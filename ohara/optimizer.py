from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, optim


# Five-step Polar Express coefficients used by nanochat. The iteration produces
# an efficient approximation to the polar factor of each Muon update.
POLAR_EXPRESS_COEFFICIENTS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def build_adamw(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
) -> optim.AdamW:
    """Build AdamW with decay on matrix weights, excluding norms and scalars."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")

    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    if not decay:
        raise ValueError("model has no trainable matrix parameters")

    return optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=betas,
    )


def _polar_express(matrix: Tensor, steps: int) -> Tensor:
    """Orthogonalize a stack of matrices with nanochat's Polar Express update."""
    if matrix.ndim != 3:
        raise ValueError("Muon expects a stack of two-dimensional gradients")
    if not 1 <= steps <= len(POLAR_EXPRESS_COEFFICIENTS):
        raise ValueError("ns_steps must be between 1 and 5")

    # BF16 is fast and numerically safe for this iteration on CUDA. FP16 is not:
    # its exponent range is too small. Keep CPU execution in FP32 for portability.
    use_bfloat16 = matrix.device.type == "cuda" and matrix.dtype != torch.float64
    x = matrix.to(torch.bfloat16) if use_bfloat16 else matrix.float()
    norm = x.float().norm(dim=(-2, -1), keepdim=True)
    x = x / (norm.to(x.dtype) * 1.01 + 1e-6)

    if matrix.size(-2) > matrix.size(-1):
        for a, b, c in POLAR_EXPRESS_COEFFICIENTS[:steps]:
            gram = x.mT @ x
            correction = b * gram + c * (gram @ gram)
            x = a * x + x @ correction
    else:
        for a, b, c in POLAR_EXPRESS_COEFFICIENTS[:steps]:
            gram = x @ x.mT
            correction = b * gram + c * (gram @ gram)
            x = a * x + correction @ x
    return x


class MuonAdamW(optim.Optimizer):
    """Nanochat-style hybrid optimizer: Muon for matrices and AdamW elsewhere.

    Muon groups must contain same-shaped 2-D parameters. AdamW groups may contain
    arbitrary dense parameters. Keeping both algorithms in one Optimizer makes
    checkpointing and AMP stepping behave exactly like a normal PyTorch optimizer.
    """

    def __init__(self, param_groups: list[dict[str, Any]]):
        if not param_groups:
            raise ValueError("param_groups cannot be empty")
        for group in param_groups:
            kind = group.get("kind")
            if kind not in {"adamw", "muon"}:
                raise ValueError(f"unknown optimizer kind: {kind!r}")
            if float(group.get("lr", 0.0)) <= 0:
                raise ValueError("every optimizer group needs a positive lr")
            if float(group.get("weight_decay", 0.0)) < 0:
                raise ValueError("weight_decay cannot be negative")
            if kind == "muon":
                params = list(group.get("params", ()))
                if not params or any(parameter.ndim != 2 for parameter in params):
                    raise ValueError("Muon groups must contain two-dimensional parameters")
                if any(parameter.shape != params[0].shape for parameter in params):
                    raise ValueError("parameters in a Muon group must have identical shapes")
                momentum = float(group.get("momentum", 0.95))
                beta2 = float(group.get("beta2", 0.9))
                if not 0 <= momentum < 1 or not 0 <= beta2 < 1:
                    raise ValueError("Muon momentum and beta2 must be in [0, 1)")
                steps = int(group.get("ns_steps", 5))
                if not 1 <= steps <= len(POLAR_EXPRESS_COEFFICIENTS):
                    raise ValueError("ns_steps must be between 1 and 5")
        super().__init__(param_groups, defaults={})

    @staticmethod
    def _adamw_step(parameter: Tensor, group: dict[str, Any], state: dict[str, Any]) -> None:
        gradient = parameter.grad
        if gradient is None:
            return
        if gradient.is_sparse:
            raise RuntimeError("MuonAdamW does not support sparse gradients")
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(parameter)
            state["exp_avg_sq"] = torch.zeros_like(parameter)

        state["step"] += 1
        beta1, beta2 = group["betas"]
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])

        parameter.mul_(1.0 - learning_rate * weight_decay)
        exp_avg.lerp_(gradient, 1.0 - beta1)
        exp_avg_sq.lerp_(gradient.square(), 1.0 - beta2)
        bias_correction1 = 1.0 - beta1 ** state["step"]
        bias_correction2 = 1.0 - beta2 ** state["step"]
        denominator = (exp_avg_sq / bias_correction2).sqrt().add_(group["eps"])
        parameter.addcdiv_(exp_avg, denominator, value=-learning_rate / bias_correction1)

    def _muon_step(self, group: dict[str, Any]) -> None:
        params: list[Tensor] = group["params"]
        active = [parameter for parameter in params if parameter.grad is not None]
        if not active:
            return
        if len(active) != len(params):
            raise RuntimeError(
                "a Muon group has missing gradients; all same-shaped matrix parameters "
                "must participate in each optimization step"
            )
        if any(parameter.grad is not None and parameter.grad.is_sparse for parameter in active):
            raise RuntimeError("MuonAdamW does not support sparse gradients")

        first = params[0]
        rows, columns = first.shape
        state = self.state[first]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                len(params), rows, columns, dtype=first.dtype, device=first.device
            )
        reduction_dim = -1 if rows >= columns else -2
        if "second_momentum_buffer" not in state:
            second_shape = (
                (len(params), rows, 1)
                if reduction_dim == -1
                else (len(params), 1, columns)
            )
            state["second_momentum_buffer"] = torch.zeros(
                second_shape, dtype=first.dtype, device=first.device
            )

        momentum_buffer = state["momentum_buffer"]
        second_momentum_buffer = state["second_momentum_buffer"]
        stacked_gradients = torch.stack([parameter.grad for parameter in params])
        stacked_parameters = torch.stack(params)

        momentum = float(group["momentum"])
        momentum_buffer.lerp_(stacked_gradients, 1.0 - momentum)
        update = stacked_gradients.lerp(momentum_buffer, momentum)
        update = _polar_express(update, int(group["ns_steps"]))

        beta2 = float(group["beta2"])
        variance = update.float().square().mean(dim=reduction_dim, keepdim=True)
        reduction_size = update.size(reduction_dim)
        original_norm = (variance.sum(dim=(-2, -1), keepdim=True) * reduction_size).sqrt()
        second_momentum_buffer.lerp_(variance.to(second_momentum_buffer.dtype), 1.0 - beta2)
        inverse_rms = second_momentum_buffer.clamp_min(1e-10).rsqrt()
        scaled_norm = (
            (variance * reduction_size) * inverse_rms.float().square()
        ).sum(dim=(-2, -1), keepdim=True).sqrt()
        scale = inverse_rms * (original_norm / scaled_norm.clamp_min(1e-10))
        update = update * scale.to(update.dtype)

        # Match nanochat's aspect-ratio adjustment and cautious weight decay.
        learning_rate = float(group["lr"]) * math.sqrt(max(1.0, rows / columns))
        weight_decay = float(group["weight_decay"])
        update = update.to(stacked_parameters.dtype)
        decay_mask = (update * stacked_parameters) >= 0
        stacked_parameters.sub_(
            learning_rate * update
            + learning_rate * weight_decay * stacked_parameters * decay_mask
        )
        torch._foreach_copy_(params, list(stacked_parameters.unbind(0)))

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["kind"] == "adamw":
                for parameter in group["params"]:
                    self._adamw_step(parameter, group, self.state[parameter])
            elif group["kind"] == "muon":
                self._muon_step(group)
            else:  # Protect against malformed state_dicts.
                raise ValueError(f"unknown optimizer kind: {group['kind']!r}")
        return loss


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    while hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
        model = model.module
    return model


def build_muon_adamw(
    model: torch.nn.Module,
    *,
    matrix_learning_rate: float = 0.02,
    embedding_learning_rate: float = 0.3,
    unembedding_learning_rate: float = 0.008,
    scalar_learning_rate: float = 0.5,
    weight_decay: float = 0.28,
    reference_hidden_size: int = 768,
    momentum: float = 0.95,
    beta2: float = 0.9,
    ns_steps: int = 5,
) -> MuonAdamW:
    """Partition a Llama into nanochat-style hybrid optimizer groups.

    Learning rates passed here are already batch-size scaled. AdamW rates receive
    the additional 1/sqrt(d_model) transfer used by nanochat; Muon matrix rates do
    not. Learned RMSNorm weights use the conservative scalar-lr × 0.01 group.
    """
    values = (
        matrix_learning_rate,
        embedding_learning_rate,
        unembedding_learning_rate,
        scalar_learning_rate,
    )
    if any(value <= 0 for value in values):
        raise ValueError("all learning rates must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay cannot be negative")
    if reference_hidden_size < 1:
        raise ValueError("reference_hidden_size must be positive")

    root = _unwrap_model(model)
    required = ("token_emb", "vocab_proj", "layers", "config")
    if any(not hasattr(root, attribute) for attribute in required):
        raise TypeError("build_muon_adamw expects an ohara Llama-compatible model")

    trainable = [parameter for parameter in root.parameters() if parameter.requires_grad]
    embedding_params = [
        parameter for parameter in root.token_emb.parameters() if parameter.requires_grad
    ]
    embedding_ids = {id(parameter) for parameter in embedding_params}
    unembedding_params = [
        parameter
        for parameter in root.vocab_proj.parameters()
        if parameter.requires_grad and id(parameter) not in embedding_ids
    ]
    reserved_ids = embedding_ids | {id(parameter) for parameter in unembedding_params}

    matrix_params = [
        parameter
        for parameter in root.layers.parameters()
        if parameter.requires_grad and parameter.ndim == 2 and id(parameter) not in reserved_ids
    ]
    matrix_ids = {id(parameter) for parameter in matrix_params}
    scalar_params = [
        parameter
        for parameter in trainable
        if id(parameter) not in reserved_ids and id(parameter) not in matrix_ids
    ]

    assigned = embedding_params + unembedding_params + matrix_params + scalar_params
    if len({id(parameter) for parameter in assigned}) != len(assigned):
        raise RuntimeError("optimizer parameter groups overlap")
    if {id(parameter) for parameter in assigned} != {id(parameter) for parameter in trainable}:
        raise RuntimeError("optimizer parameter partition is incomplete")
    if not embedding_params or not matrix_params:
        raise ValueError("model must have token embeddings and transformer matrices")

    hidden_size = int(root.config.hidden_size)
    width_scale = math.sqrt(reference_hidden_size / hidden_size)
    groups: list[dict[str, Any]] = []

    def add_adamw_group(
        name: str,
        params: list[torch.nn.Parameter],
        learning_rate: float,
        betas: tuple[float, float],
        decay: float,
    ) -> None:
        if not params:
            return
        groups.append(
            {
                "name": name,
                "kind": "adamw",
                "params": params,
                "lr": learning_rate,
                "initial_lr": learning_rate,
                "lr_scale": learning_rate / matrix_learning_rate,
                "betas": betas,
                "eps": 1e-10,
                "weight_decay": decay,
            }
        )

    add_adamw_group(
        "unembedding",
        unembedding_params,
        unembedding_learning_rate * width_scale,
        (0.8, 0.96),
        0.01,
    )
    add_adamw_group(
        "embedding",
        embedding_params,
        embedding_learning_rate * width_scale,
        (0.8, 0.995),
        0.001,
    )
    add_adamw_group(
        "norms_and_scalars",
        scalar_params,
        scalar_learning_rate * 0.01,
        (0.8, 0.95),
        0.0,
    )

    by_shape: defaultdict[torch.Size, list[torch.nn.Parameter]] = defaultdict(list)
    for parameter in matrix_params:
        by_shape[parameter.shape].append(parameter)
    for shape in sorted(by_shape, key=tuple):
        params = by_shape[shape]
        groups.append(
            {
                "name": f"matrix_{'x'.join(str(size) for size in shape)}",
                "kind": "muon",
                "params": params,
                "lr": matrix_learning_rate,
                "initial_lr": matrix_learning_rate,
                "lr_scale": 1.0,
                "momentum": momentum,
                "ns_steps": ns_steps,
                "beta2": beta2,
                "weight_decay": weight_decay,
            }
        )
    return MuonAdamW(groups)
