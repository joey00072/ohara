from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
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


def _hypersphere_update_(
    params: list[Tensor],
    directions: list[Tensor],
    learning_rate: float,
    *,
    eps: float = 1e-10,
) -> None:
    """Take a constant-norm ("hyperspherical") step, in place, over a whole group.

    For each parameter and its update direction::

        p' = p - lr * u * ||p|| / ||u||       # step size is relative to ||p||
        p  = p' * ||p|| / ||p'||              # project back onto the sphere

    The Frobenius norm of every parameter is therefore preserved exactly, which is
    what makes weight decay unnecessary: it has no first-order effect once the
    parameter is renormalized. Because the update is divided by ``||u||``, only the
    *direction* of ``u`` matters -- any constant rescaling of it cancels.

    Every quantity stays on device (no ``.item()``) and every operation is a
    ``torch._foreach_*`` call, so one group costs a handful of fused kernels
    regardless of how many parameters it holds.
    """
    if not params:
        return
    param_norms = torch._foreach_norm(params)
    direction_norms = torch._foreach_clamp_min(torch._foreach_norm(directions), eps)

    scales = torch._foreach_div(param_norms, direction_norms)
    torch._foreach_mul_(scales, learning_rate)
    torch._foreach_add_(params, torch._foreach_mul(directions, scales), alpha=-1)

    new_norms = torch._foreach_clamp_min(torch._foreach_norm(params), eps)
    torch._foreach_mul_(params, torch._foreach_div(param_norms, new_norms))


def _hypersphere_update_stacked_(
    stacked: Tensor,
    updates: Tensor,
    learning_rate: float,
    *,
    eps: float = 1e-10,
) -> None:
    """Batched :func:`_hypersphere_update_` for one stack of same-shaped matrices.

    ``stacked`` is ``(num_params, rows, columns)``; norms are taken per matrix, so
    this matches the per-parameter behaviour of the foreach version exactly while
    running as single batched kernels.
    """
    param_norms = torch.linalg.matrix_norm(stacked, keepdim=True)
    update_norms = torch.linalg.matrix_norm(updates, keepdim=True).clamp_min(eps)

    stacked.sub_(updates * (param_norms * learning_rate / update_norms))

    new_norms = torch.linalg.matrix_norm(stacked, keepdim=True).clamp_min(eps)
    stacked.mul_(param_norms / new_norms)


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
    """Hybrid optimizer: Muon for matrices, AdamW elsewhere, either optionally on a sphere.

    Muon groups must contain same-shaped 2-D parameters, or one 3-D parameter
    representing a pre-stacked batch of matrices. AdamW groups may contain
    arbitrary dense parameters. Keeping every algorithm in one Optimizer makes
    checkpointing and AMP stepping behave exactly like a normal PyTorch optimizer.

    Setting ``hypersphere=True`` on a group switches it from an additive step to a
    constant-norm one, giving the AdamH / MuonH variants from *Rethinking Language
    Model Scaling under Transferable Hypersphere Optimization*
    (https://arxiv.org/abs/2603.28743). See :func:`_hypersphere_update_`. Such a
    group must have ``weight_decay=0``; its ``lr`` is a relative step size
    (roughly a rotation angle), not an absolute one.
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
            if group.get("hypersphere", False):
                if float(group.get("weight_decay", 0.0)) != 0.0:
                    raise ValueError(
                        "hyperspherical groups must use weight_decay=0: renormalizing the "
                        "parameter cancels decay to first order. Fold it into the learning "
                        "rate instead (lr = sqrt(lr * weight_decay) of the additive recipe)."
                    )
                if any(
                    parameter.ndim < 2
                    for parameter in group.get("params", ())
                ):
                    raise ValueError(
                        "hyperspherical groups hold matrix parameters only; keep vectors "
                        "such as norm gains and biases in a plain AdamW group"
                    )
            if kind == "muon":
                params = list(group.get("params", ()))
                if not params:
                    raise ValueError("Muon groups must contain matrix parameters")
                # A single 3D parameter is a batch of matrices that is *already*
                # stacked -- mixture-of-experts weights are stored that way, one
                # tensor holding every expert. Muon's update is batched anyway, so
                # it applies directly with no stacking step.
                pre_stacked = len(params) == 1 and params[0].ndim == 3
                if not pre_stacked:
                    if any(parameter.ndim != 2 for parameter in params):
                        raise ValueError(
                            "Muon groups must contain two-dimensional parameters, or a "
                            "single pre-stacked three-dimensional parameter"
                        )
                    if any(parameter.shape != params[0].shape for parameter in params):
                        raise ValueError(
                            "parameters in a Muon group must have identical shapes"
                        )
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

    def _adamh_step(self, group: dict[str, Any]) -> None:
        """AdamH: the Adam direction, applied as a constant-norm step.

        Runs the whole group through ``torch._foreach_*`` rather than looping one
        parameter at a time, so a 40-matrix group costs a fixed number of kernels.
        """
        params = [parameter for parameter in group["params"] if parameter.grad is not None]
        if not params:
            return
        gradients = [parameter.grad for parameter in params]
        if any(gradient.is_sparse for gradient in gradients):
            raise RuntimeError("MuonAdamW does not support sparse gradients")

        states = [self.state[parameter] for parameter in params]
        for parameter, state in zip(params, states, strict=True):
            if not state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] += 1

        beta1, beta2 = group["betas"]
        eps = float(group["eps"])
        step = states[0]["step"]
        exp_avgs = [state["exp_avg"] for state in states]
        exp_avg_sqs = [state["exp_avg_sq"] for state in states]

        torch._foreach_lerp_(exp_avgs, gradients, 1.0 - beta1)
        squared = torch._foreach_mul(gradients, gradients)
        torch._foreach_lerp_(exp_avg_sqs, squared, 1.0 - beta2)

        # Bias-corrected Adam direction: m_hat / (sqrt(v_hat) + eps).
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        denominators = torch._foreach_sqrt(torch._foreach_div(exp_avg_sqs, bias_correction2))
        torch._foreach_add_(denominators, eps)
        directions = torch._foreach_div(
            torch._foreach_div(exp_avgs, bias_correction1), denominators
        )

        if len(params) == 1 and params[0].ndim == 3:
            _hypersphere_update_stacked_(params[0], directions[0], float(group["lr"]))
        else:
            _hypersphere_update_(params, directions, float(group["lr"]))

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
        # A pre-stacked parameter already carries its batch dimension, so the
        # buffers match its shape directly instead of gaining one.
        pre_stacked = len(params) == 1 and first.ndim == 3
        rows, columns = first.shape[-2:]
        batch = first.size(0) if pre_stacked else len(params)
        state = self.state[first]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(
                batch, rows, columns, dtype=first.dtype, device=first.device
            )
        hyperspherical = bool(group.get("hypersphere", False))
        reduction_dim = -1 if rows >= columns else -2
        # MuonH never consumes the second moment, so do not pay for the buffer.
        if not hyperspherical and "second_momentum_buffer" not in state:
            second_shape = (
                (batch, rows, 1) if reduction_dim == -1 else (batch, 1, columns)
            )
            state["second_momentum_buffer"] = torch.zeros(
                second_shape, dtype=first.dtype, device=first.device
            )

        momentum_buffer = state["momentum_buffer"]
        if pre_stacked:
            stacked_gradients = params[0].grad
            stacked_parameters = params[0].detach()
        else:
            stacked_gradients = torch.stack([parameter.grad for parameter in params])
            stacked_parameters = torch.stack(params)

        momentum = float(group["momentum"])
        momentum_buffer.lerp_(stacked_gradients, 1.0 - momentum)
        update = stacked_gradients.lerp(momentum_buffer, momentum)
        update = _polar_express(update, int(group["ns_steps"]))

        if hyperspherical:
            # MuonH: orthogonalized momentum, applied as a constant-norm step.
            # The second-moment rescaling below is deliberately skipped -- it
            # changes the update *direction*, which the projection does not
            # normalize away, so keeping it would no longer be MuonH. The
            # aspect-ratio learning-rate factor is skipped for the opposite
            # reason: it is a pure rescaling, so the projection cancels it.
            _hypersphere_update_stacked_(
                stacked_parameters, update.to(stacked_parameters.dtype), float(group["lr"])
            )
            if not pre_stacked:
                torch._foreach_copy_(params, list(stacked_parameters.unbind(0)))
            return

        second_momentum_buffer = state["second_momentum_buffer"]
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
        # A pre-stacked parameter shares storage with stacked_parameters, so the
        # in-place update above already landed on it; copying back would only
        # reshape-mismatch.
        if not pre_stacked:
            torch._foreach_copy_(params, list(stacked_parameters.unbind(0)))

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["kind"] == "adamw":
                if group.get("hypersphere", False):
                    self._adamh_step(group)
                else:
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


@dataclass(frozen=True)
class _LlamaParameterGroups:
    """A Llama's trainable parameters, split by the role each plays in training."""

    embedding: list[torch.nn.Parameter]
    unembedding: list[torch.nn.Parameter]  # the lm_head; empty when weights are tied
    matrix: list[torch.nn.Parameter]  # hidden linear weights
    # Mixture-of-experts weights, one (experts, in, out) tensor per projection.
    # Already a batch of matrices, so Muon takes each without stacking.
    stacked_matrix: list[torch.nn.Parameter]
    scalar: list[torch.nn.Parameter]  # norm gains, biases, anything 1-D
    hidden_size: int


def _partition_llama_parameters(model: torch.nn.Module, caller: str) -> _LlamaParameterGroups:
    """Split an ohara Llama into embedding / lm_head / hidden-matrix / vector groups.

    Every optimizer builder here shares this partition, so they cannot drift apart
    on which parameter belongs where.
    """
    root = _unwrap_model(model)
    required = ("token_emb", "vocab_proj", "layers", "config")
    if any(not hasattr(root, attribute) for attribute in required):
        raise TypeError(f"{caller} expects an ohara Llama-compatible model")

    trainable = [parameter for parameter in root.parameters() if parameter.requires_grad]
    embedding = [
        parameter for parameter in root.token_emb.parameters() if parameter.requires_grad
    ]
    embedding_ids = {id(parameter) for parameter in embedding}
    unembedding = [
        parameter
        for parameter in root.vocab_proj.parameters()
        if parameter.requires_grad and id(parameter) not in embedding_ids
    ]
    reserved_ids = embedding_ids | {id(parameter) for parameter in unembedding}

    matrix = [
        parameter
        for parameter in root.layers.parameters()
        if parameter.requires_grad and parameter.ndim == 2 and id(parameter) not in reserved_ids
    ]
    # Mixture-of-experts layers store every expert in one (experts, in, out)
    # tensor. Those are matrices too -- a whole batch of them -- and belong on
    # Muon at the matrix learning rate. Without this they fall through to the
    # scalar catch-all and train at the norm-gain rate, which silently
    # undertrains most of the model.
    stacked_matrix = [
        parameter
        for parameter in root.layers.parameters()
        if parameter.requires_grad and parameter.ndim == 3 and id(parameter) not in reserved_ids
    ]
    matrix_ids = {id(parameter) for parameter in matrix} | {
        id(parameter) for parameter in stacked_matrix
    }
    scalar = [
        parameter
        for parameter in trainable
        if id(parameter) not in reserved_ids and id(parameter) not in matrix_ids
    ]

    assigned = embedding + unembedding + matrix + stacked_matrix + scalar
    if len({id(parameter) for parameter in assigned}) != len(assigned):
        raise RuntimeError("optimizer parameter groups overlap")
    if {id(parameter) for parameter in assigned} != {id(parameter) for parameter in trainable}:
        raise RuntimeError("optimizer parameter partition is incomplete")
    if not embedding or not matrix:
        raise ValueError("model must have token embeddings and transformer matrices")

    return _LlamaParameterGroups(
        embedding=embedding,
        unembedding=unembedding,
        matrix=matrix,
        stacked_matrix=stacked_matrix,
        scalar=scalar,
        hidden_size=int(root.config.hidden_size),
    )


def _group_matrices_by_shape(
    params: list[torch.nn.Parameter],
) -> list[tuple[torch.Size, list[torch.nn.Parameter]]]:
    """Bucket matrices by shape so each bucket can be stacked into one batched step."""
    by_shape: defaultdict[torch.Size, list[torch.nn.Parameter]] = defaultdict(list)
    for parameter in params:
        by_shape[parameter.shape].append(parameter)
    return [(shape, by_shape[shape]) for shape in sorted(by_shape, key=tuple)]


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

    partition = _partition_llama_parameters(model, "build_muon_adamw")
    embedding_params = partition.embedding
    unembedding_params = partition.unembedding
    matrix_params = partition.matrix
    scalar_params = partition.scalar

    width_scale = math.sqrt(reference_hidden_size / partition.hidden_size)
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

    # Each stacked tensor is its own Muon group: it is already the batch.
    for parameter in partition.stacked_matrix:
        groups.append(
            {
                "name": f"expert_{'x'.join(str(size) for size in parameter.shape)}",
                "kind": "muon",
                "params": [parameter],
                "lr": matrix_learning_rate,
                "initial_lr": matrix_learning_rate,
                "lr_scale": 1.0,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "beta2": beta2,
                "ns_steps": ns_steps,
            }
        )

    for shape, params in _group_matrices_by_shape(matrix_params):
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


def build_adamh(
    model: torch.nn.Module,
    *,
    learning_rate: float = 0.007,
    adam_learning_rate: float = 6e-4,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> MuonAdamW:
    """AdamH: constant-norm Adam on every matrix, plain AdamW on everything else.

    From *Rethinking Language Model Scaling under Transferable Hypersphere
    Optimization* (https://arxiv.org/abs/2603.28743). Each matrix keeps the
    Frobenius norm it was initialized with, so there is no weight decay to tune;
    ``learning_rate`` is the *relative* step size ||dW||/||W||.

    To port an AdamW recipe, use ``learning_rate = sqrt(lr * weight_decay)``. The
    default is that formula applied to ohara's own AdamW defaults
    (``sqrt(5e-4 * 0.1) = 0.0071``).

    Args:
        learning_rate: Relative step size for the hyperspherical matrix groups.
        adam_learning_rate: Absolute LR for embeddings and vector parameters,
            which stay on ordinary AdamW.
    """
    if learning_rate <= 0 or adam_learning_rate <= 0:
        raise ValueError("all learning rates must be positive")

    partition = _partition_llama_parameters(model, "build_adamh")
    # The paper puts every Linear weight, lm_head included, on AdamH.
    matrices = partition.matrix + partition.unembedding
    groups: list[dict[str, Any]] = []

    for shape, params in _group_matrices_by_shape(matrices):
        groups.append(
            {
                "name": f"adamh_{'x'.join(str(size) for size in shape)}",
                "kind": "adamw",
                "hypersphere": True,
                "params": params,
                "lr": learning_rate,
                "initial_lr": learning_rate,
                "lr_scale": 1.0,
                "betas": betas,
                "eps": eps,
                "weight_decay": 0.0,
            }
        )

    # Each expert stack is one Parameter but contains many independent matrices;
    # keep each expert matrix on its own constant-norm sphere.
    for parameter in partition.stacked_matrix:
        groups.append(
            {
                "name": f"adamh_expert_{'x'.join(str(size) for size in parameter.shape)}",
                "kind": "adamw",
                "hypersphere": True,
                "params": [parameter],
                "lr": learning_rate,
                "initial_lr": learning_rate,
                "lr_scale": 1.0,
                "betas": betas,
                "eps": eps,
                "weight_decay": 0.0,
            }
        )

    adam_params = partition.embedding + partition.scalar
    if adam_params:
        groups.append(
            {
                "name": "embedding_and_scalars",
                "kind": "adamw",
                "params": adam_params,
                "lr": adam_learning_rate,
                "initial_lr": adam_learning_rate,
                "lr_scale": adam_learning_rate / learning_rate,
                "betas": betas,
                "eps": eps,
                "weight_decay": 0.0,
            }
        )
    return MuonAdamW(groups)


def build_muonh_adamh(
    model: torch.nn.Module,
    *,
    learning_rate: float = 0.075,
    adam_learning_rate: float = 6e-4,
    momentum: float = 0.95,
    ns_steps: int = 5,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> MuonAdamW:
    """MuonH on hidden matrices, AdamH on the lm_head, plain AdamW on the rest.

    From *Rethinking Language Model Scaling under Transferable Hypersphere
    Optimization* (https://arxiv.org/abs/2603.28743). Both hyperspherical
    algorithms normalize the update to a fixed fraction of ``||W||``, which is why
    they share one ``learning_rate`` -- the paper's central practical claim, and the
    reason there is no separate matrix/unembedding rate to retune here.

    To port a Muon recipe, use ``learning_rate = sqrt(lr * weight_decay)``. The
    default is that formula applied to ohara's nanochat defaults
    (``sqrt(0.02 * 0.28) = 0.0748``).

    Args:
        learning_rate: Relative step size shared by the MuonH and AdamH groups.
        adam_learning_rate: Absolute LR for embeddings and vector parameters.
        momentum: Muon momentum. Its scale cancels in the projection, so this only
            sets how much history the update direction carries.
        ns_steps: Newton-Schulz iterations used to orthogonalize the update.
    """
    if learning_rate <= 0 or adam_learning_rate <= 0:
        raise ValueError("all learning rates must be positive")

    partition = _partition_llama_parameters(model, "build_muonh_adamh")
    groups: list[dict[str, Any]] = []

    for shape, params in _group_matrices_by_shape(partition.matrix):
        groups.append(
            {
                "name": f"muonh_{'x'.join(str(size) for size in shape)}",
                "kind": "muon",
                "hypersphere": True,
                "params": params,
                "lr": learning_rate,
                "initial_lr": learning_rate,
                "lr_scale": 1.0,
                "momentum": momentum,
                "ns_steps": ns_steps,
                "beta2": 0.9,  # unused on the hyperspherical path
                "weight_decay": 0.0,
            }
        )

    for parameter in partition.stacked_matrix:
        groups.append(
            {
                "name": f"muonh_expert_{'x'.join(str(size) for size in parameter.shape)}",
                "kind": "muon",
                "hypersphere": True,
                "params": [parameter],
                "lr": learning_rate,
                "initial_lr": learning_rate,
                "lr_scale": 1.0,
                "momentum": momentum,
                "ns_steps": ns_steps,
                "beta2": 0.9,  # unused on the hyperspherical path
                "weight_decay": 0.0,
            }
        )

    # The lm_head is a matrix but is not orthogonalized: the paper keeps it on
    # AdamH, at the same relative learning rate.
    for shape, params in _group_matrices_by_shape(partition.unembedding):
        groups.append(
            {
                "name": f"adamh_unembedding_{'x'.join(str(size) for size in shape)}",
                "kind": "adamw",
                "hypersphere": True,
                "params": params,
                "lr": learning_rate,
                "initial_lr": learning_rate,
                "lr_scale": 1.0,
                "betas": betas,
                "eps": eps,
                "weight_decay": 0.0,
            }
        )

    adam_params = partition.embedding + partition.scalar
    if adam_params:
        groups.append(
            {
                "name": "embedding_and_scalars",
                "kind": "adamw",
                "params": adam_params,
                "lr": adam_learning_rate,
                "initial_lr": adam_learning_rate,
                "lr_scale": adam_learning_rate / learning_rate,
                "betas": betas,
                "eps": eps,
                "weight_decay": 0.0,
            }
        )
    return MuonAdamW(groups)
