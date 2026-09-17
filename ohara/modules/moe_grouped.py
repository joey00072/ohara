"""Top-k MoE with grouped expert matmuls and optional shared experts.

CUDA BF16 uses three grouped matmuls; other configurations use a per-expert
reference loop. Dispatch runs eagerly because grouped-mm tracing has
data-dependent shape guards. Routing uses FP32 logits and optional quantile
balancing, applied once per optimizer step.

Shared experts follow DeepSeekMoE: https://arxiv.org/abs/2401.06066.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ohara.modules.router import RouterLinear
from ohara.modules.quantile import update_bias, valid_tokens


def _grouped_mm_available() -> bool:
    return hasattr(torch, "_grouped_mm")


class GroupedMoE(nn.Module):
    """Top-k routed experts with optional always-on shared experts.

    Args:
        dim: Model width.
        hidden_dim: Width of each expert.
        num_experts: Number of routed experts.
        num_experts_per_tok: Number of routed experts selected per token.
        num_shared_experts: Number of shared experts.
        gate_fn: Router weighting function: sigmoid or softmax.
        normalize_weights: Normalize selected sigmoid weights for top-k > 1.
            Softmax always normalizes for top-k > 1. Top-1 keeps its probability.
        quantile_balancing: Update routing bias from per-step global quantiles.
        shared_exclusive: Remove the routed output's projection onto the shared
            output before adding them. Requires shared experts.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        num_shared_experts: int = 1,
        gate_fn: str = "sigmoid",
        normalize_weights: bool = True,
        quantile_balancing: bool = True,
        shared_exclusive: bool = False,
    ) -> None:
        super().__init__()
        if not 1 <= num_experts_per_tok <= num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        if num_shared_experts < 0:
            raise ValueError("num_shared_experts cannot be negative")
        if gate_fn not in ("softmax", "sigmoid"):
            raise ValueError("gate_fn must be 'softmax' or 'sigmoid'")
        if quantile_balancing and num_experts_per_tok >= num_experts:
            raise ValueError(
                "quantile balancing reads the (k+1)-th logit as a threshold, so it "
                "needs num_experts_per_tok < num_experts"
            )

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.gate_fn = gate_fn
        self.normalize_weights = normalize_weights
        self.quantile_balancing = quantile_balancing
        if shared_exclusive and num_shared_experts == 0:
            raise ValueError("shared_exclusive needs at least one shared expert to reject against")
        self.shared_exclusive = shared_exclusive

        # Routed experts, stacked. SwiGLU: down(silu(gate(x)) * up(x)).
        self.w_gate = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w_up = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w_down = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))

        # Shared experts process every token as one dense projection.
        if num_shared_experts > 0:
            shared_hidden = hidden_dim * num_shared_experts
            self.shared_gate = nn.Linear(dim, shared_hidden, bias=False)
            self.shared_up = nn.Linear(dim, shared_hidden, bias=False)
            self.shared_down = nn.Linear(shared_hidden, dim, bias=False)

        self.router = RouterLinear(dim, num_experts, bias=False)

        # Checkpoint the solved bias; optimizer-step statistics are transient.
        self.register_buffer("router_bias", torch.zeros(num_experts))
        self.register_buffer("qb_samples", torch.empty(0, num_experts), persistent=False)
        self.register_buffer(
            "expert_counts", torch.zeros(num_experts, dtype=torch.long), persistent=False
        )
        self.reset_parameters()

    def _apply(self, fn, recurse: bool = True):
        """Move balancing state without applying reduced-precision casts to it."""
        fp32_buffers = {
            name: self._buffers[name].detach().clone()
            for name in ("router_bias", "qb_samples")
            if self._buffers.get(name) is not None
        }
        result = super()._apply(fn, recurse=recurse)
        for name, value in fp32_buffers.items():
            converted = self._buffers[name]
            self._buffers[name] = (
                converted.float()
                if value.is_meta
                else value.to(device=converted.device, dtype=torch.float32)
            )
        return result

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        key = prefix + "router_bias"
        if key in state_dict and state_dict[key].dtype != torch.float32:
            state_dict[key] = state_dict[key].float()
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    # -- routing ---------------------------------------------------------

    def _route(self, flat_x: torch.Tensor):
        """Return (expert_indices, expert_weights, logits, alpha) for each token."""
        # fp32 for the router: quantile balancing works on quantiles of logit
        # differences, and in bf16 too many of those differences collide.
        logits = self.router(flat_x)

        alpha = None
        if self.quantile_balancing:
            # Take k+1; the extra one is not a routed expert, it is the threshold.
            top = torch.topk(logits + self.router_bias, self.num_experts_per_tok + 1, dim=-1)
            alpha = top.values[:, -1:]
            expert_indices = top.indices[:, : self.num_experts_per_tok]
        else:
            expert_indices = torch.topk(logits, self.num_experts_per_tok, dim=-1).indices

        # Weights come from the *unbiased* logits: the bias decides which experts
        # run, letting it through here would also change their outputs. This is
        # the router's only gradient path, since top-k itself is not differentiable.
        selected = logits.gather(-1, expert_indices)
        if self.gate_fn == "softmax":
            weights = (
                logits.softmax(dim=-1).gather(-1, expert_indices)
                if self.num_experts_per_tok == 1
                else selected.softmax(dim=-1)
            )
        else:
            weights = torch.sigmoid(selected)
            if self.normalize_weights and self.num_experts_per_tok > 1:
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return expert_indices, weights, logits, alpha

    # -- dispatch --------------------------------------------------------

    # Data-dependent grouped-mm shape guards prevent Dynamo tracing.
    @torch.compiler.disable
    def _dispatch_grouped(
        self, flat_x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """Run every expert in three grouped matmuls."""
        flat_indices = expert_indices.reshape(-1)

        # Sort (token, expert) pairs by expert so each expert owns a contiguous
        # span, then hand the span boundaries to the grouped kernel.
        order = torch.argsort(flat_indices, stable=True)
        rows = order // self.num_experts_per_tok
        counts = torch.bincount(flat_indices, minlength=self.num_experts)
        offsets = counts.cumsum(0).to(torch.int32)

        # _grouped_mm has no AutocastCUDA registration. Pick the autocast dtype
        # explicitly or BF16-mixed training silently runs these GEMMs in FP32.
        compute_dtype = (
            torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else flat_x.dtype
        )
        x_sorted = flat_x[rows].to(compute_dtype)
        gate = torch._grouped_mm(x_sorted, self.w_gate.to(compute_dtype), offs=offsets)
        up = torch._grouped_mm(x_sorted, self.w_up.to(compute_dtype), offs=offsets)
        hidden = F.silu(gate) * up
        y = torch._grouped_mm(hidden, self.w_down.to(compute_dtype), offs=offsets)

        y = y * expert_weights.reshape(-1)[order].unsqueeze(-1).to(y.dtype)
        out = torch.zeros_like(flat_x)
        # Scatter-add: a token is routed to several experts.
        out.index_add_(0, rows, y.to(out.dtype))
        return out

    def _dispatch_reference(
        self, flat_x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """Readable per-expert loop. Used on CPU and as the correctness oracle."""
        out = torch.zeros_like(flat_x)
        flat_indices = expert_indices.reshape(-1)
        flat_weights = expert_weights.reshape(-1)
        rows = torch.arange(flat_indices.numel(), device=flat_x.device) // self.num_experts_per_tok
        for expert in range(self.num_experts):
            mask = flat_indices == expert
            if not bool(mask.any()):
                continue
            token_rows = rows[mask]
            xe = flat_x[token_rows]
            hidden = F.silu(xe @ self.w_gate[expert]) * (xe @ self.w_up[expert])
            ye = (hidden @ self.w_down[expert]) * flat_weights[mask].unsqueeze(-1).to(xe.dtype)
            out.index_add_(0, token_rows, ye.to(out.dtype))
        return out

    def _shared(self, x: torch.Tensor) -> torch.Tensor:
        return self.shared_down(F.silu(self.shared_gate(x)) * self.shared_up(x))

    @staticmethod
    def _reject(target: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        """Remove the component of target along basis, per token, in FP32.

        A zero basis leaves target unchanged, including at zero-initialized startup.
        """
        unit = F.normalize(basis.float(), dim=-1)
        projection = (target.float() * unit).sum(dim=-1, keepdim=True) * unit
        return (target.float() - projection).to(target.dtype)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        valid = valid_tokens(x, padding_mask)
        batch, seq_len, dim = x.shape
        flat_x = x.reshape(batch * seq_len, dim)

        expert_indices, expert_weights, logits, alpha = self._route(flat_x)

        compute_dtype = (
            torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else flat_x.dtype
        )
        # CUDA grouped GEMMs require bf16 and 16-byte aligned row strides.
        # Other configurations remain valid through the reference dispatcher.
        use_grouped = (
            flat_x.is_cuda
            and _grouped_mm_available()
            and compute_dtype == torch.bfloat16
            and self.dim % 8 == 0
            and self.hidden_dim % 8 == 0
        )
        dispatch = self._dispatch_grouped if use_grouped else self._dispatch_reference
        out = dispatch(flat_x, expert_indices, expert_weights.to(flat_x.dtype))

        if self.num_shared_experts > 0:
            shared = self._shared(flat_x)
            if self.shared_exclusive:
                out = self._reject(out, shared)
            out = out + shared

        if self.training:
            if self.quantile_balancing:
                self._accumulate_qb(logits.detach(), alpha.detach(), valid)
            self.expert_counts += torch.bincount(
                (expert_indices if valid is None else expert_indices[valid]).detach().reshape(-1),
                minlength=self.num_experts
            )
        return out.view(batch, seq_len, dim)

    # -- balancing -------------------------------------------------------

    @torch.no_grad()
    def _accumulate_qb(
        self, logits: torch.Tensor, alpha: torch.Tensor, valid: torch.Tensor | None = None
    ) -> None:
        samples = (logits - alpha).detach().float()
        if valid is not None:
            samples = samples[valid]
        self.qb_samples = torch.cat((self.qb_samples, samples), dim=0)

    @torch.no_grad()
    def reset_qb_stats(self) -> None:
        self.qb_samples = self.qb_samples.new_empty((0, self.num_experts))

    @torch.no_grad()
    def apply_qb_update(self, process_group=None) -> None:
        """Solve over all valid tokens once per optimizer step."""
        update_bias(self, process_group)

    @torch.no_grad()
    def expert_load(self, reset: bool = True) -> torch.Tensor:
        counts = self.expert_counts.clone()
        if reset:
            self.expert_counts.zero_()
        return counts

    # -- init ------------------------------------------------------------

    @torch.no_grad()
    def reset_parameters(self, init_std: float | None = None) -> None:
        bound = (3.0**0.5) * (init_std if init_std else self.dim**-0.5)
        # Match the dense init: inputs uniform, output projection zero, so a fresh
        # expert starts as a no-op on the residual stream.
        nn.init.uniform_(self.w_gate, -0.4 * bound, 0.4 * bound)
        nn.init.uniform_(self.w_up, -0.4 * bound, 0.4 * bound)
        nn.init.zeros_(self.w_down)
        if self.num_shared_experts > 0:
            nn.init.uniform_(self.shared_gate.weight, -0.4 * bound, 0.4 * bound)
            nn.init.uniform_(self.shared_up.weight, -0.4 * bound, 0.4 * bound)
            nn.init.zeros_(self.shared_down.weight)
        nn.init.normal_(self.router.weight, mean=0.0, std=self.dim**-0.5)
        self.router_bias.zero_()
