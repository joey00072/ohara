"""Fine-grained mixture of experts with a shared expert, dispatched as grouped matmuls.

:class:`ohara.modules.moe.MoE` runs one contiguous slice per expert in a Python
loop. That is clear and correct, and it is fine at 8 experts. It stops being fine
at 256: the loop issues 256 tiny GEMMs per layer -- 3,072 for a 12-layer model --
and ``counts.tolist()`` forces a host sync every layer, which also breaks
``torch.compile`` into fragments.

This module keeps the same routing but stacks the expert weights into single
``(num_experts, in, out)`` tensors and dispatches every expert in one
``torch._grouped_mm``. One kernel replaces the loop, no host sync, static graph.

Two design points beyond that, both from the DeepSeek MoE line of work
(https://arxiv.org/abs/2401.06066, https://arxiv.org/abs/2412.19437):

**Shared experts.** A fraction of the feed-forward is always active for every
token. Routed experts then no longer each have to relearn the common
transformation, so they specialise instead of duplicating. This is the "shared
expert isolation" of DeepSeekMoE.

**Fine granularity.** Many narrow experts rather than a few wide ones. With E
experts and top-k routing there are C(E, k) possible combinations, so shrinking
experts while raising k buys combinatorially more specialisation at equal FLOPs.

Load balancing stays quantile balancing (Jianlin Su; Kimi K2/K3), matching
:mod:`ohara.modules.moe`: the router bias is solved in closed form each step, so
there is no auxiliary loss and no balancing coefficient to tune. DeepSeek-V3
reaches the same place by a different update rule.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _grouped_mm_available() -> bool:
    return hasattr(torch, "_grouped_mm")


class GroupedMoE(nn.Module):
    """Top-k routed experts plus always-on shared experts, batched into grouped GEMMs.

    Args:
        dim: model width.
        hidden_dim: width of a *single* expert. Keep ``hidden_dim * (k + shared)``
            equal to the dense feed-forward width to hold FLOPs per token fixed.
        num_experts: routed experts per layer.
        num_experts_per_tok: how many routed experts each token uses.
        num_shared_experts: experts every token always passes through.
        gate_fn: ``sigmoid`` (DeepSeek-V3) or ``softmax``. Sigmoid scores each
            expert independently, which behaves better as ``num_experts`` grows
            because softmax over hundreds of logits drives every weight tiny.
        normalize_weights: rescale the chosen weights to sum to 1. With sigmoid
            this keeps the residual contribution at a stable scale.
        quantile_balancing: closed-form router bias, no auxiliary loss.
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

        # Routed experts, stacked. SwiGLU: down(silu(gate(x)) * up(x)).
        self.w_gate = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w_up = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.w_down = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))

        # Shared experts are dense: every token uses them, so there is nothing to
        # route and a plain matmul of the full batch is the fastest form.
        if num_shared_experts > 0:
            shared_hidden = hidden_dim * num_shared_experts
            self.shared_gate = nn.Linear(dim, shared_hidden, bias=False)
            self.shared_up = nn.Linear(dim, shared_hidden, bias=False)
            self.shared_down = nn.Linear(shared_hidden, dim, bias=False)

        self.router = nn.Linear(dim, num_experts, bias=False)

        # A buffer, not a Parameter: quantile balancing solves for it directly, so
        # the optimizer must never touch it. Checkpointed, so a resumed run starts
        # already balanced.
        self.register_buffer("router_bias", torch.zeros(num_experts))
        self.register_buffer("qb_beta_sum", torch.zeros(num_experts), persistent=False)
        self.register_buffer("qb_beta_count", torch.zeros(()), persistent=False)
        self.register_buffer(
            "expert_counts", torch.zeros(num_experts, dtype=torch.long), persistent=False
        )
        self.reset_parameters()

    # -- routing ---------------------------------------------------------

    def _route(self, flat_x: torch.Tensor):
        """Return (expert_indices, expert_weights, logits, alpha) for each token."""
        # fp32 for the router: quantile balancing works on quantiles of logit
        # differences, and in bf16 too many of those differences collide.
        logits = self.router(flat_x).float()

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
            weights = selected.softmax(dim=-1)
        else:
            weights = torch.sigmoid(selected)
            if self.normalize_weights:
                weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return expert_indices, weights, logits, alpha

    # -- dispatch --------------------------------------------------------

    # torch._grouped_mm carries shape guards that dynamo cannot discharge -- the
    # group offsets are data-dependent, so tracing fails on an internal check.
    # Running this one function eagerly costs little: it is already three fused
    # kernels, and everything around it (norms, attention, the head) still
    # compiles. The alternative is not compiling the model at all.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        flat_x = x.reshape(batch * seq_len, dim)

        expert_indices, expert_weights, logits, alpha = self._route(flat_x)

        use_grouped = flat_x.is_cuda and _grouped_mm_available()
        dispatch = self._dispatch_grouped if use_grouped else self._dispatch_reference
        out = dispatch(flat_x, expert_indices, expert_weights.to(flat_x.dtype))

        if self.num_shared_experts > 0:
            out = out + self._shared(flat_x)

        if self.training:
            if self.quantile_balancing:
                self._accumulate_qb(logits.detach(), alpha.detach())
            self.expert_counts += torch.bincount(
                expert_indices.detach().reshape(-1), minlength=self.num_experts
            )
        return out.view(batch, seq_len, dim)

    # -- balancing -------------------------------------------------------

    @torch.no_grad()
    def _accumulate_qb(self, logits: torch.Tensor, alpha: torch.Tensor) -> None:
        # An expert wins a token when its logit beats that token's threshold alpha.
        # So the bias giving an expert exactly its fair share is the fair-share-th
        # largest of (logit - alpha) across the batch. No loss, no coefficient.
        s_minus_alpha = logits - alpha
        num_tokens = s_minus_alpha.size(0)
        fair_share = max(1, num_tokens * self.num_experts_per_tok // self.num_experts)
        beta = torch.topk(s_minus_alpha.t(), fair_share, dim=-1).values[:, -1]
        self.qb_beta_sum += beta
        self.qb_beta_count += 1

    @torch.no_grad()
    def apply_qb_update(self) -> None:
        """Fold accumulated statistics into the router bias. Once per optimizer step."""
        if float(self.qb_beta_count) == 0:
            return
        beta = self.qb_beta_sum / self.qb_beta_count
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(beta, op=dist.ReduceOp.AVG)
        bias = -beta
        # Adding a constant to every expert cannot change the top-k, so remove the
        # mean to stop the vector drifting over a long run.
        self.router_bias.copy_(bias - bias.mean())
        self.qb_beta_sum.zero_()
        self.qb_beta_count.zero_()

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
