from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from ohara.modules.mlp import MLP_MAP


class MoE(nn.Module):
    """Top-k mixture of experts with optional quantile balancing.

    Routing is the usual: a linear gate scores every expert, the top-k win the token,
    and their outputs are combined with the gate weights. Two things are worth knowing:

    - Dispatch sorts the (token, expert) pairs by expert and runs one contiguous slice
      per expert, instead of replicating the input k times and masking it E times.
    - Load balancing is quantile balancing (Jianlin Su, used in Kimi K2/K3), which needs
      no auxiliary loss and no loss coefficient to tune. See ``apply_qb_update``.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        mlp: str = "swiglu",
        gate_fn: str = "softmax",
        quantile_balancing: bool = True,
    ):
        super().__init__()
        assert 1 <= num_experts_per_tok <= num_experts
        if quantile_balancing:
            assert num_experts_per_tok < num_experts, (
                "quantile balancing reads the (k+1)-th logit as a threshold, "
                f"so it needs num_experts_per_tok < num_experts (got {num_experts_per_tok} == {num_experts})"
            )
        assert gate_fn in ("softmax", "sigmoid")

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.gate_fn = gate_fn
        self.quantile_balancing = quantile_balancing

        mlp_block = MLP_MAP[mlp]  # SwiGLU is default

        self.experts = nn.ModuleList([mlp_block(dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts, bias=False)

        # A buffer, not a Parameter: quantile balancing solves for it in closed form, so the
        # optimizer must never touch it. Checkpointed, so a resumed run starts already balanced.
        self.register_buffer("router_bias", torch.zeros(num_experts))
        # Scratch for the balancing solve, averaged over the micro-steps of one optimizer step.
        # Not checkpointed: meaningless outside the step it was accumulated in.
        self.register_buffer("qb_beta_sum", torch.zeros(num_experts), persistent=False)
        self.register_buffer("qb_beta_count", torch.zeros(()), persistent=False)
        # How many tokens each expert saw, for monitoring. See `expert_load`.
        self.register_buffer(
            "expert_counts", torch.zeros(num_experts, dtype=torch.long), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        flat_x = x.reshape(batch_size * seq_len, dim)  # (N, dim)

        # fp32 for the router: quantile balancing works on quantiles of logit differences,
        # and in bf16 too many of those differences round to the same value.
        logits = self.gate(flat_x).float()  # (N, num_experts)

        alpha = None
        if self.quantile_balancing:
            # Take k+1. The extra one is not a routed expert, it is the balancing threshold.
            top_vals, top_idx = torch.topk(
                logits + self.router_bias, self.num_experts_per_tok + 1, dim=-1
            )
            alpha = top_vals[:, -1:]  # (N, 1)
            expert_indices = top_idx[:, : self.num_experts_per_tok]  # (N, k)
        else:
            expert_indices = torch.topk(logits, self.num_experts_per_tok, dim=-1).indices

        # Weights come from the *unbiased* logits: the bias picks which experts run, letting it
        # through here would change their outputs too. This is also the gate's only path to a
        # gradient, since top-k selection is not differentiable.
        selected = logits.gather(-1, expert_indices)  # (N, k)
        if self.gate_fn == "softmax":
            expert_weights = selected.softmax(dim=-1)
        else:
            expert_weights = torch.sigmoid(selected)

        output = self._dispatch(flat_x, expert_indices, expert_weights.to(x.dtype))

        if self.training:
            if self.quantile_balancing:
                self._accumulate_qb(logits.detach(), alpha.detach())
            self.expert_counts += torch.bincount(
                expert_indices.detach().reshape(-1), minlength=self.num_experts
            )

        return output.view(batch_size, seq_len, dim)

    def _dispatch(
        self, flat_x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """Run each expert once, over a contiguous slice of tokens routed to it."""
        k = self.num_experts_per_tok
        flat_indices = expert_indices.reshape(-1)  # (N * k,)

        # Sort the (token, expert) pairs by expert so each expert's tokens are contiguous.
        order = torch.argsort(flat_indices, stable=True)
        counts = torch.bincount(flat_indices, minlength=self.num_experts)
        rows = order.div(k, rounding_mode="floor")  # token owning each sorted pair
        weights = expert_weights.reshape(-1)[order].unsqueeze(-1)  # (N * k, 1)

        output = torch.zeros_like(flat_x)
        start = 0
        for expert, count in zip(self.experts, counts.tolist()):
            if count == 0:  # nothing routed here this batch
                continue
            token_rows = rows[start : start + count]
            y = expert(flat_x[token_rows]) * weights[start : start + count]
            # Scatter-add, since a token can be routed to several experts.
            output.index_add_(0, token_rows, y.to(output.dtype))
            start += count
        return output

    @torch.no_grad()
    def _accumulate_qb(self, logits: torch.Tensor, alpha: torch.Tensor) -> None:
        # An expert is picked for a token when its logit beats that token's threshold alpha.
        # So the bias that would hand an expert exactly its fair share is the fair-share-th
        # largest of (its logit - alpha) over the batch. No loss, no coefficient to tune.
        s_minus_alpha = logits - alpha  # (N, num_experts)
        num_tokens = s_minus_alpha.size(0)
        fair_share = max(1, num_tokens * self.num_experts_per_tok // self.num_experts)
        beta = torch.topk(s_minus_alpha.t(), fair_share, dim=-1).values[:, -1]  # (num_experts,)
        self.qb_beta_sum += beta
        self.qb_beta_count += 1

    @torch.no_grad()
    def apply_qb_update(self) -> None:
        """Fold the accumulated statistics into the router bias. Call once per optimizer step."""
        if self.qb_beta_count.item() == 0:
            return
        beta = self.qb_beta_sum / self.qb_beta_count
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(beta, op=dist.ReduceOp.AVG)
        bias = -beta
        # Adding a constant to every expert changes nothing about which ones win the top-k,
        # so subtract the mean to keep the whole vector from drifting over a long run.
        self.router_bias.copy_(bias - bias.mean())
        self.qb_beta_sum.zero_()
        self.qb_beta_count.zero_()

    @torch.no_grad()
    def expert_load(self, reset: bool = True) -> torch.Tensor:
        """Tokens routed to each expert since the last reset. Perfect balance is uniform."""
        counts = self.expert_counts.clone()
        if reset:
            self.expert_counts.zero_()
        return counts

    def reset_parameters(self, init_std=None):
        gate_std = init_std or (self.dim ** (-0.5))
        nn.init.trunc_normal_(
            self.gate.weight,
            mean=0.0,
            std=gate_std,
            a=-3 * gate_std,
            b=3 * gate_std,
        )
        self.router_bias.zero_()

        for expert in self.experts:
            if hasattr(expert, "reset_parameters"):
                expert.reset_parameters(init_std=init_std)


def apply_qb_update(module: nn.Module) -> None:
    """Apply the quantile balancing update to every MoE in a model.

    Drop this into the training loop right next to `optimizer.step()`.
    """
    for submodule in module.modules():
        if isinstance(submodule, MoE) and submodule.quantile_balancing:
            submodule.apply_qb_update()


def expert_load(module: nn.Module, reset: bool = True) -> torch.Tensor | None:
    """Stack the per-expert token counts of every MoE in a model: (num_moe_layers, num_experts)."""
    counts = [m.expert_load(reset=reset) for m in module.modules() if isinstance(m, MoE)]
    return torch.stack(counts) if counts else None


def maximal_violation(counts: torch.Tensor) -> torch.Tensor:
    """MaxVio load imbalance: how far the busiest expert runs above an even split. 0 is balanced."""
    counts = counts.float()
    mean = counts.mean(dim=-1, keepdim=True)
    return ((counts.max(dim=-1, keepdim=True).values - mean) / mean).squeeze(-1)
