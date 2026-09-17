"""Exact quantile statistics for logit-based expert balancing.

Quantiles cannot be averaged across microbatches or ranks. Keep detached score
differences until the optimizer step, then solve over the union of tokens. This
costs O(tokens * experts) scratch space per step, but preserves the existing
unbounded logit routing without histogram clipping or changing score semantics.
"""

import torch
import torch.distributed as dist


def valid_tokens(x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Validate a True-means-padding mask and flatten its complement."""
    if padding_mask is None:
        return None
    if padding_mask.dtype != torch.bool or padding_mask.shape != x.shape[:-1]:
        raise ValueError("padding_mask must be boolean with shape (batch, sequence)")
    if padding_mask.device != x.device:
        raise ValueError("padding_mask must be on the input device")
    return ~padding_mask.reshape(-1)


@torch.no_grad()
def update_bias(module, process_group=None) -> None:
    samples = module.qb_samples
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size(process_group)
        size = torch.tensor([samples.size(0)], device=samples.device, dtype=torch.int64)
        sizes = [torch.empty_like(size) for _ in range(world_size)]
        dist.all_gather(sizes, size, group=process_group)
        lengths = [int(n.item()) for n in sizes]
        max_length = max(lengths)
        if max_length:
            padded = samples.new_zeros((max_length, module.num_experts))
            padded[: samples.size(0)].copy_(samples)
            gathered = [torch.empty_like(padded) for _ in range(world_size)]
            dist.all_gather(gathered, padded, group=process_group)
            samples = torch.cat([part[:n] for part, n in zip(gathered, lengths)])
    if samples.size(0):
        # Ceil gives the first order statistic reaching the desired token share.
        rank = (
            samples.size(0) * module.num_experts_per_tok + module.num_experts - 1
        ) // module.num_experts
        beta = samples.kthvalue(samples.size(0) - rank + 1, dim=0).values
        bias = -beta
        module.router_bias.copy_(bias - bias.mean())
    module.reset_qb_stats()
