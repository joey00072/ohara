"""Contiguous sequence shards with causal halos and recurrent state exchange.

QSA gathers KV and compressed index keys, not hidden states or queries. GDN
uses FLA's state-based CP on CUDA; the CPU oracle exchanges affine state maps.
All ranks must use equal local lengths and participate in forward/backward.
"""

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


class _Gather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        pieces = [torch.empty(x.shape, device=x.device, dtype=x.dtype)
                  for _ in range(dist.get_world_size(group))]
        dist.all_gather(pieces, x.contiguous(), group=group)
        return torch.stack(pieces)

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        rank = dist.get_rank(ctx.group)
        if dist.get_backend(ctx.group) == "nccl":
            result = torch.empty_like(grad[rank])
            dist.reduce_scatter_tensor(result, grad.flatten(0, 1), group=ctx.group)
        else:
            grad = grad.clone()
            dist.all_reduce(grad, group=ctx.group)
            result = grad[rank].contiguous()
        return result, None


class _Sum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        result = x.clone()
        dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, grad):
        result = grad.clone()
        dist.all_reduce(result, group=ctx.group)
        return result, None


@dataclass(frozen=True)
class ContextParallel:
    group: object
    removed_tail: int = 0

    @property
    def size(self):
        return dist.get_world_size(self.group)

    @property
    def rank(self):
        return dist.get_rank(self.group)

    def for_mtp(self, step):
        return replace(self, removed_tail=step)

    def positions(self, x):
        return torch.arange(x.size(1), device=x.device) + self.rank * x.size(1)

    def valid_mask(self, x):
        return (self.positions(x) < self.size * x.size(1) - self.removed_tail)[None].expand(x.size(0), -1)

    def gather(self, x, dim=1):
        return torch.cat(_Gather.apply(x, self.group).unbind(0), dim=dim)

    def halo(self, x, width, fill=0):
        """Prepend at most width tokens of history, even when local chunks are short."""
        if not width:
            return x
        tails = _Gather.apply(x[:, -width:], self.group).unbind(0)
        # Keep every rank's collective in the autograd graph, including rank zero.
        anchor = sum(t.sum() * 0 for t in tails)
        if self.rank:
            prefix = torch.cat(tails[:self.rank], 1)[:, -width:]
        else:
            prefix = x[:, :0]
        missing = width - prefix.size(1)
        padding = x.new_full((x.size(0), missing, *x.shape[2:]), fill)
        return torch.cat((padding, prefix, x), 1) + anchor

    def shift(self, x, amount, fill):
        """Move future tokens left across rank boundaries; used only for token IDs."""
        global_x = self.gather(x)
        start = self.rank * x.size(1) + amount
        part = global_x[:, start:start + x.size(1)]
        padding = x.new_full((x.size(0), x.size(1) - part.size(1), *x.shape[2:]), fill)
        return torch.cat((part, padding), 1)

    def mean_loss(self, local_sum, local_count):
        count = local_count.detach().clone()
        dist.all_reduce(count, group=self.group)
        return local_sum * self.size / count.clamp_min(1)

    def router_loss(self, probabilities, indices, shape):
        valid = self.valid_mask(probabilities.reshape(*shape, -1)).reshape(-1)
        count = valid.sum()
        dist.all_reduce(count, group=self.group)
        experts = probabilities.size(-1)
        load = torch.bincount(indices[valid].reshape(-1), minlength=experts).float()
        dist.all_reduce(load, group=self.group)
        mass = _Sum.apply((probabilities * valid[:, None]).sum(0), self.group)
        return experts * (load * mass).sum() / (count.clamp_min(1).square() * indices.size(-1))

    def delta(self, q, k, v, g, beta, backend):
        from .ops import delta_reference, fla_delta

        if backend != "torch" and q.is_cuda:
            if q.size(0) != 1:
                raise ValueError("FLA context parallelism currently requires micro-batch size 1")
            from fla.ops.cp import build_cp_context

            lengths = torch.tensor([0, q.size(1) * self.size], device=q.device, dtype=torch.long)
            cp = build_cp_context(lengths, group=self.group)
            return fla_delta()(q=q, k=k, v=v, g=g, beta=beta,
                               use_qk_l2norm_in_kernel=True, cp_context=cp)[0]
        # An affine map S_out = M @ S_in + B summarizes each sequence shard.
        normalized = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
        transition = torch.eye(k.size(-1), device=k.device).expand(k.size(0), k.size(2), -1, -1)
        state = torch.zeros(*transition.shape[:-1], v.size(-1), device=k.device)
        for t in range(k.size(1)):
            key = normalized[:, t]
            decay = g[:, t].float().exp()[..., None, None]
            strength = beta[:, t].float()[..., None, None]
            erase = key[..., :, None] * key[..., None, :]
            update = (torch.eye(k.size(-1), device=k.device) - strength * erase) * decay
            transition = update @ transition
            state = update @ state + strength * key[..., :, None] * v[:, t].float()[..., None, :]
        transitions = _Gather.apply(transition, self.group).unbind(0)
        states = _Gather.apply(state, self.group).unbind(0)
        initial = torch.zeros_like(state)
        for i in range(self.rank):
            initial = transitions[i] @ initial + states[i]
        initial = initial + sum(x.sum() * 0 for x in (*transitions, *states))
        return delta_reference(q, k, v, g, beta, initial)[0]


def causal_conv(conv, x, context=None):
    """Depthwise convolution on B,T,C with a differentiable left halo."""
    width = conv.dilation[0] * (conv.kernel_size[0] - 1)
    content = (context.halo(x, width) if context is not None
               else F.pad(x.transpose(1, 2), (width, 0)).transpose(1, 2))
    return conv(content.transpose(1, 2)).transpose(1, 2)


@torch.no_grad()
def synchronize_gradients(model, context):
    """Average replicated gradients; FSDP/DTensor shards already reduce theirs."""
    for parameter in model.parameters():
        if parameter.grad is not None and not isinstance(parameter, DTensor):
            dist.all_reduce(parameter.grad, group=context.group)
            parameter.grad.div_(context.size)
