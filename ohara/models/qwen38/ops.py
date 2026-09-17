"""Kernel dispatch and small reference operations for the Qwen3.8 example."""

from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(shape))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * (1 + self.weight.float())).to(x.dtype)


def rope(x, positions, dim, theta):
    frequencies = theta ** (-torch.arange(0, dim, 2, device=x.device).float() / dim)
    angles = positions.float()[:, None] * frequencies
    cos, sin = angles.cos()[None, :, None], angles.sin()[None, :, None]
    left, right = x[..., :dim].float().chunk(2, dim=-1)
    rotated = torch.cat((left * cos - right * sin, right * cos + left * sin), dim=-1)
    return torch.cat((rotated.to(x.dtype), x[..., dim:]), dim=-1)


def delta_reference(q, k, v, g, beta, initial_state=None):
    """Sequential FP32 oracle, including gradients and an optional initial state."""
    dtype = q.dtype
    q, k = (x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)
            for x in (q, k))
    q = q * q.size(-1) ** -0.5
    v, g, beta = v.float(), g.float(), beta.float()
    state = (q.new_zeros(q.size(0), q.size(2), q.size(3), v.size(3))
             if initial_state is None else initial_state.float())
    outputs = []
    for t in range(q.size(1)):
        state = state * g[:, t].exp()[..., None, None]
        error = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], state)
        state = state + k[:, t, :, :, None] * (beta[:, t, :, None] * error)[:, :, None, :]
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], state))
    return torch.stack(outputs, dim=1).to(dtype), state


@lru_cache(None)
def fla_delta():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    except ImportError as error:
        raise RuntimeError(
            "CUDA Gated DeltaNet requires flash-linear-attention. Install it with uv, "
            "or select --backend torch for the correctness path."
        ) from error
    return chunk_gated_delta_rule


def delta(q, k, v, g, beta, backend):
    if backend == "torch" or (backend == "auto" and not q.is_cuda):
        return delta_reference(q, k, v, g, beta)[0]
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("CUDA delta kernels require CUDA FP16 or BF16 inputs")
    output, _ = fla_delta()(
        q=q, k=k, v=v, g=g, beta=beta, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )
    return output


def sparse_attention_reference(q, k, v, indices):
    """Gather only selected keys; caller bounds query-chunk memory."""
    batch, length, heads, dim = q.shape
    groups = heads // k.size(2)
    rows = torch.arange(batch, device=q.device)[:, None, None]
    selected_k = k[rows, indices.clamp_min(0)]
    selected_v = v[rows, indices.clamp_min(0)]
    # Batch queries independently so SDPA never constructs a sequence-square mask.
    selected_k = selected_k.reshape(batch * length, -1, k.size(2), dim).transpose(1, 2)
    selected_v = selected_v.reshape(batch * length, -1, k.size(2), dim).transpose(1, 2)
    query = q.reshape(batch * length, heads, 1, dim)
    mask = (indices >= 0).reshape(batch * length, 1, 1, -1)
    result = F.scaled_dot_product_attention(
        query, selected_k, selected_v, attn_mask=mask, enable_gqa=groups > 1,
    )
    return result.reshape(batch, length, heads, dim)


def sparse_attention(q, k, v, indices, backend):
    if backend == "torch" or (backend == "auto" and not q.is_cuda):
        return sparse_attention_reference(q, k, v, indices)
    if not q.is_cuda:
        raise ValueError("CUDA sparse attention requires CUDA tensors")
    from ohara.kernels.qwen38_sparse import sparse_attention as kernel

    return kernel(q, k, v, indices)
