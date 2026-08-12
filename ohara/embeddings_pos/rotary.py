"""Rotary position embeddings (RoPE).

Paper: RoFormer, https://arxiv.org/abs/2104.09864

Two interfaces live here, and every model in this repo uses one of them:

- :func:`precompute_freqs_cis` + :func:`apply_rope` — the llama-style functional
  form. Angles are computed once and cached as buffers on the model, then
  applied to q/k in ``(batch, seq_len, num_heads, head_dim)`` layout.
- :class:`RoPE` — a module that computes its own angles on the fly and can
  rotate only the first ``dims`` features. Used by phi, where the checkpoint
  only rotates part of each head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tuple[Tensor, Tensor]:
    """Precompute the cos/sin rotation angles for positions ``0..end``.

    Returns two ``(end, dim // 2)`` tensors, meant to be registered as buffers
    and sliced per forward pass.
    """
    # torch.arange(0, dim, 2) -> 2(i-1)/d for i = 1, 2, ..., d//2
    # [: (dim // 2)] truncates the odd-dim case
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # one angle per (position, frequency)

    # e^(i.t) = cos(t) + i.sin(t)
    return torch.cos(freqs), torch.sin(freqs)


def reshape_for_broadcast(freqs_cis: Tensor, x: Tensor) -> Tensor:
    """Reshape ``(seq_len, head_dim // 2)`` angles to broadcast against ``x``."""
    ndim = x.dim()
    assert ndim > 1
    assert freqs_cis.shape == (
        x.shape[1],
        x.shape[-1],
    ), f"{freqs_cis.shape=}, {(x.shape[1], x.shape[-1])=}"

    # Keep the sequence (dim 1) and frequency (last) dims; broadcast the rest.
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rope(q: Tensor, k: Tensor, cis: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """Rotate ``q`` and ``k`` by the precomputed angles in ``cis``.

    Both tensors are ``(batch, seq_len, num_heads, head_dim)``. ``cis`` is the
    ``(cos, sin)`` pair from :func:`precompute_freqs_cis`, already sliced to the
    positions being processed.

    The trick: read each pair of adjacent features as one complex number,
    ``[x, y, x1, y1, ...] -> x + iy, x1 + iy1``. Multiplying a complex number by
    ``cos + i.sin`` rotates it, so the whole vector rotates in chunks of two.
    """
    _, seq_len, _, _ = q.shape

    freqs_cos, freqs_sin = cis
    freqs_cos, freqs_sin = freqs_cos[:seq_len], freqs_sin[:seq_len]

    # (..., n) -> (..., n // 2, 2), i.e. (B,T,nh,C) -> (B,T,nh,C//2,2)
    q_cis = q.float().reshape(q.shape[:-1] + (-1, 2))
    k_cis = k.float().reshape(k.shape[:-1] + (-1, 2))

    # Split the trailing pair into real and imaginary parts.
    xq_r, xq_i = q_cis.unbind(-1)
    xk_r, xk_i = k_cis.unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)  # (1, T, 1, C//2)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)
    # with a, b = real, imaginary part of q/k and c, d = cos, sin
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Restack the pairs and flatten back: [r, i, r2, i2, ...]
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(q), xk_out.type_as(q)


class RoPE(nn.Module):
    """Rotary position embedding as a module, with partial-rotation support.

    The default implementation rotates feature pairs that are half the rotated
    width apart; ``traditional=True`` rotates adjacent pairs instead, which is
    slightly less efficient. Features past ``dims`` are passed through unchanged.

    Args:
        dims: Number of leading features to rotate. The rest is left alone.
        traditional: Rotate adjacent pairs rather than half-stride pairs.
        base: Base for the angular frequency of each dimension.
        scale: Scale applied to positions before computing angles.
    """

    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        if dims < 2 or dims % 2 != 0:
            # An odd width would split into halves of different sizes and
            # silently broadcast into the wrong shape.
            raise ValueError(f"RoPE dims must be a positive even number, got {dims}")
        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = scale

    def extra_repr(self) -> str:
        return f"{self.dims}, traditional={self.traditional}"

    def _compute_rope(self, costheta: Tensor, sintheta: Tensor, x: Tensor) -> Tensor:
        x1 = x[..., : self.dims // 2]
        x2 = x[..., self.dims // 2 : self.dims]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            return torch.cat([rx1, rx2, x[..., self.dims :]], dim=-1)
        return torch.cat([rx1, rx2], dim=-1)

    def _compute_traditional_rope(self, costheta: Tensor, sintheta: Tensor, x: Tensor) -> Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            raise NotImplementedError("RoPE doesn't implement partial traditional application")

        return torch.cat([rx1[..., None], rx2[..., None]], dim=-1).flatten(-2)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-2], shape[-1])
        num_positions = x.shape[1] + offset
        costheta, sintheta = self.create_cos_sin_theta(
            num_positions,
            self.dims,
            offset=offset,
            base=self.base,
            scale=self.scale,
            dtype=x.dtype,
            device=x.device,
        )

        rope = self._compute_traditional_rope if self.traditional else self._compute_rope
        return rope(costheta, sintheta, x).reshape(shape)

    @staticmethod
    def create_cos_sin_theta(
        num_positions: int,
        dims: int,
        offset: int = 0,
        base: float = 10000,
        scale: float = 1.0,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:
        half_dims = dims // 2
        positions = torch.arange(offset, num_positions, dtype=dtype, device=device) * scale
        freqs = torch.exp(
            -torch.arange(0.0, half_dims, dtype=dtype, device=device) * (math.log(base) / half_dims)
        )
        theta = positions.reshape(-1, 1) * freqs.reshape(1, -1)
        return torch.cos(theta), torch.sin(theta)
