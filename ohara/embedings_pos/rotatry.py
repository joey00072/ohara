from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

import math


class RotatryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        cis = RotatryEmbedding.precompute_freqs_cis(dim, max_seq_len)
        self.register_buffer("cos", cis[0])
        self.register_buffer("sin", cis[1])

    @staticmethod
    def rotate_half(x: Tensor):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @staticmethod
    def apply_rotary_pos_emb(
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        position_ids: int | None = None,
        unsqueeze_dim: int = 1,
    ):
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        x_embed = (x * cos) + (RotatryEmbedding.rotate_half(x) * sin)
        return x_embed

    def forward(self, x, position_ids=None):
        return self.apply_rotary_pos_emb(x, self.cos, self.sin, position_ids=position_ids)

    @staticmethod
    def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        # [: (dim // 2)] for odd number truncation
        # torch.arange(0, dim, 2) -> 2(i-1)//d while i= 1,2,..,(d//2)

        t = torch.arange(end, device=freqs.device)
        freqs = torch.outer(t, freqs).float()  # gives diffrent angle vector

        # e^it = cos(t) + i sin(t)
        freqs_cos = torch.cos(freqs)  # real
        freqs_sin = torch.sin(freqs)  # imaginary
        return freqs_cos, freqs_sin


# rotary embedding
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # [: (dim // 2)] for odd number truncation
    # torch.arange(0, dim, 2) -> 2(i-1)//d while i= 1,2,..,(d//2)

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()  # gives diffrent angle vector

    # e^it = cos(t) + i sin(t)
    freqs_cos = torch.cos(freqs)  # real
    freqs_sin = torch.sin(freqs)  # imaginary
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.dim()
    assert 1 < ndim
    assert freqs_cis.shape == (
        x.shape[1],
        x.shape[-1],
    ), f"{freqs_cis.shape=}, {(x.shape[1], x.shape[-1])=}"

    # keep 2nd (T) and last(freq) dim same else make dim 1 for freq_cis
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # print(shape)
    return freqs_cis.view(shape)


def apply_rope(k, q, cis):
    # Idea suppose vector v = [x,y,x1,y1,...] # v.shape = dim
    # convert vetor into complex num # ie two vec one real, one imagery
    # [x,y,x1,y1,...] -> x+iy, x1+iy1
    # Multiplying by complex num == roatate vector
    # => (x + iy) * (cos + isin) -> x'+iy'
    # restack
    # x'+iy' -> [x',y',x1',y1'...]
    # you roated vector in chunks of two lfg!!!
    _, seq_len, _, _ = q.shape

    freqs_cos, freqs_sin = cis
    freqs_cos, freqs_sin = freqs_cos[:seq_len], freqs_sin[:seq_len]

    #  rehsape a shape (...,n )-> (..., n//2,2)
    q_cis = q.float().reshape(
        q.shape[:-1] + (-1, 2)
    )  # (B,T,nhead,C) -> (B,T,nhead,Cc,2) # Cc = C//2
    k_cis = k.float().reshape(k.shape[:-1] + (-1, 2))  # (B,T,nhead,C) -> (B,T,nhead,Cc,2)

    xq_r, xq_i = q_cis.unbind(-1)  # (B,T,nhead,Cc,2) -> ((B,T,Cc), (B,T,Cc)) split into two tuple
    xk_r, xk_i = k_cis.unbind(-1)  # (B,T,nhead,Cc,2) -> ((B,T,Cc), (B,T,Cc))

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)  # freqs.shape = (1,T,1,Cc)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # e+if = (a+ib) * (c+di) = (ac-bd) + i (ad+bc)
    # a = xq_r , b = xq_i
    # c = fcos , d = fsin
    # ...
    # e = (ac-bd) = xq_r * freqs_cos - xq_i * freqs_sin
    # f = (c+di)  = xq_r * freqs_sin + xq_i * freqs_cos

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin  # (ac-bd)   # shape =  # (B,T,nhead,Cc)
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos  # (ad+bc) * i
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin  # (ac-bd)
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos  # (ad+bc) * i

    # now we stack r,i -> [r,i,r2,i2]
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1)  # (B,T,nhead,Cc,2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1)  # (B,T,nhead,Cc,2)

    # flatten last two dimensions
    xq_out = xq_out.flatten(3)  # (B,T,nhead,C)
    xk_out = xk_out.flatten(3)  # (B,T,nhead,C)

    return xq_out.type_as(q), xk_out.type_as(q)


class RoPE(nn.Module):
    """Implements the rotary positional encoding.

    The traditional implementation rotates consecutive pairs of elements in the
    feature dimension while the default implementation rotates pairs with
    stride half the feature dimensions for efficiency.

    For more details see `RoFormer: Enhanced Transformer with Rotary Position
    Embedding <https://arxiv.org/abs/2104.09864>`_.

    Args:
        dims (int): The feature dimensions to be rotated. If the input feature
            is larger than dims then the rest is left unchanged.
        traditional (bool, optional): If set to True choose the traditional
            implementation which is slightly less efficient. Default: ``False``.
        base (float, optional): The base used to compute angular frequency for
            each dimension in the positional encodings. Default: ``10000``.
        scale (float, optional): The scale used to scale the positions. Default: ``1.0``.
    """

    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
    ):
        super().__init__()

        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = scale

    def _extra_repr(self):
        return f"{self.dims}, traditional={self.traditional}"

    def _compute_rope(self, costheta, sintheta, x):
        x1 = x[..., : self.dims // 2]
        x2 = x[..., self.dims // 2 : self.dims]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            rx = torch.concatenate([rx1, rx2, x[..., self.dims :]], axis=-1)
        else:
            rx = torch.concatenate([rx1, rx2], axis=-1)

        return rx

    def _compute_traditional_rope(self, costheta, sintheta, x):
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            raise NotImplementedError("RoPE doesn't implement partial traditional application")

        rx = torch.cat([rx1[..., None], rx2[..., None]], axis=-1)

        return rx

    def forward(self, x: Tensor, offset: int = 0):
        shape = x.shape
        x = x.reshape(-1, shape[-2], shape[-1])
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N, self.dims, offset=offset, base=self.base, scale=self.scale, dtype=x.dtype
        )

        rope = self._compute_traditional_rope if self.traditional else self._compute_rope
        rx = rope(costheta, sintheta, x)

        return torch.reshape(rx, shape)

    @staticmethod
    def create_cos_sin_theta(
        N: int,
        D: int,
        offset: int = 0,
        base: float = 10000,
        scale: float = 1.0,
        dtype=torch.float32,
    ):
        D = D // 2
        positions = torch.arange(offset, N, dtype=dtype) * scale
        freqs = torch.exp(-torch.arange(0.0, D, dtype=dtype) * (math.log(base) / D))
        theta = torch.reshape(positions, (-1, 1)) * torch.reshape(freqs, (1, -1))
        return torch.cos(theta), torch.sin(theta)
