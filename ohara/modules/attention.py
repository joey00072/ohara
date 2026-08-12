"""Standalone causal self-attention block.

Models in :mod:`ohara.models` build attention inline so each file reads
top-to-bottom, but experiments that only need a drop-in attention module can use
this one.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ohara.embeddings_pos.rotary import apply_rope

TensorTuple = tuple[Tensor, ...]


class CausalAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        res_dropout: float = 0.0,
        idx: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads: int = num_heads
        self.head_dim: int = d_model // num_heads
        self.idx = idx

        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_dropout = nn.Dropout(res_dropout)

        self.flash_attn: bool = hasattr(F, "scaled_dot_product_attention")

        self.reset_parameters()

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        freqs_cis: TensorTuple | None = None,
        verbose: bool = False,
        **kwargs: dict,
    ) -> Tensor | tuple[Tensor, dict]:
        """Attend over ``x``.

        With ``verbose=True`` the attention matrix is returned alongside the
        output, which forces the slow path since flash attention never
        materializes it.
        """
        batch, seq_len, d_model = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # (B, T, C) -> (B, T, num_heads, head_dim)
        shape = (batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(shape)
        q = q.view(shape)
        v = v.view(shape)

        if freqs_cis is not None:
            q, k = apply_rope(q, k, freqs_cis)

        k = k.transpose(1, 2)  # (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mtx = None
        if self.flash_attn and not verbose:
            output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            else:
                causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
                attn_mtx = attn_mtx.masked_fill(~causal, float("-inf"))
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            output = torch.matmul(self.attn_dropout(attn_mtx), v)

        # Concatenate the heads back into the residual stream.
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = self.proj(output)
        output = self.res_dropout(output)

        if verbose:
            return output, {"idx": self.idx, "attn_mtx": attn_mtx}
        return output

    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        init_std = init_std or (self.head_dim ** (-0.5))

        for w in (self.key, self.query, self.value):
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.proj.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )


# Historic name kept because it is the one this repo has always used.
CasualAttention = CausalAttention
Attention = CausalAttention
