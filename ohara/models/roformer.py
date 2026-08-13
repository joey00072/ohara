"""RoFormer: a GPT-shaped decoder that uses rotary position embeddings.

Paper: https://arxiv.org/abs/2104.09864

This is the same block layout as :mod:`ohara.models.gpt`, with the learned
position embedding replaced by RoPE applied to q/k inside attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ohara.embeddings_pos.rotary import apply_rope, precompute_freqs_cis
from ohara.modules.mlp import MLP


@dataclass
class Config:
    vocab_size: int = 65
    seq_len: int = 64
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.2
    multiple_of: int = 4
    bias: bool = True


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = d_model // config.num_heads

        self.key = nn.Linear(d_model, d_model, bias=config.bias)
        self.query = nn.Linear(d_model, d_model, bias=config.bias)
        self.value = nn.Linear(d_model, d_model, bias=config.bias)
        self.proj = nn.Linear(d_model, d_model, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)

        self.flash_attn = hasattr(F, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # RoPE expects (B, T, num_heads, head_dim).
        shape = (batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(shape)
        q = q.view(shape)
        v = v.view(shape)

        q, k = apply_rope(q, k, freqs_cis)

        k = k.transpose(1, 2)  # (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn:
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
            attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx = self.attn_dropout(attn_mtx)
            output = torch.matmul(attn_mtx, v)

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = self.proj(output)
        return self.res_dropout(output)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.attn = Attention(config)
        self.ff = MLP(
            dim=config.d_model,
            multiple_of=config.multiple_of,
            dropout=config.dropout,
            activation_fn="gelu",
            bias=config.bias,
        )

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, x, freqs_cis, mask=None):
        x = x + self.attn(self.norm1(x), freqs_cis, mask)
        x = x + self.ff(self.norm2(x))
        return x


class RoFormer(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        head_dim = config.d_model // config.num_heads
        if head_dim % 2 != 0:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        self.config = config

        self.word_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        self.norm = nn.LayerNorm(config.d_model)
        self.vocab_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, config.seq_len * 2)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        if hasattr(F, "scaled_dot_product_attention"):
            self.mask = None
        else:
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full((1, 1, config.seq_len, config.seq_len), float("-inf"))
            self.register_buffer("mask", torch.triu(mask, diagonal=1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("input token IDs must have shape (batch, sequence)")
        seq_len = x.size(1)
        if seq_len > self.config.seq_len:
            raise ValueError("input exceeds the configured seq_len")

        x = self.word_emb(x)
        freqs_cis = (self.freqs_cos[:seq_len], self.freqs_sin[:seq_len])

        for layer in self.layers:
            x = layer(x, freqs_cis, self.mask)

        x = self.norm(x)
        return self.vocab_proj(x)
