"""A minimal GPT: learned position embeddings, LayerNorm, causal attention.

The feed-forward block is selectable, so this covers both the original GPT-2
shape (``mlp="mlp"``, a GELU/SiLU MLP) and the SwiGLU variant (``mlp="swiglu"``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ohara.modules.mlp import MLP_MAP


@dataclass
class Config:
    vocab_size: int = 65
    max_sequence_length: int = 64
    hidden_size: int = 128
    num_attention_heads: int = 4
    num_hidden_layers: int = 4
    dropout: float = 0.2
    multiple_of: int = 4
    bias: bool = False
    mlp: str = "mlp"  # any key of ohara.modules.mlp.MLP_MAP


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = hidden_size // config.num_attention_heads

        self.key = nn.Linear(hidden_size, hidden_size, bias=config.bias)
        self.query = nn.Linear(hidden_size, hidden_size, bias=config.bias)
        self.value = nn.Linear(hidden_size, hidden_size, bias=config.bias)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)

        self.flash_attn = hasattr(F, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, hidden_size = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # (B, T, C) -> (B, num_heads, T, head_dim)
        shape = (batch, seq_len, self.num_attention_heads, self.head_dim)
        k = k.view(shape).transpose(1, 2)
        q = q.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)

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
            output = torch.matmul(attn_mtx, v)  # (B, num_heads, T, head_dim)

        # Concatenate the heads back into the residual stream.
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, hidden_size)
        output = self.proj(output)
        return self.res_dropout(output)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.attn = Attention(config)
        self.ff = MLP_MAP[config.mlp](
            dim=config.hidden_size,
            multiple_of=config.multiple_of,
            dropout=config.dropout,
            bias=config.bias,
        )

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if config.mlp not in MLP_MAP:
            raise ValueError(f"mlp must be one of {sorted(MLP_MAP)}, got {config.mlp!r}")
        self.config = config

        self.word_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_sequence_length, config.hidden_size)

        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])

        self.norm = nn.LayerNorm(config.hidden_size)
        self.vocab_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if hasattr(F, "scaled_dot_product_attention"):
            self.mask = None
        else:
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full(
                (1, 1, config.max_sequence_length, config.max_sequence_length), float("-inf")
            )
            self.register_buffer("mask", torch.triu(mask, diagonal=1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("input token IDs must have shape (batch, sequence)")
        seq_len = x.size(1)
        if seq_len > self.config.max_sequence_length:
            raise ValueError("input exceeds max_sequence_length")

        positions = torch.arange(seq_len, device=x.device)
        x = self.word_emb(x) + self.pos_emb(positions)

        for layer in self.layers:
            x = layer(x, self.mask)

        x = self.norm(x)
        return self.vocab_proj(x)
