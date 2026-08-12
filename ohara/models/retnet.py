"""RetNet: retention instead of softmax attention.

Paper: Retentive Network, https://arxiv.org/abs/2307.08621

Retention replaces the softmax with a decay mask (from :class:`XPos`) and a
gated output projection. Only the parallel form is implemented here; the
recurrent and chunked forms are what make RetNet fast at inference and are still
on the list.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ohara.embeddings_pos.xpos import XPos
from ohara.modules.mlp import SwiGLU
from ohara.modules.norm import RMSNorm


@dataclass
class Config:
    vocab_size: int = 65
    seq_len: int = 64
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.2
    multiple_of: int = 4
    bias: bool = False
    eps: float = 1e-5


class Retention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.scaling = self.head_dim**-0.5

        self.key = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.query = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.value = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.gate = nn.Linear(self.d_model, self.d_model, bias=config.bias)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=config.bias)

        self.norm = RMSNorm(self.head_dim, config.eps)

    def forward(self, x: torch.Tensor, decay_mask: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k = self.key(x) * self.scaling
        q = self.query(x)
        v = self.value(x)
        g = self.gate(x)

        # (B, T, C) -> (B, num_heads, T, head_dim)
        shape = (batch, seq_len, self.num_heads, self.head_dim)
        k = k.view(shape).transpose(1, 2)
        q = q.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)

        ret_mtx = torch.matmul(q, k.transpose(2, 3))
        # Normalize before applying decay, as in the reference implementation.
        ret_mtx = ret_mtx / ret_mtx.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)
        ret_mtx = ret_mtx * decay_mask[:, :seq_len, :seq_len]

        output = torch.matmul(ret_mtx, v)  # (B, num_heads, T, head_dim)
        output = self.norm(output)

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        output = output * torch.nn.functional.silu(g)
        return self.proj(output)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.attn = Retention(config)
        self.ff = SwiGLU(
            dim=config.d_model,
            multiple_of=config.multiple_of,
            dropout=config.dropout,
            bias=config.bias,
        )

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

    def forward(self, x, decay_mask):
        x = x + self.attn(self.norm1(x), decay_mask)
        x = x + self.ff(self.norm2(x))
        return x


class RetNet(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.config = config

        self.word_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        self.norm = nn.LayerNorm(config.d_model)
        self.vocab_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # XPos gives the per-head decay mask; the cos/sin pair is only needed by
        # the recurrent form, which is not implemented yet.
        _, decay_mask = XPos(config.d_model, config.num_heads).forward(slen=config.seq_len)
        self.register_buffer("decay_mask", decay_mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("input token IDs must have shape (batch, sequence)")
        if x.size(1) > self.config.seq_len:
            raise ValueError("input exceeds the configured seq_len")

        x = self.word_emb(x)

        for layer in self.layers:
            x = layer(x, self.decay_mask)

        x = self.norm(x)
        return self.vocab_proj(x)
