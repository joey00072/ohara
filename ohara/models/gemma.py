from __future__ import annotations

import os
import math
import json
from dataclasses import dataclass


import torch
import torch.nn as nn
import torch.nn.functional as F


from torch import Tensor
from safetensors import safe_open
from ohara.utils.load import download_hf_model

from tqdm import tqdm
from typing import Tuple

from ohara.modules.mlp import GEGLU
from ohara.modules.norm import RMSNorm
from ohara.embedings_pos.rotatry import precompute_freqs_cis
from ohara.embedings_pos.rotatry import apply_rope


@dataclass
class GemmaConfig:
    vocab_size: int = 51200
    seq_len: int = 2048
    d_model: int = 2048
    intermediate_size = 16 * 2048
    num_heads: int = 32
    num_kv_heads: int = 1
    num_layers: int = 32
    dropout: float = 0.2
    multiple_of: int = 4
    bias: bool = True
    eps: float = 1e-5
    rotary_dim: float = 0.4


class GemmaAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.d_model = d_model
        self.head_dim = self.d_model // num_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.scaling: float = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(
            self.d_model, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        )

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        input_shape = x.shape
        assert len(input_shape) == 3

        batch_size, seq_len, d_model = x.shape

        qkv: Tensor = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q: Tensor = q.view(batch_size, -1, self.num_heads, self.head_dim)
        k: Tensor = k.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        v: Tensor = v.view(batch_size, -1, self.num_kv_heads, self.head_dim)

        # Positional embedding.
        q = apply_rope(q, freqs_cis=freqs_cis)
        k = apply_rope(k, freqs_cis=freqs_cis)

        # TODO: add code for kv chache

        # Grouped Query Attention
        if self.num_kv_heads != self.num_heads:
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(value, self.num_queries_per_kv, dim=2)

        q = q.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        attn_mtx = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        attn_mtx = attn_mtx + mask
        attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(q)

        output = torch.matmul(attn_mtx, v)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        return output


class Block(nn.Module):
    def __init__(
        self,
        config: GemmaConfig,
    ):
        super().__init__()
        self.self_attn = GemmaAttention(
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
        )
        self.mlp = GEGLU(
            dim=config.d_model,
            hidden_dim=config.intermediate_size,
        )
        self.ln1 = RMSNorm(config.d_model, eps=config.eps)
        self.ln2 = RMSNorm(config.d_model, eps=config.eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.self_attn(x=self.ln1(x), freqs_cis=freqs_cis, mask=mask)
        x = x + self.mlp(self.ln2(x))

        return x


class Gemma(nn.Module):
    def __init__(self, model_args: GemmaConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = model_args

        self.token_emb = nn.Embedding(model_args.vocab_size, model_args.d_model)

        self.layers = nn.ModuleList(
            [Block(model_args) for _ in range(model_args.num_layers)]
        )

        self.norm = nn.LayerNorm(model_args.d_model)
        self.vocab_proj = nn.Linear(
            model_args.d_model, model_args.vocab_size, bias=False
        )

        self.token_emb.weight = self.vocab_proj.weight

        self.cos, self.sin = precompute_freqs_cis(
            model_args.d_model // model_args.num_heads, model_args.seq_len * 2
        )

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full(
                (1, 1, model_args.seq_len, model_args.seq_len), float("-inf")
            )
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def forward(self, x: torch.Tensor):
        batch, seqlen = x.shape
        x = self.token_emb(x)

        device = self.token_emb.weight.device
        freqs_cis = self.cos[:seqlen].to(device), self.sin[:seqlen].to(device)

        for layer in self.layers:
            x = layer(x, self.mask, freqs_cis)

        x = self.norm(x)
        x = self.vocab_proj(x)
        return x
