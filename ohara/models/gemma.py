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
from ohara.embeddings_pos.rotary import precompute_freqs_cis
from ohara.embeddings_pos.rotary import apply_rope


@dataclass
class GemmaConfig:
    vocab_size: int = 51200
    max_sequence_length: int = 2048
    hidden_size: int = 2048
    intermediate_size = 16 * 2048
    num_attention_heads: int = 32
    num_key_value_heads: int = 1
    num_hidden_layers: int = 32
    dropout: float = 0.2
    multiple_of: int = 4
    bias: bool = True
    eps: float = 1e-5
    rotary_dim: float = 0.4


class GemmaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
    ):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        assert self.num_attention_heads % self.num_key_value_heads == 0
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads

        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // num_attention_heads

        self.q_size = self.num_attention_heads * self.head_dim
        self.kv_size = self.num_key_value_heads * self.head_dim

        self.scaling: float = self.head_dim**-0.5

        self.qkv_proj = nn.Linear(
            self.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
        )

        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        input_shape = x.shape
        assert len(input_shape) == 3

        batch_size, seq_len, hidden_size = x.shape

        qkv: Tensor = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q: Tensor = q.view(batch_size, -1, self.num_attention_heads, self.head_dim)
        k: Tensor = k.view(batch_size, -1, self.num_key_value_heads, self.head_dim)
        v: Tensor = v.view(batch_size, -1, self.num_key_value_heads, self.head_dim)

        # Positional embedding.
        q, k = apply_rope(q, k, freqs_cis)

        # TODO: add code for kv chache

        # Grouped Query Attention
        if self.num_key_value_heads != self.num_attention_heads:
            key = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
            value = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)
        else:
            key = k
            value = v

        q = q.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        attn_mtx = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if mask is not None:
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
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
        )
        self.mlp = GEGLU(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
        )
        self.ln1 = RMSNorm(config.hidden_size, eps=config.eps)
        self.ln2 = RMSNorm(config.hidden_size, eps=config.eps)

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

        self.token_emb = nn.Embedding(model_args.vocab_size, model_args.hidden_size)

        self.layers = nn.ModuleList(
            [Block(model_args) for _ in range(model_args.num_hidden_layers)]
        )

        self.norm = nn.LayerNorm(model_args.hidden_size)
        self.vocab_proj = nn.Linear(model_args.hidden_size, model_args.vocab_size, bias=False)

        self.token_emb.weight = self.vocab_proj.weight

        self.cos, self.sin = precompute_freqs_cis(
            model_args.hidden_size // model_args.num_attention_heads,
            model_args.max_sequence_length * 2,
        )

        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
        mask = torch.full(
            (1, 1, model_args.max_sequence_length, model_args.max_sequence_length),
            float("-inf"),
        )
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor):
        batch, seqlen = x.shape
        x = self.token_emb(x)

        device = self.token_emb.weight.device
        freqs_cis = self.cos[:seqlen].to(device), self.sin[:seqlen].to(device)

        for layer in self.layers:
            x = layer(x, freqs_cis, self.mask)

        x = self.norm(x)
        x = self.vocab_proj(x)
        return x
