import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
from ohara.modules.mlp import SwiGLU
from ohara.modules.norm import RMSNorm

from ohara.embedings_pos.rotatry import precompute_freqs_cis
from ohara.embedings_pos.rotatry import apply_rope

from typing import Callable

@dataclass
class Config:
    vocab_size: int = 65
    seq_len: int = 64
    d_model: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    num_kv_heads: int = 0
    num_layers: int = 4
    dropout: flaot = 0.2
    multiple_of: int = 4
    bias: int = False
    weight_tying: bool = False


def squared_relu(x):
    x = F.relu(x)
    return x**2

MAP = {
        "silu": F.silu,
        "relu": F.relu,
        "squared_relu": squared_relu,
      }   

class GLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        dropout: float | None = None,
        activation:str = "silu",
        bias: bool = False,
    ):
        super().__init__()

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x
        
        self.activation = MAP[activation]

    def forward(self, x):
        return self.dropout(self.w2(self.activation(self.w1(x)) * self.w3(x)))


class MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        multiple_of: int = 4,
        dropout: float | None = None,
        activation:str = "silu",
        bias: bool = True,
    ):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.activation = MAP[activation]

        self.dropout = nn.Dropout(dropout) if dropout else lambda x: x

    def forward(self, x):
        x = self.w1(x)
        x = self.activation_fn(x)
        x = self.w2(x)
        x = self.dropout(x)
        return x

class Attention(nn.Module):
    def __init__(self, model_args: Config):
        super().__init__()
        d_model = model_args.d_model
        self.num_heads = model_args.num_heads
        self.head_dim = model_args.d_model // model_args.num_heads
        self.num_kv_heads = (
            model_args.num_heads if model_args.num_kv_heads == 0 else model_args.num_kv_heads
        )
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.key = nn.Linear(d_model, self.head_dim * self.num_heads)
        self.query = nn.Linear(d_model, self.head_dim * self.num_kv_heads)
        self.value = nn.Linear(d_model, self.head_dim * self.num_kv_heads)
        self.proj = nn.Linear(d_model, d_model, model_args.bias)
            
        self.attn_dropout = nn.Dropout(model_args.dropout)
        self.res_dropout = nn.Dropout(model_args.dropout)

        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor  # type hint for lsp
        q: torch.Tensor  # ignore
        v: torch.Tensor

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(
            batch, seq_len, self.num_heads, self.head_dim
        )  # shape = (B, seq_len, num_heads, head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        # Grouped Query Attention
        if self.num_kv_heads != self.num_heads:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

        k = k.transpose(1, 2)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,  # order impotent
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx = self.attn_dropout(attn_mtx)

            output = torch.matmul(attn_mtx, v)  # (batch, n_head, seq_len, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


class Block(nn.Module):
    def __init__(self, model_args: Config):
        super().__init__()

        self.attn = Attention(model_args)
        self.ff = SwiGLU(
            dim=model_args.d_model,
            hidden_dim=model_args.hidden_dim,
            dropout=model_args.dropout,
            bias=model_args.bias,
        )

        self.norm1 = RMSNorm(model_args.d_model)
        self.norm2 = RMSNorm(model_args.d_model)

    def forward(self, x, mask, freqs_cis):
        x = x + self.attn(self.norm1(x), mask, freqs_cis)
        x = x + self.ff(self.norm2(x))
        return x


class GPTLM(nn.Module):
    def __init__(self, model_args: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = model_args

        self.token_emb = nn.Embedding(model_args.vocab_size, model_args.d_model)

        self.layers = nn.ModuleList([Block(model_args) for _ in range(model_args.num_layers)])

        self.norm = RMSNorm(model_args.d_model)
        self.vocab_proj = nn.Linear(model_args.d_model, model_args.vocab_size, bias=False)

        if model_args.weight_tying:
            self.token_emb.weight = self.vocab_proj.weight

        cos,isin = precompute_freqs_cis(
            model_args.d_model // model_args.num_heads, model_args.seq_len * 2
        )
        self.register_buffer("freq_cos",cos)
        self.register_buffer("freq_sin",isin)

        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full((1, 1, model_args.seq_len, model_args.seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        batch, seqlen = x.shape
        x = self.token_emb(x)
        device = self.token_emb.weight.device
        freqs_cis = self.freq_cos[:seqlen], self.freq_sin[:seqlen]
        
        for layer in self.layers:
            x = layer(x, self.mask, freqs_cis)

        x = self.norm(x)
        x = self.vocab_proj(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
