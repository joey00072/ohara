import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
from ohara.modules.mlp import GLU, MLP
from ohara.modules.norm import RMSNorm

from ohara.embedings_pos.rotatry import precompute_freqs_cis
from ohara.embedings_pos.rotatry import apply_rope

from typing import Callable

from huggingface_hub import PyTorchModelHubMixin

from collections import OrderedDict

import lightning as L


@dataclass
class Config(OrderedDict):
    vocab_size: int
    seq_len: int

    d_model: int
    hidden_dim: int

    num_heads: int
    num_kv_heads: int = 0

    num_layers: int = 4

    dropout: float = 0.2
    bias: bool = False
    weight_tying: bool = False

    activation: Callable = "silu"  # "relu", "gelu", "silu" etc
    mlp: str = "GLU"  # MLP or GLU

    exp: int = 2


MLP_BLOCK = {"MLP": MLP, "GLU": GLU}


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.num_kv_heads = config.num_heads if config.num_kv_heads == 0 else config.num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.key = nn.Linear(config.exp * d_model, self.head_dim * self.num_heads, config.bias)
        self.query = nn.Linear(config.exp * d_model, self.head_dim * self.num_kv_heads, config.bias)
        self.value = nn.Linear(config.exp * d_model, self.head_dim * self.num_kv_heads, config.bias)
        self.proj = nn.Linear(d_model, d_model * config.exp, config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)

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
    def __init__(self, config: Config):
        super().__init__()

        self.attn = Attention(config)
        self.ff = MLP_BLOCK[config.mlp](
            dim=config.d_model,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            bias=config.bias,
        )

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

    def forward(self, x, mask, freqs_cis):
        x = x + self.attn(self.norm1(x), mask, freqs_cis)
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_layers)])

        self.norm = RMSNorm(config.d_model)
        self.vocab_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.weight_tying:
            self.token_emb.weight = self.vocab_proj.weight

        cos, isin = precompute_freqs_cis(config.d_model // config.num_heads, config.seq_len * 2)
        self.register_buffer("freq_cos", cos)
        self.register_buffer("freq_sin", isin)

        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full((1, 1, config.seq_len, config.seq_len), float("-inf"))
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


class ModelingLM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.model = Transformer(self.config)

    def forward(self, x: torch.Tensor):
        return self.model(x)


if __name__ == "__main__":
    config = Config(
        vocab_size=10,
        seq_len=10,
        d_model=128,
        hidden_dim=128,
        num_heads=4,
        num_kv_heads=0,
        num_layers=4,
        dropout=0.2,
        bias=False,
        weight_tying=False,
        activation="silu",
        mlp="GLU",
    )

    model = ModelingLM(config).eval()
    print(model)
    hf_name = "joey00072/test_001"
    model.push_to_hub(hf_name)
    x = torch.arange(10).reshape(1, 10)
    print(model(x).sum(dim=-1))
    new_model = ModelingLM.from_pretrained(hf_name).eval()
    print(new_model(x).sum(dim=-1))

    # assert torch.isclose(model.model.vocab_proj.weight, new_model.model.vocab_proj.weight)
