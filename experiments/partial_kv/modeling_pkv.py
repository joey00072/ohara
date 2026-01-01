import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
from ohara.modules.mlp import GLU, MLP,MLP_MAP
from ohara.modules.norm import RMSNorm

from ohara.embeddings_pos.rotary import precompute_freqs_cis
from ohara.embeddings_pos.rotary import apply_rope

from huggingface_hub import PyTorchModelHubMixin

from collections import OrderedDict

from pkv import Config, KAttention, Attention, PartialKVAttention


ATTENTION_MAP = {"attention": Attention, "k_is_v": KAttention, "partial_kv": PartialKVAttention}



class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.attn = ATTENTION_MAP[config.attention_type](config)
        self.ff = MLP_MAP[config.mlp](
            dim=config.d_model,
            hidden_dim=config.hidden_dim,
            activation_fn=config.activation,
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

        
        cos, isin = precompute_freqs_cis(config.rope_head_dim, config.seq_len * 2)
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
        activation="relu_squared",
        mlp="GLU",
    )

    model = ModelingLM(config).eval()
    print(model)
