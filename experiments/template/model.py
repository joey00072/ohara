from typing import assert_type
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
from ohara.modules.mlp import GLU, MLP, ACT2FN
from ohara.modules.norm import RMSNorm

from ohara.embeddings_pos.rotary import precompute_freqs_cis
from ohara.embeddings_pos.rotary import apply_rope

from huggingface_hub import PyTorchModelHubMixin

from collections import OrderedDict


@dataclass
class Config(OrderedDict):
    vocab_size: int
    max_sequence_length: int

    hidden_size: int
    intermediate_size: int

    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int = 0

    num_hidden_layers: int = 4

    dropout: float = 0.2
    bias: bool = False
    weight_tying: bool = False

    activation: str = "silu"  # "relu", "gelu", "silu" etc
    mlp: str = "GLU"  # MLP or GLU

    use_spda: bool = False


MLP_BLOCK = {"MLP": MLP, "GLU": GLU}


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        hidden_size = config.hidden_size
        self.hidden_size = hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads

        self.key = nn.Linear(hidden_size, self.head_dim * self.num_key_value_heads, config.bias)
        self.query = nn.Linear(hidden_size, self.head_dim * self.num_attention_heads, config.bias)
        self.value = nn.Linear(hidden_size, self.head_dim * self.num_key_value_heads, config.bias)
        self.proj = nn.Linear(self.head_dim * self.num_attention_heads, hidden_size, config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)

        self.flash_attn = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention") and not config.use_spda
        )

        self.reset_parameters()

    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        init_std = init_std or (self.head_dim ** (-0.5))

        for w in [self.key, self.query, self.value]:
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

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor  # type hint for lsp
        q: torch.Tensor  # ignore
        v: torch.Tensor

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(
            batch, seq_len, self.num_key_value_heads, self.head_dim
        )  # shape = (B, seq_len, num_key_value_heads, head_dim)
        q = q.view(batch, seq_len, self.num_attention_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_key_value_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        # Grouped Query Attention
        if self.num_key_value_heads != self.num_attention_heads:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

        k = k.transpose(1, 2)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,  # order important
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
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.attn = Attention(config)
        self.ff: MLP | GLU = MLP_BLOCK[config.mlp](
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            activation_fn=config.activation,
            dropout=config.dropout,
            bias=config.bias,
        )

        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

    def forward(self, x, mask, freqs_cis):
        x = x + self.attn(self.norm1(x), mask, freqs_cis)
        x = x + self.ff(self.norm2(x))
        return x

    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        self.attn.reset_parameters(init_std, factor)
        self.ff.reset_parameters(init_std, factor)


class Transformer(nn.Module):
    def __init__(self, config: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])

        self.norm = RMSNorm(config.hidden_size)
        self.vocab_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.weight_tying:
            self.token_emb.weight = self.vocab_proj.weight

        cos, isin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, config.max_sequence_length * 2
        )
        self.register_buffer("freq_cos", cos)
        self.register_buffer("freq_sin", isin)

        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full(
                (1, 1, config.max_sequence_length, config.max_sequence_length), float("-inf")
            )
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

    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        layer: Block
        torch.nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.vocab_proj.weight, mean=0.0, std=0.02)
        for layer in self.layers:
            layer.reset_parameters(init_std, factor)
        self.norm.reset_parameters()


class ModelingLM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.model = Transformer(self.config)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, return_outputs: bool = False):
        logits = self.model(x)
        if return_outputs:
            return logits, None
        return logits

    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        self.model.reset_parameters(init_std, factor)


if __name__ == "__main__":
    hidden_size = 128
    num_attention_heads = 4
    config = Config(
        vocab_size=10,
        max_sequence_length=10,
        hidden_size=hidden_size,
        intermediate_size=128,
        head_dim=hidden_size // num_attention_heads,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        num_hidden_layers=4,
        dropout=0.2,
        bias=False,
        weight_tying=False,
        activation="relu_squared",
        mlp="GLU",
    )

    model = ModelingLM(config).eval()
    print(model)
