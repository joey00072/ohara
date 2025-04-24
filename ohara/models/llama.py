import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
from ..modules.mlp import SwiGLU
from ..modules.norm import RMSNorm

from ohara.embedings_pos.rotatry import precompute_freqs_cis
from ohara.embedings_pos.rotatry import apply_rope



@dataclass
class Config:
    vocab_size: int = 65
    seq_len: int = 64
    d_model: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    num_kv_heads: int = 0
    num_layers: int = 4
    dropout: float = 0.2
    multiple_of: int = 4
    bias: int = False
    weight_tying: bool = False
    rope_theta: float = 100000


class KVCache:
    def __init__(self, shape, max_seq_length, idx: int | None = None, device=None, dtype=None):
        self.idx = idx
        self.key: torch.Tensor = torch.zeros(shape, device=device, dtype=dtype)
        self.value: torch.Tensor = torch.zeros(shape, device=device, dtype=dtype)
        self.max_seq_length = max_seq_length

    def forward(self, keys: torch.Tensor, values: torch.Tensor, start_pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, T, _, _ = keys.shape
        self.key[:bsz, start_pos : start_pos + T] = keys
        self.value[:bsz, start_pos : start_pos + T] = values
        keys = self.key[:bsz, : start_pos + T]
        values = self.value[:bsz, : start_pos + T]
        return keys, values


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d_model = cfg.d_model
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.d_model // cfg.num_heads
        self.num_kv_heads = cfg.num_heads if cfg.num_kv_heads == 0 else cfg.num_kv_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.key = nn.Linear(d_model, self.head_dim * self.num_kv_heads, cfg.bias)
        self.query = nn.Linear(d_model, self.head_dim * self.num_heads, cfg.bias)
        self.value = nn.Linear(d_model, self.head_dim * self.num_kv_heads, cfg.bias)
        self.proj = nn.Linear(self.head_dim * self.num_heads, d_model, cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.res_dropout = nn.Dropout(cfg.dropout)

        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis, kv_cache: KVCache | None = None, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor
        q: torch.Tensor
        v: torch.Tensor

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim)
        q = q.view(batch, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Apply RoPE with position offset if using KV cache
        offset = position_ids if kv_cache else 0
        freqs_cos, freqs_sin = freqs_cis
        if offset:
            freqs_cos = freqs_cos[offset:offset + seq_len]
            freqs_sin = freqs_sin[offset:offset + seq_len]
        q, k = apply_rope(q, k, (freqs_cos, freqs_sin))

        # Apply KV cache if provided
        if kv_cache is not None:
            k, v = kv_cache.forward(k, v, position_ids)

        # Grouped Query Attention
        if self.num_kv_heads != self.num_heads:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
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
                attn_mtx = attn_mtx + mask[:, :, :seq_len, :k.size(2)]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx = self.attn_dropout(attn_mtx)
            output = torch.matmul(attn_mtx, v)

        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.head_dim * self.num_heads)
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.attn = Attention(cfg)
        self.ff = SwiGLU(
            dim=cfg.d_model,
            hidden_dim=cfg.hidden_dim,
            dropout=cfg.dropout,
            bias=cfg.bias,
        )

        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)

    def forward(self, x, mask, freqs_cis, kv_cache: KVCache | None = None, position_ids: torch.Tensor | None = None):
        x = x + self.attn(self.norm1(x), mask, freqs_cis, kv_cache, position_ids)
        x = x + self.ff(self.norm2(x))
        return x


class LLAMA(nn.Module):
    def __init__(self, cfg: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])

        self.norm = RMSNorm(cfg.d_model)
        self.vocab_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.weight_tying:
            self.token_emb.weight = self.vocab_proj.weight

        cos, isin = precompute_freqs_cis(cfg.d_model // cfg.num_heads, cfg.seq_len * 2)
        self.register_buffer("freq_cos", cos)
        self.register_buffer("freq_sin", isin)

        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full((1, 1, cfg.seq_len, cfg.seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor, kv_cache: list[KVCache] | None = None, position_ids: torch.Tensor | None = None):
        batch, seqlen = x.shape
        x = self.token_emb(x)
        device = self.token_emb.weight.device
        freqs_cis = self.freq_cos[:seqlen], self.freq_sin[:seqlen]

        # Handle KV caching mask adjustment
        mask = self.mask
        if kv_cache is not None:
            x = x[:, position_ids:]
            mask = None

        # Forward through layers with KV cache
        for idx, layer in enumerate(self.layers):
            cache = kv_cache[idx] if kv_cache is not None else None
            x = layer(x, mask, freqs_cis, cache, position_ids)

        x = self.norm(x)
        x = self.vocab_proj(x)
        return x

    def build_kv_cache(self) -> list[KVCache]:
        """Build an empty KV cache suitable for the model's configuration."""
        shape = (
            1,
            self.config.seq_len,
            self.config.num_heads,
            self.config.d_model // self.config.num_heads,
        )
        kv_cache = []
        dtype = self.token_emb.weight.dtype
        device = self.token_emb.weight.device

        for idx in range(self.config.num_layers):
            kv_cache.append(KVCache(shape, self.config.seq_len, idx, device=device, dtype=dtype))
        return kv_cache

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    @classmethod
    def from_pretrained(cls, hf_name: str):
        from ohara.utils.load import download_hf_model
        import json
        
        path_name = download_hf_model(hf_name)
        with open(path_name + "/config.json", "r") as f:
            config = json.load(f)
        print(config)
        
        config = Config(
            
        )
        
