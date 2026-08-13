import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass

from ohara.embeddings_pos.rotary import apply_rope, precompute_freqs_cis
from ohara.modules.kv_cache import KVCache
from ohara.modules.mlp import SwiGLU
from ohara.modules.norm import RMSNorm


@dataclass
class Config:
    vocab_size: int = 65
    max_sequence_length: int = 64
    hidden_size: int = 128
    intermediate_size: int = 256
    num_attention_heads: int = 4
    num_key_value_heads: int = 0
    num_hidden_layers: int = 4
    dropout: float = 0.2
    multiple_of: int = 4
    bias: bool = False
    weight_tying: bool = False
    rope_theta: float = 100000
    init_style: str = "standard"


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden_size = cfg.hidden_size
        self.num_attention_heads = cfg.num_attention_heads
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.num_key_value_heads = (
            cfg.num_attention_heads if cfg.num_key_value_heads == 0 else cfg.num_key_value_heads
        )
        assert self.num_attention_heads % self.num_key_value_heads == 0
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads

        self.key = nn.Linear(hidden_size, self.head_dim * self.num_key_value_heads, cfg.bias)
        self.query = nn.Linear(hidden_size, self.head_dim * self.num_attention_heads, cfg.bias)
        self.value = nn.Linear(hidden_size, self.head_dim * self.num_key_value_heads, cfg.bias)
        self.proj = nn.Linear(self.head_dim * self.num_attention_heads, hidden_size, cfg.bias)

        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.res_dropout = nn.Dropout(cfg.dropout)

        self.flash_attn = hasattr(torch.nn.functional, "scaled_dot_product_attention")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        freqs_cis,
        kv_cache: KVCache | None = None,
        position_ids: int | None = None,
    ) -> torch.Tensor:
        batch, seq_len, hidden_size = x.shape

        k: torch.Tensor
        q: torch.Tensor
        v: torch.Tensor

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(batch, seq_len, self.num_key_value_heads, self.head_dim)
        q = q.view(batch, seq_len, self.num_attention_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_key_value_heads, self.head_dim)

        freqs_cos, freqs_sin = freqs_cis
        q, k = apply_rope(q, k, (freqs_cos, freqs_sin))

        # Apply KV cache if provided
        if kv_cache is not None:
            assert position_ids is not None
            k, v = kv_cache.forward(k, v, position_ids)

        # Grouped Query Attention
        if self.num_key_value_heads != self.num_attention_heads:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        cache_mask = None
        if kv_cache is not None:
            query_positions = position_ids + torch.arange(seq_len, device=q.device)
            key_positions = torch.arange(k.size(2), device=q.device)
            cache_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=cache_mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=kv_cache is None,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            if cache_mask is not None:
                attn_mtx = attn_mtx.masked_fill(
                    ~cache_mask.view(1, 1, seq_len, k.size(2)),
                    float("-inf"),
                )
            elif mask is not None:
                attn_mtx = attn_mtx + mask[:, :, :seq_len, : k.size(2)]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx = self.attn_dropout(attn_mtx)
            output = torch.matmul(attn_mtx, v)

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.head_dim * self.num_attention_heads)
        )
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.attn = Attention(cfg)
        self.ff = SwiGLU(
            dim=cfg.hidden_size,
            hidden_dim=cfg.intermediate_size,
            dropout=cfg.dropout,
            bias=cfg.bias,
        )

        self.norm1 = RMSNorm(cfg.hidden_size)
        self.norm2 = RMSNorm(cfg.hidden_size)

    def forward(
        self,
        x,
        mask,
        freqs_cis,
        kv_cache: KVCache | None = None,
        position_ids: int | None = None,
    ):
        x = x + self.attn(self.norm1(x), mask, freqs_cis, kv_cache, position_ids)
        x = x + self.ff(self.norm2(x))
        return x


class Llama(nn.Module):
    def __init__(self, cfg: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if cfg.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if cfg.hidden_size < 1 or cfg.num_hidden_layers < 1:
            raise ValueError("hidden_size and num_hidden_layers must be positive")
        if cfg.intermediate_size < 1:
            raise ValueError("intermediate_size must be positive")
        if cfg.num_attention_heads < 1 or cfg.hidden_size % cfg.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        if cfg.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")
        kv_heads = cfg.num_attention_heads if cfg.num_key_value_heads == 0 else cfg.num_key_value_heads
        if kv_heads < 1 or cfg.num_attention_heads % kv_heads != 0:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if not 0.0 <= cfg.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if cfg.init_style not in {"standard", "nanochat"}:
            raise ValueError("init_style must be 'standard' or 'nanochat'")
        if cfg.init_style == "nanochat" and cfg.weight_tying:
            raise ValueError("nanochat initialization requires untied embeddings")

        self.config = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])

        self.norm = RMSNorm(cfg.hidden_size)
        self.vocab_proj = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        if cfg.weight_tying:
            self.token_emb.weight = self.vocab_proj.weight

        cos, isin = precompute_freqs_cis(
            head_dim,
            cfg.max_sequence_length * 2,
            theta=cfg.rope_theta,
        )
        self.register_buffer("freq_cos", cos)
        self.register_buffer("freq_sin", isin)

        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full(
                (1, 1, cfg.max_sequence_length, cfg.max_sequence_length), float("-inf")
            )
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self.apply(self._init_weights)
        if cfg.init_style == "nanochat":
            self._init_nanochat_weights()

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: list[KVCache] | None = None,
        position_ids: int | torch.Tensor | None = None,
    ):
        if x.ndim != 2:
            raise ValueError("input token IDs must have shape (batch, sequence)")
        if isinstance(position_ids, torch.Tensor):
            if position_ids.numel() != 1:
                raise ValueError("position_ids must be a scalar cache position")
            position_ids = int(position_ids.item())

        start_pos = 0
        mask = self.mask
        if kv_cache is not None:
            if len(kv_cache) != len(self.layers):
                raise ValueError("KV cache must contain one entry per model layer")
            if position_ids is None or position_ids < 0:
                raise ValueError("a non-negative position_ids is required with KV cache")
            start_pos = position_ids
            mask = None
        elif position_ids is not None:
            raise ValueError("position_ids is only valid when using a KV cache")

        sequence_length = x.size(1)
        if sequence_length < 1:
            raise ValueError("input sequence cannot be empty")
        if start_pos + sequence_length > self.config.max_sequence_length:
            raise ValueError("input exceeds max_sequence_length")

        x = self.token_emb(x)
        freqs_cis = (
            self.freq_cos[start_pos : start_pos + sequence_length],
            self.freq_sin[start_pos : start_pos + sequence_length],
        )

        # Forward through layers with KV cache
        for idx, layer in enumerate(self.layers):
            cache = kv_cache[idx] if kv_cache is not None else None
            x = layer(x, mask, freqs_cis, cache, start_pos if cache is not None else None)

        x = self.norm(x)
        x = self.vocab_proj(x)
        return x

    def build_kv_cache(self, batch_size: int = 1) -> list[KVCache]:
        """Build an empty KV cache suitable for the model's configuration."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        kv_heads = (
            self.config.num_attention_heads
            if self.config.num_key_value_heads == 0
            else self.config.num_key_value_heads
        )
        shape = (
            batch_size,
            self.config.max_sequence_length,
            kv_heads,
            self.config.hidden_size // self.config.num_attention_heads,
        )
        kv_cache = []
        dtype = self.token_emb.weight.dtype
        device = self.token_emb.weight.device

        for idx in range(self.config.num_hidden_layers):
            kv_cache.append(
                KVCache(shape, self.config.max_sequence_length, idx, device=device, dtype=dtype)
            )
        return kv_cache

    def num_scaling_params(self) -> dict[str, int]:
        """Return parameter groups used by compute-optimal scaling analysis."""
        token_embeddings = self.token_emb.weight.numel()
        lm_head = (
            0
            if self.vocab_proj.weight is self.token_emb.weight
            else self.vocab_proj.weight.numel()
        )
        transformer_matrices = sum(
            parameter.numel()
            for layer in self.layers
            for parameter in layer.parameters()
            if parameter.ndim >= 2
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        norms_and_scalars = total - token_embeddings - lm_head - transformer_matrices
        if norms_and_scalars < 0:
            raise RuntimeError("parameter groups overlap")
        return {
            "token_embeddings": token_embeddings,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "norms_and_scalars": norms_and_scalars,
            "total": total,
            # Match nanochat's cleanest convention: transformer matrices + output head.
            "effective": transformer_matrices + lm_head,
        }

    def estimate_flops(self, sequence_length: int | None = None) -> float:
        """Estimate forward+backward FLOPs per token using nanochat's convention."""
        sequence_length = sequence_length or self.config.max_sequence_length
        if not 1 <= sequence_length <= self.config.max_sequence_length:
            raise ValueError("sequence_length must be within the configured context window")

        # Embedding lookup is not a matmul. The output projection is a matmul even
        # when its physical weight is tied to the token embedding.
        layer_matrix_parameters = sum(
            parameter.numel()
            for layer in self.layers
            for parameter in layer.parameters()
            if parameter.ndim >= 2
        )
        matmul_parameters = layer_matrix_parameters + self.vocab_proj.weight.numel()
        attention_flops = (
            12
            * self.config.num_hidden_layers
            * self.config.num_attention_heads
            * (self.config.hidden_size // self.config.num_attention_heads)
            * sequence_length
        )
        return float(6 * matmul_parameters + attention_flops)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def _init_nanochat_weights(self) -> None:
        """Apply nanochat's width-transferable initialization to this Llama."""
        torch.nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.vocab_proj.weight, mean=0.0, std=0.001)
        bound = math.sqrt(3.0) * self.config.hidden_size**-0.5
        for block in self.layers:
            torch.nn.init.uniform_(block.attn.query.weight, -bound, bound)
            torch.nn.init.uniform_(block.attn.key.weight, -bound, bound)
            torch.nn.init.uniform_(block.attn.value.weight, -bound, bound)
            torch.nn.init.zeros_(block.attn.proj.weight)
            torch.nn.init.uniform_(block.ff.up.weight, -0.4 * bound, 0.4 * bound)
            torch.nn.init.uniform_(block.ff.gate.weight, -0.4 * bound, 0.4 * bound)
            torch.nn.init.zeros_(block.ff.down.weight)
            block.norm1.reset_parameters()
            block.norm2.reset_parameters()
        self.norm.reset_parameters()

    @classmethod
    def from_pretrained(cls, hf_name: str):
        raise NotImplementedError(
            "Llama.from_pretrained is not implemented yet. "
            "Use a model-specific loader or initialize `Llama(Config(...))` directly."
        )
