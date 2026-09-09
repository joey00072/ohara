"""Qwen3 decoder model with native Hugging Face safetensors loading.

This keeps Ohara's small ``Config -> model -> KV cache`` shape while matching
Qwen3's checkpoint layout: explicit head dimensions, Q/K head normalization,
grouped-query attention, half-rotation RoPE, and tied token/output embeddings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from ohara.embeddings_pos.rotary import precompute_freqs_cis
from ohara.modules.kv_cache import KVCache
from ohara.modules.mlp import SwiGLU
from ohara.modules.norm import RMSNorm


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    max_sequence_length: int = 40960
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    attention_dropout: float = 0.0
    attention_bias: bool = False
    weight_tying: bool = True
    initializer_range: float = 0.02
    bos_token_id: int = 151643
    eos_token_id: int = 151645

    @classmethod
    def from_hf_config(cls, payload: dict[str, Any]) -> "Qwen3Config":
        if payload.get("model_type") != "qwen3":
            raise ValueError(f"expected a qwen3 config, got {payload.get('model_type')!r}")
        if payload.get("use_sliding_window") or "sliding_attention" in payload.get(
            "layer_types", []
        ):
            raise ValueError("sliding-window Qwen3 checkpoints are not supported")
        rope_theta = payload.get("rope_theta", 1_000_000.0)
        # HF uses rope_scaling in older configs and rope_parameters in newer
        # ones. Neither may silently change a scaled checkpoint into plain RoPE.
        for key in ("rope_scaling", "rope_parameters"):
            rope = payload.get(key)
            if rope is None:
                continue
            if not isinstance(rope, dict):
                raise ValueError(f"Qwen3 {key} must be a mapping")
            allowed = {"rope_type", "type", "rope_theta"}
            if (
                any(rope.get(name, "default") != "default" for name in ("rope_type", "type"))
                or set(rope) - allowed
            ):
                raise ValueError(
                    f"scaled or per-layer Qwen3 RoPE in {key} is not supported; "
                    "only default RoPE is implemented"
                )
            rope_theta = rope.get("rope_theta", rope_theta)
        return cls(
            vocab_size=int(payload["vocab_size"]),
            max_sequence_length=int(payload["max_position_embeddings"]),
            hidden_size=int(payload["hidden_size"]),
            intermediate_size=int(payload["intermediate_size"]),
            num_hidden_layers=int(payload["num_hidden_layers"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            num_key_value_heads=int(payload["num_key_value_heads"]),
            head_dim=int(
                payload.get("head_dim", payload["hidden_size"] // payload["num_attention_heads"])
            ),
            rms_norm_eps=float(payload.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope_theta),
            attention_dropout=float(payload.get("attention_dropout", 0.0)),
            attention_bias=bool(payload.get("attention_bias", False)),
            weight_tying=bool(payload.get("tie_word_embeddings", True)),
            initializer_range=float(payload.get("initializer_range", 0.02)),
            bos_token_id=int(payload.get("bos_token_id", 151643)),
            eos_token_id=int(payload.get("eos_token_id", 151645)),
        )

    def to_hf_config(self) -> dict[str, Any]:
        return {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_sequence_length,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "attention_dropout": self.attention_dropout,
            "attention_bias": self.attention_bias,
            "tie_word_embeddings": self.weight_tying,
            "initializer_range": self.initializer_range,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "hidden_act": "silu",
            "use_cache": True,
        }


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Qwen repeats each half-width frequency over both halves of a head, then
    # rotates [first_half, second_half]. This differs from Llama's adjacent pairs.
    cos = torch.cat((cos, cos), dim=-1).to(query.dtype).view(1, cos.size(0), 1, -1)
    sin = torch.cat((sin, sin), dim=-1).to(query.dtype).view(1, sin.size(0), 1, -1)
    return query * cos + _rotate_half(query) * sin, key * cos + _rotate_half(key) * sin


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout

        self.query = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.key = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.value = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: tuple[torch.Tensor, torch.Tensor],
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        query = self.query(x).view(batch, seq_len, self.num_heads, self.head_dim)
        key = self.key(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        value = self.value(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = _apply_rope(query, key, *freqs)

        if kv_cache is not None:
            key, value = kv_cache.forward(key, value, start_pos)
        query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        causal = None
        if kv_cache is not None:
            query_positions = start_pos + torch.arange(seq_len, device=x.device)
            key_positions = torch.arange(key.size(-2), device=x.device)
            causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=causal,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=kv_cache is None,
            enable_gqa=self.num_queries_per_kv > 1,
        )
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.proj(output)


class Qwen3Block(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.attn = Qwen3Attention(config)
        self.ff = SwiGLU(
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            dropout=0.0,
            bias=False,
        )
        self.norm1 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs: tuple[torch.Tensor, torch.Tensor],
        kv_cache: KVCache | None = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs, kv_cache, start_pos)
        return x + self.ff(self.norm2(x))


class Qwen3(nn.Module):
    supports_thinking = True

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if config.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for rotary embeddings")
        if config.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Qwen3Block(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.vocab_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.weight_tying:
            self.vocab_proj.weight = self.token_emb.weight

        cos, sin = precompute_freqs_cis(
            config.head_dim,
            config.max_sequence_length,
            theta=config.rope_theta,
        )
        self.register_buffer("freq_cos", cos, persistent=False)
        self.register_buffer("freq_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(
        self,
        token_ids: torch.Tensor,
        kv_cache: list[KVCache] | None = None,
        position_ids: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("input token IDs must have shape (batch, sequence)")
        if isinstance(position_ids, torch.Tensor):
            if position_ids.numel() != 1:
                raise ValueError("position_ids must be a scalar cache position")
            position_ids = int(position_ids.item())
        if kv_cache is None:
            if position_ids is not None:
                raise ValueError("position_ids is only valid when using a KV cache")
            start_pos = 0
        else:
            if len(kv_cache) != len(self.layers):
                raise ValueError("KV cache must contain one entry per model layer")
            if position_ids is None or position_ids < 0:
                raise ValueError("a non-negative position_ids is required with KV cache")
            start_pos = position_ids

        seq_len = token_ids.size(1)
        if seq_len < 1:
            raise ValueError("input sequence cannot be empty")
        if start_pos + seq_len > self.config.max_sequence_length:
            raise ValueError("input exceeds max_sequence_length")

        x = self.token_emb(token_ids)
        freqs = (
            self.freq_cos[start_pos : start_pos + seq_len],
            self.freq_sin[start_pos : start_pos + seq_len],
        )
        for index, layer in enumerate(self.layers):
            cache = kv_cache[index] if kv_cache is not None else None
            x = layer(x, freqs, cache, start_pos)
        return self.vocab_proj(self.norm(x))

    def build_kv_cache(
        self, batch_size: int = 1, *, max_sequence_length: int | None = None, int8: bool = False
    ) -> list[KVCache]:
        max_sequence_length = (
            self.config.max_sequence_length if max_sequence_length is None else max_sequence_length
        )
        if not 1 <= max_sequence_length <= self.config.max_sequence_length:
            raise ValueError("cache length must be in [1, max_sequence_length]")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        shape = (
            batch_size,
            max_sequence_length,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        return [
            KVCache(
                shape,
                max_sequence_length,
                index,
                device=self.token_emb.weight.device,
                dtype=self.token_emb.weight.dtype,
                int8=int8,
            )
            for index in range(self.config.num_hidden_layers)
        ]

    @staticmethod
    def _hf_to_ohara_key(key: str) -> str:
        if key == "model.embed_tokens.weight":
            return "token_emb.weight"
        if key == "model.norm.weight":
            return "norm.weight"
        if key == "lm_head.weight":
            return "vocab_proj.weight"
        key = key.removeprefix("model.")
        replacements = (
            (".self_attn.q_proj.", ".attn.query."),
            (".self_attn.k_proj.", ".attn.key."),
            (".self_attn.v_proj.", ".attn.value."),
            (".self_attn.o_proj.", ".attn.proj."),
            (".self_attn.q_norm.", ".attn.q_norm."),
            (".self_attn.k_norm.", ".attn.k_norm."),
            (".mlp.gate_proj.", ".ff.gate."),
            (".mlp.up_proj.", ".ff.up."),
            (".mlp.down_proj.", ".ff.down."),
            (".input_layernorm.", ".norm1."),
            (".post_attention_layernorm.", ".norm2."),
        )
        for old, new in replacements:
            key = key.replace(old, new)
        return key

    @staticmethod
    def _ohara_to_hf_key(key: str) -> str:
        if key == "token_emb.weight":
            return "model.embed_tokens.weight"
        if key == "norm.weight":
            return "model.norm.weight"
        if key == "vocab_proj.weight":
            return "lm_head.weight"
        replacements = (
            (".attn.query.", ".self_attn.q_proj."),
            (".attn.key.", ".self_attn.k_proj."),
            (".attn.value.", ".self_attn.v_proj."),
            (".attn.proj.", ".self_attn.o_proj."),
            (".attn.q_norm.", ".self_attn.q_norm."),
            (".attn.k_norm.", ".self_attn.k_norm."),
            (".ff.gate.", ".mlp.gate_proj."),
            (".ff.up.", ".mlp.up_proj."),
            (".ff.down.", ".mlp.down_proj."),
            (".norm1.", ".input_layernorm."),
            (".norm2.", ".post_attention_layernorm."),
        )
        for old, new in replacements:
            key = key.replace(old, new)
        return "model." + key

    def save_pretrained(self, save_directory: str | Path) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        tensors = {
            self._ohara_to_hf_key(key): value.detach().cpu().contiguous().clone()
            for key, value in state.items()
            if not (self.config.weight_tying and key == "vocab_proj.weight")
        }
        save_file(tensors, save_directory / "model.safetensors", metadata={"format": "pt"})
        (save_directory / "config.json").write_text(
            json.dumps(self.config.to_hf_config(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> "Qwen3":
        source = Path(model_name_or_path)
        if source.is_file():
            model_directory = source.parent
            explicit_weights = source
        elif source.is_dir():
            model_directory = source
            explicit_weights = None
        else:
            model_directory = Path(
                snapshot_download(
                    str(model_name_or_path),
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    local_files_only=local_files_only,
                    token=token,
                    allow_patterns=("config.json", "*.safetensors", "*.safetensors.index.json"),
                )
            )
            explicit_weights = None

        config_path = model_directory / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"model config not found: {config_path}")
        config = Qwen3Config.from_hf_config(json.loads(config_path.read_text(encoding="utf-8")))
        with torch.device("meta"):
            model = cls(config)
        for module in model.modules():
            for buffer_name, buffer in module._buffers.items():
                if buffer is not None:
                    module._buffers[buffer_name] = torch.zeros(
                        buffer.shape, dtype=buffer.dtype, device=device
                    )
        cos, sin = precompute_freqs_cis(
            config.head_dim,
            config.max_sequence_length,
            theta=config.rope_theta,
        )
        model.freq_cos, model.freq_sin = cos.to(device), sin.to(device)

        if explicit_weights is not None:
            weight_files = [explicit_weights]
        else:
            index_path = model_directory / "model.safetensors.index.json"
            if index_path.is_file():
                index = json.loads(index_path.read_text(encoding="utf-8"))
                weight_map = index.get("weight_map")
                if not isinstance(weight_map, dict) or not weight_map:
                    raise ValueError(f"invalid safetensors index: {index_path}")
                weight_files = [
                    model_directory / filename for filename in dict.fromkeys(weight_map.values())
                ]
            else:
                weights_path = model_directory / "model.safetensors"
                if not weights_path.is_file():
                    raise FileNotFoundError(f"model weights not found: {weights_path}")
                weight_files = [weights_path]

        expected = set(model.state_dict())
        loaded: set[str] = set()
        for weights_path in weight_files:
            shard = {
                cls._hf_to_ohara_key(key): value
                for key, value in load_file(weights_path, device=str(device)).items()
            }
            duplicates = loaded.intersection(shard)
            if duplicates:
                raise ValueError(f"duplicate tensors across shards: {sorted(duplicates)[:5]}")
            loaded.update(shard)
            shard = {
                key: value.to(dtype=dtype)
                if dtype is not None and value.is_floating_point()
                else value
                for key, value in shard.items()
            }
            model.load_state_dict(shard, strict=False, assign=True)

        allowed_missing = {"vocab_proj.weight"} if config.weight_tying else set()
        missing = sorted(expected - loaded - allowed_missing)
        unexpected = sorted(loaded - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing keys: {missing[:5]}")
            if unexpected:
                details.append(f"unexpected keys: {unexpected[:5]}")
            raise RuntimeError(
                "checkpoint does not match Qwen3 config (" + "; ".join(details) + ")"
            )
        if config.weight_tying:
            if "token_emb.weight" in loaded:
                model.vocab_proj.weight = model.token_emb.weight
            else:
                model.token_emb.weight = model.vocab_proj.weight
        return model.eval()
