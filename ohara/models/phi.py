"""Phi-2, in the shape the microsoft/phi-2 checkpoint expects.

Two details set it apart from llama: attention and the MLP run in parallel off
the same normed input, and RoPE rotates only ``rotary_dim`` of each head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from torch import Tensor

from ohara.embeddings_pos.rotary import RoPE
from ohara.modules.kv_cache import KVCache
from ohara.utils.load import download_hf_model


@dataclass
class PhiConfig:
    vocab_size: int = 51200
    max_sequence_length: int = 2048
    hidden_size: int = 2560
    num_attention_heads: int = 32
    num_hidden_layers: int = 32
    dropout: float = 0.2
    multiple_of: int = 4
    bias: bool = True
    eps: float = 1e-5
    rotary_dim: float = 0.4


def new_gelu(x: Tensor) -> Tensor:
    return (
        0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
    )


class MLP(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = new_gelu(x)
        return self.fc2(x)


class PhiMHA(nn.Module):
    def __init__(self, layer_idx, hidden_size, num_attention_heads, rotary_dim) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

        self.rope = RoPE(int(rotary_dim * self.head_dim), traditional=False)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        kv_cache: KVCache | None = None,
        position_ids: int | None = None,
    ) -> Tensor:
        batch_size, seq_length, _ = x.shape

        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)

        k = k.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        q = q.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_attention_heads, self.head_dim)

        start_pos = 0
        if kv_cache is not None:
            if position_ids is None:
                raise ValueError("position_ids is required when using a KV cache")
            start_pos = int(position_ids)
            k, v = kv_cache.forward(k, v, start_pos)

        # (B, num_heads, seq_len, head_dim). Attention math runs in fp32.
        k = k.transpose(1, 2).to(torch.float32)
        q = q.transpose(1, 2).to(torch.float32)
        v = v.transpose(1, 2).to(torch.float32)

        # Queries start at the current cache position; keys always start at 0.
        q = self.rope(q, offset=start_pos)
        k = self.rope(k)

        scale = math.sqrt(1 / q.shape[-1])
        scores = (q @ k.transpose(-1, -2)) * scale

        if kv_cache is not None:
            # The cache holds every key up to now, so build the causal mask from
            # absolute positions rather than reusing the fixed prompt-sized one.
            query_positions = start_pos + torch.arange(seq_length, device=q.device)
            key_positions = torch.arange(k.size(2), device=q.device)
            allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            scores = scores.masked_fill(~allowed, float("-inf"))
        elif mask is not None:
            scores = scores + mask[:, :, :seq_length, :seq_length]

        scores = torch.softmax(scores, dim=-1).type_as(v)
        output = (scores @ v).type_as(x)

        output = output.transpose(1, 2).reshape(batch_size, seq_length, self.hidden_size)
        return self.dense(output)


class Block(nn.Module):
    def __init__(
        self,
        config: PhiConfig,
        block_idx: int | None = None,
    ) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(config.hidden_size, eps=config.eps)
        self.block_idx = block_idx

        self.mixer = PhiMHA(
            block_idx, config.hidden_size, config.num_attention_heads, config.rotary_dim
        )
        self.mlp = MLP(config.hidden_size, config.multiple_of * config.hidden_size)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        kv_cache: KVCache | None = None,
        position_ids: int | None = None,
    ) -> Tensor:
        # Phi runs attention and the MLP in parallel off the same normed input.
        residual = x
        x = self.ln(x)
        return self.mixer(x, mask, kv_cache, position_ids) + self.mlp(x) + residual


class Phi(nn.Module):
    def __init__(self, config: PhiConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_hidden_layers)])

        self.ln = nn.LayerNorm(config.hidden_size, eps=config.eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

        mask = torch.full(
            (1, 1, config.max_sequence_length, config.max_sequence_length), float("-inf")
        )
        # Derived from the config, so it is rebuilt rather than checkpointed.
        self.register_buffer("mask", torch.triu(mask, diagonal=1), persistent=False)

    def forward(
        self,
        x: Tensor,
        kv_cache: list[KVCache] | None = None,
        position_ids: int | None = None,
    ) -> Tensor:
        """``x`` holds only the tokens that have not been cached yet."""
        if kv_cache is not None and len(kv_cache) != len(self.layers):
            raise ValueError("KV cache must contain one entry per model layer")

        x = self.wte(x).to(self.wte.weight.dtype)
        for idx, layer in enumerate(self.layers):
            cache = kv_cache[idx] if kv_cache is not None else None
            x = layer(x, self.mask, cache, position_ids=position_ids)

        x = self.ln(x)
        return self.lm_head(x)

    def loss(self, logits: Tensor, labels: Tensor):
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def build_kv_cache(self, batch_size: int = 1) -> list[KVCache]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        shape = (
            batch_size,
            self.config.max_sequence_length,
            self.config.num_attention_heads,
            self.config.hidden_size // self.config.num_attention_heads,
        )
        dtype = self.wte.weight.dtype
        device = self.wte.weight.device
        return [
            KVCache(shape, self.config.max_sequence_length, idx, device=device, dtype=dtype)
            for idx in range(self.config.num_hidden_layers)
        ]

    @staticmethod
    def _huggingface_state_dict_map(num_hidden_layers: int) -> dict[str, str]:
        """Map this implementation's parameter names onto microsoft/phi-2's."""
        mapping = {
            "wte.weight": "model.embed_tokens.weight",
            "ln.weight": "model.final_layernorm.weight",
            "ln.bias": "model.final_layernorm.bias",
            "lm_head.weight": "lm_head.weight",
            "lm_head.bias": "lm_head.bias",
        }
        attention = {"q_proj": "q_proj", "k_proj": "k_proj", "v_proj": "v_proj", "dense": "dense"}
        for idx in range(num_hidden_layers):
            for suffix in ("weight", "bias"):
                mapping[f"layers.{idx}.ln.{suffix}"] = (
                    f"model.layers.{idx}.input_layernorm.{suffix}"
                )
                for name in ("fc1", "fc2"):
                    mapping[f"layers.{idx}.mlp.{name}.{suffix}"] = (
                        f"model.layers.{idx}.mlp.{name}.{suffix}"
                    )
                for ours, theirs in attention.items():
                    mapping[f"layers.{idx}.mixer.{ours}.{suffix}"] = (
                        f"model.layers.{idx}.self_attn.{theirs}.{suffix}"
                    )
        return mapping

    @classmethod
    def from_pretrained(cls, name: str) -> "Phi":
        """Download ``name`` from the Hub and load it into a fp16 Phi."""
        config = PhiConfig()
        model = cls(config).half()

        path_name = download_hf_model(name)
        weights: dict[str, Tensor] = {}
        for shard in sorted(Path(path_name).glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as reader:
                for key in reader.keys():
                    weights[key] = reader.get_tensor(key)
        if not weights:
            raise FileNotFoundError(f"no safetensors shards found in {path_name}")

        mapping = cls._huggingface_state_dict_map(config.num_hidden_layers)
        missing = sorted(set(mapping.values()) - set(weights))
        if missing:
            raise KeyError(f"checkpoint is missing expected tensors: {missing[:5]}")

        # load_state_dict validates every name and shape for us.
        state_dict = {ours: weights[theirs] for ours, theirs in mapping.items()}
        model.load_state_dict(state_dict, strict=True)
        return model
