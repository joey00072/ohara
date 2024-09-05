from __future__ import annotations

import os
import math
import json
from dataclasses import dataclass


import torch
import torch.nn as nn

from torch import Tensor
from safetensors import safe_open
from ohara.utils.load import download_hf_model

from tqdm import tqdm


@dataclass
class PhiConfig:
    vocab_size: int = 51200
    seq_len: int = 2048
    d_model: int = 2560
    num_heads: int = 32
    num_layers: int = 32
    dropout: float = 0.2
    multiple_of: int = 4
    bias: bool = True
    eps: float = 1e-5
    rotary_dim: float = 0.4


class KVCache:
    def __init__(self, shape, max_seq_length, idx: int | None = None, device=None, dtype=None):
        self.idx = idx
        self.key: torch.Tensor = torch.zeros(shape, device=device, dtype=dtype)
        self.value: torch.Tensor = torch.zeros(shape, device=device, dtype=dtype)
        self.max_seq_length = max_seq_length

    def forward(self, keys: Tensor, values: Tensor, start_pos: Tensor) -> tuple[Tensor, Tensor]:
        bsz, T, _, _ = keys.shape
        # print(f"start_pos={start_pos}, T={T}, bsz={bsz}")
        self.key[:bsz, start_pos : start_pos + T] = keys
        self.value[:bsz, start_pos : start_pos + T] = values
        keys = self.key[:bsz, : start_pos + T]
        values = self.value[:bsz, : start_pos + T]
        return keys, values


class RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        traditional: bool = False,
        base: float = 10000,
        scale: float = 1.0,
    ):
        super().__init__()

        self.dims = dims
        self.traditional = traditional
        self.base = base
        self.scale = scale

    def _extra_repr(self):
        return f"{self.dims}, traditional={self.traditional}"

    def _compute_rope(self, costheta, sintheta, x) -> Tensor:
        x1 = x[..., : self.dims // 2]
        x2 = x[..., self.dims // 2 : self.dims]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        if self.dims < x.shape[-1]:
            rx = torch.concatenate([rx1, rx2, x[..., self.dims :]], axis=-1)
        else:
            rx = torch.concatenate([rx1, rx2], axis=-1)

        return rx

    def _compute_traditional_rope(self, costheta, sintheta, x)->Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rx1 = x1 * costheta - x2 * sintheta
        rx2 = x1 * sintheta + x2 * costheta

        assert self.dims < x.shape[-1], "RoPE doesn't implement partial traditional application"

        rx = torch.cat([rx1[..., None], rx2[..., None]], axis=-1)

        return rx

    def forward(self, x: Tensor, offset: int = 0):
        shape = x.shape
        x = x.reshape(-1, shape[-2], shape[-1])
        N = x.shape[1] + offset
        costheta, sintheta = RoPE.create_cos_sin_theta(
            N,
            self.dims,
            offset=offset,
            base=self.base,
            scale=self.scale,
            dtype=x.dtype,
            device=x.device,
        )

        rope = self._compute_traditional_rope if self.traditional else self._compute_rope
        rx = rope(costheta, sintheta, x)

        return torch.reshape(rx, shape)

    @staticmethod
    def create_cos_sin_theta(
        N: int,
        D: int,
        offset: int = 0,
        base: float = 10000,
        scale: float = 1.0,
        dtype=torch.float32,
        device=None,
    ) -> tuple[Tensor,Tensor]:
        D = D // 2
        positions = torch.arange(offset, N, dtype=dtype, device=device) * scale
        freqs = torch.exp(-torch.arange(0.0, D, dtype=dtype, device=device) * (math.log(base) / D))
        theta = torch.reshape(positions, (-1, 1)) * torch.reshape(freqs, (1, -1))
        return torch.cos(theta), torch.sin(theta)


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


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x)


class PhiMHA(nn.Module):
    def __init__(self, layer_idx, dim, num_heads, rotary_dim) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.layer_idx = layer_idx

        self.k_proj = nn.Linear(dim, dim)
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.dense = nn.Linear(dim, dim)

        self.rope = RoPE(int(rotary_dim * self.head_dim), traditional=False)

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
        kv_cache: KVCache | None = None,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        batch_size, seq_length, d_model = x.shape
        # print(f"{batch_size}, {seq_length}, {d_model}")
        k = self.k_proj(x)
        q = self.q_proj(x)
        v = self.v_proj(x)

        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)

        if kv_cache is not None:
            k, v = kv_cache.forward(k, v, position_ids)

        k: Tensor = k.transpose(1, 2).to(torch.float32)  # shape = (B, num_heads, seq_len, head_dim)
        q: Tensor = q.transpose(1, 2).to(torch.float32)
        v: Tensor = v.transpose(1, 2).to(torch.float32)

        offset = position_ids if kv_cache else 0

        q = self.rope.forward(q, offset=offset).to(torch.float32)
        k = self.rope.forward(k).to(torch.float32)

        # Finally perform the attention computation
        scale = math.sqrt(1 / q.shape[-1])
        scores = (q @ k.transpose(-1, -2)) * scale

        if mask is not None:
            mask = mask[:, :, :seq_length, :seq_length]
            scores = scores + mask

        scores = torch.softmax(scores, dim=-1).type_as(v)
        output = (scores @ v).type_as(x)
        # output = F.scaled_dot_product_attention(
        #     q,
        #     k,
        #     v,
        #     attn_mask=None,
        #     dropout_p=0,
        #     is_causal=True,
        # ).type_as(x)

        output = output.transpose(1, 2).type_as(x)
        output = output.reshape(batch_size, seq_length, self.dim)

        return self.dense(output)

    def scaled_dot_product_attention(self, q, k, v, mask):
        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(-1, -2)) * scale
        if mask is not None:
            scores = scores + mask
        scores = torch.softmax(scores, dim=-1).type_as(v)
        output = scores @ v
        return output


class Block(nn.Module):
    def __init__(
        self,
        config: PhiConfig,
        block_idx: int | None = None,
    ) -> None:
        super().__init__()

        self.ln = LayerNorm(config.d_model, eps=config.eps)
        self.block_idx = block_idx

        self.mixer = PhiMHA(block_idx, config.d_model, config.num_heads, config.rotary_dim)
        self.mlp = MLP(config.d_model, config.multiple_of * config.d_model)

    def forward(
        self,
        x: torch.FloatTensor,
        mask: torch.BoolTensor | None = None,
        kv_cache: list[KVCache] | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        residual = x
        x = self.ln(x)
        return self.mixer(x, mask, kv_cache, position_ids) + self.mlp(x) + residual


class Phi(nn.Module):
    def __init__(self, config: PhiConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_layers)])

        self.ln = LayerNorm(config.d_model, eps=config.eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

        mask = torch.full((1, 1, config.seq_len, config.seq_len), float("-inf"))

        (
            1,
            config.seq_len,
            config.num_heads,
            config.d_model // config.num_heads,
        )

        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x, kv_cache: list[KVCache] | None = None, position_ids=None):
        mask = self.mask
        if kv_cache is not None:
            x = x[:, position_ids:]
            mask = None
        # print(f"{x.shape=} {kv_cache=}")
        x = self.wte(x)
        # print(f"{x.sum(dim=-1)=}")
        cache = None
        for idx, layer in enumerate(self.layers):
            if kv_cache is not None:
                cache = kv_cache[idx]
            x = layer(x.to(self.wte.weight.dtype), mask, cache, position_ids=position_ids)

        x = self.ln(x)
        x = self.lm_head(x)
        return x

    def loss(self, logits: Tensor, labels: Tensor):
        loss = self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def build_kv_cache(self) -> list[KVCache]:
        shape = (
            1,
            self.config.seq_len,
            self.config.num_heads,
            self.config.d_model // self.config.num_heads,
        )
        kv_cache = []
        dtype = self.wte.weight.dtype
        device = self.wte.weight.device

        for idx in range(self.config.num_layers):
            kv_cache.append(KVCache(shape, self.config.seq_len, idx, device=device, dtype=dtype))
        return kv_cache

    @staticmethod
    def from_pretrained(name: str) -> nn.Module:
        config = PhiConfig()
        with torch.device("cuda"):
            model = Phi(config).half()
        # return model

        path_name = download_hf_model(name)
        with open(os.path.join(path_name, "config.json"), encoding="utf-8") as f:
            json.load(f)

        files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        weights = {}
        for f in files:
            w = safe_open(os.path.join(path_name, f), framework="pt", device="cpu")
            for k in w.keys():
                # print(k)
                weights[k] = w.get_tensor(k)

        assert model.wte.weight.data.shape == weights["model.embed_tokens.weight"].shape
        model.wte.weight.data = weights["model.embed_tokens.weight"]

        assert model.lm_head.weight.data.shape == weights["lm_head.weight"].shape
        model.lm_head.weight.data = weights["lm_head.weight"]

        assert model.lm_head.bias.data.shape == weights["lm_head.bias"].shape
        model.lm_head.bias.data = weights["lm_head.bias"]

        assert model.ln.weight.data.shape == weights["model.final_layernorm.weight"].shape
        model.ln.weight.data = weights["model.final_layernorm.weight"]

        assert model.ln.bias.data.shape == weights["model.final_layernorm.bias"].shape
        model.ln.bias.data = weights["model.final_layernorm.bias"]

        for idx, layer in tqdm(enumerate(model.layers), total=len(model.layers)):
            layer: Block

            assert (
                layer.mlp.fc1.weight.data.shape
                == weights[f"model.layers.{idx}.mlp.fc1.weight"].shape
            )
            layer.mlp.fc1.weight.data = weights[f"model.layers.{idx}.mlp.fc1.weight"]

            assert (
                layer.mlp.fc1.bias.data.shape == weights[f"model.layers.{idx}.mlp.fc1.bias"].shape
            )
            layer.mlp.fc1.bias.data = weights[f"model.layers.{idx}.mlp.fc1.bias"]

            assert (
                layer.mlp.fc2.weight.data.shape
                == weights[f"model.layers.{idx}.mlp.fc2.weight"].shape
            )
            layer.mlp.fc2.weight.data = weights[f"model.layers.{idx}.mlp.fc2.weight"]

            assert (
                layer.mlp.fc2.bias.data.shape == weights[f"model.layers.{idx}.mlp.fc2.bias"].shape
            )
            layer.mlp.fc2.bias.data = weights[f"model.layers.{idx}.mlp.fc2.bias"]

            assert (
                layer.mixer.q_proj.weight.data.shape
                == weights[f"model.layers.{idx}.self_attn.q_proj.weight"].shape
            )
            layer.mixer.q_proj.weight.data = weights[f"model.layers.{idx}.self_attn.q_proj.weight"]

            assert (
                layer.mixer.q_proj.bias.data.shape
                == weights[f"model.layers.{idx}.self_attn.q_proj.bias"].shape
            )
            layer.mixer.q_proj.bias.data = weights[f"model.layers.{idx}.self_attn.q_proj.bias"]

            assert (
                layer.mixer.k_proj.weight.data.shape
                == weights[f"model.layers.{idx}.self_attn.k_proj.weight"].shape
            )
            layer.mixer.k_proj.weight.data = weights[f"model.layers.{idx}.self_attn.k_proj.weight"]

            assert (
                layer.mixer.k_proj.bias.data.shape
                == weights[f"model.layers.{idx}.self_attn.k_proj.bias"].shape
            )
            layer.mixer.k_proj.bias.data = weights[f"model.layers.{idx}.self_attn.k_proj.bias"]

            assert (
                layer.mixer.v_proj.weight.data.shape
                == weights[f"model.layers.{idx}.self_attn.v_proj.weight"].shape
            )
            layer.mixer.v_proj.weight.data = weights[f"model.layers.{idx}.self_attn.v_proj.weight"]

            assert (
                layer.mixer.v_proj.bias.data.shape
                == weights[f"model.layers.{idx}.self_attn.v_proj.bias"].shape
            )
            layer.mixer.v_proj.bias.data = weights[f"model.layers.{idx}.self_attn.v_proj.bias"]

            assert (
                layer.mixer.dense.weight.data.shape
                == weights[f"model.layers.{idx}.self_attn.dense.weight"].shape
            )
            layer.mixer.dense.weight.data = weights[f"model.layers.{idx}.self_attn.dense.weight"]

            assert (
                layer.mixer.dense.bias.data.shape
                == weights[f"model.layers.{idx}.self_attn.dense.bias"].shape
            )
            layer.mixer.dense.bias.data = weights[f"model.layers.{idx}.self_attn.dense.bias"]

            assert (
                layer.ln.weight.data.shape
                == weights[f"model.layers.{idx}.input_layernorm.weight"].shape
            )
            layer.ln.weight.data = weights[f"model.layers.{idx}.input_layernorm.weight"]

            assert (
                layer.ln.bias.data.shape
                == weights[f"model.layers.{idx}.input_layernorm.bias"].shape
            )
            layer.ln.bias.data = weights[f"model.layers.{idx}.input_layernorm.bias"]

        return model
