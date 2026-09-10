import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint

from ohara.embeddings_pos.rotary import apply_rope, precompute_freqs_cis
from ohara.modules.kv_cache import KVCache
from ohara.modules.mlp import SwiGLU
from ohara.modules.moe import MoE
from ohara.modules.moe_grouped import GroupedMoE
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
    rms_norm_eps: float = 1e-5
    moe_expert_hidden_dim: int | None = None
    rope_theta: float = 100000
    init_style: str = "standard"
    # Mixture of experts. 0 experts keeps every layer's feed-forward dense.
    # With moe_layer_interval=N, every Nth layer is an MoE and the rest stay dense,
    # which is the usual way to buy capacity without paying routing cost everywhere.
    moe_num_experts: int = 0
    moe_experts_per_tok: int = 2
    moe_layer_interval: int = 1
    moe_gate_fn: str = "softmax"
    moe_quantile_balancing: bool = True
    # Fine-grained MoE: many narrow routed experts plus always-on shared experts,
    # dispatched with grouped matmuls instead of a per-expert Python loop. Required
    # in practice above ~32 experts, where the loop's per-expert GEMMs and host
    # sync dominate. See ohara/modules/moe_grouped.py.
    moe_grouped: bool = False
    moe_num_shared_experts: int = 0
    moe_normalize_weights: bool = True
    # Make the routed sum orthogonal to the shared expert's output before adding
    # them (vector rejection, as in exclusive self-attention). Enforces shared
    # expert isolation instead of hoping for it. See ohara/modules/moe_grouped.py.
    moe_shared_exclusive: bool = False


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

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        cache_mask = None
        if kv_cache is not None:
            query_positions = position_ids + torch.arange(seq_len, device=q.device)
            key_positions = torch.arange(k.size(2), device=q.device)
            cache_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)

        output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=cache_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=kv_cache is None,
            enable_gqa=self.num_key_value_heads != self.num_attention_heads,
        )

        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.head_dim * self.num_attention_heads)
        )
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


def uses_moe(cfg: Config, layer_idx: int) -> bool:
    """Whether this layer's feed-forward is a mixture of experts."""
    if cfg.moe_num_experts < 1:
        return False
    return layer_idx % cfg.moe_layer_interval == 0


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int = 0):
        super().__init__()

        self.attn = Attention(cfg)
        self.is_moe = uses_moe(cfg, layer_idx)
        if self.is_moe and cfg.moe_grouped:
            self.ff = GroupedMoE(
                dim=cfg.hidden_size,
                hidden_dim=cfg.moe_expert_hidden_dim or cfg.intermediate_size,
                num_experts=cfg.moe_num_experts,
                num_experts_per_tok=cfg.moe_experts_per_tok,
                num_shared_experts=cfg.moe_num_shared_experts,
                gate_fn=cfg.moe_gate_fn,
                normalize_weights=cfg.moe_normalize_weights,
                quantile_balancing=cfg.moe_quantile_balancing,
                shared_exclusive=cfg.moe_shared_exclusive,
            )
        elif self.is_moe:
            self.ff = MoE(
                dim=cfg.hidden_size,
                hidden_dim=cfg.moe_expert_hidden_dim or cfg.intermediate_size,
                num_experts=cfg.moe_num_experts,
                num_experts_per_tok=cfg.moe_experts_per_tok,
                gate_fn=cfg.moe_gate_fn,
                quantile_balancing=cfg.moe_quantile_balancing,
            )
        else:
            self.ff = SwiGLU(
                dim=cfg.hidden_size,
                hidden_dim=cfg.intermediate_size,
                dropout=cfg.dropout,
                bias=cfg.bias,
            )

        self.norm1 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.norm2 = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

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
        if cfg.moe_expert_hidden_dim is not None and cfg.moe_expert_hidden_dim < 1:
            raise ValueError("moe_expert_hidden_dim must be positive")
        if cfg.intermediate_size < 1:
            raise ValueError("intermediate_size must be positive")
        if cfg.num_attention_heads < 1 or cfg.hidden_size % cfg.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        if head_dim % 2 != 0:
            raise ValueError("attention head dimension must be even for rotary embeddings")
        if cfg.max_sequence_length < 2:
            raise ValueError("max_sequence_length must be at least 2")
        kv_heads = (
            cfg.num_attention_heads if cfg.num_key_value_heads == 0 else cfg.num_key_value_heads
        )
        if kv_heads < 1 or cfg.num_attention_heads % kv_heads != 0:
            raise ValueError("num_key_value_heads must divide num_attention_heads")
        if not 0.0 <= cfg.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if cfg.init_style not in {"standard", "nanochat"}:
            raise ValueError("init_style must be 'standard' or 'nanochat'")
        if cfg.init_style == "nanochat" and cfg.weight_tying:
            raise ValueError("nanochat initialization requires untied embeddings")
        if cfg.moe_num_experts < 0 or cfg.moe_num_shared_experts < 0:
            raise ValueError("MoE expert counts cannot be negative")
        if cfg.moe_layer_interval < 1:
            raise ValueError("moe_layer_interval must be positive")
        if cfg.moe_gate_fn not in {"softmax", "sigmoid"}:
            raise ValueError("moe_gate_fn must be 'softmax' or 'sigmoid'")
        if cfg.moe_grouped and cfg.moe_num_experts == 0:
            raise ValueError("moe_grouped requires moe_num_experts > 0")
        if cfg.moe_num_shared_experts > 0 and not cfg.moe_grouped:
            raise ValueError("shared experts require moe_grouped")
        if not cfg.moe_normalize_weights and not cfg.moe_grouped:
            raise ValueError("moe_normalize_weights only applies to grouped MoE")

        self.config = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        self.layers = nn.ModuleList([Block(cfg, idx) for idx in range(cfg.num_hidden_layers)])

        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
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

        self.mask = None

        self.apply(self._init_weights)
        if cfg.init_style == "nanochat":
            self._init_nanochat_weights()

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        for name in ("freq_cos", "freq_sin"):
            state_dict[prefix + name] = getattr(self, name)
        state_dict.pop(prefix + "mask", None)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: list[KVCache] | None = None,
        position_ids: int | torch.Tensor | None = None,
        *,
        targets: torch.Tensor | None = None,
        loss_chunk_size: int | None = None,
        ignore_index: int = -1,
        return_loss_details: bool = False,
    ):
        if x.ndim != 2:
            raise ValueError("input token IDs must have shape (batch, sequence)")
        if targets is not None:
            if targets.shape != x.shape:
                raise ValueError("targets must match the input batch and sequence dimensions")
            if loss_chunk_size is None or loss_chunk_size < 1:
                raise ValueError("loss_chunk_size must be positive when targets are provided")
            if return_loss_details and torch.is_grad_enabled():
                raise ValueError("return_loss_details is only available with gradients disabled")
        elif return_loss_details:
            raise ValueError("return_loss_details requires targets")
        if isinstance(position_ids, torch.Tensor):
            if position_ids.numel() != 1:
                raise ValueError("position_ids must be a scalar cache position")
            position_ids = int(position_ids.item())

        start_pos = 0
        mask = self.mask
        if kv_cache is not None:
            if targets is not None:
                raise ValueError("targets cannot be used with a KV cache")
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
        if targets is None:
            return self.vocab_proj(x)
        return self._chunked_loss(
            x,
            targets,
            chunk_size=loss_chunk_size,
            ignore_index=ignore_index,
            return_details=return_loss_details,
        )

    def _chunked_loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        *,
        chunk_size: int,
        ignore_index: int,
        return_details: bool,
    ):
        """Project and score token chunks without retaining full-vocabulary logits."""
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        flat_targets = targets.reshape(-1)
        loss_chunks = []
        prediction_chunks = []
        total_loss = hidden.new_zeros((), dtype=torch.float32)

        for hidden_chunk, target_chunk in zip(
            flat_hidden.split(chunk_size), flat_targets.split(chunk_size), strict=True
        ):
            if return_details:
                logits = self.vocab_proj(hidden_chunk).float()
                token_loss = F.cross_entropy(
                    logits,
                    target_chunk,
                    ignore_index=ignore_index,
                    reduction="none",
                )
                total_loss = total_loss + token_loss.sum()
                loss_chunks.append(token_loss)
                prediction_chunks.append(logits.argmax(dim=-1))
                continue

            def project_and_score(h: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                logits = self.vocab_proj(h)
                return F.cross_entropy(
                    logits.float(),
                    labels,
                    ignore_index=ignore_index,
                    reduction="sum",
                )

            chunk_loss = (
                checkpoint(project_and_score, hidden_chunk, target_chunk, use_reentrant=False)
                if torch.is_grad_enabled()
                else project_and_score(hidden_chunk, target_chunk)
            )
            total_loss = total_loss + chunk_loss

        if return_details:
            return total_loss, torch.cat(loss_chunks), torch.cat(prediction_chunks)
        return total_loss

    def build_kv_cache(
        self, batch_size: int = 1, *, max_sequence_length: int | None = None, int8: bool = False
    ) -> list[KVCache]:
        """Build an empty KV cache suitable for the model's configuration."""
        max_sequence_length = (
            self.config.max_sequence_length if max_sequence_length is None else max_sequence_length
        )
        if not 1 <= max_sequence_length <= self.config.max_sequence_length:
            raise ValueError("cache length must be in [1, max_sequence_length]")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        kv_heads = (
            self.config.num_attention_heads
            if self.config.num_key_value_heads == 0
            else self.config.num_key_value_heads
        )
        shape = (
            batch_size,
            max_sequence_length,
            kv_heads,
            self.config.hidden_size // self.config.num_attention_heads,
        )
        kv_cache = []
        dtype = self.token_emb.weight.dtype
        device = self.token_emb.weight.device

        for idx in range(self.config.num_hidden_layers):
            kv_cache.append(
                KVCache(shape, max_sequence_length, idx, device=device, dtype=dtype, int8=int8)
            )
        return kv_cache

    def num_scaling_params(self) -> dict[str, int]:
        """Return parameter groups used by compute-optimal scaling analysis."""
        token_embeddings = self.token_emb.weight.numel()
        lm_head = (
            0 if self.vocab_proj.weight is self.token_emb.weight else self.vocab_proj.weight.numel()
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

    def active_matmul_parameters(self) -> int:
        """Matrix parameters a single token actually passes through, across all layers.

        For a dense model this is just every matrix in every block. For an MoE it is
        *not*: a token is routed to ``num_experts_per_tok`` of ``num_experts``, so
        only that fraction of the expert weights participates. Counting all of them
        would inflate the FLOPs-per-token estimate by ``num_experts / k`` and report
        an MFU several times higher than the hardware is really achieving.
        """
        total = 0
        for block in self.layers:
            total += sum(p.numel() for p in block.attn.parameters() if p.ndim >= 2)
            if isinstance(block.ff, GroupedMoE):
                ff = block.ff
                per_expert = ff.dim * ff.hidden_dim * 3
                total += per_expert * ff.num_experts_per_tok
                # Shared experts run for every token, so they count in full.
                total += per_expert * ff.num_shared_experts
                total += ff.router.weight.numel()
            elif block.is_moe:
                per_expert = sum(p.numel() for p in block.ff.experts[0].parameters() if p.ndim >= 2)
                total += per_expert * block.ff.num_experts_per_tok
                total += block.ff.gate.weight.numel()
            else:
                total += sum(p.numel() for p in block.ff.parameters() if p.ndim >= 2)
        return total

    def estimate_flops(self, sequence_length: int | None = None) -> float:
        """Estimate forward+backward FLOPs per token using nanochat's convention."""
        sequence_length = sequence_length or self.config.max_sequence_length
        if not 1 <= sequence_length <= self.config.max_sequence_length:
            raise ValueError("sequence_length must be within the configured context window")

        # Embedding lookup is not a matmul. The output projection is a matmul even
        # when its physical weight is tied to the token embedding.
        layer_matrix_parameters = self.active_matmul_parameters()
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
        elif isinstance(module, GroupedMoE):
            # These stacked weights are Parameters rather than Linear children,
            # so initialize them explicitly under the standard Llama scheme.
            torch.nn.init.normal_(module.w_gate, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.w_up, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.w_down, mean=0.0, std=0.02)
            module.router_bias.zero_()

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
            if isinstance(block.ff, GroupedMoE):
                block.ff.reset_parameters()
                block.norm1.reset_parameters()
                block.norm2.reset_parameters()
                continue
            experts = block.ff.experts if block.is_moe else [block.ff]
            for expert in experts:
                torch.nn.init.uniform_(expert.up.weight, -0.4 * bound, 0.4 * bound)
                torch.nn.init.uniform_(expert.gate.weight, -0.4 * bound, 0.4 * bound)
                torch.nn.init.zeros_(expert.down.weight)
            if block.is_moe:
                # The router starts small and unbiased so early routing is near-uniform;
                # quantile balancing takes over from there.
                torch.nn.init.normal_(
                    block.ff.gate.weight, mean=0.0, std=self.config.hidden_size**-0.5
                )
                block.ff.router_bias.zero_()
            block.norm1.reset_parameters()
            block.norm2.reset_parameters()
        self.norm.reset_parameters()

    def save_pretrained(self, save_directory: str | Path) -> None:
        """Save weights and architecture in the standard Hugging Face layout.

        The resulting directory contains ``model.safetensors`` and ``config.json``
        and can be loaded with :meth:`from_pretrained`. Rotary tables are derived
        from the config and deliberately omitted from the weights file.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        tensors = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in self.state_dict().items()
            if key not in {"freq_cos", "freq_sin"}
        }
        save_file(tensors, save_directory / "model.safetensors", metadata={"format": "pt"})

        config = {
            "architectures": ["Llama"],
            "architecture": "ohara.models.llama.Llama",
            "model_type": "ohara_llama",
            **asdict(self.config),
        }
        (save_directory / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
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
    ) -> "Llama":
        """Load an Ohara safetensors model from a directory or the Hub.

        Both a single ``model.safetensors`` and the standard
        ``model.safetensors.index.json`` sharded layout are supported. Config
        metadata fields used by Hugging Face are ignored; all :class:`Config`
        fields, including routing settings that cannot be inferred from tensor
        shapes, are restored from ``config.json``.
        """
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
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        config_fields = {field.name for field in fields(Config)}
        config = Config(**{key: value for key, value in payload.items() if key in config_fields})

        with torch.device("meta"):
            model = cls(config)
        for module in model.modules():
            for buffer_name, buffer in module._buffers.items():
                if buffer is not None:
                    module._buffers[buffer_name] = torch.zeros(
                        buffer.shape, dtype=buffer.dtype, device=device
                    )
        cos, sin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads,
            config.max_sequence_length * 2,
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
            if not weights_path.is_file():
                raise FileNotFoundError(f"safetensors shard not found: {weights_path}")
            shard = load_file(weights_path, device=str(device))
            duplicates = loaded.intersection(shard)
            if duplicates:
                raise ValueError(
                    f"duplicate tensors across safetensors shards: {sorted(duplicates)[:5]}"
                )
            loaded.update(shard)
            shard = {
                key: value.to(dtype=dtype)
                if dtype is not None and value.is_floating_point()
                else value
                for key, value in shard.items()
            }
            model.load_state_dict(shard, strict=False, assign=True)

        unexpected = sorted(loaded - expected)
        allowed_missing = {"freq_cos", "freq_sin"}
        if config.weight_tying and loaded.intersection({"token_emb.weight", "vocab_proj.weight"}):
            # Standard safetensors writers may store only one side of a tied pair.
            allowed_missing.update({"token_emb.weight", "vocab_proj.weight"})
        missing = sorted(expected - loaded - allowed_missing)
        if unexpected or missing:
            details = []
            if missing:
                details.append(f"missing keys: {missing[:5]}")
            if unexpected:
                details.append(f"unexpected keys: {unexpected[:5]}")
            raise RuntimeError(
                "checkpoint does not match Llama config (" + "; ".join(details) + ")"
            )

        if config.weight_tying:
            if "token_emb.weight" in loaded:
                model.vocab_proj.weight = model.token_emb.weight
            else:
                model.token_emb.weight = model.vocab_proj.weight
        return model.eval()
