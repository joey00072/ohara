"""Streaming chat inference: load a finetuned checkpoint and talk to it.

``ohara.inference.Inference`` generates a completion and prints it; a chat UI
needs the opposite arrangement — incremental text handed back to a caller that
decides what to do with it. :class:`ChatEngine` wraps a finetuned model with the
conversation rendering from :mod:`ohara.chat` and yields decoded deltas as they
are sampled.

Decoding is incremental in a way that matters for BPE: a token can decode to a
partial UTF-8 sequence or split a word, so text is emitted by diffing the decode
of the full generated sequence rather than decoding tokens one at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch
from transformers import PreTrainedTokenizerBase

from ohara.chat import (
    ASSISTANT_END,
    load_chat_tokenizer,
    render_for_completion,
    special_token_ids,
)
from ohara.models.llama import Config, Llama


@dataclass
class SamplingConfig:
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 0
    max_new_tokens: int = 512

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be in [0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k cannot be negative")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")


def sample_next_token(
    logits: torch.Tensor,
    config: SamplingConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token from the final position's logits.

    ``temperature == 0`` is greedy. Otherwise top-k is applied before top-p, so
    top-k bounds the candidate set and top-p trims it further by probability mass.
    """
    logits = logits[:, -1].float()
    if config.temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / config.temperature
    if config.top_k > 0:
        k = min(config.top_k, logits.size(-1))
        threshold = logits.topk(k, dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    if 0.0 < config.top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        # Keep the smallest prefix whose mass reaches top_p; the shift keeps the
        # single most likely token even when it alone exceeds the threshold.
        drop = cumulative - sorted_probs > config.top_p
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
        return torch.gather(sorted_indices, -1, choice)
    return torch.multinomial(probs, num_samples=1, generator=generator)


# Checkpoints can carry either or both wrapper prefixes, and in either order:
# DDP adds "module." and torch.compile adds "_orig_mod.", so a compiled model
# wrapped in DDP saves keys like "module._orig_mod.layers.0...". Strip whatever
# is on the front until nothing is left to strip.
_WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def strip_wrapper_prefixes(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in _WRAPPER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def _clean_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {strip_wrapper_prefixes(key): value for key, value in state.items()}


def config_from_state_dict(
    state: Mapping[str, torch.Tensor], *, moe_experts_per_tok: int = 2
) -> Config:
    """Recover a model config from checkpoint tensor shapes.

    Checkpoints written by the training entrypoints hold a bare state dict, so
    the architecture is inferred rather than read back. Everything is determined
    by a shape except two things:

    - the head dimension, which comes from the rotary buffer;
    - ``moe_experts_per_tok``, which is a routing choice and leaves no trace in
      any tensor shape. It defaults to 2 and must be passed if a checkpoint was
      trained with another top-k, otherwise the model loads with the right
      weights but routes differently than it was trained to.
    """
    vocab_size, hidden_size = state["token_emb.weight"].shape
    layer_indices = {
        int(key.split(".")[1]) for key in state if key.startswith("layers.")
    }
    num_layers = 1 + max(layer_indices)

    # An MoE layer stores its feed-forward under ff.experts.<n>.*; a dense one
    # stores ff.up/gate/down directly.
    moe_layers = sorted(
        {
            int(key.split(".")[1])
            for key in state
            if key.startswith("layers.") and ".ff.experts." in key
        }
    )
    if moe_layers:
        num_experts = 1 + max(
            int(key.split(".")[4])
            for key in state
            if key.startswith(f"layers.{moe_layers[0]}.ff.experts.")
        )
        intermediate_size = state[f"layers.{moe_layers[0]}.ff.experts.0.up.weight"].shape[0]
        # Layers are laid out at a fixed stride, so the gap between the first two
        # recovers the interval (a single MoE layer implies every layer).
        interval = moe_layers[1] - moe_layers[0] if len(moe_layers) > 1 else 1
    else:
        num_experts = 0
        intermediate_size = state["layers.0.ff.up.weight"].shape[0]
        interval = 1

    query_out = state["layers.0.attn.query.weight"].shape[0]
    key_out = state["layers.0.attn.key.weight"].shape[0]

    freq_cos = state.get("freq_cos")
    if freq_cos is None:
        raise ValueError("checkpoint has no rotary buffer; cannot infer head dimension")
    head_dim = int(freq_cos.shape[-1]) * 2
    num_heads = query_out // head_dim
    num_kv_heads = key_out // head_dim
    max_sequence_length = int(freq_cos.shape[0]) // 2
    tied = state["vocab_proj.weight"].data_ptr() == state["token_emb.weight"].data_ptr()

    return Config(
        vocab_size=int(vocab_size),
        hidden_size=int(hidden_size),
        intermediate_size=int(intermediate_size),
        max_sequence_length=max_sequence_length,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=0 if num_kv_heads == num_heads else num_kv_heads,
        dropout=0.0,
        weight_tying=tied,
        moe_num_experts=num_experts,
        moe_experts_per_tok=moe_experts_per_tok if num_experts else 2,
        moe_layer_interval=interval,
    )


class ChatEngine:
    """A finetuned model plus its tokenizer, ready to answer conversations."""

    def __init__(
        self,
        model: Llama,
        tokenizer: PreTrainedTokenizerBase,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if dtype is None:
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=self.dtype).eval()
        self.tokenizer = tokenizer
        self.specials = special_token_ids(tokenizer)
        self.assistant_end_id = self.specials[ASSISTANT_END]
        self.eos_token_id = tokenizer.eos_token_id
        # Ids at or above this exist only to pad the matmul to a tensor-core
        # friendly width. They decode to nothing, so they must never be sampled.
        self.tokenizer_vocab_size = len(tokenizer)
        self.max_sequence_length = model.config.max_sequence_length

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        tokenizer_dir: str | Path | None = None,
        tokenizer_name: str = "EleutherAI/gpt-neo-125m",
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "ChatEngine":
        """Load a checkpoint written by ``examples/train_sft.py``.

        The SFT script saves the chat tokenizer beside the checkpoint (``sft.pt``
        -> ``sft/``). That directory is preferred when present, since it is the
        exact vocabulary the model was trained against; otherwise the tokenizer
        is rebuilt from ``tokenizer_name`` and extended with the chat tokens.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        if tokenizer_dir is None:
            candidate = checkpoint_path.with_suffix("")
            tokenizer_dir = candidate if (candidate / "tokenizer.json").exists() else None
        if tokenizer_dir is not None:
            tokenizer = load_chat_tokenizer(
                tokenizer_dir=tokenizer_dir, prefer_hf=False, local_files_only=True
            )
        else:
            tokenizer = load_chat_tokenizer(hf_name=tokenizer_name)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = _clean_state_dict(checkpoint.get("model", checkpoint))
        config = config_from_state_dict(state)
        # A model trained with --pad-vocab-to is wider than its tokenizer; those
        # extra rows are unreachable padding and are masked out before sampling.
        # A model *narrower* than the tokenizer is genuinely broken: it cannot
        # represent tokens the tokenizer can produce.
        if config.vocab_size < len(tokenizer):
            raise ValueError(
                f"checkpoint vocabulary ({config.vocab_size:,}) is smaller than the "
                f"tokenizer ({len(tokenizer):,}). Point --tokenizer-dir at the "
                "directory saved next to the checkpoint."
            )
        model = Llama(config)
        model.load_state_dict(state, strict=True)
        return cls(model, tokenizer, device=device, dtype=dtype)

    def metadata(self, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
        """Model facts worth showing in a UI, merged with any saved SFT metadata."""
        info: dict[str, Any] = {
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "layers": self.model.config.num_hidden_layers,
            "hidden_size": self.model.config.hidden_size,
            "heads": self.model.config.num_attention_heads,
            "context_length": self.max_sequence_length,
            "vocab_size": self.model.config.vocab_size,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
        }
        if checkpoint_path is not None:
            sidecar = Path(checkpoint_path).with_suffix("") / "ohara_chat.json"
            if sidecar.exists():
                try:
                    info["sft"] = json.loads(sidecar.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
        return info

    def render_prompt(self, messages: Sequence[Mapping[str, Any]]) -> list[int]:
        """Render a conversation, dropping old turns that do not fit the context.

        The window has to leave room for the reply, so entire leading turns are
        dropped (never a partial turn) until the prompt fits.
        """
        budget = self.max_sequence_length
        turns = list(messages)
        while turns:
            ids = render_for_completion(self.tokenizer, turns, max_tokens=budget)
            if len(ids) < budget:
                return ids
            # Drop the oldest user/assistant pair and retry.
            turns = turns[2:] if len(turns) > 2 else turns[1:]
        raise ValueError("the latest message alone does not fit in the context window")

    @torch.inference_mode()
    def generate_stream(
        self,
        messages: Sequence[Mapping[str, Any]],
        config: SamplingConfig | None = None,
        *,
        seed: int | None = None,
    ) -> Iterator[str]:
        """Yield decoded text deltas for the assistant's reply."""
        config = config or SamplingConfig()
        prompt_ids = self.render_prompt(messages)
        room = self.max_sequence_length - len(prompt_ids)
        if room < 1:
            raise ValueError("no room left in the context window for a reply")
        max_new_tokens = min(config.max_new_tokens, room)

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        kv_cache = self.model.build_kv_cache(batch_size=1)
        tokens = torch.tensor(prompt_ids, dtype=torch.long, device=self.device).view(1, -1)
        model_input = tokens
        position = 0
        produced: list[int] = []
        emitted = ""

        for _ in range(max_new_tokens):
            logits = self.model(model_input, kv_cache, position)
            position += model_input.size(1)
            if logits.size(-1) > self.tokenizer_vocab_size:
                logits[..., self.tokenizer_vocab_size:] = float("-inf")
            next_token = sample_next_token(logits, config, generator)
            token_id = int(next_token.item())
            if token_id == self.assistant_end_id or token_id == self.eos_token_id:
                break
            produced.append(token_id)
            model_input = next_token

            # Decode the whole reply and emit only what is new: a single token
            # may not be valid text on its own.
            text = self.tokenizer.decode(produced, skip_special_tokens=True)
            if len(text) > len(emitted):
                yield text[len(emitted):]
                emitted = text

    def generate(
        self,
        messages: Sequence[Mapping[str, Any]],
        config: SamplingConfig | None = None,
        *,
        seed: int | None = None,
    ) -> str:
        """Generate a complete reply."""
        return "".join(self.generate_stream(messages, config, seed=seed))
