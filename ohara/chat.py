"""Conversation rendering for chat finetuning, in nanochat's token layout.

Pretraining sees raw text; a chat model additionally needs to know where a turn
starts and stops, and which tokens it is supposed to *produce* rather than merely
read. Both come from the same place: a small set of special tokens that wrap each
message, plus a per-token supervision mask that is 1 only inside assistant turns.

The token layout mirrors ``ref/nanochat``::

    <bos> <|user_start|> ... <|user_end|> <|assistant_start|> ... <|assistant_end|> ...

with two differences that follow from ohara using Hugging Face tokenizers rather
than a tokenizer trained in-repo:

- there is no ``<|bos|>`` of our own; the wrapped tokenizer's BOS (or EOS, if it
  has no BOS) delimits documents, exactly as it does during pretraining.
- the conversation tokens are *added* to an existing vocabulary, so
  ``len(tokenizer)`` grows. Build the model from the chat tokenizer in both
  pretraining and SFT and the sizes line up; see :func:`load_chat_tokenizer`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerBase

from ohara.tokenizer import get_tokenizer


# Wrap user and assistant turns. The python/output pairs let an assistant turn
# carry a tool call and the interpreter's reply to it; the reply is not
# supervised, because at test time those tokens come from Python, not the model.
USER_START = "<|user_start|>"
USER_END = "<|user_end|>"
ASSISTANT_START = "<|assistant_start|>"
ASSISTANT_END = "<|assistant_end|>"
PYTHON_START = "<|python_start|>"
PYTHON_END = "<|python_end|>"
OUTPUT_START = "<|output_start|>"
OUTPUT_END = "<|output_end|>"

CHAT_SPECIAL_TOKENS: tuple[str, ...] = (
    USER_START,
    USER_END,
    ASSISTANT_START,
    ASSISTANT_END,
    PYTHON_START,
    PYTHON_END,
    OUTPUT_START,
    OUTPUT_END,
)

# Targets carrying this value are dropped from the loss. Matches Trainer's default.
IGNORE_INDEX = -1


def add_chat_tokens(tokenizer: PreTrainedTokenizerBase) -> int:
    """Add the conversation special tokens to ``tokenizer`` in place.

    Returns the number of tokens actually added, which is 0 when the tokenizer
    already carries them. Adding is idempotent, so callers may apply this to a
    tokenizer loaded from a finetuned checkpoint without growing the vocabulary
    a second time.
    """
    missing = [token for token in CHAT_SPECIAL_TOKENS if token not in tokenizer.get_vocab()]
    if not missing:
        return 0
    added = tokenizer.add_special_tokens({"additional_special_tokens": missing})
    return int(added)


def load_chat_tokenizer(
    *,
    hf_name: str | None = None,
    tokenizer_dir: str | Path = "./tokenizer",
    prefer_hf: bool = True,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
    **kwargs: Any,
) -> PreTrainedTokenizerBase:
    """Load a tokenizer and extend it with the chat special tokens."""
    tokenizer = get_tokenizer(
        hf_name=hf_name,
        tokenizer_dir=tokenizer_dir,
        prefer_hf=prefer_hf,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        **kwargs,
    )
    add_chat_tokens(tokenizer)
    return tokenizer


def boundary_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """The document-delimiting token: BOS if the tokenizer has one, else EOS."""
    token_id = tokenizer.bos_token_id
    if token_id is None:
        token_id = tokenizer.eos_token_id
    if token_id is None:
        raise ValueError("tokenizer must define a BOS or EOS token to delimit conversations")
    return int(token_id)


def special_token_ids(tokenizer: PreTrainedTokenizerBase) -> dict[str, int]:
    """Map each chat special token to its id, failing loudly if any is missing."""
    vocab = tokenizer.get_vocab()
    missing = [token for token in CHAT_SPECIAL_TOKENS if token not in vocab]
    if missing:
        raise ValueError(
            "tokenizer is missing chat special tokens: "
            + ", ".join(missing)
            + ". Load it with ohara.chat.load_chat_tokenizer or call add_chat_tokens."
        )
    return {token: int(vocab[token]) for token in CHAT_SPECIAL_TOKENS}


def normalize_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return messages as a strictly alternating user/assistant list.

    A leading system message is folded into the first user message, which is how
    nanochat handles it: the model never learns a separate system role, so the
    instruction is simply prepended to the turn it applies to. Any other
    deviation from user/assistant alternation raises, since silently training on
    a mis-rendered conversation is worse than dropping it.
    """
    if not messages:
        raise ValueError("conversation has no messages")

    normalized = [dict(message) for message in messages]
    if normalized[0].get("role") == "system":
        if len(normalized) < 2 or normalized[1].get("role") != "user":
            raise ValueError("a system message must be followed by a user message")
        system_content = normalized[0]["content"]
        user_content = normalized[1]["content"]
        if not isinstance(system_content, str) or not isinstance(user_content, str):
            raise ValueError("system and user messages must be strings")
        normalized[1]["content"] = f"{system_content}\n\n{user_content}"
        normalized = normalized[1:]

    for index, message in enumerate(normalized):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.get("role") != expected:
            raise ValueError(
                f"message {index} has role {message.get('role')!r}, expected {expected!r}; "
                "conversations must alternate user/assistant"
            )
    return normalized


def render_conversation(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int = 2048,
) -> tuple[list[int], list[int]]:
    """Tokenize a conversation into ids plus an assistant-only supervision mask.

    ``mask[i] == 1`` marks a token the assistant is expected to generate. The
    turn-closing ``<|assistant_end|>`` is supervised too, so the model learns to
    stop; everything the model only ever reads (user turns, tool output, the
    structural tokens that precede a reply) is masked out.

    Both lists are truncated to ``max_tokens``, which bounds memory for the rare
    very long conversation rather than dropping it.
    """
    if max_tokens < 1:
        raise ValueError("max_tokens must be at least 1")
    normalized = normalize_messages(messages)
    specials = special_token_ids(tokenizer)

    ids: list[int] = []
    mask: list[int] = []

    def add(token_ids: int | Iterable[int], mask_value: int) -> None:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        else:
            token_ids = list(token_ids)
        ids.extend(token_ids)
        mask.extend([mask_value] * len(token_ids))

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    add(boundary_token_id(tokenizer), 0)
    for message in normalized:
        content = message["content"]
        if message["role"] == "user":
            if not isinstance(content, str):
                raise ValueError("user messages must be strings")
            add(specials[USER_START], 0)
            add(encode(content), 0)
            add(specials[USER_END], 0)
            continue

        add(specials[ASSISTANT_START], 0)
        if isinstance(content, str):
            add(encode(content), 1)
        elif isinstance(content, list):
            for part in content:
                part_type = part.get("type")
                text = part.get("text", "")
                if part_type == "text":
                    add(encode(text), 1)
                elif part_type == "python":
                    add(specials[PYTHON_START], 1)
                    add(encode(text), 1)
                    add(specials[PYTHON_END], 1)
                elif part_type == "python_output":
                    # Produced by the interpreter at test time, so never supervised.
                    add(specials[OUTPUT_START], 0)
                    add(encode(text), 0)
                    add(specials[OUTPUT_END], 0)
                else:
                    raise ValueError(f"unknown assistant content part type: {part_type!r}")
        else:
            raise ValueError(f"unknown assistant content type: {type(content)}")
        add(specials[ASSISTANT_END], 1)

    return ids[:max_tokens], mask[:max_tokens]


def render_for_completion(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[Mapping[str, Any]],
    *,
    max_tokens: int = 2048,
) -> list[int]:
    """Render a prompt primed for the assistant to continue.

    Drops a trailing assistant message if present (evaluation harnesses hold the
    reference answer in it) and appends ``<|assistant_start|>`` so the next
    sampled token begins the reply.
    """
    normalized = [dict(message) for message in messages]
    if normalized and normalized[-1].get("role") == "assistant":
        normalized.pop()
    if not normalized:
        raise ValueError("conversation has no prompt messages")
    ids, _ = render_conversation(tokenizer, normalized, max_tokens=max_tokens)
    ids.append(special_token_ids(tokenizer)[ASSISTANT_START])
    return ids


def training_pair(
    ids: Sequence[int],
    mask: Sequence[int],
    *,
    ignore_index: int = IGNORE_INDEX,
) -> tuple[list[int], list[int]]:
    """Shift a rendered conversation into (inputs, targets) for next-token loss.

    Position ``i`` predicts token ``i + 1``, so a target is supervised when the
    *predicted* token is an assistant token. Unsupervised positions become
    ``ignore_index`` and drop out of the loss.
    """
    if len(ids) != len(mask):
        raise ValueError("ids and mask must have the same length")
    if len(ids) < 2:
        raise ValueError("a conversation must render to at least two tokens")
    inputs = list(ids[:-1])
    targets = [
        token if mask_value == 1 else ignore_index
        for token, mask_value in zip(ids[1:], mask[1:])
    ]
    return inputs, targets


def visualize(
    tokenizer: PreTrainedTokenizerBase,
    ids: Sequence[int],
    mask: Sequence[int],
) -> str:
    """Render ids with supervised tokens in green and masked ones in red."""
    green, red, reset = "\033[92m", "\033[91m", "\033[0m"
    pieces = []
    for token_id, mask_value in zip(ids, mask):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        pieces.append(f"{green if mask_value == 1 else red}{text}{reset}")
    return "|".join(pieces)


@torch.no_grad()
def resize_token_embeddings(model: nn.Module, vocab_size: int) -> nn.Module:
    """Grow a model's embedding and output head to ``vocab_size``.

    Needed only when finetuning a checkpoint that was pretrained *without* the
    chat tokens. New rows are drawn from the same distribution the corresponding
    matrix was initialized with, so the added tokens start out as unremarkable
    as any other unseen token rather than as high-logit attractors.
    """
    old_vocab_size = model.token_emb.weight.size(0)
    if vocab_size == old_vocab_size:
        return model
    if vocab_size < old_vocab_size:
        raise ValueError("refusing to shrink the vocabulary")

    hidden_size = model.token_emb.weight.size(1)
    device = model.token_emb.weight.device
    dtype = model.token_emb.weight.dtype
    tied = model.vocab_proj.weight is model.token_emb.weight

    embedding = nn.Embedding(vocab_size, hidden_size, device=device, dtype=dtype)
    embedding.weight.normal_(mean=0.0, std=0.02)
    embedding.weight[:old_vocab_size] = model.token_emb.weight
    model.token_emb = embedding

    if tied:
        model.vocab_proj = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
        model.vocab_proj.weight = model.token_emb.weight
    else:
        head = nn.Linear(hidden_size, vocab_size, bias=False, device=device, dtype=dtype)
        # nanochat-style unembeddings start near zero; keep new rows consistent.
        head.weight.normal_(mean=0.0, std=0.001)
        head.weight[:old_vocab_size] = model.vocab_proj.weight
        model.vocab_proj = head

    model.config.vocab_size = vocab_size
    return model
