from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


LOCAL_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


@dataclass
class TokenizerLoadResult:
    tokenizer: PreTrainedTokenizerBase
    source: str
    identifier: str


def _has_local_tokenizer(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.joinpath(fname).exists() for fname in LOCAL_TOKENIZER_FILES)


def _load_hf_tokenizer(
    identifier: str,
    *,
    use_fast: bool = True,
    local_files_only: bool = False,
    cache_dir: str | Path | None = None,
    **kwargs: Any,
) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(
        identifier,
        use_fast=use_fast,
        local_files_only=local_files_only,
        cache_dir=cache_dir,
        **kwargs,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.padding_side is None:
        tokenizer.padding_side = "right"
    return tokenizer


def load_tokenizer(
    *,
    hf_name: str | None = None,
    tokenizer_dir: str | Path = "./tokenizer",
    prefer_hf: bool = True,
    fallback_hf_name: str | None = None,
    use_fast: bool = True,
    cache_dir: str | Path | None = None,
    **kwargs: Any,
) -> TokenizerLoadResult:
    """
    Load tokenizer with explicit priority order.

    Priority order:
    - prefer_hf=True:  HF -> local -> fallback HF
    - prefer_hf=False: local -> HF -> fallback HF
    """
    local_dir = Path(tokenizer_dir)
    has_local = _has_local_tokenizer(local_dir)

    attempts: list[tuple[str, str, bool]] = []
    if prefer_hf:
        if hf_name:
            attempts.append(("hf", hf_name, False))
        if has_local:
            attempts.append(("local", str(local_dir), True))
    else:
        if has_local:
            attempts.append(("local", str(local_dir), True))
        if hf_name:
            attempts.append(("hf", hf_name, False))

    if fallback_hf_name and fallback_hf_name not in {hf_name, None}:
        attempts.append(("fallback_hf", fallback_hf_name, False))

    errors: list[str] = []
    for source, identifier, local_only in attempts:
        try:
            tokenizer = _load_hf_tokenizer(
                identifier,
                use_fast=use_fast,
                local_files_only=local_only,
                cache_dir=cache_dir,
                **kwargs,
            )
            return TokenizerLoadResult(tokenizer=tokenizer, source=source, identifier=identifier)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{source}({identifier}): {type(exc).__name__}: {exc}")

    if not attempts:
        raise RuntimeError(
            "No tokenizer source configured. Pass `hf_name` or provide local tokenizer files in "
            f"`{local_dir}`."
        )
    raise RuntimeError("Failed to load tokenizer. " + " | ".join(errors))


def get_tokenizer(
    *,
    hf_name: str | None = None,
    tokenizer_dir: str | Path = "./tokenizer",
    prefer_hf: bool = True,
    fallback_hf_name: str | None = None,
    use_fast: bool = True,
    cache_dir: str | Path | None = None,
    **kwargs: Any,
) -> PreTrainedTokenizerBase:
    result = load_tokenizer(
        hf_name=hf_name,
        tokenizer_dir=tokenizer_dir,
        prefer_hf=prefer_hf,
        fallback_hf_name=fallback_hf_name,
        use_fast=use_fast,
        cache_dir=cache_dir,
        **kwargs,
    )
    return result.tokenizer


def get_token_bytes(
    tokenizer: PreTrainedTokenizerBase,
    *,
    device: str | torch.device = "cpu",
    cache_path: str | Path | None = None,
    include_special: bool = False,
) -> torch.Tensor:
    """
    Build a token_id -> utf8 byte-length tensor.
    Special tokens are zeroed by default for BPB-style evaluation.
    """
    if cache_path is not None:
        cache_file = Path(cache_path)
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return torch.load(f, map_location=device)

    vocab_size = len(tokenizer)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    token_bytes = torch.zeros(vocab_size, dtype=torch.int32, device="cpu")

    for token_id in range(vocab_size):
        if not include_special and token_id in special_ids:
            continue
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        token_bytes[token_id] = len(text.encode("utf-8"))

    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            torch.save(token_bytes, f)

    return token_bytes.to(device=device)


def tokenizer_info(tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
    return {
        "name_or_path": getattr(tokenizer, "name_or_path", "unknown"),
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "special_tokens": list(getattr(tokenizer, "all_special_tokens", []) or []),
    }
