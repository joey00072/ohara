"""Validated rank-local input checkpoint state for zero-worker loaders."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _loader(loader):
    while hasattr(loader, "dataloader"):
        loader = loader.dataloader
    return loader


def _file_identity(path):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return str(path)
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    # Hash once per dataset instance, streaming in bounded memory. Size/mtime
    # alone cannot establish that an in-place edit preserves the corpus.
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path) if path.is_dir() else file.name).encode())
        with file.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def _contract(loader, *, gradient_accumulation_steps, data_rank, data_world_size):
    dataset = loader.dataset
    if not hasattr(dataset, "_resume_identity"):
        source = getattr(dataset, "bin_path", getattr(dataset, "dataset_name", None))
        tokenizer = getattr(dataset, "tokenizer", None)
        backend = getattr(tokenizer, "backend_tokenizer", None)
        token_config = {
            "backend": backend.to_str() if backend is not None else None,
            "name": getattr(tokenizer, "name_or_path", None),
            "vocab": tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else None,
            "special_tokens": getattr(tokenizer, "special_tokens_map", None),
        }
        dataset._resume_identity = {
            "source": ([_file_identity(path) for path in dataset._source_files]
                       if hasattr(dataset, "_source_files") else
                       _file_identity(source) if source is not None else None),
            "metadata": getattr(dataset, "metadata", None),
            "tokenizer": hashlib.sha256(json.dumps(token_config, sort_keys=True, default=str).encode()).hexdigest(),
        }
    return {
        "dataset_class": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "identity": dataset._resume_identity,
        "dataset_options": {key: getattr(dataset, key, None) for key in (
            "name", "revision", "split", "boundary_token_id", "max_length", "text_column", "shuffle", "shuffle_buffer_size", "seed", "infinite",
        )},
        "training_recipe": getattr(dataset, "training_recipe", None),
        "batch_size": loader.batch_size,
        "num_workers": loader.num_workers,
        "drop_last": loader.drop_last,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "data_rank": data_rank,
        "data_world_size": data_world_size,
    }


def capture_input_state(loader, *, gradient_accumulation_steps, data_rank, data_world_size):
    loader = _loader(loader)
    if loader.num_workers:
        return {"version": 1, "resumable": False, "reason": "exact input resume requires num_workers=0; worker prefetch state is unavailable"}
    if not hasattr(loader.dataset, "state_dict"):
        return {"version": 1, "resumable": False, "reason": "dataset has no resumable iterator state"}
    dataset = loader.dataset
    if getattr(dataset, "infinite", True) is False:
        return {"version": 1, "resumable": False, "reason": "finite iterable loader cycling is not exactly resumable; use infinite=True"}
    if hasattr(dataset, "dataset_name") and not Path(dataset.dataset_name).expanduser().exists():
        if not re.fullmatch(r"[0-9a-fA-F]{40}", getattr(dataset, "revision", None) or ""):
            return {"version": 1, "resumable": False, "reason": "remote streaming exact resume requires an immutable 40-character --dataset-revision commit"}
    try:
        cursor = loader.dataset.state_dict()
    except ValueError as exc:
        return {"version": 1, "resumable": False, "reason": str(exc)}
    return {
        "version": 1, "resumable": True,
        "contract": _contract(loader, gradient_accumulation_steps=gradient_accumulation_steps,
                              data_rank=data_rank, data_world_size=data_world_size),
        "cursor": cursor,
    }


def restore_input_state(loader, state, *, gradient_accumulation_steps, data_rank, data_world_size):
    loader = _loader(loader)
    if not state or state.get("version") != 1:
        raise ValueError("checkpoint lacks a validated input contract; legacy exact resume is unsupported")
    if not state.get("resumable"):
        raise ValueError(state.get("reason", "checkpoint input is not exactly resumable"))
    if hasattr(loader.dataset, "_load_stream"):
        # Resolve the same local split files before validating their identity.
        loader.dataset._load_stream()
    current = _contract(loader, gradient_accumulation_steps=gradient_accumulation_steps,
                        data_rank=data_rank, data_world_size=data_world_size)
    mismatch = [key for key in current if current[key] != state["contract"].get(key)]
    if mismatch:
        raise ValueError("checkpoint input contract mismatch: " + ", ".join(mismatch))
    loader.dataset.load_state_dict(state["cursor"])
