"""Flat pre-tokenized corpora, memory-mapped at training time.

Streaming text and tokenizing inside the training loop makes the tokenizer part
of the critical path: the GPUs idle while the CPU turns bytes into ids, and the
same documents get re-tokenized every epoch. Measured on a 2xA100 ClimbMix run,
that showed up as a steady ~4.0s step punctuated by a ~15s stall every ~17
steps, dragging the mean to 4.8s.

The fix is to tokenize once, ahead of time, into a flat array of token ids on
disk. Training then memory-maps that array and slices contiguous blocks out of
it, which costs a memcpy and no Python.

Layout is deliberately simple: one ``.bin`` of little-endian uint16 or uint32 ids, and
a ``.json`` sidecar recording the tokenizer, token count and dtype so a corpus
cannot be silently paired with the wrong vocabulary. uint16 holds any vocabulary
up to 65,536, and larger vocabularies use uint32.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info

from ohara.data_parallel import DataParallelIterableDataset
from transformers import PreTrainedTokenizerBase


# uint16 holds any vocabulary up to 65,536 and halves read bandwidth against
# uint32. Larger vocabularies (Qwen's 151,936, for instance) need the wider type,
# so the writer picks per corpus and the sidecar records which was used.
TOKEN_DTYPE = np.uint16
DTYPES = {"uint16": np.dtype("<u2"), "uint32": np.dtype("<u4")}
MAX_VOCAB = np.iinfo(TOKEN_DTYPE).max + 1


def dtype_for_vocab(vocab_size: int) -> type[np.unsignedinteger]:
    """Narrowest unsigned type that can hold every id in this vocabulary."""
    if vocab_size <= np.iinfo(np.uint16).max + 1:
        return np.uint16
    if vocab_size <= np.iinfo(np.uint32).max + 1:
        return np.uint32
    raise ValueError(f"vocabulary of {vocab_size:,} is too large for uint32 token bins")


def _sidecar_path(bin_path: str | Path) -> Path:
    return Path(bin_path).with_suffix(".json")


def write_token_bin(
    documents: Iterable[str],
    tokenizer: PreTrainedTokenizerBase,
    bin_path: str | Path,
    *,
    boundary_token_id: int | None = None,
    batch_size: int = 1024,
    flush_every: int = 64 * 1024 * 1024,
    progress_every: int = 1_000_000,
    log: bool = True,
    build_options: dict[str, object] | None = None,
) -> dict[str, object]:
    """Tokenize ``documents`` into a flat unsigned ``.bin`` plus a JSON sidecar.

    Each document is prefixed with a boundary token (BOS, else EOS) so packed
    No separate EOS is appended: tokenizers with distinct BOS/EOS need a
    different document recipe to learn explicit stopping. Packed
    blocks keep document starts visible to the model, matching how
    :class:`ohara.dataset.StreamingTextDataset` builds its stream.
    """
    token_type = dtype_for_vocab(len(tokenizer))
    token_dtype = DTYPES[token_type.__name__]
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if boundary_token_id is None:
        boundary_token_id = tokenizer.bos_token_id
        if boundary_token_id is None:
            boundary_token_id = tokenizer.eos_token_id
    if boundary_token_id is None:
        raise ValueError("tokenizer must define a BOS or EOS token to delimit documents")

    bin_path = Path(bin_path)
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = bin_path.with_name(f".{bin_path.name}.tmp")

    total_tokens = 0
    total_documents = 0
    pending: list[np.ndarray] = []
    pending_tokens = 0
    next_report = progress_every

    def tokenize(batch: list[str]) -> None:
        """Encode a batch of documents into `pending`, each prefixed with BOS."""
        nonlocal total_documents, pending_tokens
        encoded = tokenizer(batch, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            block = np.empty(len(ids) + 1, dtype=token_dtype)
            block[0] = boundary_token_id
            block[1:] = np.asarray(ids, dtype=token_dtype)
            pending.append(block)
            pending_tokens += block.size
        total_documents += len(batch)

    try:
        with open(temporary, "wb") as handle:

            def flush() -> None:
                nonlocal total_tokens, pending_tokens, next_report
                if not pending:
                    return
                np.concatenate(pending).tofile(handle)
                total_tokens += pending_tokens
                pending.clear()
                pending_tokens = 0
                if log and total_tokens >= next_report:
                    print(f"  {total_documents:,} docs -> {total_tokens:,} tokens", flush=True)
                    next_report = total_tokens + progress_every

            batch: list[str] = []
            for document in documents:
                if not isinstance(document, str) or not document:
                    continue
                batch.append(document)
                if len(batch) >= batch_size:
                    tokenize(batch)
                    batch.clear()
                    if pending_tokens >= flush_every:
                        flush()
            if batch:
                tokenize(batch)
            flush()

        if total_tokens == 0:
            raise RuntimeError("no tokens were written; the document stream was empty")
        metadata: dict[str, object] = {
            "tokens": int(total_tokens),
            "documents": int(total_documents),
            "dtype": token_type.__name__,
            "vocab_size": len(tokenizer),
            "tokenizer": str(getattr(tokenizer, "name_or_path", "unknown")),
            "boundary_token_id": int(boundary_token_id),
            "build_options": build_options,
        }
        sidecar = _sidecar_path(bin_path)
        temporary_sidecar = sidecar.with_name(f".{sidecar.name}.tmp")
        try:
            temporary_sidecar.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            # A pair of files cannot be renamed atomically. Leave a marker if
            # publication is interrupted, including same-size replacements.
            pending_publication = bin_path.with_suffix(bin_path.suffix + ".pending")
            pending_publication.touch()
            temporary_sidecar.replace(sidecar)
            temporary.replace(bin_path)
            pending_publication.unlink()
        finally:
            temporary_sidecar.unlink(missing_ok=True)
    finally:
        temporary.unlink(missing_ok=True)

    if log:
        print(f"wrote {bin_path}: {total_tokens:,} tokens from {total_documents:,} documents")
    return metadata


def read_token_bin_metadata(bin_path: str | Path) -> dict[str, object]:
    if Path(bin_path).with_suffix(Path(bin_path).suffix + ".pending").exists():
        raise ValueError(f"incomplete token-bin publication: {bin_path}; rebuild the bin")
    sidecar = _sidecar_path(bin_path)
    if not sidecar.exists():
        raise FileNotFoundError(f"missing token-bin sidecar: {sidecar}")
    return json.loads(sidecar.read_text(encoding="utf-8"))


class TokenBinDataset(DataParallelIterableDataset, IterableDataset):
    """Yield ``(inputs, targets)`` blocks from a memory-mapped token array.

    Blocks are handed out round-robin across the flattened (rank, worker) grid,
    so every shard sees a disjoint stride of the corpus and the union covers it
    exactly once per epoch. The mapping is deterministic given ``seed`` and
    ``epoch``, which keeps a resumed run reproducible.
    """

    def __init__(
        self,
        bin_path: str | Path,
        *,
        max_length: int,
        tokenizer: PreTrainedTokenizerBase | None = None,
        shuffle: bool = True,
        seed: int = 42,
        infinite: bool = True,
        start_block: int = 0,
        data_rank: int | None = None,
        data_world_size: int | None = None,
    ) -> None:
        super().__init__()
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if start_block < 0:
            raise ValueError("start_block cannot be negative")
        if (data_rank is None) != (data_world_size is None):
            raise ValueError("data_rank and data_world_size must be provided together")

        if data_world_size is not None and data_world_size < 1:
            raise ValueError("data_world_size must be at least 1")
        if data_rank is not None and not 0 <= data_rank < data_world_size:
            raise ValueError("data_rank must be between 0 and data_world_size - 1")

        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"token bin not found: {self.bin_path}")
        self.metadata = read_token_bin_metadata(self.bin_path)
        recorded = str(self.metadata.get("dtype", "uint16"))
        if recorded not in DTYPES:
            raise ValueError(f"{self.bin_path} records an unsupported dtype: {recorded}")
        self.token_dtype = DTYPES[recorded]
        if tokenizer is not None and self.metadata["vocab_size"] != len(tokenizer):
            raise ValueError(
                f"{self.bin_path} was tokenized with a {self.metadata['vocab_size']:,} token "
                f"vocabulary but the tokenizer has {len(tokenizer):,}; re-run pretokenization"
            )

        expected_size = int(self.metadata["tokens"]) * self.token_dtype.itemsize
        if self.bin_path.stat().st_size != expected_size:
            raise ValueError(f"{self.bin_path} size does not match its token-bin metadata")
        self.max_length = max_length
        self.shuffle = shuffle
        self.seed = seed
        self.infinite = infinite
        self.start_block = start_block
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        # One extra token per block supplies the shifted target.
        self.block_size = max_length + 1
        self.num_blocks = int(self.metadata["tokens"]) // self.block_size
        if self.num_blocks < 1:
            raise ValueError(
                f"{self.bin_path} holds {self.metadata['tokens']:,} tokens, "
                f"which is fewer than one block of {self.block_size:,}"
            )
        self._tokens: np.memmap | None = None

    def _memmap(self) -> np.memmap:
        # Opened lazily so the memmap is created inside each worker process
        # rather than inherited across a fork.
        if self._tokens is None:
            self._tokens = np.memmap(self.bin_path, dtype=self.token_dtype, mode="r", shape=(int(self.metadata["tokens"]),))
        return self._tokens

    def _shard(self) -> tuple[int, int]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rank, world_size = self.data_rank, self.data_world_size
        if rank is None or world_size is None:
            distributed = dist.is_available() and dist.is_initialized()
            rank = dist.get_rank() if distributed else 0
            world_size = dist.get_world_size() if distributed else 1
        return rank * num_workers + worker_id, world_size * num_workers

    def validate_capacity(self, num_workers: int = 0) -> None:
        _, ranks = self._shard()
        # Called in the parent process before workers exist.
        shards = ranks * max(1, num_workers)
        if self.num_blocks < shards:
            raise ValueError(f"token-bin has {self.num_blocks} blocks for {shards} rank/worker shards")

    def state_dict(self) -> dict:
        return {"version": 1, "blocks_consumed": getattr(self, "_blocks_consumed", self.start_block)}

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1 or state["blocks_consumed"] < 0:
            raise ValueError("invalid token-bin cursor")
        if state["blocks_consumed"] > 0:
            self._ohara_iterator_started = True
        self.start_block = int(state["blocks_consumed"])
        self._blocks_consumed = self.start_block

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        self._mark_iterator_started()
        tokens = self._memmap()
        shard_id, num_shards = self._shard()
        if self.num_blocks < num_shards:
            raise ValueError(f"token-bin has {self.num_blocks} blocks for {num_shards} rank/worker shards")
        blocks_per_epoch = len(range(shard_id, self.num_blocks, num_shards))
        epoch, skip = divmod(self.start_block, blocks_per_epoch)
        self._blocks_consumed = self.start_block
        if not self.infinite and epoch > 0:
            return
        while True:
            order = np.arange(self.num_blocks)
            if self.shuffle:
                np.random.default_rng(self.seed + epoch).shuffle(order)
            for position in range(shard_id + skip * num_shards, self.num_blocks, num_shards):
                start = int(order[position]) * self.block_size
                block = np.asarray(tokens[start : start + self.block_size], dtype=np.int64)
                chunk = torch.from_numpy(block)
                self._blocks_consumed += 1
                yield chunk[:-1], chunk[1:]
            if not self.infinite:
                return
            epoch += 1
            skip = 0
