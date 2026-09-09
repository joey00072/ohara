"""Top-k logit distillation from a cached teacher.

Distillation normally means running the teacher forward on every batch, which
makes the teacher part of the training cost and muddies the question "is
distilling faster than training from scratch?". But our corpus is fixed and
pre-tokenized, so the teacher's output for a given block is *deterministic*.
Compute it once, memory-map it, and every student run afterwards pays nothing.

That also separates the two questions worth measuring:

- **sample efficiency** — at equal tokens, does the distilled student reach a
  lower loss? The cache makes this a clean comparison, since both arms have the
  same step cost.
- **wall-clock efficiency** — does it win once the teacher's own forward pass is
  counted? That is the cache build time, reported separately and amortized over
  however many student runs use it.

**Why the logsumexp is stored.** Keeping only the top-k logits and renormalizing
them to sum to one throws away how much probability mass lives *outside* the
top-k -- and that is real information about how confident the teacher is. Storing
the teacher's logsumexp costs one float per position and makes the top-k
probabilities exact::

    p_i     = exp(logit_i - logsumexp)      # exact, for each of the k
    p_other = 1 - sum(p_i)                  # the residual mass, now known

The loss then matches a (k + 1)-way distribution: the k explicit tokens plus one
"everything else" bucket. Because the probabilities are exact rather than
renormalized, temperature is unnecessary -- a real temperature would need the
full logit vector, which is the thing we are trying not to store.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from ohara.tokenbin import DTYPES, read_token_bin_metadata

INDEX_DTYPE = np.uint32
VALUE_DTYPE = np.float16
LOGSUMEXP_DTYPE = np.float32


def cache_paths(prefix: str | Path) -> dict[str, Path]:
    prefix = Path(prefix)
    return {
        "index": prefix.with_name(prefix.name + ".idx.bin"),
        "value": prefix.with_name(prefix.name + ".val.bin"),
        "logsumexp": prefix.with_name(prefix.name + ".lse.bin"),
        "meta": prefix.with_name(prefix.name + ".json"),
    }


def cache_bytes_per_block(seq_len: int, top_k: int) -> int:
    """Storage one cached block costs, for sizing a run before building it."""
    return seq_len * top_k * (
        np.dtype(INDEX_DTYPE).itemsize + np.dtype(VALUE_DTYPE).itemsize
    ) + seq_len * np.dtype(LOGSUMEXP_DTYPE).itemsize


class TeacherCache:
    """Memory-mapped top-k teacher logits, addressed by block index."""

    def __init__(self, prefix: str | Path) -> None:
        paths = cache_paths(prefix)
        if not paths["meta"].exists():
            raise FileNotFoundError(f"missing teacher cache metadata: {paths['meta']}")
        self.metadata = json.loads(paths["meta"].read_text(encoding="utf-8"))
        self.paths = paths
        self.num_blocks = int(self.metadata["blocks"])
        self.seq_len = int(self.metadata["seq_len"])
        self.top_k = int(self.metadata["top_k"])
        self.teacher = str(self.metadata["teacher"])
        self.vocab_size = int(self.metadata["vocab_size"])
        if self.num_blocks < 1 or self.seq_len < 1 or not 1 <= self.top_k <= self.vocab_size:
            raise ValueError("invalid teacher cache dimensions")
        sizes = {"index": self.num_blocks * self.seq_len * self.top_k * np.dtype(INDEX_DTYPE).itemsize,
                 "value": self.num_blocks * self.seq_len * self.top_k * np.dtype(VALUE_DTYPE).itemsize,
                 "logsumexp": self.num_blocks * self.seq_len * np.dtype(LOGSUMEXP_DTYPE).itemsize}
        for name, expected in sizes.items():
            if paths[name].stat().st_size != expected:
                raise ValueError(f"teacher cache {name} size does not match metadata")
        self._index: np.memmap | None = None
        self._value: np.memmap | None = None
        self._logsumexp: np.memmap | None = None

    def _open(self) -> None:
        # Opened lazily so each dataloader worker maps the files itself rather
        # than inheriting handles across a fork.
        if self._index is None:
            shape = (self.num_blocks, self.seq_len, self.top_k)
            self._index = np.memmap(self.paths["index"], dtype=INDEX_DTYPE, mode="r", shape=shape)
            self._value = np.memmap(self.paths["value"], dtype=VALUE_DTYPE, mode="r", shape=shape)
            self._logsumexp = np.memmap(
                self.paths["logsumexp"],
                dtype=LOGSUMEXP_DTYPE,
                mode="r",
                shape=(self.num_blocks, self.seq_len),
            )

    def block(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._open()
        assert self._value is not None and self._logsumexp is not None
        # np.array(..., copy=True) rather than asarray: a memmap slice is not
        # writable, and torch.from_numpy on a read-only buffer warns and yields a
        # tensor whose in-place use is undefined.
        return (
            torch.from_numpy(np.array(self._index[index], dtype=np.int64)),
            torch.from_numpy(np.array(self._value[index], dtype=np.float32)),
            torch.from_numpy(np.array(self._logsumexp[index], dtype=np.float32)),
        )


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_index: torch.Tensor,
    teacher_logit: torch.Tensor,
    teacher_logsumexp: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy against the teacher over its top-k plus an "other" bucket.

    Returns a summed loss (not a mean), so a caller can divide by the same token
    count it uses for the hard-label loss and keep the two terms commensurate.

    All of the teacher's probabilities are exact, recovered from the stored
    logsumexp, so no renormalisation or temperature is involved. The residual
    mass is clamped at zero: it is mathematically non-negative, but float16
    storage of the top-k logits can push the sum a hair over one.
    """
    student_logits = student_logits.float()
    teacher_probs = torch.exp(teacher_logit - teacher_logsumexp.unsqueeze(-1))
    other_prob = (1.0 - teacher_probs.sum(dim=-1)).clamp_min(0.0)

    # Student's own normaliser over the full vocabulary, then the k explicit
    # log-probabilities and whatever log-mass is left for everything else.
    student_logsumexp = torch.logsumexp(student_logits, dim=-1)
    student_topk = torch.gather(student_logits, -1, teacher_index)
    student_logprobs = student_topk - student_logsumexp.unsqueeze(-1)
    per_token = -(teacher_probs * student_logprobs).sum(dim=-1)
    if teacher_index.size(-1) < student_logits.size(-1):
        excluded = student_logits.scatter(-1, teacher_index, float("-inf"))
        student_other = torch.logsumexp(excluded, dim=-1) - student_logsumexp
        per_token = per_token - other_prob * student_other

    if valid_mask is not None:
        per_token = per_token * valid_mask
    return per_token.sum()


class DistillTokenBinDataset(IterableDataset):
    """Token-bin blocks paired with their cached teacher predictions.

    Yields ``(inputs, targets, teacher_index, teacher_logit, teacher_logsumexp)``.
    Only the blocks the cache covers are iterated, so both arms of an A/B see
    exactly the same data whether or not they use the teacher terms.
    """

    def __init__(
        self,
        bin_path: str | Path,
        cache_prefix: str | Path,
        *,
        max_length: int,
        shuffle: bool = True,
        seed: int = 42,
        infinite: bool = True,
        data_rank: int | None = None,
        data_world_size: int | None = None,
        student_vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        self.bin_path = Path(bin_path)
        self.metadata = read_token_bin_metadata(self.bin_path)
        self.token_dtype = DTYPES[str(self.metadata.get("dtype", "uint16"))]
        self.cache = TeacherCache(cache_prefix)
        if student_vocab_size is not None and student_vocab_size != self.cache.vocab_size:
            raise ValueError(f"student vocabulary {student_vocab_size} must match teacher cache vocabulary {self.cache.vocab_size}")
        expected_tokens = self.cache.num_blocks * (max_length + 1)
        if int(self.cache.metadata["tokens_covered"]) != expected_tokens:
            raise ValueError("teacher cache tokens_covered does not match block geometry")
        if expected_tokens > int(self.metadata["tokens"]):
            raise ValueError("teacher cache covers more tokens than the token bin")
        if self.bin_path.stat().st_size != int(self.metadata["tokens"]) * np.dtype(self.token_dtype).itemsize:
            raise ValueError("token bin size does not match metadata")
        source_hash = self.cache.metadata.get("token_bin_sha256")
        if source_hash is not None:
            with self.bin_path.open("rb") as handle:
                actual_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            if source_hash != actual_hash:
                raise ValueError("teacher cache was built from a different token bin")
        elif Path(self.cache.metadata["token_bin"]).resolve() != self.bin_path.resolve():
            raise ValueError("teacher cache token_bin does not match this corpus; rebuild cache")
        if self.cache.seq_len != max_length:
            raise ValueError(
                f"teacher cache holds seq_len={self.cache.seq_len} but training uses "
                f"{max_length}; rebuild the cache to match"
            )
        self.max_length = max_length
        self.block_size = max_length + 1
        self.num_blocks = self.cache.num_blocks
        self.shuffle = shuffle
        self.seed = seed
        self.infinite = infinite
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        if data_world_size is not None and data_world_size < 1:
            raise ValueError("data_world_size must be positive")
        if data_rank is not None and (data_rank < 0 or (data_world_size is not None and data_rank >= data_world_size)):
            raise ValueError("data_rank must be within data_world_size")
        self._tokens: np.memmap | None = None

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

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        if self._tokens is None:
            self._tokens = np.memmap(self.bin_path, dtype=self.token_dtype, mode="r")
        shard_id, num_shards = self._shard()
        if shard_id >= self.num_blocks:
            return
        epoch = 0
        while True:
            order = np.arange(self.num_blocks)
            if self.shuffle:
                np.random.default_rng(self.seed + epoch).shuffle(order)
            for position in range(shard_id, self.num_blocks, num_shards):
                block_index = int(order[position])
                start = block_index * self.block_size
                block = np.asarray(
                    self._tokens[start : start + self.block_size], dtype=np.int64
                )
                chunk = torch.from_numpy(block)
                index, logit, logsumexp = self.cache.block(block_index)
                yield chunk[:-1], chunk[1:], index, logit, logsumexp
            if not self.infinite:
                return
            epoch += 1


@torch.no_grad()
def build_teacher_cache(
    teacher,
    bin_path: str | Path,
    cache_prefix: str | Path,
    *,
    seq_len: int,
    top_k: int = 8,
    num_blocks: int | None = None,
    batch_blocks: int = 4,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    log_every: int = 200,
) -> dict[str, object]:
    """Run the teacher over a token bin and store its top-k predictions.

    Blocks are read in file order (not shuffled) so a cache is reproducible and
    rebuilding a prefix replaces its previous cache files.
    """
    bin_path = Path(bin_path)
    metadata = read_token_bin_metadata(bin_path)
    token_dtype = DTYPES[str(metadata.get("dtype", "uint16"))]
    tokens = np.memmap(bin_path, dtype=token_dtype, mode="r")

    block_size = seq_len + 1
    available = tokens.size // block_size
    total_blocks = available if num_blocks is None else min(num_blocks, available)
    if total_blocks < 1:
        raise ValueError("token bin holds fewer than one block")

    paths = cache_paths(cache_prefix)
    paths["index"].parent.mkdir(parents=True, exist_ok=True)
    teacher = teacher.to(device=device, dtype=dtype).eval()

    vocab_size = int(teacher.config.vocab_size)
    if not 1 <= top_k <= vocab_size:
        raise ValueError("top_k must be within the teacher vocabulary")
    if batch_blocks < 1:
        raise ValueError("batch_blocks must be positive")
    written = 0
    with (
        open(paths["index"], "wb") as index_file,
        open(paths["value"], "wb") as value_file,
        open(paths["logsumexp"], "wb") as logsumexp_file,
    ):
        for start_block in range(0, total_blocks, batch_blocks):
            count = min(batch_blocks, total_blocks - start_block)
            rows = np.stack(
                [
                    np.asarray(
                        tokens[(start_block + offset) * block_size :
                               (start_block + offset) * block_size + block_size],
                        dtype=np.int64,
                    )
                    for offset in range(count)
                ]
            )
            batch = torch.from_numpy(rows).to(device)
            # Position j of the input predicts token j+1, so the teacher sees the
            # first seq_len tokens and its outputs align with the student's targets.
            logits = teacher(batch[:, :-1]).logits
            # Bound the fp32 temporary independently of batch and context length.
            for chunk in logits.reshape(-1, logits.size(-1)).split(64):
                chunk = chunk.float()
                values, indices = torch.topk(chunk, top_k, dim=-1)
                logsumexp = torch.logsumexp(chunk, dim=-1)
                # Center before fp16 storage; the existing loss format uses a
                # zero normalizer for these log probabilities.
                values = values - logsumexp.unsqueeze(-1)
                indices.cpu().numpy().astype(INDEX_DTYPE).tofile(index_file)
                values.cpu().numpy().astype(VALUE_DTYPE).tofile(value_file)
                np.zeros(len(chunk), dtype=LOGSUMEXP_DTYPE).tofile(logsumexp_file)
            written += count
            if log_every and (start_block // batch_blocks) % log_every == 0:
                print(f"  cached {written:,}/{total_blocks:,} blocks", flush=True)

    with bin_path.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    payload: dict[str, object] = {
        "blocks": written,
        "seq_len": seq_len,
        "top_k": top_k,
        "teacher": str(getattr(teacher, "name_or_path", teacher.__class__.__name__)),
        "vocab_size": vocab_size,
        "token_bin": str(bin_path.resolve()),
        "token_bin_sha256": source_hash,
        "value_format": "log_probability",
        "tokens_covered": written * block_size,
    }
    paths["meta"].write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"teacher cache: {written:,} blocks, {written * block_size:,} tokens")
    return payload


def hard_label_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard next-token cross-entropy, returned as (summed loss, token count)."""
    flat_logits = logits.float().reshape(-1, logits.size(-1))
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index
    loss = F.cross_entropy(
        flat_logits, flat_targets, ignore_index=ignore_index, reduction="sum"
    )
    return loss, valid.sum()
