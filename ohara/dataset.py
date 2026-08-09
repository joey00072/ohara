from __future__ import annotations


from datasets import load_from_disk, load_dataset

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torch.distributed as dist
from pathlib import Path

import os
import requests
import random

from transformers import PreTrainedTokenizerBase

from ohara.tokenizer import load_tokenizer, TokenizerLoadResult

PATH = Path("./data")
# "google/byt5-small"
# "NeelNanda/gpt-neox-tokenizer-digits"


def _resolve_tokenizer(
    tokenizer: PreTrainedTokenizerBase | str | None,
    *,
    cache_dir: str | Path | None = None,
) -> PreTrainedTokenizerBase:
    if tokenizer is None:
        tokenizer = "NeelNanda/gpt-neox-tokenizer-digits"
    if isinstance(tokenizer, str):
        result: TokenizerLoadResult = load_tokenizer(
            hf_name=tokenizer,
            prefer_hf=True,
            cache_dir=cache_dir,
            use_fast=True,
        )
        tokenizer = result.tokenizer
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def get_tokenizer(
    tokenizer: PreTrainedTokenizerBase | str | None = None,
    *,
    cache_dir: str | Path | None = None,
) -> PreTrainedTokenizerBase:
    """Backward-compatible tokenizer loader used by older scripts."""
    return _resolve_tokenizer(tokenizer, cache_dir=cache_dir)


class PreTokenizedDataset(IterableDataset):
    def __init__(
        self,
        dataset_name: str = "JeanKaddour/minipile",
        tokenizer: PreTrainedTokenizerBase | str | None = None,
        split: str = "train",
        path: Path = PATH,
        microbatch_size: int = 32,
        min_length: int = 512,
        max_length: int = 2048,
        hf=False,
        cache_dir=None,
    ):
        self.tokenizer = _resolve_tokenizer(tokenizer, cache_dir=cache_dir)
        self.length = len(self.tokenizer)
        self.PAD = self.tokenizer.pad_token_id
        if self.PAD is None:
            raise ValueError("tokenizer must define a PAD or EOS token")

        self.microbatch_size = microbatch_size
        self.vocab_size = len(self.tokenizer)
        self.min_length = min_length
        self.max_length = max_length + 1
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name

        fpath = path
        if path == PATH:
            fpath = str(
                f"{self.dataset_name.replace('/','-')}--{self.tokenizer.name_or_path.replace('/','-')}"
            )
            fpath = str(path.joinpath(fpath).joinpath(split))
        if hf:
            self.ds = load_dataset(dataset_name)[split]
        else:
            self.ds = load_from_disk(fpath)
        if len(self.ds) == 0:
            raise ValueError(f"dataset split is empty: {fpath}")

    def __iter__(self) -> torch.Tensor:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers

        while True:
            for index in range(shard_id, len(self.ds), num_shards):
                x = torch.tensor(self.ds[index]["input_ids"], dtype=torch.long)
                if x.shape[0] > self.max_length:
                    x = x[: self.max_length]
                x = F.pad(x, (0, self.max_length - x.shape[0]), "constant", value=self.PAD)
                yield x[:-1], x[1:]


class StreamingTextDataset(IterableDataset):
    """Tokenize and greedily pack a streaming Hugging Face text dataset."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerBase | str,
        *,
        split: str,
        max_length: int,
        name: str | None = None,
        text_column: str = "text",
        shuffle: bool = False,
        shuffle_buffer_size: int = 1_000,
        seed: int = 42,
        cache_dir: str | Path | None = None,
        data_rank: int | None = None,
        data_world_size: int | None = None,
        start_block: int = 0,
    ) -> None:
        super().__init__()
        if max_length < 2:
            raise ValueError("max_length must be at least 2")
        if shuffle_buffer_size < 1:
            raise ValueError("shuffle_buffer_size must be at least 1")
        if (data_rank is None) != (data_world_size is None):
            raise ValueError("data_rank and data_world_size must be provided together")
        if data_world_size is not None and data_world_size < 1:
            raise ValueError("data_world_size must be at least 1")
        if data_rank is not None and not 0 <= data_rank < data_world_size:
            raise ValueError("data_rank must be between 0 and data_world_size - 1")
        if start_block < 0:
            raise ValueError("start_block cannot be negative")
        self.dataset_name = dataset_name
        self.name = name
        self.split = split
        self.max_length = max_length
        self.text_column = text_column
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.cache_dir = cache_dir
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.start_block = start_block
        self.tokenizer = _resolve_tokenizer(tokenizer, cache_dir=cache_dir)
        self.boundary_token_id = self.tokenizer.bos_token_id
        if self.boundary_token_id is None:
            self.boundary_token_id = self.tokenizer.eos_token_id
        if self.boundary_token_id is None:
            raise ValueError("tokenizer must define a BOS or EOS token for document boundaries")

    def _load_stream(self):
        local_path = Path(self.dataset_name).expanduser()
        if local_path.exists():
            builders = {
                ".txt": "text",
                ".json": "json",
                ".jsonl": "json",
                ".parquet": "parquet",
            }
            if local_path.is_file():
                suffix = local_path.suffix.lower()
                if suffix not in builders:
                    raise ValueError(f"unsupported local dataset file: {local_path}")
                data_files = {self.split: str(local_path)}
                builder = builders[suffix]
            else:
                matches = []
                builder = None
                for suffix, candidate_builder in builders.items():
                    candidates = sorted(local_path.glob(f"{self.split}*{suffix}"))
                    if candidates:
                        matches = candidates
                        builder = candidate_builder
                        break
                if not matches or builder is None:
                    raise FileNotFoundError(
                        f"no {self.split!r} text, JSONL, JSON, or Parquet files in {local_path}"
                    )
                data_files = {self.split: [str(path) for path in matches]}
            return load_dataset(
                builder,
                data_files=data_files,
                split=self.split,
                streaming=True,
                cache_dir=self.cache_dir,
            )

        kwargs = {
            "path": self.dataset_name,
            "split": self.split,
            "streaming": True,
            "cache_dir": self.cache_dir,
        }
        if self.name is not None:
            kwargs["name"] = self.name
        return load_dataset(**kwargs)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        distributed = dist.is_available() and dist.is_initialized()
        rank = self.data_rank
        world_size = self.data_world_size
        if rank is None or world_size is None:
            rank = dist.get_rank() if distributed else 0
            world_size = dist.get_world_size() if distributed else 1
        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers

        base_stream = self._load_stream()
        token_buffer: list[int] = []
        buffer_start = 0
        blocks_to_skip = self.start_block
        epoch = 0
        while True:
            stream = base_stream
            if self.shuffle:
                stream = stream.shuffle(
                    seed=self.seed + epoch,
                    buffer_size=self.shuffle_buffer_size,
                )
            yielded_document = False
            for document_index, row in enumerate(stream):
                if document_index % num_shards != shard_id:
                    continue
                if self.text_column not in row:
                    raise KeyError(
                        f"dataset row does not contain text column {self.text_column!r}"
                    )
                text = row[self.text_column]
                if not isinstance(text, str) or not text:
                    continue
                yielded_document = True
                token_buffer.append(self.boundary_token_id)
                token_buffer.extend(
                    self.tokenizer.encode(text, add_special_tokens=False)
                )
                block_size = self.max_length + 1
                while len(token_buffer) - buffer_start >= block_size:
                    block_end = buffer_start + block_size
                    block = torch.tensor(token_buffer[buffer_start:block_end], dtype=torch.long)
                    buffer_start = block_end
                    if blocks_to_skip > 0:
                        blocks_to_skip -= 1
                    else:
                        yield block[:-1], block[1:]
                if buffer_start >= block_size * 16:
                    token_buffer = token_buffer[buffer_start:]
                    buffer_start = 0
            if not yielded_document:
                raise RuntimeError(
                    f"no usable documents found in {self.dataset_name!r} split {self.split!r}"
                )
            epoch += 1


class TinyShakespeareDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | str | None = None,
        path: Path = PATH,
        min_length: int = 512,
        max_length: int = 512,
        cache_dir=None,
    ):
        self.tokenizer = _resolve_tokenizer(tokenizer, cache_dir=cache_dir)
        self.length = len(self.tokenizer)
        self.PAD = self.tokenizer.pad_token_id

        self.vocab_size = len(self.tokenizer)
        self.min_length = min_length
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.dataset_name = "tinyshakespeare"

        self.data_path = path.joinpath(self.dataset_name + ".txt")

        try:  # ugly ik
            with open(self.data_path) as f:
                self.data = torch.Tensor(self.tokenizer.encode(f.read())).long()
        except Exception:
            self.download_data()
            with open(self.data_path) as f:
                self.data = torch.Tensor(self.tokenizer.encode(f.read())).long()

        self.length = len(self.data)

    def __iter__(self) -> torch.Tensor:
        while True:
            idx = random.randint(0, (self.length - self.max_length - 1))
            x = self.data[idx : idx + self.max_length + 1]
            yield x[:-1][: self.max_length], x[1:][: self.max_length]

    def download_data(self):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        if response.status_code == 200:
            with open(self.data_path, "w", encoding="utf-8") as file:
                file.write(response.text)
        else:
            raise Exception(f"Failed to download data. Status code: {response.status_code}")


if __name__ == "__main__":
    tokenizer = _resolve_tokenizer("microsoft/phi-2")
    dataset = PreTokenizedDataset(
        dataset_name="roneneldan/TinyStories", tokenizer=tokenizer, cache_dir="hf_cache"
    )
    dataloder = DataLoader(dataset, batch_size=2)

    for data, target in dataloder:
        print(data.shape, target.shape)
        exit(0)
        # print(tokenizer.decode(data.tolist()))
