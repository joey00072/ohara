from __future__ import annotations


from itertools import cycle
from datasets import load_from_disk, load_dataset

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
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
        self.toks_cycle = cycle(self.ds)

    def __iter__(self) -> torch.Tensor:
        while True:
            x = next(self.toks_cycle)["input_ids"]
            x = torch.tensor(x, dtype=torch.long)
            if x.shape[0] > self.max_length:
                x = x[: self.max_length]
            x = F.pad(x, (0, self.max_length - x.shape[0]), "constant", value=self.PAD)
            yield x[:-1][: self.max_length], x[1:][: self.max_length]


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
