from __future__ import annotations


from transformers import AutoTokenizer
from itertools import cycle
from datasets import load_from_disk

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path

import requests
import random

PATH = Path("./data")
# "google/byt5-small"
# "NeelNanda/gpt-neox-tokenizer-digits"


def get_tokenizer(self, tokenizer: AutoTokenizer | str = None):
    self.tokenizer = tokenizer
    if tokenizer is None:
        tokenizer = "NeelNanda/gpt-neox-tokenizer-digits"
    if isinstance(tokenizer, str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer, cache_dir=self.cache_dir, use_fast=True
        )


class MiniPile(IterableDataset):
    def __init__(
        self,
        dataset_name: str = "JeanKaddour/minipile",
        tokenizer: AutoTokenizer = None,
        split: str = "train",
        path: Path = PATH,
        microbatch_size: int = 32,
        min_length: int = 512,
        max_length: int = 2048,
        cache_dir=None,
    ):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.length = len(self.tokenizer)
        self.PAD = tokenizer.pad_token_id

        self.microbatch_size = microbatch_size
        self.vocab_size = len(tokenizer)
        self.min_length = min_length
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.dataset_name = dataset_name

        fpath = path
        if path == PATH:
            fpath = str(
                f"{self.dataset_name.replace('/','-')}--{self.tokenizer.name_or_path.replace('/','-')}"
            )
            fpath = path.joinpath(fpath).joinpath(split)

        self.ds = load_from_disk(fpath)
        self.toks_cycle = cycle(self.ds)

    def __iter__(self) -> torch.Tensor:
        while True:
            x = next(self.toks_cycle)["input_ids"]
            x = torch.tensor(x, dtype=torch.long)
            x = F.pad(x, (0, self.max_length - x.shape[0]), "constant", value=self.PAD)
            yield x[:-1], x[1:]


class TinyShakespeareDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer = None,
        path: Path = PATH,
        min_length: int = 512,
        max_length: int = 512,
        cache_dir=None,
    ):
        assert tokenizer is not None, "must pass tokenizer"
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.length = len(self.tokenizer)
        self.PAD = tokenizer.pad_token_id

        self.vocab_size = len(tokenizer)
        self.min_length = min_length
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.dataset_name = "tinyshakespeare"

        self.data_path = path.joinpath(self.dataset_name + ".txt")

        try:  # ugly ik
            with open(self.data_path) as f:
                self.data = torch.Tensor(tokenizer.encode(f.read())).long()
        except Exception:
            self.download_data()
            with open(self.data_path) as f:
                self.data = torch.Tensor(tokenizer.encode(f.read())).long()

        self.length = len(self.data)

    def __iter__(self) -> torch.Tensor:
        while True:
            idx = random.randint(0, (self.length - self.max_length - 1))
            x = self.data[idx : idx + self.max_length + 1]
            yield x[:-1], x[1:]

    def download_data(self):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        response = requests.get(url)
        if response.status_code == 200:
            with open(self.data_path, "w", encoding="utf-8") as file:
                file.write(response.text)
        else:
            raise Exception(
                f"Failed to download data. Status code: {response.status_code}"
            )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    dataset = TinyShakespeareDataset(tokenizer=tokenizer)
    dataloder = DataLoader(dataset, batch_size=2)

    for data, target in dataloder:
        print(data.shape, target.shape)
        # print(tokenizer.decode(data.tolist()))
