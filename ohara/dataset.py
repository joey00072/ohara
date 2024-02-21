import os
from transformers import AutoTokenizer
from itertools import cycle
from datasets import load_dataset,load_from_disk
from datasets.download import DownloadMode

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset,DataLoader,IterDataPipe
from pathlib import Path


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
        split:str="train",
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
        if path==PATH:
            fpath = str(f"{self.dataset_name.replace('/','-')}--{self.tokenizer.name_or_path.replace('/','-')}")
            fpath = path.joinpath(fpath).joinpath(split)

        self.ds = load_from_disk(fpath)
        self.toks_cycle = cycle(self.ds)


    def __iter__(self) -> torch.Tensor:
        while True:
            x = next(self.toks_cycle)["input_ids"]
            x = torch.tensor(x, dtype=torch.long)
            x = F.pad(x, (0, self.max_length - x.shape[0]), "constant", value=self.PAD)
            yield x[:-1], x[1:]


if __name__=="__main__":
    tokenizer = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits")
    dataset = MiniPile(tokenizer=tokenizer,split="test")
    dataloader = DataLoader(dataset=dataset,batch_size=8)
    for data,target in dataloader:
        print(data.shape)
        break
