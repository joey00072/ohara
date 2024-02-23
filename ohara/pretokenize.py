from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer
from datasets import load_dataset, DownloadMode


class DatasetPreprocessor:
    def __init__(
        self,
        dataset_name: str = "JeanKaddour/minipile",
        tokenizer_name: str = "NeelNanda/gpt-neox-tokenizer-digits",
        min_length: int = 512,
        max_length: int = 2049,
        splits: list[str] = ["train", "test"],
        output_dir: Path = Path("./data"),
        hf_cache: Path = Path("./hf_cache"),
    ):
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.min_length = min_length
        self.max_length = max_length
        self.splits = splits
        self.output_dir = output_dir
        self.hf_cache = hf_cache

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.padding_side = "right"
        self.PAD = self.tokenizer.pad_token_id
        self.length = self.tokenizer.vocab_size

        self.num_proc = max(os.cpu_count() // 2, 1)

    def load_and_preprocess_dataset(self, split):
        dataset = load_dataset(
            self.dataset_name,
            split=split,
            download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
        )

        tokenized = (
            dataset.map(
                self.tokenize_fn,
                batched=True,
                remove_columns=["text"],
                num_proc=self.num_proc,
            )
            .shuffle(seed=42)
            .filter(self.filter_fn, num_proc=self.num_proc)
        )

        return tokenized

    def tokenize_fn(self, x):
        return self.tokenizer(x["text"], max_length=self.max_length, truncation=True)

    def filter_fn(self, x):
        return len(x["input_ids"]) >= self.min_length

    def save_pre_tokenized_dataset(self, dataset, split):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        fpath = str(
            f"{self.dataset_name.replace('/','-')}--{self.tokenizer.name_or_path.replace('/','-')}"
        )
        dataset.save_to_disk(self.output_dir.joinpath(fpath).joinpath(split))

        print(f"Dataset saved to {self.output_dir}")

    def process_and_save(self):
        for split in self.splits:
            tokenized_dataset = self.load_and_preprocess_dataset(split)
            self.save_pre_tokenized_dataset(tokenized_dataset, split)
            print(f"{split} Processing and saving completed.")


if __name__ == "__main__":
    # Example usage
    preprocessor = DatasetPreprocessor(splits=["test"])
    preprocessor.process_and_save()
