from __future__ import annotations

import os
from pathlib import Path

from transformers import AutoTokenizer
from datasets import load_dataset, DownloadMode


class DatasetPreprocessor:
    def __init__(
        self,
        dataset_name: str = "JeanKaddour/minipile",
        tokenizer_name: str = "NeelNanda/gpt-neox-tokenizer-digits",
        min_length: int = 512,
        max_length: int = 2049,
        splits: list[str] | None = None,
        revision: str | None = None,
        num_proc: int | None = None,
        output_dir: Path = Path("./data"),
        hf_cache: Path = Path("./hf_cache"),
    ):
        if splits is None:
            splits = ["train", "test"]
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.min_length = min_length
        self.max_length = max_length
        self.splits = splits
        self.output_dir = output_dir
        self.hf_cache = hf_cache
        self.revision = revision

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.padding_side = "right"
        self.PAD = self.tokenizer.pad_token_id
        self.length = self.tokenizer.vocab_size

        self.num_proc = num_proc if num_proc else max(os.cpu_count() - 3, 1)

    def load_and_preprocess_dataset(self, split, remove_columns=None):
        if remove_columns is None:
            remove_columns = ["text"]
        dataset = load_dataset(
            self.dataset_name,
            split=split,
            download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
            revision=self.revision,
            cache_dir=self.hf_cache,
        )

        tokenized = (
            dataset.map(
                self.tokenize_fn,
                batched=True,
                remove_columns=remove_columns,
                num_proc=self.num_proc,
            )
            .shuffle(seed=31415)
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
        fpath = str(self.output_dir.joinpath(fpath).joinpath(split))
        dataset.save_to_disk(fpath)

        print(f"Dataset saved to {self.output_dir}")
    
    def push_pre_tokenized_dataset(self, dataset,split,hf_username):
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
        fpath = str(
            f"{hf_username}/pretokenized__{self.dataset_name.replace('/','_')}__{self.tokenizer.name_or_path.replace('/','__')}"
        )
        # fpath = str(self.output_dir.joinpath(fpath).joinpath(split))
        dataset.push_to_hub(fpath,split=split)

        print(f"Dataset Pused to hf {fpath}")


    def process_and_save(self, remove_columns=None,push=False,hf_username=None):
        if remove_columns is None:
            remove_columns = ["text"]
        for split in self.splits:
            tokenized_dataset = self.load_and_preprocess_dataset(
                split, remove_columns=remove_columns
            )
            self.save_pre_tokenized_dataset(tokenized_dataset, split)
            print(f"{split} Processing and saving completed.")
            if push:
                assert hf_username is not None
                self.push_pre_tokenized_dataset(tokenized_dataset,split,hf_username)
                
                


class OpenHermesDatasetPreprocessor(DatasetPreprocessor):
    def load_and_preprocess_dataset(self, split, remove_columns=None):
        if remove_columns is None:
            remove_columns = ["text"]
        dataset = load_dataset(
            self.dataset_name,
            split=split,
            download_mode=DownloadMode.REUSE_CACHE_IF_EXISTS,
            revision=self.revision,
        )

        def map_hermes_to_chatml_format(item):
            lst = []
            for row in item["conversations"]:
                if row["from"] == "system":
                    lst.append({"role": "system", "content": row["value"]})
                if row["from"] == "gpt":
                    lst.append({"role": "assistant", "content": row["value"]})
                if row["from"] == "human":
                    lst.append({"role": "user", "content": row["value"]})
            return {"input_ids": self.tokenizer.apply_chat_template(lst)}

        tokenized = (
            dataset.map(
                map_hermes_to_chatml_format,
                num_proc=self.num_proc,
            )
            .shuffle(seed=42)
            .filter(self.filter_fn, num_proc=self.num_proc)
        )

        return tokenized

    def filter_fn(self, x):
        return len(x["input_ids"]) >= self.min_length and len(x["input_ids"]) <= self.max_length


if __name__ == "__main__":
    # Example usage
    # preprocessor = DatasetPreprocessor(splits=["test"])
    # preprocessor.process_and_save()

    preprocessor = OpenHermesDatasetPreprocessor(
        dataset_name="teknium/OpenHermes-2.5",
        tokenizer_name="philschmid/gemma-tokenizer-chatml",
        revision="f7e624a58ce3642ec50483cb6039468ee8c3c464",
        splits=["train"],
        num_proc=1,
    )

    preprocessor.process_and_save(remove_columns=None)
