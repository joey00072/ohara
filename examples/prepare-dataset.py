import sys
from rich import print
from ohara.pretokenize import DatasetPreprocessor, OpenHermesDatasetPreprocessor

from pathlib import Path


def prepare_openhermes_2_5(
    dataset="teknium/OpenHermes-2.5",
    tokenizer="philschmid/gemma-tokenizer-chatml",
    revision="f7e624a58ce3642ec50483cb6039468ee8c3c464",
):
    print(f"Pretokenizing {dataset=} with {tokenizer=}")
    preprocessor = OpenHermesDatasetPreprocessor(
        dataset_name=dataset,
        tokenizer_name=tokenizer,
        revision=revision,
        splits=["train"],
        num_proc=1,
    )

    preprocessor.process_and_save(remove_columns=None)


def tinystories(
    dataset="roneneldan/TinyStories",
    tokenizer="microsoft/phi-2",
    push:bool=False,
    hf_username:str="joey00072",
):
    print(f"Pretokenizing {dataset=} with {tokenizer=}")
    preprocessor = DatasetPreprocessor(
        dataset_name=dataset,
        tokenizer_name=tokenizer,
        splits=["train", "validation"],
        hf_cache=Path("./hf_cache"),
    )
    preprocessor.process_and_save(push=push,hf_username=hf_username)


def minipile(
    dataset="JeanKaddour/minipile",
    tokenizer="EleutherAI/gpt-neo-125m",
    push:bool=False,
    hf_username:str="joey00072",
):
    print(f"Pretokenizing {dataset=} with {tokenizer=}")
    preprocessor = DatasetPreprocessor(
        dataset_name=dataset,
        tokenizer_name=tokenizer,
        splits=["train", "validation"],
        hf_cache=Path("./hf_cache"),
    )
    preprocessor.process_and_save(push=push,hf_username=hf_username)
    


datasets = {"openhermes": prepare_openhermes_2_5, "tinystories": tinystories, "minipile": minipile}


if __name__ == "__main__":
    minipile(push=True,hf_username="joey00072")
    # tinystories(push=True,hf_username="joey00072")
