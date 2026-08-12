"""Download a dataset and pretokenize it to disk for faster training.

    uv run python examples/prepare_dataset.py tinystories
    uv run python examples/prepare_dataset.py fineweb-edu --push --hf-username you

The result lands in ./data and is what :class:`ohara.dataset.PreTokenizedDataset`
reads. Depending on the dataset this takes a while.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from ohara.pretokenize import DatasetPreprocessor, OpenHermesDatasetPreprocessor

HF_CACHE = Path("./hf_cache")


@dataclass(frozen=True)
class Recipe:
    """One named dataset/tokenizer pairing to pretokenize."""

    dataset: str
    tokenizer: str
    splits: list[str] = field(default_factory=lambda: ["train", "validation"])
    name: str | None = None
    revision: str | None = None
    preprocessor: type[DatasetPreprocessor] = DatasetPreprocessor


RECIPES: dict[str, Recipe] = {
    "tinystories": Recipe(
        dataset="roneneldan/TinyStories",
        tokenizer="microsoft/phi-2",
    ),
    "minipile": Recipe(
        dataset="JeanKaddour/minipile",
        tokenizer="EleutherAI/gpt-neo-125m",
    ),
    "fineweb-edu": Recipe(
        dataset="HuggingFaceFW/fineweb-edu",
        tokenizer="EleutherAI/gpt-neo-125m",
        name="sample-10BT",
    ),
    "openhermes": Recipe(
        dataset="teknium/OpenHermes-2.5",
        tokenizer="philschmid/gemma-tokenizer-chatml",
        splits=["train"],
        revision="f7e624a58ce3642ec50483cb6039468ee8c3c464",
        preprocessor=OpenHermesDatasetPreprocessor,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recipe", choices=sorted(RECIPES), help="which dataset to prepare")
    parser.add_argument("--tokenizer", default=None, help="override the recipe's tokenizer")
    parser.add_argument("--push", action="store_true", help="also push the result to the Hub")
    parser.add_argument("--hf-username", default=None, help="Hub account to push to")
    parser.add_argument("--num-proc", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.push and not args.hf_username:
        raise SystemExit("--push requires --hf-username")

    recipe = RECIPES[args.recipe]
    tokenizer = args.tokenizer or recipe.tokenizer
    print(f"Pretokenizing dataset={recipe.dataset!r} with tokenizer={tokenizer!r}")

    kwargs = {
        "dataset_name": recipe.dataset,
        "tokenizer_name": tokenizer,
        "splits": recipe.splits,
        "hf_cache": HF_CACHE,
        "num_proc": args.num_proc,
    }
    if recipe.name is not None:
        kwargs["name"] = recipe.name
    if recipe.revision is not None:
        kwargs["revision"] = recipe.revision

    preprocessor = recipe.preprocessor(**kwargs)
    preprocessor.process_and_save(push=args.push, hf_username=args.hf_username)


if __name__ == "__main__":
    main()
