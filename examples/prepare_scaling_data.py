from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from datasets import load_dataset

from ohara.tokenizer import get_tokenizer


NANOCHAT_CLIMBMIX_BASE_URL = (
    "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
)
NANOCHAT_CLIMBMIX_VALIDATION_SHARD = 6_542
DEFAULT_TOKENIZER = "EleutherAI/gpt-neo-125m"


def nanochat_climbmix_files(train_shards: int) -> tuple[list[str], str]:
    """Return NanoChat's training shards and its pinned validation shard."""
    if not 1 <= train_shards <= NANOCHAT_CLIMBMIX_VALIDATION_SHARD:
        raise ValueError(
            "climbmix train shards must be between 1 and "
            f"{NANOCHAT_CLIMBMIX_VALIDATION_SHARD:,}"
        )
    train = [
        f"{NANOCHAT_CLIMBMIX_BASE_URL}/shard_{index:05d}.parquet"
        for index in range(train_shards)
    ]
    validation = (
        f"{NANOCHAT_CLIMBMIX_BASE_URL}/"
        f"shard_{NANOCHAT_CLIMBMIX_VALIDATION_SHARD:05d}.parquet"
    )
    return train, validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage a bounded text corpus for offline scaling sweeps. With no source override, "
            "this uses the same ClimbMix parquet source and held-out shard as NanoChat."
        )
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="custom Hugging Face dataset; overrides the default NanoChat ClimbMix source",
    )
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument(
        "--train-file",
        action="append",
        default=None,
        help="custom local/remote source file; repeat for multiple files",
    )
    parser.add_argument(
        "--validation-file",
        action="append",
        default=None,
        help="custom local/remote validation file; repeat for multiple files",
    )
    parser.add_argument(
        "--climbmix-train-shards",
        type=int,
        default=10,
        help="number of NanoChat ClimbMix train shards exposed to the bounded staging pass",
    )
    parser.add_argument("--train-documents", type=int, default=100_000)
    parser.add_argument("--validation-documents", type=int, default=10_000)
    parser.add_argument("--output-dir", default="./data/scaling_corpus")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_TOKENIZER,
        help=(
            "tokenize while staging and record a token count next to each split "
            "(stats.<split>.json) so scaling_laws.py can refuse sweeps that would "
            "repeatedly re-epoch a corpus that is too small for the planned token horizon"
        ),
    )
    parser.add_argument(
        "--tokenizer-batch-size",
        type=int,
        default=256,
        help="documents per tokenizer call while computing exact corpus token counts",
    )
    parser.add_argument("--tokenizer-local-files-only", action="store_true")
    return parser.parse_args()


def _count_batch_tokens(tokenizer, texts: list[str]) -> int:
    """Count model tokens plus one document-boundary token per document."""
    if callable(tokenizer):
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return sum(len(token_ids) + 1 for token_ids in encoded["input_ids"])
    return sum(
        len(tokenizer.encode(text, add_special_tokens=False)) + 1 for text in texts
    )


def stage_split(
    *,
    dataset_name: str | None,
    dataset_config: str | None,
    source_files: list[str] | None,
    source_split: str,
    output_split: str,
    text_column: str,
    max_documents: int,
    output_dir: Path,
    force: bool,
    skip_existing: bool,
    tokenizer=None,
    tokenizer_batch_size: int = 256,
) -> tuple[int, int, int | None] | None:
    if max_documents < 1:
        raise ValueError("document limits must be positive")
    if tokenizer_batch_size < 1:
        raise ValueError("tokenizer batch size must be positive")
    output = output_dir.joinpath(f"{output_split}.jsonl")
    # Deliberately does not start with "{output_split}": StreamingTextDataset
    # discovers local corpus files by globbing "{split}*{suffix}" (including
    # .json), so a "{split}.stats.json" name would itself get picked up and
    # fed to the model as training data.
    stats_path = output_dir.joinpath(f"stats.{output_split}.json")
    if output.exists() and not force:
        if skip_existing:
            return None
        raise FileExistsError(f"{output} already exists; pass --force to replace it")

    if source_files is not None:
        if not source_files:
            raise ValueError("source_files cannot be empty")
        suffixes = {Path(urlparse(source).path).suffix.lower() for source in source_files}
        builders = {".txt": "text", ".json": "json", ".jsonl": "json", ".parquet": "parquet"}
        if len(suffixes) != 1 or next(iter(suffixes)) not in builders:
            raise ValueError(f"source files must share a supported suffix: {source_files}")
        suffix = next(iter(suffixes))
        kwargs = {
            "path": builders[suffix],
            "data_files": {source_split: source_files},
            "split": source_split,
            "streaming": True,
        }
    else:
        if dataset_name is None:
            raise ValueError("dataset_name is required when source_files are not provided")
        kwargs = {
            "path": dataset_name,
            "split": source_split,
            "streaming": True,
        }
        if dataset_config is not None:
            kwargs["name"] = dataset_config
    stream = load_dataset(**kwargs)

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    documents = 0
    utf8_bytes = 0
    tokens = 0 if tokenizer is not None else None
    token_batch: list[str] = []
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            for row in stream:
                text = row.get(text_column)
                if not isinstance(text, str) or not text:
                    continue
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                documents += 1
                utf8_bytes += len(text.encode("utf-8"))
                if tokenizer is not None:
                    token_batch.append(text)
                    if len(token_batch) >= tokenizer_batch_size:
                        tokens += _count_batch_tokens(tokenizer, token_batch)
                        token_batch.clear()
                if documents >= max_documents:
                    break
            if tokenizer is not None and token_batch:
                tokens += _count_batch_tokens(tokenizer, token_batch)
        if documents == 0:
            raise RuntimeError(f"no usable text found in split {source_split!r}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    if tokens is not None:
        stats_temp = stats_path.with_name(f".{stats_path.name}.tmp-{os.getpid()}")
        try:
            stats_temp.write_text(
                json.dumps(
                    {"documents": documents, "utf8_bytes": utf8_bytes, "tokens": tokens},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            os.replace(stats_temp, stats_path)
        finally:
            stats_temp.unlink(missing_ok=True)
    return documents, utf8_bytes, tokens


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    train_files = args.train_file
    validation_files = args.validation_file
    if args.dataset is None and train_files is None and validation_files is None:
        train_files, validation_file = nanochat_climbmix_files(args.climbmix_train_shards)
        validation_files = [validation_file]
        print(
            f"source: NanoChat ClimbMix ({len(train_files)} train shards; "
            f"validation shard {NANOCHAT_CLIMBMIX_VALIDATION_SHARD:05d})"
        )
    elif args.dataset is None and (train_files is None or validation_files is None):
        raise ValueError(
            "custom file staging requires both --train-file and --validation-file"
        )
    elif args.dataset is not None and (train_files is not None or validation_files is not None):
        raise ValueError("use either --dataset or custom source files, not both")

    tokenizer = None
    if args.tokenizer is not None:
        tokenizer = get_tokenizer(
            hf_name=args.tokenizer,
            prefer_hf=True,
            local_files_only=args.tokenizer_local_files_only,
        )
    for source_split, output_split, limit, source_files in (
        (args.train_split, "train", args.train_documents, train_files),
        (
            args.validation_split,
            "validation",
            args.validation_documents,
            validation_files,
        ),
    ):
        staged = stage_split(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            source_files=source_files,
            source_split=source_split,
            output_split=output_split,
            text_column=args.text_column,
            max_documents=limit,
            output_dir=output_dir,
            force=args.force,
            skip_existing=args.skip_existing,
            tokenizer=tokenizer,
            tokenizer_batch_size=args.tokenizer_batch_size,
        )
        if staged is None:
            print(f"reuse {output_split}: {output_dir / f'{output_split}.jsonl'}")
            continue
        documents, utf8_bytes, tokens = staged
        message = (
            f"staged {output_split}: {documents:,} documents, "
            f"{utf8_bytes / 1024 / 1024:.2f} MiB"
        )
        if tokens is not None:
            message += f", {tokens:,} tokens"
        message += f" -> {output_dir / f'{output_split}.jsonl'}"
        print(message)
    # datasets/fsspec may still be retiring HTTP worker callbacks here.
    time.sleep(5.0)


if __name__ == "__main__":
    main()
