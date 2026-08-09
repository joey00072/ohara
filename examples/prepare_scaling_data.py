from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage a bounded Hugging Face text corpus once for offline scaling sweeps"
    )
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--validation-file", default=None)
    parser.add_argument("--train-documents", type=int, default=100_000)
    parser.add_argument("--validation-documents", type=int, default=10_000)
    parser.add_argument("--output-dir", default="./data/scaling_corpus")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def stage_split(
    *,
    dataset_name: str,
    dataset_config: str | None,
    source_file: str | None,
    source_split: str,
    output_split: str,
    text_column: str,
    max_documents: int,
    output_dir: Path,
    force: bool,
    skip_existing: bool,
) -> tuple[int, int] | None:
    if max_documents < 1:
        raise ValueError("document limits must be positive")
    output = output_dir.joinpath(f"{output_split}.jsonl")
    if output.exists() and not force:
        if skip_existing:
            return None
        raise FileExistsError(f"{output} already exists; pass --force to replace it")

    if source_file is not None:
        suffix = Path(urlparse(source_file).path).suffix.lower()
        builders = {".txt": "text", ".json": "json", ".jsonl": "json", ".parquet": "parquet"}
        if suffix not in builders:
            raise ValueError(f"unsupported source file: {source_file}")
        kwargs = {
            "path": builders[suffix],
            "data_files": {source_split: source_file},
            "split": source_split,
            "streaming": True,
        }
    else:
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
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            for row in stream:
                text = row.get(text_column)
                if not isinstance(text, str) or not text:
                    continue
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                documents += 1
                utf8_bytes += len(text.encode("utf-8"))
                if documents >= max_documents:
                    break
        if documents == 0:
            raise RuntimeError(f"no usable text found in split {source_split!r}")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return documents, utf8_bytes


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    for source_split, output_split, limit, source_file in (
        (args.train_split, "train", args.train_documents, args.train_file),
        (
            args.validation_split,
            "validation",
            args.validation_documents,
            args.validation_file,
        ),
    ):
        staged = stage_split(
            dataset_name=args.dataset,
            dataset_config=args.dataset_config,
            source_file=source_file,
            source_split=source_split,
            output_split=output_split,
            text_column=args.text_column,
            max_documents=limit,
            output_dir=output_dir,
            force=args.force,
            skip_existing=args.skip_existing,
        )
        if staged is None:
            print(f"reuse {output_split}: {output_dir / f'{output_split}.jsonl'}")
            continue
        documents, utf8_bytes = staged
        print(
            f"staged {output_split}: {documents:,} documents, "
            f"{utf8_bytes / 1024 / 1024:.2f} MiB -> {output_dir / f'{output_split}.jsonl'}"
        )
    # datasets/fsspec may still be retiring HTTP worker callbacks here.
    time.sleep(5.0)


if __name__ == "__main__":
    main()
