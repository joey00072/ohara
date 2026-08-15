"""Tokenize a staged corpus once into flat token bins for fast training.

    python examples/pretokenize_corpus.py --corpus ./data/scaling_corpus

Reads the ``{split}.jsonl`` files written by ``prepare_scaling_data.py`` and
writes ``{split}.bin`` beside them. Training then memory-maps those instead of
tokenizing text inside the training loop, which is what keeps the GPUs waiting.

Run it once per (corpus, tokenizer) pair; the sidecar records which tokenizer
produced each bin so a mismatch fails loudly at load rather than training on
garbage ids.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

from ohara.chat import add_chat_tokens
from ohara.tokenbin import read_token_bin_metadata, write_token_bin
from ohara.tokenizer import get_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-tokenize a staged corpus into token bins")
    parser.add_argument("--corpus", default="./data/scaling_corpus")
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neo-125m")
    parser.add_argument("--tokenizer-local-files-only", action="store_true")
    parser.add_argument(
        "--chat-tokens",
        action="store_true",
        help="reserve the conversation special tokens, matching --chat-tokens at train time",
    )
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path, text_column: str) -> Iterator[str]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get(text_column)
            if isinstance(text, str) and text:
                yield text


def main() -> None:
    args = parse_args()
    corpus = Path(args.corpus)
    if not corpus.is_dir():
        raise NotADirectoryError(f"corpus directory not found: {corpus}")

    tokenizer = get_tokenizer(
        hf_name=args.tokenizer,
        prefer_hf=True,
        local_files_only=args.tokenizer_local_files_only,
    )
    if args.chat_tokens:
        add_chat_tokens(tokenizer)
    print(f"tokenizer {args.tokenizer}: vocab {len(tokenizer):,}")

    for split in (item.strip() for item in args.splits.split(",") if item.strip()):
        source = corpus / f"{split}.jsonl"
        if not source.exists():
            raise FileNotFoundError(f"missing staged split: {source}")
        destination = corpus / f"{split}.bin"

        if destination.exists() and not args.force:
            existing = read_token_bin_metadata(destination)
            if existing["vocab_size"] == len(tokenizer):
                print(f"reuse {destination}: {existing['tokens']:,} tokens")
                continue
            raise FileExistsError(
                f"{destination} was built with a {existing['vocab_size']:,} token vocabulary "
                f"but this run has {len(tokenizer):,}; pass --force to rebuild"
            )

        print(f"tokenizing {source} -> {destination}")
        started = time.perf_counter()
        metadata = write_token_bin(
            read_jsonl(source, args.text_column),
            tokenizer,
            destination,
            batch_size=args.batch_size,
        )
        elapsed = time.perf_counter() - started
        rate = metadata["tokens"] / max(elapsed, 1e-9)
        print(
            f"  {metadata['tokens']:,} tokens in {elapsed / 60:.1f}m "
            f"({rate / 1e6:.2f}M tokens/s), {destination.stat().st_size / 1e9:.2f} GB"
        )


if __name__ == "__main__":
    main()
