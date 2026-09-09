"""Precompute a teacher's top-k logits over a pre-tokenized corpus.

    python examples/distill_cache.py --teacher Qwen/Qwen3-0.6B-Base \
        --bin data/qwen_corpus/train.bin --out runs/qwen_top8 --blocks 76700

Run once. Every student trained against this cache afterwards pays nothing for
the teacher, which is what makes "is distillation faster than training from
scratch?" answerable: the teacher's cost becomes a single measurable number
rather than a tax on every step.

The corpus must be tokenized with the *teacher's* tokenizer -- top-k indices are
positions in the teacher's vocabulary, and token boundaries have to line up.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from ohara.distill import build_teacher_cache, cache_bytes_per_block


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache a teacher's top-k logits")
    parser.add_argument("--teacher", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--bin", required=True, help="token bin tokenized with the teacher's tokenizer")
    parser.add_argument("--out", required=True, help="cache prefix")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--blocks",
        type=int,
        default=None,
        help="how many blocks to cache; default is the whole bin",
    )
    parser.add_argument("--batch-blocks", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true", help="only report the size it would take")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_block = cache_bytes_per_block(args.seq_len, args.top_k)
    if args.blocks:
        print(
            f"cache size: {args.blocks:,} blocks x {per_block / 1e6:.2f} MB = "
            f"{args.blocks * per_block / 1e9:.2f} GB "
            f"({args.blocks * (args.seq_len + 1) / 1e6:.0f}M tokens covered)"
        )
    if args.dry_run:
        return

    print(f"loading teacher {args.teacher}")
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, dtype=torch.bfloat16)
    teacher.name_or_path = args.teacher

    started = time.perf_counter()
    payload = build_teacher_cache(
        teacher,
        Path(args.bin),
        Path(args.out),
        seq_len=args.seq_len,
        top_k=args.top_k,
        num_blocks=args.blocks,
        batch_blocks=args.batch_blocks,
        device=args.device,
    )
    elapsed = time.perf_counter() - started
    tokens = int(payload["tokens_covered"])
    print(
        f"built in {elapsed / 60:.1f} min "
        f"({tokens / max(elapsed, 1e-9):,.0f} teacher tokens/s) -> {args.out}.*"
    )


if __name__ == "__main__":
    main()
