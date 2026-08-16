"""Serve a browser chat UI for a finetuned ohara checkpoint.

    python examples/chat_web.py --checkpoint ./ckpt/sft.pt

Then open http://localhost:8080. To reach it from a laptop while the model runs
on a remote box, either bind publicly with ``--host 0.0.0.0`` or, better, leave
it on localhost and forward the port over ssh:

    ssh -N -L 8080:localhost:8080 root@<host>
"""

from __future__ import annotations

import argparse

import torch

from ohara.chat_engine import ChatEngine, SamplingConfig
from ohara.webui import serve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with a finetuned ohara model in a browser")
    parser.add_argument("--checkpoint", default="./ckpt/sft.pt")
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="tokenizer saved beside the checkpoint (default: <checkpoint stem>/)",
    )
    parser.add_argument(
        "--tokenizer",
        default="EleutherAI/gpt-neo-125m",
        help="fallback tokenizer when no directory was saved next to the checkpoint",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dtype",
        default=None,
        choices=("float32", "bfloat16", "float16"),
        help="default: bfloat16 on cuda, float32 on cpu",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype) if args.dtype else None

    engine = ChatEngine.from_checkpoint(
        args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        tokenizer_name=args.tokenizer,
        device=args.device,
        dtype=dtype,
    )
    info = engine.metadata(args.checkpoint)
    print(
        f"loaded {args.checkpoint}: {info['parameters'] / 1e6:.1f}M params, "
        f"{info['layers']} layers, ctx {info['context_length']}, "
        f"on {info['device']} in {info['dtype']}"
    )

    sampling = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_new_tokens=min(args.max_new_tokens, info["context_length"]),
    )
    serve(
        engine,
        host=args.host,
        port=args.port,
        checkpoint_path=args.checkpoint,
        sampling=sampling,
    )


if __name__ == "__main__":
    main()
