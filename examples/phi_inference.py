"""Download microsoft/phi-2 and generate from it in fp16.

    uv run python examples/phi_inference.py --prompt "Once upon a time"
    uv run python examples/phi_inference.py --interactive
"""

from __future__ import annotations

import argparse
import time

from transformers import AutoTokenizer

from ohara.inference import Inference
from ohara.models.phi import Phi
from ohara.utils import auto_accelerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="microsoft/phi-2")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-p", type=float, default=0.2)
    parser.add_argument("--no-kv-cache", action="store_true", help="disable the KV cache")
    parser.add_argument("--interactive", action="store_true", help="keep prompting for input")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = auto_accelerator()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Downloads the checkpoint from the Hub and loads it into ohara's own Phi.
    model = Phi.from_pretrained(args.model).to(device).eval()

    engine = Inference(model, tokenizer, device, use_kv_cache=not args.no_kv_cache)

    while True:
        prompt = input("Prompt: ") if args.interactive else args.prompt
        if args.interactive and prompt.strip() in ("", "exit"):
            break

        start = time.perf_counter()
        print(
            engine.generate(
                prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                stream=False,
            )
        )
        print(f"Time taken: {time.perf_counter() - start:.2f}s")

        if not args.interactive:
            break


if __name__ == "__main__":
    main()
