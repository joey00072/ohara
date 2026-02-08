from __future__ import annotations

import argparse
import json
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM

from ohara.core_eval import evaluate_core
from ohara.tokenizer import get_tokenizer
from ohara.utils import auto_accelerator


def main():
    parser = argparse.ArgumentParser(description="Run CORE-style evaluation")
    parser.add_argument(
        "--hf-model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model id",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=100,
        help="max samples per CORE task (-1 for all)",
    )
    parser.add_argument(
        "--eval-bundle-dir",
        type=str,
        default="./eval_bundle",
        help="local directory for eval bundle",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "bfloat16"],
        help="inference dtype",
    )
    args = parser.parse_args()

    device = auto_accelerator()
    tokenizer = get_tokenizer(hf_name=args.hf_model, prefer_hf=True)

    model_dtype = None
    if args.dtype == "float32":
        model_dtype = torch.float32
    elif args.dtype == "bfloat16":
        model_dtype = torch.bfloat16
    elif device.type == "cuda":
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=model_dtype)
    model.to(device)
    model.eval()

    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=model_dtype)
        if device.type == "cuda" and model_dtype in {torch.float16, torch.bfloat16}
        else nullcontext()
    )

    with autocast_ctx:
        results = evaluate_core(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_per_task=args.max_per_task,
            eval_bundle_dir=args.eval_bundle_dir,
        )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
