from __future__ import annotations

import argparse
import json
from contextlib import nullcontext

import torch
from transformers import AutoModelForCausalLM

from ohara.chat import add_chat_tokens
from ohara.chat_engine import config_from_state_dict, strip_wrapper_prefixes
from ohara.core_eval import evaluate_core
from ohara.models.llama import Llama
from ohara.tokenizer import get_tokenizer
from ohara.utils import auto_accelerator


def main():
    parser = argparse.ArgumentParser(description="Run CORE-style evaluation")
    parser.add_argument(
        "--hf-model",
        type=str,
        default="openai-community/gpt2",
        help="HuggingFace model id (ignored when --checkpoint is given)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="evaluate an ohara checkpoint (.pt) instead of a HuggingFace model",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="EleutherAI/gpt-neo-125m",
        help="tokenizer for --checkpoint",
    )
    parser.add_argument(
        "--chat-tokens",
        action="store_true",
        help="extend the tokenizer with the chat special tokens, as training did",
    )
    parser.add_argument(
        "--moe-experts-per-tok",
        type=int,
        default=2,
        help="top-k for MoE checkpoints; no tensor shape records it",
    )
    parser.add_argument("--label", type=str, default=None, help="name for the result row")
    parser.add_argument("--output-json", type=str, default=None)
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
    if args.checkpoint:
        tokenizer = get_tokenizer(hf_name=args.tokenizer, prefer_hf=True)
        if args.chat_tokens:
            add_chat_tokens(tokenizer)
    else:
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

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        state = checkpoint.get("model", checkpoint)
        state = {strip_wrapper_prefixes(k): v for k, v in state.items()}
        config = config_from_state_dict(state, moe_experts_per_tok=args.moe_experts_per_tok)
        model = Llama(config)
        model.load_state_dict(state, strict=False)
        model = model.to(device=device, dtype=model_dtype)
        print(
            f"loaded {args.checkpoint}: {config.num_hidden_layers}L "
            f"hidden={config.hidden_size} vocab={config.vocab_size} "
            + (
                f"moe={config.moe_num_experts}x top-{config.moe_experts_per_tok}"
                if config.moe_num_experts
                else "dense"
            )
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=model_dtype)
        model = model.to(device)
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

    label = args.label or (args.checkpoint or args.hf_model)
    print(json.dumps(results, indent=2))
    print(f"CORE  {label}: {results['core_metric']:.4f}")

    if args.output_json:
        from pathlib import Path

        payload = {"label": label, **results}
        Path(args.output_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
