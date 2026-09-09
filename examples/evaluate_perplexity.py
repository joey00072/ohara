from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ohara.models.qwen3 import Qwen3
from ohara.models.llama import Llama
from huggingface_hub import hf_hub_download
from ohara.perplexity import fixed_block_perplexity, sliding_window_perplexity
from ohara.utils import auto_accelerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate explicitly defined token perplexity")
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local directory")
    parser.add_argument("--backend", choices=("hf", "ohara"), default="hf")
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--protocol", choices=("fixed", "sliding"), default="fixed")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = auto_accelerator()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.backend == "ohara":
        config_path = Path(args.model) / "config.json"
        if not config_path.is_file():
            config_path = Path(hf_hub_download(args.model, "config.json"))
        model_type = json.loads(config_path.read_text())["model_type"]
        model_classes = {"ohara_llama": Llama, "qwen3": Qwen3}
        if model_type not in model_classes:
            raise ValueError(f"unsupported Ohara model_type: {model_type}")
        model = model_classes[model_type].from_pretrained(args.model, device=device, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            attn_implementation="eager",
        ).to(device).eval()

    split = load_dataset(args.dataset, args.dataset_config, split=args.split)
    text = "\n\n".join(split[args.text_column])
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
        verbose=False,
    ).input_ids[0]

    common = {
        "device": device,
        "sequence_length": args.sequence_length,
    }
    if args.protocol == "fixed":
        metrics = fixed_block_perplexity(
            model,
            token_ids,
            batch_size=args.batch_size,
            **common,
        )
    else:
        metrics = sliding_window_perplexity(
            model,
            token_ids,
            stride=args.stride,
            **common,
        )

    result = {
        "model": args.model,
        "backend": args.backend,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "dtype": args.dtype,
        "add_special_tokens": False,
        "document_separator": "\\n\\n",
        "source_tokens": token_ids.numel(),
        "sequence_length": args.sequence_length,
        **metrics,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output_json is not None:
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
