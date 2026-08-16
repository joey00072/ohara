"""Export a training checkpoint to safetensors plus a config sidecar.

    python examples/export_safetensors.py --checkpoint ckpt/moe_d12.pt --out export/moe-base

Training checkpoints are pickled ``.pt`` files holding optimizer state, RNG state
and wrapper-prefixed weights. That is the right format to resume from and the
wrong one to publish: it executes arbitrary code on load, it is several times
larger than the weights, and the architecture is only recoverable by guessing
from tensor shapes.

This writes ``model.safetensors`` (weights only, no code execution) next to a
``config.json`` recording the architecture. The config matters more than it
looks: ``moe_experts_per_tok`` leaves no trace in any tensor shape, so without it
a mixture-of-experts checkpoint loads with correct weights but routes differently
than it was trained to.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from ohara.chat_engine import config_from_state_dict, strip_wrapper_prefixes


# Rebuilt at construction time from max_sequence_length, and counted per step
# rather than learned. Publishing them would only bloat the file.
DERIVED_BUFFERS = ("freq_cos", "freq_sin", "qb_beta_sum", "qb_beta_count", "expert_counts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a checkpoint to safetensors")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--moe-experts-per-tok",
        type=int,
        default=2,
        help="top-k the model was trained with; not recoverable from tensor shapes",
    )
    parser.add_argument("--moe-gate-fn", choices=("softmax", "sigmoid"), default="softmax")
    parser.add_argument(
        "--moe-no-normalize-weights",
        action="store_true",
        help="the grouped checkpoint was trained without normalizing sigmoid weights",
    )
    parser.add_argument(
        "--keep-derived-buffers",
        action="store_true",
        help="also export rotary and router-statistics buffers",
    )
    return parser.parse_args()


def export(
    checkpoint_path: Path,
    out_dir: Path,
    *,
    moe_experts_per_tok: int = 2,
    moe_gate_fn: str = "softmax",
    moe_normalize_weights: bool = True,
    keep_derived_buffers: bool = False,
) -> dict[str, object]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw = checkpoint.get("model", checkpoint)
    state = {strip_wrapper_prefixes(key): value for key, value in raw.items()}

    config = config_from_state_dict(
        state,
        moe_experts_per_tok=moe_experts_per_tok,
        moe_gate_fn=moe_gate_fn,
        moe_normalize_weights=moe_normalize_weights,
    )

    tensors: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        if not keep_derived_buffers and key.endswith(DERIVED_BUFFERS):
            continue
        # safetensors rejects shared storage and needs a contiguous layout, so
        # clone rather than risk an alias between tied or viewed tensors.
        tensors[key] = value.detach().cpu().contiguous().clone()

    if not tensors:
        raise RuntimeError("no tensors found in checkpoint")

    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "model.safetensors"
    save_file(
        tensors,
        weights_path,
        metadata={"format": "pt", "source": checkpoint_path.name},
    )

    config_payload = {
        "architecture": "ohara.models.llama.Llama",
        "iteration": checkpoint.get("idx"),
        **asdict(config),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2) + "\n", encoding="utf-8"
    )

    return {
        "tensors": len(tensors),
        "parameters": sum(t.numel() for t in tensors.values()),
        "bytes": weights_path.stat().st_size,
        "config": config_payload,
    }


def main() -> None:
    args = parse_args()
    result = export(
        Path(args.checkpoint),
        Path(args.out),
        moe_experts_per_tok=args.moe_experts_per_tok,
        moe_gate_fn=args.moe_gate_fn,
        moe_normalize_weights=not args.moe_no_normalize_weights,
        keep_derived_buffers=args.keep_derived_buffers,
    )
    config = result["config"]
    print(f"exported {args.checkpoint} -> {args.out}/model.safetensors")
    print(f"  {result['tensors']} tensors, {result['parameters']:,} params, "
          f"{result['bytes'] / 1e9:.2f} GB")
    print(
        f"  {config['num_hidden_layers']}L hidden={config['hidden_size']} "
        f"vocab={config['vocab_size']} "
        + (
            f"moe={config['moe_num_experts']}x top-{config['moe_experts_per_tok']}"
            if config["moe_num_experts"]
            else "dense"
        )
    )


if __name__ == "__main__":
    main()
