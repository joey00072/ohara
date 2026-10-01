"""Matched-token Llama / Qwen Next pretraining; see docs/pretraining-comparison.md."""

import argparse
import hashlib
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ohara.models.llama import Config as LlamaConfig, Llama
from ohara.models.qwen38 import Config as QwenConfig, Qwen38


def configs(sequence_length=512):
    return {
        "llama": LlamaConfig(
            vocab_size=50304, hidden_size=512, intermediate_size=1456,
            num_hidden_layers=16, num_attention_heads=8, num_key_value_heads=4,
            max_sequence_length=sequence_length, dropout=0.0,
        ),
        "qwen": QwenConfig(
            vocab_size=50304, hidden_size=512, num_layers=8,
            num_heads=8, num_kv_heads=2, head_dim=64, rotary_dim=32,
            linear_key_heads=4, linear_value_heads=8,
            linear_key_dim=64, linear_value_dim=64, residual_rank=64,
            expert_dim=256, index_dim=32, token_budget=128, query_chunk_size=128,
            ngram_vocab=97649, ngram_heads=4, ngram_dim=512, eos_token_id=50256,
            backend="cuda", mtp_steps=1,
        ),
    }


def make_model(name, config):
    return Llama(config) if name == "llama" else Qwen38(config)


def parameter_counts(model):
    total = sum(p.numel() for p in model.parameters())
    tables = model.ngram.embedding.weight.numel() if isinstance(model, Qwen38) else 0
    mtp = sum(p.numel() for p in model.mtp.parameters()) if isinstance(model, Qwen38) else 0
    return {"total": total, "non_table": total - tables, "ngram_tables": tables, "mtp": mtp}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer

    args.data.mkdir(parents=True, exist_ok=True)
    if any(args.data.iterdir()):
        raise FileExistsError("use an empty data directory")
    dataset_id, tokenizer_id = "HuggingFaceFW/fineweb-edu", "EleutherAI/gpt-neo-125m"
    api = HfApi()
    dataset_revision = api.dataset_info(dataset_id).sha
    tokenizer_revision = api.model_info(tokenizer_id).sha
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, revision=tokenizer_revision)
    stream = load_dataset(dataset_id, "sample-10BT", split="train", streaming=True,
                          revision=dataset_revision).shuffle(seed=args.seed, buffer_size=1000)
    limits = {"train": args.train_tokens + 1, "validation": args.validation_tokens + 1}
    arrays = {split: np.lib.format.open_memmap(args.data / f"{split}.npy", mode="w+",
              dtype=np.uint16, shape=(limit,)) for split, limit in limits.items()}
    counts = dict.fromkeys(limits, 0)
    documents = dict.fromkeys(limits, 0)
    seen = set()
    for batch in stream.iter(batch_size=128):
        texts = batch["text"]
        encoded = tokenizer(texts, add_special_tokens=False, return_attention_mask=False,
                            return_token_type_ids=False)["input_ids"]
        for text, ids in zip(texts, encoded, strict=True):
            key = hashlib.sha256(text.encode()).digest()
            if key in seen:
                continue
            seen.add(key)
            # Whole documents belong to one split, including repeated occurrences.
            split = "validation" if int.from_bytes(key[:8], "big") % 50 == 0 else "train"
            remaining = limits[split] - counts[split]
            if remaining <= 0:
                continue
            ids = ([tokenizer.eos_token_id] + ids)[:remaining]
            if ids and max(ids) >= 50304:
                raise ValueError("unexpected tokenizer vocabulary")
            arrays[split][counts[split]:counts[split] + len(ids)] = ids
            counts[split] += len(ids)
            documents[split] += 1
        print(json.dumps({"tokens": counts, "documents": documents}), flush=True)
        if all(counts[split] == limit for split, limit in limits.items()):
            break
    if counts != limits:
        raise RuntimeError(f"insufficient corpus: {counts}")
    for array in arrays.values():
        array.flush()
    metadata = {
        "dataset": dataset_id, "subset": "sample-10BT", "dataset_revision": dataset_revision,
        "tokenizer": tokenizer_id, "tokenizer_revision": tokenizer_revision,
        "eos_token_id": tokenizer.eos_token_id, "vocab_size": len(tokenizer),
        "seed": args.seed, "tokens": counts, "documents": documents,
        "split": "SHA256(text) first 8 bytes modulo 50; 0 held out; exact-text deduplicated",
        "sha256": {split: digest(args.data / f"{split}.npy") for split in limits},
    }
    (args.data / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def batch_at(tokens, offset, batch_size, sequence_length, device):
    count = batch_size * sequence_length
    block = torch.from_numpy(np.array(tokens[offset:offset + count + 1], dtype=np.int64)).to(device)
    if block.numel() != count + 1:
        raise ValueError("token budget exceeds the prepared corpus")
    return block[:-1].reshape(batch_size, sequence_length), block[1:].reshape(batch_size, sequence_length)


def losses(model, inputs, targets):
    precision = torch.autocast("cuda", dtype=torch.bfloat16) if inputs.is_cuda else nullcontext()
    with precision:
        if isinstance(model, Qwen38):
            output = model(inputs, targets, loss_chunk_size=512)
            return output.loss, output.loss - output.auxiliary_loss
        loss = model(inputs, targets=targets, loss_chunk_size=512) / targets.numel()
        return loss, loss


@torch.no_grad()
def evaluate(model, tokens, args):
    stride = args.micro_batch * args.sequence_length
    batches = (len(tokens) - 1) // stride
    if batches < 1:
        raise ValueError("validation corpus is smaller than a batch")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total = 0.0
    try:
        for index in range(batches):
            inputs, targets = batch_at(
                tokens, index * stride, args.micro_batch, args.sequence_length, device
            )
            _, ce = losses(model, inputs, targets)
            if not torch.isfinite(ce):
                raise RuntimeError("non-finite validation loss")
            total += ce.item()
        return total / batches
    finally:
        model.train(was_training)


def learning_rate(step, steps, peak, warmup):
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("this comparison requires a CUDA GPU")
    if not 0 < args.warmup < args.steps:
        raise ValueError("warmup must be positive and smaller than steps")
    output = args.output / args.model
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output}")
    metadata = json.loads((args.data / "metadata.json").read_text())
    for split, expected in metadata["sha256"].items():
        if digest(args.data / f"{split}.npy") != expected:
            raise ValueError(f"{split} data checksum mismatch")
    if metadata["eos_token_id"] != 50256 or metadata["vocab_size"] != 50257:
        raise ValueError("data must use the shared GPT-2 tokenizer")
    tokens = np.load(args.data / "train.npy", mmap_mode="r")
    validation = np.load(args.data / "validation.npy", mmap_mode="r")
    per_step = args.micro_batch * args.sequence_length * args.accumulation
    if len(tokens) < per_step * args.steps + 1:
        raise ValueError("prepare enough unique tokens for this run")
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    cfg = configs(args.sequence_length)[args.model]
    model = make_model(args.model, cfg).cuda()
    counts = parameter_counts(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                                  eps=1e-8, weight_decay=0.1, fused=True)
    recipe = {
        "model": args.model, "config": asdict(cfg), "parameters": counts,
        "data": metadata, "steps": args.steps, "tokens_per_step": per_step,
        "micro_batch": args.micro_batch, "accumulation": args.accumulation,
        "sequence_length": args.sequence_length, "peak_lr": args.lr,
        "warmup": args.warmup, "seed": args.seed, "eval_every": args.eval_every,
        "optimizer": "AdamW, betas=(0.9,0.95), eps=1e-8, weight_decay=0.1, clip=1",
        "schedule": "linear warmup; cosine decay to 10% of peak",
        "precision": "FP32 parameters and Adam moments; BF16 autocast",
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
    }
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    print(json.dumps(recipe), flush=True)
    started = time.perf_counter()
    with (output / "metrics.jsonl").open("w", buffering=1) as log:
        def record(values):
            line = json.dumps(values)
            log.write(line + "\n")
            print(line, flush=True)

        record({"step": 0, "tokens": 0, "validation_ce": evaluate(model, validation, args),
                "elapsed_seconds": time.perf_counter() - started})
        for step in range(1, args.steps + 1):
            step_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(step, args.steps, args.lr, args.warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr
            objective_sum = torch.zeros((), device="cuda")
            ce_sum = torch.zeros((), device="cuda")
            for micro in range(args.accumulation):
                offset = (step - 1) * per_step + micro * args.micro_batch * args.sequence_length
                inputs, targets = batch_at(tokens, offset, args.micro_batch, args.sequence_length, "cuda")
                objective, ce = losses(model, inputs, targets)
                (objective / args.accumulation).backward()
                objective_sum += objective.detach() / args.accumulation
                ce_sum += ce.detach() / args.accumulation
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            values = {"step": step, "tokens": step * per_step, "train_ce": ce_sum.item(),
                      "objective": objective_sum.item(), "grad_norm": norm.item(), "lr": lr,
                      "step_seconds": time.perf_counter() - step_started}
            if step % args.eval_every == 0 or step == args.steps:
                values["validation_ce"] = evaluate(model, validation, args)
            values["elapsed_seconds"] = time.perf_counter() - started
            record(values)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "recipe": recipe, "step": args.steps, "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state()}, output / "final.pt")


def plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    reference = None
    summary = {}
    for name, color in (("llama", "#2563eb"), ("qwen", "#d97706")):
        folder = args.output / name
        recipe = json.loads((folder / "recipe.json").read_text())
        rows = [json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()]
        matched = {key: recipe[key] for key in (
            "data", "steps", "tokens_per_step", "micro_batch", "accumulation",
            "sequence_length", "peak_lr", "warmup", "seed", "eval_every",
            "optimizer", "schedule", "precision",
        )}
        if reference is not None and matched != reference:
            raise ValueError("the runs do not use matching data and training recipes")
        reference = matched
        if not rows or rows[-1]["step"] != recipe["steps"] or "validation_ce" not in rows[-1]:
            raise ValueError(f"{name} run has not completed")
        counts = recipe["parameters"]
        label = (f"Llama {counts['total'] / 1e6:.1f}M" if name == "llama" else
                 f"Qwen Next {counts['non_table'] / 1e6:.1f}M + {counts['ngram_tables'] / 1e6:.1f}M n-gram")
        training = [row for row in rows if "train_ce" in row]
        window = min(16, len(training))
        x = np.array([row["tokens"] / 1e6 for row in training])
        y = np.array([row["train_ce"] for row in training])
        if window:
            axes[0].plot(x, y, color=color, alpha=0.15, linewidth=0.7)
            axes[0].plot(x[window - 1:], np.convolve(y, np.ones(window) / window, mode="valid"),
                         color=color, label=label, linewidth=2)
        validation = [row for row in rows if "validation_ce" in row]
        summary[name] = {"parameters": counts, "tokens": rows[-1]["tokens"],
                         "final_validation_ce": validation[-1]["validation_ce"],
                         "final_train_ce_mean_16": float(y[-16:].mean()),
                         "elapsed_seconds": rows[-1]["elapsed_seconds"]}
        axes[1].plot([row["tokens"] / 1e6 for row in validation],
                     [row["validation_ce"] for row in validation], marker="o", markersize=3,
                     color=color, label=label, linewidth=2)
    for ax, title in zip(axes, ("Training · 16-step trailing mean", "Held-out validation"), strict=True):
        ax.set(title=title, xlabel="Training tokens (millions)", ylabel="Next-token cross-entropy (nats)")
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8)
    fig.suptitle("Small pretraining comparison · same data, tokenizer and token budget", fontsize=13)
    for extension in ("png", "svg"):
        fig.savefig(args.output / f"loss.{extension}", dpi=180)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("describe", "prepare", "train", "plot"))
    parser.add_argument("--model", choices=("llama", "qwen"))
    parser.add_argument("--data", type=Path, default=Path("data/pretraining-comparison"))
    parser.add_argument("--output", type=Path, default=Path("runs/pretraining-comparison"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-tokens", type=int, default=8_388_608)
    parser.add_argument("--validation-tokens", type=int, default=131_072)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=32)
    args = parser.parse_args()
    for name in ("train_tokens", "validation_tokens", "steps", "sequence_length", "micro_batch", "accumulation", "warmup", "lr", "eval_every"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"{name} must be finite and positive")
    if args.action == "describe":
        for name, cfg in configs(args.sequence_length).items():
            with torch.device("meta"):
                model = make_model(name, cfg)
            print(json.dumps({"model": name, "parameters": parameter_counts(model), "config": asdict(cfg)}, indent=2))
    elif args.action == "train":
        if args.model is None:
            parser.error("train requires --model")
        train(args)
    elif args.action == "prepare":
        prepare(args)
    else:
        plot(args)


if __name__ == "__main__":
    main()
