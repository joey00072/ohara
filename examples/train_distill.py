"""Train a student with hard-label loss, cached teacher logits, or a blend.

    # baseline: pure next-token cross-entropy
    uv run python examples/train_distill.py --distill-alpha 1.0 --teacher-cache runs/qwen_top8 ...

    # distillation: half hard labels, half teacher top-k
    uv run python examples/train_distill.py --distill-alpha 0.5 --teacher-cache runs/qwen_top8 ...

Both arms of an A/B run this same script and differ only in ``--distill-alpha``,
so no code-path difference can confound the comparison. At ``alpha=1.0`` the
teacher terms are skipped entirely and the loss is identical to ordinary
pretraining.

The loss is::

    L = alpha * CE(hard labels) + (1 - alpha) * CE(teacher top-k + "other")

Both terms are summed over tokens and divided by the same token count, so the
weighting means what it looks like it means.

Validation is always pure cross-entropy on held-out data, which keeps
bits-per-byte comparable between arms even though their training objectives
differ. Bits-per-byte is also byte-normalised, so it stays comparable across
tokenizers -- useful here, since this script runs on a Qwen-tokenized corpus
while earlier runs used gpt-neo.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ohara.distill import DistillTokenBinDataset, TeacherCache, distillation_loss, hard_label_loss
from ohara.models.llama import Config, Llama
from ohara.modules.moe import apply_qb_update
from ohara.optimizer import build_muon_adamw
from ohara.runtime import (
    EngineConfig,
    OharaEngine,
    ParallelConfig,
    PrecisionConfig,
    PrecisionMode,
)
from ohara.scaling import CosineWeightDecayScheduler, MuonMomentumScheduler, WarmupStableDecayScheduler
from ohara.tokenbin import TokenBinDataset
from ohara.tokenizer import get_token_bytes, get_tokenizer
from ohara.tracking import BACKENDS as TRACKING_BACKENDS, create_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a student with optional distillation")
    parser.add_argument("--corpus", default="./data/qwen_corpus")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help=(
            "student vocabulary; must match the teacher's output width so cached "
            "top-k indices are in range (Qwen3 pads 151,669 to 151,936)"
        ),
    )
    parser.add_argument("--teacher-cache", required=True, help="prefix of a cache from distill_cache.py")
    parser.add_argument(
        "--distill-alpha",
        type=float,
        default=1.0,
        help="weight on hard labels; 1.0 is pure cross-entropy, 0.5 an even blend",
    )
    # shapes
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=300)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=448)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=6)
    # moe
    parser.add_argument("--moe-num-experts", type=int, default=64)
    parser.add_argument("--moe-experts-per-tok", type=int, default=4)
    parser.add_argument("--moe-num-shared-experts", type=int, default=1)
    parser.add_argument("--moe-gate-fn", choices=("softmax", "sigmoid"), default="sigmoid")
    # optimization
    parser.add_argument("--matrix-learning-rate", type=float, default=0.02)
    parser.add_argument("--embedding-learning-rate", type=float, default=0.3)
    parser.add_argument("--unembedding-learning-rate", type=float, default=0.008)
    parser.add_argument("--scalar-learning-rate", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.28)
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--warmdown-ratio", type=float, default=0.65)
    parser.add_argument("--final-lr-fraction", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    # runtime
    parser.add_argument("--precision", default=PrecisionMode.BF16_MIXED.value,
                        choices=[mode.value for mode in PrecisionMode])
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-path", default="./ckpt/distill.pt")
    parser.add_argument("--logger", choices=TRACKING_BACKENDS, default="auto")
    parser.add_argument("--project", default="ohara-distill")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--result-json", default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, engine, token_bytes, batches: int) -> dict[str, float]:
    """Pure cross-entropy on held-out data, so both arms are measured identically."""
    was_training = model.training
    model.eval()
    total_nats = torch.zeros((), device=engine.device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=engine.device, dtype=torch.float64)
    total_bytes = torch.zeros((), device=engine.device, dtype=torch.float64)

    iterator = iter(loader)
    for _ in range(batches):
        data, target = engine.to_device(next(iterator))
        with engine.autocast_context():
            logits = model(data)
        loss, count = hard_label_loss(logits, target)
        total_nats += loss.double()
        total_tokens += count.double()
        total_bytes += token_bytes[target.reshape(-1)].sum().double()

    if was_training:
        model.train()
    total_nats = engine.all_reduce(total_nats, "sum")
    total_tokens = engine.all_reduce(total_tokens, "sum")
    total_bytes = engine.all_reduce(total_bytes, "sum")
    tokens = float(total_tokens)
    nats = float(total_nats)
    byte_count = float(total_bytes)
    mean = nats / max(tokens, 1.0)
    return {
        "loss": mean,
        "ppl": math.exp(min(mean, 60.0)),
        "bpb": nats / (math.log(2) * byte_count) if byte_count > 0 else float("nan"),
        "accuracy": float("nan"),
    }


def run() -> None:
    args = parse_args()
    if not 0.0 <= args.distill_alpha <= 1.0:
        raise ValueError("distill-alpha must be in [0, 1]")
    distilling = args.distill_alpha < 1.0
    if distilling and not args.teacher_cache:
        raise ValueError("--distill-alpha below 1.0 needs --teacher-cache")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    engine = OharaEngine(
        EngineConfig(
            precision=PrecisionConfig(mode=PrecisionMode(args.precision)),
            parallel=ParallelConfig(tp=1),
        )
    )
    engine.launch()

    tokenizer = get_tokenizer(hf_name=args.tokenizer, prefer_hf=True)
    cache = TeacherCache(args.teacher_cache)
    vocab_size = args.vocab_size or cache.vocab_size
    if vocab_size != cache.vocab_size or vocab_size < len(tokenizer):
        raise ValueError("student vocabulary must match teacher cache and cover the tokenizer")

    config = Config(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        max_sequence_length=args.seq_len,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        dropout=0.0,
        weight_tying=False,
        init_style="nanochat",
        moe_num_experts=args.moe_num_experts,
        moe_experts_per_tok=args.moe_experts_per_tok,
        moe_grouped=args.moe_num_experts > 0,
        moe_num_shared_experts=args.moe_num_shared_experts if args.moe_num_experts else 0,
        moe_gate_fn=args.moe_gate_fn,
    )
    raw_model = Llama(config)
    model = torch.compile(raw_model, dynamic=False) if args.compile else raw_model
    model = engine.prepare(model)

    optimizer = build_muon_adamw(
        model,
        matrix_learning_rate=args.matrix_learning_rate,
        embedding_learning_rate=args.embedding_learning_rate,
        unembedding_learning_rate=args.unembedding_learning_rate,
        scalar_learning_rate=args.scalar_learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer = engine.prepare_optimizers(optimizer)[0]

    corpus = Path(args.corpus)
    train_ds = DistillTokenBinDataset(
        corpus / "train.bin", args.teacher_cache,
        max_length=args.seq_len, shuffle=True, seed=args.seed,
        student_vocab_size=vocab_size,
    )
    blocks = train_ds.num_blocks
    val_ds = TokenBinDataset(
        corpus / "validation.bin", max_length=args.seq_len, shuffle=False, seed=args.seed
    )

    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers,
                   "pin_memory": engine.device.type == "cuda"}
    train_loader = DataLoader(train_ds, **loader_args)
    val_loader = DataLoader(val_ds, **loader_args)

    scheduler = WarmupStableDecayScheduler(
        learning_rate=args.matrix_learning_rate, max_iters=args.max_iters,
        warmup_iters=args.warmup_iters, warmdown_ratio=args.warmdown_ratio,
        final_lr_fraction=args.final_lr_fraction,
    )
    momentum = MuonMomentumScheduler(max_iters=args.max_iters, warmdown_ratio=args.warmdown_ratio,
                                     warmup_iters=min(200, args.max_iters // 4))
    decay = CosineWeightDecayScheduler(weight_decay=args.weight_decay, max_iters=args.max_iters)

    token_bytes = get_token_bytes(tokenizer, device=engine.device)
    if token_bytes.numel() < vocab_size:
        token_bytes = torch.nn.functional.pad(token_bytes, (0, vocab_size - token_bytes.numel()))

    tracker = None
    if engine.is_global_zero:
        tracker = create_logger(
            args.logger, project=args.project, run_name=args.run_name,
            config={**vars(args), "vocab_size": vocab_size, "blocks": blocks,
                    "parameters": sum(p.numel() for p in raw_model.parameters())},
        )
        engine.loggers = [tracker]
        mode = "distill" if distilling else "hard-label only"
        print(
            f"{mode}: alpha={args.distill_alpha} vocab={vocab_size:,} "
            f"params={sum(p.numel() for p in raw_model.parameters()):,} "
            f"active={raw_model.active_matmul_parameters():,} blocks={blocks:,} "
            f"max_iters={args.max_iters:,} tokens/step="
            f"{args.batch_size * args.seq_len * args.grad_accum_steps:,}",
            flush=True,
        )

    initial = evaluate(model, val_loader, engine, token_bytes, args.eval_batches)
    if engine.is_global_zero:
        print(f"starting val_loss={initial['loss']:.4f} val_bpb={initial['bpb']:.4f}", flush=True)

    train_iter = iter(train_loader)
    started = time.perf_counter()
    model.train()

    for step in range(1, args.max_iters + 1):
        engine.synchronize()
        step_start = time.perf_counter()
        lr = scheduler(step)
        for group in optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
            if group.get("kind") == "muon":
                group["momentum"] = momentum(step)
                group["weight_decay"] = decay(step)

        hard_sum = distill_sum = 0.0
        for micro in range(args.grad_accum_steps):
            batch = next(train_iter)
            batch = [engine.to_device(item) for item in batch]
            data, target = batch[0], batch[1]
            sync = engine.no_backward_sync(model, enabled=micro < args.grad_accum_steps - 1)
            with sync:
                with engine.autocast_context():
                    logits = model(data)
                hard, count = hard_label_loss(logits, target)
                tokens = count.clamp_min(1)
                loss = args.distill_alpha * hard / tokens
                hard_sum += float(hard.detach()) / float(tokens)
                if distilling:
                    index, value, logsumexp = batch[2], batch[3], batch[4]
                    soft = distillation_loss(logits, index, value, logsumexp, valid_mask=target != -1)
                    loss = loss + (1.0 - args.distill_alpha) * soft / tokens
                    distill_sum += float(soft.detach()) / float(tokens)
                engine.backward(loss / args.grad_accum_steps)

        if args.grad_clip_norm:
            engine.clip_gradients(model, optimizer, max_norm=args.grad_clip_norm)
        engine.optimizer_step(optimizer)
        optimizer.zero_grad(set_to_none=True)
        apply_qb_update(model)

        engine.synchronize()
        elapsed = time.perf_counter() - step_start
        accum = args.grad_accum_steps
        if engine.is_global_zero:
            print(
                f"iter: {step} | loss: {hard_sum / accum:.4f} | lr: {lr:e} | "
                f"time: {elapsed:.4f}s | tok/s: "
                f"{args.batch_size * args.seq_len * accum / elapsed:,.2f}"
                + (f" | distill: {distill_sum / accum:.4f}" if distilling else ""),
                flush=True,
            )

        if args.eval_every and (step % args.eval_every == 0 or step == args.max_iters):
            metrics = evaluate(model, val_loader, engine, token_bytes, args.eval_batches)
            if engine.is_global_zero:
                print(
                    f"iter: {step} | val_loss: {metrics['loss']:.4f} | "
                    f"val_ppl: {metrics['ppl']:.2f} | val_bpb: {metrics['bpb']:.4f} | "
                    f"val_acc: 0.0000",
                    flush=True,
                )
                if tracker:
                    tracker.log_metrics(
                        {"validation_loss": metrics["loss"], "validation_bpb": metrics["bpb"]},
                        step=step,
                    )

    wall = time.perf_counter() - started
    final = evaluate(model, val_loader, engine, token_bytes, args.eval_batches)
    if engine.is_global_zero:
        print(f"final val_loss={final['loss']:.4f} val_bpb={final['bpb']:.4f} "
              f"train_time_sec={wall:.0f}", flush=True)
        Path(args.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": raw_model.state_dict(), "model_config": asdict(config),
                    "idx": args.max_iters}, args.checkpoint_path)
        if args.result_json:
            Path(args.result_json).write_text(
                json.dumps({"distill_alpha": args.distill_alpha, "vocab_size": vocab_size,
                            "val_loss": final["loss"], "val_bpb": final["bpb"],
                            "max_iters": args.max_iters, "train_time_sec": wall}, indent=2) + "\n",
                encoding="utf-8",
            )
        if tracker:
            tracker.finish()
    engine.close()
    time.sleep(2.0)


if __name__ == "__main__":
    run()
