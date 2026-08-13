from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ohara.dataset import StreamingTextDataset
from ohara.lr_scheduler import CosineScheduler
from ohara.models.llama import Config, Llama
from ohara.optimizer import build_adamh, build_adamw, build_muon_adamw, build_muonh_adamh
from ohara.scaling import (
    CosineWeightDecayScheduler,
    MuonMomentumScheduler,
    WarmupStableDecayScheduler,
)
from ohara.runtime import (
    EngineConfig,
    OharaEngine,
    ParallelConfig,
    PrecisionConfig,
    PrecisionMode,
    TensorParallelPlan,
)
from ohara.tokenizer import get_token_bytes, get_tokenizer
from ohara.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small Llama on streamed text")
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neo-125m")
    parser.add_argument("--tokenizer-local-files-only", action="store_true")
    parser.add_argument("--token-bytes-cache", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--checkpoint-path", default="./ckpt/model.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--optimizer",
        choices=("adamw", "muon", "adamh", "muonh"),
        default="adamw",
        help="adamh/muonh are the constant-norm variants; they ignore --weight-decay "
        "and read --hypersphere-learning-rate instead",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--matrix-learning-rate", type=float, default=0.02)
    parser.add_argument("--embedding-learning-rate", type=float, default=0.3)
    parser.add_argument("--unembedding-learning-rate", type=float, default=0.008)
    parser.add_argument("--scalar-learning-rate", type=float, default=0.5)
    parser.add_argument(
        "--hypersphere-learning-rate",
        type=float,
        default=None,
        help="relative step size for adamh/muonh; defaults to sqrt(lr * weight_decay) "
        "of the matching additive recipe",
    )
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-momentum-warmup-iters", type=int, default=400)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--intermediate-size", type=int, default=1_024)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--init-style", choices=("standard", "nanochat"), default="standard"
    )
    parser.add_argument(
        "--weight-tying",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("cosine", "wsd"),
        default="cosine",
        help="cosine or nanochat-style warmup/stable/warmdown",
    )
    parser.add_argument("--warmdown-ratio", type=float, default=0.65)
    parser.add_argument("--final-lr-fraction", type=float, default=0.05)
    parser.add_argument("--evaluate-bpb", action="store_true")
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--scaling-depth", type=int, default=None)
    parser.add_argument("--flops-budget", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp", type=int, default=int(os.environ.get("OHARA_TP", "1")))
    parser.add_argument(
        "--precision",
        choices=[mode.value for mode in PrecisionMode],
        default=PrecisionMode.BF16_MIXED.value,
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    if args.hidden_size % args.num_heads != 0:
        raise ValueError("hidden-size must be divisible by num-heads")
    if args.grad_accum_steps < 1:
        raise ValueError("grad-accum-steps must be at least 1")
    if args.batch_size < 1 or args.max_iters < 1:
        raise ValueError("batch-size and max-iters must be at least 1")
    if args.eval_every < 0 or args.eval_batches < 1 or args.save_every < 0:
        raise ValueError("invalid evaluation or checkpoint interval")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    optimizer_lrs = (
        args.learning_rate,
        args.matrix_learning_rate,
        args.embedding_learning_rate,
        args.unembedding_learning_rate,
        args.scalar_learning_rate,
    )
    if any(value <= 0 for value in optimizer_lrs) or args.min_lr < 0 or args.weight_decay < 0:
        raise ValueError(
            "learning rates must be positive; min-lr and weight-decay cannot be negative"
        )
    if args.muon_momentum_warmup_iters < 0 or args.grad_clip_norm < 0:
        raise ValueError("optimizer warmup and gradient clipping cannot be negative")
    if not 0 <= args.warmdown_ratio <= 1:
        raise ValueError("warmdown-ratio must be in [0, 1]")
    if not 0 <= args.final_lr_fraction <= 1:
        raise ValueError("final-lr-fraction must be in [0, 1]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    engine = OharaEngine(
        EngineConfig(
            precision=PrecisionConfig(mode=PrecisionMode(args.precision)),
            parallel=ParallelConfig(tp=args.tp),
            tensor_parallel=TensorParallelPlan.llama_default(degree=args.tp),
        )
    )
    engine.launch()

    tokenizer = get_tokenizer(
        hf_name=args.tokenizer,
        prefer_hf=True,
        local_files_only=args.tokenizer_local_files_only,
    )
    vocab_size = len(tokenizer)
    model_cfg = Config(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        max_sequence_length=args.seq_len,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        dropout=args.dropout,
        multiple_of=4,
        weight_tying=args.weight_tying,
        init_style=args.init_style,
    )
    raw_model = Llama(model_cfg)
    model = raw_model
    model = engine.prepare(model)
    if args.optimizer in ("muon", "muonh") and args.tp != 1:
        raise ValueError("the hybrid Muon optimizers currently require --tp 1")

    if args.optimizer == "muon":
        optimizer = build_muon_adamw(
            model,
            matrix_learning_rate=args.matrix_learning_rate,
            embedding_learning_rate=args.embedding_learning_rate,
            unembedding_learning_rate=args.unembedding_learning_rate,
            scalar_learning_rate=args.scalar_learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler_learning_rate = args.matrix_learning_rate
    elif args.optimizer in ("adamh", "muonh"):
        # On a sphere the learning rate is a relative step size, and weight decay
        # has no first-order effect. sqrt(lr * wd) carries an additive recipe over.
        additive_lr = (
            args.matrix_learning_rate if args.optimizer == "muonh" else args.learning_rate
        )
        scheduler_learning_rate = args.hypersphere_learning_rate
        if scheduler_learning_rate is None:
            scheduler_learning_rate = math.sqrt(additive_lr * args.weight_decay)
            if scheduler_learning_rate <= 0:
                raise ValueError(
                    "cannot derive a hyperspherical learning rate from --weight-decay 0; "
                    "pass --hypersphere-learning-rate explicitly"
                )
        builder = build_muonh_adamh if args.optimizer == "muonh" else build_adamh
        optimizer = builder(
            model,
            learning_rate=scheduler_learning_rate,
            adam_learning_rate=args.embedding_learning_rate
            if args.optimizer == "muonh"
            else args.learning_rate,
        )
    else:
        optimizer = build_adamw(
            model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler_learning_rate = args.learning_rate
    optimizer = engine.prepare_optimizers(optimizer)[0]

    common_dataset_args = {
        "dataset_name": args.dataset,
        "name": args.dataset_config,
        "tokenizer": tokenizer,
        "max_length": args.seq_len,
        "text_column": args.text_column,
        "data_rank": engine.data_parallel_rank,
        "data_world_size": engine.data_parallel_world_size,
    }
    train_ds = StreamingTextDataset(
        **common_dataset_args,
        split=args.train_split,
        shuffle=True,
        seed=args.seed,
    )
    val_ds = StreamingTextDataset(
        **common_dataset_args,
        split=args.validation_split,
        shuffle=False,
        seed=args.seed,
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": engine.device.type == "cuda",
    }
    train_dl = DataLoader(
        train_ds,
        **loader_args,
        generator=torch.Generator().manual_seed(args.seed),
    )
    val_dl = DataLoader(
        val_ds,
        **loader_args,
        generator=torch.Generator().manual_seed(args.seed + 1),
    )
    train_dl, val_dl = engine.prepare_dataloaders(train_dl, val_dl)

    if args.lr_schedule == "wsd":
        scheduler = WarmupStableDecayScheduler(
            learning_rate=scheduler_learning_rate,
            max_iters=args.max_iters,
            warmup_iters=args.warmup_iters,
            warmdown_ratio=args.warmdown_ratio,
            final_lr_fraction=args.final_lr_fraction,
        )
    else:
        scheduler = CosineScheduler(
            learning_rate=scheduler_learning_rate,
            min_lr=args.min_lr,
            warmup_iters=args.warmup_iters,
            max_iters=args.max_iters,
        )
    optimizer_hparams = None
    if args.optimizer == "muon":
        momentum_scheduler = MuonMomentumScheduler(
            max_iters=args.max_iters,
            warmdown_ratio=args.warmdown_ratio,
            warmup_iters=args.muon_momentum_warmup_iters,
        )
        weight_decay_scheduler = CosineWeightDecayScheduler(
            weight_decay=args.weight_decay,
            max_iters=args.max_iters,
        )

        def optimizer_hparams(iteration: int) -> dict[str, float]:
            return {
                "momentum": momentum_scheduler(iteration),
                "weight_decay": weight_decay_scheduler(iteration),
            }

    token_bytes = (
        get_token_bytes(
            tokenizer,
            device=engine.device,
            cache_path=args.token_bytes_cache,
        )
        if args.evaluate_bpb
        else None
    )
    trainer = Trainer(
        engine=engine,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        get_lr=scheduler,
        get_optimizer_hparams=optimizer_hparams,
        micro_batch=args.grad_accum_steps,
        max_iters=args.max_iters,
        eval_iters=args.eval_every,
        save_ckpt_iters=args.save_every,
        ignore_index=-1,
        print_every=1,
        eval_val_batches=args.eval_batches,
        eval_train_batches=0,
        grad_clip_norm=args.grad_clip_norm or None,
        timing_warmup_steps=10,
        checkpoint_path=args.checkpoint_path,
        token_bytes=token_bytes,
    )

    start_iter = 0
    if args.resume:
        checkpoint = engine.load(
            args.checkpoint_path,
            {"model": model, "optimizer": optimizer},
        )
        start_iter = int(checkpoint["idx"])
        saved_accumulation = int(
            checkpoint.get("gradient_accumulation_steps", args.grad_accum_steps)
        )
        if saved_accumulation != args.grad_accum_steps:
            raise ValueError(
                "checkpoint gradient accumulation does not match --grad-accum-steps"
            )
        batches_consumed = int(
            checkpoint.get(
                "train_batches_consumed",
                start_iter * args.grad_accum_steps,
            )
        )
        if batches_consumed > 0 and args.num_workers != 0:
            raise ValueError("exact --resume currently requires --num-workers 0")
        train_ds.start_block = batches_consumed * args.batch_size
        trainer.train_batches_consumed = batches_consumed
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        if engine.is_global_zero:
            print(
                f"resumed checkpoint={args.checkpoint_path} at iter={start_iter}; "
                f"skipping {train_ds.start_block:,} previously consumed blocks"
            )

    parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum_steps
    initial = trainer.evaluate(trainer.val_dataloader, args.eval_batches)
    if engine.is_global_zero:
        print(
            f"device={engine.device} precision={args.precision} params={parameter_count:,} "
            f"tokens/step/rank={tokens_per_step:,} random_loss={math.log(vocab_size):.4f}"
        )
        print(
            f"starting val_loss={initial['loss']:.4f} val_ppl={initial['ppl']:.2f} "
            f"val_bits/token={initial['bits_per_token']:.4f}"
        )

    training_started = time.perf_counter()
    trainer.train(start_iter=start_iter)
    training_wall_time = time.perf_counter() - training_started
    final = trainer.evaluate(trainer.val_dataloader, args.eval_batches)

    if engine.is_global_zero:
        final_message = (
            f"final val_loss={final['loss']:.4f} val_ppl={final['ppl']:.2f} "
            f"val_bits/token={final['bits_per_token']:.4f}"
        )
        if "bpb" in final:
            final_message += f" val_bpb={final['bpb']:.4f}"
        print(final_message)

    if args.result_json is not None and engine.is_global_zero:
        counts = raw_model.num_scaling_params()
        flops_per_token = raw_model.estimate_flops(args.seq_len)
        total_batch_size = (
            args.batch_size
            * args.seq_len
            * args.grad_accum_steps
            * engine.data_parallel_world_size
        )
        tokens_trained = args.max_iters * total_batch_size
        result = {
            "flops_budget": (
                args.flops_budget
                if args.flops_budget is not None
                else flops_per_token * tokens_trained
            ),
            "depth": (
                args.scaling_depth
                if args.scaling_depth is not None
                else args.num_layers
            ),
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "num_heads": args.num_heads,
            "sequence_length": args.seq_len,
            "params_token_embeddings": counts["token_embeddings"],
            "params_lm_head": counts["lm_head"],
            "params_transformer": counts["transformer_matrices"],
            "params_norms_and_scalars": counts["norms_and_scalars"],
            "params_total": counts["total"],
            "params_effective": counts["effective"],
            "flops_per_token": flops_per_token,
            "total_batch_size": total_batch_size,
            "num_iterations": args.max_iters,
            "tokens_trained": tokens_trained,
            "actual_training_flops": flops_per_token * tokens_trained,
            "tokens_per_effective_param": tokens_trained / counts["effective"],
            "optimizer": args.optimizer,
            "optimizer_muon": float(args.optimizer == "muon"),
            "initialization_nanochat": float(args.init_style == "nanochat"),
            "learning_rate": scheduler_learning_rate,
            "matrix_learning_rate": args.matrix_learning_rate,
            "embedding_learning_rate": args.embedding_learning_rate,
            "unembedding_learning_rate": args.unembedding_learning_rate,
            "scalar_learning_rate": args.scalar_learning_rate,
            "weight_decay": args.weight_decay,
            "initial_val_loss": initial["loss"],
            "initial_val_bpb": initial.get("bpb", float("nan")),
            "val_loss": final["loss"],
            "val_bpb": final.get("bpb", float("nan")),
            "val_bits_per_token": final["bits_per_token"],
            "train_time_sec": training_wall_time,
        }
        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_name(f".{result_path.name}.tmp-{os.getpid()}")
        try:
            temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, result_path)
        finally:
            temporary.unlink(missing_ok=True)
    engine.close()
    # Give remote-stream cleanup callbacks a chance to finish before CPython
    # tears down extension-module thread states. This applies regardless of the
    # accelerator: the readers and tokenizer workers live on the CPU.
    time.sleep(5.0)


if __name__ == "__main__":
    run()
