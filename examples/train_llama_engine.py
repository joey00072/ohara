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

from ohara.chat import add_chat_tokens
from ohara.dataset import StreamingTextDataset
from ohara.lr_scheduler import CosineScheduler
from ohara.tokenbin import TokenBinDataset
from ohara.tracking import BACKENDS as TRACKING_BACKENDS, create_logger
from ohara.models.llama import Config, Llama
from ohara.modules.moe import apply_qb_update
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
    parser.add_argument(
        "--chat-tokens",
        action="store_true",
        help=(
            "reserve the conversation special tokens in the vocabulary now, so a later "
            "SFT pass does not have to grow the embedding matrix (see ohara.chat)"
        ),
    )
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
    # Mixture of experts. 0 keeps every feed-forward dense.
    parser.add_argument("--moe-num-experts", type=int, default=0)
    parser.add_argument("--moe-experts-per-tok", type=int, default=2)
    parser.add_argument(
        "--moe-layer-interval",
        type=int,
        default=1,
        help="make every Nth layer an MoE and leave the rest dense (1 = all layers)",
    )
    parser.add_argument("--moe-gate-fn", choices=("softmax", "sigmoid"), default="softmax")
    parser.add_argument(
        "--moe-grouped",
        action="store_true",
        help="dispatch experts with grouped matmuls; required above ~32 experts",
    )
    parser.add_argument(
        "--moe-num-shared-experts",
        type=int,
        default=0,
        help="always-active experts per layer (DeepSeek-style shared expert isolation)",
    )
    parser.add_argument(
        "--moe-no-normalize-weights",
        action="store_true",
        help="do not rescale routed weights to sum to 1 (grouped sigmoid gating only)",
    )
    parser.add_argument(
        "--moe-no-quantile-balancing",
        action="store_true",
        help="disable closed-form router balancing (it needs no aux loss, so keep it on)",
    )
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
    parser.add_argument(
        "--logger",
        choices=TRACKING_BACKENDS,
        default="auto",
        help="auto prefers wandb when a key is set, else falls back to local trackio",
    )
    parser.add_argument("--project", default="ohara")
    parser.add_argument(
        "--run-name",
        default=None,
        # Not "--run": torchrun has --run-path, and argparse rejects the prefix
        # as ambiguous when the script is launched under it.
        help="run name shown in the tracker",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model; costs a warmup but is the single largest MFU win",
    )
    parser.add_argument(
        "--no-token-bins",
        action="store_true",
        help="ignore pre-tokenized {split}.bin files and tokenize text inline",
    )
    parser.add_argument(
        "--pad-vocab-to",
        type=int,
        default=1,
        help=(
            "round the model vocabulary up to a multiple of this for tensor-core "
            "friendly matmuls (nanochat uses 64). Padded rows are never valid targets"
        ),
    )
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
    if args.chat_tokens:
        add_chat_tokens(tokenizer)
    tokenizer_vocab_size = len(tokenizer)
    if args.pad_vocab_to < 1:
        raise ValueError("pad-vocab-to must be at least 1")
    # An odd vocabulary (gpt-neo's 50257, or 50265 with chat tokens) makes the
    # embedding and output matmuls fall off the tensor-core fast path. Padding up
    # only adds rows that no target ever selects, so the loss is unchanged.
    vocab_size = (
        (tokenizer_vocab_size + args.pad_vocab_to - 1) // args.pad_vocab_to
    ) * args.pad_vocab_to
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
        moe_num_experts=args.moe_num_experts,
        moe_experts_per_tok=args.moe_experts_per_tok,
        moe_layer_interval=args.moe_layer_interval,
        moe_gate_fn=args.moe_gate_fn,
        moe_quantile_balancing=not args.moe_no_quantile_balancing,
        moe_grouped=args.moe_grouped,
        moe_num_shared_experts=args.moe_num_shared_experts,
        moe_normalize_weights=not args.moe_no_normalize_weights,
    )
    raw_model = Llama(model_cfg)
    model = raw_model
    if args.compile:
        # Batch shapes are fixed for the whole run, so dynamic=False lets inductor
        # specialize. Compile before the engine wraps the module for DDP.
        model = torch.compile(model, dynamic=False)
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

    # Only rank zero opens a tracking run; the other ranks would create duplicate
    # runs for the same job. engine.log_dict already forwards from rank zero only.
    if engine.is_global_zero:
        tracker = create_logger(
            args.logger,
            project=args.project,
            run_name=args.run_name,
            config={
                **vars(args),
                "vocab_size": vocab_size,
                "parameters": sum(p.numel() for p in raw_model.parameters()),
                "world_size": engine.data_parallel_world_size,
            },
        )
        engine.loggers = [tracker]
    else:
        tracker = None

    # Prefer pre-tokenized bins when they exist beside the corpus: tokenizing in
    # the training loop leaves the GPUs waiting on the CPU, and re-tokenizes the
    # same documents every epoch. See examples/pretokenize_corpus.py.
    corpus_dir = Path(args.dataset)
    train_bin = corpus_dir / f"{args.train_split}.bin"
    val_bin = corpus_dir / f"{args.validation_split}.bin"
    use_token_bins = not args.no_token_bins and train_bin.exists() and val_bin.exists()

    if use_token_bins:
        bin_args = {
            "max_length": args.seq_len,
            "tokenizer": tokenizer,
            "data_rank": engine.data_parallel_rank,
            "data_world_size": engine.data_parallel_world_size,
        }
        train_ds = TokenBinDataset(train_bin, shuffle=True, seed=args.seed, **bin_args)
        val_ds = TokenBinDataset(val_bin, shuffle=False, seed=args.seed, **bin_args)
        if engine.is_global_zero:
            print(
                f"token bins: train {train_ds.metadata['tokens']:,} tokens, "
                f"val {val_ds.metadata['tokens']:,} tokens"
            )
    else:
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

    token_bytes = None
    if args.evaluate_bpb:
        token_bytes = get_token_bytes(
            tokenizer,
            device=engine.device,
            cache_path=args.token_bytes_cache,
        )
        if token_bytes.numel() < vocab_size:
            # --pad-vocab-to widened the model past the tokenizer. The padded ids
            # decode to nothing and are never targets, so they carry zero bytes and
            # cannot affect bits-per-byte; they just have to exist for the lookup.
            token_bytes = torch.nn.functional.pad(
                token_bytes, (0, vocab_size - token_bytes.numel())
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
        apply_router_balancing=(
            apply_qb_update
            if args.moe_num_experts > 0 and not args.moe_no_quantile_balancing
            else None
        ),
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
        vocab_note = (
            f" vocab={tokenizer_vocab_size:,}->{vocab_size:,}"
            if vocab_size != tokenizer_vocab_size
            else ""
        )
        print(
            f"device={engine.device} precision={args.precision} params={parameter_count:,} "
            f"tokens/step/rank={tokens_per_step:,} max_iters={args.max_iters:,}{vocab_note} "
            # Padded rows are never targets, so chance level is set by the real vocabulary.
            f"random_loss={math.log(tokenizer_vocab_size):.4f}"
            f"{' compiled' if args.compile else ''}"
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
    if tracker is not None:
        tracker.finish()
    engine.close()
    # Give remote-stream cleanup callbacks a chance to finish before CPython
    # tears down extension-module thread states. This applies regardless of the
    # accelerator: the readers and tokenizer workers live on the CPU.
    time.sleep(5.0)


if __name__ == "__main__":
    run()
