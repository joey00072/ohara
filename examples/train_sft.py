"""Supervised finetuning of a pretrained Llama into a chat model.

Picks up where ``train_llama_engine.py`` leaves off: loads a pretrained
checkpoint, swaps the raw-text stream for packed conversations with
assistant-only loss masks, and runs a short low-LR pass.

    python examples/train_sft.py --pretrained-checkpoint ./ckpt/base.pt

Following nanochat's ``chat_sft``: weight decay stays at zero (pretraining's
cosine schedule already ramped it there), the learning rate starts at a fraction
of the pretraining rate, and half the run is spent warming down. The model is
small and the mixture is small, so SFT is minutes of work against hours of
pretraining — the point is to teach the conversation format, not new knowledge.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ohara.chat import (
    ASSISTANT_END,
    CHAT_SPECIAL_TOKENS,
    load_chat_tokenizer,
    resize_token_embeddings,
    special_token_ids,
)
from ohara.chat_engine import config_from_state_dict, strip_wrapper_prefixes
from ohara.models.llama import Llama
from ohara.optimizer import build_adamw, build_muon_adamw
from ohara.runtime import (
    EngineConfig,
    OharaEngine,
    ParallelConfig,
    PrecisionConfig,
    PrecisionMode,
)
from ohara.scaling import CosineWeightDecayScheduler, MuonMomentumScheduler, WarmupStableDecayScheduler
from ohara.modules.moe import apply_qb_update
from ohara.sft import ConversationDataset, build_mixture
from ohara.tokenizer import get_token_bytes
from ohara.tracking import BACKENDS as TRACKING_BACKENDS, create_logger
from ohara.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised finetune a pretrained Llama")
    # Model / checkpoints
    parser.add_argument("--pretrained-checkpoint", default="./ckpt/base.pt")
    parser.add_argument("--checkpoint-path", default="./ckpt/sft.pt")
    parser.add_argument("--tokenizer", default="EleutherAI/gpt-neo-125m")
    parser.add_argument("--tokenizer-local-files-only", action="store_true")
    parser.add_argument("--token-bytes-cache", default=None)
    parser.add_argument(
        "--moe-experts-per-tok",
        type=int,
        default=2,
        help="top-k the base was trained with; no tensor shape records it",
    )
    # Data mixture
    parser.add_argument("--smoltalk-limit", type=int, default=None)
    parser.add_argument("--mmlu-epochs", type=int, default=1)
    parser.add_argument("--mmlu-limit", type=int, default=None)
    parser.add_argument("--gsm8k-epochs", type=int, default=1)
    parser.add_argument("--gsm8k-limit", type=int, default=None)
    parser.add_argument("--val-smoltalk-limit", type=int, default=2_000)
    parser.add_argument("--cache-dir", default=None)
    # Shapes
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="max context length (default: inherit from the pretrained checkpoint)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    # Horizon
    parser.add_argument("--max-iters", type=int, default=800)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=200)
    # Optimization
    parser.add_argument("--optimizer", choices=("muon", "adamw"), default="muon")
    parser.add_argument("--matrix-learning-rate", type=float, default=0.02)
    parser.add_argument("--embedding-learning-rate", type=float, default=0.3)
    parser.add_argument("--unembedding-learning-rate", type=float, default=0.004)
    parser.add_argument("--scalar-learning-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--init-lr-frac",
        type=float,
        default=0.8,
        help="start SFT at this fraction of the pretraining learning rate",
    )
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--warmdown-ratio", type=float, default=0.5)
    parser.add_argument("--final-lr-fraction", type=float, default=0.0)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="pretraining decays this to zero; SFT continues from there",
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    # Tracking
    parser.add_argument(
        "--logger",
        choices=TRACKING_BACKENDS,
        default="auto",
        help="auto prefers wandb when a key is set, else falls back to local trackio",
    )
    parser.add_argument("--project", default="ohara-sft")
    parser.add_argument(
        "--run-name",
        default=None,
        # Not "--run": torchrun has --run-path, and argparse rejects the prefix
        # as ambiguous when the script is launched under it.
        help="run name shown in the tracker",
    )
    # Runtime
    parser.add_argument(
        "--precision",
        choices=[mode.value for mode in PrecisionMode],
        default=PrecisionMode.BF16_MIXED.value,
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tp", type=int, default=int(os.environ.get("OHARA_TP", "1")))
    parser.add_argument("--evaluate-bpb", action="store_true")
    parser.add_argument("--result-json", default=None)
    return parser.parse_args()


def load_pretrained(
    checkpoint_path: Path,
    vocab_size: int,
    seq_len: int | None,
    moe_experts_per_tok: int = 2,
):
    """Rebuild the pretrained model from its checkpoint and adapt it for chat.

    The architecture is recovered from tensor shapes by
    :func:`ohara.chat_engine.config_from_state_dict`, so this works against any
    checkpoint the pretraining entrypoint wrote -- dense or mixture-of-experts,
    compiled or not.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"pretrained checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    state = {strip_wrapper_prefixes(key): value for key, value in state.items()}

    config = config_from_state_dict(state, moe_experts_per_tok=moe_experts_per_tok)
    pretrained_vocab = config.vocab_size

    if seq_len is not None:
        if seq_len > config.max_sequence_length:
            raise ValueError(
                f"--seq-len {seq_len} exceeds the pretrained context window "
                f"of {config.max_sequence_length}"
            )
        config.max_sequence_length = seq_len

    model = Llama(config)
    # Rotary buffers are sized from max_sequence_length, which may have shrunk.
    state = {
        key: value
        for key, value in state.items()
        if not key.startswith(("freq_cos", "freq_sin"))
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    ignorable = ("freq_cos", "freq_sin", "qb_beta_sum", "qb_beta_count", "expert_counts")
    unexpected = [key for key in unexpected if not key.endswith(ignorable)]
    missing = [key for key in missing if not key.endswith(ignorable)]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint does not match the reconstructed model: "
            f"missing={missing} unexpected={unexpected}"
        )

    # A base trained with --pad-vocab-to is *wider* than the tokenizer. Those rows
    # are unreachable padding, so keep them rather than trying to shrink the model.
    if pretrained_vocab < vocab_size:
        resize_token_embeddings(model, vocab_size)
    return model, config, pretrained_vocab


def run() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.grad_accum_steps < 1 or args.max_iters < 1:
        raise ValueError("batch-size, grad-accum-steps and max-iters must be at least 1")
    if not 0 < args.init_lr_frac <= 1:
        raise ValueError("init-lr-frac must be in (0, 1]")
    if args.weight_decay < 0 or args.grad_clip_norm < 0:
        raise ValueError("weight-decay and grad-clip-norm cannot be negative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    engine = OharaEngine(
        EngineConfig(
            precision=PrecisionConfig(mode=PrecisionMode(args.precision)),
            parallel=ParallelConfig(tp=args.tp),
        )
    )
    engine.launch()

    tokenizer = load_chat_tokenizer(
        hf_name=args.tokenizer,
        prefer_hf=True,
        local_files_only=args.tokenizer_local_files_only,
    )
    vocab_size = len(tokenizer)

    raw_model, config, pretrained_vocab = load_pretrained(
        Path(args.pretrained_checkpoint),
        vocab_size,
        args.seq_len,
        moe_experts_per_tok=args.moe_experts_per_tok,
    )
    seq_len = config.max_sequence_length
    model = engine.prepare(raw_model)

    if engine.is_global_zero:
        print(
            f"loaded {args.pretrained_checkpoint}: depth={config.num_hidden_layers} "
            f"hidden={config.hidden_size} heads={config.num_attention_heads} "
            f"ctx={seq_len} params={sum(p.numel() for p in raw_model.parameters()):,}"
        )
        if pretrained_vocab != vocab_size:
            print(
                f"grew vocabulary {pretrained_vocab:,} -> {vocab_size:,} for "
                f"{len(CHAT_SPECIAL_TOKENS)} chat special tokens"
            )

    if args.optimizer == "muon":
        if args.tp != 1:
            raise ValueError("the hybrid Muon optimizer currently requires --tp 1")
        optimizer = build_muon_adamw(
            model,
            matrix_learning_rate=args.matrix_learning_rate * args.init_lr_frac,
            embedding_learning_rate=args.embedding_learning_rate * args.init_lr_frac,
            unembedding_learning_rate=args.unembedding_learning_rate * args.init_lr_frac,
            scalar_learning_rate=args.scalar_learning_rate * args.init_lr_frac,
            weight_decay=args.weight_decay,
        )
        scheduler_learning_rate = args.matrix_learning_rate * args.init_lr_frac
    else:
        optimizer = build_adamw(
            model,
            learning_rate=args.learning_rate * args.init_lr_frac,
            weight_decay=args.weight_decay,
        )
        scheduler_learning_rate = args.learning_rate * args.init_lr_frac
    optimizer = engine.prepare_optimizers(optimizer)[0]

    # Rank zero only, so ranks do not open duplicate runs for one job.
    if engine.is_global_zero:
        tracker = create_logger(
            args.logger,
            project=args.project,
            run_name=args.run_name,
            config={
                **vars(args),
                "vocab_size": vocab_size,
                "parameters": sum(p.numel() for p in raw_model.parameters()),
                "depth": config.num_hidden_layers,
                "world_size": engine.data_parallel_world_size,
            },
        )
        engine.loggers = [tracker]
    else:
        tracker = None

    if engine.is_global_zero:
        print("building SFT mixture (this downloads SmolTalk / MMLU / GSM8K on first run)")
    train_conversations = build_mixture(
        split="train",
        smoltalk_limit=args.smoltalk_limit,
        mmlu_epochs=args.mmlu_epochs,
        mmlu_limit=args.mmlu_limit,
        gsm8k_epochs=args.gsm8k_epochs,
        gsm8k_limit=args.gsm8k_limit,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    val_conversations = build_mixture(
        split="val",
        smoltalk_limit=args.val_smoltalk_limit,
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    if engine.is_global_zero:
        print(
            f"mixture: {len(train_conversations):,} train / "
            f"{len(val_conversations):,} val conversations"
        )

    dataset_args = {
        "tokenizer": tokenizer,
        "max_length": seq_len,
        "buffer_size": args.buffer_size,
        "data_rank": engine.data_parallel_rank,
        "data_world_size": engine.data_parallel_world_size,
    }
    train_ds = ConversationDataset(
        train_conversations, shuffle=True, seed=args.seed, **dataset_args
    )
    val_ds = ConversationDataset(
        val_conversations, shuffle=False, seed=args.seed + 1, **dataset_args
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": engine.device.type == "cuda",
    }
    train_dl = DataLoader(train_ds, **loader_args)
    val_dl = DataLoader(val_ds, **loader_args)
    train_dl, val_dl = engine.prepare_dataloaders(train_dl, val_dl)

    scheduler = WarmupStableDecayScheduler(
        learning_rate=scheduler_learning_rate,
        max_iters=args.max_iters,
        warmup_iters=args.warmup_iters,
        warmdown_ratio=args.warmdown_ratio,
        final_lr_fraction=args.final_lr_fraction,
    )
    optimizer_hparams = None
    if args.optimizer == "muon":
        momentum_scheduler = MuonMomentumScheduler(
            max_iters=args.max_iters,
            warmdown_ratio=args.warmdown_ratio,
            warmup_iters=0,
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
        get_token_bytes(tokenizer, device=engine.device, cache_path=args.token_bytes_cache)
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
        print_every=10,
        eval_val_batches=args.eval_batches,
        grad_clip_norm=args.grad_clip_norm or None,
        checkpoint_path=args.checkpoint_path,
        token_bytes=token_bytes,
        apply_router_balancing=apply_qb_update if config.moe_num_experts > 0 else None,
    )

    initial = trainer.evaluate(trainer.val_dataloader, args.eval_batches)
    if engine.is_global_zero:
        print(
            f"before SFT: val_loss={initial['loss']:.4f} val_ppl={initial['ppl']:.2f} "
            f"val_acc={initial['accuracy']:.4f}"
        )

    started = time.perf_counter()
    trainer.train()
    wall_time = time.perf_counter() - started
    final = trainer.evaluate(trainer.val_dataloader, args.eval_batches)

    if engine.is_global_zero:
        print(
            f"after SFT: val_loss={final['loss']:.4f} val_ppl={final['ppl']:.2f} "
            f"val_acc={final['accuracy']:.4f} time={wall_time / 60:.1f}m"
        )

    # The chat serving path needs the tokenizer that produced these ids, so save
    # it next to the checkpoint rather than rebuilding it from flags later.
    if engine.is_global_zero:
        tokenizer_dir = Path(args.checkpoint_path).with_suffix("")
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_dir)
        metadata = {
            "vocab_size": vocab_size,
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "max_sequence_length": config.max_sequence_length,
            "weight_tying": config.weight_tying,
            "tokenizer": args.tokenizer,
            "assistant_end_id": special_token_ids(tokenizer)[ASSISTANT_END],
            "initial_val_loss": initial["loss"],
            "val_loss": final["loss"],
            "val_accuracy": final["accuracy"],
            "max_iters": args.max_iters,
            "train_time_sec": wall_time,
        }
        (tokenizer_dir / "ohara_chat.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"saved chat tokenizer and metadata to {tokenizer_dir}")

        if args.result_json is not None:
            result_path = Path(args.result_json)
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if tracker is not None:
        tracker.finish()
    engine.close()
    time.sleep(2.0)


if __name__ == "__main__":
    run()
