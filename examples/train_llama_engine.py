from __future__ import annotations

import math
import os
import time

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from ohara.dataset import PreTokenizedDataset
from ohara.lr_scheduler import CosineScheduler
from ohara.models.llama import Config, Llama
from ohara.runtime import (
    EngineConfig,
    OharaEngine,
    ParallelConfig,
    PrecisionConfig,
    PrecisionMode,
    TensorParallelPlan,
)
from ohara.tokenizer import get_tokenizer
from ohara.utils import BetterCycle


def run() -> None:
    tokenizer = get_tokenizer(hf_name="microsoft/phi-2", prefer_hf=True)

    model_cfg = Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=256,
        max_sequence_length=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        multiple_of=4,
    )
    model = Llama(model_cfg)

    train_ds = PreTokenizedDataset(
        dataset_name="roneneldan/TinyStories",
        tokenizer=tokenizer,
        split="train",
        max_length=256,
    )
    val_ds = PreTokenizedDataset(
        dataset_name="roneneldan/TinyStories",
        tokenizer=tokenizer,
        split="validation",
        max_length=256,
    )

    train_dl = DataLoader(train_ds, batch_size=8)
    val_dl = DataLoader(val_ds, batch_size=8)

    optimizer = optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = CosineScheduler(
        learning_rate=5e-4,
        min_lr=0.0,
        warmup_iters=100,
        max_iters=10000,
    )

    tp_degree = int(os.environ.get("OHARA_TP", "1"))
    engine = OharaEngine(
        EngineConfig(
            precision=PrecisionConfig(mode=PrecisionMode.BF16_MIXED),
            parallel=ParallelConfig(tp=tp_degree),
            tensor_parallel=TensorParallelPlan.llama_default(degree=tp_degree),
        )
    )
    train_dl, val_dl = engine.prepare_dataloaders(train_dl, val_dl)
    model, optimizer = engine.prepare(model, optimizer)

    train_cycle = BetterCycle(iter(train_dl))
    val_cycle = BetterCycle(iter(val_dl))

    ignore_index = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
    micro_batch = 4
    max_iters = 10000

    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, max_iters + 1):
        t0 = time.perf_counter()
        lr = scheduler(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        step_loss = 0.0
        for micro_idx in range(micro_batch):
            x, y = next(train_cycle)
            with engine.no_backward_sync(model, enabled=micro_idx < micro_batch - 1):
                logits: torch.Tensor = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=ignore_index,
                )
                step_loss += float(loss.detach().item())
                engine.backward(loss / micro_batch)

        engine.optimizer_step(optimizer)
        optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0 and engine.is_global_zero:
            dt = time.perf_counter() - t0
            print(f"step={step} loss={step_loss / micro_batch:.4f} lr={lr:.3e} time={dt:.3f}s")

        if step % 200 == 0:
            model.eval()
            with torch.no_grad():
                x, y = next(val_cycle)
                logits = model(x)
                val_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=ignore_index,
                )
                val_loss_t = val_loss.detach().to(engine.device)
                val_loss_t = engine.all_reduce(val_loss_t)
                if engine.is_global_zero:
                    ppl = math.exp(min(float(val_loss_t.item()), 20.0))
                    print(f"eval step={step} loss={val_loss_t.item():.4f} ppl={ppl:.2f}")
            model.train()

        if step % 1000 == 0:
            engine.save(
                "./ckpt/model_engine.pt",
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                },
            )


if __name__ == "__main__":
    run()
