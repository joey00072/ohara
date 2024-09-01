import sys
import time
import math
from datetime import datetime
import os

from typing import Any, Dict
from dataclasses import dataclass, field


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from ohara.lr_scheduler import CosineScheduler
from ohara.dataset import PreTokenizedDataset
from ohara.utils import BetterCycle

from ohara.utils import auto_accelerator, random_name, model_summary


from torch.utils.data import DataLoader
from transformers import AutoTokenizer


from rich import print, traceback

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.loggers import TensorBoardLogger

from model import ModelingLM, Config
from monkey_patcher import dm_linear_monkey_patch


traceback.install()

torch.set_float32_matmul_precision("high")

# ================================================================================================

### CONFIGS
wandb_project_name = "Ohara-MODEL"
wandb_run_name = random_name()

learning_rate: float = 5e-4
min_learning_rate: float = 0.0

warmup_iters: int = 1000
max_iters: int = 10_000
batch_size: int = 8
micro_batch: int = 4
eval_iters: int = 100
save_ckpt_iters: int = 2000

multiple_of: int = 4
d_model: int = 1024 // 4
hidden_dim = int(d_model * multiple_of)
seq_len: int = 256
num_layers: int = 32 // 3  # 44
num_heads: int = 32

MONKEY_PATCH = False
model_name = f"joey00072/model_name{'Baseline' if MONKEY_PATCH is None else str(MONKEY_PATCH)}"

assert d_model % num_heads == 0

compile_model = not bool(sys.platform == "darwin")

### Dataset and Tokenizer
dataset_name = "joey00072/pretokenized__JeanKaddour_minipile__EleutherAI__gpt-neo-125m"  # run pretokeinze first
tokenizer_name = "EleutherAI/gpt-neo-125m"

### Setup
device = auto_accelerator()  # auto chose device (cuda, mps)

# for restarting training from last checkpoint
resume_training = False
checkpoint_path = "./ckpt/model.pt"


# ================================================================================================


@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: nn.Module,
    dataloader: DataLoader,
    max_iters: int,
    ignore_index: int = -1,
    device=None,
) -> Tensor:
    fabric.barrier()
    model.eval()
    with torch.no_grad():
        losses = torch.zeros(max_iters, device=device)
        for idx, (data, target) in enumerate(dataloader):
            if idx >= max_iters:
                break
            logits: torch.Tensor = model(data)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=ignore_index,
            )
            losses[idx] = loss.item()
        val_loss = losses.mean()

    model.train()
    fabric.barrier()
    return val_loss


def train(
    fabric: L.Fabric,
    model: nn.Module,
    optimizer: optim.Optimizer,
    micro_batch: int,
    train_dataloader,
    val_dataloader,
    eval_iters: int,
    save_ckpt_iters: int,
    get_lr,
    ignore_index=-1,
    start_iter: int = 0,
):
    fabric.launch()
    ignore_index = ignore_index if ignore_index else -1
    # sanity test
    validate(fabric, model, val_dataloader, 5, device=device)
    # cycling loader so you can run it indefinitely
    train_dataloader = BetterCycle(iter(train_dataloader))
    val_dataloader = BetterCycle(iter(val_dataloader))
    (data, target) = next(val_dataloader)
    tokens_per_iter = int(math.prod(data.shape) * micro_batch)

    micro_batch_loss: float = 0
    idx: int = start_iter
    while True:
        micro_batch_loss = 0
        if idx >= max_iters:
            break
        idx += 1
        start_time: float = time.perf_counter()

        lr = get_lr(idx)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # ...
        for _ in range(micro_batch):
            (data, target) = next(train_dataloader)
            with fabric.no_backward_sync(model, enabled=micro_batch > 1):
                logits: torch.Tensor = model(data)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1),
                    ignore_index=ignore_index,
                )
                loss = loss / micro_batch
                micro_batch_loss += loss.item()
                fabric.backward(loss)

        optimizer.step()
        optimizer.zero_grad()

        curr_time: float = time.perf_counter()
        elapsed_time: float = curr_time - start_time
        print(
            f"iter: {idx} | loss: {micro_batch_loss:.4f} | lr: {lr:e} | time: {elapsed_time:.4f}s"
        )

        if idx % eval_iters == 0:
            val_loss = validate(fabric, model, val_dataloader, 100, device=device)
            try:
                fabric.log_dict(
                    {
                        "training_loss": micro_batch_loss,
                        "test_loss": val_loss,
                        "iter": idx,
                        "tokens": idx * tokens_per_iter,
                        "lr": lr,
                        "time": elapsed_time,
                    },
                    step=idx,
                )
            except Exception as e:
                print(f"Error logging: {e}")

        if idx % save_ckpt_iters == 0:
            state = {"model": model, "optimizer": optimizer, "idx": idx, "lr": lr}
            fabric.save("./ckpt/model.pt", state)
            model.config.ckpt_iter = idx
            model.push_to_hub(model_name, commit_message=f"checkpoint iter: {idx}")


def main():
    hyper_params = {
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "max_iters": max_iters,
        "eval_iters": eval_iters,
        "batch_size": batch_size,
        "micro_batch": micro_batch,
        "d_model": d_model,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "multiple_of": multiple_of,
        "compile_model": compile_model,
        "tokenizer_name": tokenizer_name,
        "dataset_name": dataset_name,
        "resume_training": resume_training,
        "save_ckpt_iters": save_ckpt_iters,
    }
    # fabric init
    logger = WandbLogger(project=wandb_project_name, resume=resume_training)
    # logger = TensorBoardLogger(root_dir="./logs", name=wandb_project_name)
    fabric = L.Fabric(loggers=[logger], precision="bf16-mixed")
    fabric.logger.log_hyperparams(hyper_params)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    config = Config(
        vocab_size=tokenizer.vocab_size,
        seq_len=seq_len,
        d_model=d_model,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=0,
        num_layers=num_layers,
        dropout=0.2,
        bias=False,
        weight_tying=True,
        activation="relu_squared",
        mlp="GLU",
    )

    train_ds = PreTokenizedDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split="train",
        hf=True,
        max_length=seq_len,
    )
    test_ds = PreTokenizedDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split="validation",
        hf=True,
        max_length=seq_len,
    )

    train_dataloader = DataLoader(train_ds, batch_size=batch_size)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)
    train_dataloader, test_dataloader = fabric.setup_dataloaders(train_dataloader, test_dataloader)

    model = ModelingLM(config)

    if MONKEY_PATCH:
        target_layers = ["key", "value", "query", "proj", "w1", "w2", "w3"]
        dm_linear_monkey_patch(model, target_layers)
        print("\n>>>> Applied monkey patching <<<<\n")

    model: L.LightningModule = fabric.setup(model)
    print("=" * 100)
    if compile_model:
        model = torch.compile(model)
    # model.gradient_checkpointing_enable()
    print(model)
    print(model_summary(model))
    import time

    get_lr = CosineScheduler(
        learning_rate=learning_rate,
        min_lr=min_learning_rate,
        warmup_iters=warmup_iters,
        max_iters=max_iters,
    )

    # inputs = torch.tensor(tokenizer.encode("The")).unsqueeze(0).clone().detach()
    optimizer = optim.AdamW(model.parameters(), lr=get_lr(0))
    optimizer = fabric.setup_optimizers(optimizer)

    start_iter = 0
    if resume_training and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = fabric.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['idx']
        print(f"Resuming from iteration: {start_iter}")

    # Let's GO!!
    train(
        fabric,
        model,
        optimizer,
        micro_batch=micro_batch,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        eval_iters=eval_iters,
        save_ckpt_iters=save_ckpt_iters,
        get_lr=get_lr,
        ignore_index=tokenizer.pad_token_id,
        start_iter=start_iter,
    )

    # TODO: Replace this inference
    start: float = time.time()
    max_tokens = 200
    model = model.eval()
    inputs = tokenizer.encode("Once")
    with torch.no_grad():
        inputs = torch.tensor(inputs).reshape(1, -1).to(device).clone().detach()
        for _idx in range(max_tokens):
            logits = model(inputs)
            max_logits = torch.argmax(logits, dim=-1)
            inputs: torch.Tensor = torch.cat((inputs, max_logits[:, -1:]), dim=-1)
            inputs.shape[1] - 1
            print(tokenizer.decode(inputs.tolist()[0][-1]), end="", flush=True)

    end: float = time.time()

    print(f"Time: {end-start}")


if __name__ == "__main__":
    main()
