import sys
import time
import math
from datetime import datetime

from typing import Any, Dict
from dataclasses import dataclass, field


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from ohara.models.llama import LLAMA, Config
from ohara.lr_scheduler import CosineScheduler
from ohara.dataset import PreTokenizedDataset

from ohara.utils import auto_accelerator, random_name, model_summary


from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import wandb
from rich import print, traceback


traceback.install()


### CONFIGS
wandb_project_name = "Ohara_LLAMA"
wandb_run_name = "xoxoxox"  # random_name()

learning_rate: float = 5e-4
min_learning_rate: float = 0.0

wornup_iters: int = 1000
max_iters: int = 100_000
batch_size: int = 32
micro_batch: int = 4
eval_iters: int = 100

d_model: int = 1024 // 16
seq_len: int = 256
num_layers: int = 4
num_heads: int = 4
multiple_of: int = 4

assert d_model % num_heads == 0

compile_model = not bool(sys.platform == "darwin")

### Dataset and Tokenizer
# dataset_name = "roneneldan/TinyStories"  # run pretokeinze first
# tokenizer_name = "microsoft/phi-2"

dataset_name = "JeanKaddour/minipile"  # run pretokeinze first
tokenizer_name = "EleutherAI/gpt-neo-125m"


### Setup
device = auto_accelerator()  # auto chose device (cuda, mps)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    max_iters: int,
    ignore_index: int = -1,
    device=None,
) -> int:
    model.eval()
    with torch.no_grad():
        losses = torch.zeros(max_iters, device=device)
        for idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
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
    return val_loss


def train(
    model,
    optimizer: optim.Optimizer,
    micro_batch: int,
    train_dataloader,
    val_dataloader,
    eval_iters: int,
    get_lr,
    ignore_index=-1,
    logger=None,
):
    model.to(device)

    ignore_index = ignore_index if ignore_index else -1

    # sanity test
    validate(model, val_dataloader, 5, device=device)

    (data, target) = next(iter(train_dataloader))
    tokerns_per_iter = int(math.prod(data.shape) * micro_batch)

    micro_batch_loss = 0
    idx = 0
    while True:
        micro_batch_loss = 0
        if idx >= max_iters:
            break
        idx += 1

        lr = get_lr(idx)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        for _ in range(micro_batch):
            (data, target) = next(iter(train_dataloader))
            data, target = data.to(device), target.to(device)

            logits: torch.Tensor = model(data)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                ignore_index=ignore_index,
            )
            loss = loss / micro_batch
            loss.backward()
            micro_batch_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        print(f"iter: {idx} | loss: {micro_batch_loss:.4f} | lr: {lr:e} ")

        if idx % eval_iters == 0:
            val_loss = validate(model, val_dataloader, 100, device=device)
            try:
                logger.log(
                    {
                        "traning_loss": micro_batch_loss / micro_batch,
                        "test_loss": val_loss,
                        "iter": idx,
                        "tokens": idx * tokerns_per_iter,
                    },
                    step=idx,
                )
            except Exception as e:
                print(f"Error logging: {e}")


def main():
    wandb.init(project=wandb_project_name)
    # wandb=None

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    config = Config(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_layers,
        multiple_of=multiple_of,
    )

    # loss should be close to cross_entropy before traning
    print(f"{tokenizer.vocab_size=} CrossEntropy={-math.log(1/tokenizer.vocab_size)} ")

    train_ds = PreTokenizedDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split="train",
        max_length=seq_len,
    )
    test_ds = PreTokenizedDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split="validation",
        max_length=seq_len,
    )

    train_dataloader = DataLoader(train_ds, batch_size=batch_size)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)

    model = LLAMA(config).to(device)

    if compile_model:
        model = torch.compile(model)

    print(model)
    print(model_summary(model))

    get_lr = CosineScheduler(
        learning_rate=learning_rate,
        min_lr=min_learning_rate,
        warmup_iters=wornup_iters,
        max_iters=max_iters,
    )

    # inputs = torch.tensor(tokenizer.encode("The")).unsqueeze(0).clone().detach()
    optimzer = optim.AdamW(model.parameters(), lr=get_lr(0))

    train(
        model,
        optimzer,
        micro_batch=micro_batch,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        eval_iters=eval_iters,
        get_lr=get_lr,
        ignore_index=tokenizer.pad_token_id,
        logger=wandb,
    )

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
