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


from model import GPTLM, Config
from ohara.lr_scheduler import CosineScheduler
from ohara.dataset import PreTokenizedDataset
from ohara.utils import BetterCycle

from ohara.utils import auto_accelerator, random_name, model_summary
from ohara.inference import Inference

from torch.utils.data import DataLoader
from transformers import AutoTokenizer


from rich import print, traceback

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.loggers import TensorBoardLogger
traceback.install()


### CONFIGS
wandb_project_name = "Ohara-LLAMA-Fabric"
wandb_run_name = random_name()

learning_rate: float = 3e-4
min_learning_rate: float = 0.0

warmup_iters: int = 1000
max_iters: int = 10_000
total_batch_size = 524288
batch_size: int = 8*2
micro_batch: int = 4
eval_iters: int = 100
save_ckpt_iters: int = 1000

d_model: int = 128
ffn_hidden_dim:int = 128 *4
seq_len: int = 256
num_layers: int = 8
num_heads: int = 8
multiple_of: int = 4
weight_tying = True

assert d_model % num_heads == 0, f"{d_model=} {num_heads=} {d_model%num_heads}"

compile_model = False
compile_mode = "reduce-overhead"
torch.set_float32_matmul_precision('high') #'medium' 

### Dataset and Tokenizer
# dataset_name = "roneneldan/TinyStories"  # run pretokeinze first
# tokenizer_name = "microsoft/phi-2"

dataset_name = "JeanKaddour/minipile"  # run pretokeinze first
tokenizer_name = "EleutherAI/gpt-neo-125m"


### Setup
device = auto_accelerator()  # auto chose device (cuda, mps)


# fabric
strategy: str = "auto"
precision:str = "bf16-mixed"  # "32-true", "16-mixed", "16-true", "bf16-mixed", "bf16-true"

# for restarting training from last checkout
resume_traning = False

tokenizer = None
@torch.no_grad()
def validate(
    fabric: L.Fabric,
    model: nn.Module,
    dataloader: DataLoader,
    max_iters: int,
    ignore_index: int = -1,
    device=None,
) -> int:
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
):
    fabric.launch()
    ignore_index = ignore_index if ignore_index else -1
    # sanity test
    validate(fabric, model, val_dataloader, 5, device=device)
    # cyclining loder so you can runit indefinitely
    train_dataloader = BetterCycle(iter(train_dataloader))
    val_dataloader = BetterCycle(iter(val_dataloader))
    (data, target) = next(val_dataloader)
    print(f"{data.shape=}")
    tokerns_per_iter = int(math.prod(data.shape) * micro_batch)

    micro_batch_loss: float = 0
    idx: int = 0
    while True:
        start_time: float = time.perf_counter()
        micro_batch_loss = 0
        if idx >= max_iters:
            break
        idx += 1

        lr = get_lr(idx)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # ...
        for _ in range(micro_batch):
            (data, target) = next(train_dataloader)
            with fabric.no_backward_sync(model, enabled=micro_batch == 1):
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
                        "traning_loss": micro_batch_loss,
                        "test_loss": val_loss,
                        "iter": idx,
                        "tokens": idx * tokerns_per_iter,
                        "lr": lr,
                        "time": elapsed_time,
                    },
                    step=idx,
                )
                model = model.eval()
                
                pipline = Inference(model=model,tokenizer=tokenizer,device="cuda",use_kv_cache=False,max_tokens=100)
                text = pipline.generate("the",temperature=1.1, top_p=0.2, stream=True)
                print(text)
                model = model.train()
            except Exception as e:
                print(f"Error logging: {e}")

        if idx % save_ckpt_iters == 0:
            state = {"model": model, "optimizer": optimizer, "idx": idx, "lr": lr}
            fabric.save("./ckpt/model.pt", state)


def main():
    global tokenizer
    hyper_params = {
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "max_iters": max_iters,
        "eval_iters": eval_iters,
        "batch_size": batch_size,
        "micro_batch": micro_batch,
        "d_model": d_model,
        "ffn_hidden_dim":ffn_hidden_dim,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "num_heads": num_layers,
        "multiple_of": multiple_of,
        "compile_model": compile_model,
        "tokenizer_name": tokenizer_name,
        "dataset_name": dataset_name,
        "resume_traning": resume_traning,
        "save_ckpt_iters": save_ckpt_iters,
    }
    loggers = []
    # wandb_logger = WandbLogger(project=params["wandb_project_name"], resume=params["resume_traning"])
    # loggers.append(wandb_logger)
    tensorboard_logger = TensorBoardLogger(root_dir="logs", name=wandb_run_name)
    loggers.append(tensorboard_logger)

    fabric: L.Fabric = L.Fabric(strategy=strategy, precision=precision, loggers=loggers)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    config = Config(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        hidden_dim=ffn_hidden_dim,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_layers,
        multiple_of=multiple_of,
        weight_tying=weight_tying,
    )

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
    train_dataloader, test_dataloader = fabric.setup_dataloaders(train_dataloader, test_dataloader)

    model = GPTLM(config)
    model: L.LightningModule = fabric.setup(model)

    if compile_model:
        model = torch.compile(model,mode=compile_mode)

    print(model)
    print(model_summary(model))

    get_lr = CosineScheduler(
        learning_rate=learning_rate,
        min_lr=min_learning_rate,
        warmup_iters=warmup_iters,
        max_iters=max_iters,
    )

    # inputs = torch.tensor(tokenizer.encode("The")).unsqueeze(0).clone().detach()
    optimzer = optim.AdamW(model.parameters(), lr=get_lr(0))
    optimzer = fabric.setup_optimizers(optimzer)
    # Lets GO!!
    train(
        fabric,
        model,
        optimzer,
        micro_batch=micro_batch,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        eval_iters=eval_iters,
        save_ckpt_iters=save_ckpt_iters,
        get_lr=get_lr,
        ignore_index=tokenizer.pad_token_id,
    )

    # TODO: Replace this inference
    start: float = time.time()
    max_tokens = 200
    model = model.eval()
    
    pipline = Inference(model=model,tokenizer=tokenizer)
    text = pipline.generate("The cat")
    print(text)
    end: float = time.time()

    print(f"Time: {end-start}")


if __name__ == "__main__":
    main()
