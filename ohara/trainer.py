import sys
import time
import math
from datetime import datetime
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch import Tensor
from typing import Any, Callable

from ohara.models.llama import Llama, Config
from ohara.lr_scheduler import CosineScheduler
from ohara.dataset import PreTokenizedDataset
from ohara.utils import auto_accelerator, model_summary, BetterCycle

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.loggers import TensorBoardLogger

import wandb
from rich import print, traceback

traceback.install()


class Trainer:
    def __init__(
        self,
        fabric: L.Fabric,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        get_lr: Callable[[int], float],
        micro_batch: int,
        max_iters: int,
        eval_iters: int,
        save_ckpt_iters: int,
        ignore_index: int = -1,
        push_to_hub: bool = False,
        model_name: str = "",
        log_iter_loss: bool = False,
        iter_loss_window: int = 100,
        cudagraph_mark_step_begin: bool = False,
    ):
        self.fabric = fabric
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = BetterCycle(iter(train_dataloader))
        self.val_dataloader = BetterCycle(iter(val_dataloader))
        self.get_lr = get_lr
        self.micro_batch = micro_batch
        self.max_iters = max_iters
        self.eval_iters = eval_iters
        self.save_ckpt_iters = save_ckpt_iters
        self.ignore_index = ignore_index
        self.push_to_hub = push_to_hub
        self.model_name = model_name
        self.log_iter_loss = log_iter_loss
        self.iter_loss_window = iter_loss_window
        self.cudagraph_mark_step_begin = cudagraph_mark_step_begin
        self.iter_loss_history: deque[float] = deque(maxlen=iter_loss_window)

        (data, target) = next(self.val_dataloader)
        self.tokens_per_iter = int(math.prod(data.shape) * micro_batch)

    def _maybe_cudagraph_step_begin(self) -> None:
        if not self.cudagraph_mark_step_begin:
            return
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

    @torch.no_grad()
    def calculate_loss(self, dataloader: DataLoader, num_batches: int) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        losses = torch.zeros(num_batches, device=self.fabric.device)
        for idx, (data, target) in enumerate(dataloader):
            if idx >= num_batches:
                break
            self._maybe_cudagraph_step_begin()
            logits: torch.Tensor = self.model(data)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=self.ignore_index,
            )
            losses[idx] = loss.item()
        if was_training:
            self.model.train()
        return losses.mean()

    def log_function(self, idx: int, lr: float, elapsed_time: float) -> None:
        train_loss = self.calculate_loss(self.train_dataloader, 100)
        val_loss = self.calculate_loss(self.val_dataloader, 100)

        print(
            f"iter: {idx} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | lr: {lr:e} | time: {elapsed_time:.4f}s"
        )

        try:
            self.fabric.log_dict(
                {
                    "training_loss": train_loss,
                    "validation_loss": val_loss,
                    "iter": idx,
                    "tokens": idx * self.tokens_per_iter,
                    "lr": lr,
                    "time": elapsed_time,
                },
                step=idx,
            )
        except Exception as e:
            print(f"Error logging: {e}")

    def train(self, start_iter: int = 0):
        self.fabric.launch()
        # sanity test
        self.calculate_loss(self.val_dataloader, 5)
        self.model.train()

        idx: int = start_iter
        while True:
            if idx >= self.max_iters:
                break
            idx += 1
            start_time: float = time.perf_counter()

            lr = self.get_lr(idx)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            micro_batch_loss = 0
            for _ in range(self.micro_batch):
                (data, target) = next(self.train_dataloader)
                with self.fabric.no_backward_sync(self.model, enabled=_ < self.micro_batch - 1):
                    self._maybe_cudagraph_step_begin()
                    logits: torch.Tensor = self.model(data)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target.view(-1),
                        ignore_index=self.ignore_index,
                    )
                    loss = loss / self.micro_batch
                    micro_batch_loss += loss.item()
                    self.fabric.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            curr_time: float = time.perf_counter()
            elapsed_time: float = curr_time - start_time
            tokens_per_sec = self.tokens_per_iter / max(elapsed_time, 1e-9)
            print(
                f"iter: {idx} | loss: {micro_batch_loss:.4f} | lr: {lr:e} | time: {elapsed_time:.4f}s | tok/s: {tokens_per_sec:.2f}"
            )

            if self.log_iter_loss:
                self.iter_loss_history.append(micro_batch_loss)
                iter_loss_avg = sum(self.iter_loss_history) / len(self.iter_loss_history)
                try:
                    self.fabric.log_dict(
                        {
                            "train_iter_loss": micro_batch_loss,
                            "train_iter_loss_100": iter_loss_avg,
                            "iter": idx,
                            "tokens": idx * self.tokens_per_iter,
                            "lr": lr,
                            "time": elapsed_time,
                            "tokens_per_sec": tokens_per_sec,
                        },
                        step=idx,
                    )
                except Exception as e:
                    print(f"Error logging iter loss: {e}")

            if idx % self.eval_iters == 0:
                self.model.eval()
                self.log_function(idx, lr, elapsed_time)
                self.model.train()

            if idx % self.save_ckpt_iters == 0:
                state = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "idx": idx,
                    "lr": lr,
                }
                self.fabric.save("./ckpt/model.pt", state)
                self.model.config.ckpt_iter = idx
                if self.push_to_hub:
                    self.model.push_to_hub(
                        self.model_name, commit_message=f"checkpoint iter: {idx}"
                    )


def main():
    # wandb
    project_name: str = "Ohara-LLAMA-Trainer"

    # dataset and tokenizer
    pretrained_model: str = "microsoft/phi-2"
    dataset_name: str = "roneneldan/TinyStories"

    # learning_rate
    learning_rate: float = 5e-4
    min_lr: float = 0.0

    warmup_iters: int = 1000
    max_iters: int = 100_000
    eval_iters: int = 100

    # bactch size
    batch_size: int = 32
    micro_batch: int = 4

    # Model Args
    hidden_size: int = 128
    max_sequence_length: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    multiple_of: int = 4
    max_length: int = 256

    # system
    compile_model: bool = not (sys.platform != "darwin")
    device: torch.device = auto_accelerator()  # select accelerator eg cuda, mps
    dtype: torch.dtype = torch.float32  # We will use fancy dtypes in future

    logger: Any = wandb.init(project=project_name)

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    config: Config = Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        max_sequence_length=max_sequence_length,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        multiple_of=multiple_of,
    )

    model: nn.Module = Llama(config).to(device)
    if compile_model:
        model = torch.compile(model)

    print("-" * 100)
    print(model)
    print(model_summary(model))
    print("-" * 100)

    train_ds: PreTokenizedDataset = PreTokenizedDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split="train",
        max_length=max_length,
    )
    test_ds: PreTokenizedDataset = PreTokenizedDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split="validation",
        max_length=max_length,
    )

    train_dataloader: DataLoader = DataLoader(train_ds, batch_size=batch_size)
    val_dataloader: DataLoader = DataLoader(test_ds, batch_size=batch_size)

    optimizer: optim.AdamW = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler: CosineScheduler = CosineScheduler(
        learning_rate=learning_rate,
        min_lr=min_lr,
        warmup_iters=warmup_iters,
        max_iters=max_iters,
    )

    trainer: Trainer = Trainer(
        fabric=L.Fabric(accelerator="auto", devices="auto", precision="bf16-mixed"),
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        get_lr=scheduler,
        micro_batch=micro_batch,
        max_iters=max_iters,
        eval_iters=eval_iters,
        save_ckpt_iters=1000,
        ignore_index=tokenizer.pad_token_id,
        push_to_hub=False,
        model_name="",
    )

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
