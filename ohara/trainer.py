import sys
import time
import math
from collections import deque
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from typing import Any, Callable

from ohara.models.llama import Llama, Config
from ohara.lr_scheduler import CosineScheduler
from ohara.dataset import PreTokenizedDataset
from ohara.utils import auto_accelerator, model_summary, BetterCycle

from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import lightning as L

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
        print_every: int = 1,
        eval_val_batches: int = 100,
        eval_train_batches: int = 0,
        grad_clip_norm: float | None = None,
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
        self.print_every = max(1, int(print_every))
        self.eval_val_batches = max(1, int(eval_val_batches))
        self.eval_train_batches = max(0, int(eval_train_batches))
        self.grad_clip_norm = grad_clip_norm
        self.cudagraph_mark_step_begin = cudagraph_mark_step_begin
        self.iter_loss_history: deque[float] = deque(maxlen=iter_loss_window)

        (data, target) = next(self.val_dataloader)
        self.tokens_per_iter = int(math.prod(data.shape) * micro_batch)

    def _maybe_cudagraph_step_begin(self) -> None:
        if not self.cudagraph_mark_step_begin:
            return
        if hasattr(torch, "compiler") and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

    @staticmethod
    def _is_distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _all_reduce_sum(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._is_distributed():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, num_batches: int) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()

        total_loss = torch.zeros((), device=self.fabric.device, dtype=torch.float64)
        total_tokens = torch.zeros((), device=self.fabric.device, dtype=torch.float64)
        total_correct = torch.zeros((), device=self.fabric.device, dtype=torch.float64)

        for _ in range(max(1, int(num_batches))):
            data, target = next(dataloader)
            self._maybe_cudagraph_step_begin()
            logits: torch.Tensor = self.model(data)

            flat_logits = logits.view(-1, logits.size(-1))
            flat_target = target.reshape(-1)
            valid = flat_target != self.ignore_index
            valid_count = valid.sum()
            if valid_count.item() == 0:
                continue

            loss_sum = F.cross_entropy(
                flat_logits,
                flat_target,
                ignore_index=self.ignore_index,
                reduction="sum",
            )
            preds = flat_logits.argmax(dim=-1)
            correct = (preds.eq(flat_target) & valid).sum()

            total_loss += loss_sum
            total_tokens += valid_count
            total_correct += correct

        self._all_reduce_sum(total_loss)
        self._all_reduce_sum(total_tokens)
        self._all_reduce_sum(total_correct)

        tokens = max(float(total_tokens.item()), 1.0)
        mean_loss = float((total_loss / tokens).item())
        ppl = float(math.exp(min(mean_loss, 20.0)))
        bpb = float(mean_loss / math.log(2))
        accuracy = float((total_correct / tokens).item())

        if was_training:
            self.model.train()

        return {
            "loss": mean_loss,
            "ppl": ppl,
            "bpb": bpb,
            "accuracy": accuracy,
            "tokens": float(total_tokens.item()),
        }

    @torch.no_grad()
    def calculate_loss(self, dataloader: DataLoader, num_batches: int) -> torch.Tensor:
        metrics = self.evaluate(dataloader, num_batches)
        return torch.tensor(metrics["loss"], device=self.fabric.device)

    def log_function(
        self,
        idx: int,
        lr: float,
        elapsed_time: float,
        train_step_loss: float | None = None,
        tokens_per_sec: float | None = None,
    ) -> None:
        train_eval = (
            self.evaluate(self.train_dataloader, self.eval_train_batches)
            if self.eval_train_batches > 0
            else None
        )
        val_eval = self.evaluate(self.val_dataloader, self.eval_val_batches)

        message_parts = [
            f"iter: {idx}",
            f"val_loss: {val_eval['loss']:.4f}",
            f"val_ppl: {val_eval['ppl']:.2f}",
            f"val_bpb: {val_eval['bpb']:.4f}",
            f"val_acc: {val_eval['accuracy']:.4f}",
        ]
        if train_eval is not None:
            message_parts.insert(1, f"train_loss: {train_eval['loss']:.4f}")
            message_parts.insert(2, f"train_ppl: {train_eval['ppl']:.2f}")
        message_parts.extend([f"lr: {lr:e}", f"time: {elapsed_time:.4f}s"])
        print(" | ".join(message_parts))

        logs: dict[str, float] = {
            "validation_loss": val_eval["loss"],
            "validation_ppl": val_eval["ppl"],
            "validation_bpb": val_eval["bpb"],
            "validation_accuracy": val_eval["accuracy"],
            "iter": idx,
            "tokens": idx * self.tokens_per_iter,
            "lr": lr,
            "time": elapsed_time,
        }
        if train_eval is not None:
            logs.update(
                {
                    "training_eval_loss": train_eval["loss"],
                    "training_eval_ppl": train_eval["ppl"],
                    "training_eval_bpb": train_eval["bpb"],
                    "training_eval_accuracy": train_eval["accuracy"],
                }
            )
        if train_step_loss is not None:
            logs["training_step_loss"] = float(train_step_loss)
        if tokens_per_sec is not None:
            logs["tokens_per_sec"] = float(tokens_per_sec)

        try:
            self.fabric.log_dict(logs, step=idx)
        except Exception as e:
            print(f"Error logging: {e}")

    def train(self, start_iter: int = 0):
        if not getattr(self.fabric, "_launched", False):
            self.fabric.launch()

        # sanity eval
        _ = self.evaluate(self.val_dataloader, 1)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        idx: int = start_iter
        while True:
            if idx >= self.max_iters:
                break
            idx += 1
            start_time: float = time.perf_counter()

            lr = self.get_lr(idx)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            micro_batch_loss = 0.0
            for micro_step in range(self.micro_batch):
                (data, target) = next(self.train_dataloader)
                sync_context = (
                    self.fabric.no_backward_sync(self.model, enabled=micro_step < self.micro_batch - 1)
                    if self.micro_batch > 1
                    else nullcontext()
                )
                with sync_context:
                    self._maybe_cudagraph_step_begin()
                    logits: torch.Tensor = self.model(data)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target.view(-1),
                        ignore_index=self.ignore_index,
                    )
                    loss_value = float(loss.detach().item())
                    if not math.isfinite(loss_value):
                        raise RuntimeError(f"Non-finite loss detected at iter={idx}: {loss_value}")
                    micro_batch_loss += loss_value
                    self.fabric.backward(loss / self.micro_batch)

            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                self.fabric.clip_gradients(
                    self.model,
                    self.optimizer,
                    max_norm=float(self.grad_clip_norm),
                )

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            step_loss = micro_batch_loss / max(1, self.micro_batch)

            curr_time: float = time.perf_counter()
            elapsed_time: float = curr_time - start_time
            tokens_per_sec = self.tokens_per_iter / max(elapsed_time, 1e-9)
            if idx % self.print_every == 0:
                print(
                    f"iter: {idx} | loss: {step_loss:.4f} | lr: {lr:e} | time: {elapsed_time:.4f}s | tok/s: {tokens_per_sec:.2f}"
                )

            if self.log_iter_loss:
                self.iter_loss_history.append(step_loss)
                iter_loss_avg = sum(self.iter_loss_history) / len(self.iter_loss_history)
                try:
                    self.fabric.log_dict(
                        {
                            "train_iter_loss": step_loss,
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

            if self.eval_iters > 0 and idx % self.eval_iters == 0:
                self.model.eval()
                self.log_function(
                    idx=idx,
                    lr=lr,
                    elapsed_time=elapsed_time,
                    train_step_loss=step_loss,
                    tokens_per_sec=tokens_per_sec,
                )
                self.model.train()

            if self.save_ckpt_iters > 0 and idx % self.save_ckpt_iters == 0:
                state = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "idx": idx,
                    "lr": lr,
                    "train_step_loss": step_loss,
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
    compile_model: bool = sys.platform != "darwin"
    device: torch.device = auto_accelerator()  # select accelerator eg cuda, mps

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

    fabric = L.Fabric(accelerator="auto", devices="auto", precision="bf16-mixed")
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    model = fabric.setup(model)
    optimizer = fabric.setup_optimizers(optimizer)

    trainer: Trainer = Trainer(
        fabric=fabric,
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
        eval_val_batches=100,
        eval_train_batches=0,
        grad_clip_norm=1.0,
    )

    trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()
