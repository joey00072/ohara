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

from typing import Any, Callable, Protocol

from ohara.models.llama import Llama, Config
from ohara.lr_scheduler import CosineScheduler
from ohara.runtime import EngineConfig, OharaEngine
from ohara.dataset import PreTokenizedDataset
from ohara.tokenizer import get_tokenizer
from ohara.utils import model_summary, BetterCycle

from torch.utils.data import DataLoader

import wandb
from rich import print, traceback

traceback.install()


class RuntimeEngine(Protocol):
    device: torch.device
    is_global_zero: bool

    def launch(self) -> None: ...
    def prepare_dataloaders(self, *dataloaders: DataLoader): ...
    def prepare(self, module: nn.Module, *optimizers: optim.Optimizer): ...
    def prepare_optimizers(self, *optimizers: optim.Optimizer): ...
    def no_backward_sync(self, module: nn.Module, enabled: bool = True): ...
    def backward(self, loss: torch.Tensor) -> None: ...
    def clip_gradients(
        self, model: nn.Module, optimizer: optim.Optimizer, max_norm: float
    ) -> None: ...
    def optimizer_step(self, optimizer: optim.Optimizer) -> None: ...
    def save(self, path: str, state: dict[str, Any]) -> None: ...
    def log_dict(self, payload: dict[str, Any], step: int | None = None) -> None: ...


class Trainer:
    def __init__(
        self,
        engine: RuntimeEngine,
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
        flops_per_token: float | None = None,
        peak_flops: float | None = None,
        timing_warmup_steps: int = 10,
        cudagraph_mark_step_begin: bool = False,
    ):
        self.engine = engine
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
        self.flops_per_token = flops_per_token
        self.peak_flops = peak_flops
        self.timing_warmup_steps = max(0, int(timing_warmup_steps))
        self.cudagraph_mark_step_begin = cudagraph_mark_step_begin
        self.iter_loss_history: deque[float] = deque(maxlen=iter_loss_window)
        self.total_training_time_s: float = 0.0
        self.timed_steps: int = 0

        (data, target) = next(self.val_dataloader)
        self.tokens_per_iter = int(math.prod(data.shape) * micro_batch)
        self.world_size = dist.get_world_size() if self._is_distributed() else 1
        self.global_tokens_per_iter = self.tokens_per_iter * self.world_size

        if self.flops_per_token is None:
            self.flops_per_token = self._infer_flops_per_token()
        if self.peak_flops is None:
            self.peak_flops = self._infer_peak_flops()

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

    def _infer_flops_per_token(self) -> float | None:
        model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model, "estimate_flops"):
            try:
                return float(model.estimate_flops())
            except Exception:
                pass
        try:
            # Rough training-rule estimate often used for decoder-only models.
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return float(6.0 * num_params)
        except Exception:
            return None

    def _infer_peak_flops(self) -> float | None:
        device = self.engine.device
        if device.type != "cuda" or not torch.cuda.is_available():
            return None

        device_index = device.index if device.index is not None else torch.cuda.current_device()
        name = torch.cuda.get_device_name(device_index).upper()
        # Approximate BF16 tensor FLOPs/s by SKU family.
        peak_map = {
            "H100": 9.89e14,
            "H200": 9.89e14,
            "A100": 3.12e14,
            "A800": 3.12e14,
            "L40": 1.81e14,
            "L40S": 3.62e14,
            "RTX 4090": 1.65e14,
            "RTX 6000 ADA": 1.46e14,
        }
        for key, value in peak_map.items():
            if key in name:
                return value
        return None

    def _compute_perf_metrics(
        self, idx: int, elapsed_time: float
    ) -> tuple[float, float | None, float | None, float | None]:
        tokens_per_sec = self.global_tokens_per_iter / max(elapsed_time, 1e-9)

        mfu = None
        if self.flops_per_token and self.peak_flops:
            flops_per_sec = self.flops_per_token * tokens_per_sec
            denom = self.peak_flops * max(1, self.world_size)
            if denom > 0:
                mfu = 100.0 * (flops_per_sec / denom)

        if idx > self.timing_warmup_steps:
            self.total_training_time_s += elapsed_time
            self.timed_steps += 1

        eta_seconds = None
        avg_step = None
        if self.timed_steps > 0:
            avg_step = self.total_training_time_s / self.timed_steps
            eta_seconds = max(self.max_iters - idx, 0) * avg_step

        return tokens_per_sec, mfu, eta_seconds, avg_step

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, num_batches: int) -> dict[str, float]:
        was_training = self.model.training
        self.model.eval()

        total_loss = torch.zeros((), device=self.engine.device, dtype=torch.float64)
        total_tokens = torch.zeros((), device=self.engine.device, dtype=torch.float64)
        total_correct = torch.zeros((), device=self.engine.device, dtype=torch.float64)

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
        return torch.tensor(metrics["loss"], device=self.engine.device)

    def log_function(
        self,
        idx: int,
        lr: float,
        elapsed_time: float,
        train_step_loss: float | None = None,
        tokens_per_sec: float | None = None,
        mfu: float | None = None,
        eta_seconds: float | None = None,
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
        if tokens_per_sec is not None:
            message_parts.append(f"tok/s: {tokens_per_sec:,.0f}")
        if mfu is not None:
            message_parts.append(f"mfu: {mfu:.2f}%")
        if eta_seconds is not None:
            message_parts.append(f"eta: {eta_seconds / 60.0:.1f}m")
        print(" | ".join(message_parts))

        logs: dict[str, float] = {
            "validation_loss": val_eval["loss"],
            "validation_ppl": val_eval["ppl"],
            "validation_bpb": val_eval["bpb"],
            "validation_accuracy": val_eval["accuracy"],
            "iter": idx,
            "tokens": idx * self.global_tokens_per_iter,
            "lr": lr,
            "time": elapsed_time,
            "total_training_time_s": self.total_training_time_s,
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
        if mfu is not None:
            logs["mfu"] = float(mfu)
        if eta_seconds is not None:
            logs["eta_seconds"] = float(eta_seconds)

        try:
            self.engine.log_dict(logs, step=idx)
        except Exception as e:
            print(f"Error logging: {e}")

    def train(self, start_iter: int = 0):
        self.engine.launch()

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
                    self.engine.no_backward_sync(
                        self.model, enabled=micro_step < self.micro_batch - 1
                    )
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
                    self.engine.backward(loss / self.micro_batch)

            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                self.engine.clip_gradients(
                    self.model,
                    self.optimizer,
                    max_norm=float(self.grad_clip_norm),
                )

            self.engine.optimizer_step(self.optimizer)
            self.optimizer.zero_grad(set_to_none=True)

            step_loss = micro_batch_loss / max(1, self.micro_batch)

            curr_time: float = time.perf_counter()
            elapsed_time: float = curr_time - start_time
            tokens_per_sec, mfu, eta_seconds, _ = self._compute_perf_metrics(idx, elapsed_time)
            if idx % self.print_every == 0:
                base_msg = (
                    f"iter: {idx} | loss: {step_loss:.4f} | lr: {lr:e} | time: {elapsed_time:.4f}s "
                    f"| tok/s: {tokens_per_sec:,.2f}"
                )
                if mfu is not None:
                    base_msg += f" | mfu: {mfu:.2f}%"
                if eta_seconds is not None:
                    base_msg += f" | eta: {eta_seconds / 60.0:.1f}m"
                print(base_msg)

            if self.log_iter_loss:
                self.iter_loss_history.append(step_loss)
                iter_loss_avg = sum(self.iter_loss_history) / len(self.iter_loss_history)
                try:
                    self.engine.log_dict(
                        {
                            "train_iter_loss": step_loss,
                            "train_iter_loss_100": iter_loss_avg,
                            "iter": idx,
                            "tokens": idx * self.global_tokens_per_iter,
                            "lr": lr,
                            "time": elapsed_time,
                            "tokens_per_sec": tokens_per_sec,
                            "total_training_time_s": self.total_training_time_s,
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
                    mfu=mfu,
                    eta_seconds=eta_seconds,
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
                self.engine.save("./ckpt/model.pt", state)
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
    engine = OharaEngine(EngineConfig())
    engine.launch()
    device: torch.device = engine.device

    logger: Any = wandb.init(project=project_name)

    tokenizer = get_tokenizer(hf_name=pretrained_model, prefer_hf=True)

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

    train_dataloader, val_dataloader = engine.prepare_dataloaders(train_dataloader, val_dataloader)
    model, optimizer = engine.prepare(model, optimizer)

    trainer: Trainer = Trainer(
        engine=engine,
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
