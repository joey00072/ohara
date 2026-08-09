import gc
import time
import math
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

from typing import Any, Callable, Mapping, Protocol

from ohara.utils import BetterCycle

from torch.utils.data import DataLoader


class RuntimeEngine(Protocol):
    device: torch.device
    is_global_zero: bool
    data_parallel_world_size: int

    def launch(self) -> None: ...
    def prepare_dataloaders(self, *dataloaders: DataLoader): ...
    def prepare(self, module: nn.Module, *optimizers: optim.Optimizer): ...
    def prepare_optimizers(self, *optimizers: optim.Optimizer): ...
    def to_device(self, value: Any) -> Any: ...
    def autocast_context(self): ...
    def no_backward_sync(self, module: nn.Module, enabled: bool = True): ...
    def backward(self, loss: torch.Tensor) -> None: ...
    def clip_gradients(
        self, model: nn.Module, optimizer: optim.Optimizer, max_norm: float
    ) -> torch.Tensor: ...
    def optimizer_step(self, optimizer: optim.Optimizer) -> None: ...
    def synchronize(self) -> None: ...
    def save(self, path: str | Path, state: dict[str, Any]) -> None: ...
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
        get_optimizer_hparams: Callable[[int], Mapping[str, float]] | None = None,
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
        checkpoint_path: str | Path = "./ckpt/model.pt",
        token_bytes: torch.Tensor | None = None,
    ):
        if micro_batch < 1:
            raise ValueError("micro_batch must be at least 1")
        if max_iters < 1:
            raise ValueError("max_iters must be at least 1")
        if eval_iters < 0 or save_ckpt_iters < 0:
            raise ValueError("eval_iters and save_ckpt_iters cannot be negative")
        if eval_val_batches < 1:
            raise ValueError("eval_val_batches must be at least 1")
        if iter_loss_window < 1:
            raise ValueError("iter_loss_window must be at least 1")
        self.engine = engine
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = BetterCycle(train_dataloader)
        self.val_dataloader = BetterCycle(val_dataloader)
        self.get_lr = get_lr
        self.get_optimizer_hparams = get_optimizer_hparams
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
        self.checkpoint_path = Path(checkpoint_path)
        if token_bytes is not None and token_bytes.ndim != 1:
            raise ValueError("token_bytes must be a one-dimensional vocabulary lookup")
        self.token_bytes = (
            self.engine.to_device(token_bytes) if token_bytes is not None else None
        )
        self.iter_loss_history: deque[float] = deque(maxlen=iter_loss_window)
        self.total_training_time_s: float = 0.0
        self.timed_steps: int = 0
        self.train_batches_consumed: int = 0
        self._validation_batches: list[tuple[torch.Tensor, torch.Tensor]] = []

        self.tokens_per_iter = 0
        self.world_size = getattr(engine, "data_parallel_world_size", 1)
        self.global_tokens_per_iter = 0

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

        requested_batches = max(1, int(num_batches))
        is_validation = dataloader is self.val_dataloader
        use_cache = is_validation and len(self._validation_batches) >= requested_batches
        if use_cache:
            evaluation_iterator = iter(self._validation_batches[:requested_batches])
        else:
            # Do not advance persistent data iterators during evaluation. The
            # validation prefix is cached after tokenization/device transfer so
            # later evaluations are comparable and require no network access.
            source = dataloader.iterable if isinstance(dataloader, BetterCycle) else dataloader
            evaluation_iterator = BetterCycle(source)
            if is_validation:
                self._validation_batches = [
                    self.engine.to_device(next(evaluation_iterator))
                    for _ in range(requested_batches)
                ]
                evaluation_iterator.close()
                gc.collect()
                evaluation_iterator = iter(self._validation_batches)

        total_loss = torch.zeros((), device=self.engine.device, dtype=torch.float64)
        total_tokens = torch.zeros((), device=self.engine.device, dtype=torch.float64)
        total_correct = torch.zeros((), device=self.engine.device, dtype=torch.float64)
        total_bpb_nats = torch.zeros((), device=self.engine.device, dtype=torch.float64)
        total_bytes = torch.zeros((), device=self.engine.device, dtype=torch.float64)

        for _ in range(requested_batches):
            data, target = self.engine.to_device(next(evaluation_iterator))
            self._maybe_cudagraph_step_begin()
            with self.engine.autocast_context():
                logits: torch.Tensor = self.model(data)

            # Match nanochat: reduced-precision model compute, FP32 loss math.
            flat_logits = logits.float().reshape(-1, logits.size(-1))
            flat_target = target.reshape(-1)
            valid = flat_target != self.ignore_index
            valid_count = valid.sum()
            if valid_count.item() == 0:
                continue

            token_losses = F.cross_entropy(
                flat_logits,
                flat_target,
                ignore_index=self.ignore_index,
                reduction="none",
            )
            loss_sum = token_losses.sum()
            preds = flat_logits.argmax(dim=-1)
            correct = (preds.eq(flat_target) & valid).sum()

            if self.token_bytes is not None:
                if self.token_bytes.numel() < flat_logits.size(-1):
                    raise ValueError("token_bytes is smaller than the model vocabulary")
                safe_target = torch.where(valid, flat_target, torch.zeros_like(flat_target))
                byte_counts = self.token_bytes[safe_target]
                count_for_bpb = valid & byte_counts.gt(0)
                total_bpb_nats += token_losses[count_for_bpb].sum()
                total_bytes += byte_counts[count_for_bpb].sum()

            total_loss += loss_sum
            total_tokens += valid_count
            total_correct += correct

        if isinstance(evaluation_iterator, BetterCycle):
            # In particular, release remote validation streams before opening
            # the training stream. Some Arrow/fsspec versions cannot finalize
            # two live remote Parquet readers safely at interpreter shutdown.
            evaluation_iterator.close()
            gc.collect()

        self._all_reduce_sum(total_loss)
        self._all_reduce_sum(total_tokens)
        self._all_reduce_sum(total_correct)
        self._all_reduce_sum(total_bpb_nats)
        self._all_reduce_sum(total_bytes)

        tokens = float(total_tokens.item())
        if tokens <= 0:
            if was_training:
                self.model.train()
            raise RuntimeError("evaluation produced no valid target tokens")
        mean_loss = float((total_loss / tokens).item())
        ppl = float(math.exp(mean_loss))
        bits_per_token = float(mean_loss / math.log(2))
        accuracy = float((total_correct / tokens).item())

        if was_training:
            self.model.train()

        metrics = {
            "loss": mean_loss,
            "ppl": ppl,
            "bits_per_token": bits_per_token,
            "accuracy": accuracy,
            "tokens": float(total_tokens.item()),
        }
        bytes_count = float(total_bytes.item())
        if self.token_bytes is not None:
            metrics["bpb"] = (
                float(total_bpb_nats.item()) / (math.log(2) * bytes_count)
                if bytes_count > 0
                else float("inf")
            )
            metrics["bytes"] = bytes_count
        return metrics

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
            f"val_bits/token: {val_eval['bits_per_token']:.4f}",
            f"val_acc: {val_eval['accuracy']:.4f}",
        ]
        if "bpb" in val_eval:
            message_parts.insert(4, f"val_bpb: {val_eval['bpb']:.4f}")
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
        if self.engine.is_global_zero:
            print(" | ".join(message_parts))

        logs: dict[str, float] = {
            "validation_loss": val_eval["loss"],
            "validation_ppl": val_eval["ppl"],
            "validation_bits_per_token": val_eval["bits_per_token"],
            "validation_accuracy": val_eval["accuracy"],
            "iter": idx,
            "tokens": idx * self.global_tokens_per_iter,
            "lr": lr,
            "time": elapsed_time,
            "total_training_time_s": self.total_training_time_s,
        }
        if "bpb" in val_eval:
            logs["validation_bpb"] = val_eval["bpb"]
        if train_eval is not None:
            logs.update(
                {
                    "training_eval_loss": train_eval["loss"],
                    "training_eval_ppl": train_eval["ppl"],
                    "training_eval_bits_per_token": train_eval["bits_per_token"],
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
        if not 0 <= start_iter <= self.max_iters:
            raise ValueError("start_iter must be between 0 and max_iters")

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        idx: int = start_iter
        while True:
            if idx >= self.max_iters:
                break
            idx += 1
            self.engine.synchronize()
            start_time: float = time.perf_counter()

            lr = self.get_lr(idx)
            if not math.isfinite(lr) or lr < 0:
                raise RuntimeError(f"invalid learning rate at iter={idx}: {lr}")
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr * float(param_group.get("lr_scale", 1.0))
            if self.get_optimizer_hparams is not None:
                optimizer_hparams = self.get_optimizer_hparams(idx)
                unknown = set(optimizer_hparams) - {"momentum", "weight_decay"}
                if unknown:
                    raise ValueError(
                        "unknown scheduled optimizer hyperparameters: "
                        + ", ".join(sorted(unknown))
                    )
                for param_group in self.optimizer.param_groups:
                    if param_group.get("kind") != "muon":
                        continue
                    if "momentum" in optimizer_hparams:
                        param_group["momentum"] = float(optimizer_hparams["momentum"])
                    if "weight_decay" in optimizer_hparams:
                        param_group["weight_decay"] = float(
                            optimizer_hparams["weight_decay"]
                        )

            accumulated_batches = [
                self.engine.to_device(next(self.train_dataloader))
                for _ in range(self.micro_batch)
            ]
            self.train_batches_consumed += self.micro_batch
            total_valid_tokens = sum(
                (target != self.ignore_index).sum()
                for _, target in accumulated_batches
            )
            if total_valid_tokens.item() == 0:
                raise RuntimeError(f"training batch at iter={idx} has no valid target tokens")

            accumulated_loss_sum = torch.zeros(
                (), device=self.engine.device, dtype=torch.float32
            )
            for micro_step, (data, target) in enumerate(accumulated_batches):
                if self.tokens_per_iter == 0:
                    self.tokens_per_iter = int(data.numel() * self.micro_batch)
                    self.global_tokens_per_iter = self.tokens_per_iter * self.world_size
                sync_context = (
                    self.engine.no_backward_sync(
                        self.model, enabled=micro_step < self.micro_batch - 1
                    )
                    if self.micro_batch > 1
                    else nullcontext()
                )
                with sync_context:
                    self._maybe_cudagraph_step_begin()
                    with self.engine.autocast_context():
                        logits: torch.Tensor = self.model(data)
                        loss_sum = F.cross_entropy(
                            logits.float().reshape(-1, logits.size(-1)),
                            target.reshape(-1),
                            ignore_index=self.ignore_index,
                            reduction="sum",
                        )
                    if not torch.isfinite(loss_sum.detach()):
                        raise RuntimeError(
                            f"Non-finite loss detected at iter={idx}, micro_step={micro_step}: "
                            f"{float(loss_sum.detach())}"
                        )
                    accumulated_loss_sum += loss_sum.detach()
                    self.engine.backward(loss_sum / total_valid_tokens)

            if self.grad_clip_norm is not None and self.grad_clip_norm > 0:
                grad_norm = self.engine.clip_gradients(
                    self.model,
                    self.optimizer,
                    max_norm=float(self.grad_clip_norm),
                )
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm detected at iter={idx}: {float(grad_norm)}"
                    )

            self.engine.optimizer_step(self.optimizer)
            self.optimizer.zero_grad(set_to_none=True)

            step_loss_tensor = accumulated_loss_sum / total_valid_tokens
            if not torch.isfinite(step_loss_tensor):
                raise RuntimeError(
                    f"Non-finite loss detected at iter={idx}: {float(step_loss_tensor)}"
                )
            step_loss = float(step_loss_tensor)

            self.engine.synchronize()
            curr_time: float = time.perf_counter()
            elapsed_time: float = curr_time - start_time
            tokens_per_sec, mfu, eta_seconds, _ = self._compute_perf_metrics(idx, elapsed_time)
            if idx % self.print_every == 0 and self.engine.is_global_zero:
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

            if self.eval_iters > 0 and (idx % self.eval_iters == 0 or idx == self.max_iters):
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

            if self.save_ckpt_iters > 0 and (
                idx % self.save_ckpt_iters == 0 or idx == self.max_iters
            ):
                state = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "idx": idx,
                    "lr": lr,
                    "train_step_loss": step_loss,
                    "gradient_accumulation_steps": self.micro_batch,
                    "train_batches_consumed": self.train_batches_consumed,
                    "torch_rng_state": torch.get_rng_state(),
                }
                if torch.cuda.is_available():
                    state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
                self.engine.save(self.checkpoint_path, state)
                model_config = getattr(self.model, "config", None)
                if model_config is not None:
                    model_config.ckpt_iter = idx
                if self.push_to_hub:
                    self.model.push_to_hub(
                        self.model_name, commit_message=f"checkpoint iter: {idx}"
                    )

        self.close()

    def close(self) -> None:
        """Release streaming iterators before Python begins interpreter shutdown."""
        self.train_dataloader.close()
        self.val_dataloader.close()
        gc.collect()
