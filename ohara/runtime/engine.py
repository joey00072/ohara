from __future__ import annotations

import os
from contextlib import nullcontext
from collections.abc import Mapping
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, cast

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset, RandomSampler

from .config import EngineConfig
from .enums import Backend, PrecisionMode, ReduceType, StrategyType
from .strategy import BaseStrategy, DDPStrategy, SingleStrategy
from .tensor_parallel import TensorParallelPlan, apply_tensor_parallel
from .topology import ParallelTopology


class _DeviceDataLoader:
    """Yield batches from a dataloader on the engine's compute device."""

    def __init__(self, dataloader: DataLoader, engine: OharaEngine) -> None:
        self.dataloader = dataloader
        self.engine = engine
        self.epoch = 0

    def __iter__(self):
        sampler = getattr(self.dataloader, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(self.epoch)
            self.epoch += 1
        for batch in self.dataloader:
            yield self.engine.to_device(batch)

    def __len__(self) -> int:
        return len(self.dataloader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataloader, name)


def _to_reduce_op(reduce_type: ReduceType) -> Any:
    if reduce_type == ReduceType.SUM or reduce_type == ReduceType.MEAN:
        return dist.ReduceOp.SUM
    if reduce_type == ReduceType.MAX:
        return dist.ReduceOp.MAX
    if reduce_type == ReduceType.MIN:
        return dist.ReduceOp.MIN
    raise ValueError(f"Unsupported reduce type: {reduce_type}")


class OharaEngine:
    def __init__(
        self, config: EngineConfig | None = None, loggers: list[Any] | None = None
    ) -> None:
        self.config = config or EngineConfig()
        self.loggers = loggers or []
        self._launched = False
        self._device = torch.device("cpu")
        self._strategy: BaseStrategy = SingleStrategy()
        try:
            self.topology = ParallelTopology.from_config(1, self.config.parallel)
        except ValueError:
            self.topology = ParallelTopology(
                world_size=1,
                dp_replicate=1,
                dp_shard=1,
                tp=1,
                pp=1,
                cp=1,
                ep=1,
            )
        self._scaler = self._build_grad_scaler(enabled=False)
        self._world_mesh: DeviceMesh | None = None
        self._tp_mesh: DeviceMesh | None = None
        self._dp_group: Any | None = None
        self._dp_rank: int = 0
        self._dp_world_size: int = 1

    @staticmethod
    def _build_grad_scaler(enabled: bool):
        try:
            grad_scaler_cls = cast(Any, torch.amp).GradScaler
            return grad_scaler_cls("cuda", enabled=enabled)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=enabled)

    @property
    def launched(self) -> bool:
        return self._launched

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def world_size(self) -> int:
        return self.topology.world_size

    @property
    def global_rank(self) -> int:
        return self.topology.global_rank

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def data_parallel_rank(self) -> int:
        """Rank used to shard input data (TP ranks intentionally share it)."""
        return self.topology.dp_rank

    @property
    def data_parallel_world_size(self) -> int:
        """Number of independent data-parallel replicas."""
        return self.topology.dp_world_size

    def launch(self, function: Callable[..., Any] | None = None, *args: Any, **kwargs: Any) -> Any:
        if self._launched:
            if function is not None:
                return function(*args, **kwargs)
            return None

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        distributed_requested = world_size > 1 or int(os.environ.get("RANK", "0")) > 0
        selected_backend: str | None = None

        if distributed_requested and not dist.is_initialized():
            backend = self.config.distributed.backend
            if backend == Backend.NCCL and not torch.cuda.is_available():
                backend = Backend.GLOO
            selected_backend = backend.value

            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                if backend == Backend.NCCL and local_rank < torch.cuda.device_count():
                    torch.cuda.set_device(local_rank)

            dist.init_process_group(
                backend=backend.value,
                timeout=timedelta(seconds=self.config.distributed.timeout_seconds),
            )

        if dist.is_initialized():
            world_size = dist.get_world_size()
            selected_backend = dist.get_backend()

        self.topology = ParallelTopology.from_config(world_size, self.config.parallel)

        if torch.cuda.is_available() and (
            not distributed_requested or selected_backend == Backend.NCCL.value
        ):
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            self._device = torch.device(f"cuda:{local_rank}")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        strategy = self.config.strategy
        if strategy == StrategyType.AUTO:
            strategy = StrategyType.DDP if self.topology.dp_enabled else StrategyType.SINGLE
        if strategy == StrategyType.DDP and not self.topology.distributed_enabled:
            strategy = StrategyType.SINGLE

        if self.topology.tp_enabled:
            self._build_meshes()

        self._configure_data_parallel_groups(strategy)

        if strategy == StrategyType.DDP:
            self._strategy = DDPStrategy(init_sync=not self.topology.tp_enabled)
        else:
            self._strategy = SingleStrategy()

        use_grad_scaler = (
            self.config.precision.mode == PrecisionMode.FP16_MIXED and self._device.type == "cuda"
        )
        self._scaler = self._build_grad_scaler(enabled=use_grad_scaler)
        self._launched = True
        if function is not None:
            return function(*args, **kwargs)
        return None

    def prepare(self, module: torch.nn.Module, *optimizers: Optimizer):
        module = self.prepare_module(module)
        if not optimizers:
            return module
        wrapped_optimizers = self.prepare_optimizers(*optimizers)
        if len(wrapped_optimizers) == 1:
            return module, wrapped_optimizers[0]
        return (module, *wrapped_optimizers)

    def prepare_module(
        self, module: torch.nn.Module, *, move_to_device: bool = True
    ) -> torch.nn.Module:
        self.launch()
        if move_to_device:
            if self.config.precision.mode == PrecisionMode.BF16_TRUE:
                module = module.to(device=self._device, dtype=torch.bfloat16)
            else:
                module = module.to(self._device)
        module = self._maybe_apply_tensor_parallel(module)
        return self._strategy.setup_module(
            module, device=self._device, process_group=self._dp_group
        )

    # Fabric parity aliases
    def setup(self, module: torch.nn.Module, *optimizers: Optimizer):
        return self.prepare(module, *optimizers)

    def setup_module(
        self, module: torch.nn.Module, *, move_to_device: bool = True
    ) -> torch.nn.Module:
        return self.prepare_module(module, move_to_device=move_to_device)

    def setup_optimizers(self, *optimizers: Optimizer):
        return self.prepare_optimizers(*optimizers)

    def setup_dataloaders(self, *dataloaders: DataLoader):
        return self.prepare_dataloaders(*dataloaders)

    def _build_meshes(self) -> None:
        if not dist.is_initialized():
            raise RuntimeError(
                "Distributed process group must be initialized before building TP mesh"
            )

        dim_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp", "ep")
        dim_sizes = (
            self.topology.pp,
            self.topology.dp_replicate,
            self.topology.dp_shard,
            self.topology.cp,
            self.topology.tp,
            self.topology.ep,
        )
        self._world_mesh = init_device_mesh(
            self._device.type,
            dim_sizes,
            mesh_dim_names=dim_names,
        )
        self._tp_mesh = self._world_mesh["tp"]

    def _configure_data_parallel_groups(self, strategy: StrategyType) -> None:
        self._dp_group = None
        self._dp_world_size = self.topology.dp_world_size if strategy == StrategyType.DDP else 1
        self._dp_rank = self.topology.dp_rank if strategy == StrategyType.DDP else 0

        if strategy != StrategyType.DDP:
            return
        if not dist.is_initialized() or self.topology.dp_world_size <= 1:
            return

        if (
            not self.topology.tp_enabled
            and self.topology.pp == 1
            and self.topology.cp == 1
            and self.topology.ep == 1
        ):
            self._dp_group = None
            return

        current = self.topology.global_rank
        for group_ranks in self.topology.iter_all_data_parallel_groups():
            pg = dist.new_group(ranks=group_ranks)
            if current in group_ranks:
                self._dp_group = pg

    def _resolve_tensor_parallel_plan(self, module: torch.nn.Module) -> TensorParallelPlan:
        plan = self.config.tensor_parallel
        if not self.topology.tp_enabled:
            return plan

        if plan.degree in (0, 1):
            plan = TensorParallelPlan.llama_default(degree=self.topology.tp)
        if plan.degree != self.topology.tp:
            raise ValueError(
                f"TensorParallelPlan.degree={plan.degree} does not match topology.tp={self.topology.tp}"
            )

        hidden_size = getattr(getattr(module, "config", None), "hidden_size", None)
        plan.validate(hidden_size=hidden_size)
        return plan

    def _maybe_apply_tensor_parallel(self, module: torch.nn.Module) -> torch.nn.Module:
        if not self.topology.tp_enabled:
            return module
        if self._tp_mesh is None:
            self._build_meshes()
        plan = self._resolve_tensor_parallel_plan(module)
        assert self._tp_mesh is not None
        return apply_tensor_parallel(module, self._tp_mesh, plan)

    def prepare_optimizers(self, *optimizers: Optimizer):
        self.launch()
        return optimizers

    def _replace_sampler(self, dataloader: DataLoader) -> DataLoader:
        if not hasattr(dataloader, "dataset"):
            return dataloader
        dataset = dataloader.dataset
        if isinstance(dataset, IterableDataset):
            return dataloader
        if isinstance(getattr(dataloader, "sampler", None), DistributedSampler):
            return dataloader
        if dataloader.batch_size is None:
            return dataloader

        sampler = DistributedSampler(
            dataset,
            num_replicas=self._dp_world_size,
            rank=self._dp_rank,
            shuffle=isinstance(dataloader.sampler, RandomSampler),
            drop_last=dataloader.drop_last,
        )

        kwargs: dict[str, Any] = {
            "dataset": dataset,
            "batch_size": dataloader.batch_size,
            "sampler": sampler,
            "num_workers": dataloader.num_workers,
            "collate_fn": dataloader.collate_fn,
            "pin_memory": dataloader.pin_memory,
            "drop_last": dataloader.drop_last,
            "timeout": dataloader.timeout,
            "worker_init_fn": dataloader.worker_init_fn,
            "persistent_workers": dataloader.persistent_workers,
        }
        if dataloader.num_workers > 0:
            kwargs["prefetch_factor"] = dataloader.prefetch_factor
        return DataLoader(**kwargs)

    def prepare_dataloaders(self, *dataloaders: DataLoader):
        self.launch()
        if len(dataloaders) == 0:
            raise ValueError("At least one dataloader is required")
        distributed = (
            [self._replace_sampler(dl) for dl in dataloaders]
            if self._dp_world_size > 1
            else list(dataloaders)
        )
        wrapped = [
            dl if isinstance(dl, _DeviceDataLoader) else _DeviceDataLoader(dl, self)
            for dl in distributed
        ]
        return tuple(wrapped) if len(wrapped) > 1 else wrapped[0]

    def to_device(self, value: Any) -> Any:
        """Recursively move tensors in a nested batch to the compute device."""
        if isinstance(value, torch.Tensor):
            return value.to(
                self._device,
                non_blocking=self._device.type == "cuda",
            )
        if isinstance(value, Mapping):
            moved = {key: self.to_device(item) for key, item in value.items()}
            try:
                return type(value)(moved)
            except TypeError:
                return moved
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            return type(value)(*(self.to_device(item) for item in value))
        if isinstance(value, tuple):
            return tuple(self.to_device(item) for item in value)
        if isinstance(value, list):
            return [self.to_device(item) for item in value]
        return value

    def backward(self, loss: torch.Tensor) -> None:
        if self._scaler.is_enabled():
            self._scaler.scale(loss).backward()
        else:
            loss.backward()

    def no_sync(self, module: torch.nn.Module, enabled: bool = True):
        return self._strategy.no_sync(module, enabled=enabled)

    def no_backward_sync(self, module: torch.nn.Module, enabled: bool = True):
        return self.no_sync(module, enabled=enabled)

    def clip_gradients(
        self, model: torch.nn.Module, optimizer: Optimizer, max_norm: float
    ) -> torch.Tensor:
        if self._scaler.is_enabled():
            self._scaler.unscale_(optimizer)
        return clip_grad_norm_(model.parameters(), max_norm=max_norm)

    def optimizer_step(self, optimizer: Optimizer) -> None:
        if self._scaler.is_enabled():
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()

    def barrier(self) -> None:
        if dist.is_initialized():
            if dist.get_backend() == Backend.NCCL.value and self._device.type == "cuda":
                dist.barrier(device_ids=[self._device.index])
            else:
                dist.barrier()

    def synchronize(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)

    def broadcast(self, obj: Any, src: int = 0) -> Any:
        if not dist.is_initialized():
            return obj
        payload = [obj]
        dist.broadcast_object_list(payload, src=src)
        return payload[0]

    def all_reduce(
        self,
        tensor: torch.Tensor,
        reduce_type: ReduceType | str = ReduceType.MEAN,
        *,
        group: Any | None = None,
        reduce_op: ReduceType | str | None = None,
    ) -> torch.Tensor:
        selected = reduce_op if reduce_op is not None else reduce_type
        if isinstance(selected, str):
            selected = ReduceType(selected)
        if not dist.is_initialized():
            return tensor
        dist.all_reduce(tensor, op=_to_reduce_op(selected), group=group)
        if selected == ReduceType.MEAN:
            world = dist.get_world_size(group=group) if group is not None else self.world_size
            tensor /= world
        return tensor

    def all_gather(self, tensor: torch.Tensor, group: Any | None = None) -> list[torch.Tensor]:
        if not dist.is_initialized():
            return [tensor]
        world = dist.get_world_size(group=group) if group is not None else self.world_size
        gathered = [torch.zeros_like(tensor) for _ in range(world)]
        dist.all_gather(gathered, tensor, group=group)
        return gathered

    def save(self, path: str | Path, state: dict[str, Any]) -> None:
        path = Path(path)
        if self.is_global_zero:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            try:
                torch.save(state, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        self.barrier()

    def load(self, path: str | Path, state: dict[str, Any] | None = None) -> dict[str, Any]:
        path = Path(path)
        payload: dict[str, Any] = torch.load(
            path,
            map_location=self._device,
            weights_only=False,
        )
        if state is None:
            return payload

        for key, obj in state.items():
            if key not in payload:
                continue
            value = payload[key]
            if hasattr(obj, "load_state_dict"):
                obj.load_state_dict(value)
            else:
                state[key] = value
        return payload

    def log_dict(self, payload: dict[str, Any], step: int | None = None) -> None:
        if not self.is_global_zero:
            return
        for logger in self.loggers:
            if hasattr(logger, "log_metrics"):
                logger.log_metrics(payload, step=step)
            elif hasattr(logger, "log_dict"):
                logger.log_dict(payload, step=step)

    def autocast_context(self):
        mode = self.config.precision.mode
        if self._device.type != "cuda":
            return nullcontext()
        if mode == PrecisionMode.FP16_MIXED:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if mode == PrecisionMode.BF16_MIXED:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def close(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
