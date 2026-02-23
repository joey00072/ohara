from __future__ import annotations

from dataclasses import dataclass, field

from .enums import Backend, PrecisionMode, StrategyType
from .tensor_parallel import TensorParallelPlan


@dataclass
class DistributedConfig:
    backend: Backend = Backend.NCCL
    timeout_seconds: int = 1800


@dataclass
class ParallelConfig:
    dp_replicate: int = 1
    dp_shard: int = 1
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1
    infer_dp_shard_from_world_size: bool = True


@dataclass
class PrecisionConfig:
    mode: PrecisionMode = PrecisionMode.BF16_MIXED


@dataclass
class EngineConfig:
    strategy: StrategyType = StrategyType.AUTO
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    precision: PrecisionConfig = field(default_factory=PrecisionConfig)
    tensor_parallel: TensorParallelPlan = field(default_factory=TensorParallelPlan)
