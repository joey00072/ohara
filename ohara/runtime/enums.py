from __future__ import annotations

from enum import StrEnum


class Backend(StrEnum):
    NCCL = "nccl"
    GLOO = "gloo"
    MPI = "mpi"


class StrategyType(StrEnum):
    AUTO = "auto"
    SINGLE = "single"
    DDP = "ddp"


class PrecisionMode(StrEnum):
    FP32 = "fp32"
    FP16_MIXED = "fp16_mixed"
    BF16_MIXED = "bf16_mixed"
    BF16_TRUE = "bf16_true"


class ReduceType(StrEnum):
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


class PipelineScheduleType(StrEnum):
    ONE_FWD_ONE_BWD = "1f1b"
    GPIPE = "gpipe"


class ParallelDimension(StrEnum):
    DP_REPLICATE = "dp_replicate"
    DP_SHARD = "dp_shard"
    TP = "tp"
    PP = "pp"
    CP = "cp"
    EP = "ep"


class TensorParallelStyle(StrEnum):
    COLWISE = "colwise"
    ROWWISE = "rowwise"
