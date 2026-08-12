"""ohara: small, readable implementations of LLM papers, plus a training stack.

The names below are the ones most scripts need. Everything else is one import
away, e.g. ``from ohara.models.phi import Phi`` or
``from ohara.modules.mlp import SwiGLU``.

Attributes are resolved on first access so that ``import ohara`` costs about as
much as ``import torch``; pulling in the dataset and trainer modules eagerly
would also load ``datasets``, ``transformers`` and ``lightning``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# public name -> module that defines it
_EXPORTS = {
    # models
    "Config": "ohara.models.llama",
    "Llama": "ohara.models.llama",
    # data
    "PreTokenizedDataset": "ohara.dataset",
    "StreamingTextDataset": "ohara.dataset",
    "TinyShakespeareDataset": "ohara.dataset",
    # training
    "CosineScheduler": "ohara.lr_scheduler",
    "Scheduler": "ohara.lr_scheduler",
    "Trainer": "ohara.trainer",
    # runtime / parallelism
    "Backend": "ohara.runtime",
    "DistributedConfig": "ohara.runtime",
    "EngineConfig": "ohara.runtime",
    "OharaEngine": "ohara.runtime",
    "ParallelConfig": "ohara.runtime",
    "PipelinePlan": "ohara.runtime",
    "PipelineScheduleType": "ohara.runtime",
    "PipelineStageSpec": "ohara.runtime",
    "PrecisionConfig": "ohara.runtime",
    "PrecisionMode": "ohara.runtime",
    "ReduceType": "ohara.runtime",
    "StrategyType": "ohara.runtime",
    "TensorParallelPlan": "ohara.runtime",
    "TensorParallelRule": "ohara.runtime",
    "TensorParallelStyle": "ohara.runtime",
}

__all__ = [
    "Backend",
    "Config",
    "CosineScheduler",
    "DistributedConfig",
    "EngineConfig",
    "Llama",
    "OharaEngine",
    "ParallelConfig",
    "PipelinePlan",
    "PipelineScheduleType",
    "PipelineStageSpec",
    "PreTokenizedDataset",
    "PrecisionConfig",
    "PrecisionMode",
    "ReduceType",
    "Scheduler",
    "StrategyType",
    "StreamingTextDataset",
    "TensorParallelPlan",
    "TensorParallelRule",
    "TensorParallelStyle",
    "TinyShakespeareDataset",
    "Trainer",
]

assert sorted(__all__) == sorted(_EXPORTS), "__all__ and _EXPORTS must stay in sync"


def __getattr__(name: str):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # cache so later lookups skip this path
    return value


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:  # let type checkers and IDEs see the real symbols
    from .dataset import PreTokenizedDataset, StreamingTextDataset, TinyShakespeareDataset
    from .lr_scheduler import CosineScheduler, Scheduler
    from .models.llama import Config, Llama
    from .runtime import (
        Backend,
        DistributedConfig,
        EngineConfig,
        OharaEngine,
        ParallelConfig,
        PipelinePlan,
        PipelineScheduleType,
        PipelineStageSpec,
        PrecisionConfig,
        PrecisionMode,
        ReduceType,
        StrategyType,
        TensorParallelPlan,
        TensorParallelRule,
        TensorParallelStyle,
    )
    from .trainer import Trainer
