from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from .enums import TensorParallelStyle


@dataclass
class TensorParallelRule:
    module_pattern: str
    style: TensorParallelStyle


@dataclass
class TensorParallelPlan:
    degree: int = 1
    sequence_parallel: bool = False
    rules: list[TensorParallelRule] = field(default_factory=list)

    def validate(self, hidden_size: int | None = None) -> None:
        if self.degree < 1:
            raise ValueError("tensor parallel degree must be >= 1")
        if hidden_size is not None and hidden_size % self.degree != 0:
            raise ValueError(
                f"hidden_size={hidden_size} must be divisible by tensor parallel degree={self.degree}"
            )

    @classmethod
    def llama_default(cls, degree: int) -> "TensorParallelPlan":
        return cls(
            degree=degree,
            sequence_parallel=False,
            rules=[
                TensorParallelRule("layers.*.attn.query", TensorParallelStyle.COLWISE),
                TensorParallelRule("layers.*.attn.key", TensorParallelStyle.COLWISE),
                TensorParallelRule("layers.*.attn.value", TensorParallelStyle.COLWISE),
                TensorParallelRule("layers.*.attn.proj", TensorParallelStyle.ROWWISE),
                TensorParallelRule("layers.*.ff.up", TensorParallelStyle.COLWISE),
                TensorParallelRule("layers.*.ff.gate", TensorParallelStyle.COLWISE),
                TensorParallelRule("layers.*.ff.down", TensorParallelStyle.ROWWISE),
            ],
        )

    def style_for(self, module_fqn: str) -> TensorParallelStyle | None:
        for rule in self.rules:
            if fnmatch.fnmatch(module_fqn, rule.module_pattern):
                return rule.style
        return None


def _to_torch_style(style: TensorParallelStyle):
    if style == TensorParallelStyle.COLWISE:
        return ColwiseParallel()
    if style == TensorParallelStyle.ROWWISE:
        return RowwiseParallel()
    raise ValueError(f"Unsupported tensor parallel style: {style}")


def apply_tensor_parallel(
    module: nn.Module,
    tp_mesh: DeviceMesh,
    plan: TensorParallelPlan,
) -> nn.Module:
    plan.validate()

    layer_plan: dict[str, Any] = {}
    for fqn, child in module.named_modules():
        if not fqn:
            continue
        if not isinstance(child, nn.Linear):
            continue
        style = plan.style_for(fqn)
        if style is None:
            continue
        layer_plan[fqn] = _to_torch_style(style)

    if not layer_plan:
        raise ValueError(
            "No modules matched tensor parallel rules. "
            "Provide matching TensorParallelRule patterns for your model."
        )

    parallelize_module(module, tp_mesh, layer_plan)
    return module
