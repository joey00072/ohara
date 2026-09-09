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
        if self.sequence_parallel:
            raise ValueError("sequence_parallel is not implemented")
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
    root = module
    while isinstance(getattr(root, "_orig_mod", None), nn.Module):
        root = root._orig_mod
    if getattr(root, "_ohara_tensor_parallel", False):
        raise ValueError("tensor parallelism has already been applied to this model")

    layer_plan: dict[str, Any] = {}
    attentions = []
    for fqn, child in root.named_modules():
        if not fqn:
            continue
        if not isinstance(child, nn.Linear):
            continue
        style = plan.style_for(fqn)
        if style is None:
            continue
        dimension = child.out_features if style == TensorParallelStyle.COLWISE else child.in_features
        if dimension % plan.degree:
            raise ValueError(f"{fqn} dimension {dimension} must be divisible by tp={plan.degree}")
        layer_plan[fqn] = _to_torch_style(style)

    for fqn, child in root.named_modules():
        if not hasattr(child, "num_attention_heads") or not hasattr(child, "num_key_value_heads"):
            continue
        qkv = [f"{fqn}.{name}" for name in ("query", "key", "value")]
        if not any(name in layer_plan for name in qkv):
            continue
        if not all(plan.style_for(name) == TensorParallelStyle.COLWISE for name in qkv):
            raise ValueError(f"{fqn}: query, key and value must all be column-sharded")
        if plan.style_for(f"{fqn}.proj") != TensorParallelStyle.ROWWISE:
            raise ValueError(f"{fqn}: attention output must be row-sharded")
        for attr in ("num_attention_heads", "num_key_value_heads"):
            if getattr(child, attr) % plan.degree:
                raise ValueError(f"{fqn}.{attr} must be divisible by tp={plan.degree}")
        attentions.append(child)

    if not layer_plan:
        raise ValueError(
            "No modules matched tensor parallel rules. "
            "Provide matching TensorParallelRule patterns for your model."
        )

    parallelize_module(root, tp_mesh, layer_plan)
    for attention in attentions:
        attention.num_attention_heads //= plan.degree
        attention.num_key_value_heads //= plan.degree
    root._ohara_tensor_parallel = True
    return module
