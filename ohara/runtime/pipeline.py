from __future__ import annotations

from dataclasses import dataclass, field

from .enums import PipelineScheduleType


@dataclass
class PipelineStageSpec:
    module_names: list[str]


@dataclass
class PipelinePlan:
    degree: int = 1
    micro_batch_size: int = 1
    schedule: PipelineScheduleType = PipelineScheduleType.ONE_FWD_ONE_BWD
    stages: list[PipelineStageSpec] = field(default_factory=list)

    def validate(self, local_batch_size: int) -> None:
        if self.degree < 1:
            raise ValueError("pipeline degree must be >= 1")
        if self.micro_batch_size < 1:
            raise ValueError("pipeline micro_batch_size must be >= 1")
        if local_batch_size % self.micro_batch_size != 0:
            raise ValueError(
                f"local_batch_size={local_batch_size} must be divisible by micro_batch_size={self.micro_batch_size}"
            )
        if self.stages and len(self.stages) % self.degree != 0:
            raise ValueError(
                f"number of virtual stages={len(self.stages)} must be divisible by pipeline degree={self.degree}"
            )
