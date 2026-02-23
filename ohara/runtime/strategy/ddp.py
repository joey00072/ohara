from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any
from torch.nn.parallel import DistributedDataParallel

from .base import BaseStrategy


class DDPStrategy(BaseStrategy):
    def __init__(self, *, init_sync: bool = True) -> None:
        self._init_sync = init_sync

    def setup_module(
        self,
        module: nn.Module,
        *,
        device: torch.device,
        process_group: Any | None = None,
    ) -> nn.Module:
        if device.type == "cuda":
            return DistributedDataParallel(
                module,
                device_ids=[device.index],
                output_device=device.index,
                process_group=process_group,
                init_sync=self._init_sync,
            )
        return DistributedDataParallel(
            module,
            process_group=process_group,
            init_sync=self._init_sync,
        )
