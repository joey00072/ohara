from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any

from .base import BaseStrategy


class SingleStrategy(BaseStrategy):
    def setup_module(
        self,
        module: nn.Module,
        *,
        device: torch.device,
        process_group: Any | None = None,
    ) -> nn.Module:
        return module
