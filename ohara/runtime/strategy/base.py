from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any

import torch
import torch.nn as nn


class BaseStrategy:
    def setup_module(
        self,
        module: nn.Module,
        *,
        device: torch.device,
        process_group: Any | None = None,
    ) -> nn.Module:
        raise NotImplementedError

    def no_sync(self, module: nn.Module, enabled: bool):
        if not enabled:
            return nullcontext()
        no_sync_fn = getattr(module, "no_sync", None)
        if callable(no_sync_fn):
            ctx = no_sync_fn()
            if isinstance(ctx, AbstractContextManager):
                return ctx
        return nullcontext()
