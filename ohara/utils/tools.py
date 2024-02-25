from __future__ import annotations

from typing import Any
from collections.abc import Iterable
from dataclasses import dataclass

import torch

def accelerator_device(device=None):
    if isinstance(device,torch.device):
        return device
    if isinstance(device,str):
        # assert device in ["cpu","cuda","mps","xla"]
        return torch.device(device)
    assert device is None, "Pass Valid Device"
    accelerator = "cpu"
    if torch.cuda.is_available():
        accelerator = "cuda"
    if torch.backends.mps.is_built():
        accelerator = "mps"
    return torch.device(accelerator)

@dataclass
class BetterCycle:
    iterable : Iterable
    idx:int = 0
    _iterator:Iterable= None

    def __iter__(self) -> BetterCycle:
        return self

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)

        try:
            return next(self._iterator)
        except StopIteration:
            self.idx += 1
            self._iterator = iter(self.iterable)
            return next(self._iterator)



if __name__=="__main__":
    data = [0,1,2,3]

    cyc = BetterCycle(data)

    for idx, item in enumerate(cyc):
        print(f"{idx}: {item}")
        if idx == 7:
            break
