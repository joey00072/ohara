from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from dataclasses import dataclass, field
from torch import Tensor
import os
import time
import pygame
import numpy as np


def next_state(tensor: Tensor) -> Tensor:
    """
    Get 2D tensor and return the next Game of Life state with warping logic.
    """
    next_tensor = torch.zeros_like(tensor)
    rows, cols = tensor.shape

    for i in range(rows):
        for j in range(cols):
            live_neighbours = (
                tensor[(i - 1) % rows, (j - 1) % cols]
                + tensor[(i - 1) % rows, j % cols]
                + tensor[(i - 1) % rows, (j + 1) % cols]
                + tensor[i % rows, (j - 1) % cols]
                + tensor[i % rows, (j + 1) % cols]
                + tensor[(i + 1) % rows, (j - 1) % cols]
                + tensor[(i + 1) % rows, j % cols]
                + tensor[(i + 1) % rows, (j + 1) % cols]
            )

            stay_alive = (tensor[i, j] == 1) & ((live_neighbours == 2) | (live_neighbours == 3))
            become_alive = (tensor[i, j] == 0) & (live_neighbours == 3)

            next_tensor[i, j] = stay_alive | become_alive

    return next_tensor


class LifeDataset(IterableDataset):
    """
    Iterable dataset for Game of Life states.
    """

    def __init__(self, size, samples):
        self.size = size
        self.samples = samples
        self._reinit()

    def _reinit(self):
        self.world = torch.randn(self.size, self.size)
        self.world[self.world < 0] = 0
        self.world[self.world > 0] = 1
        self.world = self.world.float()
        self.pasts = []

    def tensor_matches_any(self, tensor):
        stacked_tensors = torch.stack(self.pasts)
        matches = torch.all(torch.eq(tensor, stacked_tensors), dim=list(range(1, tensor.dim() + 1)))
        return torch.any(matches).item()

    def __iter__(self):
        for _ in range(self.samples):
            self.past
            self.world = next_state(self.world)
            if self.tensor_matches_any():
                ...
            yield self.world


if __name__ == "__main__":
    size = 10  # Example size
    samples = 5  # Example number of samples
    dataset = LifeDataset(size, samples)

    for state in dataset:
        print(state)
