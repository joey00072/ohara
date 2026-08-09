import math

from dataclasses import dataclass
from typing import Any


@dataclass
class Scheduler:
    learning_rate: float = 3e-4

    def __call__(self, *args: Any, **kwds: Any) -> float:
        return self.learning_rate


@dataclass
class CosineScheduler:
    learning_rate: float = 3e-4  # Karpathy constant
    min_lr: float = 3e-5  # 1/10 of lr as per chichilla papaer
    warmup_iters: int = 1000
    max_iters: int = 100_0000

    def __post_init__(self) -> None:
        if self.learning_rate < 0 or self.min_lr < 0:
            raise ValueError("learning rates cannot be negative")
        if self.min_lr > self.learning_rate:
            raise ValueError("min_lr cannot exceed learning_rate")
        if self.warmup_iters < 0:
            raise ValueError("warmup_iters cannot be negative")
        if self.max_iters < 1:
            raise ValueError("max_iters must be at least 1")
        if self.warmup_iters >= self.max_iters:
            raise ValueError("warmup_iters must be smaller than max_iters")

    def __call__(self, iteration: int) -> float:
        if iteration < 0:
            raise ValueError("iteration cannot be negative")
        if self.warmup_iters > 0 and iteration < self.warmup_iters:
            return self.learning_rate * iteration / self.warmup_iters

        if iteration > self.max_iters:
            return self.min_lr

        decay_ratio = (iteration - self.warmup_iters) / (self.max_iters - self.warmup_iters)
        assert 0 <= decay_ratio <= 1

        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)


# TODO: jagged cosine learning rate for relora
# I belive parameter efficiency pretraning is possibe only one way to find out
# relora would be good start

if __name__ == "__main__":
    scheduler = CosineScheduler(learning_rate=0.1, min_lr=0.001, warmup_iters=5, max_iters=100)

    print(scheduler)
