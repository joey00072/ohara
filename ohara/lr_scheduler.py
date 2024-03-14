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

    def __call__(self, iteration) -> float:
        if iteration < self.warmup_iters:
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
