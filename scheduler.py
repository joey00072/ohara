import math
from dataclasses import dataclass


@dataclass
class CosineWithWarmupLR:
    learning_rate: float = 5e-4
    min_lr: float = 0.0
    warmup_iters: int = 1000
    lr_decay_iters: int = 100000  # ~= max iters as per Chinchilla

    def get_lr(self, it):
        # 1) Linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * it / self.warmup_iters

        # 2) If it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr

        # 3) In between, use cosine decay down to min learning rate
        decay_ratio = (it - self.warmup_iters) / (
            self.lr_decay_iters - self.warmup_iters
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)
