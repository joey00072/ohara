import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
import math
from scheduler import CosineWithWarmupLR

import gpt


@torch.no_grad()
def estimate_loss(model: nn.Module, eval_iters: int, iter_batches, ctx):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            data, target = next(batch_iter)
            with ctx:
                logits: torch.Tensor = model(data)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1
                )
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


def main():
    config = gpt.Config()
    model = gpt.GPT(config)
    print(model)


if __name__ == "__main__":
    main()
