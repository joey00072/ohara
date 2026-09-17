"""Measure sparse attention forward/backward on a single GPU."""

import argparse

import torch
from triton.testing import do_bench

from ohara.kernels.qwen38_sparse import sparse_attention
from ohara.models.qwen38.attention import select_blocks
from ohara.models.qwen38.ops import sparse_attention_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--budget", type=int, default=64)
    args = parser.parse_args()
    if args.length < 4 or args.budget < 4 or args.budget % 4:
        parser.error("length must be >= 4; budget must be a positive multiple of 4")
    torch.manual_seed(42)
    q = torch.randn(1, args.length, 8, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, args.length, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    scores = torch.randn(1, args.length, args.length // 4, device="cuda")
    indices, _, _ = select_blocks(scores, torch.arange(args.length, device="cuda"), 4, args.budget)
    gradient = torch.randn_like(q)
    print(f"{torch.cuda.get_device_name()}: BF16, length={args.length}, budget={args.budget}, heads=8/2, dim=64")
    for name, operation in (("gather+SDPA", sparse_attention_reference), ("indexed Triton", sparse_attention)):
        def run(operation=operation):
            out = operation(q, k, v, indices)
            torch.autograd.grad(out, (q, k, v), gradient)

        run()  # Compile before timing or counting transient allocations.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        run()
        torch.cuda.synchronize()
        memory = (torch.cuda.max_memory_allocated() - baseline) / 2**20
        milliseconds = do_bench(run, warmup=100, rep=500)
        print(f"{name}: forward+backward={milliseconds:.3f} ms, extra_peak={memory:.1f} MiB")


if __name__ == "__main__":
    main()
