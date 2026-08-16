"""Benchmark dense vs mixture-of-experts throughput at matched FLOPs.

    python examples/bench_moe.py --device-batch-size 8

MoE is often reached for in the hope of "more MFU". It does not work that way:
MFU is achieved FLOPs over peak FLOPs, and routing tokens to experts adds work
that is *not* matmul — a sort, a bincount, a device sync, and one small GEMM per
expert instead of one large one. At equal FLOPs per token an MoE will generally
show **lower** MFU than the dense model it replaces.

What MoE buys is parameters at constant compute, so the fair question is loss per
FLOP, not utilisation. This script measures the throughput half of that trade so
the cost is at least known: it matches FLOPs per token by shrinking each expert
to ``intermediate_size // experts_per_tok`` and reports step time and MFU side by
side.
"""

from __future__ import annotations

import argparse
import time

import torch

from ohara.models.llama import Config, Llama
from ohara.modules.moe import apply_qb_update, expert_load, maximal_violation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare dense and MoE throughput")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--vocab-size", type=int, default=50265)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--device-batch-size", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--experts-per-tok", type=int, default=2)
    parser.add_argument("--moe-layer-interval", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--peak-flops", type=float, default=312e12, help="A100 bf16 dense")
    return parser.parse_args()


def build(args: argparse.Namespace, *, moe: bool) -> Llama:
    # Shrinking each expert by experts_per_tok keeps FLOPs per token equal to the
    # dense model, so the two rows below differ in parameters, not in compute.
    intermediate = (
        args.intermediate_size // args.experts_per_tok if moe else args.intermediate_size
    )
    return Llama(
        Config(
            vocab_size=args.vocab_size,
            hidden_size=args.hidden_size,
            intermediate_size=intermediate,
            max_sequence_length=args.seq_len,
            num_hidden_layers=args.num_layers,
            num_attention_heads=args.num_heads,
            dropout=0.0,
            weight_tying=False,
            init_style="nanochat",
            moe_num_experts=args.num_experts if moe else 0,
            moe_experts_per_tok=args.experts_per_tok,
            moe_layer_interval=args.moe_layer_interval,
        )
    )


def benchmark(model: Llama, args: argparse.Namespace, *, moe: bool) -> dict[str, float]:
    device = torch.device("cuda")
    raw = model
    model = model.to(device)
    if args.compile:
        model = torch.compile(model, dynamic=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    tokens = args.device_batch_size * args.seq_len
    data = torch.randint(0, args.vocab_size, (args.device_batch_size, args.seq_len), device=device)
    target = torch.randint(0, args.vocab_size, (args.device_batch_size, args.seq_len), device=device)

    durations: list[float] = []
    for step in range(args.warmup + args.steps):
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(data)
            loss = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)), target.reshape(-1)
            )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if moe:
            apply_qb_update(raw)
        torch.cuda.synchronize()
        if step >= args.warmup:
            durations.append(time.perf_counter() - started)

    durations.sort()
    median = durations[len(durations) // 2]
    flops_per_token = raw.estimate_flops(args.seq_len)
    result = {
        "step_s": median,
        "tokens_per_s": tokens / median,
        "mfu": 100.0 * flops_per_token * (tokens / median) / args.peak_flops,
        "flops_per_token": flops_per_token,
        "params": sum(p.numel() for p in raw.parameters()),
        "active": raw.active_matmul_parameters(),
        "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    if moe:
        counts = expert_load(raw, reset=False)
        if counts is not None:
            result["maxvio"] = float(maximal_violation(counts).mean())
    del model, raw, optimizer
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs a CUDA device")
    torch.set_float32_matmul_precision("high")

    print(
        f"{args.num_layers}L hidden={args.hidden_size} seq={args.seq_len} "
        f"device_batch={args.device_batch_size} compile={args.compile}"
    )
    print(
        f"MoE: {args.num_experts} experts, top-{args.experts_per_tok}, "
        f"every {args.moe_layer_interval} layer(s), expert width "
        f"{args.intermediate_size // args.experts_per_tok}\n"
    )

    rows = {}
    for name, is_moe in (("dense", False), ("moe", True)):
        rows[name] = benchmark(build(args, moe=is_moe), args, moe=is_moe)

    header = f"{'':6} {'step':>9} {'tok/s':>11} {'MFU':>8} {'params':>12} {'active':>12} {'mem':>8}"
    print(header)
    print("-" * len(header))
    for name, r in rows.items():
        print(
            f"{name:6} {r['step_s']:8.3f}s {r['tokens_per_s']:11,.0f} {r['mfu']:7.1f}% "
            f"{r['params'] / 1e6:11.1f}M {r['active'] / 1e6:11.1f}M {r['peak_mem_gb']:7.1f}G"
        )

    dense, moe = rows["dense"], rows["moe"]
    print(
        f"\nFLOPs/token match: dense {dense['flops_per_token']:,.0f} vs "
        f"moe {moe['flops_per_token']:,.0f} "
        f"({moe['flops_per_token'] / dense['flops_per_token']:.3f}x)"
    )
    print(f"MoE has {moe['params'] / dense['params']:.2f}x the parameters")
    print(f"MoE costs {moe['step_s'] / dense['step_s']:.2f}x the wall time per step")
    print(f"MoE MFU is {moe['mfu'] / dense['mfu']:.2f}x dense")
    if "maxvio" in moe:
        print(f"router MaxVio after {args.steps + args.warmup} steps: {moe['maxvio']:.3f} (0 = balanced)")


if __name__ == "__main__":
    main()
