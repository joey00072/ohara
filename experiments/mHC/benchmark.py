import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from model import Block, Config, HyperBlock, ManifoldHyperConnections
from sinkhorn import sinkhorn_log
from ohara.embeddings_pos.rotary import precompute_freqs_cis


@dataclass
class BenchConfig:
    batch_size: int
    seq_len: int
    hidden_size: int
    expansion_rate: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    mhc_sinkhorn_iters: int
    mhc_sinkhorn_eps: float
    dropout: float
    include_common: bool
    backward: bool
    iters: int
    warmup: int
    device: torch.device
    dtype: torch.dtype


class IdentityAttn(nn.Module):
    def forward(self, x, mask=None, freqs_cis=None):
        return x


class IdentityFFN(nn.Module):
    def forward(self, x):
        return x


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timeit(fn, iters: int, warmup: int, device: torch.device, backward: bool, zero_grad_fn):
    if backward:
        torch.set_grad_enabled(True)
    else:
        torch.set_grad_enabled(False)

    for _ in range(warmup):
        out = fn()
        if backward:
            out.float().sum().backward()
            if zero_grad_fn is not None:
                zero_grad_fn()

    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn()
        if backward:
            out.float().sum().backward()
            if zero_grad_fn is not None:
                zero_grad_fn()
    _sync(device)
    end = time.perf_counter()

    torch.set_grad_enabled(True)
    return (end - start) / iters


def _parse_dtype(value: str) -> torch.dtype:
    value = value.lower()
    if value in ("float32", "fp32"):
        return torch.float32
    if value in ("float16", "fp16"):
        return torch.float16
    if value in ("bfloat16", "bf16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {value}")


def _build_config(cfg: BenchConfig, connection_type: str) -> Config:
    return Config(
        vocab_size=32000,
        max_sequence_length=cfg.seq_len,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.hidden_size * 4,
        head_dim=cfg.head_dim,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        num_hidden_layers=1,
        dropout=cfg.dropout,
        bias=False,
        weight_tying=False,
        activation="silu",
        mlp="GLU",
        connection_type=connection_type,
        expansion_rate=cfg.expansion_rate,
        mhc_sinkhorn_iters=cfg.mhc_sinkhorn_iters,
        mhc_sinkhorn_eps=cfg.mhc_sinkhorn_eps,
    )


def _build_mask(cfg: BenchConfig, device: torch.device, dtype: torch.dtype):
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        return None
    mask = torch.full(
        (1, 1, cfg.seq_len, cfg.seq_len),
        float("-inf"),
        device=device,
        dtype=dtype,
    )
    return torch.triu(mask, diagonal=1)


def _build_freqs(cfg: BenchConfig, device: torch.device, dtype: torch.dtype):
    cos, sin = precompute_freqs_cis(cfg.head_dim, cfg.seq_len * 2)
    return cos.to(device=device, dtype=dtype), sin.to(device=device, dtype=dtype)


def _zero_grads(module: nn.Module, *tensors: torch.Tensor):
    def _clear():
        module.zero_grad(set_to_none=True)
        for t in tensors:
            if t.grad is not None:
                t.grad = None

    return _clear


def _attach_identities(block: nn.Module) -> None:
    block.attn = IdentityAttn()
    block.ff = IdentityFFN()


def _bench_components(cfg: BenchConfig):
    device = cfg.device
    dtype = cfg.dtype

    x = torch.randn(
        cfg.batch_size,
        cfg.seq_len,
        cfg.hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=cfg.backward,
    )
    x_h = torch.randn(
        cfg.batch_size,
        cfg.seq_len,
        cfg.expansion_rate,
        cfg.hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=cfg.backward,
    )

    mask = _build_mask(cfg, device, dtype)
    freqs_cis = _build_freqs(cfg, device, dtype)

    results = []

    # Residual baseline
    res_cfg = _build_config(cfg, "residual")
    res_block = Block(res_cfg).to(device=device, dtype=dtype)
    if not cfg.include_common:
        _attach_identities(res_block)
    res_fn = lambda: res_block(x, mask, freqs_cis)
    res_time = _timeit(res_fn, cfg.iters, cfg.warmup, device, cfg.backward, _zero_grads(res_block, x))
    results.append(("residual/block", res_time))

    # mHC parts
    mhc_cfg = _build_config(cfg, "mhc")
    mhc_block = HyperBlock(mhc_cfg, layer_id=0).to(device=device, dtype=dtype)
    if not cfg.include_common:
        _attach_identities(mhc_block)

    mhc_conn = ManifoldHyperConnections(
        dim=cfg.hidden_size,
        rate=cfg.expansion_rate,
        sinkhorn_iters=cfg.mhc_sinkhorn_iters,
        sinkhorn_eps=cfg.mhc_sinkhorn_eps,
    ).to(device=device, dtype=dtype)
    mhc_conn_no = ManifoldHyperConnections(
        dim=cfg.hidden_size,
        rate=cfg.expansion_rate,
        sinkhorn_iters=0,
        sinkhorn_eps=cfg.mhc_sinkhorn_eps,
    ).to(device=device, dtype=dtype)

    def _mhc_conn():
        h_pre, h_post, h_res = mhc_conn(x_h)
        return h_pre.sum() + h_post.sum() + h_res.sum()

    def _mhc_conn_no_sinkhorn():
        h_pre, h_post, h_res = mhc_conn_no(x_h)
        return h_pre.sum() + h_post.sum() + h_res.sum()

    mhc_conn_time = _timeit(
        _mhc_conn, cfg.iters, cfg.warmup, device, cfg.backward, _zero_grads(mhc_conn, x_h)
    )
    mhc_conn_no_time = _timeit(
        _mhc_conn_no_sinkhorn, cfg.iters, cfg.warmup, device, cfg.backward, _zero_grads(mhc_conn_no, x_h)
    )
    results.append(("mhc/conn_forward(sinkhorn)", mhc_conn_time))
    results.append(("mhc/conn_forward(no_sinkhorn)", mhc_conn_no_time))

    log_alpha = torch.randn(
        cfg.batch_size,
        cfg.seq_len,
        cfg.expansion_rate,
        cfg.expansion_rate,
        device=device,
        dtype=dtype,
        requires_grad=cfg.backward,
    )
    sinkhorn_fn = lambda: sinkhorn_log(log_alpha, cfg.mhc_sinkhorn_iters, cfg.mhc_sinkhorn_eps).sum()
    sinkhorn_time = _timeit(
        sinkhorn_fn, cfg.iters, cfg.warmup, device, cfg.backward, _zero_grads(mhc_conn, log_alpha)
    )
    results.append(("mhc/sinkhorn_log", sinkhorn_time))

    mhc_apply_attn = lambda: mhc_block._apply_mhc_attn(x_h, mhc_block.attn_conn, mhc_block.attn_norm, mask, freqs_cis).sum()
    mhc_apply_ffn = lambda: mhc_block._apply_mhc_ffn(x_h, mhc_block.ffn_conn, mhc_block.ffn_norm).sum()
    mhc_apply_attn_time = _timeit(
        mhc_apply_attn, cfg.iters, cfg.warmup, device, cfg.backward, _zero_grads(mhc_block, x_h)
    )
    mhc_apply_ffn_time = _timeit(
        mhc_apply_ffn, cfg.iters, cfg.warmup, device, cfg.backward, _zero_grads(mhc_block, x_h)
    )
    results.append(("mhc/apply_attn_path", mhc_apply_attn_time))
    results.append(("mhc/apply_ffn_path", mhc_apply_ffn_time))

    mhc_block_fn = lambda: mhc_block(x_h, mask, freqs_cis)
    mhc_block_time = _timeit(
        mhc_block_fn, cfg.iters, cfg.warmup, device, cfg.backward, _zero_grads(mhc_block, x_h)
    )
    results.append(("mhc/block", mhc_block_time))

    return results


def _print_results(results, baseline_name: str, backward: bool):
    baseline_time = None
    for name, t in results:
        if name == baseline_name:
            baseline_time = t
            break

    if baseline_time is None:
        baseline_time = results[0][1]
        baseline_name = results[0][0]

    name_width = max(len(name) for name, _ in results) + 2
    mode = "fwd+bwd" if backward else "fwd"
    print(f"mode: {mode} | baseline: {baseline_name}")
    print(f"{'component':<{name_width}}  ms/iter   x_baseline")
    for name, t in results:
        ms = t * 1e3
        ratio = t / baseline_time if baseline_time else float("nan")
        print(f"{name:<{name_width}}  {ms:>7.3f}   {ratio:>8.3f}")

def _find_time(results, name: str):
    for item_name, t in results:
        if item_name == name:
            return t
    return None


def _print_combined_table(fwd_results, bwd_results, baseline_name: str):
    baseline_fwd = _find_time(fwd_results, baseline_name)
    baseline_bwd = _find_time(bwd_results, baseline_name)
    if baseline_fwd is None or baseline_bwd is None:
        return

    fwd_map = {name: t for name, t in fwd_results}
    bwd_map = {name: t for name, t in bwd_results}
    components = [name for name, _ in fwd_results if name in bwd_map]

    print("combined (fwd + fwd+bwd)")
    print(
        "component                      base_fwd_ms  fwd_ms  fwd_slowdown%  "
        "base_bwd_ms  bwd_ms  bwd_slowdown%  total_slowdown%"
    )
    for name in components:
        fwd_t = fwd_map[name]
        bwd_t = bwd_map[name]
        fwd_slow = (fwd_t / baseline_fwd - 1.0) * 100.0 if baseline_fwd else float("nan")
        bwd_slow = (bwd_t / baseline_bwd - 1.0) * 100.0 if baseline_bwd else float("nan")
        total_slow = (
            (fwd_t + bwd_t) / (baseline_fwd + baseline_bwd) - 1.0
            if (baseline_fwd + baseline_bwd)
            else float("nan")
        )
        print(
            f"{name:<30}  {baseline_fwd*1e3:>11.3f}  {fwd_t*1e3:>6.3f}  {fwd_slow:>13.2f}  "
            f"{baseline_bwd*1e3:>11.3f}  {bwd_t*1e3:>6.3f}  {bwd_slow:>13.2f}  "
            f"{total_slow*100.0:>14.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark mHC components vs residual baseline.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--expansion-rate", type=int, default=4)
    parser.add_argument("--num-attn-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--mhc-sinkhorn-iters", type=int, default=20)
    parser.add_argument("--mhc-sinkhorn-eps", type=float, default=1e-6)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--include-common", action="store_true")
    parser.add_argument(
        "--mode",
        choices=["fwd", "bwd", "both"],
        default="both",
        help="Run forward only (fwd), forward+backward (bwd), or both.",
    )
    parser.add_argument("--backward", action="store_true", help="Alias for --mode bwd.")
    args = parser.parse_args()

    if args.num_kv_heads is None:
        args.num_kv_heads = args.num_attn_heads

    if args.hidden_size % args.num_attn_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attn_heads")

    device = (
        torch.device("cuda")
        if args.device == "auto" and torch.cuda.is_available()
        else torch.device(args.device if args.device != "auto" else "cpu")
    )
    dtype = _parse_dtype(args.dtype)
    if device.type == "cpu" and dtype in (torch.float16, torch.bfloat16):
        raise ValueError("float16/bfloat16 on CPU is not supported for this benchmark")

    mode = "bwd" if args.backward else args.mode
    modes = [("fwd", False), ("bwd", True)] if mode == "both" else [(mode, mode == "bwd")]

    results_by_mode = {}
    for mode_name, backward in modes:
        cfg = BenchConfig(
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            hidden_size=args.hidden_size,
            expansion_rate=args.expansion_rate,
            num_attention_heads=args.num_attn_heads,
            num_key_value_heads=args.num_kv_heads,
            head_dim=args.hidden_size // args.num_attn_heads,
            mhc_sinkhorn_iters=args.mhc_sinkhorn_iters,
            mhc_sinkhorn_eps=args.mhc_sinkhorn_eps,
            dropout=args.dropout,
            include_common=args.include_common,
            backward=backward,
            iters=args.iters,
            warmup=args.warmup,
            device=device,
            dtype=dtype,
        )

        torch.manual_seed(0)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(0)

        results = _bench_components(cfg)
        _print_results(results, "residual/block", cfg.backward)
        results_by_mode[mode_name] = results
        if mode_name != modes[-1][0]:
            print()

    if "fwd" in results_by_mode and "bwd" in results_by_mode:
        _print_combined_table(results_by_mode["fwd"], results_by_mode["bwd"], "residual/block")


if __name__ == "__main__":
    main()
