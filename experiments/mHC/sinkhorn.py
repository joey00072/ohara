import math
import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


def _sinkhorn_log(log_alpha: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    log_alpha = log_alpha - log_alpha.amax(dim=(-2, -1), keepdim=True)
    for _ in range(iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


_MAX_FUSED_BLOCK = 64


def _next_power_of_2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def _num_warps(block_n: int, block_m: int) -> int:
    block = max(block_n, block_m)
    if block <= 16:
        return 2
    if block <= 32:
        return 4
    return 8


def _use_triton(log_alpha: torch.Tensor, iters: int) -> bool:
    if not _TRITON_AVAILABLE:
        return False
    if not log_alpha.is_cuda:
        return False
    if log_alpha.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if iters <= 0:
        return False
    if log_alpha.ndim < 2:
        return False
    n = log_alpha.shape[-2]
    m = log_alpha.shape[-1]
    if _next_power_of_2(n) > _MAX_FUSED_BLOCK or _next_power_of_2(m) > _MAX_FUSED_BLOCK:
        return False
    return True


if _TRITON_AVAILABLE:

    @triton.jit
    def _sinkhorn_fwd_kernel(
        x_ptr,
        out_ptr,
        stride_b,
        stride_n,
        stride_m,
        n,
        m,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        ITERS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        mask_n = offs_n < n
        mask_m = offs_m < m
        mask = mask_n[:, None] & mask_m[None, :]

        ptrs = x_ptr + pid * stride_b + offs_n[:, None] * stride_n + offs_m[None, :] * stride_m
        x = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)

        row_max = tl.max(x, axis=1)
        max_val = tl.max(row_max, axis=0)
        x = x - max_val

        for _ in tl.static_range(0, ITERS):
            row_max = tl.max(x, axis=1)
            row_max = tl.where(mask_n, row_max, 0.0)
            row_exp = tl.exp(x - row_max[:, None])
            row_sum = tl.sum(row_exp, axis=1)
            row_sum = tl.where(mask_n, row_sum, 1.0)
            row_lse = tl.log(row_sum) + row_max
            x = x - row_lse[:, None]

            col_max = tl.max(x, axis=0)
            col_max = tl.where(mask_m, col_max, 0.0)
            col_exp = tl.exp(x - col_max[None, :])
            col_sum = tl.sum(col_exp, axis=0)
            col_sum = tl.where(mask_m, col_sum, 1.0)
            col_lse = tl.log(col_sum) + col_max
            x = x - col_lse[None, :]

        out_ptrs = out_ptr + pid * stride_b + offs_n[:, None] * stride_n + offs_m[None, :] * stride_m
        tl.store(out_ptrs, tl.exp(x), mask=mask)

    @triton.jit
    def _sinkhorn_bwd_kernel(
        x_ptr,
        grad_out_ptr,
        grad_in_ptr,
        stride_b,
        stride_n,
        stride_m,
        n,
        m,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        ITERS: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        mask_n = offs_n < n
        mask_m = offs_m < m
        mask = mask_n[:, None] & mask_m[None, :]

        ptrs = x_ptr + pid * stride_b + offs_n[:, None] * stride_n + offs_m[None, :] * stride_m
        x = tl.load(ptrs, mask=mask, other=-float("inf")).to(tl.float32)

        row_max = tl.max(x, axis=1)
        max_val = tl.max(row_max, axis=0)
        x = x - max_val

        x0 = x
        for _ in tl.static_range(0, ITERS):
            row_max = tl.max(x, axis=1)
            row_max = tl.where(mask_n, row_max, 0.0)
            row_exp = tl.exp(x - row_max[:, None])
            row_sum = tl.sum(row_exp, axis=1)
            row_sum = tl.where(mask_n, row_sum, 1.0)
            row_lse = tl.log(row_sum) + row_max
            x = x - row_lse[:, None]

            col_max = tl.max(x, axis=0)
            col_max = tl.where(mask_m, col_max, 0.0)
            col_exp = tl.exp(x - col_max[None, :])
            col_sum = tl.sum(col_exp, axis=0)
            col_sum = tl.where(mask_m, col_sum, 1.0)
            col_lse = tl.log(col_sum) + col_max
            x = x - col_lse[None, :]

        grad_ptrs = (
            grad_out_ptr + pid * stride_b + offs_n[:, None] * stride_n + offs_m[None, :] * stride_m
        )
        g = tl.load(grad_ptrs, mask=mask, other=0.0).to(tl.float32)
        g = g * tl.exp(x)

        for t in tl.static_range(0, ITERS):
            idx = tl.full((), ITERS - 1 - t, tl.int32)
            x_step = x0
            x_after_row = x_step
            x_after_col = x_step
            for i in tl.static_range(0, ITERS):
                i_val = tl.full((), i, tl.int32)
                row_max = tl.max(x_step, axis=1)
                row_max = tl.where(mask_n, row_max, 0.0)
                row_exp = tl.exp(x_step - row_max[:, None])
                row_sum = tl.sum(row_exp, axis=1)
                row_sum = tl.where(mask_n, row_sum, 1.0)
                row_lse = tl.log(row_sum) + row_max
                x_row = x_step - row_lse[:, None]

                col_max = tl.max(x_row, axis=0)
                col_max = tl.where(mask_m, col_max, 0.0)
                col_exp = tl.exp(x_row - col_max[None, :])
                col_sum = tl.sum(col_exp, axis=0)
                col_sum = tl.where(mask_m, col_sum, 1.0)
                col_lse = tl.log(col_sum) + col_max
                x_col = x_row - col_lse[None, :]

                use = i_val <= idx
                use_eq = i_val == idx
                x_step = tl.where(use, x_col, x_step)
                x_after_row = tl.where(use_eq, x_row, x_after_row)
                x_after_col = tl.where(use_eq, x_col, x_after_col)

            exp_col = tl.exp(x_step)
            col_sum = tl.sum(g, axis=0)
            col_sum = tl.where(mask_m, col_sum, 0.0)
            g = g - exp_col * col_sum[None, :]

            exp_row = tl.exp(x_after_row)
            row_sum = tl.sum(g, axis=1)
            row_sum = tl.where(mask_n, row_sum, 0.0)
            g = g - exp_row * row_sum[:, None]

        out_ptrs = (
            grad_in_ptr + pid * stride_b + offs_n[:, None] * stride_n + offs_m[None, :] * stride_m
        )
        tl.store(out_ptrs, g, mask=mask)


class _SinkhornTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_alpha: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
        x = log_alpha.float()
        shape = x.shape
        n = shape[-2]
        m = shape[-1]
        batch = math.prod(shape[:-2]) if x.ndim > 2 else 1
        x = x.contiguous().view(batch, n, m)

        out = torch.empty_like(x, dtype=log_alpha.dtype)
        block_n = _next_power_of_2(n)
        block_m = _next_power_of_2(m)
        grid = (batch,)
        _sinkhorn_fwd_kernel[grid](
            x,
            out,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            n,
            m,
            BLOCK_N=block_n,
            BLOCK_M=block_m,
            ITERS=iters,
            num_warps=_num_warps(block_n, block_m),
        )

        ctx.save_for_backward(log_alpha)
        ctx.iters = iters
        ctx.shape = shape
        ctx.orig_dtype = log_alpha.dtype

        return out.view(shape)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        iters = ctx.iters
        shape = ctx.shape
        orig_dtype = ctx.orig_dtype
        if iters == 0:
            return None, None, None
        (log_alpha,) = ctx.saved_tensors
        n = shape[-2]
        m = shape[-1]
        batch = math.prod(shape[:-2]) if len(shape) > 2 else 1

        x = log_alpha.float().contiguous().view(batch, n, m)
        grad_out = grad_out.float().contiguous().view(batch, n, m)
        grad_in = torch.empty_like(x, dtype=orig_dtype)
        block_n = _next_power_of_2(n)
        block_m = _next_power_of_2(m)
        grid = (batch,)
        _sinkhorn_bwd_kernel[grid](
            x,
            grad_out,
            grad_in,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            n,
            m,
            BLOCK_N=block_n,
            BLOCK_M=block_m,
            ITERS=iters,
            num_warps=_num_warps(block_n, block_m),
        )

        return grad_in.view(shape), None, None


def sinkhorn_log(log_alpha: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    if _use_triton(log_alpha, iters):
        return _SinkhornTritonFn.apply(log_alpha, iters, eps)
    orig_dtype = log_alpha.dtype
    return _sinkhorn_log(log_alpha.float(), iters, eps).to(orig_dtype)
