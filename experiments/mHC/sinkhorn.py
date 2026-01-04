import torch


def _sinkhorn_log(log_alpha: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    log_alpha = log_alpha - log_alpha.amax(dim=(-2, -1), keepdim=True)
    for _ in range(iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
    return torch.exp(log_alpha)


def sinkhorn_log(log_alpha: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    orig_dtype = log_alpha.dtype
    return _sinkhorn_log(log_alpha.float(), iters, eps).to(orig_dtype)
