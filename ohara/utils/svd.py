import torch
from typing import Tuple

def svd_approx(W: torch.Tensor, r: int = None) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """
    Perform SVD approximation of a matrix with an optional rank specification.
    
    Parameters:
    W (torch.Tensor): The input matrix to be approximated.
    r (int, optional): The rank for the approximation. If None, the full rank is used.

    Returns:
    Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]: 
        A tuple containing the matrices A and B (where A @ B is the approximation of W), 
        the rank used for the approximation, and the Frobenius norm of the difference between W and its approximation.
    """
    U, S, V = torch.linalg.svd(W, full_matrices=False)
    if r is None:
        r = S.size(0)
    else:
        r = min(r, S.size(0))

    A = U[:, :r]
    B = torch.diag(S[:r]) @ V[:r, :]
    approx_W = A @ B
    frobenius_norm = torch.linalg.norm(W - approx_W)
    return A, B, r, frobenius_norm