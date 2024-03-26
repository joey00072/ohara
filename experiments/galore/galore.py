from __future__ import annotations

import torch
import torch.nn as nn

from torch import Tensor
# minimum entropy reimplementation https://github.com/jiaweizzhao/GaLore/tree/master


class GaLoreProjector:
    def __init__(
        self,
        rank: int,
        verbose: bool = False,
        update_proj_gap: int = 200,
        scale: float = 1.0,
    ) -> None:
        assert rank > 0, "rank must be a positive integer"
        assert update_proj_gap > 0, "good value is above 100"
        assert scale > 0, "scale must be a greater than 0"

        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix: Tensor | None = None

    def project(self, full_rank_grad: Tensor, iter: int) -> Tensor:
        out_features, in_features = full_rank_grad.shape
        if out_features >= in_features:
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="right"
                )

            low_rank_grad = full_rank_grad @ self.ortho_matrix.t()
        else:
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(
                    full_rank_grad, self.rank, type="left"
                )

            low_rank_grad = self.ortho_matrix.t() @ full_rank_grad

        return low_rank_grad

    def project_back(self, low_rank_grad) -> Tensor:
        out_features, in_features = low_rank_grad.shape

        if out_features >= in_features:
            full_rank_grad = low_rank_grad @ self.ortho_matrix
        else:
            full_rank_grad = self.ortho_matrix @ low_rank_grad

        return full_rank_grad * self.scale

    def get_orthogonal_matrix(self, weights: nn.Parameter, rank: int, type: str) -> Tensor:
        dtype = weights.data.dtype
        device = weights.data.device
        matrix = weights.data.float()  # svd is only supported in float

        U, Sig, Vt = torch.linalg.svd(matrix, full_matrices=False)

        # this is where you will project the orthogonal matrix
        # I am not comfertable with this, they are taking only one part of svd decomposition
        # to project gradient in lower dimension each part of svd UΣVt have diffrent meaning
        # U represent the eigenvectors in columns space
        # V represent the basis vector, in rows space
        # paper just assuimg useing only U or V is sufficient to capture gradient most impormant part
        # which is not the case in mathamatical sense of svd (both U and V are important),
        # will it scale? its just U and V sufficient for learning deep neural network?
        # here is video for svd vibes check https://youtu.be/nbBvuuNVfco?si=G5bLJvOyreTOzQfC
        if type == "right":
            low_rank_matrix = Vt[:rank, :]

        elif type == "left":
            low_rank_matrix: Tensor = U[:, :rank]

        low_rank_matrix = low_rank_matrix.to(device).to(dtype)
        return low_rank_matrix


# TODO: AdamW with glore projection

if __name__ == "__main__":
    torch.manual_seed(0)
    rank: int = 2
    verbose: bool = False
    update_proj_gap: int = 200
    scale: float = 1
    proj_type: str = "std"
    projector = GaLoreProjector(rank, verbose, update_proj_gap, scale, proj_type)
    print(projector.ortho_matrix)
    linear = nn.Linear(6, 5, bias=False)
    params = linear.parameters()
    x = torch.rand(50, 6)
    y: Tensor = linear(x)
    y.tanh().sum().backward()

    # ret = projector.project(linear.weight.grad,0)
    # print(ret)

    xgrad = torch.rand(linear.weight.shape)
    projector.project(torch.rand(linear.weight.shape), 0)
    import os
    import time

    for step in range(2, 1000):
        os.system("clear")
        print(f"------------------------")
        ret = projector.project(xgrad, step)
        print(ret)
        print(f"---------{step}---------")
        time.sleep(0.1)
        bk = projector.project_back(ret)

        print(bk)
        exit(0)
