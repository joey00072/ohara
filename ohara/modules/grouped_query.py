import torch
from torch import Tensor


def repeat_for_grouped_query(x: Tensor, num_queries: int):
    # assuming shape batch, seq_len, num_heads, head_dim
    x = torch.repeat_interleave(x, num_queries, dim=2)
