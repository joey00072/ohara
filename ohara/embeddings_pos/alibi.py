import torch


def get_alibi_mask(number_of_heads, max_seq_len):
    """Return per-head distance penalties with shape (heads, sequence, sequence).

    Add to a causal attention mask before softmax; this bias does not mask
    future positions itself. Paper: https://arxiv.org/abs/2108.12409.
    """

    nh = number_of_heads
    n = max_seq_len

    rows = torch.arange(n).view(1, -1, 1)
    cols = torch.arange(n).view(1, 1, -1)

    matrix = rows - cols
    matrix = torch.where(matrix < 1, (torch.zeros_like(matrix)), matrix)

    matrix = matrix.expand(nh, -1, -1) * -1
    m = 1 / 2 ** (torch.arange(1, nh + 1) / (nh / 8))
    return matrix * m.view(nh, 1, 1)
