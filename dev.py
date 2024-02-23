import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        self.linear = nn.Linear(dim)
