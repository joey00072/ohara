import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        self.linear = nn.Linear(dim)
