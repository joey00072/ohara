import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


## TODO WR
class Network(nn.Module):
    def __init__(self, dim, hdim):
        super().__init__()
        self.up = nn.Linear(dim, dim)
        self.down = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.up(x)
        x = F.silu(x)
        return self.down(x)


def log_prob(logits, labels):
    return torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)


dim = 10
seq = 3
batch = 1

policy = Network(dim, dim)
ref_model = Network(dim, dim)
ref_model.load_state_dict(policy.state_dict())
ref_model = ref_model.eval()

optimizer = optim.AdamW(policy.parameters(), lr=1e-2)
labels = torch.tensor([1, 2, 3]).unsqueeze(0)
inputs = torch.rand((batch, seq, dim))

for _idx in range(100):
    logits = policy(inputs)
    pi = log_prob(logits, labels).sum()
    with torch.no_grad():
        logits = ref_model(inputs)
        ref = log_prob(logits, labels).sum()

    print(pi, ref, pi - ref)
    loss = F.sigmoid(pi - ref)
    print(loss)
    exit(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
