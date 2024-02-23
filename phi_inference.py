import os
import time

import random
# import pretty_errors

from transformers import AutoTokenizer
from ohara.models.phi import Phi, PhiConfig, Block, PhiMHA

from ohara.models.mamba import MambaBlock, MambaConfig

import torch
import torch.nn as nn

from tqdm import tqdm
import torch.nn.functional as F


from ohara.inference import Inference  # Need to use this in future

kv = True
accelerator = "cpu"
if torch.cuda.is_available():
    accelerator = "cuda"
else:
    accelerator = "mps"

device = torch.device(accelerator)

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model: Phi = Phi.from_pretrained(model_name).to(device).eval().to(torch.float16)
prompt = "def dfs(node):"

inputs = tokenizer.encode(prompt)
print(kv)
kv_cache = model.build_kv_cache() if kv == "True" else None


max_tokens = 1000
input_pos = 0
inputs = tokenizer.encode(prompt)


def distill_model(
    Teacher: nn.Module,
    Student: nn.Module,
    d_model,
    seq_len=16,
    batch_size=16,
    iters=100,
    log_iter=10,
):
    Teacher.eval()
    Student.train()

    Teacher = Teacher.to(device).to(torch.float32).eval()
    Student = Student.to(device).to(torch.float32)

    optimizer = torch.optim.AdamW(Student.parameters(), lr=4e-3)

    avg_loss = 0
    for it in range(iters):
        for _ in range(5):
            inputs = torch.randn((batch_size, seq_len, d_model)).to(device)

            with torch.no_grad():
                labels = Teacher(inputs)

            pred = Student(inputs)

            loss = F.mse_loss(pred, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            avg_loss += loss.item()

        # if it % log_iter == 0:
        print(f"Iter: {it} Loss: {avg_loss/5}")
        avg_loss = 0

    return Student


class MLP(nn.Module):
    def __init__(self, d_model=64):
        super().__init__()
        hidden = d_model * 5
        self.up = nn.Linear(d_model, hidden)
        self.down = nn.Linear(hidden, d_model)

    def forward(self, x):
        x = F.relu(self.up(x))
        return self.down(x)


for idx, layer in enumerate(model.layers):
    # if idx!=30:
    #     continue
    layer: Block
    S = MLP(d_model=model.config.d_model)
    T = layer.mlp
    distill_model(T, S, model.config.d_model)
    layer.mlp = S.to(torch.float16)
model = model.to(device)

print("=" * 100)
print(model)
print("=" * 100)

start: float = time.time()
model = model.eval()
with torch.no_grad():
    inputs = torch.tensor(inputs).reshape(1, -1).to(device)
    print(tokenizer.decode(inputs.tolist()[0]), end="")
    for _idx in range(max_tokens):
        logits = model(inputs, kv_cache, input_pos)
        max_logits = torch.argmax(logits, dim=-1)
        # print(f"{idx=} {max_logits.shape}")
        inputs: torch.Tensor = torch.cat((inputs, max_logits[:, -1:]), dim=-1)
        input_pos = inputs.shape[1] - 1
        print(tokenizer.decode(inputs.tolist()[0][-1]), end="", flush=True)

    end: float = time.time()

print(f"Time: {end-start}")


print(tokenizer.decode(inputs.tolist()[0]))
