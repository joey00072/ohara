import os
import time

# import pretty_errors

from transformers import AutoTokenizer
from ohara.models.phi import Phi, PhiConfig, Block,PhiMHA

from ohara.models.mamba import MambaBlock, MambaConfig

import torch
import torch.nn as nn

from tqdm import tqdm
import torch.nn.functional as F

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


mamba_cfg = MambaConfig(
    d_model=model.config.d_model,
    n_layers=1,
)
mamba = MambaBlock(mamba_cfg).to(device).to(torch.float16)
attn = 

def distill_model(
    Teacher: nn.Module,
    Student: nn.Module,
    d_model,
    seq_len=100,
    batch_size=8,
    iters=100,
    log_iter=100,
):
    Teacher.eval()
    Student.train()

    Teacher = Teacher.to(device).to(torch.float32)
    Student = Student.to(device).to(torch.float32)

    optimizer = torch.optim.AdamW(Student.parameters(), lr=1e-5)

    avg_loss = 0
    for it in range(iters):
        inputs = torch.rand((batch_size, seq_len, d_model)).to(device)

        with torch.no_grad():
            labels = Teacher(inputs)

        pred = Student(inputs)

        loss = F.mse_loss(pred, labels)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        avg_loss += loss.item()

        # if it % log_iter == 0:
        print(f"Iter: {it} Loss: {avg_loss/log_iter}")
        avg_loss = 0

    return Student

model = model.cpu()
model.layers[0].mixer = distill_model(
    model.layers[0].mixer, mamba, model.config.d_model
).eval().to(model.wte.weight.dtype)

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
