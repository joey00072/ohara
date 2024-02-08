import os
import time

# import pretty_errors

from transformers import AutoTokenizer
from ohara.models.phi import Phi,PhiConfig,Block


import torch
import torch.nn as nn

from tqdm import tqdm


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

print(model)
max_tokens = 1000
input_pos = 0
inputs = tokenizer.encode(prompt)


print("="*100)

start: float = time.time()
model = model.eval()
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
