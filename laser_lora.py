import os
import json
import time
import transformers 
# import pretty_errors

from transformers import AutoTokenizer,AutoModelForCausalLM
from ohara.phi.phi import Phi,Block
from pprint import pprint

import torch
import torch.nn as nn

from tqdm import tqdm
from typing import Tuple


kv = os.getenv("KV")
device = torch.device("mps")

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model:Phi = Phi.from_pretrained(model_name).to(device).eval().to(torch.float16)
prompt = "def dfs(node):"

inputs = tokenizer.encode(prompt)
print(kv)
kv_cache = model.build_kv_cache() if kv=="True" else None

print(model) 
max_tokens = 50
input_pos = 0
inputs = tokenizer.encode(prompt)

start:float = time.time()

inputs = torch.tensor(inputs).reshape(1,-1).to(device)
for idx in range(max_tokens):
    logits = model(inputs,kv_cache,input_pos)
    max_logits = torch.argmax(logits,dim=-1)
    # print(f"{idx=} {max_logits.shape}")
    inputs :torch.Tensor= torch.cat((inputs,max_logits[:,-1:]),dim=-1)
    input_pos=inputs.shape[1]-1
    exit(0)
    
end:float = time.time()

print(f"Time: {end-start}")
    

print(inputs)
print(tokenizer.decode(inputs.tolist()[0]))