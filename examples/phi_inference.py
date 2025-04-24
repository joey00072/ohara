import time

import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import AutoTokenizer, PreTrainedModel
from typing import Optional, Union


# import pretty_errors

from transformers import AutoTokenizer
from ohara.models.phi import Phi, Block
from ohara.utils import auto_accelerator
from ohara.inference import Inference

kv = True
device = auto_accelerator()

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model: Phi = Phi.from_pretrained(model_name).to(device).eval()

print(kv)
kv_cache = model.build_kv_cache() if kv == True else None

model = model.to(device)

# model = torch.compile(model)

print("=" * 100)
print(model)
print("=" * 100)

max_new_tokens = 100

infer = Inference(model, tokenizer, device)

while True:
    prompt = "Once upon a time" #+ input("Prompt: ")
    if prompt == "":
        continue
    if prompt == "exit":
        break
    start = time.perf_counter()
    print(infer.generate(prompt, temperature=1.1, top_p=0.2, stream=False, max_new_tokens=max_new_tokens))
    end = time.perf_counter()
    print(f"Time taken: {end - start} seconds")
    # exit()