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

model_name = "cognitivecomputations/dolphin-2_6-phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model: Phi = Phi.from_pretrained(model_name).to(device).eval().to(torch.float16)

print(kv)
kv_cache = model.build_kv_cache() if kv == True else None

model = model.to(device)


print("=" * 100)
print(model)
print("=" * 100)

infer = Inference(model, tokenizer, device)

while True:
    prompt = input("Prompt: ")
    if prompt == "":
        continue
    if prompt == "exit":
        break
    print(infer.generate(prompt, temperature=1.1, top_p=0.2, stream=True))

