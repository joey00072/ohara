import time

# import pretty_errors

from transformers import AutoTokenizer
from ohara.models.phi import Phi, Block
from ohara.utils import auto_accelerator


import torch
import torch.nn as nn

import torch.nn.functional as F


kv = True
device = auto_accelerator()

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model: Phi = Phi.from_pretrained(model_name).to(device).eval().to(torch.float16)

print(kv)
kv_cache = model.build_kv_cache() if kv == True else None


import torch
from transformers import AutoTokenizer, PreTrainedModel
from typing import Optional, Union
import time


class ModelInference:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        device: str,
        temperature: float = 1.0,
        top_p: float = 0.0,
        max_tokens: int = 50,
        use_kv_cache: bool = True,
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.max_tokens = max_tokens
        self.use_kv_cache = use_kv_cache
        if use_kv_cache and hasattr(self.model, "build_kv_cache"):
            self.kv_cache = self.model.build_kv_cache()
        else:
            self.kv_cache = None

    def sampler(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        logits = logits[:, -1]  # Get the last token logits
        if temperature == 1:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)  # Sort probabilities
        probs_sum = torch.cumsum(probs_sort, dim=-1)  # Cumulative sum
        mask = probs_sum - probs_sort > top_p  # Create a mask for tokens to exclude
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  # Normalize probabilities
        next_token = torch.multinomial(probs_sort, num_samples=1)  # Sample a token
        next_token = torch.gather(probs_idx, -1, next_token)  # Get the original token index
        return next_token

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        if temperature is None:
            temperature = self.default_temperature
        if top_p is None:
            top_p = self.default_top_p

        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_pos = 0
        start_time = time.time()
        with torch.no_grad():
            print(self.tokenizer.decode(inputs.tolist()[0]), end="")
            for _ in range(self.max_tokens):
                logits = (
                    self.model(inputs, self.kv_cache, input_pos)
                    if self.use_kv_cache
                    else self.model(inputs)
                )
                next_token = self.sampler(logits, temperature=temperature, top_p=top_p)
                inputs = torch.cat((inputs, next_token[:, -1:]), dim=-1)
                input_pos = inputs.shape[1] - 1
                print(self.tokenizer.decode(inputs.tolist()[0][-1]), end="", flush=True)
            end_time = time.time()
        print(f"\nTime: {end_time - start_time}s")


model = model.to(device)


print("=" * 100)
print(model)
print("=" * 100)


def sampler(logits, temperature=1, top_p=0.0) -> torch.Tensor:
    logits = logits[:, -1]
    if temperature == 1:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature

    probs = torch.softmax(logits, dim=-1)

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)  # (B, vocab_size)

    probs_sum = torch.cumsum(probs_sort, dim=-1)

    mask = probs_sum - probs_sort > top_p

    probs_sort[mask] = 0.0

    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    next_token = torch.multinomial(probs_sort, num_samples=1)

    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token


for _ in range(100):
    max_tokens = 50
    input_pos = 0
    prompt = "Once Upon a Time, There was a girl named"
    temperature = float(input("Temperature: "))
    top_p = float(input("Top p: "))

    inputs = tokenizer.encode(prompt)
    inputs = tokenizer.encode(prompt)
    kv_cache = model.build_kv_cache()
    start: float = time.time()
    model = model.eval()
    with torch.no_grad():
        inputs = torch.tensor(inputs).reshape(1, -1).to(device)
        print(tokenizer.decode(inputs.tolist()[0]), end="")
        for _idx in range(max_tokens):
            logits = model(inputs, kv_cache, input_pos)
            max_logits = sampler(logits, temperature=temperature, top_p=top_p)
            # print(f"{idx=} {max_logits.shape}")
            inputs: torch.Tensor = torch.cat((inputs, max_logits[:, -1:]), dim=-1)
            input_pos = inputs.shape[1] - 1
            print(tokenizer.decode(inputs.tolist()[0][-1]), end="", flush=True)

        end: float = time.time()

    print(f"\nTime: {end-start}")
    print("-" * 100)
    temperature += 0.5


print(tokenizer.decode(inputs.tolist()[0]))
