import time
import torch
import torch.nn as nn
from tqdm import tqdm

from torch import Tensor
from typing import Optional

from transformers import AutoTokenizer


class Inference:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: str = None,
        temperature: float = 1.0,
        top_p: float = 0.0,
        max_new_tokens: int = 500,
        use_kv_cache: bool = True,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif not isinstance(device, torch.device):
            device = torch.device(device)
        self.device = device
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.default_temperature = temperature
        self.default_top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.use_kv_cache = use_kv_cache
        if self.use_kv_cache and hasattr(self.model, "build_kv_cache"):
            self.kv_cache = self.model.build_kv_cache()
        else:
            self.kv_cache = None

    @staticmethod
    @torch.inference_mode()
    def sampler(logits, temperature=1, top_p=0.0) -> torch.Tensor:
        logits = logits[:, -1]
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        if temperature == 1 and top_p <= 0:
            # Keep existing default behavior deterministic.
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)
        if top_p <= 0 or top_p >= 1:
            return torch.multinomial(probs, num_samples=1)

        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)  # (B, vocab_size)

        probs_sum = torch.cumsum(probs_sort, dim=-1)

        mask = probs_sum - probs_sort > top_p

        probs_sort[mask] = 0.0

        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

        next_token = torch.multinomial(probs_sort, num_samples=1)

        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        stream: bool = True,
    ):
        if temperature is None:
            temperature = self.default_temperature
        if top_p is None:
            top_p = self.default_top_p
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if self.use_kv_cache and hasattr(self.model, "build_kv_cache"):
            self.kv_cache = self.model.build_kv_cache()
        else:
            self.kv_cache = None
        device = self.device
        inputs = self.tokenizer.encode(prompt)
        inputs = torch.tensor(inputs).reshape(1, -1).to(device)
        input_pos = 0
        start_time = time.time()
        with torch.no_grad():
            if stream:
                print(self.tokenizer.decode(inputs.tolist()[0]), end="")
            for _ in range(max_new_tokens):
                logits = (
                    self.model(inputs, self.kv_cache, input_pos)
                    if self.use_kv_cache
                    else self.model(inputs)
                )
                next_token = self.sampler(logits, temperature=temperature, top_p=top_p)
                if next_token[:, -1:].item() == self.tokenizer.eos_token_id:
                    break
                inputs = torch.cat((inputs, next_token[:, -1:]), dim=-1)
                input_pos = inputs.shape[1] - 1
                if stream:
                    print(self.tokenizer.decode(inputs.tolist()[0][-1]), end="", flush=True)
            end_time = time.time()
        if stream:
            print(f"\nTime: {end_time - start_time}s")
        return self.tokenizer.decode(inputs.squeeze(0).tolist())


if __name__ == "__main__":
    raise SystemExit(
        "This module exposes the `Inference` class. "
        "See the training/example scripts for end-to-end usage."
    )
