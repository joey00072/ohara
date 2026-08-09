from __future__ import annotations

import torch
import torch.nn as nn
import time

from transformers import PreTrainedTokenizerBase


class Inference:
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: str | torch.device | None = None,
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
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        stream: bool = True,
    ):
        if temperature is None:
            temperature = self.default_temperature
        if top_p is None:
            top_p = self.default_top_p
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")
        if self.use_kv_cache and hasattr(self.model, "build_kv_cache"):
            self.kv_cache = self.model.build_kv_cache()
        else:
            self.kv_cache = None
        device = self.device
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            raise ValueError("prompt must encode to at least one token")
        inputs = torch.tensor(token_ids, dtype=torch.long, device=device).reshape(1, -1)
        model_config = getattr(self.model, "config", None)
        max_sequence_length = getattr(model_config, "max_sequence_length", None)
        if max_sequence_length is not None:
            if inputs.size(1) > max_sequence_length:
                raise ValueError("prompt exceeds model max_sequence_length")
            max_new_tokens = min(
                max_new_tokens,
                max(0, max_sequence_length - inputs.size(1)),
            )

        generated = inputs
        model_inputs = inputs
        input_pos = 0
        start_time = time.time()
        with torch.inference_mode():
            if stream:
                print(self.tokenizer.decode(generated.tolist()[0]), end="")
            for _ in range(max_new_tokens):
                logits = (
                    self.model(model_inputs, self.kv_cache, input_pos)
                    if self.use_kv_cache
                    else self.model(generated)
                )
                next_token = self.sampler(logits, temperature=temperature, top_p=top_p)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                generated = torch.cat((generated, next_token), dim=-1)
                if self.use_kv_cache:
                    input_pos += model_inputs.size(1)
                    model_inputs = next_token
                if stream:
                    print(self.tokenizer.decode([next_token.item()]), end="", flush=True)
            end_time = time.time()
        if stream:
            print(f"\nTime: {end_time - start_time}s")
        return self.tokenizer.decode(generated.squeeze(0).tolist())


if __name__ == "__main__":
    raise SystemExit(
        "This module exposes the `Inference` class. "
        "See the training/example scripts for end-to-end usage."
    )
