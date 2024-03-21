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
        device: str,
        temperature: float = 1.0,
        top_p: float = 0.0,
        max_tokens: int = 500,
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

    @staticmethod
    @torch.inference_mode()
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

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stream: bool = True,
    ):
        if temperature is None:
            temperature = self.default_temperature
        if top_p is None:
            top_p = self.default_top_p
        self.kv_cache = self.model.build_kv_cache()

        inputs = self.tokenizer.encode(prompt)
        inputs = torch.tensor(inputs).reshape(1, -1).to(device)
        input_pos = 0
        start_time = time.time()
        with torch.no_grad():
            if stream:
                print(self.tokenizer.decode(inputs.tolist()[0]), end="")
            for _ in range(self.max_tokens):
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
        return self.tokenizer.decode(inputs.tolist()[0][-1])


if __name__ == "__main__":
    torch.manual_seed(0)

    allow_cuda = False
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would",
        # Few shot promt
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        # Zero shot prompt
        """Tell me if the following person is actually Doraemon disguised as human:
        Name: Umar Jamil
        Decision:
        """,
    ]

    model = LLaMA.build(
        checkpoints_dir="llama-2-7b/",
        tokenizer_path="tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device,
    )

    out_tokens, out_texts = model.text_completion(prompts, max_gen_len=64)
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f"{out_texts[i]}")
        print("-" * 50)
