import torch
import torch.nn as nn

from transformers import AutoTokenizer


def smapler(logits, *args, **kwargs):
    ...


def generate(
    model: nn.Module, tokenizer: AutoTokenizer, max_tokens: int = 64, *args, **kwargs
):
    """
    Generate text from a model
    """
    model.eval()

    for _ in range(max_tokens):
        input_ids = torch.tensor(tokenizer.encode("")).unsqueeze(0)
        with torch.no_grad():
            outputs = model.generate(input_ids, **kwargs)
            tokenizer.decode(outputs[0].tolist())
