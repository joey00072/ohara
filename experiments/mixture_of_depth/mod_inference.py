import time
from mixture_of_depth import ModelingMixtureOfDepth, MoD
from transformers import AutoTokenizer
from ohara.utils import auto_accelerator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


device = auto_accelerator("cpu")

tokenizer_name = "EleutherAI/gpt-neo-125m"
model_name = "joey00072/mixture-of-depth-TRex-320.0M"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = ModelingMixtureOfDepth.from_pretrained(model_name).to(device)


prompt = "There is an apple on the"
x: Tensor = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
print(f"1. {x.shape=}")
batch, seqlen = x.shape

x = model.model.token_emb(x)
print(f"2. {x.shape=}")

mod1 = model.model.layers[0]
freqs_cis = model.model.cis[0][:seqlen].to(device), model.model.cis[1][:seqlen].to(device)
mask = model.model.mask
cos, isin = freqs_cis
print(f"2.1 {cos.shape=}, {isin.shape=}")


def inference(self: MoD, x: Tensor, *args, **kwargs):
    batch_size, seq_len, dim = x.shape

    router_logits = self.router(x)
    print(f"router_logits: {router_logits.reshape(-1).softmax(dim=-1)}")

    aux_logits = F.sigmoid(self.aux_router(x))
    print(f"aux_logits: {aux_logits.reshape(-1)}")
    mask = (aux_logits > 0.5).int()
    print(f"mask: {mask.reshape(-1)=}")


x = inference(mod1, x, mask, freqs_cis)
