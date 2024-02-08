from transformers import AutoTokenizer
from tqdm import tqdm
from itertools import cycle
from datasets import load_dataset

import torch

ds = load_dataset("JeanKaddour/minipile", split="train", cache_dir="hf_cache")
ds_val = load_dataset("JeanKaddour/minipile", split="validation", cache_dir="hf_cache")

tokenizer = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits", cache_dir="hf_cache", use_fast = True)
tokenizer.padding_side = 'right'

PAD = tokenizer.pad_token_id

microbatch_size = 32
accum_batch_size = microbatch_size*4
vocab_size = len(tokenizer)
x_d = 768
x_n_min = 512
max_length = 2048
m_n = 64
layers = 14
num_proc = 8

def tokenize_fn(x):
    return tokenizer(x['text'], max_length = max_length, truncation = True)

def filter_fn(x):
    return len(x['input_ids']) >= x_n_min

toks = ds.map(tokenize_fn, batched = True, remove_columns=["text"], num_proc=num_proc).shuffle(seed=42).filter(filter_fn, num_proc=num_proc)
toks_val = ds_val.map(tokenize_fn, batched = True, remove_columns=["text"], num_proc=num_proc).filter(filter_fn, num_proc=num_proc)

toks_cycle = cycle(toks)

def get_microbatch():
    microbatch = [next(toks_cycle)['input_ids'] for i in range(microbatch_size)]
    microbatch = [torch.tensor(x) for x in microbatch]
    max_len = max([x.shape[0] for x in microbatch])
    microbatch = [torch.pad(x, (0, max_len - x.shape[0]), constant_values=PAD) for x in microbatch]
    microbatch = torch.stack(microbatch)
    return microbatch


num_microbatches = accum_batch_size // microbatch_size
num_epochs = 1
num_batches = num_epochs * len(toks) // accum_batch_size

batch = get_microbatch()

print(batch.shape)