import os
from transformers import AutoTokenizer
from tqdm import tqdm
from itertools import cycle
from datasets import load_dataset

import torch
from torch.nn.functional import pad

CACHE_DIR = "hf_cache"

ds = load_dataset("JeanKaddour/minipile", split="train", cache_dir=CACHE_DIR)
ds_val = load_dataset("JeanKaddour/minipile", split="validation", cache_dir=CACHE_DIR)

tokenizer = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits", cache_dir=CACHE_DIR, use_fast = True)
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
num_proc = os.cpu_count() // 2

def tokenize_fn(x):
    return tokenizer(x['text'], max_length = max_length, truncation = True)

def filter_fn(x):
    return len(x['input_ids']) >= x_n_min

toks = ds.map(tokenize_fn, batched = True, remove_columns=["text"], num_proc=num_proc).shuffle(seed=42).filter(filter_fn, num_proc=num_proc)
toks_val = ds_val.map(tokenize_fn, batched = True, remove_columns=["text"], num_proc=num_proc).filter(filter_fn, num_proc=num_proc)

toks_cycle = cycle(toks)


def get_microbatch():
    microbatch = [next(toks_cycle)['input_ids'] for i in range(microbatch_size)]
    microbatch = [torch.tensor(x, dtype=torch.long) for x in microbatch]  # Ensure tensors are long type for IDs
    max_len = max(x.shape[0] for x in microbatch)
    
    # Compute padding for each tensor and apply padding
    microbatch = [pad(x, (0, max_len - x.shape[0]), 'constant', value=PAD) for x in microbatch]
    
    microbatch = torch.stack(microbatch)
    return microbatch



num_microbatches = accum_batch_size // microbatch_size
num_epochs = 1
num_batches = num_epochs * len(toks) // accum_batch_size

batch = get_microbatch()

print(batch.shape)
mask = torch.where(batch != PAD, torch.tensor(1), torch.tensor(0))
print(batch)
print(mask)



# class ExpDataset(torch.utils.data.Dataset):
#     def __init__(self, tokenizer: AutoTokenizer=None):
#         self.tokenizer = tokenizer if tokenizer else  AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits", cache_dir=CACHE_DIR, use_fast = True)
#         self.tokenizer.padding_side = 'right'
#         self.length = len(self.tokenizer)
#         self.PAD = tokenizer.pad_token_id
        
#         microbatch_size = 32
#         accum_batch_size = microbatch_size*4
#         vocab_size = len(tokenizer)
#         x_d = 768
#         x_n_min = 512
#         max_length = 2048
#         m_n = 64
#         layers = 14
#         num_proc = os.cpu_count() // 2

#     def __len__(self):
#         return self.length

#     def __getitem__(self, idx):
#         return self.toks[idx]