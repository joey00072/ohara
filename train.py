import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import lightning as L
from ohara.models.llama import LLAMA, Config
from ohara.lr_scheduler import CosineScheduler
from ohara.dataset import MiniPile
from ohara.utils.info import model_summary


from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import pretty_errors


device = torch.device("mps")
@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    max_iters: int,
    ignore_index: int = -1,
    device=None,
) -> int:
    # fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(max_iters, device=device)
    for idx, (data, target) in enumerate(dataloader):
        if idx >= max_iters:
            break
        logits: torch.Tensor = model(data)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=ignore_index
        )
        losses[idx] = loss.item()
        print(loss)

    model.train()
    return losses.mean()


def train(model, optimizer:optim.Optimizer, train_dataloader, val_dataloader,ignore_index):
    
    model.to(device)
    max_iters=100
# validate(model, val_dataloader, 100)
    for idx, (data, target) in enumerate(val_dataloader):
        data,target = data.to(device),target.to(device)
        if idx >= max_iters:
            break
        logits: torch.Tensor = model(data)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=ignore_index
        )
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



def main():
    learning_rate = 1e-5
    wornup_iters = 5

    tokenizer = AutoTokenizer.from_pretrained("NeelNanda/gpt-neox-tokenizer-digits")
    config = Config(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        seq_len=2048,
        num_layers=2,
        num_heads=4,
    )
    
    print(f"{tokenizer.vocab_size=} CrossEntropy={-math.log(1/tokenizer.vocab_size)}")

    model = LLAMA(config)
    ds = MiniPile(dataset_name="JeanKaddour/minipile",tokenizer=tokenizer,split="test")
    dataloader = DataLoader(ds, batch_size=4)

    print(model)
    print(model_summary(model))
    print(model_summary(model.token_emb))
    # model = torch.compile(model)
    inputs = torch.arange(10).unsqueeze(0)
    get_lr = CosineScheduler(learning_rate=learning_rate)
    optimzer = optim.AdamW(model.parameters())
    train(model,optimzer,train_dataloader=dataloader,val_dataloader=dataloader,ignore_index=tokenizer.pad_token_id)
    tokenizer.name_or_path



if __name__ == "__main__":
    main()
