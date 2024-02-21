import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import time

import lightning as L
# from ohara.models.llama import LLAMA, Config
from ohara.models.phi import Phi,PhiConfig
from ohara.lr_scheduler import CosineScheduler
from ohara.dataset import MiniPile,TinyShakespeareDataset
from ohara.utils.info import model_summary


from lightning.fabric.utilities.throughput import ThroughputMonitor, measure_flops
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ohara.inference import Inference


# import pretty_errors


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
    micro_batch = 5 
# validate(model, val_dataloader, 100)
    micro_batch_loss = 0
    idx =0
    while True:
        micro_batch_loss = 0
        if idx >= max_iters:
                break
            
        for _ in range(micro_batch):
            idx+=1
            (data, target) = next(iter(val_dataloader))
            data,target = data.to(device),target.to(device)
            
            logits: torch.Tensor = model(data)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=ignore_index
            )
            micro_batch_loss += loss.item()
            loss.backward()
        print(f"Iter: {idx}, Loss: {micro_batch_loss/micro_batch}")
        optimizer.step()
        optimizer.zero_grad()



def main():
    learning_rate = 1e-5
    wornup_iters = 5
    batch_size = 32

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-small")
    config = PhiConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=2560//2,
        seq_len=2048,
        num_layers=2,
        num_heads=4,
        multiple_of=1,
        
    )
    
    print(f"{tokenizer.vocab_size=} CrossEntropy={-math.log(1/tokenizer.vocab_size)}")

    model = Phi(config)
    ds = TinyShakespeareDataset(tokenizer=tokenizer)
    dataloader = DataLoader(ds, batch_size=batch_size)

    print(model)
    print(model_summary(model))
    print(model_summary(model.wte))
    # model = torch.compile(model)
    inputs = torch.arange(10).unsqueeze(0)
    get_lr = CosineScheduler(learning_rate=learning_rate)
    optimzer = optim.AdamW(model.parameters())
    train(model,optimzer,train_dataloader=dataloader,val_dataloader=dataloader,ignore_index=tokenizer.pad_token_id)
    
    start: float = time.time()
    max_tokens = 200
    model = model.eval()
    with torch.no_grad():
        inputs = torch.tensor(inputs).reshape(1, -1).to(device)
        print(tokenizer.decode(inputs.tolist()[0]), end="")
        for _idx in range(max_tokens):
            logits = model(inputs)
            max_logits = torch.argmax(logits, dim=-1)
            # print(f"{idx=} {max_logits.shape}")
            inputs: torch.Tensor = torch.cat((inputs, max_logits[:, -1:]), dim=-1)
            input_pos = inputs.shape[1] - 1
            print(tokenizer.decode(inputs.tolist()[0][-1]), end="", flush=True)

    end: float = time.time()

    print(f"Time: {end-start}")
    



if __name__ == "__main__":
    main()
