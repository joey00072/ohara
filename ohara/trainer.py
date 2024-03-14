import sys
import time
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from ohara.models.llama import LLAMA, Config
from ohara.lr_scheduler import CosineScheduler
from ohara.dataset import PreTokenizedDataset

from ohara.utils import auto_accelerator, random_name, model_summary


from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import wandb
from rich import print, traceback

traceback.install()

# TODO write fabric trainer

class Trainer: 
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: optim.Optimizer,
        scheduler: CosineScheduler,
        device: torch.device,
        logger=None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.logger = logger
        
        # notes
        ## not using lot of self. on purpse. Its easy to copy code and make functional version

    @torch.no_grad()
    def validate(self, max_iters: int, ignore_index: int = -1) -> float:
        
        # type hints
        data:Tensor
        target:Tensor
        logits:Tensor
        
        
        self.model.eval()
        losses = torch.zeros(max_iters, device=self.device)
        
        for idx, (data, target) in enumerate(self.val_dataloader):
            
            if idx >= max_iters:
                break
            
            data, target = data.to(self.device), target.to(self.device)
            logits = self.model(data)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=ignore_index,
            )
            
            losses[idx] = loss.item()
            
        val_loss = losses.mean()
        self.model.train()
        
        return val_loss.item()

    def train(self, micro_batch: int, eval_iters: int, max_iters: int, ignore_index=-1):
        # type hints
        data:Tensor
        target:Tensor
        logits:Tensor
        
        self.model.to(self.device)
        ignore_index = ignore_index if ignore_index else -1
        
        # sanity test
        self.validate(5, ignore_index=ignore_index)

        idx = 0
        while idx < max_iters:
            micro_batch_loss = 0
            
            for _ in range(micro_batch):
                (data, target) = next(iter(self.train_dataloader))
                data, target = data.to(self.device), target.to(self.device)

                logits = self.model(data)
                
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target.view(-1),
                    ignore_index=ignore_index,
                )
                
                loss = loss / micro_batch
                micro_batch_loss += loss.item()
                
                loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            
            lr = self.scheduler(idx)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            if idx % eval_iters == 0:
                val_loss = self.validate(100, ignore_index=ignore_index)
                if self.logger:
                    self.logger.log(
                        {
                            "training_loss": micro_batch_loss,
                            "validation_loss": val_loss,
                            "iter": idx,
                            "tokens":  math.prod(data.shape) * micro_batch,
                        },
                        step=idx,
                    )

            idx += 1
            print(f"Iter: {idx} | Loss: {micro_batch_loss:.4f} | LR: {lr:e}")


def main():
    
    # wandb 
    project_name = "Ohara-LLAMA-Trainer"
    
    # dataset and tokenizer
    pretrained_model = "microsoft/phi-2"
    dataset_name = "roneneldan/TinyStories"
    
    # learning_rate
    learning_rate = 5e-4
    min_lr = 0.0
    
    warmup_iters = 1000 
    max_iters = 100_000 
    eval_iters = 100
    
    # bactch size
    batch_size = 32
    micro_batch = 4
    
    
    # Model Args
    d_model = 128 
    seq_len = 256
    num_layers = 4
    num_heads = 4
    multiple_of = 4
    max_length = 256
    
    # system
    compile_model = not (sys.platform != "darwin")
    device = auto_accelerator() # select accelerator eg cuda, mps
    dtype = torch.float32 # We will use fancy dtypes in future
    
    logger = wandb.init(project=project_name)
    
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    
    config = Config(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        seq_len=seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        multiple_of=multiple_of,
    )
    
    model = LLAMA(config).to(device)
    if compile_model:
        model = torch.compile(model)
    
    print("-"*100)
    print(model)
    print(model_summary(model))
    print("-"*100)
    
    train_ds = PreTokenizedDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split="train",
        max_length=max_length,
    )
    test_ds = PreTokenizedDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split="validation",
        max_length=max_length,
    )
    
    train_dataloader = DataLoader(train_ds, batch_size=batch_size)
    val_dataloader = DataLoader(test_ds, batch_size=batch_size)
    
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = CosineScheduler(
        learning_rate=learning_rate, min_lr=min_lr, warmup_iters=warmup_iters, max_iters=max_iters
    )
    
    trainer = Trainer(
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        scheduler,
        device,
        logger=logger,
    )
    
    
    trainer.train(
        micro_batch=micro_batch,
        eval_iters=eval_iters,
        max_iters=max_iters,
        ignore_index=tokenizer.pad_token_id,
    )
    
    wandb.finish()

if __name__ == "__main__":
    main()
