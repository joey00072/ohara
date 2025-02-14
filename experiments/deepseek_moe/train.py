import sys
import time
import math
from datetime import datetime
import os

from typing import Any, Dict, Callable
from dataclasses import dataclass, field


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from ohara.lr_scheduler import CosineScheduler
from ohara.dataset import PreTokenizedDataset
from ohara.utils import BetterCycle

from ohara.utils import auto_accelerator, random_name, model_summary


from torch.utils.data import DataLoader
from transformers import AutoTokenizer


from rich import print, traceback

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.fabric.loggers import TensorBoardLogger

from model import ModelingLM
from dsmoe import Config

from ohara.trainer import Trainer
from dsmoe import FFNType

traceback.install()

torch.set_float32_matmul_precision("high")

# ================================================================================================

### CONFIGS
wandb_project_name = "Ohara-DeepSeek-Moe" + (f"-{sys.argv[1]}" if len(sys.argv) > 1 else "")
wandb_run_name = random_name()

learning_rate: float = 5e-4
min_learning_rate: float = 0.0

max_iters: int = 10_000
warmup_iters: int = max_iters//10 

total_batch_size:int = 2**13 
seq_len: int = 256
batch_size: int = 4
micro_batch: int = int(total_batch_size/(seq_len*batch_size))
eval_iters: int = 100
save_ckpt_iters: int = 2000


d_model: int = 1024 
num_heads: int = 16
head_dim: int = 64


weight_tying: bool = True


##### < MOE >
mlp: str = "glu" # i want to use swiglu but it make params count 1.5x
activation_fn: str = "silu"
multiple_of: int = 1
hidden_dim = int(d_model * multiple_of)
num_layers: int = 8 #// 3  # 44
dense_layers: int = 1 # num_layers
ffn_type: str = FFNType.DSMoE
aux_free_loadbalancing: bool = True
use_aux_loss: bool = not aux_free_loadbalancing
num_experts: int = 8
num_experts_per_tok: int = 2
num_shared_experts: int = 1
expert_update_rate: float = 0.001
train_experts_biases: bool = True   

if len(sys.argv) > 1:
    if sys.argv[1] == "baseline":
        ffn_type = FFNType.Dense
        hidden_dim = int(d_model*8)
        dense_layers = num_layers
    elif sys.argv[1] == "use_aux_loss":
        aux_free_loadbalancing = False
        use_aux_loss = True
    elif sys.argv[1] == "aux_free_loadbalancing":
        aux_free_loadbalancing = True
        use_aux_loss = False

##### </ MOE >


MONKEY_PATCH = False
model_name = f"joey00072/model_name{'Baseline' if MONKEY_PATCH is None else str(MONKEY_PATCH)}"

assert d_model % num_heads == 0

compile_model = False# not bool(sys.platform == "darwin")
compile_mode:str = None # "reduce-overhead"

### Dataset and Tokenizer
dataset_name = "joey00072/pretokenized__JeanKaddour_minipile__EleutherAI__gpt-neo-125m"  # run pretokeinze first
tokenizer_name = "EleutherAI/gpt-neo-125m"

### Setup
device = auto_accelerator()  # auto chose device (cuda, mps)

# for restarting training from last checkpoint
resume_training = False
push_to_hub = False
checkpoint_path = "./ckpt/model.pt"

wandb_logger = True
tensorboard_logger = True

# ================================================================================================


class Trainer:
    def __init__(
        self,
        fabric: L.Fabric,
        model: nn.Module,
        optimizer: optim.Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        get_lr: Callable[[int], float],
        micro_batch: int,
        max_iters: int,
        eval_iters: int,
        save_ckpt_iters: int,
        ignore_index: int = -1,
        push_to_hub: bool = False,
        gamma: float = 1e-3,
        model_name: str = "",
    ):
        self.fabric = fabric
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = BetterCycle(iter(train_dataloader))
        self.val_dataloader = BetterCycle(iter(val_dataloader))
        self.get_lr = get_lr
        self.micro_batch = micro_batch
        self.max_iters = max_iters
        self.eval_iters = eval_iters
        self.save_ckpt_iters = save_ckpt_iters
        self.ignore_index = ignore_index
        self.push_to_hub = push_to_hub
        self.model_name = model_name
        self.gamma = gamma

        (data, target) = next(self.val_dataloader)
        self.tokens_per_iter = int(math.prod(data.shape) * micro_batch)

    @torch.no_grad()
    def calculate_loss(self, dataloader: DataLoader, num_batches: int) -> torch.Tensor:
        self.model.eval()
        losses = torch.zeros(num_batches, device=self.fabric.device)
        for idx, (data, target) in enumerate(dataloader):
            if idx >= num_batches:
                break
            logits, aux_loss, max_violation = self.model(data)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.reshape(-1),
                ignore_index=self.ignore_index,
            )
            losses[idx] = loss.item()
        return losses.mean()

    def log_function(self, idx: int, lr: float, elapsed_time: float, aux_loss: float, max_violation: float) -> None:
        train_loss = self.calculate_loss(self.train_dataloader, 100)
        val_loss = self.calculate_loss(self.val_dataloader, 100)
        
        print(f"iter: {idx} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | lr: {lr:e} | time: {elapsed_time:.4f}s")
        
        try:
            self.fabric.log_dict(
                {
                    "training_loss": train_loss,
                    "validation_loss": val_loss,
                    "iter": idx,
                    "tokens": idx * self.tokens_per_iter,
                    "lr": lr,
                    "time": elapsed_time,
                    "aux_loss": aux_loss,
                    "max_violation": max_violation,
                },
                step=idx,
            )
        except Exception as e:
            print(f"Error logging: {e}")

    def train(self, start_iter: int = 0):
        self.fabric.launch()
        # sanity test
        self.calculate_loss(self.val_dataloader, 5)

        idx: int = start_iter

        self.model.train()
        while True:
            if idx >= self.max_iters:
                break
            idx += 1
            start_time: float = time.perf_counter()

            lr = self.get_lr(idx)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            micro_batch_loss = 0
            micro_batch_aux_loss = 0
            micro_batch_max_violation = 0
            for _ in range(self.micro_batch):
                (data, target) = next(self.train_dataloader)
                with self.fabric.no_backward_sync(self.model, enabled=_ < self.micro_batch - 1):
                    logits: torch.Tensor
                    logits, aux_loss, max_violation = self.model(data)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target.view(-1),
                        ignore_index=self.ignore_index,
                    )+ self.gamma * aux_loss
                    loss = loss / self.micro_batch
                    micro_batch_loss += loss.item()
                    micro_batch_aux_loss += aux_loss if isinstance(aux_loss, float) else aux_loss.item()
                    micro_batch_max_violation += max_violation if isinstance(max_violation, float) else max_violation.item()
                    self.fabric.backward(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            curr_time: float = time.perf_counter()
            elapsed_time: float = curr_time - start_time
            print(
                f"iter: {idx} | loss: {micro_batch_loss:.4f} | aux_loss: {micro_batch_aux_loss:.4f} | max_violation: {micro_batch_max_violation:.4f} | lr: {lr:e} | time: {elapsed_time:.4f}s"
            )
        
            try:
                self.fabric.log_dict(
                    {
                        "iter_idx": idx,
                        "loss": micro_batch_loss/self.micro_batch,
                        "aux_loss": micro_batch_aux_loss/self.micro_batch,
                        "max_violation": micro_batch_max_violation/self.micro_batch,
                    },
                    step=idx,
                )
            except Exception as e:
                print(f"Error logging: {e}")
                
            if idx % self.eval_iters == 0:
                self.model.eval()
                self.log_function(idx, lr, elapsed_time, micro_batch_aux_loss/self.micro_batch, micro_batch_max_violation/self.micro_batch)
                self.model.train()

            if idx % self.save_ckpt_iters == 0:
                state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "idx": idx, "lr": lr}
                self.fabric.save("./ckpt/model.pt", state)
                self.model.config.ckpt_iter = idx
                if self.push_to_hub:
                    self.model.push_to_hub(self.model_name, commit_message=f"checkpoint iter: {idx}")



def main():
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    hyper_params = {
        "vocab_size": tokenizer.vocab_size,
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "max_iters": max_iters,
        "eval_iters": eval_iters,
        "batch_size": batch_size,
        "micro_batch": micro_batch,
        "d_model": d_model,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "hidden_dim": hidden_dim,
        "multiple_of": multiple_of,
        "compile_model": compile_model,
        "tokenizer_name": tokenizer_name,
        "dataset_name": dataset_name,
        "resume_training": resume_training,
        "save_ckpt_iters": save_ckpt_iters,
        "weight_tying": weight_tying,
        "ffn_type": ffn_type,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "num_shared_experts": num_shared_experts,
        "expert_update_rate": expert_update_rate,
        "train_experts_biases": train_experts_biases,
        "aux_free_loadbalancing": aux_free_loadbalancing,
        "use_aux_loss": use_aux_loss,
        "mlp": mlp,
        "activation_fn": activation_fn,
        "dense_layers": dense_layers,
    }
    
    print("="*100)  
    print(hyper_params)
    print("="*100)
    
    
    loggers = []

    if wandb_logger:
        logger = WandbLogger(project=wandb_project_name, resume=resume_training)
        loggers.append(logger)

    if tensorboard_logger:
        logger = TensorBoardLogger(root_dir="./logs", name=wandb_project_name)
        loggers.append(logger)

    # fabric init
    fabric = L.Fabric(loggers=loggers, precision="bf16-mixed")
    fabric.logger.log_hyperparams(hyper_params)

    

    config = Config(**hyper_params)
    
    print(config)

    train_ds = PreTokenizedDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split="train",
        hf=True,
        max_length=seq_len,
    )
    test_ds = PreTokenizedDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split="validation",
        hf=True,
        max_length=seq_len,
    )

    train_dataloader = DataLoader(train_ds, batch_size=batch_size)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)
    train_dataloader, test_dataloader = fabric.setup_dataloaders(train_dataloader, test_dataloader)

    model = ModelingLM(config)

    model: L.LightningModule = fabric.setup(model)
    print("=" * 100)
    if compile_model:
        model = torch.compile(model)
    # model.gradient_checkpointing_enable()
    print(model)
    print(model_summary(model))
    
    # exit()
    import time

    get_lr = CosineScheduler(
        learning_rate=learning_rate,
        min_lr=min_learning_rate,
        warmup_iters=warmup_iters,
        max_iters=max_iters,
    )

    # inputs = torch.tensor(tokenizer.encode("The")).unsqueeze(0).clone().detach()
    optimizer = optim.AdamW(model.parameters(), lr=get_lr(0))
    optimizer = fabric.setup_optimizers(optimizer)

    trainer = Trainer(
        fabric=fabric,
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        get_lr=get_lr,
        micro_batch=micro_batch,
        max_iters=max_iters,
        eval_iters=eval_iters,
        save_ckpt_iters=save_ckpt_iters,
        ignore_index=-1,
        push_to_hub=push_to_hub,
        model_name=model_name,
    )

    start_iter = 0
    if resume_training and os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = fabric.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['idx']
        print(f"Resuming from iteration: {start_iter}")

    # Let's GO!!
    trainer.train(start_iter=start_iter)

    # TODO: Replace this inference
    start: float = time.time()
    max_tokens = 200
    model = model.eval()
    inputs = tokenizer.encode("Once")
    with torch.no_grad():
        inputs = torch.tensor(inputs).reshape(1, -1).to(fabric.device).clone().detach()
        for _idx in range(max_tokens):
            logits = model(inputs)
            max_logits = torch.argmax(logits, dim=-1)
            inputs: torch.Tensor = torch.cat((inputs, max_logits[:, -1:]), dim=-1)
            print(tokenizer.decode(inputs.tolist()[0][-1]), end="", flush=True)

    end: float = time.time()

    print(f"Time: {end-start}")


if __name__ == "__main__":
    main()
