from __future__ import annotations

import sys
import time
import math
from pathlib import Path
from datetime import datetime

from typing import Any, Dict
from dataclasses import dataclass, field


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

from mixture_of_depth import ModelingMixtureOfDepth,Config

traceback.install()


### CONFIGS
wandb_project_name = "Ohara-MOD"
wandb_run_name = random_name()

hf_username = "joey00072"

learning_rate: float = 5e-4
min_learning_rate: float = 0.0

warmup_iters: int = 1000
max_iters: int = 100_000
batch_size: int = 8
micro_batch: int = 4
eval_iters: int = 100
save_ckpt_iters: int = 1000

multiple_of: int = 4
d_model: int = 128 * 4  # 512
exponantion_factor = 4
hidden_dim = int(d_model * exponantion_factor)
seq_len: int = 256
num_layers: int = 16
num_heads: int = 16
num_kv_heads: int = 4
weight_tying = True
window_size = seq_len

moe_num_experts: int = 4
moe_num_experts_per_tok: int = 2

precision = "32-true"  # "32-true", "16-mixed", "16-true", "bf16-mixed", "bf16-true"
assert d_model % num_heads == 0

compile_model = False

### Dataset and Tokenizer
dataset_name = "JeanKaddour/minipile"  # run pretokeinze first
tokenizer_name = "EleutherAI/gpt-neo-125m"

@dataclass
class ModelType:
    Baseline = {
        "mixture_of_depth": False,
        "name": "Baseline",
    }
    
    Moe = {
        "mixture_of_depth": False,
        "name": "mixture of expert",
        
    }
    
    Mod = {
        "mixture_of_depth": True,
        "name": "mixture of depth",
    }
    
    Mode = {
        "mixture_of_depth": True,
        "name": "mixture of depth and expert",
    }

@dataclass
class ModelSize:
    # by joey00072
    # roughtly same parameters for all models
    Microraptor = {
        "name": "Microraptor",
        "d_model": 128,
    }

    Velosoraptor = {
        "name": "Velosoraptor",
        "d_model": 256,
    }

    Utahraptor = {
        "name": "Utahraptor",
        "d_model": 512,
    }

    TRex = {
        "name": "TRex",
        "d_model": 1024,
    }


def get_params(model_type: str, model_size: ModelSize, vocab_size: int=False,**custom_args):
    """
    This will give you roughly same size of models
    """
    assert vocab_size, "Please provide vocab size"
    params = {
        "model_type": model_type,
        "vocab_size": vocab_size,
        "wandb_project_name": random_name(),
        "learning_rate": learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "max_iters": max_iters,
        "eval_iters": eval_iters,
        "batch_size": batch_size,
        "micro_batch": micro_batch,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "num_heads": num_layers,
        "multiple_of": multiple_of,
        "compile_model": compile_model,
        "tokenizer_name": tokenizer_name,
        "dataset_name": dataset_name,
        "resume_traning": resume_traning,
        "save_ckpt_iters": save_ckpt_iters,
        "weight_tying": weight_tying,
        "precision": precision,
        "mixture_of_expert":False,
        "moe_num_experts": moe_num_experts,
        "moe_num_experts_per_tok": moe_num_experts_per_tok,
        "ignore_index": -1,
    }
    # ik i should use enums but for class ModelSize but...
    params["model_name"] = model_size["name"]
    params["d_model"] = model_size["d_model"]
    params["hidden_dim"] = int(model_size["d_model"] * exponantion_factor)
    if model_type == ModelType.Moe or model_type == ModelType.Mode:
        params["hidden_dim"] = model_size["d_model"] * 1
        params["mixture_of_expert"]= True
    return params


### Setup
device = auto_accelerator()  # auto chose device (cuda, mps)

# for restarting training from last checkout
resume_traning = False


class FabricTrainer:
    def __init__(
        self,
        params: dict[str:Any],
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: optim.Optimizer = None,
        lr_scheduler: CosineScheduler = None,
        accelerator: str | None = None,
        precision: str | None = None,  # "32-true", "16-mixed", "16-true", "bf16-mixed", "bf16-true"
        strategy: str = "auto",
        **kwargs,
    ):
        self.params = params
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.accelerator = accelerator

        loggers = []
        # wandb_logger = WandbLogger(project=params["wandb_project_name"], resume=params["resume_traning"])
        # loggers.append(wandb_logger)
        tensorboard_logger = TensorBoardLogger(root_dir="logs", name=wandb_run_name)
        loggers.append(tensorboard_logger)

        self.fabric: L.Fabric = L.Fabric(strategy=strategy, precision=precision, loggers=loggers)

        self.train_dataloader, self.val_dataloader = self.fabric.setup_dataloaders(
            train_dataloader, val_dataloader
        )
        self.get_lr = lr_scheduler

        self.model = self.fabric.setup(model)

        if params.get("compile_model"):
            self.fabric.print("Compiling model")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        if lr_scheduler is None:
            self.get_lr = CosineScheduler(
                learning_rate=params["learning_rate"],
                min_lr=params["min_learning_rate"],
                warmup_iters=params["warmup_iters"],
                max_iters=params["max_iters"],
            )

        optimzer = optimizer if optimizer else optim.AdamW(model.parameters(), lr=self.get_lr(100))
        self.optimzer = self.fabric.setup_optimizers(optimzer)

    def fit(self):
        self.fabric.logger.log_hyperparams(self.params)
        print(model_summary(self.model))
        print("=" * 100)
        self.train(
            fabric=self.fabric,
            model=self.model,
            optimizer=self.optimzer,
            micro_batch=self.params["micro_batch"],
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            eval_iters=self.params["eval_iters"],
            save_ckpt_iters=self.params["save_ckpt_iters"],
            get_lr=self.get_lr,
            model_name=self.params["wandb_project_name"],
            max_iters=self.params["max_iters"],
            ignore_index=self.params["ignore_index"],
        )
        return self

    @torch.no_grad()
    def validate(
        fabric: L.Fabric,
        model: nn.Module,
        dataloader: DataLoader,
        max_iters: int,
        ignore_index: int = -1,
    ) -> int:
        fabric.barrier()
        model.eval()
        with torch.no_grad():
            losses = torch.zeros(max_iters, device=fabric.device)
            for idx, (data, target) in enumerate(dataloader):
                if idx >= max_iters:
                    break
                logits: torch.Tensor = model(data)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    target.reshape(-1),
                    ignore_index=ignore_index,
                )
                losses[idx] = loss.item()
            val_loss = losses.mean()

        model.train()
        fabric.barrier()
        return val_loss

    @staticmethod
    def train(
        fabric: L.Fabric,
        model: nn.Module,
        optimizer: optim.Optimizer,
        micro_batch: int,
        train_dataloader,
        val_dataloader,
        eval_iters: int,
        save_ckpt_iters: int,
        get_lr,
        max_iters,
        model_name: str = "model",
        ignore_index=-1,
    ):
        fabric.launch()
        ignore_index = ignore_index if ignore_index else -1
        # sanity test
        FabricTrainer.validate(fabric, model, val_dataloader, 5)
        # cyclining loder so you can runit indefinitely
        train_dataloader = BetterCycle(iter(train_dataloader))
        val_dataloader = BetterCycle(iter(val_dataloader))
        (data, target) = next(val_dataloader)
        tokerns_per_iter = int(math.prod(data.shape) * micro_batch)

        micro_batch_loss: float = 0
        idx: int = 0
        while True:
            micro_batch_loss = 0
            if idx >= max_iters:
                break
            idx += 1
            start_time: float = time.perf_counter()

            lr = get_lr(idx)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            # ...
            for _ in range(micro_batch):
                (data, target) = next(train_dataloader)
                with fabric.no_backward_sync(model, enabled=micro_batch == 1):
                    logits: torch.Tensor = model(data)
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        target.view(-1),
                        ignore_index=ignore_index,
                    )
                    loss = loss / micro_batch
                    micro_batch_loss += loss.item()
                    fabric.backward(loss)

            optimizer.step()
            optimizer.zero_grad()

            curr_time: float = time.perf_counter()
            elapsed_time: float = curr_time - start_time
            fabric.print(
                f"iter: {idx} | loss: {micro_batch_loss:.4f} | lr: {lr:e} | time: {elapsed_time:.4f}s"
            )

            if idx % eval_iters == 0:
                val_loss = FabricTrainer.validate(fabric, model, val_dataloader, 100)
                try:
                    fabric.log_dict(
                        {
                            "traning_loss": micro_batch_loss,
                            "test_loss": val_loss,
                            "iter": idx,
                            "tokens": idx * tokerns_per_iter,
                            "lr": lr,
                            "time": elapsed_time,
                        },
                        step=idx,
                    )
                except Exception as e:
                    fabric.print(f"Error logging: {e}")

            if idx % save_ckpt_iters == 0:
                state = {"model": model, "optimizer": optimizer, "idx": idx, "lr": lr}
                fabric.save(f"./ckpt/{model_name}/model.pt", state)


def start_run(model_size, model_type):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    size_pallet = True  # make this false for custom size

    # params = {
    #     "model_type": "Griffin",
    #     "wandb_project_name": random_name(),
    #     "learning_rate": learning_rate,
    #     "min_learning_rate": min_learning_rate,
    #     "warmup_iters": warmup_iters,
    #     "max_iters": max_iters,
    #     "eval_iters": eval_iters,
    #     "batch_size": batch_size,
    #     "micro_batch": micro_batch,
    #     "vocab_size": tokenizer.vocab_size,
    #     "d_model": d_model,
    #     "hidden_dim": hidden_dim,
    #     "seq_len": seq_len,
    #     "num_layers": num_layers,
    #     "num_heads": num_layers,
    #     "multiple_of": multiple_of,
    #     "compile_model": compile_model,
    #     "tokenizer_name": tokenizer_name,
    #     "dataset_name": dataset_name,
    #     "resume_traning": resume_traning,
    #     "save_ckpt_iters": save_ckpt_iters,
    #     "weight_tying": weight_tying,
    #      "precision": "32-true",
    #     "ignore_index": -1,
    # }

    params = get_params(model_type, model_size, vocab_size=tokenizer.vocab_size)
    config = Config(
        model_type=params["model_type"],
        vocab_size=params["vocab_size"],
        d_model=params["d_model"],
        hidden_dim=params["hidden_dim"],
        seq_len=params["seq_len"],
        num_layers=params["num_layers"],
        num_heads=params["num_heads"],
        multiple_of=params["multiple_of"],
        weight_tying=params["weight_tying"],
        mixture_of_expert=params["mixture_of_expert"],
        moe_num_experts = params["moe_num_experts"],
        moe_num_experts_per_tok = params["moe_num_experts_per_tok"],
    )

    # with torch.device("meta"):
    print(model_type)
    model = ModelingMixtureOfDepth(config)
    print(model)
    summary = model_summary(model)

    print("-" * 150)

    train_ds = PreTokenizedDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split="train",
        max_length=seq_len,
    )
    test_ds = PreTokenizedDataset(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split="validation",
        max_length=seq_len,
    )

    train_dataloader = DataLoader(train_ds, batch_size=batch_size)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size)

    get_lr = CosineScheduler(
        learning_rate=learning_rate,
        min_lr=min_learning_rate,
        warmup_iters=warmup_iters,
        max_iters=max_iters,
    )

    optimzer = optim.AdamW(model.parameters(), lr=get_lr(314))

    trainer = FabricTrainer(
        params=params,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        optimizer=optimzer,
        scheduler=get_lr,
        precision=params["precision"],
    )

    trainer.fit()
    name = f"{model_type}-{model_size['name']}-{summary['total_parameters']}"
    model.push_to_hub(str(Path(hf_username).joinpath(name)))
    tokenizer.push_to_hub(str(Path(hf_username).joinpath(name)))

    # # TODO: Replace this inference
    # start: float = time.time()
    # max_tokens = 200
    # model = model.eval()
    # inputs = tokenizer.encode("Once")
    # with torch.no_grad():
    #     inputs = torch.tensor(inputs).reshape(1, -1).to(device).clone().detach()
    #     for _idx in range(max_tokens):
    #         logits = model(inputs)
    #         max_logits = torch.argmax(logits, dim=-1)
    #         inputs: torch.Tensor = torch.cat((inputs, max_logits[:, -1:]), dim=-1)
    #         inputs.shape[1] - 1
    #         print(tokenizer.decode(inputs.tolist()[0][-1]), end="", flush=True)

    # end: float = time.time()

    # print(f"Time: {end-start}")


def main():
    model_size = ModelSize.Microraptor
    model_type = ModelType.Mod
    
    start_run(model_size, model_type)


if __name__ == "__main__":
    main()
