"""Qwen3.8 text-backbone + MTP training, with an allocation-free full-size plan.

    uv run python examples/train_qwen38.py --preset official --describe
    uv run python examples/train_qwen38.py --synthetic --steps 2 --backend torch

See docs/qwen38.md for token bins, FSDP2, kernel dependencies, and checkpoints.
"""

import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.utils.data import DataLoader

from ohara.data_resume import capture_input_state, restore_input_state
from ohara.models.qwen38 import Config, Qwen38
from ohara.models.qwen38.context_parallel import ContextParallel
from ohara.models.qwen38.distributed import build_sharded, clip_grad_norm
from ohara.models.qwen38.model import GatedResidual
from ohara.tokenbin import TokenBinDataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("tiny", "official"), default="tiny")
    parser.add_argument("--config", type=Path, help="JSON overrides for the selected preset")
    parser.add_argument("--describe", action="store_true", help="print parameter and memory counts without allocating weights")
    parser.add_argument("--plan-world-size", type=int, default=64, help="rank count for the allocation-free memory estimate")
    parser.add_argument("--train-bin", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="random-token smoke test; not a model-quality benchmark")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1, help="context ranks per training example")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--loss-chunk-size", type=int, default=128)
    parser.add_argument("--backend", choices=("auto", "torch", "cuda"), default="auto")
    parser.add_argument("--compile", action="store_true", help="compile gated residual reads and writes")
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--resume", type=Path, help="completed step directory written by this script")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class TrainingState:
    def __init__(self, model, optimizer):
        from ohara.models.qwen38.expert_parallel import ExpertParallelMoE

        if any(isinstance(module, ExpertParallelMoE) for module in model.modules()):
            raise ValueError("EP uses rank-local checkpoints; its local expert tensors cannot use this DCP wrapper")
        self.model, self.optimizer = model, optimizer

    def state_dict(self):
        model, optimizer = get_state_dict(self.model, self.optimizer)
        return {"model": model, "optimizer": optimizer}

    def load_state_dict(self, state):
        set_state_dict(self.model, self.optimizer,
                       model_state_dict=state["model"], optim_state_dict=state["optimizer"])


def contract(args, cfg, world):
    return {
        "config": asdict(cfg), "world_size": world, "sequence_length": args.seq_len,
        "micro_batch_size": args.micro_batch_size, "grad_accum_steps": args.grad_accum_steps,
        "cp_size": args.cp_size,
        "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip, "seed": args.seed, "synthetic": args.synthetic,
        "train_bin": str(args.train_bin.resolve()) if args.train_bin else None,
    }


def input_layout(recipe, rank):
    return dict(gradient_accumulation_steps=recipe["grad_accum_steps"],
                data_rank=rank // recipe["cp_size"], data_world_size=recipe["world_size"] // recipe["cp_size"])


def save_checkpoint(path, model, optimizer, loader, step, recipe, rank, distributed):
    # Never overwrite a completed checkpoint with partially saved new shards.
    exists = torch.tensor(int(path.exists()), device=next(model.parameters()).device)
    if distributed:
        dist.all_reduce(exists, op=dist.ReduceOp.MAX)
    if exists.item():
        raise FileExistsError(f"checkpoint directory already exists: {path}")
    path.mkdir(parents=True, exist_ok=True)
    dcp.save({"training": TrainingState(model, optimizer)}, checkpoint_id=path / "shards")
    local = {"rng": torch.get_rng_state(),
             "data": capture_input_state(loader, **input_layout(recipe, rank)) if loader is not None else None}
    if torch.cuda.is_available():
        local["cuda_rng"] = torch.cuda.get_rng_state()
    torch.save(local, path / f"rank-{rank}.pt")
    if distributed:
        dist.barrier()
    if rank == 0:
        (path / "complete.json").write_text(json.dumps({"step": step, "recipe": recipe}, indent=2) + "\n")
    if distributed:
        dist.barrier()


def resume_checkpoint(path, model, optimizer, loader, recipe, rank):
    metadata = json.loads((path / "complete.json").read_text())
    if metadata["recipe"] != recipe:
        raise ValueError("checkpoint data layout or training recipe differs from this run")
    dcp.load({"training": TrainingState(model, optimizer)}, checkpoint_id=path / "shards")
    local = torch.load(path / f"rank-{rank}.pt", map_location="cpu", weights_only=False)
    if loader is not None:
        restore_input_state(loader, local["data"], **input_layout(recipe, rank))
    torch.set_rng_state(local["rng"])
    if "cuda_rng" in local:
        torch.cuda.set_rng_state(local["cuda_rng"])
    return metadata["step"]


def main():
    args = parse_args()
    for name in ("seq_len", "micro_batch_size", "grad_accum_steps", "cp_size", "steps", "loss_chunk_size", "save_every", "plan_world_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.seq_len < 2 or args.learning_rate <= 0 or args.grad_clip <= 0 or args.weight_decay < 0:
        raise ValueError("invalid sequence length or optimizer settings")
    overrides = json.loads(args.config.read_text()) if args.config else {}
    overrides["backend"] = args.backend
    if args.preset == "official":
        overrides.setdefault("activation_checkpointing", True)
    cfg = Config.official(**overrides) if args.preset == "official" else Config(**overrides)
    if args.describe:
        with torch.device("meta"):
            model = Qwen38(cfg)
        total = sum(p.numel() for p in model.parameters())
        ngram = model.ngram.embedding.weight.numel()
        mtp = sum(p.numel() for p in model.mtp.parameters()) if model.mtp else 0
        print(json.dumps({
            "config": asdict(cfg), "parameters": total, "ngram_parameters": ngram,
            "backbone_parameters": total - ngram - mtp, "mtp_parameters": mtp,
            "fp32_adam_state_gib_per_rank": total * 16 / args.plan_world_size / 2**30,
            "memory_note": "Estimate includes FP32 parameters, gradients and Adam moments; excludes activations, gathered layers, workspaces and communication buffers.",
        }, indent=2))
        return
    if bool(args.train_bin) == args.synthetic:
        raise ValueError("choose exactly one of --train-bin or --synthetic")
    if args.preset == "official" and (not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) < 2):
        raise ValueError("official-size training requires a multi-GPU torchrun launch; use --describe without GPUs")

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world % args.cp_size:
        raise ValueError("cp-size must divide WORLD_SIZE")
    if args.cp_size > 1 and args.seq_len % (args.cp_size * cfg.block_size):
        raise ValueError("seq-len must be divisible by cp-size * block_size")
    if args.cp_size > 1 and args.backend != "torch" and args.micro_batch_size != 1:
        raise ValueError("FLA context parallelism requires micro-batch-size=1")
    distributed = world > 1
    rank = int(os.environ.get("RANK", "0"))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0"))) if torch.cuda.is_available() else torch.device("cpu")
    if distributed and device.type != "cuda":
        raise ValueError("FSDP2 training requires CUDA; CPU smoke tests use one process")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl")
    try:
        torch.manual_seed(args.seed)
        if distributed:
            mesh = init_device_mesh("cuda", (world,))
            model = build_sharded(cfg, mesh, device)
        else:
            model = Qwen38(cfg).to(device)
        context = None
        if args.cp_size > 1:
            cp_mesh = init_device_mesh("cuda", (world // args.cp_size, args.cp_size), mesh_dim_names=("dp", "cp"))
            context = ContextParallel(cp_mesh["cp"].get_group())
        if args.compile:
            for module in model.modules():
                if isinstance(module, GatedResidual):
                    module.compile()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                      weight_decay=args.weight_decay, fused=device.type == "cuda")
        dataset = None
        if args.train_bin:
            dataset = TokenBinDataset(args.train_bin, max_length=args.seq_len, seed=args.seed,
                                      data_rank=rank // args.cp_size, data_world_size=world // args.cp_size)
            if int(dataset.metadata["vocab_size"]) > cfg.vocab_size:
                raise ValueError("token-bin vocabulary exceeds the model vocabulary")
            dataset.validate_capacity(0)
        data_loader = (DataLoader(dataset, batch_size=args.micro_batch_size, num_workers=0,
                                  pin_memory=device.type == "cuda",
                                  generator=torch.Generator().manual_seed(args.seed))
                       if dataset is not None else None)
        recipe = contract(args, cfg, world)
        step = 0
        # Distinct synthetic batches per data rank, reproducible on resume.
        torch.manual_seed(args.seed + rank // args.cp_size)
        if args.resume:
            step = resume_checkpoint(args.resume, model, optimizer, data_loader, recipe, rank)
        loader = iter(data_loader) if data_loader is not None else None
        model.train()
        while step < args.steps:
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            mean_loss = torch.zeros((), device=device)
            for _ in range(args.grad_accum_steps):
                if loader is None:
                    tokens = torch.randint(cfg.vocab_size, (args.micro_batch_size, args.seq_len + 1), device=device)
                    inputs, labels = tokens[:, :-1], tokens[:, 1:]
                else:
                    inputs, labels = (x.to(device, non_blocking=True) for x in next(loader))
                if context is not None:
                    inputs, labels = (x.chunk(context.size, dim=1)[context.rank] for x in (inputs, labels))
                autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                with autocast:
                    output = model(inputs, labels, loss_chunk_size=args.loss_chunk_size, context=context)
                loss = output.loss / args.grad_accum_steps
                loss.backward()
                mean_loss += loss.detach()
            norm = (clip_grad_norm(model, args.grad_clip) if distributed
                    else torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
            if not bool(torch.isfinite(norm)):
                raise RuntimeError("non-finite gradient norm")
            optimizer.step()
            step += 1
            if distributed:
                dist.all_reduce(mean_loss)
                mean_loss /= world
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            tokens_per_step = world // args.cp_size * args.micro_batch_size * args.seq_len * args.grad_accum_steps
            if rank == 0:
                print(f"step={step} loss={mean_loss.item():.4f} seconds={elapsed:.3f} tokens/s={tokens_per_step / elapsed:,.0f}", flush=True)
            if args.checkpoint_dir and (step % args.save_every == 0 or step == args.steps):
                save_checkpoint(args.checkpoint_dir / f"step-{step:08d}", model, optimizer,
                                data_loader, step, recipe, rank, distributed)
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
