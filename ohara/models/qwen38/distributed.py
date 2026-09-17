"""FSDP2 backbone and owner-routed n-gram embeddings.

Embedding requests and their gradients travel to the rank owning each row.
The table is never all-gathered. Its DTensor parameter also gives DCP the global
layout needed to save and restore all shards.
"""

import math

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard

from .model import Experts, GatedDeltaNet, Qwen38
from .ngram import hash_multipliers, prime_sizes
from .ops import RMSNorm


class _Lookup(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, ids, group, start, rows_per_rank):
        world = dist.get_world_size(group)
        flat = ids.reshape(-1)
        owners = flat.div(rows_per_rank, rounding_mode="floor")
        order = owners.argsort(stable=True)
        send_counts = torch.bincount(owners, minlength=world)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=group)
        send_sizes, recv_sizes = send_counts.tolist(), recv_counts.tolist()
        received = flat.new_empty(sum(recv_sizes))
        dist.all_to_all_single(received, flat[order], recv_sizes, send_sizes, group=group)
        local_ids = received - start
        values = weight[local_ids]
        returned = weight.new_empty((flat.numel(), weight.size(1)))
        dist.all_to_all_single(returned, values.contiguous(), send_sizes, recv_sizes, group=group)
        result = torch.empty_like(returned)
        result[order] = returned
        ctx.save_for_backward(local_ids, order)
        ctx.group, ctx.send_sizes, ctx.recv_sizes = group, send_sizes, recv_sizes
        ctx.weight_shape, ctx.world = weight.shape, world
        return result.reshape(*ids.shape, weight.size(1))

    @staticmethod
    def backward(ctx, gradient):
        local_ids, order = ctx.saved_tensors
        gradient = gradient.reshape(-1, gradient.size(-1))[order].contiguous()
        received = gradient.new_empty((sum(ctx.recv_sizes), gradient.size(-1)))
        dist.all_to_all_single(received, gradient, ctx.recv_sizes, ctx.send_sizes, group=ctx.group)
        result = gradient.new_zeros(ctx.weight_shape)
        # FSDP averages backbone gradients across data ranks. Apply the same
        # convention to the embedding gradients routed from those ranks.
        result.index_add_(0, local_ids, received / ctx.world)
        return result, None, None, None, None


class RowShardedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, mesh: DeviceMesh, *, device="meta"):
        super().__init__()
        self.mesh = mesh
        self.num_embeddings, self.embedding_dim = num_embeddings, embedding_dim
        self.rows_per_rank = math.ceil(num_embeddings / mesh.size())
        self.start = self.rows_per_rank * mesh.get_local_rank()
        rows = max(0, min(self.rows_per_rank, num_embeddings - self.start))
        local = torch.empty(rows, embedding_dim, device=device)
        self.weight = nn.Parameter(DTensor.from_local(
            local, mesh, [Shard(0)], run_check=False,
            shape=torch.Size((num_embeddings, embedding_dim)), stride=(embedding_dim, 1),
        ))

    def forward(self, indices):
        return _Lookup.apply(
            self.weight.to_local(), indices, self.mesh.get_group(), self.start, self.rows_per_rank,
        )


def build_sharded(cfg, mesh, device, *, mixed_precision=True):
    """Materialize only parameter shards; construction never allocates a full model."""
    with torch.device("meta"):
        model = Qwen38(cfg)
        embedding = model.ngram.embedding
        model.ngram.embedding = RowShardedEmbedding(
            embedding.num_embeddings, embedding.embedding_dim, mesh,
        )
    ignored = {model.ngram.embedding.weight}
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16 if mixed_precision else None,
        reduce_dtype=torch.float32,
    )
    for layer in model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    if model.mtp is not None:
        fully_shard(model.mtp, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy, ignored_params=ignored)
    model.to_empty(device=device)
    if torch.device(device).type == "cpu":
        # CPU DTensor RNG has no shard offsets. Keep its reference shards
        # independent without changing the caller's random stream.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(torch.initial_seed() + mesh.get_local_rank())
            _initialize_shards(model, cfg, device)
    else:
        _initialize_shards(model, cfg, device)
    return model


def _initialize_shards(model, cfg, device):
    model.apply(model._initialize)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, RMSNorm):
                module.weight.zero_()
            if isinstance(module, GatedDeltaNet):
                module.A_log.uniform_(0.01, 16).log_()
                module.dt_bias.fill_(1)
        for module in model.modules():
            if isinstance(module, Experts):
                for weight in (module.w_gate, module.w_up, module.w_down):
                    nn.init.normal_(weight, std=cfg.init_std)
                module.router_bias.zero_()
                module.expert_counts.zero_()
        nn.init.normal_(model.ngram.embedding.weight, std=cfg.init_std)
        sizes = torch.tensor(prime_sizes(cfg.ngram_vocab, 2 * cfg.ngram_heads), device=device)
        model.ngram.sizes.copy_(sizes)
        model.ngram.offsets.copy_(sizes.cumsum(0) - sizes)
        model.ngram.multipliers.copy_(hash_multipliers(cfg.vocab_size, cfg.ngram_seed).to(device))


@torch.no_grad()
def clip_grad_norm(model, max_norm, group=None):
    """Global norm for a model whose parameters are each sharded exactly once."""
    gradients = [p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
                 for p in model.parameters() if p.grad is not None]
    total = torch.stack([g.float().square().sum() for g in gradients]).sum()
    dist.all_reduce(total, group=group)
    norm = total.sqrt()
    coefficient = (max_norm / (norm + 1e-6)).clamp(max=1)
    torch._foreach_mul_(gradients, coefficient)
    return norm


@torch.no_grad()
def clip_tensor_parallel_grad_norm(model, max_norm, mesh):
    """Count TP shards once and replicated parameters once in the global norm."""
    gradients, squares = [], []
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        sharded = isinstance(parameter, DTensor)
        if sharded and parameter.device_mesh != mesh:
            raise ValueError("TP clipping requires parameters on the supplied one-dimensional mesh")
        gradient = parameter.grad.to_local() if sharded else parameter.grad
        gradients.append(gradient)
        is_shard = sharded and any(isinstance(p, Shard) for p in parameter.placements)
        squares.append(gradient.float().square().sum() / (1 if is_shard else mesh.size()))
    total = torch.stack(squares).sum()
    dist.all_reduce(total, group=mesh.get_group())
    norm = total.sqrt()
    torch._foreach_mul_(gradients, (max_norm / (norm + 1e-6)).clamp(max=1))
    return norm


def tensor_parallel_param_groups(model):
    """Keep Tensor and DTensor parameters in separate foreach optimizer groups."""
    parameters = list(model.parameters())
    return [{"params": group} for group in (
        [p for p in parameters if isinstance(p, DTensor)],
        [p for p in parameters if not isinstance(p, DTensor)],
    ) if group]


@torch.no_grad()
def clip_expert_parallel_grad_norm(model, max_norm, group=None):
    """Clip after EP gradient synchronization, counting owner shards once."""
    from .expert_parallel import ExpertParallelMoE

    experts = [m for m in model.modules() if isinstance(m, ExpertParallelMoE)]
    if not experts:
        raise ValueError("model contains no expert-parallel blocks")
    group = experts[0].process_group if group is None else group
    world = dist.get_world_size(group) if dist.is_initialized() else 1
    sharded = {id(p) for m in experts for p in (m.w_gate, m.w_up, m.w_down)}
    gradients, squares = [], []
    for parameter in model.parameters():
        if isinstance(parameter, DTensor):
            raise ValueError("EP clipping does not yet compose with DTensor sharding")
        if parameter.grad is not None:
            gradients.append(parameter.grad)
            copies = 1 if id(parameter) in sharded else world
            squares.append(parameter.grad.float().square().sum() / copies)
    total = torch.stack(squares).sum()
    if world > 1:
        dist.all_reduce(total, group=group)
    norm = total.sqrt()
    torch._foreach_mul_(gradients, (max_norm / (norm + 1e-6)).clamp(max=1))
    return norm
