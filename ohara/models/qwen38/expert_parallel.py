"""Expert parallel routing for the Qwen3.8 sparse feed-forward block.

The router and the shared expert are replicated on an expert-parallel group.
Routed expert weights are split by contiguous global expert IDs.  A token's
top-k copies are sorted by destination rank, exchanged with a variable-size
``all_to_all_single``, evaluated by the local experts, and exchanged back.

The collectives are wrapped in an autograd function.  This is important: an
ordinary ``all_to_all_single`` call would make the routed input look detached
and silently lose gradients to the residual stream and to the router weights.
Call :func:`sync_gradients` after ``backward`` and before the optimizer step.
It averages router/shared gradients and normalizes the sharded expert gradients
over the expert group; the latter already include contributions from every
rank through the reverse all-to-all.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from ohara.modules.router import RouterLinear


def _generator(device: torch.device, seed: int) -> torch.Generator | None:
    if device.type == "meta":
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(seed % (1 << 63))
    return generator


def _group_world(group: dist.ProcessGroup | None) -> tuple[int, int]:
    if group is None:
        if not dist.is_available() or not dist.is_initialized():
            return 1, 0
        group = dist.group.WORLD
    return dist.get_world_size(group), dist.get_rank(group)


def _expert_partitions(num_experts: int, world: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return balanced contiguous ``(starts, counts)`` for all ranks."""
    base, remainder = divmod(num_experts, world)
    counts = tuple(base + (rank < remainder) for rank in range(world))
    starts = tuple(sum(counts[:rank]) for rank in range(world))
    return starts, counts


class _AllToAllRows(torch.autograd.Function):
    """Variable-row all-to-all with the inverse collective in backward."""

    @staticmethod
    def forward(ctx, values, send_sizes, recv_sizes, group):
        if values.ndim < 1:
            raise ValueError("all-to-all values must have a row dimension")
        send_sizes = tuple(int(size) for size in send_sizes)
        recv_sizes = tuple(int(size) for size in recv_sizes)
        if values.size(0) != sum(send_sizes):
            raise ValueError("send split sizes do not match the input")
        output = values.new_empty((sum(recv_sizes), *values.shape[1:]))
        dist.all_to_all_single(
            output,
            values.contiguous(),
            output_split_sizes=list(recv_sizes),
            input_split_sizes=list(send_sizes),
            group=group,
        )
        ctx.group = group
        ctx.send_sizes = send_sizes
        ctx.recv_sizes = recv_sizes
        return output

    @staticmethod
    def backward(ctx, gradient):
        restored = gradient.new_empty((sum(ctx.send_sizes), *gradient.shape[1:]))
        dist.all_to_all_single(
            restored,
            gradient.contiguous(),
            output_split_sizes=list(ctx.send_sizes),
            input_split_sizes=list(ctx.recv_sizes),
            group=ctx.group,
        )
        return restored, None, None, None


def _all_to_all_rows(values, send_sizes, recv_sizes, group):
    return _AllToAllRows.apply(values, send_sizes, recv_sizes, group)


def _exchange_sizes(send_sizes: torch.Tensor, group: dist.ProcessGroup) -> tuple[int, ...]:
    recv_sizes = torch.empty_like(send_sizes)
    dist.all_to_all_single(recv_sizes, send_sizes, group=group)
    return tuple(int(size) for size in recv_sizes.tolist())


def _local_dispatch_reference(
    values: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Readable local expert loop; it also handles empty routes."""
    output = torch.zeros_like(values)
    if values.numel() == 0:
        # Empty destinations still have to participate in the autograd graph:
        # their reverse all-to-all carries zero gradients back to source ranks.
        # ``zeros_like(empty)`` alone has no grad_fn in PyTorch.
        anchor = values.sum() * 0
        anchor = anchor + weights.sum() * 0
        anchor = anchor + (w_gate.sum() + w_up.sum() + w_down.sum()) * 0
        return output + anchor
    for expert in range(w_gate.size(0)):
        rows = torch.where(expert_ids == expert)[0]
        if rows.numel() == 0:
            continue
        selected = values.index_select(0, rows)
        hidden = F.silu(selected @ w_gate[expert]) * (selected @ w_up[expert])
        result = hidden @ w_down[expert]
        result = result * weights.index_select(0, rows).to(result.dtype).unsqueeze(-1)
        output.index_add_(0, rows, result.to(output.dtype))
    return output


def _local_dispatch_grouped(
    values: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Run the three grouped GEMMs over this rank's experts."""
    if values.numel() == 0:
        anchor = values.sum() * 0
        anchor = anchor + weights.sum() * 0
        anchor = anchor + (w_gate.sum() + w_up.sum() + w_down.sum()) * 0
        return torch.zeros_like(values) + anchor
    order = torch.argsort(expert_ids, stable=True)
    sorted_ids = expert_ids.index_select(0, order)
    counts = torch.bincount(sorted_ids, minlength=w_gate.size(0))
    offsets = counts.cumsum(0).to(torch.int32)
    sorted_values = values.index_select(0, order)
    compute_dtype = (
        torch.get_autocast_dtype("cuda")
        if torch.is_autocast_enabled("cuda")
        else sorted_values.dtype
    )
    sorted_values = sorted_values.to(compute_dtype)
    gate = torch._grouped_mm(sorted_values, w_gate.to(compute_dtype), offs=offsets)
    up = torch._grouped_mm(sorted_values, w_up.to(compute_dtype), offs=offsets)
    hidden = F.silu(gate) * up
    result = torch._grouped_mm(hidden, w_down.to(compute_dtype), offs=offsets)
    result = result * weights.index_select(0, order).to(result.dtype).unsqueeze(-1)
    output = torch.zeros_like(values)
    output.index_add_(0, order, result.to(output.dtype))
    return output


class ExpertParallelMoE(nn.Module):
    """A Qwen3.8 top-k MoE with expert weights sharded across ranks.

    ``num_experts`` may be unevenly split when the group size does not divide
    it.  The router still produces global expert IDs, while each rank only
    allocates its local contiguous slice.  ``from_dense`` is convenient for a
    parity test or a migration from :class:`ohara.models.qwen38.model.Experts`;
    production construction should instantiate this class directly so a full
    expert tensor is never allocated.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        num_experts_per_tok: int,
        *,
        num_shared_experts: int = 1,
        gate_fn: str = "softmax",
        normalize_weights: bool = True,
        process_group: dist.ProcessGroup | None = None,
        backend: str = "auto",
        init_seed: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if dim < 1 or hidden_dim < 1 or num_experts < 1:
            raise ValueError("expert dimensions and count must be positive")
        if not 1 <= num_experts_per_tok <= num_experts:
            raise ValueError("num_experts_per_tok must be in [1, num_experts]")
        if num_shared_experts not in (0, 1):
            raise ValueError("Qwen3.8 uses zero or one shared expert")
        if gate_fn not in ("softmax", "sigmoid"):
            raise ValueError("gate_fn must be 'softmax' or 'sigmoid'")
        if backend not in {"auto", "torch", "cuda"}:
            raise ValueError("backend must be auto, torch, or cuda")
        if init_std <= 0:
            raise ValueError("init_std must be positive")

        world, rank = _group_world(process_group)
        if world > 1 and process_group is None:
            process_group = dist.group.WORLD
        starts, counts = _expert_partitions(num_experts, world)
        self.process_group = process_group
        self.world_size, self.rank = world, rank
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.num_shared_experts = num_shared_experts
        self.gate_fn = gate_fn
        self.normalize_weights = normalize_weights
        self.backend = backend
        self.init_std = init_std
        # Local expert initialization uses private generators, so uneven shards
        # do not advance the replicated parameter RNG by different amounts.
        self.init_seed = (int(torch.randint(0, 2**31 - 1, (), device="cpu")) * 1_000_003
                          if init_seed is None else int(init_seed))
        self.expert_start = starts[rank]
        self.local_num_experts = counts[rank]
        self.register_buffer("expert_starts", torch.tensor(starts, dtype=torch.long, device=device))
        self.register_buffer(
            "expert_counts", torch.zeros(self.local_num_experts, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer("router_bias", torch.zeros(num_experts, device=device))

        self.w_gate = nn.Parameter(torch.empty(self.local_num_experts, dim, hidden_dim, device=device, dtype=dtype))
        self.w_up = nn.Parameter(torch.empty(self.local_num_experts, dim, hidden_dim, device=device, dtype=dtype))
        self.w_down = nn.Parameter(torch.empty(self.local_num_experts, hidden_dim, dim, device=device, dtype=dtype))
        self.router = RouterLinear(dim, num_experts, bias=False, device=device, dtype=dtype)
        if num_shared_experts:
            self.shared_gate = nn.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
            self.shared_up = nn.Linear(dim, hidden_dim, bias=False, device=device, dtype=dtype)
            self.shared_down = nn.Linear(hidden_dim, dim, bias=False, device=device, dtype=dtype)
            self.shared_output_gate = nn.Linear(dim, 1, bias=False, device=device, dtype=dtype)
        self.reset_parameters()

    @classmethod
    def from_dense(cls, dense: nn.Module, process_group: dist.ProcessGroup | None = None, *, backend: str = "auto"):
        """Create a sharded copy of a dense Qwen38 ``Experts`` module."""
        module = cls(
            dense.dim,
            dense.hidden_dim,
            dense.num_experts,
            dense.num_experts_per_tok,
            num_shared_experts=dense.num_shared_experts,
            gate_fn=dense.gate_fn,
            normalize_weights=dense.normalize_weights,
            process_group=process_group,
            backend=backend,
            device=dense.w_gate.device,
            dtype=dense.w_gate.dtype,
        )
        module.train(dense.training)
        start, stop = module.expert_start, module.expert_start + module.local_num_experts
        with torch.no_grad():
            module.w_gate.copy_(dense.w_gate[start:stop])
            module.w_up.copy_(dense.w_up[start:stop])
            module.w_down.copy_(dense.w_down[start:stop])
            module.router.weight.copy_(dense.router.weight)
            module.router_bias.copy_(dense.router_bias)
            if module.num_shared_experts:
                for name in ("shared_gate", "shared_up", "shared_down", "shared_output_gate"):
                    getattr(module, name).weight.copy_(getattr(dense, name).weight)
        original = dict(dense.named_parameters())
        for name, parameter in module.named_parameters():
            parameter.requires_grad_(original[name].requires_grad)
        return module

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # Initialize replicated tensors before local expert tensors.  Ranks
        # with uneven shard sizes then consume the same RNG stream for these
        # parameters and start with identical router/shared weights.
        nn.init.normal_(
            self.router.weight, mean=0.0, std=self.init_std,
            generator=_generator(self.router.weight.device, self.init_seed),
        )
        self.router_bias.zero_()
        if self.num_shared_experts:
            for offset, parameter in enumerate(
                (self.shared_gate.weight, self.shared_up.weight, self.shared_down.weight), 1
            ):
                nn.init.normal_(
                    parameter, mean=0.0, std=self.init_std,
                    generator=_generator(parameter.device, self.init_seed + offset),
                )
            nn.init.normal_(
                self.shared_output_gate.weight, mean=0.0, std=self.init_std,
                generator=_generator(self.shared_output_gate.weight.device, self.init_seed + 4),
            )
        expert_seed = self.init_seed + 0x9E3779B9 * (self.expert_start + 1)
        for offset, parameter in enumerate((self.w_gate, self.w_up, self.w_down), 5):
            nn.init.normal_(
                parameter, mean=0.0, std=self.init_std,
                generator=_generator(parameter.device, expert_seed + offset),
            )

    def _route(self, values: torch.Tensor):
        logits = self.router(values).float()
        # Qwen3.8 uses the unbiased router for this stage.  ``router_bias`` is
        # retained as a checkpoint-compatible hook for a later balancing pass;
        # applying it here would change expert outputs relative to ``Experts``.
        selected = torch.topk(logits, self.num_experts_per_tok, dim=-1).indices
        selected_logits = logits.gather(-1, selected)
        if self.gate_fn == "softmax":
            weights = selected_logits.softmax(-1)
        else:
            weights = selected_logits.sigmoid()
            if self.normalize_weights and self.num_experts_per_tok > 1:
                weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-9)
        return selected, weights, logits

    def _use_grouped(self, values: torch.Tensor) -> bool:
        if self.backend == "torch" or not values.is_cuda or not hasattr(torch, "_grouped_mm"):
            return False
        compute_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else values.dtype
        )
        if self.backend == "cuda" and compute_dtype != torch.bfloat16:
            raise ValueError("backend=cuda expert parallelism requires BF16")
        return compute_dtype == torch.bfloat16 and self.dim % 8 == 0 and self.hidden_dim % 8 == 0

    def _dispatch(self, values, expert_ids, weights):
        dispatcher = _local_dispatch_grouped if self._use_grouped(values) else _local_dispatch_reference
        return dispatcher(values, expert_ids, weights, self.w_gate, self.w_up, self.w_down)

    def forward(self, values: torch.Tensor, context=None):
        if values.ndim < 2 or values.size(-1) != self.dim:
            raise ValueError(f"values must have shape (..., {self.dim})")
        original_shape = values.shape
        flat = values.reshape(-1, self.dim)
        indices, weights, logits = self._route(flat)

        if self.world_size == 1:
            local_ids = indices.reshape(-1)
            rows = torch.arange(local_ids.numel(), device=flat.device) // self.num_experts_per_tok
            routed = self._dispatch(flat.index_select(0, rows), local_ids, weights.reshape(-1))
            output = torch.zeros_like(flat)
            output.index_add_(0, rows, routed)
        else:
            # ``searchsorted`` uses the same contiguous partition on every rank.
            boundaries = self.expert_starts.to(indices.device)[1:]
            owners = torch.searchsorted(boundaries, indices.reshape(-1), right=True)
            order = torch.argsort(owners, stable=True)
            rows = torch.arange(indices.numel(), device=flat.device) // self.num_experts_per_tok
            send_sizes_tensor = torch.bincount(owners, minlength=self.world_size)
            send_sizes = tuple(int(size) for size in send_sizes_tensor.tolist())
            recv_sizes = _exchange_sizes(send_sizes_tensor, self.process_group)
            sorted_rows = rows.index_select(0, order)
            sorted_ids = indices.reshape(-1).index_select(0, order)
            sorted_values = flat.index_select(0, sorted_rows)
            sorted_weights = weights.reshape(-1).index_select(0, order).to(flat.dtype).unsqueeze(-1)
            received_values = _all_to_all_rows(sorted_values, send_sizes, recv_sizes, self.process_group)
            received_weights = _all_to_all_rows(sorted_weights, send_sizes, recv_sizes, self.process_group).squeeze(-1)
            received_ids = sorted_ids.new_empty(sum(recv_sizes))
            dist.all_to_all_single(
                received_ids,
                (sorted_ids - self.expert_starts.to(sorted_ids.device)[owners.index_select(0, order)]).contiguous(),
                output_split_sizes=list(recv_sizes),
                input_split_sizes=list(send_sizes),
                group=self.process_group,
            )
            local_output = self._dispatch(received_values, received_ids, received_weights)
            returned = _all_to_all_rows(local_output, recv_sizes, send_sizes, self.process_group)
            output = torch.zeros_like(flat)
            output.index_add_(0, sorted_rows, returned)

        if self.num_shared_experts:
            shared = self.shared_down(F.silu(self.shared_gate(flat)) * self.shared_up(flat))
            output = output + shared * self.shared_output_gate(flat).sigmoid()

        if self.training:
            # Count incoming routes, so every owner records all source ranks'
            # traffic.  Counting local source IDs would attribute remote
            # experts to the wrong local slot.
            if self.world_size == 1:
                received_ids = indices.reshape(-1)
            if received_ids.numel():
                self.expert_counts += torch.bincount(
                    received_ids, minlength=self.local_num_experts
                )
        if not self.training:
            return output.reshape(original_shape), flat.new_zeros(())

        if context is not None:
            auxiliary = context.router_loss(
                logits.softmax(-1), indices, values.shape[:2],
            )
        else:
            # Counts are global so this remains meaningful under EP.  The
            # router mass is reduced by sync_replicated_gradients after loss
            # backward, matching the local mean-loss convention.
            global_counts = torch.zeros(self.num_experts, device=flat.device, dtype=torch.float32)
            local_ids = indices.reshape(-1)
            global_counts.index_add_(0, local_ids, torch.ones_like(local_ids, dtype=torch.float32))
            if self.world_size > 1:
                dist.all_reduce(global_counts, group=self.process_group)
            load = global_counts / global_counts.sum().clamp_min(1)
            auxiliary = self.num_experts * (load.detach() * logits.softmax(-1).mean(0)).sum()
        return output.reshape(original_shape), auxiliary

    @torch.no_grad()
    def expert_load(self, reset: bool = True) -> torch.Tensor:
        local = self.expert_counts.float()
        global_load = torch.zeros(self.num_experts, device=local.device)
        global_load[self.expert_start:self.expert_start + self.local_num_experts] = local
        if self.world_size > 1:
            dist.all_reduce(global_load, group=self.process_group)
        if reset:
            self.expert_counts.zero_()
        return global_load.to(torch.long)

    def replicated_parameters(self) -> Iterable[nn.Parameter]:
        yield self.router.weight
        if self.num_shared_experts:
            yield from (
                self.shared_gate.weight,
                self.shared_up.weight,
                self.shared_down.weight,
                self.shared_output_gate.weight,
            )


@torch.no_grad()
def sync_replicated_gradients(
    module: ExpertParallelMoE,
    process_group: dist.ProcessGroup | None = None,
    *,
    average: bool = True,
) -> None:
    """Reduce router/shared gradients over the EP group after ``backward``.

    Expert weights are already sharded and receive all source-rank routes via
    the reverse all-to-all.  Only replicated parameters need this reduction.
    Set ``average=False`` when the surrounding optimizer expects summed grads.
    """
    group = process_group if process_group is not None else module.process_group
    world, _ = _group_world(group)
    if world == 1:
        return
    for parameter in module.replicated_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
        dist.all_reduce(gradient, group=group)
        if average:
            gradient.div_(world)


def _sharded_parameters(module: ExpertParallelMoE) -> Iterable[nn.Parameter]:
    yield module.w_gate
    yield module.w_up
    yield module.w_down


@torch.no_grad()
def normalize_sharded_gradients(
    module: ExpertParallelMoE,
    process_group: dist.ProcessGroup | None = None,
    *,
    average: bool = True,
) -> None:
    """Normalize local expert gradients after all-to-all backward.

    The reverse collective has already summed contributions from all source
    ranks into the owning expert shard.  A mean-loss optimizer therefore needs
    one division by the EP group size, with no second all-reduce.
    """
    if not average:
        return
    group = process_group if process_group is not None else module.process_group
    world, _ = _group_world(group)
    if world == 1:
        return
    for parameter in _sharded_parameters(module):
        if parameter.grad is None:
            continue
        gradient = parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
        gradient.div_(world)


@torch.no_grad()
def sync_gradients(
    module: ExpertParallelMoE,
    process_group: dist.ProcessGroup | None = None,
    *,
    average: bool = True,
) -> None:
    """Prepare all EP gradients for an optimizer step using mean losses."""
    sync_replicated_gradients(module, process_group, average=average)
    normalize_sharded_gradients(module, process_group, average=average)


def expert_parallelize(
    module: nn.Module,
    process_group: dist.ProcessGroup | None = None,
    *,
    backend: str = "auto",
) -> nn.Module:
    """Replace every Qwen38 dense expert block with a sharded block.

    This migration helper is intended for tiny parity tests and checkpoints
    loaded into a dense model.  A full-scale run should construct EP blocks
    from configuration before loading weights, so it never materializes the
    complete expert tensor on every rank.
    """
    from .model import Experts

    replacements = []
    for parent in module.modules():
        for name, child in tuple(parent.named_children()):
            if isinstance(child, Experts):
                replacements.append((parent, name, child))
    for parent, name, dense in replacements:
        setattr(parent, name, ExpertParallelMoE.from_dense(
            dense, process_group, backend=backend,
        ))
    return module


@torch.no_grad()
def sync_model_gradients(module: nn.Module, process_group=None, *, average=True):
    """Reduce replicated model gradients and normalize owner expert gradients."""
    from torch.distributed.tensor import DTensor

    experts = [m for m in module.modules() if isinstance(m, ExpertParallelMoE)]
    if not experts:
        raise ValueError("module contains no ExpertParallelMoE blocks")
    group = experts[0].process_group if process_group is None else process_group
    if any(m.process_group != group for m in experts):
        raise ValueError("all EP blocks must use the supplied process group")
    if any(isinstance(p, DTensor) for p in module.parameters()):
        raise ValueError("EP synchronization does not yet compose with DTensor sharding")
    world, _ = _group_world(group)
    if world == 1:
        return
    shards = {id(p) for m in experts for p in (m.w_gate, m.w_up, m.w_down)}
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        if id(parameter) not in shards:
            dist.all_reduce(parameter.grad, group=group)
        if average:
            parameter.grad.div_(world)
