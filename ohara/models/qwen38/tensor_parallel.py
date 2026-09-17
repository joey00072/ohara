"""Small tensor parallel building blocks for the Qwen38 example.

The Qwen38 expert hidden dimension is the useful first TP boundary: gate and
up projections keep a slice of the expert hidden dimension on each rank, and
the down projection sums those slices. Inputs and router/shared weights are
replicated within the TP group. The module deliberately keeps the collective
boundaries explicit so it can later be nested under FSDP2.

This is model parallelism, not data parallelism. Every rank in mesh must run
the same input batch and use a one-dimensional TP mesh. The training loss
can be backpropagated once per rank without a TP divide; only an external sum
of replicated scalar losses needs one normalization. The all-reduce backward
paths then produce the same gradients as a single full model.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.nn import functional as F

from ohara.modules.router import RouterLinear


def _grouped_mm_available() -> bool:
    return hasattr(torch, "_grouped_mm")


def _mesh_info(mesh: DeviceMesh) -> tuple[int, int, object]:
    if mesh.ndim != 1:
        raise ValueError("Qwen38 tensor parallelism needs a one-dimensional mesh")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("initialize torch.distributed before constructing TP modules")
    world = mesh.size()
    if world < 1:
        raise ValueError("tensor parallel mesh must contain at least one rank")
    return world, mesh.get_local_rank(), mesh.get_group()


def _require_divisible(size: int, world: int, name: str) -> None:
    if size % world:
        raise ValueError(f"{name}={size} must be divisible by tensor parallel degree={world}")


def _global_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    # DTensor needs the global stride even though only the local slice is
    # allocated. A meta tensor computes it without reserving the full weight.
    return torch.empty(shape, device="meta").stride()


@torch.no_grad()
def _normal_(value: torch.Tensor, std: float, seed: int) -> None:
    if value.is_meta:
        return
    generator = torch.Generator(device=value.device)
    generator.manual_seed(seed)
    value.normal_(0.0, std, generator=generator)


def _sharded_parameter(
    shape: tuple[int, ...],
    mesh: DeviceMesh,
    dim: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None,
    init_std: float,
    init_seed: int,
) -> nn.Parameter:
    world, rank, _ = _mesh_info(mesh)
    _require_divisible(shape[dim], world, f"weight dimension {dim}")
    local_shape = list(shape)
    local_shape[dim] //= world
    local = torch.empty(tuple(local_shape), device=device, dtype=dtype)
    value = DTensor.from_local(
        local,
        mesh,
        [Shard(dim)],
        run_check=False,
        shape=torch.Size(shape),
        stride=_global_stride(shape),
    )
    parameter = nn.Parameter(value)
    # Keep parameter streams disjoint across ranks. A plain seed + rank would
    # make rank 1's gate stream collide with rank 0's up stream.
    _normal_(local, init_std, init_seed + rank * 1_000_003)
    return parameter


def _local_slice(full: torch.Tensor, mesh: DeviceMesh, dim: int) -> torch.Tensor:
    world, rank, _ = _mesh_info(mesh)
    _require_divisible(full.size(dim), world, f"weight dimension {dim}")
    size = full.size(dim) // world
    return full.narrow(dim, rank * size, size).contiguous()


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, group):
        ctx.group = group
        result = value.contiguous().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=group)
        return result

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        # The row-parallel output is replicated, so every TP rank receives the
        # same loss gradient. Its local input is already a distinct shard;
        # reducing this gradient would multiply row weight/input gradients.
        return gradient, None


class _ReplicatedGradient(torch.autograd.Function):
    """Copy to each TP rank; sum gradients when the source is replicated."""

    @staticmethod
    def forward(ctx, value: torch.Tensor, group):
        ctx.group = group
        return value

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        result = gradient.contiguous().clone()
        dist.all_reduce(result, op=dist.ReduceOp.SUM, group=ctx.group)
        return result, None


def _all_reduce_sum(value: torch.Tensor, group, world: int) -> torch.Tensor:
    return value if world == 1 else _AllReduceSum.apply(value, group)


def _replicated_gradient(value: torch.Tensor, group, world: int) -> torch.Tensor:
    return value if world == 1 else _ReplicatedGradient.apply(value, group)


class TensorParallelExperts(nn.Module):
    """Qwen38-style routed experts sharded over the expert hidden dimension.

    Expert count stays replicated, which keeps routing local and avoids an
    expert-to-expert all-to-all. Each rank stores [E, D, H/tp] gate/up weights
    and [E, H/tp, D] down weights. The routed result is summed across the TP
    group. The optional shared expert and router are replicated; their
    parameter and input gradients use explicit reductions.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        mesh: DeviceMesh,
        *,
        num_shared_experts: int = 1,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        init_std: float = 0.02,
        init_seed: int | None = None,
    ) -> None:
        super().__init__()
        if min(dim, hidden_dim, num_experts, top_k) < 1:
            raise ValueError("expert dimensions and top_k must be positive")
        if top_k > num_experts:
            raise ValueError("top_k exceeds num_experts")
        if num_shared_experts < 0:
            raise ValueError("num_shared_experts cannot be negative")
        if init_std <= 0:
            raise ValueError("init_std must be positive")
        if init_seed is None:
            init_seed = int(torch.randint(0, 2**31 - 1, (), device="cpu")) * 1_000_003
        world, _, group = _mesh_info(mesh)
        _require_divisible(hidden_dim, world, "hidden_dim")
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_shared_experts = num_shared_experts
        self.mesh = mesh
        self.world_size = world
        self._tp_group = group
        local_hidden = hidden_dim // world
        self.w_gate = _sharded_parameter(
            (num_experts, dim, hidden_dim),
            mesh,
            2,
            device=device,
            dtype=dtype,
            init_std=init_std,
            init_seed=init_seed,
        )
        self.w_up = _sharded_parameter(
            (num_experts, dim, hidden_dim),
            mesh,
            2,
            device=device,
            dtype=dtype,
            init_std=init_std,
            init_seed=init_seed + 1,
        )
        self.w_down = _sharded_parameter(
            (num_experts, hidden_dim, dim),
            mesh,
            1,
            device=device,
            dtype=dtype,
            init_std=init_std,
            init_seed=init_seed + 2,
        )
        self.router = RouterLinear(dim, num_experts, bias=False, device=device, dtype=dtype)
        _normal_(self.router.weight, init_std, init_seed + 3)
        if num_shared_experts:
            shared_dim = num_shared_experts * hidden_dim
            self.shared_gate = nn.Linear(dim, shared_dim, bias=False, device=device, dtype=dtype)
            self.shared_up = nn.Linear(dim, shared_dim, bias=False, device=device, dtype=dtype)
            self.shared_down = nn.Linear(shared_dim, dim, bias=False, device=device, dtype=dtype)
            self.shared_output_gate = nn.Linear(dim, 1, bias=False, device=device, dtype=dtype)
            for index, module in enumerate((
                self.shared_gate, self.shared_up, self.shared_down, self.shared_output_gate
            )):
                _normal_(module.weight, init_std, init_seed + 4 + index)
        else:
            self.shared_gate = self.shared_up = self.shared_down = self.shared_output_gate = None
        self.local_hidden_dim = local_hidden

    @torch.no_grad()
    def load_full_state_dict(self, state: Mapping[str, torch.Tensor], *, prefix: str = "") -> None:
        """Load ordinary full tensors, slicing only at the TP boundary.

        This is useful when a single-rank checkpoint initializes a TP
        experiment. Production checkpoint writes should use the DTensor state
        dict/DCP path so no rank gathers full expert weights.
        """
        for name, parameter in (
            ("w_gate", self.w_gate),
            ("w_up", self.w_up),
            ("w_down", self.w_down),
        ):
            full = state[prefix + name]
            expected = tuple(parameter.shape)
            if tuple(full.shape) != expected:
                raise ValueError(f"{prefix + name} shape {tuple(full.shape)} != {expected}")
            dim = parameter.placements[0].dim
            parameter.to_local().copy_(
                _local_slice(full, self.mesh, dim).to(parameter.device, parameter.dtype)
            )
        for name, module in (
            ("router", self.router),
            ("shared_gate", self.shared_gate),
            ("shared_up", self.shared_up),
            ("shared_down", self.shared_down),
            ("shared_output_gate", self.shared_output_gate),
        ):
            if module is None:
                continue
            full = state[prefix + name + ".weight"]
            if tuple(full.shape) != tuple(module.weight.shape):
                raise ValueError(f"{prefix + name}.weight has an incompatible shape")
            module.weight.copy_(full.to(module.weight.device, module.weight.dtype))

    def _dispatch(
        self,
        flat: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        local_gate = self.w_gate.to_local()
        local_up = self.w_up.to_local()
        local_down = self.w_down.to_local()
        compute_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else flat.dtype
        )
        if (
            flat.is_cuda
            and _grouped_mm_available()
            and compute_dtype == torch.bfloat16
            and self.dim % 8 == 0
            and self.local_hidden_dim % 8 == 0
        ):
            return self._dispatch_grouped(
                flat, expert_indices, expert_weights, local_gate, local_up, local_down
            )
        output = flat.new_zeros(flat.shape)
        for expert in range(self.num_experts):
            selected = expert_indices == expert
            token_rows, choices = torch.where(selected)
            if token_rows.numel() == 0:
                continue
            values = flat[token_rows]
            gate = values @ local_gate[expert]
            up = values @ local_up[expert]
            hidden = F.silu(gate) * up
            values = hidden @ local_down[expert]
            values = values * expert_weights[token_rows, choices].unsqueeze(-1).to(values.dtype)
            output.index_add_(0, token_rows, values)
        return output

    @staticmethod
    @torch.compiler.disable
    def _dispatch_grouped(
        flat: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        local_gate: torch.Tensor,
        local_up: torch.Tensor,
        local_down: torch.Tensor,
    ) -> torch.Tensor:
        """Use three grouped GEMMs for aligned CUDA BF16 expert slices."""
        top_k = expert_indices.size(-1)
        flat_indices = expert_indices.reshape(-1)
        order = torch.argsort(flat_indices, stable=True)
        rows = order // top_k
        offsets = torch.bincount(
            flat_indices, minlength=local_gate.size(0)
        ).cumsum(0).to(torch.int32)
        compute_dtype = (
            torch.get_autocast_dtype("cuda")
            if torch.is_autocast_enabled("cuda")
            else flat.dtype
        )
        values = flat[rows].to(compute_dtype)
        gate = torch._grouped_mm(values, local_gate.to(compute_dtype), offs=offsets)
        up = torch._grouped_mm(values, local_up.to(compute_dtype), offs=offsets)
        hidden = F.silu(gate) * up
        values = torch._grouped_mm(hidden, local_down.to(compute_dtype), offs=offsets)
        values = values * expert_weights.reshape(-1)[order].to(values.dtype).unsqueeze(-1)
        output = torch.zeros_like(flat)
        output.index_add_(0, rows, values.to(output.dtype))
        return output

    def forward(
        self,
        value: torch.Tensor,
        context=None,
        *,
        expert_indices: torch.Tensor | None = None,
        expert_weights: torch.Tensor | None = None,
        return_routing: bool = False,
    ):
        if value.size(-1) != self.dim:
            raise ValueError(f"experts expect input width {self.dim}")
        flat = value.reshape(-1, self.dim)
        logits = self.router(flat)
        probabilities = logits.float().softmax(-1)
        if (expert_indices is None) != (expert_weights is None):
            raise ValueError("expert_indices and expert_weights must be supplied together")
        if expert_indices is None:
            selected = logits.topk(self.top_k, dim=-1).indices
            weights = logits.gather(-1, selected).softmax(-1)
        else:
            selected, weights = expert_indices, expert_weights
            if selected.shape != (flat.size(0), self.top_k) or weights.shape != selected.shape:
                raise ValueError("expert routing must have shape (tokens, top_k)")
            selected = selected.to(device=flat.device, dtype=torch.long)
            weights = weights.to(device=flat.device)
        # Each expert hidden slice contributes a distinct input gradient. The
        # router and shared expert see the original replicated input and must
        # keep their ordinary per-rank gradients.
        routed_input = _replicated_gradient(flat, self._tp_group, self.world_size)
        routed_weights = _replicated_gradient(weights, self._tp_group, self.world_size)
        routed = self._dispatch(routed_input, selected, routed_weights)
        routed = _all_reduce_sum(routed, self._tp_group, self.world_size)
        if self.shared_gate is not None:
            shared = self.shared_down(F.silu(self.shared_gate(flat)) * self.shared_up(flat))
            shared = shared * self.shared_output_gate(flat).sigmoid()
            routed = routed + shared
        if not self.training:
            auxiliary = value.new_zeros(())
        elif context is None:
            load = torch.bincount(selected.reshape(-1), minlength=self.num_experts).float()
            load = load / selected.numel()
            auxiliary = self.num_experts * (load * probabilities.mean(0)).sum()
        else:
            if not hasattr(context, "router_loss"):
                raise TypeError("TP expert context must provide router_loss")
            auxiliary = context.router_loss(probabilities, selected, value.shape[:2])
        output = routed.reshape_as(value)
        if return_routing:
            return output, auxiliary, selected, weights
        return output, auxiliary


def tensor_parallelize(model, mesh):
    """Replace backbone and MTP experts for a small-model TP experiment."""
    from .model import Experts

    replacements = [(parent, name, child) for parent in model.modules()
                    for name, child in parent.named_children() if isinstance(child, Experts)]
    for parent, name, old in replacements:
        if isinstance(old.w_gate, DTensor):
            raise ValueError("this TP conversion expects an unsharded model")
        new = TensorParallelExperts(
            old.dim, old.hidden_dim, old.num_experts, old.num_experts_per_tok, mesh,
            num_shared_experts=old.num_shared_experts, device=old.w_gate.device,
            dtype=old.w_gate.dtype,
        )
        new.load_full_state_dict(old.state_dict())
        new.train(old.training)
        original = dict(old.named_parameters())
        for key, parameter in new.named_parameters():
            parameter.requires_grad_(original[key].requires_grad)
        setattr(parent, name, new)
    return model
