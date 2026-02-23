from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from .config import ParallelConfig


@dataclass
class ParallelTopology:
    world_size: int
    dp_replicate: int
    dp_shard: int
    tp: int
    pp: int
    cp: int
    ep: int

    @classmethod
    def from_config(cls, world_size: int, config: ParallelConfig) -> "ParallelTopology":
        dp_replicate = int(config.dp_replicate)
        dp_shard = int(config.dp_shard)
        tp = int(config.tp)
        pp = int(config.pp)
        cp = int(config.cp)
        ep = int(config.ep)

        base = dp_replicate * tp * pp * cp * ep
        if base < 1:
            raise ValueError("Invalid parallel dimensions: base product must be >= 1")

        if config.infer_dp_shard_from_world_size:
            if world_size % base != 0:
                raise ValueError(
                    f"world_size={world_size} is not divisible by dp_replicate*tp*pp*cp*ep={base}"
                )
            dp_shard = world_size // base

        return cls(
            world_size=world_size,
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            tp=tp,
            pp=pp,
            cp=cp,
            ep=ep,
        )

    def __post_init__(self) -> None:
        dims = (self.dp_replicate, self.dp_shard, self.tp, self.pp, self.cp, self.ep)
        if any(d < 1 for d in dims):
            raise ValueError("All parallel dimensions must be >= 1")

        product = self.dp_replicate * self.dp_shard * self.tp * self.pp * self.cp * self.ep
        if product != self.world_size:
            raise ValueError(
                f"Invalid parallel dimensions: product={product} does not match world_size={self.world_size}"
            )

    @property
    def dp_world_size(self) -> int:
        return self.dp_replicate * self.dp_shard

    @property
    def tp_enabled(self) -> bool:
        return self.tp > 1

    @property
    def pp_enabled(self) -> bool:
        return self.pp > 1

    @property
    def distributed_enabled(self) -> bool:
        return self.world_size > 1

    @property
    def global_rank(self) -> int:
        return int(os.environ.get("RANK", "0"))

    @property
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", "0"))

    def _coordinates(self, rank: int) -> tuple[int, int, int, int, int, int]:
        dims = (self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp, self.ep)
        values: list[int] = []
        x = int(rank)
        for dim in reversed(dims):
            values.append(x % dim)
            x //= dim
        return tuple(reversed(values))  # type: ignore[return-value]

    def _rank_from_coordinates(self, coords: tuple[int, int, int, int, int, int]) -> int:
        pp_rank, dp_replicate_rank, dp_shard_rank, cp_rank, tp_rank, ep_rank = coords
        dims = (self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp, self.ep)
        vals = (pp_rank, dp_replicate_rank, dp_shard_rank, cp_rank, tp_rank, ep_rank)
        rank = 0
        for value, dim in zip(vals, dims, strict=True):
            rank = rank * dim + value
        return rank

    @property
    def dp_rank(self) -> int:
        return self.dp_replicate_rank * self.dp_shard + self.dp_shard_rank

    @property
    def dp_enabled(self) -> bool:
        return self.dp_world_size > 1

    def data_parallel_group_ranks(self, rank: int | None = None) -> list[int]:
        rank = self.global_rank if rank is None else rank
        pp_rank, _, _, cp_rank, tp_rank, ep_rank = self._coordinates(rank)
        ranks: list[int] = []
        for dp_replicate_rank in range(self.dp_replicate):
            for dp_shard_rank in range(self.dp_shard):
                coords = (pp_rank, dp_replicate_rank, dp_shard_rank, cp_rank, tp_rank, ep_rank)
                ranks.append(self._rank_from_coordinates(coords))
        return ranks

    def tensor_parallel_group_ranks(self, rank: int | None = None) -> list[int]:
        rank = self.global_rank if rank is None else rank
        pp_rank, dp_replicate_rank, dp_shard_rank, cp_rank, _, ep_rank = self._coordinates(rank)
        ranks: list[int] = []
        for tp_rank in range(self.tp):
            coords = (pp_rank, dp_replicate_rank, dp_shard_rank, cp_rank, tp_rank, ep_rank)
            ranks.append(self._rank_from_coordinates(coords))
        return ranks

    def iter_all_data_parallel_groups(self) -> Iterable[list[int]]:
        for pp_rank in range(self.pp):
            for cp_rank in range(self.cp):
                for tp_rank in range(self.tp):
                    for ep_rank in range(self.ep):
                        ranks: list[int] = []
                        for dp_replicate_rank in range(self.dp_replicate):
                            for dp_shard_rank in range(self.dp_shard):
                                coords = (
                                    pp_rank,
                                    dp_replicate_rank,
                                    dp_shard_rank,
                                    cp_rank,
                                    tp_rank,
                                    ep_rank,
                                )
                                ranks.append(self._rank_from_coordinates(coords))
                        yield ranks

    @property
    def pp_rank(self) -> int:
        return self._coordinates(self.global_rank)[0]

    @property
    def dp_replicate_rank(self) -> int:
        return self._coordinates(self.global_rank)[1]

    @property
    def dp_shard_rank(self) -> int:
        return self._coordinates(self.global_rank)[2]

    @property
    def tp_rank(self) -> int:
        return self._coordinates(self.global_rank)[4]

    @property
    def is_first_pp_stage(self) -> bool:
        return self.pp_rank == 0

    @property
    def is_last_pp_stage(self) -> bool:
        return self.pp_rank == (self.pp - 1)
