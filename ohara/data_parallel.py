from __future__ import annotations


class DataParallelIterableDataset:
    """Engine-bindable data-parallel coordinates for iterable datasets.

    Subclasses must call ``_mark_iterator_started`` when iteration begins and
    use ``data_rank``/``data_world_size`` instead of the global distributed
    coordinates when those values are set.
    """

    data_rank: int | None
    data_world_size: int | None

    def configure_data_parallel(self, rank: int, world_size: int) -> None:
        if world_size < 1:
            raise ValueError("data_world_size must be at least 1")
        if not 0 <= rank < world_size:
            raise ValueError("data_rank must be between 0 and data_world_size - 1")

        current = (
            getattr(self, "data_rank", None),
            getattr(self, "data_world_size", None),
        )
        requested = (rank, world_size)
        if current == requested:
            return
        if current != (None, None):
            raise ValueError(
                f"dataset data topology {current[0]}/{current[1]} does not match "
                f"engine data topology {rank}/{world_size}"
            )
        if getattr(self, "_ohara_iterator_started", False):
            raise RuntimeError("cannot assign data topology after dataset iteration has started")
        self.data_rank = rank
        self.data_world_size = world_size

    def _mark_iterator_started(self) -> None:
        self._ohara_iterator_started = True
