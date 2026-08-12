"""Human-readable random run names, e.g. ``pikachu-2026_08_12_11_30_02``."""

from __future__ import annotations

import random
from datetime import datetime
from functools import lru_cache
from pathlib import Path

_NAMES_FILE = Path(__file__).parent / "data" / "pokemon_names.txt"


@lru_cache(maxsize=1)
def pokemon_names() -> tuple[str, ...]:
    """The name pool, read from disk once."""
    return tuple(_NAMES_FILE.read_text(encoding="utf-8").split())


def random_name() -> str:
    """A random name suffixed with the current local timestamp."""
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return f"{random.choice(pokemon_names())}-{stamp}"
