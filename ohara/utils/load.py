from __future__ import annotations

import os

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

HF_CACHE_DIR = HF_HUB_CACHE


def get_model_path(model_name: str) -> tuple[str | None, str | None]:
    """Locate an already-downloaded snapshot of ``model_name``.

    Returns ``(path, error)``; exactly one of the two is ``None``.
    """
    snapshots = os.path.join(
        HF_CACHE_DIR, "models--" + model_name.replace("/", "--"), "snapshots"
    )
    if not os.path.isdir(snapshots):
        return None, f"{model_name} is not in the local Hugging Face cache"
    main_ref = os.path.join(os.path.dirname(snapshots), "refs", "main")
    if os.path.isfile(main_ref):
        with open(main_ref, encoding="utf-8") as handle:
            revision = handle.read().strip()
        candidate = os.path.join(snapshots, revision)
        if os.path.isdir(candidate):
            return candidate, None
    revisions = sorted(os.listdir(snapshots), key=lambda name: os.path.getmtime(os.path.join(snapshots, name)), reverse=True)
    if not revisions:
        return None, f"{model_name} has no snapshot revisions on disk"
    return os.path.join(snapshots, revisions[0]), None


def download_hf_model(model_name: str) -> str:
    """Download ``model_name`` from the Hub and return the local path."""
    return snapshot_download(model_name)
