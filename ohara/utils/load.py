from __future__ import annotations

import os

from huggingface_hub import snapshot_download

HF_CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/")


def get_model_path(model_name: str) -> tuple[str | None, str | None]:
    """Locate an already-downloaded snapshot of ``model_name``.

    Returns ``(path, error)``; exactly one of the two is ``None``.
    """
    snapshots = os.path.join(
        HF_CACHE_DIR, "models--" + model_name.replace("/", "--"), "snapshots"
    )
    if not os.path.isdir(snapshots):
        return None, f"{model_name} is not in the local Hugging Face cache"
    revisions = os.listdir(snapshots)
    if not revisions:
        return None, f"{model_name} has no snapshot revisions on disk"
    return os.path.join(snapshots, revisions[0]), None


def download_hf_model(model_name: str) -> str:
    """Download ``model_name`` from the Hub and return the local path."""
    return snapshot_download(model_name)
