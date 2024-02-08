from __future__ import annotations

import os

# Load model directly
from transformers import AutoModelForCausalLM

model_name = "microsoft/phi-2"


def get_model_path(model_name: str) -> tuple[str, str | None]:
    CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub/")
    model_name = os.path.join("models--" + model_name.replace("/", "--"), "snapshots")
    path = os.path.join(CACHE_DIR, model_name)
    if not os.path.exists(path):
        return None, "Not exits"
    return os.path.join(path, os.listdir(path)[0]), None


def download_hf_model(model_name: str) -> str:
    """
    download model from huggingface and return path of saved model
    """
    path, err = get_model_path(model_name)
    print(path, err)
    AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    path, err = get_model_path(model_name)
    return path


if __name__ == "__main__":
    print(download_hf_model(model_name))
