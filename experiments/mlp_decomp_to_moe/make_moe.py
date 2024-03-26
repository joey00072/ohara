from __future__ import annotations

import torch
import os
import numpy as np
from sklearn.preprocessing import normalize
from k_means_constrained import KMeansConstrained

from torch import Tensor
from dataclasses import dataclass


@dataclass
class ModelConfig:
    filename: str
    folder: str
    split_num: int = 8


def load_ffn_weight(filename: str, template: str, layer: int) -> np.ndarray:
    model: dict[str, Tensor] = torch.load(filename, map_location="cpu")
    key = template.format(layer)
    return model[key].numpy()


def split_weights_into_experts(
    ffn_weight: np.ndarray, split_num: int
) -> list[np.ndarray]:
    
    ffn_weight_norm = normalize(ffn_weight)
    kmeans = KMeansConstrained(
        n_clusters=split_num,
        size_min=ffn_weight.shape[0] // split_num,
        size_max=ffn_weight.shape[0] // split_num,
        random_state=0,
    ).fit(ffn_weight_norm)
    
    experts = [ffn_weight[kmeans.labels_ == i] for i in range(split_num)]
    return experts

## TODO: create router