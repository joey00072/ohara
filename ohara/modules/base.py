import torch

from dataclasses import dataclass


@dataclass
class Config:
    vocab_size: int = 51200

    seq_len: int = 2048
    d_model: int = 2048
    intermediate_size = 16 * 2048
    multiple_of: int = 4

    num_heads: int = 32
    num_kv_heads: int = 1

    num_layers: int = 32

    dropout: float = 0.2
    bias: bool = True

    eps: float = 1e-5
    rotary_dim: float = 0.4
