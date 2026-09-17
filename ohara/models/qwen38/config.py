from dataclasses import dataclass


@dataclass
class Config:
    """Small training defaults; ``official()`` describes the released text backbone."""

    vocab_size: int = 256
    hidden_size: int = 128
    num_layers: int = 4
    attention_interval: int = 4
    num_heads: int = 4
    num_kv_heads: int = 1
    head_dim: int = 32
    rotary_dim: int = 16
    rope_theta: float = 10_000_000.0
    linear_key_heads: int = 2
    linear_value_heads: int = 4
    linear_key_dim: int = 32
    linear_value_dim: int = 32
    conv_size: int = 4
    branches: int = 4
    residual_rank: int = 16
    num_experts: int = 8
    top_k: int = 2
    expert_dim: int = 64
    index_heads: int = 4
    index_dim: int = 32
    block_size: int = 4
    token_budget: int = 32
    query_chunk_size: int = 32
    ngram_layer: int = 2  # One-based, as in the released config.
    ngram_vocab: int = 257
    ngram_heads: int = 2  # Heads per n-gram order (bigrams and trigrams).
    ngram_dim: int = 128
    ngram_seed: int = 1234
    eos_token_id: int = 0
    rms_eps: float = 1e-6
    init_std: float = 0.02
    router_aux_coef: float = 0.001
    index_aux_coef: float = 1.0
    backend: str = "auto"  # auto, torch, cuda; cuda requires FLA and Triton.
    activation_checkpointing: bool = False
    mtp_steps: int = 1
    mtp_loss_coef: float = 0.1

    def __post_init__(self):
        positive = (
            "vocab_size", "hidden_size", "num_layers", "attention_interval", "num_heads",
            "num_kv_heads", "head_dim", "linear_key_heads", "linear_value_heads",
            "linear_key_dim", "linear_value_dim", "conv_size", "branches", "residual_rank",
            "num_experts", "top_k", "expert_dim", "index_heads", "index_dim", "block_size",
            "token_budget", "query_chunk_size", "ngram_vocab", "ngram_heads", "ngram_dim",
        )
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.num_heads % self.num_kv_heads or self.linear_value_heads % self.linear_key_heads:
            raise ValueError("query/value head counts must be divisible by their key head counts")
        if self.rotary_dim < 2 or self.rotary_dim % 2 or self.rotary_dim > min(self.head_dim, self.index_dim):
            raise ValueError("rotary_dim must be positive, even, and fit both attention and index heads")
        if self.top_k > self.num_experts:
            raise ValueError("top_k exceeds num_experts")
        if self.token_budget % self.block_size:
            raise ValueError("token_budget must be divisible by block_size")
        if not 1 <= self.ngram_layer <= self.num_layers:
            raise ValueError("ngram_layer must identify an existing layer (one-based)")
        if self.ngram_dim % (2 * self.ngram_heads):
            raise ValueError("ngram_dim must be divisible by twice ngram_heads")
        if not 0 <= self.eos_token_id < self.vocab_size:
            raise ValueError("eos_token_id must be in the vocabulary")
        if self.backend not in {"auto", "torch", "cuda"}:
            raise ValueError("backend must be auto, torch, or cuda")
        if self.rms_eps <= 0 or self.rope_theta <= 0 or self.init_std <= 0:
            raise ValueError("normalization, RoPE, and initialization constants must be positive")
        if self.router_aux_coef < 0 or self.index_aux_coef < 0:
            raise ValueError("auxiliary loss coefficients must be nonnegative")
        if self.mtp_steps < 0 or self.mtp_loss_coef < 0:
            raise ValueError("MTP steps and loss coefficient must be nonnegative")

    @classmethod
    def official(cls, **overrides):
        values = dict(
            vocab_size=248320, hidden_size=2560, num_layers=48, num_heads=24,
            num_kv_heads=2, head_dim=256, rotary_dim=64, linear_key_heads=16,
            linear_value_heads=48, linear_key_dim=128, linear_value_dim=128,
            residual_rank=320, num_experts=512, top_k=10, expert_dim=640,
            index_dim=128, token_budget=2048, ngram_vocab=20_000_000,
            ngram_heads=8, ngram_dim=2560, eos_token_id=248044,
        )
        return cls(**(values | overrides))
