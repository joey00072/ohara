from typing import Any
import jax
import jax.numpy as np
import equinox as eqx
import pretty_errors

from jaxtyping import Array
import jax.numpy as np


class SwiGLU(eqx.Module):
    dim: int
    hidden_dim: int
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear

    def __init__(self, dim, hidden, *, key):
        super().__init__()
        key_w1, key_w2, key_w3 = jax.random.split(key, 3)
        self.dim = dim
        self.hidden_dim = hidden
        self.w1 = eqx.nn.Linear(dim, hidden, key=key_w1)
        self.w2 = eqx.nn.Linear(
            hidden, dim, key=key_w2
        )  # this is order in meta's llama
        self.w3 = eqx.nn.Linear(dim, hidden, key=key_w3)

    def __call__(self, x):
        x = jax.nn.silu(jax.vmap(self.w1)(x)) * jax.vmap(self.w3)(x)
        x = jax.vmap(self.w2)(x)
        return x


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    B, T, C = 3, 5, 256
    model = SwiGLU(dim=C, hidden=C * 4, key=key)
    x = jax.random.normal(key, (B, T, C))

    out = jax.vmap(model)(x)

    print(out.shape)
