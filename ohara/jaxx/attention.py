from typing import Any
import jax
import jax.numpy as np
import equinox as eqx
import pretty_errors


from jaxtyping import Array


import jax.numpy as np


class Attention(eqx.Module):
    dim: int
    num_heads: int
    seq_len: int
    k: eqx.nn.Linear
    q: eqx.nn.Linear
    v: eqx.nn.Linear
    proj: eqx.nn.Linear

    def __init__(self, dim, num_heads, seq_len, *, key):
        super().__init__()
        key_k, key_q, key_v, key_proj = jax.random.split(key, 4)
        self.dim = dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.k = eqx.nn.Linear(dim, dim, key=key_k)
        self.q = eqx.nn.Linear(dim, dim, key=key_q)
        self.v = eqx.nn.Linear(dim, dim, key=key_v)
        self.proj = eqx.nn.Linear(dim, dim, key=key_proj)

    def __call__(self, x):
        T, C = x.shape
        k = jax.vmap(self.k)(x)
        q = jax.vmap(self.q)(x)
        v = jax.vmap(self.v)(x)

        k = k.reshape(T, self.num_heads, C // self.num_heads)
        q = q.reshape(T, self.num_heads, C // self.num_heads)
        v = v.reshape(T, self.num_heads, C // self.num_heads)

        k = k.swapaxes(0, 1)
        q = q.swapaxes(0, 1)
        v = v.swapaxes(0, 1)

        mask = np.tril(np.ones((T, T))).reshape((1, T, T))

        attn = q @ k.swapaxes(-2, -1)
        attn = attn * C**-0.5

        attn = np.where(mask == 0, float("-inf"), attn)
        attn = jax.nn.softmax(attn, axis=-1)

        out: Array = attn @ v
        out = out.swapaxes(0, 1).reshape(T, C)

        out = jax.vmap(self.proj)(out)
        return out


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    B, T, C = 3, 5, 256
    model = Attention(dim=C, num_heads=4, seq_len=T, key=key)
    x = jax.random.normal(key, (B, T, C))

    out = jax.vmap(model)(x)

    print(out.shape)
