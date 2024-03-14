import jax
import jax.numpy as np
import equinox as eqx

from jaxtyping import Array
from rope import precompute_freqs_cis, apply_rope
from ffn import SwiGLU
from norm import RMSNorm


class Attention(eqx.Module):
    dim: int
    num_heads: int
    head_dim: int
    seq_len: int
    k: eqx.nn.Linear
    q: eqx.nn.Linear
    v: eqx.nn.Linear
    proj: eqx.nn.Linear
    freqs_cis: Array

    def __init__(self, dim, num_heads, seq_len, *, key):
        super().__init__()
        key_k, key_q, key_v, key_proj = jax.random.split(key, 4)
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.seq_len = seq_len
        self.k = eqx.nn.Linear(dim, dim, key=key_k)
        self.q = eqx.nn.Linear(dim, dim, key=key_q)
        self.v = eqx.nn.Linear(dim, dim, key=key_v)
        self.proj = eqx.nn.Linear(dim, dim, key=key_proj)

        self.freqs_cis = precompute_freqs_cis(
            self.head_dim,
            seq_len * 2,
            dtype=self.k.weight.dtype,
        )

    def __call__(self, x: Array):
        T, C = x.shape
        k = jax.vmap(self.k)(x)
        q = jax.vmap(self.q)(x)
        v = jax.vmap(self.v)(x)

        k = k.reshape(T, self.num_heads, C // self.num_heads)
        q = q.reshape(T, self.num_heads, C // self.num_heads)
        v = v.reshape(T, self.num_heads, C // self.num_heads)

        freqs_cis = np.take(self.freqs_cis, T, axis=0)
        q, k = apply_rope(q, k, freqs_cis=freqs_cis, dtype=self.k.weight.dtype)

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


class LlamaBlock(eqx.Module):
    attention: Attention
    attn_norm: RMSNorm
    mlp: SwiGLU
    mlp_norm: RMSNorm

    def __init__(self, dim, num_heads, seq_len, *, key):
        super().__init__()
        self.attention = Attention(dim, num_heads, seq_len, key=key)
        self.mlp = eqx.nn.Linear(dim, dim, use_bias=False, key=key)
        self.attn_norm = RMSNorm(dim, key=key)
        self.mlp_norm = RMSNorm(dim, key=key)

    def __call__(self, x: Array):
        x = x + self.attention(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LLAMA(eqx.Module):
    dim: int
    num_layers: int
    num_heads: int
    seq_len: int
    vocab_size: int

    embdings: Array
    blocks: tuple[LlamaBlock, ...]
    norm: RMSNorm

    def __init__(self, dim, num_layers, num_heads, seq_len, vocab_size, *, key):
        super().__init__()
        self.emb = jax.random.uniform(
            key,
            shape=(
                vocab_size,
                dim,
            ),
            minval=-1e-4,
            maxval=1e-4,
        )
        self.blocks = tuple(LlamaBlock(dim, num_heads, seq_len, key=key) for _ in range(num_layers))
        self.norm = RMSNorm(dim, key=key)

    def __call__(self, x: Array):
        jax.vmap(lambda x: self.emb[x])(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.emb.T @ x


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    B, T, C = 3, 5, 256
    # model = Attention(dim=C, num_heads=4, seq_len=T, key=key)
    model = Attention(dim=C, num_heads=4, seq_len=T, key=key)
    x = jax.random.normal(key, (B, T, C))

    out = jax.vmap(model)(x)

    print(out.shape)
