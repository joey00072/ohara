import jax
import jax.numpy as jnp
import einops
from jaxtyping import Array


def precompute_freqs_cis(
    dim: int, end: int, theta: float = 10000.0
) -> tuple[Array, Array]:
    freqs = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
    freqs = jnp.einsum("i , j -> i j", jnp.arange(end), freqs)
    freqs_cos = jnp.cos(freqs)  # real part
    freqs_sin = jnp.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def rotate_half(x) -> tuple[Array, Array]:
    x1 = x[:, ::2]
    x2 = x[:, 1::2]
    x = jnp.stack((-x2, x1), axis=-1)
    return einops.rearrange(x, "... d j -> ... (d j)")


def apply_rope(q: Array, k, freq_cis) -> tuple[Array, Array]:
    cos, sin = (einops.repeat(t, "b n -> b (n j)", j=2) for t in freq_cis)
    print(q.shape, k.shape, cos.shape, sin.shape)
    sin_q = sin[-q.shape[0] :, :]
    cos_q = cos[-q.shape[0] :, :]
    print(q.shape, k.shape, cos.shape, sin.shape)

    q = (q * cos_q) + (rotate_half(q) * sin_q)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


if __name__ == "__main__":
    B, T, nh, C = 3, 5, 4, 16

    key = jax.random.PRNGKey(0)
    k = jax.random.normal(key, (B, T, C)).reshape(B, T, nh, C // nh)
    q = jax.random.normal(key, (B, T, C)).reshape(B, T, nh, C // nh)

    freq_cis = precompute_freqs_cis(C // nh, T)

    xq_out, xk_out = apply_rope(k, q, freq_cis)
