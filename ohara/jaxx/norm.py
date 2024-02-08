import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array


class RMSNorm(eqx.Module):
    eps: float
    weight: Array

    def __init__(self, dim, eps=1e-6, *, key):
        self.eps = eps
        self.weight = jnp.ones(dim)

    def __call__(self, x: Array):
        x = x * (1 / jnp.sqrt(jnp.power(x, 2).mean(-1, keepdims=True) + self.eps))
        return self.weight * x


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    B, T, C = 3, 5, 256
    model = RMSNorm(dim=C, key=key)
    x = jax.random.normal(key, (B, T, C))

    out = jax.vmap(model)(x)

    print(out.shape)
