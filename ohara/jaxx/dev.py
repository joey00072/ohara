from ramen import *
from transformers import AutoTokenizer
import jax.numpy as np

tokenizer = AutoTokenizer.from_pretrained(
    "NeelNanda/gpt-neox-tokenizer-digits", use_fast=True
)
tokenizer.padding_side = "right"


vocab_size = len(tokenizer)
x_d = 768
x_n_min = 512
x_n_max = 2048
m_n = 64
layers = 1


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))
    freqs = jnp.einsum("i , j -> i j", jnp.arange(end), freqs)
    freqs_cos = np.cos(freqs)  # real part
    freqs_sin = np.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def rotate_half(x):
    x1 = x[:, ::2]
    x2 = x[:, 1::2]

    x = jnp.stack((-x2, x1), axis=-1)

    return rearrange(x, "... d j -> ... (d j)")


def apply_rope(q, k, freq_cis):
    cos, sin = (repeat(t, "b n -> b (n j)", j=2) for t in freq_cis)
    print(q.shape, k.shape, cos.shape, sin.shape)
    sin_q = sin[-q.shape[0] :, :]
    cos_q = cos[-q.shape[0] :, :]
    print(q.shape, k.shape, cos.shape, sin.shape)

    q = (q * cos_q) + (rotate_half(q) * sin_q)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


B, T, C = 3, 5, 64
nh = 4
hs = C // nh
q = jax.random.uniform(random.PRNGKey(0), shape=(T, C))
k = jax.random.uniform(random.PRNGKey(0), shape=(T, C))

q = q.reshape(T, nh, hs)
k = k.reshape(T, nh, hs)

freq_cis = precompute_freqs_cis(dim=hs, end=T)
q1, k1 = apply_rope(q, k, freq_cis)

# exit(0)


import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


q, k = torch.tensor(q.tolist()), torch.tensor(k.tolist())
T, C = q.shape
nh = 4
hs = C // nh
q = q.reshape(1, T, nh, hs)
k = k.reshape(1, T, nh, hs)

freq_cis = precompute_freqs_cis(hs, 5)
q1, k1 = apply_rotary_emb(q, k, freq_cis[0], freq_cis[1])
print(q1.shape)
