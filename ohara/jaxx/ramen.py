import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
from equinox import nn
from typing import List, Optional, Union, Literal
from jaxtyping import Array
from einops import repeat, rearrange
self_path = "RAMEN-Ab.ipynb"

def bin_assoc(e_i, e_j):
    a_i, b_i = e_i
    a_j, b_j = e_j
    return a_j * a_i, a_j * b_i + b_j

def fixed_pos_embedding(xshape):
    dim = xshape[-1]
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))

    sinusoid_inp = jnp.einsum("i , j -> i j", jnp.arange(xshape[0]), inv_freq)

    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)

def rotate_half(x):
    x1 = x[:, ::2]
    x2 = x[:, 1::2]

    x = jnp.stack((-x2, x1), axis=-1)

    return rearrange(x, "... d j -> ... (d j)")


def apply_rope(q, k, sincos):
    sin, cos = map(lambda t: repeat(t, "b n -> b (n j)", j=2), sincos)

    sin_q = sin[-q.shape[0]:, :]
    cos_q = cos[-q.shape[0]:, :]
    
    q = (q * cos_q) + (rotate_half(q) * sin_q)
    k = (k * cos) + (rotate_half(k) * sin)

    return q, k


def ff_dim(h_d):
    d = int(h_d * (2/3))
    return 256 * round(d / 256)

class ScaleNorm(eqx.Module):
    g: Array
    
    def __init__(self, d):
        self.g = jnp.array(jnp.sqrt(d))

    def __call__(self, x):
        norm_x = jnp.linalg.norm(x, axis=-1, ord=2, keepdims=True)
        return x * (self.g / (norm_x + 1e-8))

class FF(eqx.Module):
    in1: nn.Linear
    in2: nn.Linear
    out: nn.Linear

    def __init__(self, x_d, h_d, *, key):
        key_in1, key_in2, key_out = random.split(key, num=3)

        d = ff_dim(h_d)

        self.in1 = nn.Linear(x_d, d, use_bias=False, key=key_in1)
        self.in2 = nn.Linear(x_d, d, use_bias=False, key=key_in2)
    
        self.out = nn.Linear(d, x_d, use_bias=False, key=key_out)

    def __call__(self, xs):        
        h1 = jax.nn.silu(jax.vmap(self.in1)(xs))
        h2 = jax.vmap(self.in2)(xs)

        return jax.vmap(self.out)(h1 * h2)

class TopFF(eqx.Module):
    in1: nn.Linear
    in2: nn.Linear
    mid1: nn.Linear
    mid2: nn.Linear
    out: nn.Linear

    def __init__(self, x_d, h_d, *, key):
        key_in1, key_in2, key_mid1, key_mid2, key_out = random.split(key, num=5)

        d = ff_dim(h_d)

        self.in1 = nn.Linear(x_d, d, use_bias=False, key=key_in1)
        self.in2 = nn.Linear(x_d, d, use_bias=False, key=key_in2)

        self.mid1 = nn.Linear(d, d, use_bias=False, key=key_mid1)
        self.mid2 = nn.Linear(d, d, use_bias=False, key=key_mid2)
    
        self.out = nn.Linear(d, x_d, use_bias=False, key=key_out)

    def __call__(self, xs):        
        h1 = jax.nn.silu(jax.vmap(self.in1)(xs))
        h2 = jax.vmap(self.in2)(xs)

        hs = h1 * h2

        h1 = jax.nn.silu(jax.vmap(self.mid1)(hs))
        h2 = jax.vmap(self.mid2)(hs)

        return jax.vmap(self.out)(h1 * h2)

class AttnLayer(eqx.Module):
    q: nn.Linear
    k: nn.Linear
    v: nn.Linear
    o: nn.Linear
    r: nn.Linear

    # loosely based on https://github.com/lucidrains/local-attention-flax/blob/main/local_attention_flax/local_attention_flax.py

    x_d: int
    chunk_size: int

    def __init__(self, x_d, *, reduce=8, chunk_size=64, key):
        key_q, key_k, key_v, key_o, key_r  = random.split(key, num=5)

        self.x_d = x_d
        self.chunk_size = chunk_size

        self.q = nn.Linear(x_d, x_d // reduce, use_bias=False, key=key_q)
        self.k = nn.Linear(x_d // reduce, x_d // reduce, use_bias=False, key=key_k)
        self.v = nn.Linear(x_d, x_d, use_bias=False, key=key_v)

        self.o = nn.Linear(x_d, x_d, use_bias=False, key=key_o)
        self.r = nn.Linear(x_d, x_d, use_bias=False, key=key_r)

    def __call__(self, xs):
        qs = jax.vmap(self.q)(xs)
        ks = jax.vmap(self.k)(qs)
        vs = jax.vmap(self.v)(xs)

        qs, ks, vs = map(lambda x: rearrange(x, '(c n) d -> c n d', n=self.chunk_size), (qs, ks, vs))

        # Add overlap
        ks, vs = map(lambda x: jnp.pad(x, ((1, 0), (0, 0), (0, 0)), constant_values=0), (ks, vs))
        ks, vs = map(lambda x: jnp.concatenate([x[:-1], x[1:]], axis=1), (ks, vs))
        
        # print ks vs shape
        print("ks shape:", ks.shape)
        print("vs shape:", vs.shape)
        exit(0)
        sincos = fixed_pos_embedding(ks.shape[1:])
        pos_emb = jax.vmap(apply_rope, in_axes=(0, 0, None))

        qs, ks = pos_emb(qs, ks, sincos)

        qk = jnp.einsum('c n k, c m k -> c n m', qs, ks) / jnp.sqrt(qs.shape[-1])
        mask = jnp.tril(jnp.ones((self.chunk_size, self.chunk_size * 2), dtype=xs.dtype), self.chunk_size)[None, :, :]
        A = jax.nn.softmax(qk + (mask == 0) * float('-inf'))

        ys = jnp.einsum('c n m, c m v -> c n v', A, vs)

        ys = rearrange(ys, 'c n v -> (c n) v')

        return jax.vmap(self.o)(ys) * jax.nn.sigmoid(jax.vmap(self.r)(xs))

class MemLayer(eqx.Module):
    f: nn.Linear
    q: nn.Linear
    k: nn.Linear
    v: nn.Linear
    beta: Array
    decay: Array
    l: Array
    o: nn.Linear
    r: nn.Linear
    
    m_n: int
    x_d: int
    m_d: int
    decay_d: int

    def __init__(self, x_d, m_n, *, reduce=8, key):
        key_f, key_q, key_k, key_v, key_l, key_o, key_r = random.split(key, num=7)

        m_d = x_d // reduce
        
        self.m_d = m_d
        self.x_d = x_d
        self.m_n = m_n
        self.decay_d = self.m_d // 4
        
        self.f = nn.Linear(x_d, m_n, use_bias=False, key=key_f)
        
        self.q = nn.Linear(x_d, m_d, use_bias=False, key=key_q)
        self.k = nn.Linear(m_d, m_d, use_bias=False, key=key_k)
        self.v = nn.Linear(x_d, m_d, use_bias=False, key=key_v)
    
        self.decay = 3*jnp.ones((m_n, self.decay_d))
        self.beta = -jnp.ones((m_n, m_d,))

        self.l = random.uniform(key_l, shape=(m_n, m_d,))

        self.o = nn.Linear(m_d, x_d, use_bias=False, key=key_o)
        self.r = nn.Linear(x_d, x_d, use_bias=False, key=key_r)
    
    def __call__(self, xs):
        vs = jax.vmap(self.v)(xs) # (x_n, m_d)

        fs = jax.nn.sigmoid(jax.vmap(self.f)(xs))[:, :, None] # (x_n, m_n, 1)

        beta = jax.nn.sigmoid(self.beta)[None, :, :]
        decay = jax.nn.sigmoid(self.decay)
        decay_pad = jnp.ones((self.m_n, self.m_d - self.decay_d,), dtype=xs.dtype)
        gamma = jnp.concatenate([decay_pad, decay], axis=1)[None, :, :]

        ws = (1 - fs*beta)*gamma
        
        fvs = fs*vs[:, None, :]

        # mem = (1 - f*beta)*gamma*mem_prev + f*v
        _, mems = lax.associative_scan(bin_assoc, (ws, fvs))

        qs = jax.vmap(self.q)(xs)
        ks = jax.vmap(jax.vmap(self.k))(mems) + self.l[None, :, :]

        A = jax.nn.softmax(jnp.einsum('n m k, n k -> n m ', ks, qs) / jnp.sqrt(ks.shape[-1]))
    
        ys = jnp.einsum('n m, n m d -> n d', A, mems)

        return jax.vmap(self.o)(ys) * jax.nn.sigmoid(jax.vmap(self.r)(xs))

class RAMENLayer(eqx.Module):
    sn1: ScaleNorm
    sn2: ScaleNorm
    attn: AttnLayer
    mem: MemLayer
    ff: FF
    
    def __init__(self, x_d, m_n, factor=2, *, key):
        key_attn, key_mem, key_ff = random.split(key, num=3)

        self.sn1 = ScaleNorm(x_d)
        self.sn2 = ScaleNorm(x_d)
        self.attn = AttnLayer(x_d, key=key_attn)
        self.mem = MemLayer(x_d, m_n, key=key_mem)
        self.ff = FF(x_d, x_d*factor, key=key_ff)
    
    def __call__(self, xs):
        xn = self.sn1(xs)

        xs = self.attn(xn) + self.mem(xn) + xs
        xs = self.ff(self.sn2(xs)) + xs
        return xs
        
class RAMEN(eqx.Module):
    layers: List[RAMENLayer]
    top: TopFF
    sn: ScaleNorm

    def __init__(self, layers, x_d, m_n, *, key):
        key_layers, key_top = random.split(key, num=2)
        keys = random.split(key_layers, num=layers)

        self.layers = [RAMENLayer(x_d, m_n, key=k) for k in keys]
        self.top = TopFF(x_d, x_d*4, key=key_top)
        self.sn = ScaleNorm(x_d)
 
    def __call__(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        
        xs = self.top(self.sn(xs)) + xs
        
        return xs

class RAMENModel(eqx.Module):
    emb: Array
    ramen: RAMEN
    sn: ScaleNorm

    def __init__(self, vocab_size, layers, x_d, m_n, *, key):
        key_embed, key_ramen, key_out = random.split(key, num=3)
        
        self.emb = random.uniform(key, shape=(vocab_size, x_d,), minval=-1e-4, maxval=1e-4)
        self.ramen = RAMEN(layers, x_d, m_n, key=key_ramen)
        self.sn = ScaleNorm(x_d)
        self.emb = nn.Embedding(vocab_size, x_d, use_bias=False, key=key_embed)
    
    def __call__(self, seq):
        emb = self.emb / (1e-8 + jnp.linalg.norm(self.emb, axis=-1, ord=2, keepdims=True))
        
        xs = jax.vmap(lambda x: emb[x])(seq)

        xs = self.ramen(xs)
        xs = self.sn(xs)
        
        logits = jax.vmap(lambda x: x @ emb.T)(xs) # Tied word embeddings
        
        return logits