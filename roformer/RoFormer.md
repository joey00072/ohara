# RoFormer

Name:"RoFormer: Enhanced Transformer with Rotary Position Embedding"

Source: https://arxiv.org/abs/2104.09864

---


### What's new rotatry embedding 


Make block of two apply rotary rotation matrix on it.
[[dot product]] between two vector signify relative distance 



# Breakdown

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

```

![](./mtx.png)

$$

\Theta = \{\theta_{i} = 10000^{-2(i-1)/d}, i \in [1,2,\dots,d/2]\}


$$
```python
freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

# 2(i-1)/d =
# 2*torch.arange(0,dim//2) = torch.arange(0, dim, 2)
# [: (dim // 2)] trancation if odd num of dim
# x**-2 = 1/x**2
```