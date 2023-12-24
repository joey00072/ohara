import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np

torch.manual_seed(2)

def sliding_window_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, window_size: int = 16):
    B, T, C = q.shape
    out = torch.zeros(B, T, C, device=q.device)
    for i in range(T):
        start = max(0, i - window_size + 1)
        end = min(T, i + 1)
        key = k[:, start:end, :]
        query = q[:, i, :].unsqueeze(1)
        value = v[:, start:end, :]

        # Compute scaled dot-product attention
        wei = query @ key.transpose(-1, -2)
        wei = F.softmax(wei, dim=-1)  # Apply softmax to weights
        out[:, i, :] = (wei @ value).squeeze(1)  # Multiply weights by values

    return out

def sliding_window_attention_with_mask(q:Tensor,k:Tensor,v:Tensor,window_size:int=16):
    B,T,C = q.shape
    out = torch.zeros(T,C)
    mask = torch.zeros(T,T)
    for i in range(T):
        for j in range(T):
            if j>i:
                mask[i,j] = float("-inf")
            if i-j>window_size-1:
                mask[i,j] = float("-inf")

                
    wei = q@k.transpose(-1,-2)
    wei = wei + mask
    wei = F.softmax(wei,dim=-1)
    
    out = wei @ v
    return out

B,T,C = 5,7,8

q = torch.rand(B,T,C)
k = torch.rand(B,T,C)
v = torch.rand(B,T,C)

window_size = 3

# out = sliding_window_attention(q,k,v,window_size=window_size)
# print(out.shape)

out = sliding_window_attention_with_mask(q,k,v,window_size=window_size)
print(out)

exit(0)

import jax.numpy as jnp
from jax.nn import softmax
from jax import vmap
import jax
from jax import lax

def sliding_window_attention_jax(q, k, v, window_size=16):
    T, C = q.shape
    out = jnp.zeros((T, C))
    max_slice_size = window_size  # Set the maximum slice size

    def attention_step(i, out):
        start = lax.max(0, i - window_size + 1)
        end = lax.min(T, i + 1)
        slice_size = end - start

        # Use a fixed-size slice with masking
        k_slice = lax.dynamic_slice(k, (start, 0), (max_slice_size, C))
        v_slice = lax.dynamic_slice(v, (start, 0), (max_slice_size, C))

        # Create a mask for valid positions
        valid_mask = jnp.arange(max_slice_size) < slice_size

        # Apply the mask
        k_slice_masked = k_slice * valid_mask[:, None]
        v_slice_masked = v_slice * valid_mask[:, None]

        weights = softmax(jnp.dot(k_slice_masked, q[i]), axis=-1).reshape(-1, 1)
        weighted_sum = jnp.sum(weights * v_slice_masked, axis=0)
        return out.at[i].set(weighted_sum)

    out = lax.fori_loop(0, T, attention_step, out)
    return out

# Dummy input tensors for testing the function
B,T, C =10, 50, 128  # Example dimensions
q = jnp.ones((B,T, C))  # Query tensor
k = jnp.ones((B,T, C))  # Key tensor
v = jnp.ones((B,T, C))  # Value tensor

# Call the JAX function with the dummy inputs
output_jax = jax.vmap(sliding_window_attention_jax)(q, k, v)
print(output_jax.shape ) # Display the shape of the output for verification
