import torch



# rotary embedding 
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) 
    # [: (dim // 2)] for odd number truncation
    # torch.arange(0, dim, 2) -> 2(i-1)//d while i= 1,2,..,(d//2)
    
    t = torch.arange(end, device=freqs.device)  
    freqs = torch.outer(t, freqs).float()  # gives diffrent angle vector
    
    # e^it = cos(t) + i sin(t)
    freqs_cos = torch.cos(freqs)  # real   
    freqs_sin = torch.sin(freqs)  # imaginary 
    return freqs_cos, freqs_sin
    
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.dim()
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"{freqs_cis.shape=}, {(x.shape[1], x.shape[-1])=}"
    
    # keep 2nd (T) and last(freq) dim same else make dim 1 for freq_cis
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)] 
    # print(shape)
    return freqs_cis.view(shape)


def apply_rope(k,q,freq_cis):
    freqs_sin,freqs_cos = freq_cis
    #  rehsape a shape (...,n )-> (..., n//2,2)
    q_cis = q.float().reshape(q.shape[:-1] + (-1, 2)) # (B,T,nhead,C) -> (B,T,nhead,Cc,2) # Cc = C//2
    k_cis = k.float().reshape(k.shape[:-1] + (-1, 2)) # (B,T,nhead,C) -> (B,T,nhead,Cc,2) 
    
    xq_r, xq_i = q_cis.unbind(-1) # (B,T,nhead,Cc,2) -> ((B,T,Cc), (B,T,Cc)) split into two tuple
    xk_r, xk_i = k_cis.unbind(-1) # (B,T,nhead,Cc,2) -> ((B,T,Cc), (B,T,Cc))
    
    
    freqs_cos = reshape_for_broadcast(freq_cis[0],xq_r) # freqs.shape = (1,T,1,Cc)
    freqs_sin = reshape_for_broadcast(freq_cis[0],xq_r)
    
    
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin 
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos 
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos 
    
    # now we stack r,i -> [r,i,r2,i2]
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1)  # (B,T,nhead,Cc,2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1)  # (B,T,nhead,Cc,2)
    
    # flatten last two dimensions
    xq_out = xq_out.flatten(3) # (B,T,nhead,C) 
    xk_out = xk_out.flatten(3) # (B,T,nhead,C)
    
    return xq_out.type_as(q), xk_out.type_as(q)





B,nh,T,C = 1,1,20,8
G = 2 
w_n = 5

pos = torch.arange(T)


k=q=v = torch.rand(B,T,nh,C)

freq_cis= precompute_freqs_cis(dim=C, end=T)

k,q= apply_rope(k,q,freq_cis)

k = k.transpose(1,2)
q = q.transpose(1,2)

wei = q@k.transpose(-1,-2)

# print(torch.tril(wei))

# g_size, w_size = G, w_n
# g_pos = pos // g_size

# shift = w_n - w_n // G
# s_g_pos = g_pos + shift
# g_q, g_k = apply_rope(q,k,(s_g_pos,s_g_pos))

cos,isin  = freq_cis

print(cos)

q = torch.randn(B,T+4,C)

def rope_extension(freqs,x):
    f_seq_len,hdim = freqs.shape
    _,T,_ = x.shape
    if T<=f_seq_len:
        return x
    
    ext = T - f_seq_len
    old,rec = freqs[:ext],freqs[ext:]
    
    old:torch.Tensor 
    old = old.repeat_interleave(2, dim=0)
    print(old)
    freqs = torch.cat([old,rec])
    
    return freqs


freqs = rope_extension(cos,q)

print(freqs.shape)