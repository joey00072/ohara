import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from collections import OrderedDict

from ohara.modules.norm import RMSNorm

from ohara.embedings_pos.rotatry import precompute_freqs_cis
from ohara.embedings_pos.rotatry import apply_rope

from torch import Tensor


from rich import print, traceback
traceback.install()


@dataclass
class Config(OrderedDict):
    vocab_size: int
    seq_len: int

    d_model: int
    hidden_dim: int

    # in deepseekv2 d_model < num_heads * head_dim
    # they expanded dim*3.2 for attention
    num_heads: int
    head_dim: int

    num_layers: int = 4

    dropout: float = 0.2
    bias: bool = False
    weight_tying: bool = False

    activation: str = "silu"  # "relu", "gelu", "silu" etc
    mlp: str = "GLU"  # MLP or GLU
    
    # rope is applied partially to hdim of query and key
    rope_head_dim: int = None
    
    # rank for query is higher than key and value
    # query has more information than key and value 
    # in deepseekv2  q_lora_rank =  3 * kv_lora_rank
    kv_lora_rank: int = None
    q_lora_rank: int = None


# ======================================================================================
# ||>>>> Note: this is not one to one mapping of deepseekv2 <<<<||
# --------------------------------------------------------------------------------------
# in the code they are doing different things from paper
# eg
# 1. k_rope is projection form d_model (hidden_dim) while in paper it come from compress_kv
# 2. while q_rope comes from compress_q (in both paper and code)
# 3. there are layer norm on compressed q , kv 
# 4. norm is applied to q_nope,q_rope,k_nope and v 
#    but not to k_rope (idk why rope part of k should be normalized)
# This implimentation works (better imo) but you can't load deepseekv2 weights here
# refer to deekseekv2 for thier code https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/4461458f186c35188585855f28f77af5661ad489/modeling_deepseek.py#L682
# also there is no inference merged code for mla, there implimentation do merge linear layer
# it complicates thing for inference version of MLA
# should i wright ugly but tiny fast train code??
# ======================================================================================

# --- MLA ---
class MultiHeadLatentAttention(nn.Module):
    """
    Multi Head Latent Attention 
    paper: https://arxiv.org/pdf/2405.04434
    
    TLDR: 
    kv are low ranks, this verient of attention project q,k,v to low rank to save memory

    by joey00072 (https://github.com/joey00072)
    """
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        self.dim = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank

            
        # (attention_dim == num_head*head_dim) > d_model in deepseekv2
        self.attention_dim = self.num_heads * self.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.nope_head_dim = config.head_dim - config.rope_head_dim

        # query compression
        self.compress_q_linear = nn.Linear(self.dim, self.q_lora_rank, bias=config.bias)  # W_DQ
        self.decompress_q_nope = nn.Linear(self.q_lora_rank, self.nope_head_dim * self.num_heads, bias=config.bias)  
        self.decompress_q_rope = nn.Linear(self.q_lora_rank, self.rope_head_dim * self.num_heads, bias=config.bias) 

        # key and value compression
        self.compress_kv_linear = nn.Linear(self.dim, self.kv_lora_rank, bias=config.bias)  # W_DKV
        self.decompress_k_nope = nn.Linear(self.kv_lora_rank, self.nope_head_dim * self.num_heads, bias=config.bias)  
        self.decompress_v_linear = nn.Linear(self.kv_lora_rank, self.head_dim * self.num_heads, bias=config.bias)
        
        self.k_rope_linear = nn.Linear(self.dim, self.num_heads*self.rope_head_dim, bias=config.bias) 

        self.rope_k = nn.Linear(self.dim, self.num_heads*self.rope_head_dim, bias=config.bias)
        
        self.q_norm = RMSNorm(self.q_lora_rank)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        # self.rope_norm = RMSNorm(self.rope_head_dim) # not in deepseekv2

        self.proj = nn.Linear(self.attention_dim, self.dim, bias=config.bias)
        
        # MLA_cache =  kv_lora_rank + num_heads * rope_head_dim
        # MHA_cache = 2*d_model
        # pct_used = (kv_lora_rank + num_heads * rope_head_dim)*100 / (2*d_model)  
        # pct_saved = 100 - pct_used 

    def forward(self, x: Tensor, freqs_cis: Tensor):
        batch_size, seq_len, dim = x.shape

        compressed_q = self.compress_q_linear(x)
        norm_q = self.q_norm(compressed_q)
        query_nope:Tensor = self.decompress_q_nope(norm_q)
        query_rope:Tensor = self.decompress_q_rope(norm_q)

        compressed_kv = self.compress_kv_linear(x)
        norm_kv = self.kv_norm(compressed_kv)
        key_nope: Tensor = self.decompress_k_nope(norm_kv)
        value: Tensor = self.decompress_v_linear(norm_kv)
        
        key_rope = self.rope_k(x)
        # norm_rope = self.rope_norm(key_rope) 

        query_nope = query_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim)
        query_rope = query_rope.view(batch_size, seq_len, self.num_heads, self.rope_head_dim)
        
        key_rope = key_rope.view(batch_size, seq_len, self.num_heads, self.rope_head_dim)
        key_nope = key_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim)
        
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        q_rope,k_rope = apply_rope(query_rope,key_rope, cis=freqs_cis)
        
        q_recombined = torch.cat([query_nope, q_rope], dim=-1)
        k_recombined = torch.cat([key_nope, k_rope], dim=-1)

        output = F.scaled_dot_product_attention(q_recombined, k_recombined, value, is_causal=True)

        output = output.contiguous().view(batch_size, seq_len, self.num_heads*self.head_dim)

        output = self.proj(output)

        return output


class MLA_Inference(MultiHeadLatentAttention):
    def __init__(self,config:Config ):
        super().__init__(config)
        self.config = config
        
        W_UQ = self.decompress_q_nope.weight.data.detach()
        W_UK = self.decompress_k_nope.weight.data.detach()
        
 
        
        
    



if __name__ == "__main__":
    
    d_model = 1024
    num_heads = 16
    head_dim = 128
    kv_lora_rank = 32
    q_lora_rank = 3 * kv_lora_rank
    rope_head_dim = 64
    
    config = Config(
        vocab_size=30522,
        d_model=d_model,
        seq_len=2048,
        hidden_dim=d_model,
        num_heads=num_heads,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        rope_head_dim=rope_head_dim,
    )

    mla = MultiHeadLatentAttention(config)
    x = torch.randn(1, 10, d_model)
    freqs_cis = precompute_freqs_cis(config.rope_head_dim, config.seq_len)
    mla = torch.compile(mla)
    output = mla(x, freqs_cis)
    print(output.shape)



