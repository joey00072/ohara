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
    num_heads: int = None
    head_dim: int = None
    hidden_dim: int = None
    num_kv_heads: int = None
    num_layers: int = 4
    dropout: float = 0.2
    bias: bool = False
    weight_tying: bool = False
    activation: str = "silu"
    mlp: str = "GLU"
    rope_head_dim: int = None
    kv_lora_rank: int = None
    q_lora_rank: int = None
    attn_type: str = "mla"

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)



# ======================================================================================
# ||>>>> Note <<<<||
# --------------------------------------------------------------------------------------
# in the code they are doing different things from paper
# eg
# 1. k_rope is projection form d_model (hidden_dim) while in paper it come from compress_kv
# 2. while q_rope comes from compress_q (in both paper and code)
# 3. there are layer norm on compressed q , kv
# 4. norm is applied to q_nope,q_rope,k_nope and v
#    but not to k_rope (idk why rope part of k should be normalized)
# 5. there is no inference merged code for mla
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
        
        assert config.head_dim is not None , f"head_dim is not defined {config.head_dim=}"
        assert config.q_lora_rank is not None , f"q_lora_rank is not defined {config.q_lora_rank=}"
        assert config.kv_lora_rank is not None , f"kv_lora_rank is not defined {config.kv_lora_rank=}"
        assert config.rope_head_dim is not None , f"rope_head_dim is not defined {config.rope_head_dim=}"
        
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
        self.compress_q_linear = nn.Linear(self.dim, self.q_lora_rank, bias=False)  # W_DQ
        self.decompress_q_nope = nn.Linear(self.q_lora_rank, self.nope_head_dim * self.num_heads, bias=False)
        self.decompress_q_rope = nn.Linear(self.q_lora_rank, self.rope_head_dim * self.num_heads, bias=False)

        # key and value compression
        self.compress_kv_linear = nn.Linear(self.dim, self.kv_lora_rank, bias=False)  # W_DKV
        self.decompress_k_nope = nn.Linear(self.kv_lora_rank, self.nope_head_dim * self.num_heads, bias=False)
        self.decompress_v_linear = nn.Linear(self.kv_lora_rank, self.head_dim * self.num_heads, bias=False)
        
        self.k_rope_linear = nn.Linear(self.dim, self.rope_head_dim, bias=False)

        self.q_norm = RMSNorm(self.q_lora_rank)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        # self.rope_norm = RMSNorm(self.rope_head_dim) # not in deepseekv2

        self.proj = nn.Linear(self.num_heads*self.head_dim , self.dim, bias=False)
        
        
 
    def forward(self, x: Tensor,mask: torch.Tensor, freqs_cis: Tensor):
        batch_size, seq_len, _ = x.shape

        compressed_q = self.compress_q_linear(x)
        norm_q = self.q_norm(compressed_q)
        query_nope:Tensor = self.decompress_q_nope(norm_q)
        query_rope:Tensor = self.decompress_q_rope(norm_q)

        compressed_kv = self.compress_kv_linear(x)
        norm_kv = self.kv_norm(compressed_kv)
        key_nope: Tensor = self.decompress_k_nope(norm_kv)
        value: Tensor = self.decompress_v_linear(norm_kv)
        
        key_rope:Tensor = self.k_rope_linear(x)
        # norm_rope = self.rope_norm(key_rope)

        query_nope = query_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        query_rope = query_rope.view(batch_size, seq_len, self.num_heads, self.rope_head_dim).transpose(1,2)
        
        key_rope = key_rope.view(batch_size, seq_len, 1, self.rope_head_dim).transpose(1,2)
        key_nope = key_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        
        k_rope, q_rope = apply_rope(query_rope,key_rope, cis=freqs_cis)
        
        q_recombined = torch.empty((batch_size,self.num_heads,seq_len, self.head_dim), device=x.device)
        k_recombined = torch.empty((batch_size, self.num_heads, seq_len, self.head_dim), device=x.device)
        
        q_recombined[:,:,:,:self.nope_head_dim] = query_nope
        q_recombined[:,:,:,self.nope_head_dim:] = q_rope
        
        # k_rope = torch.repeat_interleave(k_rope, self.num_heads, dim=1) # >> you dont need to do this <<
        # 👇 broadcasting will do replication krope to all heads automagically
        k_recombined[:,:,:,:self.nope_head_dim] = key_nope
        k_recombined[:,:,:,self.nope_head_dim:] = k_rope

        output = F.scaled_dot_product_attention(q_recombined, k_recombined, value, is_causal=True)

        output = output.contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

        output = self.proj(output)

        return output


class MLA_Inference(MultiHeadLatentAttention):
    def __init__(self,config:Config):
        super().__init__(config)
        self.inference_merged = False
        
    def inference_merge(self):
        Wd_Qnope = self.decompress_q_nope.weight.detach()
        Wd_Knope = self.decompress_k_nope.weight.detach()
        Wd_V = self.decompress_v_linear.weight.detach()
        
        W_proj = self.proj.weight.detach()
        
        Wd_Qnope = Wd_Qnope.reshape(self.num_heads, Wd_Qnope.T.shape[0], -1)
        Wd_Knope = Wd_Knope.reshape(self.num_heads, Wd_Knope.T.shape[0], -1)
        
        # print(f"Wd_Qnope.shape: {Wd_Qnope.shape}, Wd_Knope.shape: {Wd_Knope.shape}")
        WdQK = Wd_Qnope @ Wd_Knope.transpose(-2, -1)
        # print(f"WdQK.shape: {WdQK.shape}")
        
        WdVO = Wd_V.T @ W_proj
        
        # print(f"WdQK.shape: {WdQK.shape}, WdVO.shape: {WdVO.shape}")
        
        self.register_buffer("WdQK", WdQK)
        
        self.inference_merged = True
        
    def forward(self,x:Tensor,freqs_cis:Tensor):
        assert self.inference_merged, "model is not merged run .inference_merge() first"


        batch_size, seq_len, _ = x.shape

        
        def _test_self_attention(x:Tensor):
            compressed_q = self.compress_q_linear(x)
            norm_q = self.q_norm(compressed_q)
            query_nope:Tensor = self.decompress_q_nope(norm_q)
            query_rope:Tensor = self.decompress_q_rope(norm_q)

            compressed_kv = self.compress_kv_linear(x)
            norm_kv = self.kv_norm(compressed_kv)
            key_nope: Tensor = self.decompress_k_nope(norm_kv)
            value: Tensor = self.decompress_v_linear(norm_kv)
            
            key_rope:Tensor = self.k_rope_linear(x)
            # norm_rope = self.rope_norm(key_rope)

            query_nope = query_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
            query_rope = query_rope.view(batch_size, seq_len, self.num_heads, self.rope_head_dim).transpose(1,2)
            
            key_rope = key_rope.view(batch_size, seq_len, 1, self.rope_head_dim).transpose(1,2)
            key_nope = key_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
            
            value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
            
            k_rope, q_rope = apply_rope(query_rope,key_rope, cis=freqs_cis)
            
            q_recombined = torch.empty((batch_size,self.num_heads,seq_len, self.head_dim), device=x.device)
            k_recombined = torch.empty((batch_size, self.num_heads, seq_len, self.head_dim), device=x.device)
            
            q_recombined[:,:,:,:self.nope_head_dim] = query_nope
            q_recombined[:,:,:,self.nope_head_dim:] = q_rope
            
            # k_rope = torch.repeat_interleave(k_rope, self.num_heads, dim=1) # >> you dont need to do this <<
            # 👇 broadcasting will do replication krope to all heads automagically
            k_recombined[:,:,:,:self.nope_head_dim] = key_nope
            k_recombined[:,:,:,self.nope_head_dim:] = k_rope

            output = F.scaled_dot_product_attention(q_recombined, k_recombined, value, is_causal=True)

            output = output.contiguous().view(batch_size, seq_len, self.num_heads * self.head_dim)

            output = self.proj(output)

            return output
        
        
        def _test_flat_attention(x:Tensor):
            compressed_q = self.compress_q_linear(x)
            norm_q = self.q_norm(compressed_q)
            query_nope:Tensor = self.decompress_q_nope(norm_q)
        


def mla_reformulation_test(config:Config):
    
    mla = MultiHeadLatentAttention(config)
    mla_inference = MLA_Inference(config)
    
    x = torch.randn(2, 10, config.d_model)
    freqs_cis = precompute_freqs_cis(config.rope_head_dim, config.seq_len)
    
    mla_inference.load_state_dict(mla.state_dict())
    mla_inference.inference_merge()
    
    output_inference = mla_inference(x, freqs_cis)
    print(torch.allclose(output, output_inference))

if __name__ == "__main__":
    
    d_model = 1024
    num_heads = 16
    head_dim = 128
    kv_lora_rank = 64
    q_lora_rank = 3 * kv_lora_rank
    rope_head_dim = 32
    
    config = Config(
        vocab_size=30522,
        d_model=d_model,
        seq_len=2048,
        num_heads=num_heads,
        head_dim=head_dim,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        rope_head_dim=rope_head_dim,
    )

    mla = MultiHeadLatentAttention(config)
    x = torch.randn(2, 10, d_model)
    freqs_cis = precompute_freqs_cis(config.rope_head_dim, config.seq_len)
    # mla = torch.compile(mla)
    output = mla(x, freqs_cis)
    print(output.shape)
    
 