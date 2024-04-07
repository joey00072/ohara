import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from dataclasses import dataclass

import ohara.models.llama as llama
from ohara.modules.mlp import SwiGLU, MLP
from ohara.modules.moe import MoE
from ohara.modules.norm import RMSNorm
from ohara.embedings_pos.rotatry import precompute_freqs_cis

from huggingface_hub import PyTorchModelHubMixin
import lightning as L


@dataclass
class Config(llama.Config):
    """
    capacity_factor is between [0,1), According to the paper, 0.12 (ie 12% token is will pass directly to the next layer)
    """

    capacity_factor: float = 0.12

    ffn: str = "swiglu"  # mlp, swiglu
    mixture_of_expert:bool = False
    moe_num_experts: int = 4
    moe_num_experts_per_tok: int = 2
    
    mixture_of_depth: bool = True
    
    model_type:bool = "mixture_of_depth"
    sliding_window_attention = False
    window_size: int = 128
    weight_tying: bool = True


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()

        self.attn = llama.Attention(cfg)
        self.ff = self._ffn(cfg)

        self.norm1 = RMSNorm(cfg.d_model)
        self.norm2 = RMSNorm(cfg.d_model)

    def _ffn(self, cfg: Config):
        if cfg.mixture_of_expert:
            return MoE(
                dim=cfg.d_model,
                num_experts=cfg.moe_num_experts,
                num_experts_per_tok=cfg.moe_num_experts_per_tok,
                hidden_dim=cfg.hidden_dim,
                mlp=cfg.ffn,
            )
        if cfg.ffn == "mlp":
            return MLP(cfg.d_model, cfg.hidden_dim, cfg.dropout)
        if cfg.ffn == "swiglu":
            return SwiGLU(cfg.d_model, cfg.hidden_dim, cfg.dropout)

    def forward(self, x, mask, freqs_cis):
        x = x + self.attn(self.norm1(x), mask, freqs_cis)
        x = x + self.ff(self.norm2(x))
        return x


class MoD(nn.Module):
    """
    Paper: https://arxiv.org/abs/2404.02258
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.seq_len = cfg.seq_len
        self.capacity_factor = cfg.capacity_factor
        self.dim = cfg.d_model

        self.transformer_decoder_block = Block(cfg)
        self.router = nn.Linear(self.dim, 1, bias=False)
        self.aux_router = nn.Linear(self.dim, 1, bias=False)

    def forward(self, x: Tensor, mode="train", *args, **kwargs):
        batch_size, S, dim = x.shape

        if mode == "inference":
            return self.inference(x, *args, **kwargs)
        # S = seq_len, C = capacity  , C = int(seq_length * capacity_factor)
        #  page 6 above eq 1 | ( C<S ) | here top_k = beta
        top_k = int(S * self.capacity_factor)  # may be i should use math.ceil

        # eq1 page 6
        # scaler weights for each token
        router_logits = self.router(x)  # (x) batch,seq_len,dim -> r batch,seq_len,1

        #  𝑟𝑙> 𝑃𝛽 (R)  ... eqution 1
        token_weights, token_index = torch.topk(router_logits, top_k, dim=1, sorted=False)

        # now we have idx, we can copy this weights to another tensor and pass them to attn+mlp

        # since its auto regressive model we need to keep casual nature of it
        # that why we need sort the tokens by idx before we pass it to attn
        selected_tokens, index = torch.sort(token_index, dim=1)

        # selecting router wight by idx ( in sorted maner)
        r_weights = torch.gather(token_weights, dim=1, index=index)

        # select idx for copying for original tensor
        indices_expanded = selected_tokens.expand(-1, -1, C)

        # This are fillted topk tokens with capactiy C
        filtered_x = torch.gather(input=x, dim=1, index=indices_expanded)  # -> batch, capacity, dim

        x_out = self.transformer_decoder_block(filtered_x)

        # muliply by router weights, this add router in gradient stream
        xw_out = token_weights * x_out

        # batch_indices = torch.arange(batch_size).unsqueeze(-1).expand(-1, top_k)
        # # https://discuss.pytorch.org/t/when-inplace-operation-are-allowed-and-when-not/169583/2
        # out = x.clone()
        # # add back to resuidal strean
        # out[batch_indices, selected_tokens.squeeze(-1),: ] += xw_out
        # ^ this can be done with torch.scatter_add
        out = torch.scatter_add(input=x, dim=1, index=selected_tokens, src=xw_out)
        return out, self.aux_loss(x, router_logits, selected_tokens)

    def aux_loss(self, x: Tensor, router_logits: Tensor, selected_tokens: Tensor):
        # Page 7, Section 3.5 sampling
        router_targets = torch.zeros_like(router_logits).view(
            -1
        )  # i think torch to scatter will work here TODO
        router_targets[selected_tokens.view(-1)] = 1.0
        aux_router_logits = self.aux_router(x.detach().view(B * T, -1))
        aux_router_logits = F.sigmoid(aux_router_logits)  # keep output in range [0,1)
        return F.binary_cross_entropy(aux_router_logits.view(-1), router_targets)

    def inference(self, x: Tensor, *args, **kwargs):
        assert False, "TODO: will implement inference soon"


class TranformerDecoder(nn.Module):
    def __init__(self, cfg: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        
        if cfg.mixture_of_depth:
            DecoderBlock = MoD
        else:
            DecoderBlock = Block
            
        self.layers = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.num_layers)])

        self.norm = RMSNorm(cfg.d_model)
        self.vocab_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.token_emb.weight = self.vocab_proj.weight

        self.cis = precompute_freqs_cis(cfg.d_model // cfg.num_heads, cfg.seq_len * 2)

        if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full((1, 1, cfg.seq_len, cfg.seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        batch, seqlen = x.shape
        x = self.token_emb(x)
        device = self.token_emb.weight.device
        freqs_cis = self.cis[0][:seqlen].to(device), self.cis[1][:seqlen].to(device)

        for layer in self.layers:
            x = layer(x, self.mask, freqs_cis)

        x = self.norm(x)
        x = self.vocab_proj(x)
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class ModelingMixtureOfDepth(PyTorchModelHubMixin,L.LightningModule):
    def __init__(self, cfg: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = cfg
        self.model = TranformerDecoder(cfg)

    def forward(self, x: torch.Tensor, mode: str):
        if mode == "inference":
            return self.model.inference(x)
        else:
            return self.model(x)

if __name__ == "__main__":
    B, T, C = 1, 10, 4
    torch.manual_seed(3)
    capacity_factor = 0.3
    x = torch.randn(B, T, C)
    cfg = Config(d_model=C, seq_len=T, capacity_factor=capacity_factor)
    model = MoD(cfg)

    y = model(x, mode="inference")

    print(x)
    print(y)
