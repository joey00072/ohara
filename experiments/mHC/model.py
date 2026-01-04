import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass
from ohara.modules.mlp import GLU, MLP
from ohara.modules.norm import RMSNorm


from ohara.embeddings_pos.rotary import precompute_freqs_cis
from ohara.embeddings_pos.rotary import apply_rope

from huggingface_hub import PyTorchModelHubMixin

from collections import OrderedDict

try:
    from .sinkhorn import sinkhorn_log
except ImportError:
    from sinkhorn import sinkhorn_log


@dataclass
class Config(OrderedDict):
    vocab_size: int
    max_sequence_length: int

    hidden_size: int
    intermediate_size: int

    head_dim: int
    num_attention_heads: int
    num_key_value_heads: int = 0

    num_hidden_layers: int = 4
    expansion_rate: int = 1

    dropout: float = 0.2
    bias: bool = False
    weight_tying: bool = False

    activation: str = "silu"  # "relu", "gelu", "silu" etc
    mlp: str = "GLU"  # MLP or GLU

    use_spda: bool = False
    connection_type: str = "residual"  # "residual", "hc", "mhc"
    hc_dynamic: bool = True
    hc_dynamic_scale: float = 0.01
    mhc_sinkhorn_iters: int = 20
    mhc_sinkhorn_eps: float = 1e-6


MLP_BLOCK = {"MLP": MLP, "GLU": GLU}

class HyperConnections(nn.Module):
    def __init__(
        self,
        dim: int,
        rate: int,
        layer_id: int,
        dynamic: bool = True,
        dynamic_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.rate = rate
        self.dynamic = dynamic
        self.dynamic_scale = dynamic_scale

        self.norm = nn.LayerNorm(dim)

        self.dynamic_alpha_fn = nn.Parameter(torch.zeros(dim, rate + 1))
        self.dynamic_beta_fn = nn.Parameter(torch.zeros(dim))

        init_alpha0 = torch.zeros(rate, 1)
        init_alpha0[layer_id % rate, 0] = 1.0
        init_alphan = torch.eye(rate)
        static_alpha = torch.cat((init_alpha0, init_alphan), dim=1)

        self.static_alpha = nn.Parameter(static_alpha)
        self.static_beta = nn.Parameter(torch.ones(rate))

    def width_connection(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        norm_h = self.norm(h)

        if self.dynamic:
            dynamic_alpha = torch.matmul(norm_h, self.dynamic_alpha_fn) * self.dynamic_scale
            alpha = dynamic_alpha + self.static_alpha[None, None, :, :]
        else:
            alpha = self.static_alpha[None, None, :, :]

        alpha_t = alpha.transpose(-1, -2)
        b, l, n, d = h.shape
        k = alpha_t.size(-2)
        mix_h = torch.bmm(
            alpha_t.reshape(b * l, k, n),
            h.reshape(b * l, n, d),
        ).view(b, l, k, d)

        if self.dynamic:
            dynamic_beta = (norm_h * self.dynamic_beta_fn).sum(dim=-1) * self.dynamic_scale
            beta = dynamic_beta + self.static_beta[None, None, :]
        else:
            beta = self.static_beta[None, None, :]

        return mix_h, beta

    def depth_connection(
        self,
        mix_h: torch.Tensor,
        h_o: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        h = h_o.unsqueeze(2) * beta.unsqueeze(-1)
        h = h + mix_h[..., 1:, :]
        return h


class ManifoldHyperConnections(nn.Module):
    def __init__(
        self,
        dim: int,
        rate: int,
        sinkhorn_iters: int = 20,
        sinkhorn_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.rate = rate
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_eps = sinkhorn_eps

        flat_dim = dim * rate
        self.norm = RMSNorm(flat_dim)

        self.phi_pre = nn.Parameter(torch.zeros(flat_dim, rate))
        self.phi_post = nn.Parameter(torch.zeros(flat_dim, rate))
        self.phi_res = nn.Parameter(torch.zeros(flat_dim, rate * rate))

        self.b_pre = nn.Parameter(torch.zeros(rate))
        self.b_post = nn.Parameter(torch.zeros(rate))
        self.b_res = nn.Parameter(torch.zeros(rate, rate))

        self.alpha_pre = nn.Parameter(torch.ones(1))
        self.alpha_post = nn.Parameter(torch.ones(1))
        self.alpha_res = nn.Parameter(torch.ones(1))

    def _sinkhorn(self, log_alpha: torch.Tensor) -> torch.Tensor:
        return sinkhorn_log(log_alpha, self.sinkhorn_iters, self.sinkhorn_eps)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, rate, dim = x.shape
        flat = x.reshape(batch, seq_len, rate * dim)
        flat = self.norm(flat)
        phi = torch.cat((self.phi_pre, self.phi_post, self.phi_res), dim=1)
        proj = flat @ phi
        r = self.rate
        pre = self.alpha_pre * proj[..., :r] + self.b_pre
        post = self.alpha_post * proj[..., r : 2 * r] + self.b_post
        res = self.alpha_res * proj[..., 2 * r :] + self.b_res.reshape(-1)
        res = res.view(batch, seq_len, rate, rate)

        h_pre = torch.sigmoid(pre)
        h_post = 2.0 * torch.sigmoid(post)
        h_res = self._sinkhorn(res)
        return h_pre, h_post, h_res


class Attention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        hidden_size = config.hidden_size
        self.hidden_size = hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0
        self.num_queries_per_kv = self.num_attention_heads // self.num_key_value_heads

        self.key = nn.Linear(hidden_size, self.head_dim * self.num_key_value_heads, config.bias)
        self.query = nn.Linear(hidden_size, self.head_dim * self.num_attention_heads, config.bias)
        self.value = nn.Linear(hidden_size, self.head_dim * self.num_key_value_heads, config.bias)
        self.proj = nn.Linear(self.head_dim * self.num_attention_heads, hidden_size, config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.res_dropout = nn.Dropout(config.dropout)

        self.flash_attn = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention") and not config.use_spda
        )

        self.reset_parameters()

    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        init_std = init_std or (self.head_dim ** (-0.5))

        for w in [self.key, self.query, self.value]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.proj.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, freqs_cis) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        k: torch.Tensor  # type hint for lsp
        q: torch.Tensor  # ignore
        v: torch.Tensor

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        k = k.view(
            batch, seq_len, self.num_key_value_heads, self.head_dim
        )  # shape = (B, seq_len, num_key_value_heads, head_dim)
        q = q.view(batch, seq_len, self.num_attention_heads, self.head_dim)
        v = v.view(batch, seq_len, self.num_key_value_heads, self.head_dim)

        q, k = apply_rope(q, k, freqs_cis)

        # Grouped Query Attention
        if self.num_key_value_heads != self.num_attention_heads:
            k = torch.repeat_interleave(k, self.num_queries_per_kv, dim=2)
            v = torch.repeat_interleave(v, self.num_queries_per_kv, dim=2)

        k = k.transpose(1, 2)  # shape = (B, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.flash_attn:
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,  # order important
                attn_mask=None,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=True,
            )
        else:
            attn_mtx = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
            attn_mtx = attn_mtx + mask[:, :, :seq_len, :seq_len]
            attn_mtx = F.softmax(attn_mtx.float(), dim=-1).type_as(k)
            attn_mtx = self.attn_dropout(attn_mtx)

            output = torch.matmul(attn_mtx, v)  # (batch, n_head, seq_len, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_size)

        # final projection into the residual stream
        output = self.proj(output)
        output = self.res_dropout(output)
        return output


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.attn = Attention(config)
        self.ff: MLP | GLU = MLP_BLOCK[config.mlp](
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            activation_fn=config.activation,
            dropout=config.dropout,
            bias=config.bias,
        )

        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

    def forward(self, x, mask, freqs_cis):
        x = x + self.attn(self.norm1(x), mask, freqs_cis)
        x = x + self.ff(self.norm2(x))
        return x

    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        self.attn.reset_parameters(init_std, factor)
        self.ff.reset_parameters(init_std, factor)


class HyperBlock(nn.Module):
    def __init__(self, config: Config, layer_id: int):
        super().__init__()
        self.connection_type = config.connection_type
        self.rate = config.expansion_rate

        self.attn = Attention(config)
        self.ff: MLP | GLU = MLP_BLOCK[config.mlp](
            dim=config.hidden_size,
            hidden_dim=config.intermediate_size,
            activation_fn=config.activation,
            dropout=config.dropout,
            bias=config.bias,
        )

        self.attn_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)

        if self.connection_type == "hc":
            self.attn_conn = HyperConnections(
                dim=config.hidden_size,
                rate=self.rate,
                layer_id=layer_id,
                dynamic=config.hc_dynamic,
                dynamic_scale=config.hc_dynamic_scale,
            )
            self.ffn_conn = HyperConnections(
                dim=config.hidden_size,
                rate=self.rate,
                layer_id=layer_id,
                dynamic=config.hc_dynamic,
                dynamic_scale=config.hc_dynamic_scale,
            )
        elif self.connection_type == "mhc":
            self.attn_conn = ManifoldHyperConnections(
                dim=config.hidden_size,
                rate=self.rate,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                sinkhorn_eps=config.mhc_sinkhorn_eps,
            )
            self.ffn_conn = ManifoldHyperConnections(
                dim=config.hidden_size,
                rate=self.rate,
                sinkhorn_iters=config.mhc_sinkhorn_iters,
                sinkhorn_eps=config.mhc_sinkhorn_eps,
            )
        else:
            raise ValueError(f"Unsupported connection_type: {self.connection_type}")

    def _apply_hc(
        self,
        x: torch.Tensor,
        conn: HyperConnections,
        norm: RMSNorm,
        fn,
        mask: torch.Tensor,
        freqs_cis,
    ) -> torch.Tensor:
        mix_h, beta = conn.width_connection(x)
        h = norm(mix_h[..., 0, :])
        h = fn(h, mask, freqs_cis) if fn is self.attn else fn(h)
        x = conn.depth_connection(mix_h, h, beta)
        return x

    def _apply_mhc_attn(
        self,
        x: torch.Tensor,
        conn: ManifoldHyperConnections,
        norm: RMSNorm,
        mask: torch.Tensor,
        freqs_cis,
    ) -> torch.Tensor:
        h_pre, h_post, h_res = conn(x)
        h0 = (h_pre.unsqueeze(-1) * x).sum(dim=2)
        h0 = norm(h0)
        h = self.attn(h0, mask, freqs_cis)
        out = h_post.unsqueeze(-1) * h.unsqueeze(2)
        b, l, n, m = h_res.shape
        res = torch.bmm(
            h_res.reshape(b * l, n, m),
            x.reshape(b * l, m, x.size(-1)),
        ).view(b, l, n, x.size(-1))
        return res + out

    def _apply_mhc_ffn(
        self,
        x: torch.Tensor,
        conn: ManifoldHyperConnections,
        norm: RMSNorm,
    ) -> torch.Tensor:
        h_pre, h_post, h_res = conn(x)
        h0 = (h_pre.unsqueeze(-1) * x).sum(dim=2)
        h0 = norm(h0)
        h = self.ff(h0)
        out = h_post.unsqueeze(-1) * h.unsqueeze(2)
        b, l, n, m = h_res.shape
        res = torch.bmm(
            h_res.reshape(b * l, n, m),
            x.reshape(b * l, m, x.size(-1)),
        ).view(b, l, n, x.size(-1))
        return res + out

    def _apply_mhc(
        self,
        x: torch.Tensor,
        conn: ManifoldHyperConnections,
        norm: RMSNorm,
        fn,
        mask: torch.Tensor,
        freqs_cis,
    ) -> torch.Tensor:
        if fn is self.attn:
            return self._apply_mhc_attn(x, conn, norm, mask, freqs_cis)
        return self._apply_mhc_ffn(x, conn, norm)

    def forward(self, x, mask, freqs_cis):
        if self.connection_type == "hc":
            x = self._apply_hc(x, self.attn_conn, self.attn_norm, self.attn, mask, freqs_cis)
            x = self._apply_hc(x, self.ffn_conn, self.ffn_norm, self.ff, mask, freqs_cis)
        else:
            x = self._apply_mhc_attn(x, self.attn_conn, self.attn_norm, mask, freqs_cis)
            x = self._apply_mhc_ffn(x, self.ffn_conn, self.ffn_norm)
        return x

    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        self.attn.reset_parameters(init_std, factor)
        self.ff.reset_parameters(init_std, factor)
        self.attn_norm.reset_parameters()
        self.ffn_norm.reset_parameters()


class Transformer(nn.Module):
    def __init__(self, config: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.config = config
        if config.connection_type not in {"residual", "hc", "mhc"}:
            raise ValueError(f"Unknown connection_type: {config.connection_type}")
        self.connection_type = config.connection_type
        self.expansion_rate = config.expansion_rate
        self.use_hyper = self.connection_type in {"hc", "mhc"}

        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)

        if self.use_hyper:
            self.layers = nn.ModuleList(
                [HyperBlock(config, layer_id=idx) for idx in range(config.num_hidden_layers)]
            )
        else:
            self.layers = nn.ModuleList([Block(config) for _ in range(config.num_hidden_layers)])

        self.norm = RMSNorm(config.hidden_size)
        self.vocab_proj = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.weight_tying:
            self.token_emb.weight = self.vocab_proj.weight

        cos, isin = precompute_freqs_cis(
            config.hidden_size // config.num_attention_heads, config.max_sequence_length * 2
        )
        self.register_buffer("freq_cos", cos)
        self.register_buffer("freq_sin", isin)

        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            print("WARNING: using slow attention | upgrade pytorch to 2.0 or above")
            mask = torch.full(
                (1, 1, config.max_sequence_length, config.max_sequence_length), float("-inf")
            )
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        batch, seqlen = x.shape
        x = self.token_emb(x)
        freqs_cis = self.freq_cos[:seqlen], self.freq_sin[:seqlen]

        if self.use_hyper:
            x = x.unsqueeze(2).expand(-1, -1, self.expansion_rate, -1).contiguous()
            for layer in self.layers:
                x = layer(x, self.mask, freqs_cis)
            x = x.sum(dim=2)
        else:
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

    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        layer: Block
        torch.nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.vocab_proj.weight, mean=0.0, std=0.02)
        for layer in self.layers:
            layer.reset_parameters(init_std, factor)
        self.norm.reset_parameters()


class ModelingLM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: Config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.model = Transformer(self.config)
        self.reset_parameters()

    def forward(self, x: torch.Tensor, return_outputs: bool = False):
        logits = self.model(x)
        if return_outputs:
            return logits, None
        return logits

    def reset_parameters(self, init_std: float | None = None, factor: float = 1.0) -> None:
        self.model.reset_parameters(init_std, factor)


if __name__ == "__main__":
    hidden_size = 128
    num_attention_heads = 4
    config = Config(
        vocab_size=10,
        max_sequence_length=10,
        hidden_size=hidden_size,
        intermediate_size=128,
        head_dim=hidden_size // num_attention_heads,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        num_hidden_layers=4,
        dropout=0.2,
        bias=False,
        weight_tying=False,
        activation="relu_squared",
        mlp="GLU",
    )

    model = ModelingLM(config).eval()
    print(model)
