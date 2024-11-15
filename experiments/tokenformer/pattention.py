import torch
import torch.nn as nn
import torch.nn.functional as F

from ohara.modules.activations import ACT2FN

import math



def gelu_l2_norm(inputs, dim=-1):
    nonlinear_outputs = F.gelu(inputs)
    norm_outputs = (
        nonlinear_outputs / torch.norm(nonlinear_outputs, p=2, dim=dim, keepdim=True)
        * math.sqrt(nonlinear_outputs.shape[dim])
    )
    return norm_outputs

new_fn = {"gelu_l2_norm": gelu_l2_norm}
ACT2FN.update(new_fn)


class Pattention(nn.Module):
    """
    Pattention is ducking click bait...
    check down from renamed version
    only contribution of paper is using gelu_l2_norm allows
    you to expand weight matrix without (maybe)
    https://arxiv.org/pdf/2410.23168
    """

    def __init__(
        self,
        key_param_tokens,
        value_param_tokens,
        param_token_num,
        activation_fn="gelu_l2_norm",
        dropout_p=0.0,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.param_token_num = param_token_num
        self.param_key_dim = key_param_tokens
        self.param_value_dim = value_param_tokens
        self.activation = activation_fn
        print(f"activation_fn: {activation_fn}")
        self.activation = ACT2FN[activation_fn]
        self.key_param_tokens = nn.Parameter(torch.rand((self.param_token_num, self.param_key_dim)))
        self.value_param_tokens = nn.Parameter(
            torch.rand((self.param_token_num, self.param_value_dim))
        )

        self.activation_func = ACT2FN[activation_fn]

        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.key_param_tokens)
            torch.nn.init.xavier_uniform_(self.value_param_tokens)

    def forward(self, inputs, dropout_p=0.0):
        query = inputs
        key, value = self.key_param_tokens, self.value_param_tokens

        attn_weight = query @ key.transpose(-2, -1)
        attn_weight = self.activation_func(attn_weight)

        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        output = attn_weight @ value

        return output

    def expand_tokens(self, new_param_token_num: int):
        assert (
            new_param_token_num > self.param_token_num
        ), "new_param_token_num must be greater than current param_token_num"
        with torch.no_grad():
            key_param_tokens = torch.cat(
                [
                    self.key_param_tokens,
                    torch.zeros((new_param_token_num - self.param_token_num, self.param_key_dim)),
                ],
                dim=0,
            )
            value_param_tokens = torch.cat(
                [
                    self.value_param_tokens,
                    torch.zeros((new_param_token_num - self.param_token_num, self.param_value_dim)),
                ],
                dim=0,
            )

        self.key_param_tokens = nn.Parameter(key_param_tokens)
        self.value_param_tokens = nn.Parameter(value_param_tokens)
        self.param_token_num = new_param_token_num


class MLP(nn.Module):
    """
    Pattention is just mlp :)
    """

    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        activation_fn,
        dropout_p=0.0,
    ):
        super().__init__()

        self.hidden_features = hidden_features
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation_fn
        self.activation = ACT2FN[activation_fn]
        self.up_proj = nn.Parameter(torch.rand((self.hidden_features, self.in_features)))
        self.down_proj = nn.Parameter(torch.rand((self.hidden_features, self.out_features)))

        self.activation_func = ACT2FN[activation_fn]
        self.dropout = nn.Dropout(dropout_p)
        with torch.no_grad():
            torch.nn.init.xavier_uniform_(self.up_proj)
            torch.nn.init.xavier_uniform_(self.down_proj)

    def forward(self, inputs, dropout_p=0.0):
        up_proj, down_proj = self.up_proj, self.down_proj

        # up projection
        hidden_states = inputs @ up_proj.transpose(-2, -1)

        # activation and dropout
        hidden_states = self.activation_func(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # down projection
        output = hidden_states @ down_proj

        return output
