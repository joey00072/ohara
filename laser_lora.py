import os
import time

# import pretty_errors

from transformers import AutoTokenizer
from ohara.models.phi import Phi, PhiConfig, Block


import torch
import torch.nn as nn

from tqdm import tqdm


kv = os.getenv("KV")
device = torch.device("mps")

model_name = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model: Phi = Phi.from_pretrained(model_name).to(device).eval().to(torch.float16)
prompt = "def dfs(node):"

inputs = tokenizer.encode(prompt)
print(kv)
kv_cache = model.build_kv_cache() if kv == "True" else None

print(model)
max_tokens = 1000
input_pos = 0
inputs = tokenizer.encode(prompt)


class SVDLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.decomposed = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.decomposed:
            return x @ self.weight.T + self.bias
        print((x @ self.U.T).shape, "XX")
        print(self.S.T.shape, "XW")
        return (((x @ self.U.T) @ self.S.T) @ self.V.T) + self.bias

    def svd(self):
        U: torch.Tensor
        S: torch.Tensor
        V: torch.Tensor
        U, S, V = torch.linalg.svd(self.weight.detach().float(), full_matrices=False)
        self.U = nn.Parameter(U.half())
        self.S = nn.Parameter(S.half())
        self.V = nn.Parameter(V.half())
        self.decomposed = True


layer: Block
for layer in tqdm(model.layers):
    in_features = model.config.d_model
    out_features = model.config.d_model * 4

    up_proj = SVDLinear(in_features, out_features)
    down_proj = SVDLinear(out_features, in_features)

    assert (
        layer.mlp.fc1.weight.shape == up_proj.weight.shape
    ), f"{layer.mlp.fc1.weight.shape} != {up_proj.weight.shape}"
    assert (
        layer.mlp.fc1.bias.shape == up_proj.bias.shape
    ), f"{layer.mlp.fc1.bias.shape} != {up_proj.bias.shape}"
    assert (
        layer.mlp.fc2.weight.shape == down_proj.weight.shape
    ), f"{layer.mlp.fc2.weight.shape} != {down_proj.weight.shape}"
    assert (
        layer.mlp.fc2.bias.shape == down_proj.bias.shape
    ), f"{layer.mlp.fc2.bias.shape} != {down_proj.bias.shape}"

    up_proj.weight = layer.mlp.fc1.weight
    up_proj.bias = layer.mlp.fc1.bias
    down_proj.weight = layer.mlp.fc2.weight
    down_proj.bias = layer.mlp.fc2.bias

    layer.mlp.fc1 = up_proj
    layer.mlp.fc2 = down_proj

    layer.mlp.fc1.svd()
    layer.mlp.fc2.svd()


print("XX")
# exit(0)
start: float = time.time()
model = model.eval()
inputs = torch.tensor(inputs).reshape(1, -1).to(device)
print(tokenizer.decode(inputs.tolist()[0]), end="")
for _idx in range(max_tokens):
    logits = model(inputs, kv_cache, input_pos)
    max_logits = torch.argmax(logits, dim=-1)
    # print(f"{idx=} {max_logits.shape}")
    inputs: torch.Tensor = torch.cat((inputs, max_logits[:, -1:]), dim=-1)
    input_pos = inputs.shape[1] - 1
    print(tokenizer.decode(inputs.tolist()[0][-1]), end="", flush=True)

end: float = time.time()

print(f"Time: {end-start}")


print(tokenizer.decode(inputs.tolist()[0]))
