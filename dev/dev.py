import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ohara.modules.activations import ACT2FN

from tqdm import tqdm
import math


def gelu_l2_norm(inputs, dim=-1):
    nonlinear_outputs = F.gelu(inputs)
    norm_outputs = nonlinear_outputs / torch.norm(nonlinear_outputs, p=2, dim=dim, keepdim=True) * math.sqrt(nonlinear_outputs.shape[dim])
    return norm_outputs

def l2_norm_gelu(inputs, dim=-1):
    norm_outputs = inputs / torch.norm(inputs, p=2, dim=dim, keepdim=True) * math.sqrt(inputs.shape[dim])
    nonlinear_outputs = F.gelu(norm_outputs)
    return nonlinear_outputs

new_fns = {'gelu_l2_norm': gelu_l2_norm, 'l2_norm_gelu': l2_norm_gelu}
ACT2FN.update(new_fns)


class Pattention(nn.Module):
    """
    Pattention is ducking click bait...
    check down from renamed version 
    only contribution of paper is using gelu_l2_norm allows 
    you to expand weight matrix without (maybe)
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
        self.activation = ACT2FN[activation_fn]
        self.key_param_tokens = nn.Parameter(torch.rand((self.param_token_num, self.param_key_dim)))
        self.value_param_tokens = nn.Parameter(torch.rand((self.param_token_num, self.param_value_dim)))
        
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
    
    
    def expand_tokens(self, new_param_token_num:int):
        assert new_param_token_num > self.param_token_num, "new_param_token_num must be greater than current param_token_num"
        with torch.no_grad():
            key_param_tokens = torch.cat([self.key_param_tokens, torch.zeros((new_param_token_num - self.param_token_num, self.param_key_dim))], dim=0)
            value_param_tokens = torch.cat([self.value_param_tokens, torch.zeros((new_param_token_num - self.param_token_num, self.param_value_dim))], dim=0)
            
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


class DMLinear(nn.Linear):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.scale = nn.Parameter(torch.ones(1, out_features))

    def forward(self, x):
        w = self.weight

        w = w / w.norm(dim=1, keepdim=True).detach()

        out = F.linear(x, w) * self.scale

        if self.bias is not None:
            out = out + self.bias
        return out

Linear = DMLinear # nn.Linear
from ohara.modules.norm import RMSNorm

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        bias = False
        self.fc1 = Linear(28 * 28, 128,bias=bias)
        self.fc2 = Linear(128, 128,bias=bias)
        self.fc3 = Linear(128, 128,bias=bias)
        self.fc4 = Linear(128, 128,bias=bias)
        self.fc5 = Linear(128, 10,bias=bias)
        self.norm = RMSNorm(128)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.norm(self.fc1(x))
        x = x-F.tanh(self.fc4(F.silu(self.fc2(x)) * self.fc3(x)))
        x = self.fc5(x)
        return F.sigmoid(x)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Initialize the model, loss function, and optimizer
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5)

# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.2f}%")

print("Training completed!")
