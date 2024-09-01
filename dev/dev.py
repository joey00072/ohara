import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm



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
