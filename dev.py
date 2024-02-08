import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import math

batch_size = 128
# Step 1: Load and preprocess the MNIST dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

trainset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)


device = torch.device("mps")


class Linear(nn.Module):
    def __init__(self, input_features, output_features, rank=16):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(output_features))

        self.A = nn.Parameter(torch.rand((rank, input_features)))
        self.scale = nn.Parameter(torch.ones(1))
        self.B = nn.Parameter(torch.rand(output_features, rank))
        # Initialize weights and biases
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        lora = ((x @ self.A.t()) * self.scale) @ self.B.t()
        return lora + self.bias


# Step 2: Define a Neural Network with only Linear layers
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(28 * 28, 512)
        self.fc2 = Linear(512, 128)
        self.fc3 = Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


net = Net().to(device)

# Step 3: Define a Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(net.parameters(), lr=0.001)

# Step 4: Train the network
loss_values = []
for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader)):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # print every 100 mini-batches
            # print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.4f}")
            loss_values.append(running_loss / 100)
            running_loss = 0.0

print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.4f}")
print("Finished Training")


# Step 5: Plot the Loss Graph
plt.plot(loss_values)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
