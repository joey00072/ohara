from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class ReGLU(nn.Module):
    def __init__(self, dim, hidden_dim) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.up = nn.Linear(dim, hidden_dim)
        self.gate = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)

    def init(self):
        torch.nn.init.xavier_uniform_(self.up.weight)
        torch.nn.init.xavier_uniform_(self.down.weight)

    def forward(self, x):
        return self.down(F.relu(self.gate(x)) * self.up(x))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.up = nn.Linear(dim, hidden_dim)
        self.gate = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)

    def init(self):
        torch.nn.init.xavier_uniform_(self.up.weight)
        torch.nn.init.xavier_uniform_(self.down.weight)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim) -> None:
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.up = nn.Linear(dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, dim)
        self.init()

    def init(self):
        torch.nn.init.xavier_uniform_(self.up.weight)
        torch.nn.init.xavier_uniform_(self.down.weight)

    def forward(self, x):
        return self.down(F.relu(self.up(x)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        PREV = 100
        LATENT = 2
        HIDDEN = 3
        self.fc1 = nn.Linear(28 * 28, PREV)
        self.fc2 = nn.Linear(PREV, LATENT)
        self.proj = nn.Linear(LATENT, LATENT)
        self.block = MLP(LATENT, HIDDEN)
        print(f"{self.block.__class__.__name__}")
        self.fc3 = nn.Linear(LATENT, 10)

        self.ln = nn.LayerNorm(LATENT)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        res = F.tanh(self.proj(x))
        x = self.ln(x)
        x = self.block(x)
        x = x + res
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def main():
    # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=10, metavar='N',
    #                     help='number of epochs to train (default: 10)')
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')

    # parser.add_argument('--save-model', action='store_true', default=False,
    #                     help='For Saving the current Model')
    # args = parser.parse_args()

    seed = 42
    batch_size = 128
    test_batch_size = 1000
    epochs = 10
    lr = 0.01
    momentum = 0.5
    log_interval = 10
    save_model = False

    use_cuda = True

    torch.manual_seed(seed)

    device = torch.device("cuda")

    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs,
    )

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch, log_interval)
        test(model, device, test_loader)

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
