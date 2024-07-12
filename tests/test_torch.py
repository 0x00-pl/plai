import torch
import torch.nn as nn
import torch.optim as optim


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test_forward_pass():
    model = SimpleNN()
    input_data = torch.randn(10)
    output = model(input_data)

    assert output.shape == torch.Size([1]), "Output shape mismatch"


def test_loss_calculation():
    model = SimpleNN()
    criterion = nn.MSELoss()

    input_data = torch.randn(10)
    target = torch.tensor([1.0])

    output = model(input_data)
    loss = criterion(output, target)

    assert loss.dim() == 0, "Loss is not a scalar"


def test_backward_pass():
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    input_data = torch.randn(10)
    target = torch.tensor([1.0])

    output = model(input_data)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for param in model.parameters():
        assert param.grad is not None, "Gradient not computed for parameter"
