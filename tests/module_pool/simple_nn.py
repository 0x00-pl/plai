import torch
import torch.nn as nn
from torch import optim


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def check_torch_compile_forward(model, compiled_model):
    input_data = torch.randn(1, 10)
    expected_output = model(input_data)
    actual_output = compiled_model(input_data)
    assert torch.allclose(expected_output, actual_output), "Output mismatch between compiled and original model"


def check_torch_compile_backward(compiled_model):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(compiled_model.parameters(), lr=0.01)

    input_data = torch.randn(1, 10)
    target = torch.tensor([[1.0]])

    output = compiled_model(input_data)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for param in compiled_model.parameters():
        assert param.grad is not None, "Gradient not computed for parameter in compiled model"
