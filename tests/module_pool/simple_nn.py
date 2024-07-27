import torch
import torch.nn as nn


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
