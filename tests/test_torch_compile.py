import torch
import torch.nn as nn
import torch.optim as optim

from tests.module_pool.simple_nn import SimpleNN


def test_torch_compile_forward():
    model = SimpleNN()
    compiled_model = torch.compile(model, backend="aot_eager")

    input_data = torch.randn(1, 10)
    expected_output = model(input_data)
    actual_output = compiled_model(input_data)
    assert torch.allclose(expected_output, actual_output), "Output mismatch between compiled and original model"


def test_torch_compile_backward():
    model = SimpleNN()
    compiled_model = torch.compile(model, backend="aot_eager")

    criterion = nn.MSELoss()
    optimizer = optim.SGD(compiled_model.parameters(), lr=0.01)

    input_data = torch.randn(1, 10)
    target = torch.tensor([1.0])

    output = compiled_model(input_data)
    loss = criterion(output, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    for param in compiled_model.parameters():
        assert param.grad is not None, "Gradient not computed for parameter in compiled model"
