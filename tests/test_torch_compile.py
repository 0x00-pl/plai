import torch

from tests.module_pool.simple_nn import SimpleNN, check_torch_compile_forward


def test_torch_compile_forward():
    model = SimpleNN()
    compiled_model = torch.compile(model, backend="aot_eager")
    check_torch_compile_forward(model, compiled_model)
