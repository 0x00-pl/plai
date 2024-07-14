import torch

from plai.pl_torch_compiler.dummy_compiler import custom_compiler
from tests.module_pool.simple_nn import SimpleNN, check_torch_compile_forward, check_torch_compile_backward


def test_torch_compile_forward():
    model = SimpleNN()
    compiled_model = torch.compile(model, backend=custom_compiler)
    check_torch_compile_forward(model, compiled_model)


def test_torch_compile_backward():
    model = SimpleNN()
    compiled_model = torch.compile(model, backend=custom_compiler)
    check_torch_compile_backward(compiled_model)
