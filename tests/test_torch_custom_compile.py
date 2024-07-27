import torch

from plai.pl_torch_compiler import dummy_compiler, dump_compiler, plnn_compiler
from tests.module_pool.simple_nn import SimpleNN, check_torch_compile_forward


def test_torch_dummy_compile_forward():
    model = SimpleNN()
    compiled_model = torch.compile(model, backend=dummy_compiler.custom_compiler)
    check_torch_compile_forward(model, compiled_model)


def test_torch_dump_compile_forward():
    model = SimpleNN()
    custom_compiler = dump_compiler.CustomCompiler()
    compiled_model = torch.compile(model, backend=custom_compiler)
    check_torch_compile_forward(model, compiled_model)
    print('dump compile forward:')
    custom_compiler.print_nodes_info()


def test_torch_plnn_compile_forward():
    model = SimpleNN()
    custom_compiler = plnn_compiler.CustomCompiler()
    compiled_model = torch.compile(model, backend=custom_compiler)
    check_torch_compile_forward(model, compiled_model)
    print('dump compile forward:')
    print(custom_compiler.graph)
