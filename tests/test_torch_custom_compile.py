import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.utils import make_boxed_compiler

from plai.pl_torch_compiler import dummy_compiler, dump_compiler, plnn_compiler
from tests.module_pool.simple_nn import SimpleNN, check_torch_compile_forward, check_torch_compile_backward


def test_torch_dummy_compile_forward():
    model = SimpleNN()
    compiled_model = torch.compile(model, backend=dummy_compiler.custom_compiler)
    check_torch_compile_forward(model, compiled_model)


def test_torch_dummy_compile_backward():
    model = SimpleNN()
    aot_backend = aot_autograd(fw_compiler=make_boxed_compiler(dummy_compiler.custom_compiler))
    compiled_model = torch.compile(model, backend=aot_backend)
    check_torch_compile_backward(model, compiled_model)


def test_torch_dump_compile_forward():
    model = SimpleNN()
    custom_compiler = dump_compiler.CustomCompiler()
    compiled_model = torch.compile(model, backend=custom_compiler)
    check_torch_compile_forward(model, compiled_model)
    print('dump compile forward:')
    custom_compiler.print_nodes_info()


def test_torch_dump_compile_backward():
    model = SimpleNN()
    custom_compiler = dump_compiler.CustomCompiler()
    aot_backend = aot_autograd(fw_compiler=make_boxed_compiler(custom_compiler))
    compiled_model = torch.compile(model, backend=aot_backend)
    check_torch_compile_backward(model, compiled_model)
    print('dump compile backward:')
    custom_compiler.print_nodes_info()


def test_torch_plnn_compile_forward():
    model = SimpleNN()
    custom_compiler = plnn_compiler.CustomCompiler()
    compiled_model = torch.compile(model, backend=custom_compiler)
    check_torch_compile_forward(model, compiled_model)
    print('dump compile forward:')
    print(custom_compiler.graph)


def test_torch_plnn_compile_forward_autograd():
    model = SimpleNN()
    custom_compiler = plnn_compiler.CustomCompiler()
    aot_backend = aot_autograd(fw_compiler=make_boxed_compiler(custom_compiler), bw_compiler=None)
    compiled_model = torch.compile(model, backend=aot_backend)
    check_torch_compile_forward(model, compiled_model)
    print('dump compile forward:')
    print(custom_compiler.graph)


def test_torch_plnn_compile_backward_autograd():
    model = SimpleNN()
    custom_compiler = plnn_compiler.CustomCompiler()
    aot_backend = aot_autograd(fw_compiler=make_boxed_compiler(custom_compiler))
    compiled_model = torch.compile(model, backend=aot_backend)
    check_torch_compile_backward(model, compiled_model)
    print('dump compile backward:')
    print(custom_compiler.graph)
