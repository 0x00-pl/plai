import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_compiler

from plai.pl_torch_compiler import plnn_compiler
from tests.module_pool.simple_nn import SimpleNN, check_torch_compile_forward


def test_torch_plnn_compile_forward():
    model = SimpleNN()
    custom_compiler = plnn_compiler.CustomCompiler()
    compiled_model = torch.compile(model, backend=custom_compiler)
    check_torch_compile_forward(model, compiled_model)
    print('dump compile(no aot_autograd) forward:')
    print(custom_compiler.graph)


def torch_plnn_compile_autograd(enable_bw_compiler=False):
    model = SimpleNN()
    custom_compiler = plnn_compiler.CustomCompiler()
    if enable_bw_compiler:
        aot_backend = aot_autograd(fw_compiler=make_boxed_compiler(custom_compiler))
    else:
        aot_backend = aot_autograd(fw_compiler=make_boxed_compiler(custom_compiler), bw_compiler=None)
    compiled_model = torch.compile(model, backend=aot_backend)
    check_torch_compile_forward(model, compiled_model)
    return custom_compiler


def test_torch_plnn_compile_forward_autograd():
    custom_compiler = torch_plnn_compile_autograd(enable_bw_compiler=False)
    print('dump compile forward:')
    print(custom_compiler.graph)


def test_torch_plnn_compile_backward_autograd():
    custom_compiler = torch_plnn_compile_autograd(enable_bw_compiler=True)
    print('dump compile forward/backward:')
    print(custom_compiler.graph)
