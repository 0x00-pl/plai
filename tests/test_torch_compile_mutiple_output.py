import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.utils import make_boxed_compiler

from plai.pl_torch_compiler import dummy_compiler, dump_compiler, plnn_compiler
from tests.module_pool.simple_nn import SimpleNN, check_torch_compile_forward, check_torch_compile_backward


def test_torch_plnn_compile_mutiple_output():
    model = lambda x: torch.max(x, dim=0)
    custom_compiler = plnn_compiler.CustomCompiler()
    aot_backend = aot_autograd(fw_compiler=make_boxed_compiler(custom_compiler), bw_compiler=None)
    compiled_model = torch.compile(model, backend=aot_backend)
    check_torch_compile_forward(model, compiled_model)
    print('dump compile forward:')
    print(custom_compiler.graph)
