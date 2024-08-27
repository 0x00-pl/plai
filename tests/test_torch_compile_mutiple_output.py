import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import make_boxed_compiler

from plai.pl_torch_compiler import plnn_compiler


def test_torch_plnn_compile_multiple_output():
    model = (lambda x: torch.max(x, dim=0))
    custom_compiler = plnn_compiler.CustomCompiler()
    aot_backend = aot_autograd(fw_compiler=make_boxed_compiler(custom_compiler))
    compiled_model = torch.compile(model, backend=aot_backend)
    input_data = torch.randn(1, 10)
    expected_output = model(input_data)
    actual_output = compiled_model(input_data)
    assert torch.allclose(expected_output.values, actual_output.values)
    assert torch.allclose(expected_output.indices, actual_output.indices)
    print('dump compile forward:')
    print(custom_compiler.graph)
