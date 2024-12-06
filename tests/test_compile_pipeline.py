import torch
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.utils import make_boxed_compiler

from plai.pipelines.convertion_dialect_torch_to_plai import TorchToPlaiPass
from plai.pipelines.decompose_plai_addmm import DecomposePlaiAddMmPass
from plai.pl_torch_compiler import plnn_compiler
from plai.runtime import plai_numpy_backend_runtime
from plai.runtime.plai_numpy_backend_runtime import PlaiNumpyBackendRuntime
from plai.runtime.plai_numpy_runtime import PlaiNumpyRuntime
from tests.module_pool.simple_nn import SimpleNN, check_torch_compile_forward


def torch_custom_pipline(pipeline=None, runtime=None):
    model = SimpleNN()
    custom_compiler = plnn_compiler.CustomCompiler(pipeline=pipeline, runtime=runtime)
    aot_backend = aot_autograd(fw_compiler=make_boxed_compiler(custom_compiler), bw_compiler=None)
    compiled_model = torch.compile(model, backend=aot_backend)
    check_torch_compile_forward(model, compiled_model)
    print('dump compile forward:')
    print(custom_compiler.graph)


def test_torch_custom_pipline():
    torch_custom_pipline(pipeline=[])


def test_torch_custom_pipline_plai():
    pipeline = TorchToPlaiPass()
    torch_custom_pipline(pipeline=pipeline)


def test_torch_custom_pipeline_plai_opt():
    pipeline = [TorchToPlaiPass(), DecomposePlaiAddMmPass()]
    torch_custom_pipline(pipeline=pipeline)


def test_torch_custom_pipeline_plai_runtime():
    pipeline = [TorchToPlaiPass(), DecomposePlaiAddMmPass()]
    numpy_runtime = PlaiNumpyRuntime()
    torch_custom_pipline(pipeline=pipeline, runtime=numpy_runtime)


def test_torch_custom_pipeline_plai_backend_runtime():
    pipeline = [TorchToPlaiPass(), DecomposePlaiAddMmPass()]
    backend = plai_numpy_backend_runtime.Backend()
    numpy_runtime = plai_numpy_backend_runtime.PlaiNumpyBackendRuntime(backend)
    torch_custom_pipline(pipeline=pipeline, runtime=numpy_runtime)

