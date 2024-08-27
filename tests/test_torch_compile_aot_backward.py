import torch
from torch import nn, optim
from torch._dynamo.backends.common import aot_autograd
from torch._functorch._aot_autograd.utils import make_boxed_compiler

from tests.module_pool.simple_nn import SimpleNN


def custom_compiler(gm: torch.fx.GraphModule, example_inputs):
    print("Using custom compiler!")
    gm.graph.print_tabular()
    print()

    return gm.forward


def test_torch_dump_compile():
    # 初始化模型、损失函数和优化器
    model = SimpleNN()
    boxed_compiler = make_boxed_compiler(custom_compiler)  # 使用boxed_compiler包装自定义编译器, 解决aot_autograd的内存释放问题.
    aot_backend = aot_autograd(fw_compiler=boxed_compiler, bw_compiler=boxed_compiler)  # 在backward时也使用自定义编译器
    compiled_model = torch.compile(model, backend=aot_backend)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(compiled_model.parameters(), lr=0.01)

    # 创建输入数据和目标数据
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)

    # 使用自定义编译器进行前向传播和反向传播
    optimizer.zero_grad()
    outputs = compiled_model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    # 输出损失值
    print("Loss:", loss.item())
