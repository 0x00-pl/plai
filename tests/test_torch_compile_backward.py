import torch
from torch import nn, optim

from plai.pl_torch_compiler import dump_compiler, dummy_compiler
from tests.module_pool.simple_nn import SimpleNN


def custom_compiler(gm, example_inputs):
    print("Using custom compiler!")
    return gm.forward


def test_torch_dump_compile_backward():
    # 初始化模型、损失函数和优化器
    model = SimpleNN()
    compiled_model = torch.compile(model, backend=custom_compiler)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(compiled_model.parameters(), lr=0.01)

    # 创建输入数据和目标数据
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)

    # 使用自定义编译器进行前向传播和反向传播
    optimizer.zero_grad()
    outputs = compiled_model(inputs)
    loss = criterion(outputs, targets)
    torch._dynamo.config.compiled_autograd = True
    compiled_backward = torch.compile(lambda _: loss.backward(), backend=custom_compiler)
    compiled_backward(None)
    # loss.backward()
    optimizer.step()

    # 输出损失值
    print("Loss:", loss.item())
