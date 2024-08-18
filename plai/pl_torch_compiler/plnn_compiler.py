import importlib
import inspect
from typing import Tuple, Callable, List, Dict, Any

import torch
import torch.fx as fx
from torch._ops import OpOverload

from plai.core import core_dialect
from plai.core.location import NamedLocation, DummyLocation
from plai.core.module import Graph, Node
from plai.dialect import aten_dialect, torch_dialect


def get_object_from_string(full_path):
    module_path, obj_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


def torch_method_to_string(method: str) -> str:
    return method


def torch_module_to_string(module: torch.nn.Module) -> str:
    return module.__class__.__name__


def torch_function_to_string(func: Callable) -> str:
    if isinstance(func, OpOverload):
        return func.name()
    else:
        full_name = f'{inspect.getmodule(func).__name__}.{func.__name__}'
        assert get_object_from_string(full_name) == func
        return full_name


def torch_node_to_core_node(node: fx.Node, node_mapping: Callable[[fx.Node], Any]) -> Node:
    if node.op == 'placeholder':
        assert isinstance(node.target, str)
        return core_dialect.Placeholder(NamedLocation(node.target))
    elif node.op == 'call_method':
        raise NotImplementedError("call_method is not supported")
    elif node.op == 'call_module':
        raise NotImplementedError("call_module is not supported")
    elif node.op == 'call_function':
        func_name = torch_function_to_string(node.target)
        args = [node_mapping(arg) for arg in node.args]
        attrs = {k: v for k, v in node.kwargs.items()}
        if func_name == 'aten::view':
            return aten_dialect.View(args[0], args[1], DummyLocation())
        elif func_name == 'aten::detach':
            return aten_dialect.Relu(args[0], DummyLocation())
        elif func_name == 'aten::t':
            return core_dialect.Transpose(args[0], DummyLocation())
        elif func_name == 'aten::addmm':
            return aten_dialect.AddMm(
                args[0], args[1], args[2],
                attrs.get('beta', 1), attrs.get('alpha', 1),
                DummyLocation()
            )
        elif func_name == 'aten::mm':
            return aten_dialect.Mm(args[0], args[1], DummyLocation())
        elif func_name == 'aten::relu':
            return aten_dialect.Relu(args[0], DummyLocation())
        elif func_name == 'aten::max.dim':
            keepdim = args[2] if len(args) == 3 else False
            return aten_dialect.Max(args[0], args[1], keepdim, DummyLocation())
        elif func_name == 'aten::sum.dim_IntList':
            return aten_dialect.Sum(args[0], args[1], args[2], DummyLocation())
        elif func_name == 'aten::threshold_backward':
            return aten_dialect.ThresholdBackward(args[0], args[1], args[2], DummyLocation())
        elif func_name == 'torch._C._nn.linear':
            return torch_dialect.Linear(args[0], args[1], args[2], DummyLocation())
        elif func_name == 'torch.relu':
            return torch_dialect.Relu(args[0], DummyLocation())
        elif func_name == '_operator.getitem':
            raise NotImplementedError("_operator.getitem is not supported")
        else:
            raise NotImplementedError(f"Unsupported function: {func_name}")
    elif node.op == 'get_attr':
        raise NotImplementedError("get_attr is not supported")
    elif node.op == 'output':
        raise ValueError("Do not put output node in the middle of the graph")
    else:
        raise ValueError(f"Unsupported op: {node.op}")


class CustomCompiler:
    def __init__(self):
        self.graph = Graph('main_graph')
        self.node_mapping_dict: Dict[torch.fx.Node, Node] = {}

    def node_mapping(self, value):
        if isinstance(value, Tuple):
            return tuple(self.node_mapping(v) for v in value)
        elif isinstance(value, List):
            return [self.node_mapping(v) for v in value]
        elif isinstance(value, dict):
            return {k: self.node_mapping(v) for k, v in value.items()}
        elif isinstance(value, torch.fx.Node):
            return self.node_mapping_dict[value]
        else:
            return value

    def __call__(self, gm: fx.GraphModule, example_inputs: Tuple[torch.Tensor, ...]) -> Callable:
        # 遍历计算图中的所有节点并收集信息
        for node in gm.graph.nodes:
            assert isinstance(node, fx.Node)
            # mapped_args = [self.mapping_node(arg) for arg in node.args]
            # mapped_kwargs = {key: self.mapping_node(value) for key, value in node.kwargs.items()}
            if node.op == 'output':
                for i in node.args[0]:
                    assert i in self.node_mapping_dict or i is None
                    self.graph.add_output(self.node_mapping(i))
                new_node = None
            elif node.op == 'placeholder':
                new_node = self.graph.add_argument()
            elif node.op == 'get_attr':
                raise NotImplementedError("get_attr is not supported")
            else:
                new_node = torch_node_to_core_node(node, self.node_mapping)
                self.graph.add_node(new_node)

            self.node_mapping_dict[node] = new_node

        # 返回未修改的前向传播函数
        return gm.forward
