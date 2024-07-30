import importlib
import inspect
from typing import Tuple, Callable, List, Dict

import torch
import torch.fx as fx
from torch._ops import OpOverload

from plai.plnn.module import Graph, Node


def get_object_from_string(full_path):
    module_path, obj_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


class CustomCompiler:
    def __init__(self):
        self.graph = Graph('plnn_graph')
        self.node_mapping: Dict[torch.fx.Node, Node] = {}

    def mapping_node(self, value):
        if isinstance(value, Tuple):
            return tuple(self.mapping_node(v) for v in value)
        elif isinstance(value, List):
            return [self.mapping_node(v) for v in value]
        elif isinstance(value, dict):
            return {k: self.mapping_node(v) for k, v in value.items()}
        elif isinstance(value, torch.fx.Node):
            return self.node_mapping[value]
        else:
            return value

    @staticmethod
    def torch_method_to_string(method: str) -> str:
        return method

    @staticmethod
    def torch_module_to_string(module: torch.nn.Module) -> str:
        return module.__class__.__name__

    @staticmethod
    def torch_function_to_string(func: Callable) -> str:
        if isinstance(func, OpOverload):
            return str(func)
        else:
            full_name = f'{inspect.getmodule(func).__name__}.{func.__name__}'
            assert get_object_from_string(full_name) == func
            return full_name

    def __call__(self, gm: fx.GraphModule, example_inputs: Tuple[torch.Tensor, ...]) -> Callable:
        # 遍历计算图中的所有节点并收集信息
        for node in gm.graph.nodes:
            node: fx.Node
            mapped_args = [self.mapping_node(arg) for arg in node.args]
            mapped_kwargs = {key: self.mapping_node(value) for key, value in node.kwargs.items()}
            if node.op == 'placeholder':
                new_node = self.graph.add_argument(node.name, node.target)
            elif node.op == 'call_method':
                assert isinstance(node.target, str)
                new_node = self.graph.add_node(
                    node.name, self.torch_method_to_string(node.target), mapped_args,
                    mapped_kwargs
                )
            elif node.op == 'call_module':
                assert isinstance(node.target, torch.nn.Module)
                new_node = self.graph.add_node(
                    node.name, self.torch_module_to_string(node.target), mapped_args,
                    mapped_kwargs
                )
            elif node.op == 'call_function':
                new_node = self.graph.add_node(
                    node.name, self.torch_function_to_string(node.target),
                    mapped_args, mapped_kwargs
                )
            elif node.op == 'get_attr':
                raise NotImplementedError("get_attr is not supported")
            elif node.op == 'output':
                new_node = self.graph.add_output(mapped_args[0])
            else:
                raise ValueError(f"Unsupported op: {node.op}")

            self.node_mapping[node] = new_node

        # 返回未修改的前向传播函数
        return gm.forward
