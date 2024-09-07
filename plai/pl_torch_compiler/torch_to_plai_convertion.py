import importlib
from typing import Callable, Any, Dict

import torch
from torch import fx
from torch._ops import OpOverload

from plai.core import core_dialect
from plai.core.location import NamedLocation, Location
from plai.core.module import Node


def get_object_from_string(full_path):
    module_path, obj_name = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


def torch_method_to_string(method: str) -> str:
    return method


def torch_module_to_string(module: torch.nn.Module) -> str:
    return module.__class__.__name__


def torch_function_to_string(func: Callable):
    if isinstance(func, OpOverload):
        return func.name()
    else:
        return func
        # full_name = f'{inspect.getmodule(func).__name__}.{func.__name__}'
        # assert get_object_from_string(full_name) == func
        # return full_name


class Converter:
    def __init__(self):
        self.node_converter_dict = {}

    def register_convertion_function_dict(self, convertion_function_dict: Dict[str, Callable[[fx.Node], Node]]) -> None:
        for k, v in convertion_function_dict.items():
            assert k not in self.node_converter_dict, f"Duplicate key: {k}"
        self.node_converter_dict.update(convertion_function_dict)

    def get_converter(self, target) -> Callable[[list, dict, Location], Node]:
        if isinstance(target, OpOverload):
            func = target.name()
        else:
            func = target
        assert func in self.node_converter_dict, f"Unregistered function: {func}"
        return self.node_converter_dict.get(func)

    def convert_node(self, node: fx.Node, node_mapping: Callable[[fx.Node], Any]) -> Node:
        if node.op == 'placeholder':
            assert isinstance(node.target, str)
            return core_dialect.Placeholder(NamedLocation(node.target))
        elif node.op == 'call_method':
            raise NotImplementedError("call_method is not supported")
        elif node.op == 'call_module':
            raise NotImplementedError("call_module is not supported")
        elif node.op == 'call_function':
            args = [node_mapping(arg) for arg in node.args]
            attrs = {k: node_mapping(v) for k, v in node.kwargs.items()}
            converter = self.get_converter(node.target)
            return converter(args, attrs, NamedLocation(node.name))
        elif node.op == 'get_attr':
            raise NotImplementedError("get_attr is not supported")
        elif node.op == 'output':
            raise ValueError("Do not put output node in the middle of the graph")
        else:
            raise ValueError(f"Unsupported op: {node.op}")
