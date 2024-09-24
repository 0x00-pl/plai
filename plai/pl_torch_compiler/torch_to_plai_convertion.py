import importlib
import inspect
from typing import Callable, Any, Dict

from torch import fx
from torch._ops import OpOverload

from plai.core.location import NamedLocation, Location
from plai.core.module import Node


class Converter:
    def __init__(self):
        self.node_converter_dict = {}

    def register_convertion_function_dict(self, convertion_function_dict: Dict[str, Callable[[fx.Node], Node]]) -> None:
        for k, v in convertion_function_dict.items():
            assert k not in self.node_converter_dict, f"Duplicate key: {k}"
        self.node_converter_dict.update(convertion_function_dict)

    def get_converter(self, target) -> Callable[[list, dict, Location], Node]:
        if isinstance(target, OpOverload):
            func_name = target.name()
        else:
            module_name = inspect.getmodule(target).__name__
            func_name = f'{module_name}.{target.__name__}'

            try:
                checked_module = importlib.import_module(module_name)
                assert getattr(checked_module, target.__name__) == target, f"Unmatched function: {func_name}"
            except (ImportError, AttributeError, AssertionError):
                import warnings
                warnings.warn(f"verifying function failed: {func_name}")

        assert func_name in self.node_converter_dict, f"Unregistered function: {func_name}"
        return self.node_converter_dict.get(func_name)

    def convert_node(self, node: fx.Node, node_mapping: Callable[[fx.Node], Any]) -> Node:
        if node.op == 'call_method':
            raise NotImplementedError("call_method is not supported")
        elif node.op == 'call_module':
            raise NotImplementedError("call_module is not supported")
        elif node.op == 'call_function':
            args = [node_mapping(arg) for arg in node.args]
            attrs = {k: node_mapping(v) for k, v in node.kwargs.items()}
            converter = self.get_converter(node.target)
            return converter(args, attrs, NamedLocation(node.name))
        else:
            raise ValueError(f"Unsupported op: {node.op}")
