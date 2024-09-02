import importlib
import inspect
from typing import Callable, Any

import torch
from torch import fx
from torch._ops import OpOverload

from plai.core import core_dialect
from plai.core.location import NamedLocation
from plai.core.module import Node
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


def convert_node(node: fx.Node, node_mapping: Callable[[fx.Node], Any]) -> Node:
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
        attrs = {k: node_mapping(v) for k, v in node.kwargs.items()}
        if func_name == 'aten::view':
            return aten_dialect.View(args[0], args[1], NamedLocation(node.name))
        elif func_name == 'aten::detach':
            return aten_dialect.Relu(args[0], NamedLocation(node.name))
        elif func_name == 'aten::t':
            return core_dialect.Transpose(args[0], NamedLocation(node.name))
        elif func_name == 'aten::addmm':
            return aten_dialect.AddMm(
                args[0], args[1], args[2],
                attrs.get('beta', 1), attrs.get('alpha', 1),
                NamedLocation(node.name)
            )
        elif func_name == 'aten::mm':
            return aten_dialect.Mm(args[0], args[1], NamedLocation(node.name))
        elif func_name == 'aten::relu':
            return aten_dialect.Relu(args[0], NamedLocation(node.name))
        elif func_name == 'aten::max.dim':
            keepdim = args[2] if len(args) == 3 else False
            return aten_dialect.Max(args[0], args[1], keepdim, NamedLocation(node.name))
        elif func_name == 'aten::sum.dim_IntList':
            return aten_dialect.Sum(args[0], args[1], args[2], NamedLocation(node.name))
        elif func_name == 'aten::threshold_backward':
            return aten_dialect.ThresholdBackward(args[0], args[1], args[2], NamedLocation(node.name))
        elif func_name == 'torch._C._nn.linear':
            return torch_dialect.Linear(args[0], args[1], args[2], NamedLocation(node.name))
        elif func_name == 'torch.relu':
            return torch_dialect.Relu(args[0], NamedLocation(node.name))
        elif func_name == '_operator.getitem':
            return torch_dialect.GetItem(args[0], args[1], NamedLocation(node.name))
        else:
            raise NotImplementedError(f"Unsupported function: {func_name}")
    elif node.op == 'get_attr':
        raise NotImplementedError("get_attr is not supported")
    elif node.op == 'output':
        raise ValueError("Do not put output node in the middle of the graph")
    else:
        raise ValueError(f"Unsupported op: {node.op}")
