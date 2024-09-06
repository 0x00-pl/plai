from abc import abstractmethod
from typing import Callable, Optional

from plai.core import module
from plai.core.location import Location


class TorchNode(module.Node):
    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        raise ValueError('this is a dialect, should not using Build.')

    @classmethod
    def get_namespace(cls):
        return 'torch'

    @staticmethod
    @abstractmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        pass

    @classmethod
    def register_overload(cls, register: Callable[[str, Optional[Callable]], None]):
        pass

    convertion_function_dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        op_name = cls.get_op_name(sep='::')
        assert op_name not in TorchNode.convertion_function_dict, f'Duplicate key: {op_name}'
        TorchNode.convertion_function_dict[op_name] = cls.from_torch

        def _register_overload_inner(overload, func=None):
            if func is None:
                func = cls.from_torch
            op_name_overload = f'{op_name}.{overload}'
            assert op_name_overload not in TorchNode.convertion_function_dict, f'Duplicate key: {op_name_overload}'
            TorchNode.convertion_function_dict[op_name_overload] = func

        cls.register_overload(_register_overload_inner)


class Linear(TorchNode):
    def __init__(self, arg: module.Node, weight: module.Node, bias: module.Node, loc: Location = None):
        """
        out = arg * weight.T + bias
        """
        super().__init__([arg, weight, bias], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'linear'
        return Linear(args[0], args[1], args[2], loc)

    @classmethod
    def get_cls_name(cls):
        return '_C._nn.linear'

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return Linear(args[0], args[1], args[2], loc)


class Relu(TorchNode):
    def __init__(self, arg: module.Node, loc: Location = None):
        super().__init__([arg], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'relu'
        return Relu(args[0], loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return Relu(args[0], loc)
