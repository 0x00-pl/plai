import abc
from abc import abstractmethod
from typing import Callable

from plai.core.location import Location
from plai.core.node import Node
from plai.core.type_notation import TupleType


class TorchNode(Node, abc.ABC):
    @classmethod
    def get_namespace(cls):
        return 'torch'

    @staticmethod
    @abstractmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        pass

    @classmethod
    def register_torch_overload(cls, register: Callable[[str, Callable], None]):
        register(cls.get_op_name('::'), cls.from_torch)

    convertion_function_dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        def _register_torch_overload_inner(name: str, func: Callable) -> None:
            assert name not in TorchNode.convertion_function_dict, f'Duplicate key: {name}'
            TorchNode.convertion_function_dict[name] = func

        cls.register_torch_overload(_register_torch_overload_inner)


class GetItem(TorchNode):
    def __init__(self, arg: Node, key, loc: Location = None):
        super().__init__([arg], {'key': key}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return GetItem(args[0], args[1], loc)

    @classmethod
    def register_torch_overload(cls, register: Callable[[str, Callable], None]):
        name = '_operator.getitem'
        register(name, cls.from_torch)

    def inference_type_notation(self):
        key = self.attrs['key']
        value_type = Node.get_type_notation(self.operands[0])
        if isinstance(value_type, TupleType) and isinstance(key, int):
            return value_type.types[key]
        else:
            raise ValueError(f'Unsupported key type: {type(key)} for value type: {value_type}')


class Linear(TorchNode):
    def __init__(self, arg: Node, weight: Node, bias: Node, loc: Location = None):
        """
        out = arg * weight.T + bias
        """
        super().__init__([arg, weight, bias], {}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return Linear(args[0], args[1], args[2], loc)

    @classmethod
    def register_torch_overload(cls, register: Callable[[str, Callable], None]):
        name = f'{cls.get_namespace()}._C._nn.linear'
        register(name, cls.from_torch)


class Relu(TorchNode):
    def __init__(self, arg: Node, loc: Location = None):
        super().__init__([arg], {}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return Relu(args[0], loc)

    @classmethod
    def register_torch_overload(cls, register: Callable[[str, Callable], None]):
        name = f'{cls.get_namespace()}.relu'
        register(name, cls.from_torch)
