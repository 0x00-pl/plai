from abc import ABC
from typing import Callable

from plai.core.location import Location
from plai.core.node import Node
from plai.dialect.torch_dialect import TorchNode


class AtenNode(TorchNode, ABC):
    @classmethod
    def get_namespace(cls):
        return 'aten'


class Addmm(AtenNode):
    def __init__(self, bias: Node, mat1: Node, mat2: Node, beta, alpha, loc: Location = None):
        """
        out = beta * bias + alpha * (mat1 * mat2)
        """
        super().__init__([bias, mat1, mat2], {'beta': beta, 'alpha': alpha}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return Addmm(args[0], args[1], args[2], attrs.get('beta', 1), attrs.get('alpha', 1), loc)


class Mm(AtenNode):
    def __init__(self, mat1: Node, mat2: Node, loc: Location = None):
        """
        out = mat1 * mat2
        """
        super().__init__([mat1, mat2], {}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return Mm(args[0], args[1], loc)


class Sum(AtenNode):
    def __init__(self, arg: Node, dims: [int], keepdim: bool, loc: Location = None):
        super().__init__([arg], {'dims': dims, 'keepdim': keepdim}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return Sum(args[0], args[1], args[2], loc)

    @staticmethod
    def from_torch_overload_dim(args: list, attrs: dict, loc: Location = None):
        return Sum(args[0], args[1], args[2], loc)

    @classmethod
    def register_torch_overload(cls, register: Callable[[str, Callable], None]):
        register('torch::sum', cls.from_torch)
        register('torch::sum.dim_IntList', cls.from_torch_overload_dim)


class Relu(AtenNode):
    def __init__(self, arg: Node, loc: Location = None):
        super().__init__([arg], {}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return Relu(args[0], loc)


class Max(AtenNode):
    def __init__(self, arg: Node, dim: int, keepdim: bool, loc: Location = None):
        super().__init__([arg], {'dim': dim, 'keepdim': keepdim}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        raise NotImplementedError('aten.max is not implemented without overload')

    @staticmethod
    def from_torch_overload_dim(args: list, attrs: dict, loc: Location = None):
        if len(args) < 3:
            args.append(False)
        return Max(args[0], args[1], args[2], loc)

    @classmethod
    def register_torch_overload(cls, register: Callable[[str, Callable], None]):
        register(f'{cls.get_namespace()}::max.dim', cls.from_torch_overload_dim)


class ThresholdBackward(AtenNode):
    def __init__(self, grad_output: Node, arg: Node, threshold: float, loc: Location = None):
        super().__init__([grad_output, arg], {'threshold': threshold}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return ThresholdBackward(args[0], args[1], args[2], loc)


class View(AtenNode):
    def __init__(self, arg: Node, shape: [int], loc: Location = None):
        super().__init__([arg], {'shape': shape}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return View(args[0], args[1], loc)


class Transpose(AtenNode):
    def __init__(self, arg: Node, loc: Location = None):
        super().__init__([arg], {}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return Transpose(args[0], loc)

    @classmethod
    def register_torch_overload(cls, register: Callable[[str, Callable], None]):
        register(f'{cls.get_namespace()}::t', cls.from_torch)


class Detach(AtenNode):
    def __init__(self, arg: Node, loc: Location = None):
        super().__init__([arg], {}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return Detach(args[0], loc)


def register_dialect():
    pass  # do nothing, only for registration this file.
