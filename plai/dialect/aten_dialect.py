from abc import ABC
from typing import Callable

from plai.core import type_notation
from plai.core.location import Location
from plai.core.node import Node
from plai.core.type_notation import ScalarType, TensorType, TypeNotation
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

    def inference_type_notation(self):
        assert len(self.operands) == 3, f'Addmm should have 3 operands, but got {len(self.operands)}'
        [bias, mat1, mat2] = self.operands
        bias_type = Node.get_type_notation(bias)
        mat1_type = Node.get_type_notation(mat1)
        mat2_type = Node.get_type_notation(mat2)
        assert isinstance(bias_type,
                          (ScalarType, TensorType)), f'Addmm bias should be scalar or tensor, but got {bias_type}'
        assert isinstance(mat1_type, TensorType), f'Addmm mat1 should be tensor, but got {mat1_type}'
        assert isinstance(mat2_type, TensorType), f'Addmm mat2 should be tensor, but got {mat2_type}'
        assert mat1_type.element_type == mat2_type.element_type, f'Addmm mat1 and mat2 should have same dtype, but got {mat1_type} and {mat2_type}'
        assert len(mat1_type.shape) >= 1, f'Addmm mat1 should have at least 1 dimensions, but got {mat1_type}'
        assert len(mat2_type.shape) >= 2, f'Addmm mat2 should have at least 2 dimensions, but got {mat2_type}'
        assert (  #
                mat1_type.shape[-1] == mat2_type.shape[-2]  #
        ), f'Addmm mat1 and mat2 should be compatible, but got {mat1_type} and {mat2_type}'

        if len(mat1_type.shape) == 1:
            last_dim = mat1_type.shape[-1]
            out_shape = mat2_type.shape[-2:] + [last_dim]
        else:
            out_shape = type_notation.broadcast_shape(mat1_type.shape[-2:], mat2_type.shape[-2:])
            out_shape = out_shape + [mat1_type.shape[-2], mat2_type.shape[-1]]

        out_type = TensorType(out_shape, mat1_type.element_type)
        return out_type


class Mm(AtenNode):
    def __init__(self, mat1: Node, mat2: Node, loc: Location = None):
        """
        out = mat1 * mat2
        """
        super().__init__([mat1, mat2], {}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return Mm(args[0], args[1], loc)

    def inference_type_notation(self):
        assert len(self.operands) == 2, f'Mm should have 2 operands, but got {len(self.operands)}'
        [mat1, mat2] = self.operands
        mat1_type = Node.get_type_notation(mat1)
        mat2_type = Node.get_type_notation(mat2)
        assert isinstance(mat1_type, TensorType), f'Mm mat1 should be tensor, but got {mat1_type}'
        assert isinstance(mat2_type, TensorType), f'Mm mat2 should be tensor, but got {mat2_type}'
        assert len(mat1_type.shape) >= 2, f'Mm mat1 should have at least 2 dimensions, but got {mat1_type}'
        assert len(mat2_type.shape) >= 2, f'Mm mat2 should have at least 2 dimensions, but got {mat2_type}'
        assert (  #
                mat1_type.shape[-1] == mat2_type.shape[-2]  #
        ), f'Mm mat1 and mat2 should be compatible, but got {mat1_type} and {mat2_type}'

        out_shape = type_notation.broadcast_shape(mat1_type.shape[:-1], mat2_type.shape[:-2])
        out_shape = out_shape + [mat1_type.shape[-2], mat2_type.shape[-1]]

        out_type = TensorType(out_shape, mat1_type.element_type)
        return out_type


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

    def inference_type_notation(self) -> TypeNotation:
        assert len(self.operands) == 1, f'Sum should have 1 operand, but got {len(self.operands)}'
        [arg] = self.operands
        arg_type = Node.get_type_notation(arg)
        assert isinstance(arg_type, TensorType), f'Sum arg should be tensor, but got {arg_type}'
        assert len(arg_type.shape) >= 1, f'Sum arg should have at least 1 dimensions, but got {arg_type}'

        out_shape = []
        for index, size in enumerate(arg_type.shape):
            if index in self.attrs['dims']:
                if self.attrs['keepdim']:
                    out_shape.append(1)
            else:
                out_shape.append(size)

        out_type = TensorType(out_shape, arg_type.element_type)
        return out_type


class Relu(AtenNode):
    def __init__(self, arg: Node, loc: Location = None):
        super().__init__([arg], {}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return Relu(args[0], loc)

    def inference_type_notation(self) -> TypeNotation:
        assert len(self.operands) == 1, f'Relu should have 1 operand, but got {len(self.operands)}'
        [arg] = self.operands
        arg_type = Node.get_type_notation(arg)
        assert isinstance(arg_type, TensorType), f'Relu arg should be tensor, but got {arg_type}'
        assert len(arg_type.shape) >= 1, f'Relu arg should have at least 1 dimensions, but got {arg_type}'
        return arg_type


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

    def inference_type_notation(self) -> TypeNotation:
        assert len(self.operands) == 1, f'Max should have 1 operand, but got {len(self.operands)}'
        [arg] = self.operands
        arg_type = Node.get_type_notation(arg)
        assert isinstance(arg_type, TensorType), f'Max arg should be tensor, but got {arg_type}'
        assert len(arg_type.shape) >= 1, f'Max arg should have at least 1 dimensions, but got {arg_type}'
        assert self.attrs['dim'] < len(arg_type.shape), f'Max dim should be less than arg dimensions, but got {self.attrs["dim"]} and {len(arg_type.shape)}'
        if self.attrs['keepdim']:
            out_shape = arg_type.shape[:self.attrs['dim']] + [1] + arg_type.shape[self.attrs['dim'] + 1:]
        else:
            out_shape = arg_type.shape[:self.attrs['dim']] + arg_type.shape[self.attrs['dim'] + 1:]

        return TensorType(out_shape, arg_type.element_type)


class ThresholdBackward(AtenNode):
    def __init__(self, grad_output: Node, arg: Node, threshold: float, loc: Location = None):
        super().__init__([grad_output, arg], {'threshold': threshold}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return ThresholdBackward(args[0], args[1], args[2], loc)

    def inference_type_notation(self) -> TypeNotation:
        assert len(self.operands) == 2, f'ThresholdBackward should have 2 operands, but got {len(self.operands)}'
        [grad_output, arg] = self.operands
        grad_output_type = Node.get_type_notation(grad_output)
        arg_type = Node.get_type_notation(arg)
        assert isinstance(grad_output_type, TensorType), f'ThresholdBackward grad_output should be tensor, but got {grad_output_type}'
        assert isinstance(arg_type, TensorType), f'ThresholdBackward arg should be tensor, but got {arg_type}'
        assert len(grad_output_type.shape) >= 1, f'ThresholdBackward grad_output should have at least 1 dimensions, but got {grad_output_type}'
        assert len(arg_type.shape) >= 1, f'ThresholdBackward arg should have at least 1 dimensions, but got {arg_type}'
        assert grad_output_type.shape == arg_type.shape, f'ThresholdBackward grad_output and arg should have same shape, but got {grad_output_type} and {arg_type}'

        return grad_output_type


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

    def inference_type_notation(self) -> TypeNotation:
        assert len(self.operands) == 1, f'Transpose should have 1 operand, but got {len(self.operands)}'
        [arg] = self.operands
        arg_type = Node.get_type_notation(arg)
        assert isinstance(arg_type, TensorType), f'Transpose arg should be tensor, but got {arg_type}'
        assert len(arg_type.shape) >= 2, f'Transpose arg should have at least 2 dimensions, but got {arg_type}'
        out_shape = arg_type.shape[:-2] + [arg_type.shape[-1], arg_type.shape[-2]]
        out_type = TensorType(out_shape, arg_type.element_type)
        return out_type


class Detach(AtenNode):
    def __init__(self, arg: Node, loc: Location = None):
        super().__init__([arg], {}, loc)

    @staticmethod
    def from_torch(args: list, attrs: dict, loc: Location = None):
        return Detach(args[0], loc)


def register_dialect():
    pass  # do nothing, only for registration this file.
