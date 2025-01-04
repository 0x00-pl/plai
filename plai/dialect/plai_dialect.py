from abc import ABC
from typing import List

import numpy

from plai.core.location import Location
from plai.core.node import Node
from plai.core.type_notation import TypeNotation, ScalarType, TensorType, UnknownType


class PlaiNode(Node, ABC):
    @classmethod
    def get_namespace(cls):
        return 'plai'


class Constant(PlaiNode):
    def __init__(self, value, loc: Location = None):
        super().__init__([], {'value': value}, loc)
        if isinstance(value, int):
            self.value_type = ScalarType('int')
        elif isinstance(value, float):
            self.value_type = ScalarType('float')
        elif isinstance(value, bool):
            self.value_type = ScalarType('bool')
        elif isinstance(value, numpy.ndarray):
            self.value_type = TensorType(value.shape, UnknownType())
        else:
            raise ValueError(f'Unsupported constant type: {type(value)}')

    def get_value(self):
        return self.attrs['value']

    def inference_type_notation(self) -> TypeNotation:
        assert self.operands == [], 'Constant node should not have operands'
        return self.value_type


class Transpose(PlaiNode):
    def __init__(self, arg: Node, permutation: List = None, loc: Location = None):
        permutation = permutation if permutation is not None else [1, 0]
        super().__init__([arg], {'permutation': permutation}, loc)

    def inference_type_notation(self) -> TypeNotation:
        assert len(self.operands) == 1, 'Transpose node should have exactly one operand'
        operand_type = Node.get_type_notation(self.operands[0])
        assert isinstance(operand_type, TensorType), 'Transpose operand should be a tensor'

        shape = operand_type.shape
        permutation = self.attrs['permutation']
        assert len(shape) == len(permutation), 'Permutation length should match operand shape length'
        new_shape = [shape[i] for i in permutation]
        return TensorType(new_shape, operand_type.element_type)


class Relu(PlaiNode):
    def __init__(self, arg: Node, loc: Location = None):
        super().__init__([arg], {}, loc)


class AddMm(PlaiNode):
    def __init__(self, bias: Node, mat1: Node, mat2: Node, beta, alpha, loc: Location = None):
        """
        out = beta * bias + alpha * (mat1 * mat2)
        """
        super().__init__([bias, mat1, mat2], {'beta': beta, 'alpha': alpha}, loc)

    @classmethod
    def get_cls_name(cls):
        return 'add_mm'

    def get_bias(self):
        return self.operands[0]

    def get_mat1(self):
        return self.operands[1]

    def get_mat2(self):
        return self.operands[2]

    def get_alpha(self):
        return self.attrs['alpha']

    def get_beta(self):
        return self.attrs['beta']


class Add(PlaiNode):
    def __init__(self, arg1: Node, arg2: Node, loc: Location = None):
        super().__init__([arg1, arg2], {}, loc)


class Mul(PlaiNode):
    def __init__(self, arg1: Node, arg2: Node, loc: Location = None):
        super().__init__([arg1, arg2], {}, loc)


class MatMul(PlaiNode):
    def __init__(self, arg1: Node, arg2: Node, loc: Location = None):
        super().__init__([arg1, arg2], {}, loc)


def register_dialect():
    pass  # do nothing, only for registration this file.
