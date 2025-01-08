from abc import ABC
from typing import List

import numpy

from plai.core.location import Location
from plai.core.node import Node
from plai.core.type_notation import TypeNotation, ScalarType, TensorType, UnknownType, NoneType, broadcast_shape


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

    def inference_type_notation(self) -> TypeNotation:
        assert len(self.operands) == 1, 'Relu node should have exactly one operand'
        operand_type = Node.get_type_notation(self.operands[0])
        assert isinstance(operand_type, TensorType), 'Relu operand should be a tensor'
        return TensorType(operand_type.shape, operand_type.element_type)


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

    def inference_type_notation(self) -> TypeNotation:
        assert len(self.operands) == 3, 'AddMm node should have exactly three operands'
        bias_type = Node.get_type_notation(self.get_bias())
        mat1_type = Node.get_type_notation(self.get_mat1())
        mat2_type = Node.get_type_notation(self.get_mat2())
        assert isinstance(bias_type, (TensorType, ScalarType, NoneType)), 'AddMm bias should be a tensor or scalar'
        assert isinstance(mat1_type, TensorType), 'AddMm mat1 should be a tensor'
        assert isinstance(mat2_type, TensorType), 'AddMm mat2 should be a tensor'
        assert mat1_type.element_type == mat2_type.element_type, 'AddMm mat1 and mat2 should have the same element type'
        element_type = mat1_type.element_type

        mat1_shape = mat1_type.shape
        mat2_shape = mat2_type.shape
        if len(mat1_shape) == 1:
            return TensorType(mat2_shape[:-2] + [mat2_type.shape[-1]], element_type)
        else:
            common_shape = broadcast_shape(mat1_shape[:-1], mat2_shape[:-2])
            return TensorType(common_shape + [mat2_type.shape[-1]], element_type)


class Add(PlaiNode):
    def __init__(self, arg1: Node, arg2: Node, loc: Location = None):
        super().__init__([arg1, arg2], {}, loc)

    def inference_type_notation(self) -> TypeNotation:
        assert len(self.operands) == 2, 'Add node should have exactly two operands'
        operand1_type = Node.get_type_notation(self.operands[0])
        operand2_type = Node.get_type_notation(self.operands[1])

        assert isinstance(operand1_type, TensorType), 'Add operand1 should be a tensor'
        assert isinstance(operand2_type, TensorType), 'Add operand2 should be a tensor'
        assert operand1_type.element_type == operand2_type.element_type, 'Add operands should have the same element type'
        element_type = operand1_type.element_type
        shape1 = operand1_type.shape
        shape2 = operand2_type.shape
        common_shape = broadcast_shape(shape1, shape2)
        return TensorType(common_shape, element_type)


class Mul(PlaiNode):
    def __init__(self, arg1: Node, arg2: Node, loc: Location = None):
        super().__init__([arg1, arg2], {}, loc)

    def inference_type_notation(self) -> TypeNotation:
        assert len(self.operands) == 2, 'Mul node should have exactly two operands'
        operand1_type = Node.get_type_notation(self.operands[0])
        operand2_type = Node.get_type_notation(self.operands[1])

        assert isinstance(operand1_type, TensorType), 'Mul operand1 should be a tensor'
        assert isinstance(operand2_type, TensorType), 'Mul operand2 should be a tensor'
        assert operand1_type.element_type == operand2_type.element_type, 'Mul operands should have the same element type'
        element_type = operand1_type.element_type
        shape1 = operand1_type.shape
        shape2 = operand2_type.shape
        common_shape = broadcast_shape(shape1, shape2)
        return TensorType(common_shape, element_type)


class MatMul(PlaiNode):
    def __init__(self, arg1: Node, arg2: Node, loc: Location = None):
        super().__init__([arg1, arg2], {}, loc)

    def inference_type_notation(self) -> TypeNotation:
        assert len(self.operands) == 2, 'MatMul node should have exactly two operands'
        operand1_type = Node.get_type_notation(self.operands[0])
        operand2_type = Node.get_type_notation(self.operands[1])

        assert isinstance(operand1_type, TensorType), 'MatMul operand1 should be a tensor'
        assert isinstance(operand2_type, TensorType), 'MatMul operand2 should be a tensor'
        assert operand1_type.element_type == operand2_type.element_type, 'MatMul operands should have the same element type'
        element_type = operand1_type.element_type
        shape1 = operand1_type.shape
        shape2 = operand2_type.shape
        common_shape = broadcast_shape(shape1[:-1], shape2[:-2])
        return TensorType(common_shape + [shape1[-1], shape2[-1]], element_type)


def register_dialect():
    pass  # do nothing, only for registration this file.
