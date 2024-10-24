from typing import List

from plai.core import module
from plai.core.location import Location


class PlaiNode(module.Node):
    @classmethod
    def get_namespace(cls):
        return 'plai'


class Constant(PlaiNode):
    def __init__(self, value, loc: Location = None):
        super().__init__([], {'value': value}, loc)

    def get_value(self):
        return self.attrs['value']


class Transpose(PlaiNode):
    def __init__(self, arg: module.Node, permutation: List = None, loc: Location = None):
        permutation = permutation if permutation is not None else [1, 0]
        super().__init__([arg], {'permutation': permutation}, loc)


class Relu(PlaiNode):
    def __init__(self, arg: module.Node, loc: Location = None):
        super().__init__([arg], {}, loc)


class AddMm(PlaiNode):
    def __init__(self, bias: module.Node, mat1: module.Node, mat2: module.Node, beta, alpha, loc: Location = None):
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
    def __init__(self, arg1: module.Node, arg2: module.Node, loc: Location = None):
        super().__init__([arg1, arg2], {}, loc)


class Mul(PlaiNode):
    def __init__(self, arg1: module.Node, arg2: module.Node, loc: Location = None):
        super().__init__([arg1, arg2], {}, loc)


class MatMul(PlaiNode):
    def __init__(self, arg1: module.Node, arg2: module.Node, loc: Location = None):
        super().__init__([arg1, arg2], {}, loc)


def register_dialect():
    pass  # do nothing, only for registration this file.
