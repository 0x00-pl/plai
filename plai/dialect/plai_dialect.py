from typing import List

from plai.core import module
from plai.core.location import Location


class PlaiNode(module.Node):
    @classmethod
    def get_namespace(cls):
        return 'plai'


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


def register_dialect():
    pass  # do nothing, only for registration this file.
