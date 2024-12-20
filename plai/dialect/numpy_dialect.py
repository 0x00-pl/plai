from abc import ABC

from plai.core.location import Location
from plai.core.node import Node


class NumpyNode(Node, ABC):

    @classmethod
    def get_namespace(cls):
        return 'torch.nn'


class Relu(NumpyNode):
    def __init__(self, arg: Node, loc: Location = None):
        super().__init__([arg], {}, loc)

    def update_type_notation(self):
        for operand in self.operands:
            operand.update_type_notation()
