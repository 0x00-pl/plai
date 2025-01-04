from abc import ABC

from plai.core.location import Location
from plai.core.node import Node
from plai.core.type_notation import TypeNotation


class NumpyNode(Node, ABC):

    @classmethod
    def get_namespace(cls):
        return 'torch.nn'


class Relu(NumpyNode):
    def __init__(self, arg: Node, loc: Location = None):
        super().__init__([arg], {}, loc)

    def inference_type_notation(self) -> TypeNotation:
        return Node.get_type_notation(self.operands[0])
