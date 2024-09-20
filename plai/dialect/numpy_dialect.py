from plai.core import module
from plai.core.location import Location


class NumpyNode(module.Node):

    @classmethod
    def get_namespace(cls):
        return 'torch.nn'


class Relu(NumpyNode):
    def __init__(self, arg: module.Node, loc: Location = None):
        super().__init__([arg], {}, loc)
