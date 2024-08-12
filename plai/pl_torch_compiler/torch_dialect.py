from plai.core import module
from plai.core.location import Location


class TorchNode(module.Node):
    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        raise ValueError('this is a dialect, should not using Build.')

    @classmethod
    def get_namespace(cls):
        return 'torch.nn'


class Linear(TorchNode):
    def __init__(self, arg: module.Node, weight: module.Node, bias: module.Node, loc: Location = None):
        """
        out = arg * weight.T + bias
        """
        super().__init__('linear', [arg, weight, bias], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'linear'
        return Linear(args[0], args[1], args[2], loc)


class Relu(TorchNode):
    def __init__(self, arg: module.Node, loc: Location = None):
        super().__init__('relu', [arg], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'relu'
        return Relu(args[0], loc)
