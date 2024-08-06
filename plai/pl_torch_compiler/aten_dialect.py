from plai.core import module
from plai.core.location import Location


class AddMm(module.Node):
    def __init__(self, bias: module.Node, mat1: module.Node, mat2: module.Node, beta, alpha, loc: Location = None):
        """
        out = beta * bias + alpha * (mat1 * mat2)
        """
        super().__init__('addmm', [bias, mat1, mat2], {'beta': beta, 'alpha': alpha}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'addmm'
        return AddMm(args[0], args[1], args[2], attrs['beta'], attrs['alpha'], loc)


class Relu(module.Node):
    def __init__(self, arg: module.Node, loc: Location = None):
        super().__init__('relu', [arg], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'relu'
        return Relu(args[0], loc)


class Detach(module.Node):
    def __init__(self, arg: module.Node, loc: Location = None):
        super().__init__('detach', [arg], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'detach'
        return Detach(args[0], loc)
