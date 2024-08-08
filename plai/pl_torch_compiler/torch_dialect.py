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


class Mm(module.Node):
    def __init__(self, mat1: module.Node, mat2: module.Node, loc: Location = None):
        """
        out = mat1 * mat2
        """
        super().__init__('mm', [mat1, mat2], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'mm'
        return Mm(args[0], args[1], loc)


class Relu(module.Node):
    def __init__(self, arg: module.Node, loc: Location = None):
        super().__init__('relu', [arg], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'relu'
        return Relu(args[0], loc)


class Sum(module.Node):
    def __init__(self, arg: module.Node, dims: [int], keepdim: bool, loc: Location = None):
        super().__init__('sum', [arg], {'dims': dims, 'keepdim': keepdim}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'sum'
        return Sum(args[0], attrs['dims'], attrs['keepdim'], loc)


class ThresholdBackward(module.Node):
    def __init__(self, grad_output: module.Node, arg: module.Node, threshold: float, loc: Location = None):
        super().__init__('threshold_backward', [grad_output, arg], {'threshold': threshold}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'threshold_backward'
        return ThresholdBackward(args[0], args[1], args[2], loc)


class Detach(module.Node):
    def __init__(self, arg: module.Node, loc: Location = None):
        super().__init__('detach', [arg], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'detach'
        return Detach(args[0], loc)


class View(module.Node):
    def __init__(self, arg: module.Node, shape: [int], loc: Location = None):
        super().__init__('view', [arg], {'shape': shape}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'view'
        return View(args[0], attrs['shape'], loc)
