from plai.core import module
from plai.core.location import Location


class AddMm(module.Node):
    def __init__(self, input: module.Node, mat1: module.Node, mat2: module.Node, beta, alpha, loc: Location = None):
        """
        out = beta * input + α * (mat1 * mat2)
        """
        super().__init__('addmm', [input, mat1, mat2], {'beta': beta, 'alpha': alpha}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'addmm'
        return AddMm(args[0], args[1], args[2], attrs['beta'], attrs['alpha'], loc)
