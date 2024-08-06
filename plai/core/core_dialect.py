from plai.core import module
from plai.core.location import Location


class Placeholder(module.Node):
    def __init__(self, loc: Location = None):
        super().__init__('placeholder', [], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'placeholder'
        return Placeholder(loc)


class Transpose(module.Node):
    def __init__(self, arg: module.Node, loc: Location=None):
        super().__init__('transpose', [arg], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'transpose'
        return Transpose(args[0], loc)
