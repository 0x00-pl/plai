from plai.core import module
from plai.core.location import Location


class Placeholder(module.Node):
    def __init__(self, loc: Location = None):
        super().__init__('placeholder', [], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        return Placeholder(loc)


class Transpose(module.Node):
    def __init__(self, arg: module.Node, loc: Location=None):
        super().__init__('transpose', [arg], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        return Transpose(args[0], loc)
