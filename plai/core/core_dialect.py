from plai.core import module
from plai.core.location import Location


class CoreNode(module.Node):
    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        raise ValueError('this is a dialect, should not using Build.')

    @classmethod
    def get_namespace(cls):
        return ''


class Placeholder(CoreNode):
    def __init__(self, loc: Location = None):
        super().__init__('placeholder', [], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'placeholder'
        return Placeholder(loc)


class Transpose(CoreNode):
    def __init__(self, arg: module.Node, loc: Location=None):
        super().__init__('transpose', [arg], {}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'transpose'
        return Transpose(args[0], loc)
