from plai.core import node
from plai.core.location import Location


class CoreNode(node.Node):
    @classmethod
    def get_namespace(cls):
        return ''


class Placeholder(CoreNode):
    def __init__(self, loc: Location = None):
        super().__init__([], {}, loc)


class Output(CoreNode):
    def __init__(self, loc: Location = None):
        super().__init__([], {}, loc)

    def add_argument(self, arg: node.Node):
        idx = len(self.operands)
        self.operands.append(None)
        self.set_operand(idx, arg)
