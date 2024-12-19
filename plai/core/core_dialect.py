from abc import ABC

from plai.core import node
from plai.core.location import Location
from plai.core.type_notation import TypeNotation, UnknownType, TupleType


class CoreNode(node.Node, ABC):
    @classmethod
    def get_namespace(cls):
        return ''


class Placeholder(CoreNode):
    def __init__(self, placeholder_type: TypeNotation, loc: Location = None):
        super().__init__([], {}, loc)
        self.placeholder_type = placeholder_type

    def update_type_notation(self):
        return self.placeholder_type


class Output(CoreNode):
    def __init__(self, loc: Location = None):
        super().__init__([], {}, loc)

    def add_argument(self, arg: node.Node):
        idx = len(self.operands)
        self.operands.append(None)
        self.set_operand(idx, arg)

    def update_type_notation(self):
        if len(self.operands) == 1:
            return node.Node.get_type_notation(self.operands[0])
        else:
            return TupleType([node.Node.get_type_notation(operand) for operand in self.operands])
