from abc import ABC

from plai.core import node
from plai.core.location import Location
from plai.core.type_notation import TypeNotation, UnknownType, TupleType


class CoreNode(node.Node, ABC):
    @classmethod
    def get_namespace(cls):
        return ''


class Placeholder(CoreNode):
    def __init__(self, type_notation: TypeNotation, loc: Location = None):
        super().__init__([], {}, loc)
        self.type_notation = type_notation

    def update_type_notation(self):
        pass


class Output(CoreNode):
    def __init__(self, loc: Location = None):
        super().__init__([], {}, loc)

    def add_argument(self, arg: node.Node):
        idx = len(self.operands)
        self.operands.append(None)
        self.set_operand(idx, arg)

    def update_type_notation(self):
        if isinstance(self._type_notation, UnknownType):
            for operand in self.operands:
                operand.update_type_notation()

            if len(self.operands) == 1:
                self.type_notation = node.Node.get_type_notation(self.operands[0])
            else:
                self.type_notation = TupleType([node.Node.get_type_notation(operand) for operand in self.operands])
