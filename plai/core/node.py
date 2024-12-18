import re
from abc import ABC, abstractmethod
from typing import List, Dict, Union

from plai.core.location import Location
from plai.core.type_notation import TypeNotation, UnknownType, NoneType


class Node(ABC):
    def __init__(self, operands: List['Node'], attrs: dict, loc: Location = None):
        self.operands: List[Union['Node', None]] = [None] * len(operands)
        self.attrs = attrs
        self.loc = loc
        self.dead = False
        self.users = set()
        self._type_notation: TypeNotation = UnknownType()

        for idx, operand in enumerate(operands):
            self.set_operand(idx, operand)

        self.update_type_notation()

    @classmethod
    @abstractmethod
    def get_namespace(cls):
        pass

    @classmethod
    def get_cls_name(cls):
        camel_case = cls.__name__
        snake_case = re.sub(r'([A-Z])', r'_\1', camel_case).lower().lstrip('_')
        return snake_case

    @classmethod
    def get_op_name(cls, sep='.'):
        if cls.get_namespace() == '':
            return cls.get_cls_name()
        else:
            return f'{cls.get_namespace()}{sep}{cls.get_cls_name()}'

    subclass_dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        op_name = cls.get_op_name()
        assert op_name not in Node.subclass_dict
        Node.subclass_dict[op_name] = cls

    def add_user(self, user):
        self.users.add(user)

    def remove_user(self, user):
        if user in self.users:
            self.users.remove(user)

    def set_operand(self, idx: int, new_operand: 'Node'):
        old_operand = self.operands[idx]
        if old_operand is not None:
            old_operand.remove_user(self)
        self.operands[idx] = new_operand
        if new_operand is not None:
            new_operand.add_user(self)

        self._type_notation = UnknownType()

    def replace_operand(self, old_operand: 'Node', new_operand: 'Node'):
        for idx, operand in enumerate(self.operands):
            if operand == old_operand:
                self.set_operand(idx, new_operand)

    def remove(self):
        self.dead = True
        for operand in self.operands:
            if operand is not None:
                operand.remove_user(self)

    @staticmethod
    def get_node_class(op_name: str):
        assert op_name in Node.subclass_dict, f'Unregister Class with name: {op_name}'
        return Node.subclass_dict[op_name]

    @abstractmethod
    def update_type_notation(self) -> TypeNotation:
        pass

    @staticmethod
    def get_type_notation(node) -> TypeNotation:
        if isinstance(node, Node):
            if isinstance(node._type_notation, UnknownType):
                for operand in node.operands:
                    Node.get_type_notation(operand)

                node._type_notation = node.update_type_notation()
            return node._type_notation
        elif node is None:
            return NoneType()
        else:
            raise TypeError(f'Unsupported type: {type(node)}')

    def to_string(self, node_name_dict: Dict['Node', str]):
        return f'{self.get_op_name()}({", ".join(node_name_dict[i] for i in self.operands)}) ' \
               f'{self.attrs if self.attrs else ""}'

    @staticmethod
    def static_to_string(node: 'Node', node_name_dict: Dict['Node', str]):
        if node is None:
            return 'None'
        else:
            return node.to_string(node_name_dict)
