from abc import ABC, abstractmethod
from typing import List, Dict

from plai.core.location import Location


class Node(ABC):
    def __init__(self, op: str, operands: list, attrs: dict, loc: Location = None):
        self.op = op
        self.operands = operands
        self.attrs = attrs
        self.loc = loc

    subclass_dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        op_name = getattr(cls, 'op_name') if hasattr(cls, 'op_name') else cls.__name__
        op_name = op_name.lower()
        assert op_name not in Node.subclass_dict
        Node.subclass_dict[op_name] = cls

    @staticmethod
    def get_op_subclass(op_name: str):
        assert op_name in Node.subclass_dict, f'Unregister Class with name: {op_name}'
        return Node.subclass_dict[op_name]

    @staticmethod
    @abstractmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        op_cls = Node.get_op_subclass(op_name)
        assert op_cls.build is not Node.build, f"Class {op_cls.__name__} must override the build method."
        return op_cls.build(op_name, args, attrs, loc)

    def to_string(self, node_name_dict: Dict['Node', str]):
        return f'{self.op}({", ".join(node_name_dict[i] for i in self.operands)}) {self.attrs if self.attrs else ""}'

    @staticmethod
    def static_to_string(node: 'Node', node_name_dict: Dict['Node', str]):
        if node is None:
            return 'None'
        return node.to_string(node_name_dict)


class Graph:
    def __init__(self, name=''):
        self.name = name
        self.arguments: List[Node] = []
        self.nodes: List[Node] = []
        self.outputs: List[Node] = []

    def add_argument(self, node: Node):
        self.arguments.append(node)

    def add_output(self, node: Node):
        # none maybe is None
        self.outputs.append(node)

    def add_node(self, node: Node):
        self.nodes.append(node)

    def __str__(self):
        node_name_dict: Dict['Node', str] = {None: 'None'}
        node_name_dict = node_name_dict | {node: f'arg{idx}' for idx, node in enumerate(self.arguments)}
        node_name_dict = node_name_dict | {node: f'v{idx}' for idx, node in enumerate(self.nodes)}

        result = f'Graph {self.name}({", ".join(node_name_dict[i] for i in self.arguments)}): \n'
        for idx, node in enumerate(self.nodes):
            result += f'  {idx}: {node_name_dict[node]} = {node.to_string(node_name_dict)}\n'
        result += f'  output ({", ".join(node_name_dict[i] for i in self.outputs)})\n'
        return result
