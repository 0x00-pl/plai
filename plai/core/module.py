import re
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from plai.core.location import Location


class Node(ABC):
    def __init__(self, operands: List['Node'], attrs: dict, loc: Location = None):
        self.operands = operands
        self.attrs = attrs
        self.loc = loc

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

    @staticmethod
    def get_node_class(op_name: str):
        assert op_name in Node.subclass_dict, f'Unregister Class with name: {op_name}'
        return Node.subclass_dict[op_name]

    def to_string(self, node_name_dict: Dict['Node', str]):
        return f'{self.get_op_name()}({", ".join(node_name_dict[i] for i in self.operands)}) ' \
               f'{self.attrs if self.attrs else ""}'

    @staticmethod
    def static_to_string(node: 'Node', node_name_dict: Dict['Node', str]):
        if node is None:
            return 'None'
        else:
            return node.to_string(node_name_dict)


class Graph:
    def __init__(self, name=''):
        self.name = name
        self.arguments: List[Node] = []
        self.nodes: List[Node] = []
        self.outputs: List[Node] = []
        self.insert_point_index = 0

    def add_argument(self, node: Node):
        self.arguments.append(node)

    def add_output(self, node: Node):
        # node maybe is None
        self.outputs.append(node)

    def set_insert_point_after(self, node: Node = None):
        if node is None:
            self.insert_point_index = len(self.nodes)
        else:
            self.insert_point_index = self.nodes.index(node) + 1

    def set_insert_point_before(self, node: Node = None):
        if node is None:
            self.insert_point_index = 0
        else:
            self.insert_point_index = self.nodes.index(node)

    def add_node(self, node: Node):
        self.nodes.insert(self.insert_point_index, node)

    def remove_node(self, node: Node):
        remove_index = self.nodes.index(node)
        self.nodes.pop(remove_index)
        if remove_index < self.insert_point_index:
            self.insert_point_index -= 1

    def __str__(self):
        node_name_dict: Dict[Optional[Node], str] = {None: 'None'}
        node_name_dict = node_name_dict | {node: f'arg{idx}' for idx, node in enumerate(self.arguments)}
        node_name_dict = node_name_dict | {node: f'v{idx}' for idx, node in enumerate(self.nodes)}

        result = f'Graph {self.name}({", ".join(node_name_dict[i] for i in self.arguments)}): \n'
        for idx, node in enumerate(self.nodes):
            name = node_name_dict[node]
            result += f'  {idx}: {name} = {node.to_string(node_name_dict)}\n'
        result += f'  output ({", ".join(node_name_dict[i] for i in self.outputs)})\n'
        return result
