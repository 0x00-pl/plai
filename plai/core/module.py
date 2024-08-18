from abc import ABC, abstractmethod
from typing import List, Dict, Optional

from plai.core.location import Location


class Value:
    def __init__(self, node: Optional['Node'], type_notation=None):
        self.node = node
        self.type_notation = type_notation

    def __str__(self):
        return f"{self.type_notation}"

    def owner(self):
        return self.node


class Node(ABC):
    def __init__(self, operands: List[Value], attrs: dict, loc: Location = None):
        self.operands = operands
        self.attrs = attrs
        self.loc = loc
        self.outputs = self.build_outputs()

    subclass_dict = {}

    @classmethod
    def get_namespace(cls):
        raise NotImplementedError(f"Class {cls.__name__} must override the get_namespace method.")

    @classmethod
    def get_op_name(cls):
        if cls.get_namespace() == '':
            return cls.__name__.lower()
        else:
            return f'{cls.get_namespace()}.{cls.__name__.lower()}'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        op_name = cls.get_op_name()
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

    def build_outputs(self):
        return [Value(self)]

    def get_output(self) -> Value:
        assert len(self.outputs) == 1, f"Node {self} with Type {self.get_op_name()} must have exactly one output."
        return self.outputs[0]

    def get_outputs(self) -> List[Value]:
        return self.outputs

    def to_string(self, value_name_dict: Dict[Value, str]):
        return f'{self.get_op_name()}({", ".join(value_name_dict[i] for i in self.operands)}) {self.attrs if self.attrs else ""}'

    @staticmethod
    def static_to_string(node: 'Node', value_name_dict: Dict[Value, str]):
        if node is None:
            return 'None'
        return node.to_string(value_name_dict)


class Graph:
    def __init__(self, name=''):
        self.name = name
        self.arguments: List[Value] = []
        self.nodes: List[Node] = []
        self.outputs: List[Value] = []

    def add_argument(self):
        return self.arguments.append(Value(None))

    def add_output(self, value: Value):
        # value maybe is None
        self.outputs.append(value)

    def add_node(self, node: Node):
        self.nodes.append(node)

    def __str__(self):
        value_name_dict: Dict[Optional[Value], str] = {None: 'None'}
        value_name_dict = value_name_dict | {node: f'arg{idx}' for idx, node in enumerate(self.arguments)}
        value_name_dict = value_name_dict | {node: f'v{idx}' for idx, node in enumerate(self.nodes)}

        result = f'Graph {self.name}({", ".join(value_name_dict[i] for i in self.arguments)}): \n'
        for idx, node in enumerate(self.nodes):
            args_str = ', '.join(value_name_dict[i] for i in node.get_outputs())
            result += f'  {idx}: {args_str} = {node.to_string(value_name_dict)}\n'
        result += f'  output ({", ".join(value_name_dict[i] for i in self.outputs)})\n'
        return result
