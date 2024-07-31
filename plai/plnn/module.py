from typing import List, Any, Dict

from plai.plnn.source_map import Location


#
# class Arguments:
#     def __init__(self, argument_name_list: List[str], args: List[Any], kwargs: Dict[str, Any]):
#         self.kwargs = {}
#         for name, value in zip(argument_name_list, args):
#             self.kwargs[name] = value
#
#         for name, value in kwargs:
#             assert name not in self.kwargs
#             self.kwargs[name] = value
#
#     def get_arguments(self, name: str):
#         return self.kwargs[name]

class Node:
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
    def build(op_name: str, operands: list, attrs: dict, loc: Location = None):
        assert op_name in Node.subclass_dict, f'Unregister Class with name: {op_name}'
        op_cls = Node.subclass_dict[op_name]
        assert op_cls.build is not Node.build, f"Class {op_cls.__name__} must override the build method."
        return op_cls.build(op_name, operands, attrs, loc)

    def to_string(self, node_name_dict: Dict['Node', str]):
        return f'{self.op}({", ".join(node_name_dict[i] for i in self.operands)})'

    def __repr__(self):
        return self.name


class Graph:
    def __init__(self, name=''):
        self.name = name
        self.arguments_name_list: List[str] = []
        self.nodes: List[Node] = []
        self.outputs: List[Node] = []

    def add_argument(self, name, argument_name):
        self.arguments_name_list.append(argument_name)
        return self.add_node(name, 'placeholder', [argument_name], {})

    def add_output(self, outputs):
        self.outputs = list(outputs)

    def add_node(self, name, op, args, kwargs):
        new_node = Node(name, op, args, kwargs)
        self.nodes.append(new_node)
        return new_node

    def __str__(self):
        result = f'Graph {self.name}({", ".join(self.arguments_name_list)}): \n'
        for idx, node in enumerate(self.nodes):
            result += f'  {idx}: {node}\n'
        result += f'  output ({", ".join(i.name if i is not None else str(None) for i in self.outputs)})\n'
        return result
