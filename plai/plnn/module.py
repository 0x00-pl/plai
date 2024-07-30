from typing import List, Any, Dict


class Arguments:
    def __init__(self, argument_name_list: List[str], args: List[Any], kwargs: Dict[str, Any]):
        self.kwargs = {}
        for name, value in zip(argument_name_list, args):
            self.kwargs[name] = value

        for name, value in kwargs:
            assert name not in self.kwargs
            self.kwargs[name] = value

    def get_arguments(self, name: str):
        return self.kwargs[name]


class Node:
    def __init__(self, name: str, op: str, args, kwargs):
        self.name = name
        self.op = op
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        if self.op == 'placeholder':
            return f"{self.name} = placeholder({self.args[0]})"
        else:
            return f"{self.name} = {self.op} {self.args}{self.kwargs}"

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
