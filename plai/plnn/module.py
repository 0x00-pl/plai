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
    def __init__(self, name: str, op: str, target, args, kwargs):
        self.name = name
        self.op = op
        self.target = target
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return f"{self.name} = {self.op}@{self.target} {self.args}{self.kwargs}"

    def __repr__(self):
        return self.name


class Graph:
    def __init__(self, name=''):
        self.name = name
        self.nodes: List[Node] = []

    def add_node(self, name, op, target, args, kwargs):
        new_node = Node(name, op, target, args, kwargs)
        self.nodes.append(new_node)
        return new_node

    def __str__(self):
        result = f'Graph {self.name}: \n'
        for idx, node in enumerate(self.nodes):
            result += f'{idx}: {node}\n'
        result += '\n'
        return result
