import re
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Dict, Optional

from plai.core.location import Location


# class NodePattern:
#     def __init__(self, node_cls, operands: List['NodePattern|Node'], attrs: dict):
#         self.node_cls = node_cls
#         self.operands = operands
#         self.attrs = attrs


class Node(ABC):
    def __init__(self, operands: List['Node'], attrs: dict, loc: Location = None):
        self.operands = operands
        self.attrs = attrs
        self.loc = loc
        self.dead = False

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
    class Listener:
        def after_add_node(self, graph: 'Graph', node: Node, insert_point_index: int):
            pass

        def before_remove_node(self, graph: 'Graph', node: Node):
            pass

    def __init__(self, name=''):
        self.name = name
        self.arguments: List[Node] = []
        self.nodes: List[Node] = []
        self.outputs: List[Node] = []
        self.insert_point_index: int = 0
        self.listeners: List[Graph.Listener] = []
        self.add_listener(Graph.UpdateInsertPointListener())

    class UpdateInsertPointListener(Listener):
        def after_add_node(self, graph: 'Graph', node: Node, insert_point_index: int):
            if insert_point_index <= graph.insert_point_index:
                graph.insert_point_index += 1

        def before_remove_node(self, graph: 'Graph', node: Node):
            remove_index = graph.nodes.index(node)
            if remove_index < graph.insert_point_index:
                graph.insert_point_index -= 1

    def add_listener(self, listener):
        self.listeners.append(listener)

    def remove_listener(self, listener):
        self.listeners.remove(listener)

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
        for listener in self.listeners:
            listener.after_add_node(self, node, self.insert_point_index)

        return node

    def remove_node(self, node: Node):
        node.dead = True
        for listener in self.listeners:
            listener.before_remove_node(self, node)

    def do_remove_dead_node(self):
        self.nodes = [node for node in self.nodes if not node.dead]

    def replace_all_uses_with(self, old_node: Node, new_node: Node):
        # todo: add used_list in Node
        for node in self.nodes:
            for idx, operand in enumerate(node.operands):
                if operand == old_node:
                    node.operands[idx] = new_node

        for idx, output in enumerate(self.arguments):
            if output == old_node:
                self.arguments[idx] = new_node

        for idx, output in enumerate(self.outputs):
            if output == old_node:
                self.outputs[idx] = new_node

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


@contextmanager
def listener_context(graph: Graph, listener: Graph.Listener):
    graph.add_listener(listener)
    try:
        yield listener
    finally:
        graph.remove_listener(listener)
