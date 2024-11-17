from contextlib import contextmanager
from typing import List, Dict, Optional

from plai.core.core_dialect import Placeholder, Output
from plai.core.node import Node


class Graph:
    def __init__(self, name=''):
        self.name = name
        self.arguments: List[Placeholder] = []
        self.outputs = Output()
        self.nodes: List[Node] = []
        self.insert_point_index: int | None = 0
        self.listeners: List[Graph.Listener] = []
        self.add_listener(Graph.UpdateInsertPointListener())

    class Listener:
        def after_add_node(self, graph: 'Graph', node: Node):
            pass

        def before_remove_node(self, graph: 'Graph', node: Node):
            pass

        def before_remove_dead_node(self, graph: 'Graph'):
            pass

        def node_operand_changed(self, graph: 'Graph', node: Node, idx: int, old_operand: Node, new_operand: Node):
            pass

    class UpdateInsertPointListener(Listener):
        def after_add_node(self, graph: 'Graph', node: Node):
            graph.insert_point_index += 1

        def before_remove_dead_node(self, graph: 'Graph'):
            graph.insert_point_index = None

    def add_listener(self, listener):
        self.listeners.append(listener)

    def remove_listener(self, listener):
        self.listeners.remove(listener)

    def add_argument(self, node: Placeholder):
        self.arguments.append(node)

    def add_output(self, node: Node):
        self.outputs.add_argument(node)

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
        assert self.insert_point_index is not None, 'Insert point is not set.'
        self.nodes.insert(self.insert_point_index, node)

        for listener in self.listeners:
            listener.after_add_node(self, node)

        return node

    def remove_node(self, node: Node):
        node.remove()
        for listener in self.listeners:
            listener.before_remove_node(self, node)

    def do_remove_dead_node(self):
        for listener in self.listeners:
            listener.before_remove_dead_node(self)
        self.nodes = [node for node in self.nodes if not node.dead]

    def replace_all_uses_with(self, old_node: Node, new_node: Node):
        # todo: add used_list in Node
        for node in self.nodes:
            for idx, operand in enumerate(node.operands):
                if operand == old_node:
                    node.operands[idx] = new_node

                    for listener in self.listeners:
                        listener.node_operand_changed(self, node, idx, old_node, new_node)

        for idx, output in enumerate(self.arguments):
            if output == old_node:
                assert isinstance(new_node, Placeholder)
                self.arguments[idx] = new_node

        for idx, output in enumerate(self.outputs.operands):
            if output == old_node:
                self.outputs.set_operand(idx, new_node)

    def __str__(self):
        node_name_dict: Dict[Optional[Node], str] = {None: 'None'}
        node_name_dict = node_name_dict | {node: f'arg{idx}' for idx, node in enumerate(self.arguments)}
        node_name_dict = node_name_dict | {node: f'v{idx}' for idx, node in enumerate(self.nodes)}

        result = f'Graph {self.name}({", ".join(node_name_dict[i] for i in self.arguments)}): \n'
        for idx, node in enumerate(self.nodes):
            name = node_name_dict[node]
            result += f'  {idx}: {name} = {node.to_string(node_name_dict)}\n'
        result += f'  output ({", ".join(node_name_dict[i] for i in self.outputs.operands)})\n'
        return result


@contextmanager
def listener_context(graph: Graph, listener: Graph.Listener):
    graph.add_listener(listener)
    try:
        yield listener
    finally:
        graph.remove_listener(listener)
