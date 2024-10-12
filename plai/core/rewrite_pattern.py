import inspect
import typing
from abc import ABC, abstractmethod

from plai.core import module


class TraceChangedListener(module.Graph.Listener):
    def __init__(self):
        self.changed_nodes = []

    def after_add_node(self, graph: module.Graph, node: module.Node, insert_point_index: int):
        self.changed_nodes.append(node)

    def before_remove_node(self, graph: module.Graph, node: module.Node):
        pass


class RewritePattern(ABC):
    @staticmethod
    @abstractmethod
    def match_and_replace(graph: module.Graph, node: module.Node) -> bool:
        """
        :param graph:
        :param node: The node to match and replace.
        :return: True when changed.
        """
        pass


class TypedRewritePattern(RewritePattern, ABC):
    def __init__(self, node_cls: type):
        self.node_cls = node_cls


class RewritePatternList(RewritePattern):
    def __init__(self, patterns: [RewritePattern] = None):
        if patterns is None:
            patterns = []

        self.patterns = []
        self.typed_pattern_map: typing.Dict[typing.Type, typing.List[TypedRewritePattern]] = {}

        for pattern in patterns:
            self.add(pattern)

    def add(self, pattern: RewritePattern):
        if isinstance(pattern, TypedRewritePattern):
            if pattern.node_cls not in self.typed_pattern_map:
                self.typed_pattern_map[pattern.node_cls] = []
            self.typed_pattern_map[pattern.node_cls].append(pattern)
        else:
            self.patterns.append(pattern)

    def match_and_replace(self, graph: module.Graph, node: module.Node) -> bool:
        for pattern in self.patterns:
            if pattern.match_and_replace(graph, node):
                return True

        typed_pattern_list = self.get_typed_pattern_list_from_cls(type(node))
        for pattern in typed_pattern_list:
            if pattern.match_and_replace(graph, node):
                return True
        return False

    def get_typed_pattern_list_from_cls(self, node_cls: type) -> typing.List[TypedRewritePattern]:
        cls_list = inspect.getmro(node_cls)
        for cls in cls_list:
            if not isinstance(cls, module.Node):
                continue

            if cls in self.typed_pattern_map:
                return self.typed_pattern_map[type(cls)]
        return []


def rewrite_pattern_recursive(graph: module.Graph, pattern: RewritePattern, max_replace_count_factor: int = 10) -> bool:
    changed = False
    processed_replace_count = 0
    todo_node_list = list(graph.nodes)
    max_replace_count = len(todo_node_list) * max_replace_count_factor

    while todo_node_list:
        assert processed_replace_count < max_replace_count, 'Infinite loop detected.'
        next_todo_node_list = []
        with module.listener_context(graph, TraceChangedListener()) as trace_changed:
            for node in todo_node_list:
                if node.dead:
                    continue
                if pattern.match_and_replace(graph, node):
                    next_todo_node_list.append(node)
                    changed = True
                    processed_replace_count += 1

        graph.do_remove_dead_node()
        next_todo_node_list.extend(trace_changed.changed_nodes)
        todo_node_list = next_todo_node_list

    return changed
