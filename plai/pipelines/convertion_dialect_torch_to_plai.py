from plai import dialect
from plai.core import module, pipeline, rewrite_pattern


class ConvertTranspose(rewrite_pattern.TypedRewritePattern):
    def __init__(self):
        super().__init__(dialect.aten_dialect.Transpose)

    def match_and_replace(self, graph: module.Graph, node: module.Node) -> bool:
        assert isinstance(node, dialect.aten_dialect.Transpose)
        graph.set_insert_point_after(node)
        new_node = dialect.plai_dialect.Transpose(node.operands[0])
        graph.add_node(new_node)
        graph.remove_node(node)
        return True


class TorchToPlaiPass(pipeline.Pass):
    def __init__(self):
        super().__init__('torch_to_plai')

    def __call__(self, graph) -> bool:
        """
        :param graph:
        :return: True when changed.
        """
        pattern_list = rewrite_pattern.RewritePatternList([ConvertTranspose()])
        changed = rewrite_pattern.rewrite_pattern_recursive(graph, pattern_list)
        return changed
