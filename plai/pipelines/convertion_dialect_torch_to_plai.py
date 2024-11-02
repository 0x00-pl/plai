from plai import dialect
from plai.core import module, pipeline, rewrite_pattern
from plai.dialect import plai_dialect


class ConvertTranspose(rewrite_pattern.TypedRewritePattern):
    def __init__(self):
        super().__init__(dialect.aten_dialect.Transpose)

    def match_and_replace(self, graph: module.Graph, node: module.Node) -> bool:
        assert isinstance(node, dialect.aten_dialect.Transpose)
        new_node = plai_dialect.Transpose(node.operands[0])
        graph.add_node(new_node)
        graph.replace_all_uses_with(node, new_node)
        graph.remove_node(node)
        return True


class ConvertRelu(rewrite_pattern.TypedRewritePattern):
    def __init__(self):
        super().__init__(dialect.aten_dialect.Relu)

    def match_and_replace(self, graph: module.Graph, node: module.Node) -> bool:
        assert isinstance(node, dialect.aten_dialect.Relu)
        new_node = plai_dialect.Relu(node.operands[0])
        graph.add_node(new_node)
        graph.replace_all_uses_with(node, new_node)
        graph.remove_node(node)
        return True


class ConvertAddmm(rewrite_pattern.TypedRewritePattern):
    def __init__(self):
        super().__init__(dialect.aten_dialect.Addmm)

    def match_and_replace(self, graph: module.Graph, node: module.Node) -> bool:
        assert isinstance(node, dialect.aten_dialect.Addmm)
        new_node = plai_dialect.AddMm(node.operands[0], node.operands[1], node.operands[2], node.attrs['beta'],
                                      node.attrs['alpha'])
        graph.add_node(new_node)
        graph.replace_all_uses_with(node, new_node)
        graph.remove_node(node)
        return True


class TorchToPlaiPass(pipeline.Pass):
    def __init__(self):
        super().__init__()

    def __call__(self, graph: module.Graph) -> bool:
        """
        :param graph:
        :return: True when changed.
        """
        pattern_list = rewrite_pattern.RewritePatternList([ConvertTranspose(), ConvertRelu(), ConvertAddmm()])
        changed = rewrite_pattern.rewrite_pattern_recursive(graph, pattern_list)
        return changed
