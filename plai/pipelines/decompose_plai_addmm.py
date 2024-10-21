from plai import dialect
from plai.core import module, pipeline, rewrite_pattern
from plai.dialect import plai_dialect


class DecomposeAddMm(rewrite_pattern.TypedRewritePattern):
    def __init__(self):
        super().__init__(dialect.plai_dialect.AddMm)

    def match_and_replace(self, graph: module.Graph, node: module.Node) -> bool:
        assert isinstance(node, dialect.plai_dialect.AddMm)
        graph.set_insert_point_after(node)
        # Decompose AddMm to Add, Mul and Mm
        # out = beta * bias + alpha * (mat1 * mat2)

        if node.get_beta() != 1:
            beta = graph.add_node(plai_dialect.Constant(node.get_beta()))
            beta_bias = graph.add_node(plai_dialect.Mul(beta, node.get_bias()))
        else:
            beta_bias = node.get_bias()

        mat1_mat2 = graph.add_node(plai_dialect.MatMul(node.get_mat1(), node.get_mat2()))

        if node.get_alpha() != 1:
            alpha = graph.add_node(plai_dialect.Constant(node.get_alpha()))
            alpha_mat1_mat2 = graph.add_node(plai_dialect.Mul(alpha, mat1_mat2))
        else:
            alpha_mat1_mat2 = mat1_mat2

        new_node = graph.add_node(plai_dialect.Add(beta_bias, alpha_mat1_mat2))
        graph.replace_all_uses_with(node, new_node)
        graph.remove_node(node)
        return True


class DecomposePlaiAddMmPass(pipeline.Pass):
    def __init__(self):
        super().__init__()

    def __call__(self, graph: module.Graph) -> bool:
        pattern_list = rewrite_pattern.RewritePatternList([DecomposeAddMm()])
        changed = rewrite_pattern.rewrite_pattern_recursive(graph, pattern_list)
        return changed
