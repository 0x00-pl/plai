from typing import Tuple, Callable, List, Dict

import torch
import torch.fx as fx

from plai.core import core_dialect
from plai.core.location import NamedLocation
from plai.core.module import Graph, Node
from plai.dialect import aten_dialect, torch_dialect
from plai.pipelines.convertion_dialect_torch_to_plai import TorchToPlaiPass
from plai.pl_torch_compiler import torch_to_plai_convertion


class CustomCompiler:
    def __init__(self):
        self.graph = Graph('main_graph')
        self.node_mapping_dict: Dict[torch.fx.Node, Node] = {}

    @staticmethod
    def node_mapping(node, node_mapping_dict: Dict[torch.fx.Node, Node]):
        if isinstance(node, Tuple):
            return tuple(CustomCompiler.node_mapping(v, node_mapping_dict) for v in node)
        elif isinstance(node, List):
            return [CustomCompiler.node_mapping(v, node_mapping_dict) for v in node]
        elif isinstance(node, dict):
            return {k: CustomCompiler.node_mapping(v, node_mapping_dict) for k, v in node.items()}
        elif isinstance(node, torch.fx.Node):
            return node_mapping_dict[node]
        else:
            return node

    @staticmethod
    def import_graph(gm: fx.GraphModule, node_mapping_dict: Dict[torch.fx.Node, Node]) -> Graph:
        aten_dialect.register_dialect()
        converter = torch_to_plai_convertion.Converter()
        converter.register_convertion_function_dict(torch_dialect.TorchNode.convertion_function_dict)

        def local_node_mapping(n):
            return CustomCompiler.node_mapping(n, node_mapping_dict)

        graph = Graph('main_graph')
        # 遍历计算图中的所有节点并收集信息
        for node in gm.graph.nodes:
            assert isinstance(node, fx.Node)
            if node.op == 'placeholder':
                new_node = core_dialect.Placeholder(NamedLocation(node.target))
                graph.add_argument(new_node)
            elif node.op == 'output':
                for i in node.args[0]:
                    assert i in node_mapping_dict or i is None
                    graph.add_output(local_node_mapping(i))
                new_node = None
            elif node.op == 'get_attr':
                raise NotImplementedError("get_attr is not supported")
            elif node.op in ('call_method', 'call_module', 'call_function'):
                new_node = converter.convert_node(node, local_node_mapping)
                graph.add_node(new_node)
            else:
                raise ValueError(f"Unsupported op: {node.op}")

            node_mapping_dict[node] = new_node

        return graph

    def __call__(self, gm: fx.GraphModule, example_inputs: Tuple[torch.Tensor, ...]) -> Callable:
        self.graph = self.import_graph(gm, self.node_mapping_dict)

        pipeline = TorchToPlaiPass()
        changed = pipeline(self.graph)
        _ = changed

        # 返回未修改的前向传播函数
        return gm.forward
