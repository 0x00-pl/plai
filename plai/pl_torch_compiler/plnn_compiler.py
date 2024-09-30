from typing import Tuple, Callable, List, Dict

import torch
import torch.fx as fx

from plai.core import core_dialect
from plai.core.location import NamedLocation
from plai.core.module import Graph, Node
from plai.dialect import aten_dialect, torch_dialect
from plai.pl_torch_compiler import torch_to_plai_convertion


class CustomCompiler:
    def __init__(self):
        self.graph = Graph('main_graph')
        self.node_mapping_dict: Dict[torch.fx.Node, Node] = {}

    def node_mapping(self, node):
        if isinstance(node, Tuple):
            return tuple(self.node_mapping(v) for v in node)
        elif isinstance(node, List):
            return [self.node_mapping(v) for v in node]
        elif isinstance(node, dict):
            return {k: self.node_mapping(v) for k, v in node.items()}
        elif isinstance(node, torch.fx.Node):
            return self.node_mapping_dict[node]
        else:
            return node

    def __call__(self, gm: fx.GraphModule, example_inputs: Tuple[torch.Tensor, ...]) -> Callable:
        aten_dialect.register_dialect()
        converter = torch_to_plai_convertion.Converter()
        converter.register_convertion_function_dict(torch_dialect.TorchNode.convertion_function_dict)

        # 遍历计算图中的所有节点并收集信息
        for node in gm.graph.nodes:
            assert isinstance(node, fx.Node)
            if node.op == 'placeholder':
                new_node = core_dialect.Placeholder(NamedLocation(node.target))
                self.graph.add_argument(new_node)
            elif node.op == 'output':
                for i in node.args[0]:
                    assert i in self.node_mapping_dict or i is None
                    self.graph.add_output(self.node_mapping(i))
                new_node = None
            elif node.op == 'get_attr':
                raise NotImplementedError("get_attr is not supported")
            elif node.op in ('call_method', 'call_module', 'call_function'):
                new_node = converter.convert_node(node, self.node_mapping)
                self.graph.add_node(new_node)
            else:
                raise ValueError(f"Unsupported op: {node.op}")

            self.node_mapping_dict[node] = new_node

        # 返回未修改的前向传播函数
        return gm.forward
