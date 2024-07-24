from typing import Tuple, Callable, List, Dict

import torch
import torch.fx as fx

from plai.plnn.module import Graph, Node


class CustomCompiler:
    def __init__(self):
        self.graph = Graph('plnn_graph')
        self.node_mapping: Dict[torch.fx.Node, Node] = {}

    def mapping_node(self, value):
        if isinstance(value, Tuple):
            return tuple(self.mapping_node(v) for v in value)
        elif isinstance(value, List):
            return [self.mapping_node(v) for v in value]
        elif isinstance(value, dict):
            return {k: self.mapping_node(v) for k, v in value.items()}
        elif isinstance(value, torch.fx.Node):
            return self.node_mapping[value]
        else:
            raise ValueError(f"Unsupported type: {type(value)}")

    def __call__(self, gm: fx.GraphModule, example_inputs: Tuple[torch.Tensor, ...]) -> Callable:
        # 遍历计算图中的所有节点并收集信息
        for node in gm.graph.nodes:
            mapped_args = [self.mapping_node(arg) for arg in node.args]
            mapped_kwargs = {self.node_mapping[key]: self.mapping_node(value) for key, value in node.kwargs.items()}
            new_node = self.graph.add_node(node.name, node.op, node.target, mapped_args, mapped_kwargs)
            self.node_mapping[node] = new_node

        # 返回未修改的前向传播函数
        return gm.forward
