from typing import Tuple, Callable, List

import torch
import torch.fx as fx


class CustomCompiler:
    def __init__(self):
        self.nodes_info: List[dict] = []  # 用于存储节点信息的列表

    def __call__(self, gm: fx.GraphModule, example_inputs: Tuple[torch.Tensor, ...]) -> Callable:
        # 遍历计算图中的所有节点并收集信息
        for node in gm.graph.nodes:
            node_info = {"op": node.op, "name": node.name, "target": node.target, "args": node.args,
                         "kwargs": node.kwargs}
            self.nodes_info.append(node_info)  # 将节点信息添加到列表中

        # 返回未修改的前向传播函数
        return gm.forward

    def get_nodes_info(self) -> List[dict]:
        """返回收集到的节点信息"""
        return self.nodes_info

    def print_nodes_info(self) -> None:
        """格式化并打印收集到的节点信息"""
        print("GraphModule Nodes Information:")
        for idx, info in enumerate(self.nodes_info, start=1):
            print(f"{idx}: {info['name']} = {info['op']}[{info['target']}]{info['args']}{info['kwargs']}")
