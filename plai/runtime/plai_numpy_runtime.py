import typing

import numpy
import torch

from plai.core import runtime
from plai.core.core_dialect import Output
from plai.core.node import Node
from plai.dialect.plai_dialect import Transpose, MatMul, Add, Relu


class PlaiNumpyRuntime(runtime.Runtime):
    def run(self, graph, input_tensors):
        node_value_dict: typing.Dict[Node, numpy.ndarray] = {k: v.cpu().numpy() for k, v in
                                                             zip(graph.arguments, input_tensors)}

        def calc_value(node: Node):
            operand_values = [node_value_dict[operand] for operand in node.operands]
            if isinstance(node, Output):
                result = None
            elif isinstance(node, Transpose):
                result = numpy.transpose(operand_values[0], axes=node.attrs['permutation'])
            elif isinstance(node, MatMul):
                result = numpy.matmul(operand_values[0], operand_values[1])
            elif isinstance(node, Add):
                result = operand_values[0] + operand_values[1]
            elif isinstance(node, Relu):
                result = numpy.maximum(operand_values[0], 0)
            else:
                raise NotImplementedError(f"Node {node} is not supported by NumpyRuntime")

            node_value_dict[node] = result

        graph.walk(calc_value)

        return [torch.from_numpy(node_value_dict[output]) for output in graph.outputs.operands]
