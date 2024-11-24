import typing

import numpy

from plai.core import runtime
from plai.core.node import Node
from plai.dialect.plai_dialect import Transpose, MatMul, Add, Relu


class PlaiNumpyRuntime(runtime.Runtime):
    def run(self, graph, input_tensors):
        node_value_dict: typing.Dict[Node, typing.Any] = dict(zip(graph.arguments, input_tensors))
        for node in graph.nodes:
            operand_values = [node_value_dict[operand] for operand in node.operands]
            if isinstance(node, Transpose):
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

        print(graph)
        return [node_value_dict[output] for output in graph.outputs.operands]
