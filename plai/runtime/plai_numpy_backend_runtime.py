import typing

import numpy
import torch

from plai.core import runtime
from plai.core.core_dialect import Output
from plai.core.node import Node
from plai.dialect.plai_dialect import Transpose, MatMul, Add, Relu


class ValueOnDevice:
    def __init__(self, np_array: numpy.ndarray):
        self.value = np_array

    def get_value(self):
        return self.value


class Backend:

    def __init__(self):
        self.heap = set()

    def load(self, np_array: numpy.ndarray) -> ValueOnDevice:
        result = ValueOnDevice(np_array)
        self.heap.add(result)
        return result

    def store(self, device_array: ValueOnDevice):
        assert device_array in self.heap, "Device array is not on device"
        return device_array.get_value()

    def clear(self):
        self.heap.clear()

    def run_op(self, node_ty: str, operands: typing.List[ValueOnDevice], attrs: typing.Dict):
        assert all(operand in self.heap for operand in operands), "Not all operands are on device"
        operands_in_numpy = [operand.get_value() for operand in operands]
        if node_ty == 'Transpose':
            np_result = numpy.transpose(operands_in_numpy[0], axes=attrs['permutation'])
        elif node_ty == 'MatMul':
            np_result = numpy.matmul(operands_in_numpy[0], operands_in_numpy[1])
        elif node_ty == 'Add':
            np_result = operands_in_numpy[0] + operands_in_numpy[1]
        elif node_ty == 'Relu':
            np_result = numpy.maximum(operands_in_numpy[0], 0)
        else:
            raise NotImplementedError(f"Node {node_ty} is not supported by Backend")

        result = ValueOnDevice(np_result)
        self.heap.add(result)
        return result


class PlaiNumpyBackendRuntime(runtime.Runtime):
    def __init__(self, backend: Backend):
        self.backend = backend

    def run(self, graph, input_tensors):
        self.backend.clear()
        node_value_dict: typing.Dict[Node, numpy.ndarray] = {
            k: v.cpu().numpy() for k, v in zip(graph.arguments, input_tensors)
        }
        node_device_value_dict: typing.Dict[Node, ValueOnDevice] = {
            k: self.backend.load(v) for k, v in node_value_dict.items()
        }

        def calc_value(node: Node):
            operand_values = [node_device_value_dict[operand] for operand in node.operands]
            if isinstance(node, Output):
                result = None
            elif isinstance(node, Transpose):
                result = self.backend.run_op('Transpose', operand_values, node.attrs)
            elif isinstance(node, MatMul):
                result = self.backend.run_op('MatMul', operand_values, node.attrs)
            elif isinstance(node, Add):
                result = self.backend.run_op('Add', operand_values, node.attrs)
            elif isinstance(node, Relu):
                result = self.backend.run_op('Relu', operand_values, node.attrs)
            else:
                raise NotImplementedError(f"Node {node} is not supported by NumpyRuntime")

            node_device_value_dict[node] = result

        graph.walk(calc_value)

        return [torch.from_numpy(node_device_value_dict[output].get_value()) for output in graph.outputs.operands]
