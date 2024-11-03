from plai.core import runtime


class NumpyRuntime(runtime.Runtime):
    def run(self, graph, input_tensors):
        for node in graph.nodes:
            raise NotImplementedError(f"Node {node} is not supported by NumpyRuntime")
