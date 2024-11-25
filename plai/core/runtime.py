from abc import abstractmethod

from plai.core.graph import Graph


class Runtime:

    def __call__(self, graph: Graph, input_tensors) -> Graph:
        return self.run(graph, input_tensors)

    @abstractmethod
    def run(self, graph: Graph, input_tensors) -> Graph:
        pass
