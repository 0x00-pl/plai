import typing

import torch

from plai import dialect
from plai.core import module, pipeline
from plai.core.pipeline import Pipeline


class TorchToPlaiPass(pipeline.Pass):
    def __init__(self):
        super().__init__('torch_to_plai')

    def __call__(self, graph) -> bool:
        """
        :param graph:
        :return: True when changed.
        """
        return False

