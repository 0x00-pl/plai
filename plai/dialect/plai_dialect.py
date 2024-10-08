from typing import List

from plai.core import module
from plai.core.location import Location


class PlaiNode(module.Node):
    @classmethod
    def get_namespace(cls):
        return 'plai'


class Transpose(PlaiNode):
    def __init__(self, arg: module.Node, permutation: List = None, loc: Location = None):
        permutation = permutation if permutation is not None else [1, 0]
        super().__init__([arg], {'permutation': permutation}, loc)
