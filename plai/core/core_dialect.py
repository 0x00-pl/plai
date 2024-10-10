from plai.core import module
from plai.core.location import Location


class CoreNode(module.Node):
    @classmethod
    def get_namespace(cls):
        return ''


class Placeholder(CoreNode):
    def __init__(self, loc: Location = None):
        super().__init__([], {}, loc)

