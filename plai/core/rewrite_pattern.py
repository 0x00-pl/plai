from plai.core import module


class RewritePattern:
    def __init__(self, name: str):
        self.name = name

    def match_and_replace(self, node: module.Node) -> bool:
        """
        :param node: The node to match and replace.
        :return: True when changed.
        """
        return False
