from abc import abstractmethod

from plai.core import module
from plai.core.location import Location


class BuiltinNode(module.Node):
    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        raise ValueError('this is a dialect, should not using Build.')

    @classmethod
    def get_namespace(cls):
        return ''

    @staticmethod
    @abstractmethod
    def from_builtin(args: list, attrs: dict, loc: Location = None):
        pass

    convertion_function_dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        func_name = cls.get_op_name()
        assert func_name not in BuiltinNode.convertion_function_dict, f'Duplicate key: {func_name}'
        BuiltinNode.convertion_function_dict[func_name] = cls.from_builtin


class GetItem(BuiltinNode):
    def __init__(self, arg: module.Node, key, loc: Location = None):
        super().__init__([arg], {'key': key}, loc)

    @staticmethod
    def build(op_name: str, args: list, attrs: dict, loc: Location = None):
        assert op_name == 'getitem'
        return GetItem(args[0], key=args[0], loc=loc)

    @classmethod
    def get_cls_name(cls) -> str:
        return '_operator.getitem'

    @staticmethod
    def from_builtin(args: list, attrs: dict, loc: Location = None):
        return GetItem(args[0], args[1], loc)
