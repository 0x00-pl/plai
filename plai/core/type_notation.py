import typing


class TypeNotation:
    def __init__(self, name: str = '?'):
        self.name = name

    def __str__(self):
        return self.name


class UnknownType(TypeNotation):
    def __init__(self):
        super().__init__('?')


class ScalarType(TypeNotation):
    def __init__(self, name: str):
        super().__init__(name)


class TensorType(TypeNotation):
    def __init__(self, shape: typing.Sequence[int], element_type: TypeNotation):
        super().__init__()
        self.shape = shape
        self.element_type = element_type

    def __str__(self):
        return f'tensor({self.shape}, {self.element_type})'


class TupleType(TypeNotation):
    def __init__(self, types: typing.Sequence[TypeNotation]):
        super().__init__()
        self.types = types

    def __str__(self):
        return f'tuple({", ".join(str(t) for t in self.types)})'
