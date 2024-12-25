import typing


class TypeNotation:
    def __init__(self, name: str = '?'):
        self.name = name

    def __str__(self):
        return self.name


class UnknownType(TypeNotation):
    def __init__(self):
        super().__init__('?')

class NoneType(TypeNotation):
    def __init__(self):
        super().__init__('None')


class ScalarType(TypeNotation):
    def __init__(self, name: str):
        super().__init__(name)


class TensorType(TypeNotation):
    def __init__(self, shape: typing.Sequence[int], element_type: TypeNotation):
        super().__init__()
        self.shape = list(shape)
        self.element_type = element_type

    def __str__(self):
        return f'tensor({self.shape}, {self.element_type})'


class TupleType(TypeNotation):
    def __init__(self, types: typing.Collection[TypeNotation]):
        super().__init__()
        self.types = types

    def __str__(self):
        return f'tuple({", ".join(str(t) for t in self.types)})'


def broadcast_shape(shape1: typing.List[int], shape2: typing.List[int]) -> typing.List[int]:
    """
    Broadcast two shapes together.
    """
    result = []
    len1, len2 = len(shape1), len(shape2)
    for i in range(max(len1, len2)):
        dim1 = shape1[len1 - 1 - i] if i < len1 else 1
        dim2 = shape2[len2 - 1 - i] if i < len2 else 1
        if dim1 == dim2:
            result.append(dim1)
        elif dim1 == 1:
            result.append(dim2)
        elif dim2 == 1:
            result.append(dim1)
        else:
            raise ValueError(f'Incompatible shapes: {shape1}, {shape2}')
    return list(reversed(result))