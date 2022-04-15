from __future__ import annotations
import math
from numbers import Number
from numpy import isclose

from discopy.sugar import dataclass, inductive, Composable


class Matrix(Composable):
    dtype = int

    dom: int
    cod: int
    inside: list[list[dtype]]

    def __class_getitem__(cls, dtype: type):
        class C(cls):
            __name__ = __qualname__ = "Matrix[{}]".format(dtype.__name__)
        C.dtype = dtype
        return C

    def __init__(self, inside: list[list[dtype]], dom: int, cod: int):
        assert len(inside) == dom and all(len(row) == cod for row in inside)
        self.inside, self.dom, self.cod =\
            [list(map(self.dtype, row)) for row in inside], dom, cod

    def __eq__(self, other):
        if not isinstance(other, Matrix):
            return self.dom == self.cod == 1 and self.inside[0][0] == other
        return (self.dtype, self.inside, self.dom, self.cod)\
            == (other.dtype, other.inside, other.dom, other.cod)

    def is_close(self, other):
        if not isinstance(other, Matrix):
            assert self.dom == self.cod == 1
            return isclose(self.inside[0][0], other)
        assert self.dom == other.dom and self.cod == other.cod
        return all(all(isclose(x, y) for x, y in zip(self_row, other_row))
                   for self_row, other_row in zip(self.inside, other.inside))

    @classmethod
    def id(cls, x: int) -> Matrix:
        return cls([[i == j for i in range(x)] for j in range(x)], x, x)

    @inductive
    def then(self, other: Matrix) -> Matrix:
        assert self.dtype == other.dtype and self.cod == other.dom
        inside = [[sum(
            self.inside[i][j] * other.inside[j][k] for j in range(other.dom))
            for k in range(other.cod)] for i in range(self.dom)]
        return type(self)(inside, self.dom, other.cod)

    def __getitem__(self, key: int | tuple[int, ...]) -> Matrix:
        key = key if isinstance(key, tuple) else (key, )
        inside = [[self.inside[i][j] for i in key for j in range(self.cod)]]\
            if len(key) == 1 else [[self.inside[i][j] for i, j in [key]]]
        dom, cod = 1, self.cod if len(key) == 1 else 1
        return type(self)(inside, dom, cod)

    def transpose(self) -> Matrix:
        inside = [[self[j, i] for j in range(self.dom)] for i in range(self.cod)]
        return type(self)(inside, self.cod, self.dom)

    def map(self, func: Callable[[Number], Number]) -> Matrix:
        inside = [list(map(func, row)) for row in self.inside]
        return type(self)(inside, self.dom, self.cod)

    def conjugate(self) -> Matrix:
        return self.map(lambda x: x.conjugate())

    def dagger(self) -> Matrix:
        return self.conjugate().transpose()

    def __radd__(self, other: Number) -> Matrix:
        assert self.dom == self.cod == 1
        return self.inside[0][0] + other

    def __add__(self, other: Matrix) -> Matrix:
        inside = [[x + y for x, y in zip(u, v)]
                  for u, v in zip(self.inside, other.inside)]
        return type(self)(inside, self.dom, self.cod)

    @classmethod
    def zero(cls, dom: int, cod: int) -> Matrix:
        return cls([[0 for _ in range(cod)] for _ in range(dom)], dom, cod)


for converter in (bool, int, float, complex):
    # Downcasting a 1 by 1 Matrix to a scalar.
    def method(self, converter=converter):
        assert self.dom == self.cod == 1
        return converter(self.inside[0][0])
    setattr(Matrix, "__{}__".format(converter.__name__), method)
