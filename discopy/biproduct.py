from __future__ import annotations
from contextlib import contextmanager

from discopy import monoidal
from discopy.sugar import dataclass, product, inductive
from discopy.cat import Category
from discopy.matrix import Matrix
from discopy.monoidal import Ty, Diagram


@dataclass
class FakeInt:
    inside: tuple[Ty, ...] = (Ty(), )

    __index__ = lambda self: len(self.inside)
    __iter__ = property(lambda self: self.inside.__iter__)
    __add__ = lambda self, other: FakeInt(self.inside + other.inside)
    __mul__ = lambda self, other: FakeInt(
        tuple(x0 @ x1 for x0 in self for x1 in other))
    __rmul__ = lambda self, n: FakeInt(n * self.inside)
    __pow__ = lambda self, n: product(n * (self, ), unit=FakeInt())


class Diagram(monoidal.Diagram):
    def __eq__(self, other):
        if isinstance(other, Biproduct):
            return other.inside == [[self]]
        return monoidal.Diagram.__eq__(self, other)

    def direct_sum(self, *others):
        return Biproduct.cast(self).direct_sum(*others)

    __or__ = direct_sum


class Box(monoidal.Box, Diagram):
    cast = Diagram.cast


class Sum(monoidal.Sum, Box):
    id = lambda x: Sum.cast(Diagram.id(x))


Diagram.sum = Sum


class Biproduct(Matrix):
    inside: list[list[Sum]]
    dom: FakeInt
    cod: FakeInt

    dtype = Sum

    def __init__(self, inside, dom, cod):
        self.dom, self.cod, self.inside = dom, cod, [[
            self.dtype.id(x) if val == 1
            else self.dtype.zero(x, y) if val == 0
            else self.dtype.cast(val)
            for y, val in zip(cod, row)] for x, row in zip(dom, inside)]

    @contextmanager
    def fake_multiplication(self, method):
        self.dtype.__mul__ = getattr(self.dtype, method)
        yield
        delattr(self.dtype, "__mul__")

    @classmethod
    def cast(cls, old: Diagram):
        if isinstance(old, cls): return old
        return cls([[old]], FakeInt((old.dom, )), FakeInt((old.cod, )))

    @inductive
    def then(self, other: Biproduct | Diagram) -> Biproduct:
        with self.fake_multiplication("then"):
            return Matrix.then(self, self.cast(other))

    @inductive
    def tensor(self, other: Biproduct | Diagram) -> Biproduct:
        with self.fake_multiplication("tensor"):
            return Matrix.Kronecker(self, self.cast(other))

    @inductive
    def direct_sum(self, other: Biproduct | Diagram) -> Biproduct:
        with self.fake_multiplication("then"):
            return Matrix.direct_sum(self, self.cast(other))

    dagger = lambda self: self.transpose().map(lambda f: f.dagger())
    __or__ = lambda self, other: self.direct_sum(self.cast(other))
    __eq__ = lambda self, other: Matrix.__eq__(self, self.cast(other))
