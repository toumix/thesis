from __future__ import annotations

from discopy import braided
from discopy.cat import Category
from discopy.braided import Ty, hexagon
from discopy.hypergraph import coherence


class Copyable:
    @classmethod
    def discard(cls, x: Ty, is_dagger=False) -> Copyable:
        return cls.copy(x, 0, is_dagger)

    @classmethod
    def merge(cls, x: Ty, n=2, is_dagger=False) -> Copyable:
        return cls.copy(x, n, not is_dagger)

    @classmethod
    def unit(cls, x: Ty): return cls.discard(x, is_dagger=True)


class Diagram(Copyable, braided.Diagram):
    @classmethod
    def copy(cls, x: Ty, n=2, is_dagger=False) -> Diagram:
        def factory(a, b, y, _):
            assert b == 1 if is_dagger else a == 1
            return Copy(y, a)[::-1] if is_dagger else Copy(y, b)
        a, b = (n, 1) if is_dagger else (1, n)
        return coherence(factory).__func__(cls, a, b, x)


class Box(braided.Box, Diagram):
    cast = Diagram.cast

class Swap(braided.Swap, Box): pass

Diagram.swap = Diagram.braid = hexagon(Swap)

class Copy(Box):
    def __init__(self, x: Ty, n: int = 2, is_dagger=False):
        assert len(x) == 1
        name = "Copy({}, {}){}".format(x, n, "[::-1]" if is_dagger else "")
        dom, cod = (x ** n, x) if is_dagger else (x, x ** n)
        super().__init__(name, dom, cod, is_dagger=is_dagger)

    def dagger(self):
        x = self.cod if self.is_dagger else self.dom
        n = len(self.dom) if self.is_dagger else len(self.cod)
        return Copy(x, n, not self.is_dagger)


class Functor(braided.Functor):
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Copy):
            x = self(other.cod if other.is_dagger else other.dom)
            n = len(other.dom) if other.is_dagger else len(other.cod)
            method = getattr(self.cod.ar, "merge" if other.is_dagger else "copy")
            return method(x, n)
        return super().__call__(other)
