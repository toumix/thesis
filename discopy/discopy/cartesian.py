from __future__ import annotations

from discopy import braided
from discopy.cat import Category
from discopy.braided import Ty, hexagon
from discopy.hypergraph import coherence


class Diagram(braided.Diagram):
    @classmethod
    def copy(cls, x: Ty, n=2) -> Diagram:
        def factory(a, b, x, _):
            assert a == 1
            return Copy(x, b)
        return coherence(factory).__func__(cls, 1, n, x)

class Box(braided.Box, Diagram):
    cast = Diagram.cast

class Swap(braided.Swap, Box): pass

Diagram.swap = Diagram.braid = hexagon(Swap)

class Copy(Box):
    def __init__(self, x: Ty, n: int = 2):
        assert len(x) == 1
        super().__init__(name="Copy({}, {})".format(x, n), dom=x, cod=x ** n)

class Functor(braided.Functor):
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Copy):
            return self.cod.ar.copy(self(other.dom), len(other.cod))
        return super().__call__(other)
