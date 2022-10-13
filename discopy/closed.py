from __future__ import annotations

from discopy import monoidal
from discopy.cat import Category


class Ty(monoidal.Ty):
    @classmethod
    def cast(cls, old: monoidal.Ty) -> Ty:
        return old[0] if len(old) == 1 and isinstance(old[0], Exp) else cls(old)

    def __pow__(self, other):
        return Exp(self, other) if isinstance(other, Ty)\
            else super().__pow__(other)

class Exp(Ty):
    cast = Ty.cast

    def __init__(self, base, exponent):
        self.base, self.exponent = base, exponent
        super().__init__(inside=(self, ))

    def __eq__(self, other):
        return isinstance(other, type(self))\
            and (self.base, self.exponent) == (other.base, other.exponent)

    __str__ = lambda self: "({} ** {})".format(self.base, self.exponent)
    __repr__ = lambda self: "Exp({}, {})".format(
        repr(self.base), repr(self.exponent))

class Over(Exp):
    __str__ = lambda self: "({} << {})".format(self.base, self.exponent)
    __repr__ = lambda self: super().__repr__().replace("Exp", "Over")

class Under(Exp):
    __str__ = lambda self: "({} >> {})".format(self.exponent, self.base)
    __repr__ = lambda self: super().__repr__().replace("Exp", "Under")

Ty.__lshift__ = lambda self, other: Over(self, other)
Ty.__rshift__ = lambda self, other: Under(other, self)

class Diagram(monoidal.Diagram):
    curry = lambda self, n=1, left=True: Curry(self, n, left)

    @staticmethod
    def ev(base: Ty, exponent: Ty, left=True) -> Ev:
        return Ev(base << exponent if left else exponent >> base)

    def uncurry(self: Diagram, left=True) -> Diagram:
        base, exponent = self.cod.base, self.cod.exponent
        return self @ exponent >> Ev(base << exponent) if left\
            else exponent @ self >> Ev(exponent >> base)

class Box(monoidal.Box, Diagram):
    cast = Diagram.cast

class Ev(Box):
    def __init__(self, x: Exp):
        self.base, self.exponent = x.base, x.exponent
        self.left = isinstance(x, Over)
        dom, cod = (x @ self.exponent, self.base) if self.left\
            else (self.exponent @ x, self.base)
        super().__init__("Ev" + str(x), dom, cod)

class Curry(Box):
    def __init__(self, diagram: Diagram, n=1, left=True):
        self.diagram, self.n, self.left = diagram, n, left
        name = "Curry({}, {}, {})".format(diagram, n, left)
        if left:
            dom = diagram.dom[:len(diagram.dom) - n]
            cod = diagram.cod << diagram.dom[len(diagram.dom) - n:]
        else: dom, cod = diagram.dom[n:], diagram.dom[:n] >> diagram.cod
        super().__init__(name, dom, cod)

Diagram.over, Diagram.under, Diagram.exp = map(staticmethod, (Over, Under, Exp))

class Functor(monoidal.Functor):
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        for cls, attr in [(Over, "over"), (Under, "under"), (Exp, "exp")]:
            if isinstance(other, cls):
                method = getattr(self.cod.ar, attr)
                return method(self(other.base), self(other.exponent))
        if isinstance(other, Curry):
            return self.cod.ar.curry(
                self(other.diagram), len(self(other.cod.exponent)), other.left)
        if isinstance(other, Ev):
            return self.cod.ar.ev(
                self(other.base), self(other.exponent), other.left)
        return super().__call__(other)
