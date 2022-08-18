from __future__ import annotations

from discopy import cat, monoidal
from discopy.cat import Category
from discopy.sugar import dataclass


@dataclass
class Ob(cat.Ob):
    z: int = 0

    l = property(lambda self: Ob(self.name, self.z - 1))
    r = property(lambda self: Ob(self.name, self.z + 1))

    @classmethod
    def cast(cls, old: cat.Ob) -> Ob:
        return old if isinstance(old, cls) else cls(str(old), z=0)


class Ty(monoidal.Ty, Ob):
    def __init__(self, inside=()):
        monoidal.Ty.__init__(self, inside=tuple(map(Ob.cast, inside)))

    l = property(lambda self: type(self)([x.l for x in self.inside[::-1]]))
    r = property(lambda self: type(self)([x.r for x in self.inside[::-1]]))


class Diagram(monoidal.Diagram):
    def transpose(self, left=True) -> Diagram:
        if left: ... # Symmetric to the right case.
        return self.caps(self.dom.r, self.dom) @ self.id(self.cod.r)\
            >> self.id(self.dom.r) @ self @ self.id(self.cod.r)\
            >> self.id(self.dom.r) @ self.cups(self.cod, self.cod.r)

class Box(monoidal.Box, Diagram):
    cast = Diagram.cast

class Cup(Box):
    def __init__(self, x: Ty, y: Ty):
        assert len(x) == 1 and x == y.l
        super().__init__("Cup({}, {})".format(repr(x), repr(y)), x @ y, x[:0])

class Cap(Box):
    def __init__(self, x: Ty, y: Ty):
        assert len(x) == 1 and x.l == y
        super().__init__("Cap({}, {})".format(repr(x), repr(y)), x[:0], x @ y)


def nesting(factory):
    def method(cls, x: Ty, y: Ty) -> Diagram:
        if len(x) == 0: return cls.id(x[:0])
        if len(x) == 1: return factory(x, y)
        head = factory(x[0], y[-1])
        if head.dom:  # We are nesting cups.
            return x[0] @ method(cls, x[1:], y[:-1]) @ y[-1] >> head
        return head >> x[0] @ method(cls, x[1:], y[:-1]) @ y[-1]
    return classmethod(method)


Diagram.cups, Diagram.caps = nesting(Cup), nesting(Cap)


class Functor(monoidal.Functor):
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Ty) or isinstance(other, Ob) and other.z == 0:
            return super().__call__(other)
        if isinstance(other, Ob):
            if not hasattr(self.cod.ob, 'l' if other.z < 0 else 'r'):
                return self(Ob(other.name, z=0))[::-1]
            return self(other.r).l if other.z < 0 else self(other.l).r
        if isinstance(other, Cup):
            return self.cod.ar.cups(self(other.dom[:1]), self(other.dom[1:]))
        if isinstance(other, Cap):
            return self.cod.ar.caps(self(other.cod[:1]), self(other.cod[1:]))
        return super().__call__(other)
