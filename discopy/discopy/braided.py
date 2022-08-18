from typing import Callable

from discopy import monoidal
from discopy.cat import Category
from discopy.monoidal import Ty, Match


class Diagram(monoidal.Diagram):
    def simplify(self):
        for i, ((x, f, _), (y, g, _)) in enumerate(
                zip(self.inside, self.inside[1:])):
            if x == y and isinstance(f, Braid) and f == g[::-1]:
                inside = self.inside[:i] + self.inside[i + 2:]
                return self.cast(Diagram(inside, self.dom, self.cod)).simplify()
        return self

class Box(monoidal.Box, Diagram):
    cast = Diagram.cast

class Braid(Box):
    def __init__(self, x: Ty, y: Ty, is_dagger=False):
        assert len(x) == len(y) == 1
        name = "{}({}, {})[::-1]".format(type(self), y, x)\
            if is_dagger else "{}({}, {})".format(type(self), x, y)
        super().__init__(name, x @ y, y @ x, is_dagger=is_dagger)

    def dagger(self): return Braid(*self.cod, is_dagger=not self.is_dagger)

class Swap(Braid):
    def dagger(self): return Swap(*self.cod)

def hexagon(factory) -> Callable:
    def method(cls, x: Ty, y: Ty) -> Diagram:
        if len(x) == 0: return cls.id(y)
        if len(x) == 1:
            if len(y) == 1: return factory(x[0], y[0])
            return method(cls, x, y[:1]) @ cls.id(y[1:])\
                >> cls.id(y[:1]) @ method(cls, x, y[1:])  # left hexagon equation.
        return cls.id(x[:1]) @ method(cls, x[1:], y)\
            >> method(cls, x[:1], y) @ cls.id(x[1:])  # right hexagon equation.
    return classmethod(method)

Diagram.braid, Diagram.swap = hexagon(Braid), hexagon(Swap)

def naturality(self: Diagram, i: int, left=True, down=True, braid=None):
    braid = braid or self.braid
    layer, box = self.inside[i], self.inside[i].box
    if left and down:
        source = layer.left[-1] @ box >> braid(layer.left[-1], box.cod)
        target = braid(layer.left[-1], box.dom) >> box @ layer.left[-1]
    elif left: ...
    elif down: ...
    else:
        source = braid(layer.right[0], box.dom) >> box @ layer.right[0]
        target = layer.right[0] @ box >> braid(layer.right[0], box.cod)
    match = Match(top=self[:i] if down else self[:i - len(source) + 1],
                  bottom=self[i + len(source):] if down else self[i + 1:],
                  left=layer.left[:-1] if left else layer.left,
                  right=layer.right if left else layer.right[1:])
    assert self == match.subs(source)
    return match.subs(target)

Diagram.naturality = naturality

class Functor(monoidal.Functor):
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Swap):
            return self.cod.ar.swap(self(other.dom[0]), self(other.dom[1]))
        if isinstance(other, Braid) and not other.is_dagger:
            return self.cod.ar.braid(self(other.dom[0]), self(other.dom[1]))
        return super().__call__(other)
