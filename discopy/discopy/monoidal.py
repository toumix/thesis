from __future__ import annotations

from discopy import cat
from discopy.cat import Ob, Category
from discopy.sugar import dataclass, inductive, Tensorable


class Ty(Ob):
    def __init__(self, inside: Optional[tuple[Ob | str, ...]] = ()):
        self.inside = tuple(x if isinstance(x, Ob) else Ob(x) for x in inside)
        name = ' @ '.join(map(str, inside)) if inside\
            else "{}()".format(type(self).__name__)
        super().__init__(name)

    def tensor(self, *others: Ty) -> Ty:
        if all(isinstance(other, Ty) for other in others):
            inside = self.inside + sum([other.inside for other in others], ())
            return type(self)(inside)
        return NotImplemented  # This will allow whiskering on the left.

    def __getitem__(self, key):
        if isinstance(key, slice):
            return type(self)(self.inside[key])
        return type(self)((self.inside[key], ))

    __matmul__ = __add__ = tensor
    __pow__ = lambda self, n: type(self)(n * self.inside)
    __len__ = lambda self: len(self.inside)
    __repr__ = lambda self: "{}({})".format(
        type(self).__name__, ', '.join(map(repr, self.inside)))


class Layer(cat.Box):
    def __init__(self, left: Ty, box: cat.Box, right: Ty):
        self.left, self.box, self.right = left, box, right
        name = ("{} @ ".format(left) if left else "") + box.name\
            + (" @ {}".format(right) if right else "")
        dom, cod = left @ box.dom @ right, left @ box.cod @ right
        if not isinstance(dom, Ty): import pdb; pdb.set_trace()
        super().__init__(name, dom, cod)

    def __matmul__(self, other: Ty) -> Layer:
        return Layer(self.left, self.box, self.right @ other)

    def __rmatmul__(self, other: Ty) -> Layer:
        return Layer(other @ self.left, self.box, self.right)

    @classmethod
    def cast(cls, old: cat.Box) -> Layer:
        if isinstance(old, cls): return old
        return cls(old.dom[:0], old, old.dom[:0])

    __repr__ = lambda self: "Layer({}, {}, {})".format(
        *map(repr, [self.left, self.box, self.right]))



class Diagram(cat.Arrow, Tensorable):
    inside: tuple[Layer, ...]
    dom: Ty
    cod: Ty

    @inductive
    def tensor(self, other: Diagram) -> Diagram:
        layers = tuple(layer @ other.dom for layer in self.inside)\
            + tuple(self.cod @ layer for layer in other.inside)
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        return self.cast(Diagram(layers, dom, cod))

    def interchange(self, i: int, left=False) -> Diagram: ...
    def normal_form(self, left=False) -> Diagram: ...
    def draw(self, **params): ...


class Box(cat.Box, Diagram):
    def __init__(self, name: str, dom: Ty, cod: Ty, **params):
        cat.Box.__init__(self, name, dom, cod, **params)
        Diagram.__init__(self, (Layer.cast(self), ), dom, cod)

    def __eq__(self, other):
        if isinstance(other, Box):
            return cat.Box.__eq__(self, other)
        if isinstance(other, Diagram):
            return other.inside == (Layer.cast(self), )
        return False

    __hash__ = cat.Box.__hash__
    cast = Diagram.cast


class Functor(cat.Functor):
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other : Ty | Diagram) -> Ty | Diagram:
        if isinstance(other, Ty):
            return sum([self(obj) for obj in other.inside], self.cod.ob())
        if isinstance(other, Ob):
            result = self.ob[self.dom.ob((other, ))]
            return result if isinstance(result, self.cod.ob)\
                else self.cod.ob(result)
        if isinstance(other, Layer):
            return self(other.left) @ self(other.box) @ self(other.right)
        return super().__call__(other)
