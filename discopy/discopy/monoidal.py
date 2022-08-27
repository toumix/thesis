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
            return self.cast(inside)
        return NotImplemented  # This will allow whiskering on the left.

    def __getitem__(self, key):
        if isinstance(key, slice):
            return self.cast(self.inside[key])
        return self.cast((self.inside[key], ))

    __matmul__ = __add__ = tensor
    __pow__ = lambda self, n: self.cast(n * self.inside)
    __len__ = lambda self: len(self.inside)
    __repr__ = lambda self: "{}({})".format(
        type(self).__name__, ', '.join(map(repr, self.inside)))

    cast = classmethod(lambda cls, inside: cls(inside))


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

    def __iter__(self): yield self.left; yield self.box; yield self.right

    def dagger(self) -> Layer:
        return Layer(self.left, self.box.dagger(), self.right)


@dataclass
class Encoding:
    dom: Ty
    boxes_and_offsets: tuple[tuple[Box, int], ...]


@dataclass
class Match:
    top: Diagram
    bottom: Diagram
    left: Ty
    right: Ty

    def subs(self, target):
        return self.top >> self.left @ target @ self.right >> self.bottom


class Diagram(cat.Arrow, Tensorable):
    inside: tuple[Layer, ...]
    dom: Ty
    cod: Ty

    @inductive
    def tensor(self, other: Diagram) -> Diagram:
        if isinstance(other, Sum):
            self.sum.cast(self).tensor(other)
        layers = tuple(layer @ other.dom for layer in self.inside)\
            + tuple(self.cod @ layer for layer in other.inside)
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        return self.cast(Diagram(layers, dom, cod))

    def interchange(self, i: int, left=False) -> Diagram: ...
    def normal_form(self, left=False) -> Diagram: ...
    def draw(self, **params): ...

    @property
    def boxes(self) -> tuple[Diagram]:
        return tuple(box for _, box, _ in self.inside)

    @property
    def offsets(self) -> tuple[int]:
        return tuple(len(left) for left, _, _ in self.inside)

    def encode(self) -> Encoding:
        return Encoding(self.dom, tuple(zip(self.boxes, self.offsets)))

    @staticmethod
    def decode(encoding: Encoding) -> Diagram:
        diagram = Diagram.id(encoding.dom)
        for box, offset in encoding.boxes_and_offsets:
            left, right =\
                diagram.cod[:offset], diagram.cod[offset + len(box.dom):]
            diagram >>= left @ box @ right
        return diagram

    def match(self, pattern: Diagram) -> Iterator[Match]:
        for i in range(len(self) - len(pattern) + 1):
            for j in range(len(self[i].dom) + 1):
                match = Match(
                    self[:i], self[i + len(pattern):],
                    self[i].dom[:j], self[i].dom[j + len(pattern.dom):])
                well_typed =\
                    match.top.cod == match.left @ pattern.dom @ match.right and\
                    match.left @ pattern.cod @ match.right == match.bottom.dom
                if well_typed and self == match.subs(pattern): yield match

    def draw(self):
        import networkx
        from matplotlib import pyplot as plt
        from discopy.drawing import draw
        graph, position = draw(self)
        networkx.draw(graph, position,
                      labels={node: node.label for node in graph.nodes})
        plt.show()


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
            dtype = getattr(self.cod.ob, "__origin__", self.cod.ob)
            return result if isinstance(result, dtype)\
                else self.cod.ob((result, ))  # Syntactic sugar for {x: n}.
        if isinstance(other, Layer):
            return self(other.left) @ self(other.box) @ self(other.right)
        return super().__call__(other)


class Sum(cat.Sum, Box):
    id = lambda x: Sum.cast(Diagram.id(x))

    @inductive
    def tensor(self, other: Sum) -> Sum:
        terms = tuple(f @ g for f in self.terms for g in self.cast(other).terms)
        return type(self)(terms, self.dom @ other.dom, self.cod @ other.cod)


class Bubble(cat.Bubble, Box):
    pass


Diagram.bubble = lambda self, **kwargs: Bubble(self, **kwargs)
Diagram.sum = Sum
