from __future__ import annotations

from discopy import cat, monoidal
from discopy.cat import Category, Functor
from discopy.sugar import (
    dataclass, inductive, DictOrCallable, Composable, Tensorable)
from discopy.monoidal import Match


class Colour(cat.Ob):
    pass

class TyArrow(cat.Arrow, monoidal.Ty):
    @inductive
    def tensor(self, other):
        if isinstance(other, TyArrow):
            return cat.Arrow.then(self, other)
        return NotImplemented  # Allows whiskering on the left.

    __matmul__ = tensor

class Ty(cat.Box, TyArrow):
    cast = TyArrow.cast

class Layer(monoidal.Layer):
    def __init__(self, left: Ty, box: monoidal.Box, right: Ty):
        assert left.cod == box.dom.dom and box.dom.cod == right.dom
        super().__init__(left, box, right)

class Diagram(monoidal.Diagram):
    pass

class Box(monoidal.Box, Diagram):
    def __init__(self, name: str, dom: Ty, cod: Ty):
        assert (dom.dom, dom.cod) == (cod.dom, cod.cod)
        monoidal.Box.__init__(self, name, dom, cod)
        Diagram.__init__(self, (Layer.cast(self), ), dom, cod)

    cast = Diagram.cast

@dataclass
class TwoCategory:
    colours: type = Colour
    ob: type = Ty
    ar: type = Diagram

@dataclass
class TwoFunctor(monoidal.Functor):
    colours: DictOrCallable[Colour, Colour]
    ob: DictOrCallable[Ty, Ty]
    ar: DictOrCallable[Box, Diagram]

    dom: TwoCategory = TwoCategory()
    cod: TwoCategory = TwoCategory()

    def __call__(self, other):
        if isinstance(other, Colour):
            return self.colours[other]
        if isinstance(other, Ty):
            return self.ob[other]
        if isinstance(other, TyArrow):
            base_case = self.cod.ob.id(self(other.dom))
            return base_case.then(*[self(box) for box in other.inside])
        return super().__call__(other)


class Transformation(Composable, Tensorable):
    def __init__(self, inside: Callable, dom: Functor, cod: Functor):
        assert (dom.dom, dom.cod) == (cod.dom, cod.cod)
        self.inside, self.dom, self.cod = inside, dom, cod

    @staticmethod
    def id(F: Functor):
        return Transformation(lambda x: F.cod.ar.id(F(x)), dom=F, cod=F)

    @inductive
    def then(self, other: Transformation) -> Transformation:
        return Transformation(lambda x: self(x) >> other(x), self.dom, other.cod)

    @inductive
    def tensor(self, other: Transformation) -> Transformation:
        return self @ other.dom >> self.cod @ other

    def __matmul__(self, other):
        if isinstance(other, Functor):
            return Transformation(
                lambda x: other(self(x)), self.dom >> other, self.cod >> other)
        return self.tensor(other)

    def __rmatmul__(self, other):
        if isinstance(other, Functor):
            return Transformation(
                lambda x: self(other(x)), other >> self.dom, other >> self.cod)
        raise TypeError

    def __call__(self, other: Ty) -> Transformation:
        inside, dom, cod = self.inside(other), self.dom(other), self.cod(other)
        return self.cod.cod.ar(inside, dom, cod)

Cat = TwoCategory(colours=Category, ob=Functor, ar=Transformation)


class Slice(monoidal.Box):
    def __init__(self, rule: Rule, match: Match):
        dom, cod = match.subs(rule.dom), match.subs(rule.cod)
        super().__init__("Slice({}, {})".format(rule, match), dom, cod)

    @classmethod
    def cast(cls, old: Rule) -> Slice:
        x, y = old.dom.dom, old.cod.cod
        top, bottom, left, right = old.id(x), old.id(y), x[:0], y[len(y):]
        return cls(old, Match(top, bottom, left, right))

class Rewrite(Diagram):
    inside: tuple[Slice, ...]
    dom: Diagram
    cod: Diagram

class Rule(monoidal.Box, Rewrite):
    def __init__(self, name: str, dom: Diagram, cod: Diagram):
        monoidal.Box.__init__(self, name, dom, cod)
        Rewrite.__init__(self, (Slice.cast(self), ), dom, cod)
