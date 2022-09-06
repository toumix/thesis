from __future__ import annotations

from discopy.sugar import (
    dataclass, inductive, Composable, DictOrCallable, FakeDict)


@dataclass
class Ob:
    name: str
    __str__ = lambda self: self.name


@dataclass
class Arrow(Composable):
    inside: tuple[Box, ...]
    dom: Ob
    cod: Ob

    def __eq__(self, other):
        return False if not isinstance(other, Arrow)\
            else other.terms == (self, ) if isinstance(other, Sum)\
            else (self.inside, self.dom, self.cod)\
                == (other.inside, other.dom, other.cod)

    @classmethod
    def cast(cls, old: Arrow) -> Arrow:
        return old if isinstance(old, cls) else cls(old.inside, old.dom, old.cod)

    @classmethod
    def id(cls, x: Ob) -> Arrow:
        return cls.cast(Arrow((), x, x))

    def then(self, *others: Arrow) -> Arrow:
        if any(isinstance(other, Sum) for other in others):
            return self.sum.cast(self).then(*others)
        for f, g in zip((self, ) + others, others): assert f.cod == g.dom
        dom, cod = self.dom, others[-1].cod if others else self.cod
        inside = self.inside + sum([other.inside for other in others], ())
        return self.cast(Arrow(inside, dom, cod))

    def dagger(self):
        inside = tuple(box.dagger() for box in self.inside[::-1])
        return self.cast(Arrow(inside, self.cod, self.dom))

    def __getitem__(self, key: int | slice) -> Arrow:
        if isinstance(key, slice):
            if key.step == -1:
                inside = tuple(box.dagger() for box in self.inside[key])
                return self.cast(Arrow(inside, self.cod, self.dom))
            if (key.step or 1) != 1:
                raise IndexError
            inside = self.inside[key]
            if not inside:
                if (key.start or 0) >= len(self):
                    return self.id(self.cod)
                if (key.start or 0) <= -len(self):
                    return self.id(self.dom)
                return self.id(self.inside[key.start or 0].dom)
            return self.cast(Arrow(inside, inside[0].dom, inside[-1].cod))
        return self.inside[key]

    @classmethod
    def zero(cls, dom: Ob, cod: Ob) -> Arrow:
        return cls.sum((), dom, cod)

    __len__ = lambda self: len(self.inside)
    __str__ = lambda self: ' >> '.join(map(str, self.inside))\
        if self.inside else '{}.id({})'.format(type(self).__name__, self.dom)
    __add__ = lambda self, other: self.sum.cast(self) + other
    __radd__ = lambda self, other: self if not other else self + other
    __lt__ = lambda self, other: hash(self) < hash(other)  # An arbitrary order.


class Box(Arrow):
    cast = Arrow.cast

    def __init__(self, name: str, dom: Ob, cod: Ob, is_dagger=False):
        # This allows sesqui.Ty to subclass both Box and Ob.
        object.__setattr__(self, "name", name)
        self.is_dagger = is_dagger
        super().__init__((self, ), dom, cod)

    def __eq__(self, other):
        if isinstance(other, Box):
            return (self.name, self.dom, self.cod, self.is_dagger)\
                == (other.name, other.dom, other.cod, other.is_dagger)
        return isinstance(other, Arrow) and other.inside == (self, )

    def dagger(self):
        return type(self)(
            self.name, self.cod, self.dom, is_dagger=not self.is_dagger)

    def __repr__(self):
        if self.is_dagger:
            return repr(self.dagger()) + ".dagger()"
        return "Box({}, {}, {})".format(*map(repr, (
            self.name, self.dom, self.cod)))

    __str__ = lambda self: self.name + ("[::-1]" if self.is_dagger else "")
    __hash__ = lambda self: hash(repr(self))


@dataclass
class Category:
    ob: type = Ob
    ar: type = Arrow


class Functor(Composable):
    ob: dictOrCallable[Ob, Ob]
    ar: dictOrCallable[Box, Ar]
    dom: Category = Category(Ob, Arrow)
    cod: Category = Category(Ob, Arrow)

    def __init__(self, ob, ar, dom=None, cod=None):
        dom, cod = dom or type(self).dom, cod or type(self).cod
        ob = ob if hasattr(ob, "__getitem__") else FakeDict(ob)
        ar = ar if hasattr(ar, "__getitem__") else FakeDict(ar)
        self.ob, self.ar, self.dom, self.cod = ob, ar, dom, cod

    def __call__(self, other: Ob | Arrow) -> Ob | Arrow:
        if isinstance(other, Sum):
            unit = self.cod.ar.zero(self(other.dom), self(other.cod))
            return sum([self(f) for f in other.terms], unit)
        if isinstance(other, Bubble):
            method = getattr(self.cod.ar, other.method)
            return method(self(other.diagram)) if other.is_id_on_objects else\
                method(self(other.diagram), self(other.dom), self(other.cod))
        if isinstance(other, Ob): return self.ob[other]
        if isinstance(other, Box) and other.is_dagger:
            return self(other.dagger()).dagger()
        if isinstance(other, Box):
            result = self.ar[other]
            if isinstance(result, self.cod.ar): return result
            # This allows some nice syntactic sugar for the ar mapping.
            return self.cod.ar(result, self(other.dom), self(other.cod))
        if isinstance(other, Arrow):
            base_case = self.cod.ar.id(self(other.dom))
            return base_case.then(*[self(box) for box in other.inside])
        raise TypeError

    @classmethod
    def id(cls, x: Category) -> Functor:
        return cls(lambda obj: obj, lambda box: box, dom=x, cod=x)

    @inductive
    def then(self: Functor, other: Functor) -> Functor:
        assert self.cod == other.dom
        ob, ar = lambda x: other.ob[self.ob[x]], lambda f: other.ar[self.ar[f]]
        return type(self)(ob, ar, self.dom, other.cod)


class Sum(Box):
    def __init__(self, terms: tuple[Arrow, ...], dom: Ob, cod: Ob):
        assert all(f.dom == dom and f.cod == cod for f in terms)
        self.terms, name = terms, "Sum({}, {}, [{}])".format(
            dom, cod, ", ".join(map(str, terms)))
        Box.__init__(self, name, dom, cod)

    def __eq__(self, other):
        if isinstance(other, Sum):
            return (self.dom, self.cod, sorted(self.terms))\
                == (other.dom, other.cod, sorted(other.terms))
        return self.terms == (other, )

    def __add__(self, other):
        if not isinstance(other, Sum): return self + self.cast(other)
        return self.sum(self.terms + other.terms, self.dom, self.cod)

    @classmethod
    def cast(cls, old: cat.Arrow) -> Sum:
        return old if isinstance(old, cls) else cls((old, ), old.dom, old.cod)

    id = lambda x: Sum.cast(Arrow.id(x))

    @inductive
    def then(self, other):
        terms = tuple(f.then(g) for f in self.terms for g in self.cast(other).terms)
        return type(self)(terms, self.dom, other.cod)

    def dagger(self):
        return type(self)(tuple(f.dagger() for f in self.terms), self.cod, self.dom)


class Bubble(Box):
    method = "bubble"

    def __init__(self, diagram: Arrow, dom=None, cod=None, **params):
        self.diagram = diagram
        self.method = params.pop("method", type(self).method)
        name = "Bubble({}, {}, {})".format(diagram, dom, cod)
        dom, cod = dom or diagram.dom, cod or diagram.cod
        super().__init__(name, dom, cod, **params)

    def dagger(self):
        return type(self)(
            self.diagram, self.dom, self.cod, is_dagger=not self.is_dagger)

    @property
    def is_id_on_objects(self):
        return self.dom == self.diagram.dom and self.cod == self.diagram.cod


Arrow.sum, Arrow.bubble = Sum, lambda self, **kwargs: Bubble(self, **kwargs)
