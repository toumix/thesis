from __future__ import annotations

from discopy import cat
from discopy.sugar import dataclass, inductive, Callable, Composable, Tensorable


tuplify = lambda stuff: stuff if isinstance(stuff, tuple) else (stuff, )
untuplify = lambda stuff: stuff[0] if len(stuff) == 1 else stuff


@dataclass
class Function(Composable, Tensorable):
    inside: Callable
    dom: tuple[type, ...]
    cod: tuple[type, ...]

    @classmethod
    def id(cls, dom: type) -> Function:
        return cls(lambda *xs: xs, dom, dom)

    @inductive
    def then(self, other: Function) -> Function:
        assert self.cod == other.dom
        inside = lambda *args: other(*tuplify(self(*args)))
        return type(self)(inside, self.dom, other.cod)

    def __call__(self, *xs):
        return self.inside(*xs)

    @inductive
    def tensor(self, other: Function) -> Function:
        def inside(*xs):
            left, right = xs[:len(self.dom)], xs[len(self.dom):]
            return untuplify(tuplify(self(*left)) + tuplify(other(*right)))
        return type(self)(inside, self.dom + other.dom, self.cod + other.cod)

    @classmethod
    def swap(cls, x: tuple[type, ...], y: tuple[type, ...]) -> Function:
        def inside(*xs):
            assert len(xs) == len(x + y)
            return untuplify(xs[len(x):] + xs[:len(x)])
        return cls(inside, dom=x + y, cod=y + x)

    braid = swap
