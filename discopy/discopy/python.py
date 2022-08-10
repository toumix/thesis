from __future__ import annotations

from discopy import cat
from discopy.sugar import dataclass, inductive, Callable, Tensorable


tuplify = lambda stuff: stuff if isinstance(stuff, tuple) else (stuff, )
untuplify = lambda stuff: stuff[0] if len(stuff) == 1 else stuff


class Function(cat.Function, Tensorable):
    inside: Callable
    dom: tuple[type, ...]
    cod: tuple[type, ...]

    @inductive
    def tensor(self, other: Function) -> Function:
        def inside(*xs):
            left, right = xs[:len(self.dom)], xs[len(self.dom):]
            result = tuple(self(*left)) + tuple(other(*right))
            return result[0] if len(self.cod + other.cod) == 1 else result
        return Function(inside, self.dom + other.dom, self.cod + other.cod)
