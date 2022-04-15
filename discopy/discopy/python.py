from __future__ import annotations

from discopy.sugar import dataclass, inductive, Composable, Callable


tuplify = lambda stuff: stuff if isinstance(stuff, tuple) else (stuff, )
untuplify = lambda stuff: stuff[0] if len(stuff) == 1 else stuff


@dataclass
class Function(Composable):
    inside: Callable
    dom: type
    cod: type

    @staticmethod
    def id(dom: type) -> Function:
        return Function(lambda x: x, dom, dom)

    @inductive
    def then(self, other: Function) -> Function:
        assert self.cod == other.dom
        return Function(lambda x: other(self(x)), self.dom, other.cod)

    def __call__(self, x):
        return self.inside(x)
