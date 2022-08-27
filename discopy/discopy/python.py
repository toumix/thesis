from __future__ import annotations

from discopy import cat
from discopy.sugar import dataclass, inductive, Callable, Composable, Tensorable


Ty = tuple[type, ...]

tuplify = lambda stuff: stuff if isinstance(stuff, tuple) else (stuff, )
untuplify = lambda stuff: stuff[0] if len(stuff) == 1 else stuff
is_tuple = lambda typ: getattr(typ, "__origin__", None) is tuple


def exp(base: Ty, exponent: Ty) -> Ty:
    return (Callable[exponent, tuple[base]], )


@dataclass
class Function(Composable, Tensorable):
    inside: Callable
    dom: Ty
    cod: Ty

    @classmethod
    def id(cls, dom: type) -> Function:
        return cls(lambda *xs: xs, dom, dom)

    @inductive
    def then(self, other: Function) -> Function:
        assert self.cod == other.dom
        inside = lambda *args: other(*tuplify(self(*args)))
        return Function(inside, self.dom, other.cod)

    def __call__(self, *xs):
        return self.inside(*xs)

    @inductive
    def tensor(self, other: Function) -> Function:
        def inside(*xs):
            left, right = xs[:len(self.dom)], xs[len(self.dom):]
            return untuplify(tuplify(self(*left)) + tuplify(other(*right)))
        return Function(inside, self.dom + other.dom, self.cod + other.cod)

    @classmethod
    def swap(cls, x: Ty, y: Ty) -> Function:
        def inside(*xs):
            return untuplify(tuplify(xs)[len(x):] + tuplify(xs)[:len(x)])
        return cls(inside, dom=x + y, cod=y + x)

    braid = swap

    @staticmethod
    def copy(x: Ty, n: int):
        return Function(lambda *xs: n * xs, dom=x, cod=n * x)

    def curry(self, n=1, left=True) -> Function:
        inside = lambda *xs: lambda *ys: self(*(xs + ys) if left else (ys + xs))
        if left:
            dom = self.dom[:len(self.dom) - n]
            cod = exp(self.cod, self.dom[len(self.dom) - n:])
        else: dom, cod = self.dom[n:], exp(self.cod, self.dom[:n])
        return Function(inside, dom, cod)

    @staticmethod
    def ev(base: Ty, exponent: Ty, left=True) -> Function:
        if left:
            inside = lambda f, *xs: f(*xs)
            return Function(inside, exp(base, exponent) + exponent, base)
        inside = lambda *xs: xs[-1](*xs[:-1])
        return Function(inside, exponent + exp(base, exponent), base)

    def uncurry(self, left=True) -> Function:
        base, exponent = self.cod[0].__args__[-1], self.cod[0].__args__[:-1]
        base = tuple(base.__args__) if is_tuple(base) else (base, )
        return self @ exponent >> Function.ev(base, exponent) if left\
            else exponent @ self >> Function.ev(base, exponent, left=False)

    exp = under = over = staticmethod(exp)
