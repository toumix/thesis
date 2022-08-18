from __future__ import annotations

from discopy import *
from discopy.sugar import *
from discopy.cartesian import *


@dataclass
class Dict(Composable, Tensorable, Copyable):
    inside: dict[int, int]
    dom: int
    cod: int

    __getitem__ = lambda self, key: self.inside[key]

    @staticmethod
    def id(x: int = 0): return Dict({i: i for i in range(x)}, x, x)

    @inductive
    def then(self, other: Dict) -> Dict:
        return Dict(
            {i: other[self[i]] for i in range(self.dom)}, self.dom, other.cod)

    @inductive
    def tensor(self, other: Dict) -> Dict:
        inside = {i: self[i] for i in range(self.dom)}
        inside.update(
            {self.dom + i: self.cod + other[i] for i in range(other.dom)})
        return Dict(inside, self.dom + other.dom, self.cod + other.cod)

    @staticmethod
    def swap(x: int, y: int) -> Dict:
        inside = {i: i + x if i < x else i - x for i in range(x + y)}
        return Dict(inside, x + y, x + y)

    @staticmethod
    def merge(x: int, n: int) -> Dict:
        return Dict({i: i % x for i in range(n * x)}, n * x, x)

is_tuple = lambda typ: getattr(typ, "__origin__", None) is tuple
is_union = lambda typ: getattr(typ, "__origin__", None) is TaggedUnion

class Function(python.Function):
    is_additive = False

    def make_additive(self) -> Function:
        dom, cod = (
            (tuple[x], ) if len(x) != 1
            else x[0].__args__ if is_union(x[0])
            else x for x in (self.dom, self.cod))
        return AdditiveFunction(self.inside, dom, cod)

    @inductive
    def tensor(self, other: Function) -> Function:
        if other.is_additive:
            return self.tensor(other.make_multiplicative())
        return python.Function.tensor(self, other)

    @staticmethod
    def copy(x: tuple[type, ...], n: int, is_dagger=False):
        if is_dagger: return AdditiveFunction.merge(x, n)
        if n == 1: return Function.id(x)
        return Function(lambda *xs: n * xs, dom=x, cod=n * x)


class AdditiveFunction(Function):
    is_additive = True

    def make_multiplicative(self) -> Function:
        dom, cod = (
            (TaggedUnion[x], ) if len(x) != 1
            else x[0].__args__ if is_tuple(x[0])
            else x for x in (self.dom, self.cod))
        return Function(self.inside, dom, cod)

    @inductive
    def tensor(self, other: Function) -> Function:
        if not other.is_additive:
            return self.tensor(other.make_additive())
        inside = self.inside if len(self.dom) == 1 and len(other.dom) == 0\
            else other.inside if len(self.dom) == 0 and len(other.dom) == 1\
            else lambda i, x: (
                self(i, x) if i < len(self.dom) else other(i - len(self.dom), x))
        dom, cod = self.dom + other.dom, self.cod + other.cod
        return AdditiveFunction(inside, dom, cod)

    @staticmethod
    def merge(x: tuple[type, ...], n: int, is_dagger=False):
        if is_dagger: return Function.copy(x, n)
        if n == 1: return AdditiveFunction.id(x)
        return AdditiveFunction(lambda _, xs: xs, dom=n * x, cod=x)

    def __call__(self, *xs):
        if len(self.dom) == 1 and len(xs) == 2:
            i, x = xs
            assert i == 0
            return self(x)
        return super().__call__(*xs)


def test_equations():
    x = Ty('x')
    add, minus, zero = Box('+', x @ x, x), Box('-', x, x), Box('0', Ty(), x)
    copy, discard = Diagram.copy(x), Diagram.discard(x)

    add >> copy, copy @ copy >> x @ Swap(x, x) @ x >> add @ add
    add >> discard, discard @ discard
    zero >> discard, Diagram.id(Ty())
    copy >> minus @ x >> add, discard >> zero, copy >> x @ minus >> add

    Diagram.id(x)
    x @ zero >> x @ copy >> add @ x >> discard @ x
    x @ zero @ zero >> discard @ discard @ x
    discard >> zero


def test_FinSet():
    x = Ty('x')
    copy, discard, swap = Diagram.copy(x), Diagram.discard(x), Diagram.swap(x, x)
    F = lambda f: Functor({x: 1}, {}, cod=Category(int, Dict))(f.dagger())

    assert F(copy >> discard @ x) == F(Diagram.id(x)) == F(copy >> x @ discard)
    assert F(copy >> copy @ x) == F(Diagram.copy(x, 3)) == F(copy >> x @ copy)
    assert F(copy >> swap) == F(copy)

def test_neural_network():
    x = Ty('x')
    add = lambda n: Box('$+$', x ** n, x)
    ReLU = Box('$\sigma$', x, x)
    weights = [Box('w{}'.format(i), x, x) for i in range(4)]
    bias = Box('b', Ty(), x)

    network = Diagram.copy(x @ x, 2)\
    >> Diagram.tensor(*weights) @ bias >> add(5) >> ReLU

    F = Functor(ob={x: int}, ar={
            add(5): lambda *xs: sum(xs),
            ReLU: lambda x: max(0, x),
            bias: lambda: -1, **{
                weight: lambda x, w=w: x * w
                for weight, w in zip(weights, range(4))}},
        cod=Category(tuple[type, ...], Function))

    assert F(network)(42, 43) == max(0, sum([42 * 0, 43 * 1, 42 * 2, 43 * 3, -1]))

    y, z, w = map(Ty, "yzw")
    f, g = Box('f', y, z), Box('g', z, w)
    pipeline = f @ z >> Diagram.merge(z, 2) >> g

    G = Functor(
        ob={y: str, z: tuple[int, int], w: int},
        ar={f: lambda s: (len(set(s)), len(s)),
            g: F(network)},
        cod=Category(tuple[type, ...], AdditiveFunction))

    assert G(pipeline)(0, "hello") == F(network)(4, 5)
