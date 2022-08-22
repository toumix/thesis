from __future__ import annotations

from discopy import *
from discopy.python import *
from discopy.sugar import *
from discopy.cartesian import *


@dataclass
class Dict(Composable, Tensorable):
    inside: dict[int, int]
    dom: int
    cod: int

    __getitem__ = lambda self, key: self.inside[key]

    @staticmethod
    def id(x: int = 0): return Dict({i: i for i in range(x)}, x, x)

    @inductive
    def then(self, other: Dict) -> Dict:
        inside = {i: self[other[i]] for i in range(other.cod)}
        return Dict(inside, self.dom, other.cod)

    @inductive
    def tensor(self, other: Dict) -> Dict:
        inside = {i: self[i] for i in range(self.cod)}
        inside.update({
            self.cod + i: self.dom + other[i] for i in range(other.cod)})
        return Dict(inside, self.dom + other.dom, self.cod + other.cod)

    @staticmethod
    def swap(x: int, y: int) -> Dict:
        inside = {i: i + x if i < x else i - x for i in range(x + y)}
        return Dict(inside, x + y, x + y)

    @staticmethod
    def copy(x: int, n: int) -> Dict:
        return Dict({i: i % x for i in range(n * x)}, x, n * x)


def test_equations():
    x = Ty('x')
    copy, discard = Copy(x), Copy(x, 0)
    add, minus, zero = Box('+', x @ x, x), Box('-', x, x), Box('0', Ty(), x)

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
    copy, discard, swap = Diagram.copy(x), Diagram.copy(x, 0), Diagram.swap(x, x)
    F = Functor({x: 1}, {}, cod=Category(int, Dict))

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
