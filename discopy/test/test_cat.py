from discopy import *
from discopy.cat import *
from discopy.python import *
from discopy.matrix import *


def test_circuit():
    from math import sqrt

    class Circuit(Arrow): pass

    class Gate(Box, Circuit):
        cast = Circuit.cast

    Id = Circuit.id(Ob('1'))
    X, Y, Z, H = [Gate(name, Ob('1'), Ob('1')) for name in "XYZH"]

    assert (X >> Y) >> Z == X >> (Y >> Z) and X >> Id == X == Id >> X
    assert isinstance(Id, Circuit) and isinstance(X >> Y, Circuit)

    Circuit.eval = lambda self: Functor(
        ob={Ob('1'): 2},
        ar={X: [[0, 1], [1, 0]],
            Y: [[0, -1j], [1j, 0]],
            Z: [[1, 0], [0, -1]],
            H: [[x / sqrt(2) for x in row] for row in [[1, 1], [1, -1]]]},
        cod=Category(int, Matrix[complex]))(self)

    tmp, Matrix.__eq__ = Matrix.__eq__, Matrix.is_close

    for c in [X, Y, Z, H, X >> Y >> Z >> H]:
        assert (c >> c[::-1]).eval()\
            == Matrix[complex].id(2)\
            == (c[::-1] >> c).eval()

    assert (Z >> H).eval() == (H >> X).eval()
    assert (Z >> X).eval() == (X >> Z).eval().map(lambda x: -x)
    for gate in [H, Z, X]:
        assert (gate >> gate).eval() == Matrix[complex].id(2)

    Matrix.__eq__ = tmp


def test_functor():
    from typing import Iterable

    x, y, z = map(Ob, "xyz")
    f, g, h = Box('f', x, y), Box('g', y, z), Box('h', z, x)

    F = Functor(
        ob={x: y, y: z, z: x},
        ar={f: g, g: h, h: f})

    assert F(f >> g >> h) == F(f) >> F(g) >> F(h) == g >> h >> f

    G = Functor(
        ob={x: int, y: Iterable, z: int},
        ar={f: range, g: sum, h: lambda n: n * (n - 1) // 2},
        cod=Category(type, Function))
    assert G(f >> g)(42) == G(h)(42) == 861

    H = Functor(
        ob={x: 1, y: 2, z: 2},
        ar={f: [[0, 1]], g: [[0, 1], [1, 0]], h: [[1], [0]]},
        cod=Category(int, Matrix))
    assert H(f >> g) == H(h).transpose()

    I = Functor(
        ob={x: Category(Ob, Arrow), y: Category(Ob, Arrow), z: Category(int, Matrix)},
        ar={f: F, g: H},
        cod=Category(Category, Functor))
    assert I(f >> g)(h) == H(F(h)) == H(f)


def test_dagger():
    x, y, z = map(Ob, "xyz")
    f, g = Box('f', x, y), Box('g', y, z)

    assert Arrow.id(x)[::-1] == Arrow.id(x)
    assert (f >> g)[::-1] == g[::-1] >> f[::-1]


def test_neural_network():
    x, y, z = map(Ob, "xyz")

    Matrix.ReLU = lambda self: self.map(lambda x: max(x, 0))

    class Network(Arrow):
        pass

    class ReLU(Bubble, Network):
        method = "ReLU"
        cast = Network.cast

    class Box(cat.Box, Network):
        cast = Network.cast

    vector, bias = Box('vector', x, y), Box('bias', x, x)
    ones, weights = Box('ones', x, y), Box('weights', y, z)
    network = ReLU((vector + (bias >> ones)) >> weights)

    F = Functor(
        ob={x: 1, y: 4, z: 2},
        ar={vector: [[1.2, -2.3, 3.4, -4.5]],
            bias: [[-3.14]], ones: [[1, 1, 1, 1]],
            weights: [[5.6, -6.7], [7.8, -8.9],
                      [9.0, -0.1], [2.3, -3.4]]},
        cod=Category(int, Matrix[float]))

    assert F(network) == F(vector).map(lambda x: x + F(bias))\
                                  .then(F(weights)).map(lambda x: max(0, x))


def test_propositional_logic():
    import itertools

    Matrix._not = lambda self: self.map(lambda x: not x)

    class Formula(Arrow):
        pass

    class Not(Bubble, Formula):
        method = "_not"
        cast = Formula.cast

    class Proposition(Box, Formula):
        cast = Formula.cast

        def __init__(self, name):
            Box.__init__(self, name, Ob('x'), Ob('x'))

    def model(data: dict[Proposition, bool]):
        return Functor(ob={Ob('x'): 1}, ar={p: [[data[p]]] for p in data},
                       dom=Category(Ob, Formula), cod=Category(int, Matrix[bool]))

    p, q = map(Proposition, "pq")
    p_implies_q = Not(Not(q) >> p)
    not_p_or_q = Not(p) + q

    for a, b in itertools.product([0, 1], [0, 1]):
        F = model({p: a, q: b})
        assert F(p_implies_q) == (not (not F(q) and F(p)))\
            == F(not_p_or_q) == (not F(p) or F(q))
