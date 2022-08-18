from discopy import *
from discopy.rigid import *
from discopy.braided import *
from discopy.tensor import *
from discopy.python import *


x, y, z = map(Ty, "xyz")
a, b = Ty('a'), Ty('b')
f = Box('f', a, b)

def test_hexagon():
    assert Diagram.braid(x, y @ z) == Braid(x, y) @ z >> y @ Braid(x, z)
    assert Diagram.braid(x @ y, z) == x @ Braid(y, z) >> Braid(x, z) @ y


def test_simplify():
    assert (Diagram.braid(x, y @ z) >> Diagram.braid(x, y @ z)[::-1]).simplify()\
        == Diagram.id(x @ y @ z)\
        == (Diagram.braid(y @ z, x)[::-1] >> Diagram.braid(y @ z, x)).simplify()


def test_naturality():
    for braid in [
            Diagram.braid, (lambda x, y: Diagram.braid(y, x)[::-1]), Diagram.swap]:
        source, target = x @ f >> braid(x, b), braid(x, a) >> f @ x
        assert source.naturality(0, braid=braid) == target
        assert target.naturality(1, left=False, down=False, braid=braid) == source

def test_functor():
    swap_twice = Diagram.swap(x, y @ z) >> Diagram.swap(y @ z, x)

    F = Functor(
        ob={a: 1, b: 2, x: 3, y: 4, z: 5},
        ar={f: [[1-2j, 3+4j]]},
        cod=Category(tuple[int, ...], Tensor[complex]))

    assert F(f @ x >> Swap(b, x)) == F(Swap(a, x) >> x @ f)
    assert F(x @ f >> Swap(x, b)) == F(Swap(x, a) >> f @ x)
    assert F(swap_twice).is_close(Tensor.id(F(x @ y @ z)))

    G = Functor(
        ob={a: complex, b: float, x: int, y: bool, z: str},
        ar={f: lambda z: abs(z) ** 2},
        cod=Category(tuple[type, ...], Function))

    assert G(f @ x >> Swap(b, x))(1j, 2) == G(Swap(a, x) >> x @ f)(1j, 2)
    assert G(x @ f >> Swap(x, b))(2, 1j) == G(Swap(x, a) >> f @ x)(2, 1j)
    assert G(swap_twice)(42, True, "foo") == (42, True, "foo")
