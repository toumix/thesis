from discopy.symmetric import *
from discopy.tensor import *
from discopy.python import Function


x, y, z = map(Ty, "xyz")
a, b = Ty('a'), Ty('b')
f = Box('f', a, b)


def test_naturality():
    source, target = x @ f >> Swap(x, b), Swap(x, a) >> f @ x
    assert source.naturality(0, braid=Swap) == target
    assert target.naturality(1, left=False, down=False, braid=Swap) == source

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
