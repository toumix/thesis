from discopy.braided import *


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
            Diagram.braid, (lambda x, y: Diagram.braid(y, x)[::-1])]:
        source, target = x @ f >> braid(x, b), braid(x, a) >> f @ x
        assert source.naturality(0, braid=braid) == target
        assert target.naturality(1, left=False, down=False, braid=braid) == source
