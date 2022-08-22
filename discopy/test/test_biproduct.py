from discopy.biproduct import *


def test_conditional():
    unit = FakeInt()
    true = Biproduct.copy(unit, 2)\
        >> (Biproduct.id(unit) | Biproduct.zero(unit, unit))
    false = Biproduct.copy(unit, 2)\
        >> (Biproduct.zero(unit, unit) | Biproduct.id(unit))

    x, y = Ty('x'), Ty('y')
    f, g = Box('f', x, y), Box('g', x, y)
    conditional = (f | g) >> Biproduct.merge(FakeInt((y, )), 2)

    assert true @ FakeInt((x, )) >> conditional == f\
        and false @ FakeInt((x, )) >> conditional == g
