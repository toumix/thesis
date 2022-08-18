from discopy.tortile import *


def test_Kauffman():
    x = Ty(('$x$', ))
    A, A.inverse = Box('$A$', Ty(), Ty()), Box('$A^{-1}$', Ty(), Ty())

    class Polynomial(Diagram):
        def braid(x, y):
            assert x == y and len(x) == len(y) == 1
            return (A @ x @ y) + (Cup(x, y) >> A.inverse >> Cap(x, y))

    Kauffman = Functor(ob={x: x}, ar={}, cod=Category(Ty, Polynomial))

    assert Kauffman(Braid(x, x))\
        == (A @ x @ x) + (Cup(x, x) >> A.inverse >> Cap(x, x))
