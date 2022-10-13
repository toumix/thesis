from math import sqrt

from discopy import compact
from discopy.python import Function
from discopy.traced import *


def test_drawing():
    class TracedDrawing(compact.Diagram):
        trace = lambda self, n: super().trace(n).bubble()

    Draw = Functor(lambda x: x, lambda f: f, cod=Category(compact.Ty, TracedDrawing))
    Diagram.draw = lambda self, **params: Draw(self).draw(**params)

    a, b, x = map(Ty, "abx")
    Draw(Box('f', a @ x, b @ x).trace())


def test_fixed_point():
    phi = Function(lambda x=1: 1 + 1 / x, [int], [int]).fix()
    assert phi() == (1 + sqrt(5)) / 2
