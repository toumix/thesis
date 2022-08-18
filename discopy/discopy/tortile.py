from discopy import braided, pivotal
from discopy.cat import Category
from discopy.rigid import nesting
from discopy.pivotal import Ty
from discopy.braided import hexagon

class Ty(pivotal.Ty, braided.Ty): pass

class Diagram(pivotal.Diagram, braided.Diagram): pass

class Box(pivotal.Box, braided.Box, Diagram):
    cast = Diagram.cast

class Cup(pivotal.Cup, Box): pass

class Cap(pivotal.Cap, Box): pass

class Braid(braided.Braid, Box): pass

Diagram.braid = hexagon(Braid)
Diagram.cups, Diagram.caps = nesting(Cup), nesting(Cap)

class Functor(pivotal.Functor, braided.Functor):
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Braid):
            return braided.Functor.__call__(self, other)
        return pivotal.Functor.__call__(self, other)
