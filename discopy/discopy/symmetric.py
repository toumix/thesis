from discopy import braided
from discopy.cat import Category
from discopy.braided import Ty, Braid, hexagon


class Diagram(braided.Diagram):
    pass

class Box(braided.Box, Diagram):
    cast = Diagram.cast

class Swap(Braid, Box):
    def dagger(self):
        return Swap(*self.cod)

Diagram.braid = Diagram.swap = hexagon(Swap)

class Functor(braided.Functor):
    dom = cod = Category(Ty, Diagram)

    def __call__(self, other):
        if isinstance(other, Swap):
            return self.cod.ar.swap(self(other.dom[0]), self(other.dom[1]))
        return super().__call__(other)
