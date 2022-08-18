from discopy import rigid
from discopy.cat import Category
from discopy.rigid import nesting


class Ob(rigid.Ob):
    l = r = property(lambda self: self.cast(Ob(self.name, (self.z + 1) % 2)))

class Ty(rigid.Ty, Ob):
    def __init__(self, inside=()):
        rigid.Ty.__init__(self, inside=map(Ob.cast, inside))

class Diagram(rigid.Diagram): pass

class Box(rigid.Box, Diagram):
    cast = Diagram.cast

class Cup(rigid.Cup, Box):
    def dagger(self):
        return Cap(self.dom[0], self.dom[1])

class Cap(rigid.Cap, Box):
    def dagger(self):
        return Cup(self.cod[0], self.cod[1])

Diagram.cups, Diagram.caps = nesting(Cup), nesting(Cap)

class Functor(rigid.Functor):
    dom = cod = Category(Ty, Diagram)
