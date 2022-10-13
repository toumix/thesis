from discopy import symmetric
from discopy.rigid import nesting
from discopy.symmetric import Ty, Diagram, Box


class Spider(Box):
    def __init__(self, a: int, b: int, x: Ty, phase=None):
        assert len(x) == 1
        self.object, self.phase = x, phase or 0
        name = "Spider({})".format(', '.join(map(str, (a, b, x, phase))))
        super().__init__(name, dom=x ** a, cod=x ** b)

    def dagger(self):
        a, b, x = len(self.cod), len(self.dom), self.object
        phase = None if self.phase is None else -self.phase
        return Spider(a, b, x, phase)


def coherence(factory):
    def method(cls, a: int, b: int, x: Ty, phase=None) -> Diagram:
        if len(x) == 0 and phase is None: return cls.id(x)
        if len(x) == 1: return factory(a, b, x, phase)
        if phase is not None:  # Coherence for phase shifters.
            shift = cls.tensor(*[factory(1, 1, obj, phase) for obj in x])
            return method(cls, a, 1, x) >> shift >> method(cls, 1, b, x)
        if (a, b) in [(1, 0), (0, 1)]: # Coherence for (co)units.
            return cls.tensor(*[factory(a, b, obj) for obj in x])
        # Coherence for binary (co)products.
        if (a, b) in [(1, 2), (2, 1)]:
            spiders, braids = (
                factory(a, b, x[0], phase) @ method(cls, a, b, x[1:], phase),
                x[0] @ cls.braid(x[0], x[1:]) @ x[1:])
            return spiders >> braids if (a, b) == (1, 2) else braids >> spiders
        if a == 1:  # We can now assume b > 2.
            return method(cls, 1, b - 1, x)\
                >> method(cls, 1, 2, x) @ (x ** (b - 2))
        if b == 1:  # We can now assume a > 2.
            return method(cls, 2, 1, x) @ (x ** (a - 2))\
                >> method(cls, a - 1, 1, x)
        return method(cls, a, 1, x) >> method(cls, 1, b, x)
    return classmethod(method)


Diagram.spiders = coherence(Spider)
Diagram.cups = nesting(lambda x, _: Spider(0, 2, x))
Diagram.caps = nesting(lambda x, _: Spider(2, 0, x))


class Functor(symmetric.Functor):
    def __call__(self, other):
        if isinstance(other, Spider):
            a, b = len(other.dom), len(other.cod)
            x, phase = other.object, other.phase
            return self.cod.ar.spiders(a, b, self(x), phase)
        return super().__call__(other)
