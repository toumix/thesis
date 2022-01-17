@dataclass
class Ob:
    name : Any

@dataclass
class Arrow:
    dom : Ob
    cod : Ob
    boxes : List[Arrow]

    @staticmethod
    def id(x : Ob) -> Arrow:
        return Arrow(x, x, [])

    def then(self, *others : Arrow) -> Arrow:
        if not others: return self
        return Arrow(
            self.dom, others[-1].cod, self.boxes + sum(
                other.boxes for other in others, []))

class Box(Arrow):
    def __init__(self, name, dom, cod):
        self.name = name
        super().__init__(dom, cod, [self])

    def __eq__(self, other):
        if not isinstance(other, Arrow): return False
        if isinstance(other, Box):
            return (self.name, self.dom, self.cod)\
                == (other.name, other.dom, other.cod)
        return len(other.boxes) == 1 and self == other.boxes[0]

@dataclass
class Functor:
    ob : Mapping[Ob, Ob]
    ar : Mapping[Box, Arrow]
    ob_factory = Ob
    ar_factory = Arrow

    def __call__(self, other):
        if isinstance(other, Ob):
            return self.ob[other]
        if isinstance(other, Arrow):
            return ar_factory.id(self.ob[other.dom]).then(
                *self.ar[box] for box in other.boxes)
        raise TypeError
