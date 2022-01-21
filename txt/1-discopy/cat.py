@dataclass
class Ob:
    name: str

@dataclass
class Arrow:
    dom: Ob
    cod: Ob
    boxes: list[Arrow]

    @staticmethod
    def upgrade(old: Arrow) -> Arrow:
        return old

    @staticmethod
    def id(x: Ob) -> Arrow:
        return self.upgrade(Arrow(x, x, []))

    def then(self, *others: Arrow) -> Arrow:
        if not others: return self
        return self.upgrade(Arrow(
            self.dom, others[-1].cod, self.boxes + sum(
                [other.boxes for other in others], [])))

    __rshift__ = then

class Box(Arrow):
    def __init__(self, name: str, dom: Ob, cod: Ob):
        self.name = name
        super().__init__(dom, cod, [self])

    def __eq__(self, other):
        if not isinstance(other, Arrow): return False
        if isinstance(other, Box):
            return (self.name, self.dom, self.cod)\
                == (other.name, other.dom, other.cod)
        return other.boxes == [self]
