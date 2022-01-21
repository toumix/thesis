class Diagram(cat.Arrow):
    def __init__(self, dom: Ty, cod: Ty, layers: list[Layer]):
        self.layers = layers
        super().__init__(dom, cod, boxes=layers)

    @staticmethod
    def upgrade(old: cat.Arrow) -> Diagram:
        if isinstance(old, Diagram): return old
        layers = list(map(Layer.update, old.boxes))
        dom, cod = map(Ty.upgrade, (old.dom, old.cod))
        return Diagram(dom, cod, layers)

    @staticmethod
    def subclass(ar_factory):
        def upgrade(old: cat.Arrow) -> ar_factory:
            if not isinstance(old, Diagram): old = Diagram.upgrade(old)
            return ar_factory(old.dom, old.cod, old.layers)
        ar_factory.upgrade = staticmethod(upgrade)
        return ar_factory

    def tensor(self, other: Diagram) -> Diagram:
        dom, cod = self.dom @ other.dom, self.cod @ other.cod
        layers = [layer @ other.dom for layer in self.layers]
        layers += [self.cod @ layer for layer in other.layers]
        return self.upgrade(Diagram(dom, cod, layers))

    __matmul__ = tensor

    def interchange(self, i: int, j: int, left=False) -> Diagram: ...

    def normal_form(self) -> Diagram: ...

    def draw(self, **params): ...
