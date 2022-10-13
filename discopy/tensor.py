from __future__ import annotations

from discopy.sugar import Tensorable, inductive, product
from discopy.matrix import Number, Matrix


class Tensor(Matrix):
    inside: list[list[Number]]
    dom: tuple[int, ...]
    cod: tuple[int, ...]

    def downgrade(self) -> Matrix:
        return Matrix[self.dtype](
            self.inside, product(self.dom), product(self.cod))

    @classmethod
    def id(cls, x: tuple[int, ...]) -> Tensor:
        return cls(Matrix.id(product(x)).inside, x, x)

    @inductive
    def then(self, other: Tensor) -> Tensor:
        inside = Matrix.then(*map(Tensor.downgrade, (self, other))).inside
        return type(self)(inside, self.dom, other.cod)

    @inductive
    def tensor(self, other: Tensor) -> Tensor:
        inside = Matrix.Kronecker(*map(Tensor.downgrade, (self, other))).inside
        return type(self)(inside, self.dom + other.dom, self.cod + other.cod)

    def __getitem__(self, key : int | tuple) -> Tensor:
        if isinstance(key, tuple):
            key = sum(
                key[i] * product(self.dom[i + 1:]) for i in range(len(key)))
        inside = Matrix.__getitem__(self.downgrade(), key).inside
        dom, cod = ((), self.cod) if product(self.dom) == 1 else ((), ())
        return type(self)(inside, dom, cod)

    @classmethod
    def zero(cls, dom: tuple[int, ...], cod: tuple[int, ...]) -> Tensor:
        return cls([[0 for _ in range(product(cod))]
                    for _ in range(product(dom))], dom, cod)

    def transpose(self) -> Tensor:
        inside = Matrix.transpose(self.downgrade()).inside
        return type(self)(inside, self.cod, self.dom)

    is_close = lambda self, other: self.downgrade().is_close(other.downgrade())

    @classmethod
    def swap(cls, x: tuple[int, ...], y: tuple[int, ...]) -> Tensor:
        inside = [[(i0, j0) == (i1, j1)
            for j0 in range(product(y)) for i0 in range(product(x))]
            for i1 in range(product(x)) for j1 in range(product(y))]
        return cls(inside, dom=x + y, cod=y + x)

    @classmethod
    def spider(cls, a: int, b: int, n: int, phase=None) -> Tensor:
        phase = phase or n * [1]
        inside = [[sum(phase)]] if not a and not b\
            else [[phase[xs[0]] for xs in itertools.product(*b * [range(n)])
                   if all(x == xs[0] for x in xs)]]\
            if not a else cls.spider([], a + b, n).inside
        return cls(inside, dom=a * [n], cod=b * [n])

    braid = swap


for attr in ("__bool__", "__int__", "__float__", "__complex__"):
    setattr(Tensor, attr, lambda self: getattr(self.downgrade(), attr)())


Tensor.cups = classmethod(lambda cls, x, y: cls(
    [[i == j] for i in range(product(x)) for j in range(product(y))], x + y, ()))
Tensor.caps = classmethod(lambda cls, x, y: cls(
    [[i == j for i in range(product(x)) for j in range(product(y))]], (), x + y))
