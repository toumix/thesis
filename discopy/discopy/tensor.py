from __future__ import annotations

from discopy.sugar import Tensorable, inductive
from discopy.matrix import Number, Matrix


def product(x, unit=1):
    return unit if not x else product(x[1:], x[0] * unit)


class Tensor(Tensorable, Matrix):
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
        dom, cod = ((), cod) if product(self.dom) == 1 else ((), ())
        return type(self)(inside, dom, cod)

for attr in ("__bool__", "__int__", "__float__", "__complex__"):
    setattr(Tensor, attr, lambda self: getattr(self.downgrade(), attr)())
