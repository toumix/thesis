import dataclasses
from typing import Callable


def dataclass(cls):
    return dataclasses.dataclass(eq=True, frozen=True)(cls)


def inductive(method):
    def result(self, *others):
        if not others: return self
        if len(others) == 1: return method(self, others[0])
        if len(others) > 1: return result(method(self, others[0]), *others[1:])
    return result


class Composable:
    __rshift__ = __llshift__ = lambda self, other: self.then(other)
    __lshift__ = __lrshift__ = lambda self, other: other.then(self)


class Tensorable:
    @classmethod
    def whisker(cls, other):
        return other if isinstance(other, Tensorable) else cls.id(other)

    __matmul__ = lambda self, other: self.tensor(self.whisker(other))
    __rmatmul__ = lambda self, other: self.whisker(other).tensor(self)


class DictOrCallable:
    def __class_getitem__(_, source, target):
        return dict[source, target] | Callable[[source], target]


@dataclass
class FakeDict:
    inside: Callable
    __getitem__ = lambda self, key: self.inside(key)
