import itertools
from discopy.cat import Category
from discopy.monoidal import Bubble
from discopy.hypergraph import *


class Formula(Diagram):
    cut = lambda self: Cut(self)

class Cut(Bubble, Formula):
    method = "_not"
    cast = Formula.cast

class Predicate(Box, Formula):
    cast = Formula.cast

def model(size: dict[Ty, int], data: dict[Predicate, list[bool]]):
    return Functor(ob=size, ar={p: [data[p]] for p in data},
                   dom=Category(Ty, Formula),
                   cod=Category(list[int], Tensor[bool]))

def test_formula():
    objects, categories = Ty('o'), Ty('c')
    has_object, has_unit = [Predicate(p, Ty(), categories @ objects) for p in "OU"]
    is_rigid, is_cartesian = [Predicate(p, Ty(), categories) for p in "RC"]

    rigid_cartesian_implies_trivial = (
        has_object >> Formula.spiders(1, 3, categories) @ objects
        >> (is_rigid @ is_cartesian @ has_unit.cut()).dagger()).cut()

    size = {objects: 2, categories: 2}
    predicate_values = itertools.product(*size[categories] * [[0, 1]])
    relation_values = itertools.product(*size[categories] * size[objects] * [[0, 1]])

    for O, U, R, C in itertools.product(
            *(2 * [predicate_values] + 2 * [relation_values])):
        F = model(size, {has_object: O, has_unit: U, is_rigid: R, is_cartesian: C})
        is_rigid_cartesian_and_has_object = lambda i, j:\
            F(has_object)[i, j] and F(is_rigid)[i] and F(is_cartesian)[i]
        assert F(rigid_cartesian_implies_trivial) == all(
            not is_rigid_cartesian_and_has_object(i, j) or F(has_unit)[i, j]
            for i in range(size[categories]) for j in range(size[objects]))
