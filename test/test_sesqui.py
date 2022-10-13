from discopy.sesqui import *
from discopy.python import Function, Functor


a = Colour('a')
x = Ty('x', dom=a, cod=a)
f, g = Box('f', Ty.id(a), x), Box('g', x @ x, x)

def test_sesqui():
    Pyth = Category(tuple[type, ...], Function)
    List = Functor(
        ob=lambda xs: list[xs],
        ar=lambda f: lambda xs: list(map(f, xs)),
        dom=Pyth, cod=Pyth)
    Unit = Transformation(
        lambda _: lambda x: [x], dom=Functor.id(Pyth), cod=List)
    Mult = Transformation(
        lambda _: lambda xs: sum(xs, []), dom=List >> List, cod=List)

    F = TwoFunctor(
        colours={a: Pyth}, ob={x: List}, ar={f: Unit, g: Mult},
        cod=TwoCategory(Category, Functor, Transformation))

    assert F(f @ x >> g)(int)([1, 2, 3])\
        == F(x @ f >> g)(int)([1, 2, 3])\
        == F(Diagram.id(x))(int)([1, 2, 3]) == [1, 2, 3]

    assert F(g @ x >> g)(int)([[[42]]])\
        == F(x @ g >> g)(int)([[[42]]]) == [42]

def test_rules():
    left = Rule("left-unit", f @ x >> g, Diagram.id(x))
    right = Rule("right-unit", Diagram.id(x), x @ f >> g)
    rewrite = left >> right
    assert (rewrite.dom, rewrite.cod) == (f @ x >> g, x @ f >> g)
