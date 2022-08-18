from discopy.rigid import *
from discopy.tensor import *
from test.test_monoidal import Circuit, Qubits, qubit, sqrt2, H, Ket


def test_Ty():
    x, y = Ty('x'), Ty('y')
    assert Ty().l == Ty() == Ty().r
    assert (x @ y).l == y.l @ x.l and (x @ y).r == y.r @ x.r
    assert x.r.l == x == x.l.r


def test_tensor():
    x, y = map(Ty, "xy")
    t = x @ y
    f = Box('f', x, y)

    left_snake = Diagram.id(t.l).transpose(left=False)
    right_snake = Diagram.id(t).transpose(left=True)

    F = Functor(
        ob={x: 2, y: 3}, ar={f: [[1, 2, 3], [4, 5, 6]]},
        cod=Category(tuple[int, ...], Tensor[int]))

    assert F(left_snake) == F(Diagram.id(t))
    assert F(right_snake) == F(Diagram.id(t.l))
    assert F(f.transpose()) == F(f).transpose() == F(f.transpose(left=False))

    # Diagrammatic and algebraic transpose differ for tensors of order >= 2.
    assert F(f @ x).transpose() != F((f @ x).transpose())
