from math import sqrt

from discopy.cat import *
from discopy.monoidal import *
from discopy.tensor import *


def test_circuit():
    class Qubits(Ty):
        __str__ = lambda self: "qubit ** {}".format(len(self))

    qubit = Qubits('1')

    assert isinstance(qubit ** 0, Qubits) and isinstance(qubit ** 42, Qubits)


    class Circuit(Diagram): pass

    class Gate(Box, Circuit): pass

    class Bra(Box, Circuit):
        def __init__(self, *bits: bool):
            name = "Bra({})".format(', '.join(map(str, bits)))
            self.bits, dom, cod = bits, qubit ** len(bits), qubit ** 0
            Box.__init__(self, name, dom, cod)

        def dagger(self) -> Circuit: return Ket(*self.bits)

    class Ket(Box, Circuit):
        def __init__(self, *bits: bool):
            name = "Ket({})".format(', '.join(map(str, bits)))
            self.bits, dom, cod = bits, qubit ** 0, qubit ** len(bits)
            Box.__init__(self, name, dom, cod)

        def dagger(self) -> Circuit: return Bra(*self.bits)

    Gate.cast = Ket.cast = Circuit.cast

    X, Y, Z, H = [Gate(name, qubit, qubit) for name in "XYZH"]
    CX = Gate("CX", qubit ** 2, qubit ** 2)
    sqrt2 = Gate("$\\sqrt{2}$", qubit ** 0, qubit ** 0)
    assert isinstance(sqrt2 @ Ket(0, 0) >> H @ qubit >> CX, Circuit)


    class Eval(Functor):
        dom, cod = Category(Qubits, Circuit), Category(tuple[int, ...], Tensor[complex])

        def __call__(self, other):
            if isinstance(other, Ket):
                if not other.bits: return Tensor.id(())
                head, *tail = other.bits
                return Tensor[complex]([[head, not head]], (), (2, ))\
                    @ self(Ket(*tail))
            if isinstance(other, Bra):
                return self(other.dagger()).dagger()
            return super().__call__(other)

    Circuit.eval = lambda self: Eval(
        ob={qubit: 2},
        ar={X: [[0, 1], [1, 0]], Y: [[0, -1j], [1j, 0]], Z: [[1, 0], [0, -1]],
            H: [[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]],
            CX: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            sqrt2: [[sqrt(2)]]})(self)

    circuit = sqrt2 @ Ket(0, 0) >> H @ qubit >> CX
    superposition = Ket(0, 0) + Ket(1, 1)
    assert circuit.eval().is_close(superposition.eval())
