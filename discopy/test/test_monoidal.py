import itertools
from math import sqrt
from numpy import isclose

from discopy.cat import *
from discopy.monoidal import *
from discopy.tensor import *
from discopy.drawing import *


class Qubits(Ty):
    __str__ = lambda self: "qubit ** {}".format(len(self))

qubit = Qubits('1')

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


class Eval(Functor):
    dom = Category(Qubits, Circuit)
    cod = Category(tuple[int, ...], Tensor[complex])

    def __call__(self, other):
        if isinstance(other, Ket):
            if not other.bits: return Tensor.id(())
            head, *tail = other.bits
            return Tensor[complex]([[not head, head]], (), (2, ))\
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


def test_circuit():
    assert isinstance(qubit ** 0, Qubits) and isinstance(qubit ** 42, Qubits)

    assert isinstance(sqrt2 @ Ket(0, 0) >> H @ qubit >> CX, Circuit)

    circuit = sqrt2 @ Ket(0, 0) >> H @ qubit >> CX
    superposition = Ket(0, 0) + Ket(1, 1)
    assert circuit.eval().is_close(Circuit.eval(superposition))


def test_encoding():
    x, y, z = map(Ty, "xyz")
    f, g, h = Box('f', x, y), Box('g', y, z), Box('h', y @ z, x)
    encoding = Encoding(dom=x @ y, boxes_and_offsets=((f, 0), (g, 1), (h, 0)))
    assert Diagram.decode(encoding) == f @ g >> h\
        and (f @ g >> h).encode() == encoding


def test_simplify():
    def simplify(circuit, rules):
        for source, target in rules:
            for match in circuit.match(source):
                return simplify(match.subs(target), rules)
        return circuit

    rules = [(Ket(b) >> X, Ket(int(not b)))
             for b in [0, 1]] + [
             (Ket(b0) @ Ket(b1) >> CX, Ket(b0) @ Ket(int(not b1 if b0 else b1)))
             for b0 in [0, 1] for b1 in [0, 1]]
    circuit = Ket(1) @ Ket(0) >> CX >> qubit @ X

    assert simplify(circuit, rules) == Ket(1) >> qubit @ Ket(0)


def test_Born_rule():
    Born_rule = lambda x: abs(x) ** 2
    Circuit.measure = lambda self: self.bubble(method="squared_amplitude")
    Tensor.squared_amplitude = lambda self: self.map(Born_rule)

    assert isclose(Circuit.eval((Ket(0) >> H >> Bra(0)).measure()).inside[0][0], .5)

    biased_ReLU = lambda x: max(0, 2 * x.real - 1)
    Circuit.post_process = lambda self: self.bubble(method="non_linearity")
    Tensor.non_linearity = lambda self: self.map(biased_ReLU)

    circuit = Ket(0, 0) >> H @ qubit >> CX >> Bra(0, 0)
    post_processed_circuit = Circuit.post_process(circuit.measure())
    assert Circuit.eval(post_processed_circuit).inside\
        == [[biased_ReLU(Born_rule(complex(circuit.eval())))]]


def test_Peirce():
    class Formula(Diagram):
        cut = lambda self: Cut(self)

    class Cut(Bubble, Formula):
        method = "_not"
        cast = Formula.cast

    class Predicate(Box, Formula):
        cast = Formula.cast

    def model(size: dict[Ty, int], data: dict[Predicate, list[bool]]):
        return Functor(
            ob=size, ar={p: [data[p]] for p in data},
            dom=Category(Ty, Formula), cod=Category(tuple[int, ...], Tensor[bool]))

    x = Ty('x')
    dog, god, mortal = [
        Predicate(name, Ty(), x) for name in ("dog", "god", "mortal")]
    all_dogs_are_mortal = (dog.cut() >> mortal.dagger()).cut()
    gods_are_not_mortal = (god >> mortal.dagger()).cut()
    there_is_no_god_but_god = god >> (Formula.id(x).cut() >> god.dagger()).cut()

    size = {x: 2}

    for dogs, gods, mortals in itertools.product(*3 * [
            itertools.product(*size[x] * [[0, 1]])]):
        F = model(size, {dog: dogs, god: gods, mortal: mortals})
        assert F(all_dogs_are_mortal) == all(
            not F(dog)[i] or F(mortal)[i] for i in range(size[x]))
        assert F(gods_are_not_mortal) == all(
            not F(god)[i] or not F(mortal)[i] for i in range(size[x]))
        assert F(there_is_no_god_but_god) == any(F(god)[i] and not any(
            F(god)[j] and j != i for j in range(size[x])) for i in range(size[x]))


def test_drawing():
    x = Ty('x')
    f, g = Box('f', Ty(), x @ x), Box('g', x @ x, Ty())
    u, v = Box('u', Ty(), x), Box('v', x, Ty())

    def spiral(length: int) -> Diagram:
        diagram, n = u, length // 2 - 1
        for i in range(n):
            diagram >>= x ** i @ f @ x ** (i + 1)
        diagram >>= x ** n @ v @ x ** n
        for i in range(n):
            diagram >>= x ** (n - i - 1) @ g @ x ** (n - i - 1)
        return diagram

    diagram = spiral(6)

    assert graph2diagram(*draw(diagram)) == diagram
