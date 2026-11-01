"""Exact stdlib matrix algebra/edit closure; inspected, no network or secrets."""
from fractions import Fraction


N = 3
ZERO = (Fraction(0),) * (N * N)
IDENTITY = tuple(Fraction(int(i == j)) for i in range(N) for j in range(N))


def matrix(values):
    return tuple(Fraction(x) for row in values for x in row)


def unit(i, j):
    return tuple(Fraction(int(r == i and c == j))
                 for r in range(N) for c in range(N))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def scale(a, c):
    return tuple(x * c for x in a)


def mul(a, b):
    return tuple(sum((a[i * N + k] * b[k * N + j]
                      for k in range(N)), Fraction(0))
                 for i in range(N) for j in range(N))


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


class Span:
    """Incremental exact echelon basis; rows keep original-coordinate witnesses."""

    def __init__(self):
        self.basis = []
        self.rows = []
        self.pivots = []

    def coordinates(self, value):
        residual = list(value)
        coefficients = [Fraction(0)] * len(self.basis)
        for pivot, row, witness in zip(self.pivots, self.rows, self.witnesses):
            c = residual[pivot]
            if c:
                residual = [x - c * y for x, y in zip(residual, row)]
                coefficients = [x + c * y for x, y in zip(coefficients, witness)]
        if any(residual):
            return None
        return tuple(coefficients)

    @property
    def witnesses(self):
        return [r[1] for r in self._rows_with_witness]

    def append(self, value):
        if not hasattr(self, "_rows_with_witness"):
            self._rows_with_witness = []
        residual = list(value)
        old_count = len(self.basis)
        coefficients = [Fraction(0)] * old_count
        for pivot, row, witness in zip(self.pivots, self.rows, self.witnesses):
            c = residual[pivot]
            if c:
                residual = [x - c * y for x, y in zip(residual, row)]
                coefficients = [x + c * y for x, y in zip(coefficients, witness)]
        if not any(residual):
            return False
        pivot = next(i for i, x in enumerate(residual) if x)
        denominator = residual[pivot]
        row = tuple(x / denominator for x in residual)
        self._rows_with_witness = [(r, w + (Fraction(0),))
                                  for r, w in self._rows_with_witness]
        witness = tuple(-x / denominator for x in coefficients) + (1 / denominator,)
        self.basis.append(value)
        self.pivots.append(pivot)
        self.rows.append(row)
        self._rows_with_witness.append((row, witness))
        return True

    def reconstruct(self, coordinates):
        result = ZERO
        for c, basis in zip(coordinates, self.basis):
            result = add(result, scale(basis, c))
        return result


def algebra(generators):
    space = Span()
    space.append(IDENTITY)
    for value in generators:
        space.append(value)
    index = 0
    products = 0
    while index < len(space.basis):
        current = space.basis[index]
        for generator in generators:
            products += 2
            space.append(mul(generator, current))
            space.append(mul(current, generator))
        index += 1
    assert len(space.basis) <= N * N
    return space, products


def edit_closure(generators, edits):
    space = Span()
    for edit in edits:
        space.append(edit)
    index = 0
    products = 0
    while index < len(space.basis):
        current = space.basis[index]
        for generator in generators:
            products += 2
            space.append(mul(generator, current))
            space.append(mul(current, generator))
        index += 1
    assert len(space.basis) <= N * N
    return space, products


def compile_tables(a_space, m_space):
    c, left, right = {}, {}, {}
    for i, a in enumerate(a_space.basis):
        for j, b in enumerate(a_space.basis):
            c[i, j] = a_space.coordinates(mul(a, b))
            assert c[i, j] is not None
        for j, m in enumerate(m_space.basis):
            left[i, j] = m_space.coordinates(mul(a, m))
            right[j, i] = m_space.coordinates(mul(m, a))
            assert left[i, j] is not None and right[j, i] is not None
    return c, left, right


def contract(x, y, table, output_dimension):
    output = [Fraction(0)] * output_dimension
    for i, a in enumerate(x):
        if not a:
            continue
        for j, b in enumerate(y):
            if b:
                output = [z + a * b * c for z, c in zip(output, table[i, j])]
    return tuple(output)


def pair_product(pair_a, pair_b, tables, a_dimension, m_dimension):
    a, m = pair_a
    b, n = pair_b
    c, left, right = tables
    computation = contract(a, b, c, a_dimension)
    forward_edit = contract(a, n, left, m_dimension)
    backward_edit = contract(m, b, right, m_dimension)
    return computation, tuple(x + y for x, y in zip(forward_edit, backward_edit))
