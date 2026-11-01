"""Frozen exact-arithmetic independent check; run only via run_isolated.py."""
from fractions import Fraction as F
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'raw.jsonl'
SEED = 742019
rng = random.Random(SEED)
raw = RAW.open('x')


def encode(value):
    if isinstance(value, F):
        return {'num': str(value.numerator), 'den': str(value.denominator)}
    if isinstance(value, list):
        return [encode(x) for x in value]
    if isinstance(value, tuple):
        return [encode(x) for x in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    return value


def decode(value):
    if isinstance(value, list):
        return [decode(x) for x in value]
    if isinstance(value, dict):
        if set(value) == {'num', 'den'}:
            return F(int(value['num']), int(value['den']))
        return {k: decode(v) for k, v in value.items()}
    return value


def emit(kind, **kwargs):
    raw.write(json.dumps(encode({'kind': kind, **kwargs}), sort_keys=True) + '\n')
    raw.flush()
    os.fsync(raw.fileno())


def save_roundtrip(name, value):
    destination = ROOT / name
    with destination.open('x') as file:
        json.dump(encode(value), file, sort_keys=True)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    with destination.open() as file:
        restored = decode(json.load(file))
    assert restored == value, 'serialization mismatch'
    return restored


# Serialization smoke is deliberately before all numerical tests.
emit('OPEN', seed=SEED, arithmetic='Fraction', test_plan='test-plan.md')
smoke = {'S': [[F(2), F(-2, 7)], [F(-2, 7), F(1, 3)]], 'step': 0}
save_roundtrip('serialization-smoke.json', smoke)
emit('SERIALIZATION_SMOKE_PASS', state=smoke)
print('SERIALIZATION_SMOKE_PASS', flush=True)


def mat(rows):
    return [[F(x) for x in row] for row in rows]


def transpose(a):
    return [list(row) for row in zip(*a)]


def multiply(a, b):
    assert len(a[0]) == len(b)
    columns = transpose(b)
    return [[sum((x * y for x, y in zip(row, col)), F(0))
             for col in columns] for row in a]


def add(a, b):
    assert len(a) == len(b) and len(a[0]) == len(b[0])
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scale(c, a):
    return [[c * x for x in row] for row in a]


def difference(a, b):
    return add(a, scale(F(-1), b))


def direct_step(a, b, g, eta):
    # The two expressions are evaluated from the old a,b before returning.
    return (difference(a, scale(eta, multiply(g, transpose(b)))),
            difference(b, scale(eta, multiply(transpose(a), g))))


def factor_gram(a, b):
    x = a + transpose(b)
    return multiply(x, transpose(x))


def h_matrix(m, n, g, eta):
    h = [[F(i == j) for j in range(m + n)] for i in range(m + n)]
    for i in range(m):
        for j in range(n):
            h[i][m + j] = -eta * g[i][j]
            h[m + j][i] = -eta * g[i][j]
    return h


def summary_step(s, g, eta):
    m, n = len(g), len(g[0])
    h = h_matrix(m, n, g, eta)
    return multiply(multiply(h, s), transpose(h))


def blocks(s, m):
    return ([row[:m] for row in s[:m]],
            [row[m:] for row in s[:m]],
            [row[m:] for row in s[m:]])


def response(s, m, g):
    p, _, q = blocks(s, m)
    return add(multiply(p, g), multiply(g, q))


def random_matrix(rows, cols, denominator):
    return [[F(rng.randint(-3, 3), denominator) for _ in range(cols)]
            for _ in range(rows)]


case_count = 0
step_count = 0
for m, n in [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)]:
    for hidden in [1, 2, 5, 13]:
        case = f'{m}x{n}_h{hidden}'
        a = random_matrix(m, hidden, 5)
        b = random_matrix(hidden, n, 5)
        s = factor_gram(a, b)
        emit('CASE_INPUT', case=case, A=a, B=b, S=s)
        for step in range(6):
            g = random_matrix(m, n, 7)
            eta = F(1, step + 3)
            x = random_matrix(n, 1, 3)
            a, b = direct_step(a, b, g, eta)
            s = summary_step(s, g, eta)
            actual = factor_gram(a, b)
            assert s == actual, (case, step, 'full Gram')
            w = blocks(s, m)[1]
            assert w == multiply(a, b), (case, step, 'product')
            assert multiply(w, x) == multiply(a, multiply(b, x)), (case, step, 'output')
            emit('STEP_PASS', case=case, step=step + 1, G=g, eta=eta,
                 x=x, A=a, B=b, S=s, output=multiply(w, x))
            step_count += 1
        save_roundtrip(f'state-{case}.json', {'A': a, 'B': b, 'S': s, 'step': 6})
        emit('CASE_PASS', case=case, steps=6, serialized=True)
        case_count += 1

print(f'GRAM_CLOSURE_PASS cases={case_count} steps={step_count}', flush=True)

# Same current function, differing immediate response.
a1, b1 = mat([[1]]), mat([[1]])
a2, b2 = mat([[2]]), mat([[F(1, 2)]])
assert multiply(a1, b1) == multiply(a2, b2) == mat([[1]])
g, eta = mat([[F(2, 3)]]), F(1, 4)
a1, b1 = direct_step(a1, b1, g, eta)
a2, b2 = direct_step(a2, b2, g, eta)
delta = difference(multiply(a2, b2), multiply(a1, b1))
assert delta == mat([[F(-3, 8)]])
emit('CURRENT_PRODUCT_COUNTEREXAMPLE_PASS', G=g, eta=eta,
     W1=multiply(a1, b1), W2=multiply(a2, b2), delta=delta)

# Same entire initial response, different later response: no random search.
a1 = mat([[1, 1, 0, 0, 0, 0]])
b1 = transpose(mat([[0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1]]))
a2 = mat([[1, 1, 1, 0, 0, 0]])
b2 = transpose(mat([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0]]))
s1, s2 = factor_gram(a1, b1), factor_gram(a2, b2)
assert blocks(s1, 1)[1] == blocks(s2, 1)[1] == mat([[0, 0]])
for g in [mat([[1, 0]]), mat([[0, 1]]), mat([[1, 1]]), mat([[F(-2, 3), F(5, 7)]])]:
    assert response(s1, 1, g) == response(s2, 1, g) == scale(F(4), g)
    emit('INITIAL_RESPONSE_PROBE_PASS', G=g, response=response(s1, 1, g))
emit('TWO_STEP_PAIR_INPUT', A1=a1, B1=b1, A2=a2, B2=b2, S1=s1, S2=s2)
for step, g in enumerate([mat([[1, 0]]), mat([[0, 1]])], start=1):
    a1, b1 = direct_step(a1, b1, g, F(1, 2))
    a2, b2 = direct_step(a2, b2, g, F(1, 2))
    w1, w2 = multiply(a1, b1), multiply(a2, b2)
    if step == 1:
        assert w1 == w2 == mat([[-2, 0]])
    else:
        assert w1 == mat([[-2, F(-9, 4)]])
        assert w2 == mat([[-2, F(-17, 8)]])
        assert difference(w2, w1) == mat([[0, F(1, 8)]])
    emit('TWO_STEP_PAIR_PASS', step=step, G=g, eta=F(1, 2),
         A1=a1, B1=b1, A2=a2, B2=b2, W1=w1, W2=w2)

# Scalar exception: p-q is behaviorally invisible.
a1, b1 = F(1), F(2)
a2, b2 = F(2), F(1)
w, r = F(2), F(5)
for step in range(8):
    g, eta = F(rng.randint(-3, 3), 7), F(1, step + 3)
    a1, b1 = a1 - eta * g * b1, b1 - eta * a1 * g
    a2, b2 = a2 - eta * g * b2, b2 - eta * a2 * g
    w, r = w - eta * g * r + eta**2 * g**2 * w, (1 + eta**2 * g**2) * r - 4 * eta * g * w
    assert a1 * b1 == a2 * b2 == w
    assert a1*a1 + b1*b1 == a2*a2 + b2*b2 == r
    emit('SCALAR_REDUCTION_PASS', step=step + 1, g=g, eta=eta,
         a1=a1, b1=b1, a2=a2, b2=b2, w=w, r=r)

# Two unilateral appends and a separate all-zero append.
a = mat([[1, 2], [0, 1]])
b = mat([[1, 0, -1], [2, 1, 0]])
u, v = mat([[1], [-1]]), mat([[2, 0, 1]])
aa = [row + [u[i][0], F(0)] for i, row in enumerate(a)]
bb = b + mat([[0, 0, 0]]) + v
az = [row + [F(0)] for row in a]
bz = b + mat([[0, 0, 0]])
s, sa = factor_gram(a, b), factor_gram(aa, bb)
increment = mat([[0] * 5 for _ in range(5)])
du, dv = multiply(u, transpose(u)), multiply(transpose(v), v)
for i in range(2):
    for j in range(2):
        increment[i][j] = du[i][j]
for i in range(3):
    for j in range(3):
        increment[2+i][2+j] = dv[i][j]
assert difference(sa, s) == increment
emit('APPEND_INPUT', A=a, B=b, U=u, V=v, S=s, S_append=sa, increment=increment)
for step in range(4):
    g, eta = random_matrix(2, 3, 7), F(1, step + 3)
    expected_delta = scale(-eta, add(multiply(du, g), multiply(g, dv)))
    a, b = direct_step(a, b, g, eta)
    aa, bb = direct_step(aa, bb, g, eta)
    az, bz = direct_step(az, bz, g, eta)
    sa = summary_step(sa, g, eta)
    assert sa == factor_gram(aa, bb)
    assert multiply(az, bz) == multiply(a, b)
    assert all(row[-1] == 0 for row in az) and all(x == 0 for x in bz[-1])
    if step == 0:
        assert difference(multiply(aa, bb), multiply(a, b)) == expected_delta
    emit('APPEND_STEP_PASS', step=step + 1, G=g, eta=eta,
         A=a, B=b, A_append=aa, B_append=bb, S_append=sa,
         zero_path_A=[row[-1] for row in az], zero_path_B=bz[-1])

summary = {'status': 'PASS', 'seed': SEED, 'arithmetic': 'exact Fraction',
           'closure_cases': case_count, 'closure_steps': step_count,
           'current_product_counterexample': 'PASS',
           'initial_response_two_step_counterexample': 'PASS',
           'scalar_exception_steps': 8, 'append_steps': 4,
           'serialization': 'PASS', 'author_artifacts_opened': False}
save_roundtrip('summary.json', summary)
emit('FINAL_PASS', **summary)
raw.close()
print(json.dumps(summary, sort_keys=True), flush=True)
