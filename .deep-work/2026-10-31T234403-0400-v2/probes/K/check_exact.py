"""Finite exact reconstruction check. Stdlib; no network or hidden data."""
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path

BITS = list(product((0, 1), repeat=4))

def emit(record):
    print(json.dumps(record, sort_keys=True), flush=True)

def encode(tensor):
    return {''.join(map(str, s)): [tensor[s].numerator, tensor[s].denominator]
            for s in BITS}

def transform(tensor, matrix):
    out = {}
    for x in BITS:
        total = F(0)
        for y in BITS:
            term = tensor[y]
            for j in range(4):
                term *= matrix[x[j]][y[j]]
            total += term
        out[x] = total
    return out

def residual(t):
    return (t[(0,0,0,0)] * t[(1,1,1,1)]
            - t[(0,0,1,1)] * t[(1,1,0,0)]
            + t[(0,1,0,1)] * t[(1,0,1,0)]
            - t[(0,1,1,0)] * t[(1,0,0,1)])

def scalar(a, b):
    # The second vertex has already been relabelled by reversed planar port order.
    return sum((a[s] * b[s] for s in BITS), F(0))

def rank_balanced(t):
    pairs = list(product((0,1), repeat=2))
    m = [[t[a+b] for b in pairs] for a in pairs]
    rank = 0
    for col in range(4):
        pivot = next((j for j in range(rank, 4) if m[j][col]), None)
        if pivot is None:
            continue
        m[rank], m[pivot] = m[pivot], m[rank]
        scale = m[rank][col]
        m[rank] = [v/scale for v in m[rank]]
        for j in range(4):
            if j != rank:
                scale = m[j][col]
                m[j] = [a-scale*b for a,b in zip(m[j], m[rank])]
        rank += 1
    return rank

a = {s: F(0) for s in BITS}
entries = {'0000':1,'1100':1,'1010':2,'1001':3,
           '0110':4,'0101':5,'0011':6,'1111':8}
for bits, value in entries.items():
    a[tuple(map(int, bits))] = F(value)
serialized = json.dumps(encode(a), sort_keys=True)
decoded = {tuple(map(int, bits)): F(*fraction)
           for bits, fraction in json.loads(serialized).items()}
assert decoded == a
emit({'stage':'serialization_smoke', 'passed':True})

q = ((F(3,5), F(-4,5)), (F(4,5), F(3,5)))
qt = tuple(tuple(q[j][i] for j in range(2)) for i in range(2))
for i,j in product(range(2), repeat=2):
    assert sum(q[k][i]*q[k][j] for k in range(2)) == int(i == j)
aq = transform(a, q)
assert transform(aq, qt) == a
assert residual(a) == 0
assert scalar(a,a) == 156
assert scalar(aq,aq) == 156
assert rank_balanced(a) == rank_balanced(aq)
emit({'stage':'core', 'residual':str(residual(a)),
      'scalar_direct':str(scalar(a,a)), 'scalar_gauge':str(scalar(aq,aq)),
      'unfolding_rank':rank_balanced(a),
      'unfolding_rank_after_gauge':rank_balanced(aq),
      'nonzero_entries_before':sum(bool(v) for v in a.values()),
      'nonzero_entries_after':sum(bool(v) for v in aq.values()),
      'tensor_after_gauge':encode(aq)})

defect = dict(a)
defect[(1,1,1,1)] += 1
dq = transform(defect, q)
assert residual(defect) == 1
assert scalar(defect, a) == 164
assert scalar(dq, aq) == 164
assert transform(dq, qt) == defect
false_compiled = scalar(a, a)
assert false_compiled != scalar(defect, a)
result = {'stage':'defect_and_false_compression',
          'residual':str(residual(defect)),
          'scalar_direct':str(scalar(defect,a)),
          'scalar_gauge':str(scalar(dq,aq)),
          'scalar_if_exception_dropped':str(false_compiled),
          'negative_control_detected':True,
          'all_checks_passed':True,
          'neural_learning_executed':False,
          'scientific_significance_tested':False}
emit(result)
Path(__file__).with_name('exact-result.json').write_text(
    json.dumps(result, sort_keys=True, indent=2) + '\n')
