"""Interpret actual emitted closed_updates against independent factor loops."""
from fractions import Fraction as F
import copy
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUTHOR = ROOT.parent / 'K'
raw = (ROOT / 'generated-replay-raw.jsonl').open('x')


def encode(x):
    if isinstance(x, F):
        return {'num': str(x.numerator), 'den': str(x.denominator)}
    if isinstance(x, list):
        return [encode(v) for v in x]
    if isinstance(x, dict):
        return {k: encode(v) for k, v in x.items()}
    return x


def decode(x):
    if isinstance(x, list):
        return [decode(v) for v in x]
    if isinstance(x, dict):
        if set(x) == {'num', 'den'}:
            return F(int(x['num']), int(x['den']))
        return {k: decode(v) for k, v in x.items()}
    return x


def emit(kind, **fields):
    raw.write(json.dumps(encode({'kind': kind, **fields}), sort_keys=True) + '\n')
    raw.flush()
    os.fsync(raw.fileno())


def roundtrip(name, state):
    path = ROOT / name
    with path.open('x') as file:
        json.dump(encode(state), file, sort_keys=True)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    with path.open() as file:
        assert decode(json.load(file)) == state


emit('OPEN', arithmetic='Fraction', programs='actual author closed_updates')
smoke = {'q': [F(-13, 17), F(2, 19)], 'z': [F(7, 3)]}
roundtrip('generated-replay-smoke.json', smoke)
emit('SERIALIZATION_SMOKE_PASS', state=smoke)
print('SERIALIZATION_SMOKE_PASS', flush=True)


def rational(pair):
    assert isinstance(pair, list) and len(pair) == 2
    assert all(isinstance(x, int) for x in pair) and pair[1] != 0
    return F(pair[0], pair[1])


def monomial(values, powers):
    assert len(values) == len(powers)
    assert all(isinstance(p, int) and p >= 0 for p in powers)
    value = F(1)
    for x, power in zip(values, powers):
        value *= x**power
    return value


def load_program(tag):
    path = AUTHOR / (tag + '-closure.json')
    payload = path.read_bytes()
    program = json.loads(payload)
    events = program['events']
    assert [event['index'] for event in events] == list(range(len(events)))
    assert len(events) == len(program['closed_updates']) == program['dimension']
    names = program['source_variable_names']
    assert len(set(names)) == len(names)
    for event in events:
        for term in event['statistic']:
            assert len(term['powers']) == len(names)
            rational(term['coefficient'])
    counts = set()
    for update in program['closed_updates']:
        for term in update:
            counts.add(len(term['control_powers']))
            assert len(term['state_coefficients']) == len(events)
            for coefficient in term['state_coefficients']:
                rational(coefficient)
    assert len(counts) == 1
    emit('PROGRAM_LOADED', tag=tag, sha256=hashlib.sha256(payload).hexdigest(),
         state_count=len(events), source_names=names, control_count=next(iter(counts)))
    return program


def observe(program, source):
    return [sum((rational(term['coefficient']) * monomial(source, term['powers'])
                 for term in event['statistic']), F(0))
            for event in program['events']]


def execute(program, old_state, controls):
    result = []
    for update in program['closed_updates']:
        value = F(0)
        for term in update:
            coefficient = sum((rational(c) * old_state[k]
                               for k, c in enumerate(term['state_coefficients'])), F(0))
            value += monomial(controls, term['control_powers']) * coefficient
        result.append(value)
    return result


def source_vector(program, values):
    assert set(values) == set(program['source_variable_names'])
    return [values[name] for name in program['source_variable_names']]


def output_readout(program, state, expected):
    mapping = {event['origin']['output']: event['index']
               for event in program['events'] if 'output' in event['origin']}
    assert set(mapping) == set(range(len(expected)))
    readout = [state[mapping[index]] for index in range(len(expected))]
    assert readout == expected
    return readout


def mutation_detected(program, state, controls, expected):
    mutant = copy.deepcopy(program)
    changed = False
    for term in mutant['closed_updates'][0]:
        for index, pair in enumerate(term['state_coefficients']):
            value = rational(pair)
            if value:
                value += 1
                term['state_coefficients'][index] = [value.numerator, value.denominator]
                changed = True
                break
        if changed:
            break
    assert changed
    mutant_result = execute(mutant, state, controls)
    assert mutant_result != expected, 'deliberately mutated update escaped detection'
    return mutant_result


scalar = load_program('scalar')
a = [F(1, 2), F(-2, 3), F(3, 5)]
b = [F(-5, 7), F(7, 11), F(2, 9)]
commands = [F(1, 7), F(-2, 9), F(0), F(1, 5), F(-1, 11), F(2, 13)]


def scalar_values(a, b):
    return {**{f'a{i}': value for i, value in enumerate(a)},
            **{f'b{i}': value for i, value in enumerate(b)}}


q = source_vector(scalar, scalar_values(a, b))
state = observe(scalar, q)
output_readout(scalar, state, [sum((x*y for x, y in zip(a, b)), F(0))])
roundtrip('generated-replay-scalar-initial.json', {'q': q, 'state': state})
emit('SOURCE_INPUT', tag='scalar', a=a, b=b, commands=[[x] for x in commands], state=state)
for step, control in enumerate(commands, start=1):
    old_state = state
    new_a = [x - control * y for x, y in zip(a, b)]
    new_b = [y - control * x for x, y in zip(a, b)]
    a, b = new_a, new_b
    q = source_vector(scalar, scalar_values(a, b))
    expected = observe(scalar, q)
    state = execute(scalar, old_state, [control])
    assert state == expected, ('scalar', step, 'generated state mismatch')
    readout = output_readout(scalar, state, [sum((x*y for x, y in zip(a, b)), F(0))])
    if step == 1:
        mutated = mutation_detected(scalar, old_state, [control], expected)
        emit('MUTATION_DETECTED', tag='scalar', correct=expected, mutated=mutated)
    emit('GENERATED_STEP_PASS', tag='scalar', step=step, controls=[control],
         a=a, b=b, generated_state=state, source_statistic_state=expected, output=readout)
roundtrip('generated-replay-scalar-final.json', {'q': q, 'state': state})

matrix = load_program('matrix')
v = [[F(1, 2), F(-2, 3)], [F(3, 5), F(4, 7)]]
u = [[F(5, 11), F(-1, 2)], [F(2, 3), F(7, 13)]]
matrix_commands = [
    [F(1, 7), F(0), F(-2, 9), F(1, 5)],
    [F(0), F(1, 6), F(1, 8), F(0)],
    [F(-1, 5), F(2, 7), F(0), F(-1, 9)],
    [F(1, 11), F(-1, 13), F(1, 17), F(1, 19)],
    [F(0), F(0), F(0), F(0)],
    [F(2, 9), F(-1, 4), F(-2, 11), F(1, 3)],
]


def matrix_values(v, u):
    return {**{f'v{i}{k}': value for i, row in enumerate(v) for k, value in enumerate(row)},
            **{f'u{k}{j}': value for k, row in enumerate(u) for j, value in enumerate(row)}}


def direct_output(v, u):
    return [sum((v[i][k] * u[k][j] for k in range(len(u))), F(0))
            for i in range(len(v)) for j in range(len(u[0]))]


q = source_vector(matrix, matrix_values(v, u))
state = observe(matrix, q)
output_readout(matrix, state, direct_output(v, u))
roundtrip('generated-replay-matrix-initial.json', {'q': q, 'state': state})
emit('SOURCE_INPUT', tag='matrix', V=v, U=u, commands=matrix_commands, state=state)
for step, controls in enumerate(matrix_commands, start=1):
    old_state = state
    m, hidden, n = len(v), len(u), len(u[0])
    assert len(controls) == m*n
    # Independent loops implement the old-factor source micro-law directly.
    vn = [[v[i][k] - sum((controls[i*n+j] * u[k][j] for j in range(n)), F(0))
           for k in range(hidden)] for i in range(m)]
    un = [[u[k][j] - sum((v[i][k] * controls[i*n+j] for i in range(m)), F(0))
           for j in range(n)] for k in range(hidden)]
    v, u = vn, un
    q = source_vector(matrix, matrix_values(v, u))
    expected = observe(matrix, q)
    state = execute(matrix, old_state, controls)
    assert state == expected, ('matrix', step, 'generated state mismatch')
    readout = output_readout(matrix, state, direct_output(v, u))
    if step == 1:
        mutated = mutation_detected(matrix, old_state, controls, expected)
        emit('MUTATION_DETECTED', tag='matrix', correct=expected, mutated=mutated)
    emit('GENERATED_STEP_PASS', tag='matrix', step=step, controls=controls,
         V=v, U=u, generated_state=state, source_statistic_state=expected, output=readout)
roundtrip('generated-replay-matrix-final.json', {'q': q, 'state': state})

summary = {'status': 'PASS', 'actual_generated_closed_updates_executed': True,
           'state_selected_from': 'all emitted events; no target dimensions',
           'scalar_steps': len(commands), 'matrix_steps': len(matrix_commands),
           'scalar_state_count_observed': len(scalar['events']),
           'matrix_state_count_observed': len(matrix['events']),
           'mutation_controls_detected': 2, 'serialization': 'PASS',
           'arithmetic': 'exact Fraction', 'author_python_imported': False}
roundtrip('generated-replay-summary.json', summary)
emit('FINAL_PASS', **summary)
raw.close()
print(json.dumps(summary, sort_keys=True), flush=True)
