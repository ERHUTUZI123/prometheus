"""Exhaustive six-state reconstruction. Standard library only; no external I/O."""

import itertools
import json
from pathlib import Path
import random


def emit(record):
    line = json.dumps(record, sort_keys=True)
    with Path('results.jsonl').open('a') as stream:
        stream.write(line + '\n')
        stream.flush()
    print(line, flush=True)


def fit_chart(chart, tables, observed):
    """chart[s] = 3*c+a; f is inferred, with unobserved a left unspecified."""
    functions = {}
    for name, table in tables.items():
        function = {}
        for source in observed[name]:
            a, c = chart[source] % 3, chart[source] // 3
            target_a, target_c = chart[table[source]] % 3, chart[table[source]] // 3
            if c != target_c:
                return None
            if a in function and function[a] != target_a:
                return None
            function[a] = target_a
        functions[name] = function
    return functions


def find_charts(tables, observed):
    fits = []
    for chart in itertools.permutations(range(6)):
        functions = fit_chart(chart, tables, observed)
        if functions is not None:
            fits.append((chart, functions))
    return fits


def words_upto(limit):
    for length in range(limit + 1):
        yield from itertools.product(('inc', 'reset'), repeat=length)


def main():
    assert not Path('results.jsonl').exists(), 'Results must be fresh; preserve old execution.'
    labels = list(range(6))
    random.Random(83).shuffle(labels)
    encode = {(a, c): labels[3*c+a] for c in range(2) for a in range(3)}
    decode = {value: key for key, value in encode.items()}
    inc, reset, bad = {}, {}, {}
    for source, (a, c) in decode.items():
        inc[source] = encode[((a+1) % 3, c)]
        reset[source] = encode[(0, c)]
        bad_a = (a+1) % 3 if c == 0 else {0: 1, 1: 0, 2: 2}[a]
        bad[source] = encode[(bad_a, c)]
    all_states = tuple(range(6))
    full_inc = find_charts({'inc': inc}, {'inc': all_states})
    emit({'case': 'full_inc', 'chart_count': len(full_inc), 'predicted_count': 36,
          'state_encoding': labels, 'transition_table': inc})
    assert len(full_inc) == 36
    full_pair = find_charts({'inc': inc, 'reset': reset},
                            {'inc': all_states, 'reset': all_states})
    emit({'case': 'full_inc_reset', 'chart_count': len(full_pair), 'predicted_count': 12})
    assert len(full_pair) == 12
    full_bad = find_charts({'bad': bad}, {'bad': all_states})
    emit({'case': 'nonproduct_full', 'chart_count': len(full_bad), 'predicted_count': 0,
          'transition_table': bad})
    assert len(full_bad) == 0
    partial_states = (encode[(0, 0)], encode[(0, 1)])
    partial_bad = find_charts({'bad': bad}, {'bad': partial_states})
    assert partial_bad
    witness_chart, witness_functions = partial_bad[0]
    extended = dict(witness_functions['bad'])
    for active in range(3):
        extended.setdefault(active, active)
    inverse = {value: source for source, value in enumerate(witness_chart)}
    counterexamples = []
    for source in all_states:
        a, c = witness_chart[source] % 3, witness_chart[source] // 3
        predicted_target = inverse[3*c+extended[a]]
        if predicted_target != bad[source]:
            counterexamples.append({'source': source, 'predicted': predicted_target,
                                    'actual': bad[source], 'seen_source': source in partial_states})
    emit({'case': 'nonproduct_partial', 'chart_count': len(partial_bad),
          'observed_sources': partial_states, 'witness_chart': witness_chart,
          'witness_inferred_function': witness_functions,
          'witness_completion': extended, 'counterexamples': counterexamples})
    assert counterexamples
    assert all(not example['seen_source'] for example in counterexamples)
    chart, functions = full_pair[0]
    inverse = {value: source for source, value in enumerate(chart)}
    mismatch_count = 0
    comparison_count = 0
    for word in words_upto(6):
        for start in all_states:
            physical = start
            a, c = chart[start] % 3, chart[start] // 3
            for name in word:
                physical = {'inc': inc, 'reset': reset}[name][physical]
                a = functions[name][a]
            reconstructed = inverse[3*c+a]
            comparison_count += 1
            mismatch_count += reconstructed != physical
    emit({'case': 'composition', 'max_word_length': 6, 'comparison_count': comparison_count,
          'mismatch_count': mismatch_count, 'chart': chart, 'functions': functions})
    assert comparison_count == 6 * (2**7-1)
    assert mismatch_count == 0
    records = [json.loads(line) for line in Path('results.jsonl').read_text().splitlines()]
    assert len(records) == 5
    emit({'case': 'completion', 'status': 'passed_in_stated_finite_scope',
          'neural_training_executed': False, 'learned_factor_sizes': False,
          'proves_utility_or_novelty': False, 'serialization_records_before_completion': len(records)})


if __name__ == '__main__':
    main()
