#!/usr/bin/env python3
"""Inspected offline tree-relation competitor. No imports of author solvers."""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parents[1]

def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def interpret(rows, offsets, clamps):
    """Use only stored equation rows, offsets, and current query clamps.

    Unresolved unit-coefficient ports are solved by single-equation algebra.
    No target phase, desired answer, optimizer, eigensolver, or force trace.
    Non-triangular/unidentifiable cases fail explicitly.
    """
    n = len(rows[0])
    known = dict(clamps)
    solved = []
    scans = 0
    arithmetic = 0
    while len(known) < n:
        progress = False
        for index, (row, offset) in enumerate(zip(rows, offsets)):
            scans += 1
            missing = [i for i, c in enumerate(row) if c and i not in known]
            if len(missing) != 1:
                continue
            j = missing[0]
            if abs(row[j]) != 1:
                raise ValueError('nonunit coefficient is outside frozen interpreter')
            rhs = float(offset)
            for i, c in enumerate(row):
                if c and i != j:
                    if c == 1:
                        rhs -= known[i]
                    elif c == -1:
                        rhs += known[i]
                    else:
                        raise ValueError('nonunit known coefficient outside contract')
                    arithmetic += 1
            known[j] = float(wrap(rhs if row[j] == 1 else -rhs))
            solved.append((index, j))
            progress = True
        if not progress:
            raise ValueError('No one-unknown relation; direct tree inference unavailable')
    return np.array([known[i] for i in range(n)]), dict(
        solved_ports=len(solved), row_scans=scans,
        additions_subtractions=arithmetic, modulo_calls=len(solved),
        sign_flips=sum(rows[i][j] == -1 for i, j in solved))

def main():
    b = json.loads((RUN/'probes/B/results.json').read_text())
    bf = json.loads((RUN/'probes/BF/results.json').read_text())
    bf_queries = json.loads((RUN/'probes/BF/query-records.json').read_text())
    D = np.zeros((4, 5), dtype=int)
    for k in range(4):
        D[k, k] = -1
        D[k, k+1] = 1
    zeros, counts = interpret(D.tolist(), [0.]*4, {0: 0.})
    bf_e = wrap(D @ zeros)
    n_bf = len(bf_queries['tree']['starts'])
    assert n_bf == 64
    assert np.max(np.abs(bf_e)) == 0 and np.max(np.abs(zeros)) == 0
    # All initial free phases are irrelevant to the target of this exact task.
    # We do not report analog settling time for digital assignment.
    bf_check = dict(query_count=n_bf, correct_queries=n_bf,
        baseline_max_residual=float(np.max(np.abs(bf_e))),
        baseline_output_error=float(abs(zeros[-1])),
        operations_per_query=counts,
        relation_metadata='four anchored unit-difference rows; zero offsets',
        ignored_free_initial_states='these set numerical transients, not different input-output functions',
        recorded_tree_force_evaluations=bf['tree_total_force_evaluations'],
        recorded_mobile_force_evaluations=bf['mobility_total_force_evaluations'],
        mobility_training_force_evaluations=bf['training_total_force_evaluations'])

    C = np.array([b['constructed_rows']['A'], b['constructed_rows']['B']], dtype=int)
    da = b['learning']['A']['delta_learned']
    db = b['learning']['B']['delta_learned']
    rng = np.random.default_rng(20261101)
    checks = {}
    for mode in ('forward', 'inverse', 'composition'):
        max_residual = 0.
        max_oracle_error = 0.
        counts = None
        for _ in range(64):
            r, x, y, w, z = rng.uniform(-np.pi, np.pi, 5)
            if mode == 'forward':
                rows, offs = C[:1, :4], [da]
                clamps = {0:r, 1:x, 2:y}
                output, output_index = x+y-r+da, 3
            elif mode == 'inverse':
                rows, offs = C[:1, :4], [da]
                clamps = {0:r, 2:y, 3:z}
                output, output_index = z-y+r-da, 1
            else:
                rows, offs = C, [da, db]
                clamps = {0:r, 1:x, 2:y, 4:w}
                output, output_index = x+y+w-2*r+da+db, 5
            theta, counts = interpret(rows.tolist(), offs, clamps)
            max_residual = max(max_residual, float(np.max(np.abs(wrap(rows@theta-np.array(offs))))))
            max_oracle_error = max(max_oracle_error, float(abs(wrap(theta[output_index]-output))))
        checks[mode] = dict(queries=64, max_stored_relation_residual=max_residual,
            max_analytic_algebra_error=max_oracle_error, operations_per_query=counts)
        assert max_residual < 1e-12 and max_oracle_error < 1e-12

    # Additional scoped full arithmetic model for BF only. Each listed elementary
    # scalar operation costs one, sine has variable weight. No hardware claim.
    # Oja charges published mult/add/div/sqrt counts; no diagnostics or data I/O.
    accounting = {}
    for sine_weight in (1, 4, 16):
        tree_force = 6 + 4*sine_weight
        mobile_force = 25 + 4*sine_weight
        tree_steps = bf['tree_total_force_evaluations']//4
        mobile_steps = bf['mobility_total_force_evaluations']//4
        training_integration_forces = 16*2400*4
        training_diagnostic_forces = 16*8
        training_steps = 16*2400
        oja = 128*(20+13+8+2)
        base = bf['tree_total_force_evaluations']*tree_force + tree_steps*52
        inference = bf['mobility_total_force_evaluations']*mobile_force + mobile_steps*52
        acquisition = (training_integration_forces+training_diagnostic_forces)*tree_force + training_steps*52 + oja + 4
        accounting[str(sine_weight)] = dict(tree_inference_arithmetic=base,
            mobile_inference_arithmetic=inference, mobility_acquisition_arithmetic=acquisition,
            mobile_including_acquisition_ratio=(inference+acquisition)/base,
            exclusions='settling diagnostics, norm guards, coefficient setup, dispatch, memory, modulo, physical communication, Python overhead')
    output = dict(status='All frozen direct-substitution assertions passed',
        bf=bf_check, actual_B_learned_offsets=[da, db], fresh_B_queries=checks,
        BF_arithmetic_model=accounting,
        interpretation='Same rows/offsets suffice for exact noniterative tree execution; analog mechanism advantages require additional explicit constraints.')
    with (OUT/'results.json').open('x') as f:
        json.dump(output, f, indent=2)
        f.write('\n')
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()
