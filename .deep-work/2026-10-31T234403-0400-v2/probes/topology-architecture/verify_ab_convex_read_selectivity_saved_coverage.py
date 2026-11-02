"""Independent backward-coefficient check of a conditional convex-read bound.

Standard library and frozen arrays only. No NN/checkpoint/SSH or target read.
"""
import argparse
import json
import math
import time
from pathlib import Path

from analyze_ab_reference_key_lifetime_saved import saved, sha

ROOT = Path(__file__).resolve().parent


def summary(rows):
    first = [row for row in rows if row['requested_ordinal'] == 1]
    second = [row for row in rows if row['requested_ordinal'] == 2]
    answer = {}
    for name, selected in (('all', rows), ('first_write_requested', first), ('second_write_requested', second)):
        answer[name] = dict(rows=len(selected),
                           backward_requested_max_mean=math.fsum(row['maximum'] for row in selected)/len(selected),
                           backward_actual_requested_mean=math.fsum(row['actual'] for row in selected)/len(selected),
                           actual_using_saved_z_mean=math.fsum(row['saved_z_actual'] for row in selected)/len(selected),
                           maximum_using_saved_z_mean=math.fsum(row['saved_z_maximum'] for row in selected)/len(selected),
                           weak_other_coefficient_dominance_rows=sum(row['max_requested_minus_other'] <= 0 for row in selected),
                           strict_other_coefficient_dominance_rows=sum(row['max_requested_minus_other'] < 0 for row in selected),
                           maximum_margin_exact_zero_rows=sum(row['max_requested_minus_other'] == 0 for row in selected),
                           neither_literal_WRITE_has_any_coefficient_rows=sum(row['both_literal_zero'] for row in selected))
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('preserve existing proof')
    wall, cpu = time.perf_counter(), time.process_time()
    original_path = ROOT/'ab-convex-read-selectivity-saved-v1.json'
    original = json.loads(original_path.read_text())
    assert sha(original_path) == '591bdaf947ae645c177728d0ba5703c5149c4ac672ccd5f3c231aa4bc36071e0'
    assert sha(ROOT/'analyze_ab_convex_read_selectivity_saved.py') == original['source_sha256']
    public_path = ROOT/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated/data/stage3-valid.npz'
    assert sha(public_path) == original['public_data_sha256']
    public = saved(public_path, ('tokens', 'lengths'))
    models = {}
    for name, published in original['models'].items():
        path = Path(published['raw_path'])
        assert sha(path) == published['raw_sha256']
        keys = ('w', 'coverage', 'read_attention') if name.startswith('held_') else ('w', 'coverage')
        z = saved(path, keys)
        L = z['w']['shape'][1]
        w = z['w']['values']
        rows = []
        max_agreement = max_z_error = max_saved_z_coefficient_change = max_attention_norm_error = 0.
        max_unnormalized_bound_roundoff = 0.
        floor_cells = 0
        for row, extent in enumerate(public['lengths']['values']):
            word = list(public['tokens']['values'][row*128:row*128+extent])
            writes = [t for t, token in enumerate(word) if token == 9]
            assert len(writes) == 2 and set((word[0], word[writes[0]+1])) == {7, 8}
            read = extent-3
            assert word[-4:] in ([7, 10, 8, 15], [8, 10, 7, 15]) and word[read] == 10
            requested = next(index for index, key_position in enumerate((0, writes[0]+1)) if word[key_position] == word[extent-4])
            assert writes[requested] < read
            # A separate backward product, rather than the producer's forward
            # two-contribution recurrence. READ's own update is excluded.
            denominator, literal = [], [[], []]
            for slot in range(8):
                survival = 1.
                coefficients = [0.]*read
                for t in reversed(range(read)):
                    weight = w[(row*L+t)*8+slot]
                    assert 0 <= weight <= 1
                    coefficients[t] = weight*survival
                    survival *= 1-weight
                coverage = math.fsum(coefficients)
                assert abs(coverage-(1-survival)) < 1e-14
                observed_z = z['coverage']['values'][(row*L+read-1)*8+slot]
                max_z_error = max(max_z_error, abs(coverage-observed_z))
                floor_cells += observed_z < 1e-4
                denominator.append((max(coverage, 1e-4), max(observed_z, 1e-4)))
                for index in (0, 1):
                    literal[index].append(coefficients[writes[index]])
            coeff = [[literal[index][slot]/denominator[slot][0] for slot in range(8)] for index in (0, 1)]
            saved_coeff = [[literal[index][slot]/denominator[slot][1] for slot in range(8)] for index in (0, 1)]
            assert all(value >= 0 for current in coeff for value in current)
            assert all(coeff[0][slot]+coeff[1][slot] <= 1+1e-12 for slot in range(8))
            if 'read_attention' in z:
                attention = list(z['read_attention']['values'][(row*L+read)*8:(row*L+read+1)*8])
            else:
                wt = w[(row*L+read)*8:(row*L+read+1)*8]
                amount = math.fsum(wt)
                assert amount > 0
                attention = [x/amount for x in wt]
            assert all(value >= 0 for value in attention)
            mass = math.fsum(attention)
            max_attention_norm_error = max(max_attention_norm_error, abs(mass-1))
            maximum = max(coeff[requested])
            actual = math.fsum(a*p for a, p in zip(attention, coeff[requested]))
            other = math.fsum(a*p for a, p in zip(attention, coeff[1-requested]))
            # Captured FP32 softmax sums differ slightly from exact one.
            # The exact nonnegative-vector inequality is actual<=max(p)*sum(a).
            max_unnormalized_bound_roundoff = max(max_unnormalized_bound_roundoff, actual-maximum*mass)
            assert actual <= maximum*mass+2e-15
            assert min(coeff[requested])-2e-15 <= actual/mass <= maximum+2e-15
            margin = max(a-b for a, b in zip(coeff[requested], coeff[1-requested]))
            if margin <= 0:
                assert actual <= other+2e-15
            saved_actual = math.fsum(a*p for a, p in zip(attention, saved_coeff[requested]))
            max_saved_z_coefficient_change = max(max_saved_z_coefficient_change, abs(saved_actual-actual))
            expected = published['per_row'][row]
            assert expected['row'] == row and expected['requested_write'] == requested+1
            computed = dict(requested_max=maximum, requested_min=min(coeff[requested]),
                            requested_minus_other_max=margin, actual_requested=actual, actual_other=other)
            for key, value in computed.items():
                difference = abs(value-expected[key])
                max_agreement = max(max_agreement, difference)
                assert difference < 2e-12
            rows.append(dict(row=row, requested_ordinal=requested+1, maximum=maximum, actual=actual,
                             saved_z_actual=saved_actual, saved_z_maximum=max(saved_coeff[requested]),
                             max_requested_minus_other=margin,
                             both_literal_zero=all(value == 0 for current in literal for value in current)))
        models[name] = dict(raw_sha256=sha(path), rows=512, summaries=summary(rows),
                            backward_vs_published_forward_coefficient_max_abs=max_agreement,
                            actual_saved_z_reconstruction_max_abs=max_z_error,
                            actual_coefficient_change_using_saved_z_max_abs=max_saved_z_coefficient_change,
                            captured_attention_sum_max_error=max_attention_norm_error,
                            exact_nonnegative_bound_largest_roundoff=max_unnormalized_bound_roundoff,
                            READ_denominator_floor_cells=floor_cells,
                            roles_and_pre_READ_index_verified=True)
    proof = dict(status='PASS_INDEPENDENT_CONDITIONAL_CONVEX_READ_SELECTIVITY',
                 source_verifier_sha256=sha(Path(__file__)), parser_sha256=sha(ROOT/'analyze_ab_reference_key_lifetime_saved.py'),
                 original_artifact_sha256=sha(original_path), original_analysis_source_sha256=original['source_sha256'],
                 public_data_sha256=sha(public_path), models=models, NN_executions=0, SGD_updates=0,
                 targets_loaded=False, teacher_truth_loaded=False,
                 proof='At fixed history, b_j=lambda_requested,j/max(z_j,epsilon)>=0 and r has requested-vector coefficient a dot b. For a>=0 and sum(a)=1, min(b)<=a dot b<=max(b); if every b_j<=c_j then a dot b<=a dot c. Captured finite-precision attention additionally carries its measured sum(a) factor.',
                 boundaries=['Bound on literal OLDh vectors written at the two public WRITE times, not semantic value information, read fidelity or task accuracy.',
                             'Other history vectors may retain/re-encode the same operand; fixed-history coefficients are not the full derivative of the recurrent computation.',
                             'Means are means of per-row bounds, not uniform per-example ceilings.',
                             'The native trace is held_native, not original native.',
                             'Signed/unconstrained/nonlinear read maps fall outside the simplex bound; even existing nonlinear decoders or separated representations can use mixtures.',
                             'Weak no-slot advantage includes ties and zero literal-write contributions; strict dominance and zero cases reported separately.'],
                 analysis_wall_seconds=time.perf_counter()-wall, analysis_CPU_seconds=time.process_time()-cpu,
                 full_FLOPs=None, provider_effective_compute=None)
    with args.out.open('x') as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(status=proof['status'], out=str(args.out), sha256=sha(args.out), models={name: row['summaries']['first_write_requested'] for name, row in models.items()}, analysis_wall_seconds=proof['analysis_wall_seconds'])))


if __name__ == '__main__':
    main()
