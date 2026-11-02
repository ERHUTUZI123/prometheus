"""Independent standard-library saved-array check; zero NN, SGD, or SSH.

Reconstruct fixed-strength content interventions and final memory by explicit
backward recency coefficients, then recompute all paired VALID metrics.
"""
import argparse
import array
import ast
import hashlib
import itertools
import json
import math
import struct
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = 'parameter-ab-write-direction-controls-cpu-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path):
    answer = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            raw = archive.read(name)
            assert raw[:6] == b'\x93NUMPY'
            offset = 10 if raw[6] == 1 else 12
            length = int.from_bytes(raw[8:offset], 'little')
            header = ast.literal_eval(raw[offset:offset+length].decode())
            assert not header['fortran_order']
            code = {'<f4': 'f', '<f8': 'd', '<i8': 'q', '<i4': 'i', '|u1': 'B', '|b1': 'B'}[header['descr']]
            values = array.array(code)
            values.frombytes(raw[offset+length:])
            if sys.byteorder != 'little' and values.itemsize > 1:
                values.byteswap()
            assert len(values) == math.prod(header['shape'])
            answer[name[:-4]] = dict(shape=header['shape'], values=values,
                                    payload_sha256=hashlib.sha256(raw[offset+length:]).hexdigest())
    return answer


def values(saved, key):
    return saved[key]['values']


def grid(public):
    lookup, rows = {}, []
    tokens, lengths, initial, truth = (values(public, key) for key in ('tokens', 'lengths', 'initial', 'targets'))
    for row, length in enumerate(lengths):
        word = list(tokens[row*128:row*128+length])
        position, writes = 0, []
        for _ in range(2):
            key = word[position]-7
            assert key in (0, 1) and word[position+1:position+3] == [14, 1]
            position += 3
            value = 0
            while word[position] == 3:
                value += 1
                position += 1
            assert word[position] == 9 and value < 8
            position += 1
            writes.append((key, value))
        query = word[-4]-7
        assert word[-4:] == [7+query, 10, 8-query, 15]
        operands = dict(writes)
        template = (initial[row], tuple(word[position:-4]))
        key = (template, operands[0], operands[1], query, writes[0][0])
        assert key not in lookup
        lookup[key] = row
        rows.append(key)
        assert truth[row] == (3*operands[query]+operands[1-query]) % 8
    query_pairs = [(i, lookup[(g, v0, v1, 1, w)])
                   for i, (g, v0, v1, q, w) in enumerate(rows) if q == 0]
    write_pairs = [(i, lookup[(g, v0, v1, q, 1)])
                   for i, (g, v0, v1, q, w) in enumerate(rows) if w == 0]
    pairs = dict(query_all=query_pairs, query_nondegenerate=[(a, b) for a, b in query_pairs if truth[a] != truth[b]],
                 query_degenerate=[(a, b) for a, b in query_pairs if truth[a] == truth[b]],
                 write_order=write_pairs)
    for key in (0, 1):
        current = []
        for group, other, query, write in itertools.product(set(row[0] for row in rows), range(8), range(2), range(2)):
            for lo, hi in itertools.combinations(range(8), 2):
                a = (group, lo, other, query, write) if key == 0 else (group, other, lo, query, write)
                b = (group, hi, other, query, write) if key == 0 else (group, other, hi, query, write)
                current.append((lookup[a], lookup[b]))
        assert len(current) == 1792
        pairs['value_key'+str(key)] = current
    assert len(pairs['query_nondegenerate']) == 192 and len(write_pairs) == 256
    return rows, pairs, truth


def score(logp, truth, rows, pairs):
    pred = [max(range(8), key=lambda k: logp[row*8+k]) for row in range(512)]
    error = max(abs(math.fsum(math.exp(x) for x in logp[row*8:row*8+8])-1) for row in range(512))
    assert error < 2e-5 and all(math.isfinite(x) for x in logp)
    outcomes = {}
    for name, paired in pairs.items():
        both = sum(pred[a] == truth[a] and pred[b] == truth[b] for a, b in paired)
        relation = sum((pred[b]-pred[a]) % 8 == (truth[b]-truth[a]) % 8 for a, b in paired)
        outcomes[name] = dict(pairs=len(paired), both_correct_AND_relation=both,
                              both_correct_rate=both/len(paired), relation_only_count=relation,
                              relation_rate=relation/len(paired), prediction_changes=sum(pred[a] != pred[b] for a, b in paired))
    correct = sum(a == b for a, b in zip(pred, truth))
    query_probability_TV = [.5*math.fsum(abs(math.exp(logp[a*8+k])-math.exp(logp[b*8+k])) for k in range(8)) for a, b in pairs['query_nondegenerate']]
    query_logp_delta = [max(abs(logp[a*8+k]-logp[b*8+k]) for k in range(8)) for a, b in pairs['query_nondegenerate']]
    return dict(correct=correct, accuracy=correct/512, NLL=math.fsum(-logp[i*8+truth[i]] for i in range(512))/512,
                probability_normalization_max_error=error,
                query_twin_probability_TV_mean=math.fsum(query_probability_TV)/len(query_probability_TV),
                query_twin_probability_TV_max=max(query_probability_TV),
                query_twin_logp_Linf_mean=math.fsum(query_logp_delta)/len(query_logp_delta),
                query_twin_logp_Linf_max=max(query_logp_delta),
                candidate_coverage=sum(pred[i] in {(3*v0+v1) % 8, (3*v1+v0) % 8} for i, (_, v0, v1, q, w) in enumerate(rows)),
                pairs=outcomes), pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('preserve prior verification')
    began, cpu = time.perf_counter(), time.process_time()
    folder = ROOT/'runs'/RUN/'audit'
    result = json.loads((folder/'result.json').read_text())
    config = json.loads((ROOT/'versions'/('code-'+RUN)/'launch-config.json').read_text())
    assert result['status'] == 'COMPLETED_AB_FIXED_STRENGTH_DIRECTION_CONTROLS_CPU'
    for name, digest in result['source_sha256'].items():
        assert sha(ROOT/'versions'/('code-'+RUN)/name) == digest
    raw = {}
    for receipt in result['records']:
        path = folder/receipt['file']
        assert sha(path) == receipt['sha256'] and path.stat().st_size == receipt['bytes']
        raw[receipt['condition']] = npz(path)
    signals_path = folder/result['signals']['file']
    assert sha(signals_path) == result['signals']['sha256']
    sig = npz(signals_path)
    ref, sham = raw['reference'], raw['sham']
    assert ref.keys() == sham.keys()
    assert all(ref[key]['payload_sha256'] == sham[key]['payload_sha256'] for key in ref)
    for name in ('sham', 'permute_direction', 'cached_direction'):
        for key in ('w', 'coverage', 'final_coverage', 'active', 'lengths', 'initial'):
            assert raw[name][key]['payload_sha256'] == ref[key]['payload_sha256']
    valid_spec = config['input_files']['valid.npz']
    public_path = ROOT/'runs'/valid_spec['remote_relative']
    assert sha(public_path) == valid_spec['sha256']
    public = npz(public_path)
    rows, pairs, truth = grid(public)
    prefix = {}
    length = ref['active']['shape'][1]
    for row, extent in enumerate(values(public, 'lengths')):
        word = values(public, 'tokens')[row*128:row*128+extent-4]
        prefix.setdefault((values(public, 'initial')[row], tuple(word)), []).append(row)
    members = list(prefix.values())
    assert len(members) == 256 and all(len(pair) == 2 for pair in members)
    M, donor, active, selected = (values(sig, 'treatment_mask'), values(sig, 'donor_row'), values(ref, 'active'), values(ref, 'store_selected'))
    href, wref = values(ref, 'h'), values(ref, 'w')
    perm, cached = values(sig, 'permutation_content'), values(sig, 'cached_content')
    changed = {'permutation': 0, 'cached': 0}
    max_norm = {'permutation': 0., 'cached': 0.}
    max_content_reconstruction = {'permutation': 0., 'cached': 0.}
    zero_cache = pre_first_count = 0
    for t in range(length):
        eligible = [g for g, (a, b) in enumerate(members) if active[a*length+t] and active[b*length+t] and not selected[a*length+t] and not selected[b*length+t]]
        expected = {}
        if len(eligible) > 1:
            shift = 127 % len(eligible) or 1
            for index, group in enumerate(eligible):
                for row in members[group]:
                    expected[row] = members[eligible[(index+shift) % len(eligible)]][0]
        for row in range(512):
            event = row*length+t
            assert bool(M[event]) == (row in expected)
            assert donor[event] == expected.get(row, -1)
    first_times = []
    for row in range(512):
        last, first = None, -1
        for t in range(length):
            event = row*length+t
            base = event*32
            original = href[base:base+32]
            own_norm = math.sqrt(math.fsum(x*x for x in original))
            assert abs(own_norm-values(sig, 'reference_content_norm')[event]) < 1e-12
            assert bool(values(sig, 'pre_first_selected')[event]) == bool(active[event] and last is None)
            if M[event]:
                pre_first_count += last is None
                donor_base = (donor[event]*length+t)*32
                directions = {'permutation': href[donor_base:donor_base+32], 'cached': ([0.]*32 if last is None else last)}
                for name, actual in (('permutation', perm), ('cached', cached)):
                    observed = actual[base:base+32]
                    direction = directions[name]
                    denominator = math.sqrt(math.fsum(x*x for x in direction))
                    if name == 'cached':
                        zero_cache += denominator == 0
                    expected = ([0.]*32 if own_norm == 0 else [own_norm]+[0.]*31 if denominator == 0 else [x*own_norm/denominator for x in direction])
                    max_content_reconstruction[name] = max(max_content_reconstruction[name], max(abs(x-y) for x, y in zip(observed, expected)))
                    error = abs(math.sqrt(math.fsum(x*x for x in observed))-own_norm)
                    max_norm[name] = max(max_norm[name], error)
                    assert error <= 2e-5+2e-5*own_norm
                    changed[name] += math.sqrt(math.fsum((x-y)**2 for x, y in zip(observed, original))) > 1e-6
            else:
                assert list(perm[base:base+32]) == list(original) and list(cached[base:base+32]) == list(original)
            if active[event] and selected[event]:
                if first < 0:
                    first = t
                last = list(original)
        first_times.append(first)
    assert first_times == list(values(sig, 'first_selected_time'))
    classes = ('treated', 'excluded_nonchosen', 'treated_pre_first', 'treated_post_first', 'reference_selected')
    sums = {name: 0. for name in classes}
    max_lambda = max_true_lambda_saved_difference = max_z = max_final_V = max_delta = 0.
    for row in range(512):
        for slot in range(8):
            survival = saved_survival = 1.
            coefficients = [0.]*length
            saved_coefficients = [0.]*length
            for t in reversed(range(length)):
                weight = wref[(row*length+t)*8+slot]
                assert 0 <= weight <= 1
                coefficients[t] = weight*survival
                saved_coefficients[t] = weight*saved_survival
                survival *= 1-weight
                # Producer mass diagnostics subtract w from one in FP32,
                # then promote the product to float64. Reproduce that
                # rounding separately from the real-valued lambda identity.
                saved_survival *= struct.unpack('<f', struct.pack('<f', 1-weight))[0]
            masks = {name: [] for name in classes}
            for t in range(length):
                event = row*length+t
                masks['treated'].append(bool(M[event]))
                masks['excluded_nonchosen'].append(bool(active[event] and not selected[event] and not M[event]))
                masks['treated_pre_first'].append(bool(M[event] and values(sig, 'pre_first_selected')[event]))
                masks['treated_post_first'].append(bool(M[event] and not values(sig, 'pre_first_selected')[event]))
                masks['reference_selected'].append(bool(selected[event]))
            for name in classes:
                mass = math.fsum(c for c, included in zip(coefficients, masks[name]) if included)
                sums[name] += mass
                saved_mass = math.fsum(c for c, included in zip(saved_coefficients, masks[name]) if included)
                observed_mass = values(sig, name+'_effective_strength')[row*8+slot]
                error = abs(saved_mass-observed_mass)
                max_lambda = max(max_lambda, error)
                max_true_lambda_saved_difference = max(max_true_lambda_saved_difference, abs(mass-observed_mass))
                assert error < 1e-12
            error = abs(math.fsum(coefficients)-values(ref, 'final_coverage')[row*8+slot])
            max_z = max(max_z, error)
            assert error < 2e-5
            for dimension in range(32):
                bases = [((row*length+t)*32+dimension) for t in range(length)]
                memory_index = (row*8+slot)*32+dimension
                expected_ref = math.fsum(c*href[b] for c, b in zip(coefficients, bases))
                max_final_V = max(max_final_V, abs(expected_ref-values(ref, 'final_values')[memory_index]))
                for name, content in (('permute_direction', perm), ('cached_direction', cached)):
                    expected_value = math.fsum(c*content[b] for c, b in zip(coefficients, bases))
                    observed_value = values(raw[name], 'final_values')[memory_index]
                    error = abs(expected_value-observed_value)
                    max_final_V = max(max_final_V, error)
                    expected_delta = math.fsum(c*(content[b]-href[b]) for c, b in zip(coefficients, bases))
                    observed_delta = observed_value-values(ref, 'final_values')[memory_index]
                    max_delta = max(max_delta, abs(expected_delta-observed_delta))
                    assert error <= 2e-5+2e-5*abs(observed_value)
    previous_folder = ROOT/'runs/parameter-ab-fractional-store-cpu-v1/audit'
    all_logp = {name: list(values(saved, 'soft_logp')) for name, saved in raw.items()}
    previous_raw = {}
    for name in ('soft', 'native', 'soft_drop_nonchosen_STORE'):
        previous = npz(previous_folder/(name+'-NN-events-before-truth.npz'))
        logp = values(previous, 'after_logp')
        previous_raw[name] = [logp[(row*8+7)*8+k] for row in range(512) for k in range(8)]
    all_logp.update(native_previous=previous_raw['native'], drop_previous=previous_raw['soft_drop_nonchosen_STORE'])
    scores, predictions = {}, {}
    for name, logp in all_logp.items():
        scores[name], predictions[name] = score(logp, truth, rows, pairs)
    native_accuracy = scores['native_previous']['accuracy']
    for name in scores:
        scores[name]['soft_condition_minus_fixed_previous_native_accuracy'] = scores[name]['accuracy']-native_accuracy
        scores[name]['argmax_changes_vs_reference'] = sum(a != b for a, b in zip(predictions[name], predictions['reference']))
    reference_previous_error = max(abs(a-b) for a, b in zip(all_logp['reference'], previous_raw['soft']))
    for name in raw:
        reported = result['scores'][name]
        assert scores[name]['correct'] == reported['correct']
        assert abs(scores[name]['NLL']-reported['NLL']) < 5e-7
        q = scores[name]['pairs']['query_nondegenerate']
        assert (q['both_correct_AND_relation'], q['relation_only_count'], q['prediction_changes']) == (reported['query_both_correct'], reported['query_relation_consistent'], reported['query_pred_changes'])
    receipt = json.loads((ROOT/(RUN+'-coordinator-receipt.json')).read_text())
    proof = dict(status='PASS_INDEPENDENT_COVERAGE_FIXED_STRENGTH_DIRECTION_RAW', NN_executions=0, SGD_updates=0,
                 result_sha256=sha(folder/'result.json'), source_sha256=result['source_sha256'],
                 all_sham_arrays_bitwise_equal=True, sham_full_trajectory_fields=list(ref),
                 all_condition_w_and_full_coverage_bitwise_equal=True,
                 exact_paired_M_and_derangement_reconstructed=True,
                 treated_events=sum(M), actual_changed_content_events=changed,
                 first_selected_times_reconstructed=True, rows_never_selected=sum(x < 0 for x in first_times),
                 cached_e0_fallback_events=zero_cache, treated_pre_first_events=pre_first_count,
                 max_content_norm_abs_error=max_norm, max_replacement_direction_reconstruction_abs_error=max_content_reconstruction,
                 effective_retained_strength=sums, backward_lambda_vs_saved_forward_mass_max_abs=max_lambda,
                 exact_real_lambda_vs_mixed_precision_saved_mass_max_abs=max_true_lambda_saved_difference,
                 mass_rounding_scope='producer computes 1-w in float32 before float64 retained-mass multiplication; exact real-weight lambda and producer diagnostic reconstructed separately',
                 backward_lambda_z_final_max_abs=max_z, backward_lambda_V_final_max_abs=max_final_V,
                 explicit_linear_DeltaV_identity_max_abs=max_delta,
                 reference_vs_previous_soft_CPU_logp_max_abs=reference_previous_error,
                 scores=scores, actual_audit_science_wall_seconds=result['wall_seconds'],
                 actual_audit_CPU_seconds=result['CPU_seconds'], actual_audit_local_cold_wall_seconds=receipt['wall_seconds'],
                 actual_audit_CPU_peak_RSS_bytes=result['CPU_peak_RSS_bytes'], active_NN_transitions=result['active_NN_transitions'],
                 raw_compressed_bytes=sum(record['bytes'] for record in result['records'])+result['signals']['bytes'],
                 raw_array_payload_bytes=sum(record['array_payload_bytes'] for record in result['records'])+result['signals']['payload_bytes'],
                 full_FLOPs=None, provider_effective_compute=None,
                 analysis_wall_seconds=time.perf_counter()-began, analysis_CPU_seconds=time.process_time()-cpu,
                 caveat='reference-conditional content-direction intervention; norms/schedule/unchanged-write contents retain factual information; neither pure semantics nor native sufficiency/binding/learning result')
    with args.out.open('x') as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(status=proof['status'], out=str(args.out), sha256=sha(args.out), sham_bitwise=True,
                         norms=max_norm, V_reconstruction=max_final_V, DeltaV=max_delta, scores={name: {'correct': row['correct'], 'query_both': row['pairs']['query_nondegenerate']['both_correct_AND_relation'], 'query_relation': row['pairs']['query_nondegenerate']['relation_only_count']} for name, row in scores.items()}, analysis_wall_seconds=proof['analysis_wall_seconds'])))


if __name__ == '__main__':
    main()
