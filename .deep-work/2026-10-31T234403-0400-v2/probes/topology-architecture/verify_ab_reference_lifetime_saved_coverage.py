"""Saved-array verification of OWN-address lifetime audit; no NN/SSH/SGD."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

from verify_ab_write_direction_controls_coverage import grid, npz, score, sha, values

ROOT = Path(__file__).resolve().parent
RUN = 'parameter-ab-reference-lifetime-cpu-v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('retain previous saved proof')
    wall, cpu = time.perf_counter(), time.process_time()
    folder = ROOT/'runs'/RUN/'audit'
    result = json.loads((folder/'result.json').read_text())
    config = json.loads((ROOT/'versions'/('code-'+RUN)/'launch-config.json').read_text())
    protocol_path = ROOT/(RUN+'-protocol.json')
    protocol = json.loads(protocol_path.read_text())
    assert result['status'] == 'COMPLETED_AB_OWN_REFERENCE_LIFETIME_CPU'
    for leaf, digest in result['source_sha256'].items():
        assert sha(ROOT/'versions'/('code-'+RUN)/leaf) == digest
    def input_path(leaf):
        spec = config['input_files'][leaf]
        path = ROOT/'runs'/spec['remote_relative']
        assert sha(path) == spec['sha256'] and path.stat().st_size == spec['bytes']
        return path
    public = npz(input_path('valid.npz'))
    checkpoint = input_path('ab.pt')
    assert sha(checkpoint) == result['checkpoint_sha256']
    rows, pairs, truth = grid(public)
    scores, checks = {}, {}
    all_raw = {}
    for record in result['records']:
        name = record['mode']
        path = folder/record['file']
        assert sha(path) == record['sha256'] and path.stat().st_size == record['bytes']
        z = npz(path)
        all_raw[name] = z
        L = z['active']['shape'][1]
        overrides = 0
        max_attention_error = 0.
        pair_templates = {}
        for row, extent in enumerate(values(public, 'lengths')):
            word = list(values(public, 'tokens')[row*128:row*128+extent])
            writes = [i for i, token in enumerate(word) if token == 9]
            keys = [0, writes[0]+1]
            assert list(values(z, 'write_positions')[row*2:row*2+2]) == writes
            assert list(values(z, 'key_positions')[row*2:row*2+2]) == keys
            masked = word.copy()
            masked[keys[0]] = masked[keys[1]] = -1
            pair_templates.setdefault((values(public, 'initial')[row], tuple(masked)), []).append(row)
            for t in range(L):
                event = row*L+t
                active = t < extent
                held = any(left < t <= right for left, right in zip(keys, writes))
                assert bool(values(z, 'active')[event]) == active
                assert bool(values(z, 'hold_mask')[event]) == held
                if t >= extent-4:
                    assert not held
                before = values(z, 'before_address')[event*8:event*8+8]
                proposed = values(z, 'proposed_address')[event*8:event*8+8]
                after = values(z, 'after_address')[event*8:event*8+8]
                if not active:
                    continue
                if t:
                    assert list(before) == list(values(z, 'after_address')[(event-1)*8:event*8])
                if held:
                    which = next(k for k, (left, right) in enumerate(zip(keys, writes)) if left < t <= right)
                    own = values(z, 'key_snapshots')[(row*2+which)*8:(row*2+which+1)*8]
                    assert list(before) == list(own) and list(after) == list(own)
                    overrides += 1
                else:
                    assert list(after) == list(proposed)
                if t in keys:
                    which = keys.index(t)
                    own = values(z, 'key_snapshots')[(row*2+which)*8:(row*2+which+1)*8]
                    assert list(own) == list(proposed) == list(after)
                normalizer = max(before)
                exponentials = [math.exp(x-normalizer) for x in before]
                total = math.fsum(exponentials)
                attention = values(z, 'read_attention')[event*8:event*8+8]
                max_attention_error = max(max_attention_error, max(abs(a-b/total) for a, b in zip(attention, exponentials)))
                assert max_attention_error < 2e-5
                alpha = values(z, 'action_probabilities')[event*7:event*7+7]
                action = values(z, 'action_indices')[event]
                assert action == max(range(7), key=alpha.__getitem__)
                amount = float(action == 5) if name == 'native' else alpha[5]
                assert all(abs(weight-amount*a) < 2e-7 for weight, a in zip(values(z, 'w')[event*8:event*8+8], attention))
            assert list(values(z, 'final_address')[row*8:row*8+8]) == list(values(z, 'after_address')[(row*L+L-1)*8:(row*L+L)*8])
        assert overrides == record['address_override_rows']
        keyflip_TV = []
        for paired in pair_templates.values():
            assert len(paired) == 2
            a, b = paired
            position = values(z, 'write_positions')[a*2]
            left = values(z, 'read_attention')[(a*L+position)*8:(a*L+position+1)*8]
            right = values(z, 'read_attention')[(b*L+position)*8:(b*L+position+1)*8]
            keyflip_TV.append(.5*math.fsum(abs(x-y) for x, y in zip(left, right)))
        measured, pred = score(values(z, 'final_logp'), truth, rows, pairs)
        assert pred == list(values(z, 'predictions'))
        for key in ('correct', 'accuracy'):
            assert measured[key] == result['scores'][name][key]
        assert abs(measured['NLL']-result['scores'][name]['NLL']) < 5e-7
        q = measured['pairs']['query_nondegenerate']
        assert (q['both_correct_AND_relation'], q['relation_only_count'], q['prediction_changes']) == (result['scores'][name]['query_both_correct'], result['scores'][name]['query_relation_consistent'], result['scores'][name]['query_pred_changes'])
        scores['held_'+name] = measured
        checks[name] = dict(raw_sha256=sha(path), exact_own_updated_KEY_snapshots=True,
                            before_and_after_address_exact_through_and_at_WRITE=True,
                            active_address_continuity_exact=True, hold_mask_matches_public_intervals=True,
                            query_suffix_hold_mask_all_false=True, outside_hold_after_address_equals_proposed_exact=True,
                            read_attention_softmax_of_actual_before_address_max_abs=max_attention_error,
                            captured_write_weights_match_mode=True, override_events=overrides,
                            first_WRITE_KEYflip_attention_TV_mean=math.fsum(keyflip_TV)/len(keyflip_TV),
                            first_WRITE_KEYflip_attention_TV_min=min(keyflip_TV),
                            first_WRITE_KEYflip_attention_TV_max=max(keyflip_TV))
    previous = ROOT/'runs/parameter-ab-fractional-store-cpu-v1/audit'
    old_result = json.loads((previous/'result.json').read_text())
    for name in ('soft', 'native'):
        receipt = next(record for record in old_result['records'] if record['key'] == name)
        path = previous/receipt['raw_file']
        assert sha(path) == receipt['raw_sha256']
        old = values(npz(path), 'after_logp')
        logp = [old[(row*8+7)*8+k] for row in range(512) for k in range(8)]
        scores['previous_'+name], _ = score(logp, truth, rows, pairs)
    cold = json.loads((ROOT/(RUN+'-coordinator-receipt.json')).read_text())
    cost_sidecar = json.loads((ROOT/(RUN+'-diagnostic-temp-state.json')).read_text())
    output = dict(status='PASS_INDEPENDENT_COVERAGE_OWN_ADDRESS_LIFETIME_RAW',
                  source_verifier_sha256=sha(Path(__file__)), parser_helper_sha256=sha(ROOT/'verify_ab_write_direction_controls_coverage.py'),
                  actual_result_sha256=sha(folder/'result.json'), protocol_sha256=sha(protocol_path),
                  actual_source_sha256=result['source_sha256'], checkpoint_file_sha256=sha(checkpoint),
                  raw_checks=checks, scores=scores, NN_executions=0, SGD_updates=0,
                  caveat='suffix has no override and uses the original transition, not unchanged incoming state/output; oracle public interval timing, no supplied correct slot; exact frozen repair not learned HOLD test',
                  actual_cost=dict(P=result['P'], NN_bytes=result['NN_parameter_bytes'], carry_scalars=result['carry_scalars'],
                                   active_NN_transitions=result['active_NN_transitions'], transition_calls=result['NN_transition_calls'],
                                   science_wall_seconds=result['wall_seconds'], CPU_seconds=result['CPU_seconds'],
                                   local_cold_wall_seconds=cold['wall_seconds'], CPU_peak_RSS_bytes=result['CPU_peak_RSS_bytes'],
                                   checkpoint_bytes=result['checkpoint_bytes'], live_optimizer_bytes=0,
                                   raw_compressed_bytes=sum(record['bytes'] for record in result['records']),
                                   raw_array_payload_bytes=sum(record['array_payload_bytes'] for record in result['records']),
                                   diagnostic_temporary_state=cost_sidecar, full_FLOPs=None, provider_effective_compute=None),
                  analysis_wall_seconds=time.perf_counter()-wall, analysis_CPU_seconds=time.process_time()-cpu)
    with args.out.open('x') as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(status=output['status'], out=str(args.out), sha256=sha(args.out), raw_checks=checks,
                         outcomes={name: {'correct': record['correct'], 'candidate': record['candidate_coverage'], 'query_both': record['pairs']['query_nondegenerate']['both_correct_AND_relation'], 'query_relation': record['pairs']['query_nondegenerate']['relation_only_count']} for name, record in scores.items()}, analysis_wall_seconds=output['analysis_wall_seconds'])))


if __name__ == '__main__':
    main()
