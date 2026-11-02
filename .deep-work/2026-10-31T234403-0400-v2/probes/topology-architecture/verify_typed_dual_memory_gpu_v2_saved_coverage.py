"""Saved-only independent FP64 execution-repair verification, no NN/SSH."""
import argparse
import hashlib
import json
import math
import statistics
import struct
import time
from pathlib import Path

from verify_factorized_graph600_saved import npz
from verify_typed_dual_memory_cpu_saved_coverage import numeric_pt, sha

ROOT = Path(__file__).resolve().parent
RUN = 'parameter-typed-dual-memory-gpu-execution-v2'


def verify_pair(folder, declared, baseline):
    path = folder/declared['file']
    assert sha(path) == declared['sha256']
    raw = npz(path)
    names = json.loads(bytes(raw['gradient_names_json']['values']).decode())
    assert len(names) == 20 and names == json.loads(bytes(baseline['gradient_names_json']['values']).decode())
    failures, maximum_gradient, maximum_output = [], 0., 0.
    for key, descriptor in declared['fields'].items():
        a, b = raw['CPU_'+key], raw['GPU_'+key]
        assert a['shape'] == b['shape'] and a['values'] == baseline[key]['values']
        assert a['raw_bytes_sha256'] == baseline[key]['raw_bytes_sha256']
        atol, rtol = (2e-5, 2e-4) if key.startswith('grad_') else (2e-5, 2e-5)
        assert (atol, rtol) == (descriptor['atol'], descriptor['rtol'])
        maximum = max(abs(x-y) for x, y in zip(a['values'], b['values']))
        finite = all(math.isfinite(x) and math.isfinite(y) for x, y in zip(a['values'], b['values']))
        passed = all(abs(x-y) <= atol+rtol*abs(y) for x, y in zip(a['values'], b['values']))
        assert (finite, passed, maximum) == (descriptor['finite'], descriptor['allclose'], descriptor['max_abs'])
        if not passed:
            failures.append(names[int(key[5:])] if key.startswith('grad_') else key)
        if key.startswith('grad_'):
            maximum_gradient = max(maximum_gradient, maximum)
        elif key == 'output':
            maximum_output = maximum
    assert declared['fixed_tolerances_pass'] == (not failures)
    return dict(raw_sha256=sha(path), all20_checked=True, failed_fields=failures,
                fixed_tolerances_pass=not failures, max_gradient_abs=maximum_gradient, output_max_abs=maximum_output)


def info(record):
    assert record['forward_calls'] == 16 and record['scalar_infos'] == 1024
    assert record['nonzero'] == 0 and record['max_abs'] == 0 and record['backward_info_exposed'] is False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('preserve earlier verification')
    wall, cpu = time.perf_counter(), time.process_time()
    folder = ROOT/'runs'/RUN/'admit'
    result = json.loads((folder/'result.json').read_text())
    evidence = result['evidence']
    for leaf, digest in result['source_sha256'].items():
        assert sha(ROOT/'versions'/('code-'+RUN)/leaf) == digest
    hybrid = evidence['hybrid_CPU_reference']
    baseline_path = folder/'fresh-hybrid-CPU-reference.npz'
    assert sha(baseline_path) == hybrid['raw_sha256']
    baseline = npz(baseline_path)
    assert hybrid['NN_FB'] == 1 and hybrid['reload_NN_forward_only'] == 1 and hybrid['strict_v2_reload'] and hybrid['v1_checkpoint_rejected']
    info(hybrid['info'])
    info(hybrid['reload_info'])
    info(evidence['GPU_reference_info'])
    assert len(evidence['eager_info_checks']) == 13
    for record in evidence['eager_info_checks']:
        info(record)
    eager = verify_pair(folder, evidence['CPU_GPU_eager'], baseline)
    before = json.loads((folder/'EAGER_EVIDENCE_BEFORE_CAPTURE.json').read_text())
    assert all(evidence[key] == value for key, value in before.items())
    replay = None
    if result['capture']['supported']:
        assert result['actual_graph_replay_FB'] == 11 and len(evidence['graph_info_checks']) == 11
        for record in evidence['graph_info_checks']:
            info(record)
        replay = verify_pair(folder, evidence['CPU_GPU_graph'], baseline)
        assert evidence['theta_unchanged_after_graph']
    else:
        assert result['actual_graph_replay_FB'] == 0 and result['capture']['post_failure_NN_calls'] == 0
    for key in ('NN_eager_timing', 'NN_graph_timing'):
        if key in evidence:
            record = evidence[key]
            assert len(record['rows']) == record['iterations'] == 10
            assert statistics.mean(row['wall_seconds'] for row in record['rows']) == record['wall_mean']
            assert statistics.median(row['wall_seconds'] for row in record['rows']) == record['wall_median']
    checkpoint = numeric_pt(folder/'execution-v2-no-optimizer-checkpoint.pt')
    assert checkpoint['version'] == 'continuous-primitive-block-typed-dual-memory-fp64-solve-ex-v2'
    assert checkpoint['configuration']['internal_read_dtype'] == 'float64'
    assert checkpoint['configuration']['returned_read_dtype'] == 'state-float32'
    state = checkpoint['state_dict']
    assert len(state) == 20 and 'dual_identity' not in state
    digest = hashlib.sha256()
    for name, tensor in state.items():
        assert tensor['code'] == 'f'
        digest.update(name.encode())
        digest.update(str(tensor['shape']).encode())
        digest.update(struct.pack('<'+str(len(tensor['values']))+'f', *tensor['values']))
    old_cpu = json.loads((ROOT/'runs/parameter-typed-dual-memory-cpu-v1/check/result.json').read_text())
    assert digest.hexdigest() == old_cpu['actual_theta_sha256']
    assert result['optimizer_instantiated'] is False and result['SGD_updates'] == 0 and result['shared_buffer_bytes'] == 512
    admission = eager['fixed_tolerances_pass'] and replay is not None and replay['fixed_tolerances_pass']
    assert result['numerical_GPU_admission'] == admission
    cold = json.loads((ROOT/(RUN+'-coordinator-receipt.json')).read_text())
    proof = dict(status='VERIFIED_SAVED_FP64_SOLVE_EX_EXECUTION_REPAIR', source_sha256=sha(Path(__file__)),
                 result_sha256=sha(folder/'result.json'), fresh_same_policy_CPU_reference=True,
                 CPU_checkpoint_theta_FP32_hash_unchanged=True, old_FP32_attempt_remains_separate_failed_evidence=True,
                 CPU_GPU_eager=eager, CPU_GPU_graph=replay, capture_supported=result['capture']['supported'],
                 numerical_GPU_admission=admission, complete_forward_info_records_checked=16+(11 if replay is not None else 0),
                 complete_forward_info_scalar_checks=(16+(11 if replay is not None else 0))*1024,
                 forward_info_verification_scope='executed source-enforced zero-status JSON records; raw zero info tensors were not serialized; backward infos not exposed',
                 pre_capture_evidence_preserved=True, optimizer_steps=0, SGD_updates=0, NN_executions=0,
                 costs=dict(science_wall_seconds=result['wall_seconds'], CPU_seconds=result['CPU_seconds'], local_cold_wall_seconds=cold['wall_seconds'],
                            CPU_peak_RSS_bytes=result['CPU_peak_RSS_bytes'], NN_eager_FB=14,
                            CPU_hybrid_FB=1, CPU_reload_forward=1, graph_replay_FB=result['actual_graph_replay_FB'],
                            NN_eager_checked_wall_mean=evidence['NN_eager_timing']['wall_mean'],
                            NN_graph_checked_wall_mean=evidence.get('NN_graph_timing', {}).get('wall_mean'),
                            NN_complete_forward_solves=(2+14+result['actual_graph_replay_FB'])*1024,
                            solver_only_FB=14, shared_identity_bytes=512, parameter_bytes=41404, carry_scalars=384,
                            memory_before_capture=evidence['memory_before_capture'], memory_after_graph=evidence.get('memory_after_graph'),
                            full_FLOPs=None, solver_backward_FLOPs=None, exact_private_pool_bytes=None, provider_effective_compute=None),
                 boundary='One synthetic fixed-case execution/capture correspondence result; timings include outside-capture forward-info checks. Not a trained architecture, stability, capability, GPU-learning or efficiency result.',
                 analysis_wall_seconds=time.perf_counter()-wall, analysis_CPU_seconds=time.process_time()-cpu)
    with args.out.open('x') as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(status=proof['status'], out=str(args.out), sha256=sha(args.out), eager=eager, graph=replay,
                         capture_supported=result['capture']['supported'], cold_wall=cold['wall_seconds'])))


if __name__ == '__main__':
    main()
