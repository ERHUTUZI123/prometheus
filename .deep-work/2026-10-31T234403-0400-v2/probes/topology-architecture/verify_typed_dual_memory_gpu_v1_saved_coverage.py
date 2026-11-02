"""Independent standard-library verification of retained FP32 GPU failure.

No NN, Torch, NumPy, optimizer or SSH. Failure is never reclassified as pass.
"""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

from verify_factorized_graph600_saved import npz
from verify_typed_dual_memory_cpu_saved_coverage import numeric_pt, sha

ROOT = Path(__file__).resolve().parent
RUN = 'parameter-typed-dual-memory-gpu-execution-v1a'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('preserve failed-attempt verification')
    wall, cpu = time.perf_counter(), time.process_time()
    folder = ROOT/'runs'/RUN/'admit'
    result = json.loads((folder/'result.json').read_text())
    config = json.loads((ROOT/'versions'/('code-'+RUN)/'launch-config.json').read_text())
    for leaf, digest in result['source_sha256'].items():
        assert sha(ROOT/'versions'/('code-'+RUN)/leaf) == digest
    pair = result['evidence']['CPU_GPU_eager']
    path = folder/pair['file']
    assert sha(path) == pair['sha256'] and path.stat().st_size == pair['bytes']
    raw = npz(path)
    names = json.loads(bytes(raw['gradient_names_json']['values']).decode())
    assert len(names) == 20
    baseline_path = ROOT/'runs/parameter-typed-dual-memory-cpu-v1/check/dual-NN-B64-L16-all20-gradients.pt'
    assert sha(baseline_path) == result['CPU_reference_sha256']
    baseline = numeric_pt(baseline_path)
    fields, failed = {}, []
    scalar_pairs = 0
    for key, declared in pair['fields'].items():
        left, right = raw['CPU_'+key], raw['GPU_'+key]
        assert left['shape'] == right['shape']
        if key.startswith('grad_'):
            index = int(key[5:])
            original = baseline['masked_gradients'][names[index]]['values']
        else:
            original = baseline['masked_'+key]['values']
        assert list(left['values']) == list(original)
        atol, rtol = (2e-5, 2e-4) if key.startswith('grad_') else (2e-5, 2e-5)
        assert (declared['atol'], declared['rtol']) == (atol, rtol)
        maximum = max(abs(x-y) for x, y in zip(left['values'], right['values']))
        all_finite = all(math.isfinite(x) and math.isfinite(y) for x, y in zip(left['values'], right['values']))
        failures = sum(abs(x-y) > atol+rtol*abs(y) for x, y in zip(left['values'], right['values']))
        assert all_finite == declared['finite'] and (failures == 0) == declared['allclose'] and maximum == declared['max_abs']
        fields[key] = dict(parameter=names[int(key[5:])] if key.startswith('grad_') else None,
                           finite=all_finite, fixed_tolerance_pass=failures == 0,
                           violating_scalars=failures, scalars=len(left['values']), max_abs=maximum)
        scalar_pairs += len(left['values'])
        if failures:
            failed.append(key)
    assert len([key for key in failed if key.startswith('grad_')]) == 16
    assert result['numerical_GPU_admission'] is False and result['capture']['supported'] is False
    assert result['actual_graph_replay_FB'] == 0 and result['capture']['attempts'] == 1 and result['capture']['post_failure_NN_calls'] == 0
    assert 'torch.linalg.solve' in result['capture']['traceback'] and 'stream is capturing' in result['capture']['traceback']
    assert result['optimizer_instantiated'] is False and result['SGD_updates'] == 0
    before = json.loads((folder/'EAGER_EVIDENCE_BEFORE_CAPTURE.json').read_text())
    assert before == result['evidence']
    capture_file = json.loads((folder/'CAPTURE_FAILURE_CURRENT_SOLVE.json').read_text())
    assert capture_file['exception_chain'] == result['capture']['exception_chain']
    for key in ('NN_eager_timing',):
        timing = before[key]
        assert timing['iterations'] == len(timing['rows']) == 10
        assert statistics.mean(row['wall_seconds'] for row in timing['rows']) == timing['wall_mean']
        assert statistics.median(row['wall_seconds'] for row in timing['rows']) == timing['wall_median']
    solver = before['solver_only']
    assert solver['reference_FB'] == 1 and solver['warmup_FB'] == 3 and solver['timing']['iterations'] == 10
    solver_path = folder/'formed-solver-reference.npz'
    assert sha(solver_path) == solver['reference_raw_sha256']
    formed = npz(solver_path)
    assert all(math.isfinite(x) for tensor in formed.values() for x in tensor['values'])
    G, rhs, solution = (formed[key]['values'] for key in ('Gram', 'rhs', 'solution'))
    residual_max = 0.
    for batch in range(64):
        for row in range(8):
            residual = math.fsum(G[batch*64+row*8+column]*solution[batch*8+column] for column in range(8))-rhs[batch*8+row]
            residual_max = max(residual_max, abs(residual))
    cold = json.loads((ROOT/(RUN+'-coordinator-receipt.json')).read_text())
    proof = dict(status='VERIFIED_RETAINED_FP32_GPU_NUMERICAL_AND_CAPTURE_FAILURE',
                 source_sha256=sha(Path(__file__)), result_sha256=sha(folder/'result.json'),
                 source_and_raw_hashes_verified=True, CPU_masked_reference_payload_exact=True,
                 all_fields=fields, scalar_pairs=scalar_pairs, failed_gradient_tensors=16,
                 output_max_abs=fields['output']['max_abs'], gradient_max_abs=max(value['max_abs'] for key, value in fields.items() if key.startswith('grad_')),
                 all_values_finite=True, numerical_admission_pass=False, graph_supported=False,
                 capture_failure_chain_preserved=True, eager_evidence_identical_before_and_after_capture=True,
                 post_failure_NN_calls=0, graph_replay_FB=0, optimizer_steps=0, SGD_updates=0,
                 formed_solver_normal_equation_max_abs_residual=residual_max,
                 cost=dict(science_wall_seconds=result['wall_seconds'], CPU_seconds=result['CPU_seconds'],
                           local_cold_wall_seconds=cold['wall_seconds'], CPU_peak_RSS_bytes=result['CPU_peak_RSS_bytes'],
                           NN_eager_FB=14, solver_only_FB=14, failed_capture_attempts=1,
                           complete_NN_forward_solves=14*1024, semantic_active_steps_per_FB=486,
                           failed_capture_partial_work='unknown; charged in actual process clocks',
                           NN_eager_wall_mean=before['NN_eager_timing']['wall_mean'],
                           NN_eager_CUDA_stream_elapsed_mean=before['NN_eager_timing']['CUDA_stream_elapsed_mean'],
                           standalone_formed_solver_wall_mean=solver['timing']['wall_mean'],
                           memory_before_capture=before['memory_before_capture'], backend=before['backend'],
                           parameter_bytes=41404, gradient_bytes=41404, identity_bytes=256, input_bytes=result['static_input_bytes'],
                           exact_private_pool_or_driver_bytes=None, full_forward_FLOPs=None, full_training_FLOPs=None,
                           provider_effective_compute=None),
                 boundary='An actual finite-but-nonmatching FP32 GPU execution and capture rejection. No training/capability claim. A later FP64 solve_ex repair needs its own CPU reference and never reclassifies this attempt.',
                 NN_executions=0, analysis_wall_seconds=time.perf_counter()-wall, analysis_CPU_seconds=time.process_time()-cpu)
    with args.out.open('x') as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(status=proof['status'], out=str(args.out), sha256=sha(args.out),
                         failed_gradient_tensors=16, output_max_abs=proof['output_max_abs'], gradient_max_abs=proof['gradient_max_abs'],
                         cold_seconds=cold['wall_seconds'], analysis_wall_seconds=proof['analysis_wall_seconds'])))


if __name__ == '__main__':
    main()
