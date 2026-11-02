"""Restricted numeric-storage parsing of CPU admission raws, no Torch/NN/SSH.

Only saved simple tensor records are rebuilt as inert Python dictionaries.
Every pickle global is explicitly allowlisted; no model class is imported.
"""
import argparse
import array
import collections
import hashlib
import io
import itertools
import json
import math
import pickle
import struct
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = 'parameter-typed-dual-memory-cpu-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_record(storage, offset, shape, stride, requires_grad, hooks, *extra):
    data, code = storage
    indices = itertools.product(*(range(size) for size in shape)) if shape else [()]
    numbers = [data[offset+sum(index*step for index, step in zip(position, stride))] for position in indices]
    return dict(shape=tuple(shape), values=numbers, code=code)


def numeric_pt(path):
    with zipfile.ZipFile(path) as archive:
        leaf = next(name for name in archive.namelist() if name.endswith('/data.pkl'))
        prefix = leaf[:-len('data.pkl')]
        class Restricted(pickle.Unpickler):
            def find_class(self, module, name):
                if (module, name) == ('torch._utils', '_rebuild_tensor_v2'):
                    return tensor_record
                if (module, name) == ('collections', 'OrderedDict'):
                    return collections.OrderedDict
                if module == 'torch' and name in ('FloatStorage', 'DoubleStorage', 'LongStorage'):
                    return {'FloatStorage': 'f', 'DoubleStorage': 'd', 'LongStorage': 'q'}[name]
                raise ValueError('unapproved pickle global: '+module+'.'+name)
            def persistent_load(self, identifier):
                kind, code, key, location, count = identifier
                assert kind == 'storage' and location == 'cpu'
                result = array.array(code)
                result.frombytes(archive.read(prefix+'data/'+key))
                if sys.byteorder != 'little':
                    result.byteswap()
                assert len(result) == count
                return result, code
        return Restricted(io.BytesIO(archive.read(leaf))).load()


def norm(numbers):
    return math.sqrt(math.fsum(x*x for x in numbers))


def solve(matrix, rhs):
    size = len(rhs)
    augmented = [list(row)+[value] for row, value in zip(matrix, rhs)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        assert abs(scale) > 1e-15
        augmented[column] = [value/scale for value in augmented[column]]
        for row in range(size):
            if row != column:
                amount = augmented[row][column]
                augmented[row] = [value-amount*other for value, other in zip(augmented[row], augmented[column])]
    return [row[-1] for row in augmented]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('preserve prior proof')
    wall, cpu = time.perf_counter(), time.process_time()
    folder = ROOT/'runs'/RUN/'check'
    result = json.loads((folder/'result.json').read_text())
    assert result['status'] == 'CPU_TYPED_DUAL_MEMORY_ADMISSION_PASS'
    config = json.loads((ROOT/'versions'/('code-'+RUN)/'launch-config.json').read_text())
    for name, digest in config['source_sha256'].items():
        assert sha(ROOT/'versions'/('code-'+RUN)/name) == digest
    data = {}
    for name in ('algebra', 'read_gradients', 'NN_gradients', 'reload'):
        receipt = result['checks'][name]['raw']
        path = folder/receipt['file']
        assert sha(path) == receipt['sha256'] and path.stat().st_size == receipt['bytes']
        data[name] = numeric_pt(path)
    nn = data['NN_gradients']
    assert len(nn['eager_gradients']) == 20 and nn['eager_gradients'].keys() == nn['masked_gradients'].keys()
    max_gradient = max_output = 0.
    gradient_norms = {}
    scalar_pairs = 0
    pairs = [('output', nn['eager_output'], nn['masked_output']), ('loss', nn['eager_loss'], nn['masked_loss'])]
    pairs += [(name, left, nn['masked_gradients'][name]) for name, left in nn['eager_gradients'].items()]
    for name, left, right in pairs:
        assert left is not None and right is not None and left['shape'] == right['shape']
        difference = max(abs(x-y) for x, y in zip(left['values'], right['values']))
        rtol = 2e-5 if name in ('output', 'loss') else 2e-4
        assert all(math.isfinite(x) and math.isfinite(y) and abs(x-y) <= 2e-5+rtol*abs(y) for x, y in zip(left['values'], right['values']))
        scalar_pairs += len(left['values'])
        if name in ('output', 'loss'):
            max_output = max(max_output, difference)
        else:
            max_gradient = max(max_gradient, difference)
            gradient_norms[name] = norm(left['values'])
    restored = data['reload']
    state = restored['block_checkpoint']['state_dict']
    assert len(state) == 20 and 'dual_identity' not in state and 'key_composition' not in state
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(str(tensor['shape']).encode())
        digest.update(struct.pack('<'+str(len(tensor['values']))+tensor['code'], *tensor['values']))
    assert digest.hexdigest() == result['actual_theta_sha256']
    carry = restored['episode_carry']
    assert 'coverage' not in carry and carry['key_composition']['shape'] == (2, 8, 8)
    assert sum(len(tensor['values']) for tensor in carry.values()) == 2*384
    algebra = data['algebra']
    # Alternate plain-Python pivoted elimination, independent of Torch's solve
    # and the producer's SVD ridge reference.
    ridge = struct.unpack('<f', struct.pack('<f', 1e-4))[0]
    max_dual = max_read = max_equation = 0.
    for batch in range(2):
        C = algebra['C']['values'][batch*64:(batch+1)*64]
        V = algebra['V']['values'][batch*256:(batch+1)*256]
        query = algebra['keys']['values'][batch*8:(batch+1)*8]
        coverage = [math.fsum(C[row*8:row*8+8]) for row in range(8)]
        A = [[C[row*8+column]/max(coverage[row], 1e-4) for column in range(8)] for row in range(8)]
        Vbar = [[V[row*32+column]/max(coverage[row], 1e-4) for column in range(32)] for row in range(8)]
        gram = [[math.fsum(x*y for x, y in zip(A[row], A[column]))+(ridge if row == column else 0) for column in range(8)] for row in range(8)]
        rhs = [math.fsum(x*y for x, y in zip(row, query)) for row in A]
        b = solve(gram, rhs)
        actual_b = algebra['b']['values'][batch*8:(batch+1)*8]
        read = [math.fsum(b[row]*Vbar[row][column] for row in range(8)) for column in range(32)]
        actual_read = algebra['read']['values'][batch*32:(batch+1)*32]
        max_dual = max(max_dual, max(abs(x-y) for x, y in zip(b, actual_b)))
        max_read = max(max_read, max(abs(x-y) for x, y in zip(read, actual_read)))
        max_equation = max(max_equation, norm([math.fsum(x*y for x, y in zip(row, actual_b))-value for row, value in zip(gram, rhs)]))
        assert max_dual < 2e-7 and max_read < 2e-7
        assert norm(actual_b) <= norm(query)/(2*math.sqrt(1e-4))*(1+2e-6)
    grad_fixtures = {}
    for name, fixture in data['read_gradients'].items():
        C, V = fixture['C']['values'], fixture['V']['values']
        required_H = 0.
        for cell in range(16):
            z = math.fsum(C[cell*8:cell*8+8])
            value_norm = norm(V[cell*32:cell*32+32])
            if z:
                required_H = max(required_H, value_norm/z)
            else:
                assert value_norm == 0
        gradients = fixture['gradients']
        assert all(math.isfinite(x) for tensor in gradients for x in tensor['values'])
        grad_fixtures[name] = dict(gradient_norms={key: norm(tensor['values']) for key, tensor in zip(('V', 'C', 'query'), gradients)},
                                   minimum_uniform_history_H_needed_for_norm_inequality=required_H)
    cold = json.loads((ROOT/(RUN+'-coordinator-receipt.json')).read_text())
    proof = dict(status='PASS_INDEPENDENT_SAVED_DUAL_CPU_RAW', NN_executions=0, SGD_updates=0,
                 source_sha256=sha(Path(__file__)), result_sha256=sha(folder/'result.json'),
                 frozen_source_map_verified=True, raw_artifact_hashes_verified=True,
                 numeric_tensor_parser='restricted allowlist, inert Python records; no Torch/NN/model deserialization',
                 all20_gradient_pairs_finite_and_fixed_tolerance=True, scalar_pairs=scalar_pairs,
                 max_gradient_abs_difference=max_gradient, max_output_or_loss_abs_difference=max_output,
                 full_NN_synthetic_parameter_gradient_norms=gradient_norms,
                 saved_CPU_theta_hash_exact=True, C_episode_carry_and_identity_absent_from_theta=True,
                 alternate_Gaussian_ridge_b_max_abs_difference=max_dual, alternate_read_max_abs_difference=max_read,
                 ridge_normal_equation_residual=max_equation, effective_ridge_for_promoted_double_fixture=ridge,
                 gradient_fixtures=grad_fixtures,
                 caution='finite != stable. Tiny fixture fixes V~1e-3 while C~1e-10; it explores off-formation directions. No global h bound or recurrent/multikey training stability is established.',
                 one_write_check='For fixed h,k,q=k and one write from zero: C=d_raw k^T,V=d_raw h^T; normalized d_j=min(beta*k_j/eps,1),s=||d||^2||k||^2,r=s/(s+lambda)h. For beta=sigmoid(g), away clamp kinks 0<=df/dg<=2f(1-f)(1-beta)<=1/2; ||dr/dg||<=||h||/2. Does not bound coupled h/k, multikey histories, or the whole NN.',
                 actual_admission_costs=dict(science_wall_seconds=result['costs']['wall_seconds'], CPU_seconds=result['costs']['CPU_seconds'],
                                            local_cold_wall_seconds=cold['wall_seconds'], CPU_peak_RSS_bytes=result['costs']['CPU_peak_RSS_bytes'],
                                            P=result['P'], parameter_bytes=result['costs']['model_parameter_bytes'], carry_bytes_per_example=1536,
                                            identity_bytes=256, live_optimizer_bytes=0, full_FLOPs=None, provider_effective_compute=None),
                 analysis_wall_seconds=time.perf_counter()-wall, analysis_CPU_seconds=time.process_time()-cpu)
    with args.out.open('x') as stream:
        json.dump(proof, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(status=proof['status'], out=str(args.out), sha256=sha(args.out),
                         max_gradient_difference=max_gradient, alternate_read_difference=max_read,
                         gradient_fixtures=grad_fixtures, analysis_wall_seconds=proof['analysis_wall_seconds'])))


if __name__ == '__main__':
    main()
