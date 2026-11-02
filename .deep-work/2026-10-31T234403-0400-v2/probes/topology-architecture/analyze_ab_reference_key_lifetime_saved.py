"""Public-token KEY-lifetime analysis of saved frozen AB reference arrays.

Standard library only: no NN, checkpoint, target/truth, Torch, NumPy or SSH.
Attention is reconstructed from captured per-slot w divided by sum(w).
"""
import argparse
import array
import ast
import hashlib
import json
import math
import statistics
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved(path, keys):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for key in keys:
            raw = archive.read(key+'.npy')
            assert raw[:6] == b'\x93NUMPY'
            offset = 10 if raw[6] == 1 else 12
            size = int.from_bytes(raw[8:offset], 'little')
            header = ast.literal_eval(raw[offset:offset+size].decode())
            assert not header['fortran_order']
            code = {'<f4': 'f', '<f8': 'd', '<i8': 'q', '|b1': 'B'}[header['descr']]
            data = array.array(code)
            data.frombytes(raw[offset+size:])
            if sys.byteorder != 'little' and data.itemsize > 1:
                data.byteswap()
            assert len(data) == math.prod(header['shape'])
            result[key] = dict(shape=header['shape'], values=data)
    return result


def stats(numbers):
    if not numbers:
        return dict(count=0, mean=None, median=None, minimum=None, maximum=None)
    return dict(count=len(numbers), mean=math.fsum(numbers)/len(numbers),
                median=statistics.median(numbers), minimum=min(numbers), maximum=max(numbers))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('preserve saved analysis')
    began, cpu = time.perf_counter(), time.process_time()
    run = 'parameter-ab-write-direction-controls-cpu-v1'
    folder = ROOT/'runs'/run/'audit'
    result = json.loads((folder/'result.json').read_text())
    receipt = next(row for row in result['records'] if row['condition'] == 'reference')
    raw_path = folder/receipt['file']
    assert sha(raw_path) == receipt['sha256']
    raw = saved(raw_path, ('w', 'h', 'active', 'action_probabilities', 'action_indices', 'read', 'final_h', 'initial', 'lengths'))
    config = json.loads((ROOT/'versions'/('code-'+run)/'launch-config.json').read_text())
    spec = config['input_files']['valid.npz']
    public_path = ROOT/'runs'/spec['remote_relative']
    assert sha(public_path) == spec['sha256']
    public = saved(public_path, ('tokens', 'initial', 'lengths'))
    assert raw['initial']['values'] == public['initial']['values'] and raw['lengths']['values'] == public['lengths']['values']
    length = raw['active']['shape'][1]
    metadata, lookup = [], {}
    for row, extent in enumerate(public['lengths']['values']):
        word = list(public['tokens']['values'][row*128:row*128+extent])
        writes = [i for i, token in enumerate(word) if token == 9]
        assert len(writes) == 2 and word[0] in (7, 8) and word[writes[0]+1] in (7, 8)
        assert {word[0], word[writes[0]+1]} == {7, 8}
        assert word[1:3] == [14, 1]
        assert all(token == 3 for token in word[3:writes[0]])
        second = writes[0]+1
        assert word[second+1:second+3] == [14, 1]
        assert all(token == 3 for token in word[second+3:writes[1]])
        masked = word.copy()
        masked[0] = masked[second] = -1
        key = (public['initial']['values'][row], tuple(masked))
        lookup.setdefault(key, []).append(row)
        metadata.append(dict(word=word, first_value=writes[0]-3, second_value=writes[1]-second-3,
                             write_first=writes[0], write_second=writes[1]))
    pairs = []
    for group in lookup.values():
        assert len(group) == 2
        a, b = sorted(group, key=lambda row: metadata[row]['word'][0])
        assert metadata[a]['word'][0] == 7 and metadata[b]['word'][0] == 8
        left, right = metadata[a], metadata[b]
        assert left['first_value'] == right['first_value'] and left['second_value'] == right['second_value']
        differences = [i for i, (x, y) in enumerate(zip(left['word'], right['word'])) if x != y]
        assert differences == [0, left['write_first']+1]
        pairs.append((a, b))
    assert len(pairs) == 256
    zero_weight = 0
    max_store_mass_error = 0.
    minimum_weight_sum = math.inf
    def attention(row, t):
        nonlocal zero_weight, max_store_mass_error, minimum_weight_sum
        assert raw['active']['values'][row*length+t]
        weights = raw['w']['values'][(row*length+t)*8:(row*length+t+1)*8]
        denominator = math.fsum(weights)
        minimum_weight_sum = min(minimum_weight_sum, denominator)
        max_store_mass_error = max(max_store_mass_error, abs(denominator-raw['action_probabilities']['values'][(row*length+t)*7+5]))
        if denominator == 0:
            zero_weight += 1
            return None
        return [x/denominator for x in weights]
    def tv(a, b, t):
        left, right = attention(a, t), attention(b, t)
        return None if left is None or right is None else .5*math.fsum(abs(x-y) for x, y in zip(left, right))
    times = {'after_KEY': lambda v: 1, 'after_CLEAR': lambda v: 2,
             'after_SET0': lambda v: 3, 'before_first_WRITE': lambda v: 3+v}
    for step in range(1, 8):
        times['after_STEP'+str(step)] = lambda v, step=step: 3+step if v >= step else None
    all_attention, by_value = {}, {}
    for value in range(8):
        subset = [(a, b) for a, b in pairs if metadata[a]['first_value'] == value]
        assert len(subset) == 32
        by_value[str(value)] = {}
        for name, position in times.items():
            t = position(value)
            samples = [] if t is None else [tv(a, b, t) for a, b in subset]
            samples = [x for x in samples if x is not None]
            by_value[str(value)][name] = stats(samples)
    for name, position in times.items():
        samples = []
        for a, b in pairs:
            t = position(metadata[a]['first_value'])
            if t is not None:
                measured = tv(a, b, t)
                if measured is not None:
                    samples.append(measured)
        all_attention[name] = stats(samples)
    def event_statistics(events):
        masses = {key: [] for key in ('CMP', 'LOAD', 'STORE', 'ADDRESS')}
        action_counts = [0]*7
        hdelta, readnorm, hnorm = [], [], []
        for row, t in events:
            base = (row*length+t)*7
            alpha = raw['action_probabilities']['values'][base:base+7]
            chosen = raw['action_indices']['values'][row*length+t]
            assert chosen == max(range(7), key=alpha.__getitem__)
            action_counts[chosen] += 1
            masses['CMP'].append(math.fsum(alpha[:4]))
            for index, key in ((4, 'LOAD'), (5, 'STORE'), (6, 'ADDRESS')):
                masses[key].append(alpha[index])
            base = (row*length+t)*32
            before = raw['h']['values'][base:base+32]
            after = raw['h']['values'][base+32:base+64]
            assert t+1 < public['lengths']['values'][row]
            hdelta.append(math.sqrt(math.fsum((x-y)**2 for x, y in zip(after, before))))
            hnorm.append(math.sqrt(math.fsum(x*x for x in before)))
            readnorm.append(math.sqrt(math.fsum(x*x for x in raw['read']['values'][base:base+32])))
        return dict(events=len(events), action_argmax_counts=action_counts,
                    action_mass={key: stats(data) for key, data in masses.items()},
                    OLDh_norm=stats(hnorm), executed_h_delta_L2=stats(hdelta), read_norm=stats(readnorm))
    classes = {
        'first_KEY': lambda row: [0], 'first_CLEAR': lambda row: [1], 'first_SET0': lambda row: [2],
        'first_STEP': lambda row: [3] if metadata[row]['first_value'] else [],
        'all_first_value_STEPs': lambda row: list(range(3, 3+metadata[row]['first_value'])),
        'first_public_WRITE': lambda row: [metadata[row]['write_first']],
    }
    events_all, events_stratified = {}, {}
    for name, positions in classes.items():
        events_all[name] = event_statistics([(row, t) for row in range(512) for t in positions(row)])
        events_stratified[name] = {str(value): event_statistics([(row, t) for row in range(512) if metadata[row]['first_value'] == value for t in positions(row)]) for value in range(8)}
    output = dict(status='POSTHOC_AB_REFERENCE_KEY_LIFETIME_TOKEN_ANALYSIS',
                  source_sha256=sha(Path(__file__)), reference_raw_sha256=sha(raw_path),
                  original_public_VALID_sha256=sha(public_path), NN_forwards=0, SGD_updates=0,
                  targets_loaded=False, teacher_truth_loaded=False,
                  pairing=dict(pairs=256, rows=512, pairs_per_first_value=32,
                               rule='same initial/full active word with only two write-KEY positions masked; temporal STEP counts/noise/suffix fixed',
                               pair_order_sha256=hashlib.sha256(json.dumps(pairs).encode()).hexdigest(),
                               first_prefix_differences='ONLY first KEY before first WRITE'),
                  action_order=['compute0', 'compute1', 'compute2', 'compute3', 'LOAD', 'STORE', 'ADDRESS'],
                  attention_scope='captured soft w / sum(w); incoming attention at t+1 observes address after token t; no model rerun',
                  zero_weight_sum_attention_observations=zero_weight, minimum_observed_weight_sum=minimum_weight_sum,
                  max_reconstructed_store_mass_error=max_store_mass_error,
                  attention_TV_all_pairs=all_attention, attention_TV_by_first_value=by_value,
                  token_events_all_rows=events_all, token_events_by_first_value=events_stratified,
                  wall_seconds=time.perf_counter()-began, CPU_seconds=time.process_time()-cpu,
                  full_FLOPs=None, provider_effective_compute=None,
                  scope='descriptive frozen reference trajectory; no HOLD intervention/cause, no binding or learning-rule claim')
    with args.out.open('x') as stream:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps(dict(status=output['status'], out=str(args.out), sha256=sha(args.out),
                         attention={name: item['mean'] for name, item in all_attention.items()},
                         event_actions={name: {'alpha_ADDRESS': item['action_mass']['ADDRESS']['mean'], 'alpha_CMP': item['action_mass']['CMP']['mean'], 'alpha_LOAD': item['action_mass']['LOAD']['mean'], 'argmax': item['action_argmax_counts'], 'h_delta': item['executed_h_delta_L2']['mean'], 'read_norm': item['read_norm']['mean']} for name, item in events_all.items()},
                         wall_seconds=output['wall_seconds'])))


if __name__ == '__main__':
    main()
