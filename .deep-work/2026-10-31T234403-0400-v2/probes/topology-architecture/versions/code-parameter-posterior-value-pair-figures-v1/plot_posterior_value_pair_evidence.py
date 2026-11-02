"""Source-only scientific figure preparation for ROOT's existing bounded CPU runner.

Consumes six exact saved JSON/JSONL files listed in the companion manifest.
No Torch, NumPy imports, model/checkpoint loading, training, network or compiler.
Matplotlib is imported only to render; do not install dependencies if absent.
"""
import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

PINS = {
    'posterior-result.json': 'a51ec60079c726c176dd2a60725fa8422ac1dcf5277091d7360bc4f677f863f5',
    'posterior-records.jsonl': '71629e4de9b80aed5a0076aa9e50f4e04631e58fb7b4fde31692584e628ed59a',
    'posterior-coordinator-receipt.json': 'cdefa63ef677e80f6ac3ba82ebf379e6a10da0c6e181e5507764f74030e94c89',
    'raw-result.json': '281022a2064b1b07c556161102d3c9d8b33d5481228cbcb1a12c11a62b90e8c4',
    'raw-records.jsonl': '56fb5234b67a726841643059982c4141d756c38ccc0336bdfbe9e4b3e27909a2',
    'raw-coordinator-receipt.json': 'f41b13279242252b60e07bcfc950b776b42688fd9db6359e480b3db5cb2170fd',
}
COLORS = {'posterior': '#2166AC', 'raw': '#B35806'}
LABELS = {'posterior': 'POSTERIOR: h = pE', 'raw': 'RAW: h = candidate'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_exact(folder):
    payload = {}
    for leaf, digest in PINS.items():
        path = folder / leaf
        if sha(path) != digest:
            raise ValueError('Frozen input SHA differs: ' + leaf)
        content = path.read_text()
        payload[leaf] = ([json.loads(row) for row in content.splitlines()]
                         if leaf.endswith('.jsonl') else json.loads(content))
    data = {}
    for mode in ('posterior', 'raw'):
        result = payload[mode + '-result.json']
        records = payload[mode + '-records.jsonl']
        receipt = payload[mode + '-coordinator-receipt.json']
        assert result['status'] == 'ACTUAL_FOUR_STAGE_POSTERIOR_VALUE_GRAPH_COMPLETE'
        assert result['actual_new_updates_this_process'] == 2400
        assert not result['parent_NN_loaded'] and result['NN_unchanged_after_TEST']
        assert receipt['returncode'] == 0
        updates = [r for r in records if r['kind'] == 'actual_update']
        logged_curves = [r for r in records if r['kind'] == 'actual_VALID_curve']
        assert len(updates) == 2400 and len(logged_curves) == 28
        for stage, report in enumerate(result['reports']):
            phase = [r for r in updates if r['stage'] == stage]
            assert [r['step'] for r in phase] == list(range(1, 601))
            assert all(math.isfinite(r['terminal_NLL']) for r in phase)
            assert sum(r['training_wall_seconds'] for r in phase) == report['costs']['training_wall_seconds']
            assert [r['step'] for r in report['curves']] == list(range(0, 601, 100))
            for curve in report['curves']:
                match = [r for r in logged_curves if r['stage'] == stage and r['step'] == curve['step']]
                assert len(match) == 1 and match[0]['valid'] == curve['valid']
        resources = result['resources']
        assert resources['parameters'] == 13397
        data[mode] = dict(
            reports=result['reports'], tests=result['tests'],
            costs=dict(P=resources['parameters'], NN_bytes=resources['parameter_bytes'],
                       Adam_bytes=resources['optimizer_tensor_bytes'], replay_bytes=resources['replay_bytes'],
                       update_wall_seconds=sum(r['costs']['training_wall_seconds'] for r in result['reports']),
                       science_wall_seconds=result['driver_wall_seconds'],
                       local_coordinator_wall_seconds=receipt['wall_seconds'],
                       full_training_FLOPs=None, provider_effective_compute=None),
            source_sha256=result['source_sha256'])
    assert data['posterior']['source_sha256'] == data['raw']['source_sha256']
    return data


def render(data, out):
    # Existing ROOT-provided analysis dependencies; no installation or network.
    sys.path.insert(0, '/opt/prometheus-analysis-python')
    os.environ['MPLCONFIGDIR'] = str(out / '.matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
                         'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.5))
    a, b, c, d = axes.flat
    plotted = {'current_VALID': {}, 'old_VALID_during_stage3': {}, 'final_TEST': {}, 'costs': {}}
    for mode, values in data.items():
        col = COLORS[mode]
        plotted['current_VALID'][mode] = []
        for stage, report in enumerate(values['reports']):
            x = [600 * stage + row['step'] for row in report['curves']]
            y = [row['valid'][str(stage)]['soft']['accuracy'] for row in report['curves']]
            a.plot(x, y, color=col, marker='o', markersize=3,
                   label=LABELS[mode] if stage == 0 else None)
            plotted['current_VALID'][mode].append({'stage': stage, 'global_updates': x, 'accuracy': y})
        for task, style in enumerate(('-', '--', ':')):
            rows = values['reports'][3]['curves']
            x = [r['step'] for r in rows]
            y = [r['valid'][str(task)]['soft']['accuracy'] for r in rows]
            b.plot(x, y, color=col, linestyle=style, marker='o', markersize=2.5)
            plotted['old_VALID_during_stage3'].setdefault(mode, {})[str(task)] = {'updates': x, 'accuracy': y}
    a.set(title='a  Current-task VALID acquisition', xlabel='Cumulative optimizer updates', ylabel='Soft accuracy')
    for boundary in (600, 1200, 1800):
        a.axvline(boundary, color='.82', lw=.8)
    for stage, label in enumerate(('old0', 'old1', 'old2', 'new binary')):
        a.text(600 * stage + 300, 1.035, label, ha='center', va='bottom', fontsize=9)
    a.set_ylim(-.025, 1.08)
    a.set_xticks((0, 600, 1200, 1800, 2400))
    a.legend(loc='lower left')
    b.set(title='b  Old-task VALID while learning stage3', xlabel='Stage3 optimizer updates', ylabel='Soft accuracy', ylim=(-.025, 1.025))
    b.legend(handles=[Line2D([0], [0], color=COLORS[m], label=LABELS[m]) for m in ('posterior', 'raw')] +
                     [Line2D([0], [0], color='.25', linestyle=s, label='old' + str(t))
                      for t, s in enumerate(('-', '--', ':'))], loc='lower left', ncol=2)
    labels, xs = [], []
    for task in range(4):
        for split in ('test_iid', 'test_long'):
            labels.append(('old' + str(task) if task < 3 else 'new') + '\n' + ('IID' if split == 'test_iid' else 'long'))
            xs.append((task, split))
    for index, mode in enumerate(('posterior', 'raw')):
        heights = [data[mode]['tests'][str(t)][s]['soft']['accuracy'] for t, s in xs]
        positions = [j + (index - .5) * .36 for j in range(8)]
        c.bar(positions, heights, width=.34, color=COLORS[mode], label=LABELS[mode])
        plotted['final_TEST'][mode] = [{'task': t, 'split': s, 'accuracy': h} for (t, s), h in zip(xs, heights)]
    c.set(title='c  Frozen final TEST: IID and long', ylabel='Soft accuracy', ylim=(0, 1.03))
    c.set_xticks(range(8), labels)
    c.legend(loc='upper right')
    cost_fields = ('update_wall_seconds', 'science_wall_seconds', 'local_coordinator_wall_seconds')
    cost_styles = ('', '//', '..')
    for index, mode in enumerate(('posterior', 'raw')):
        costs = data[mode]['costs']
        plotted['costs'][mode] = costs
        for j, (key, hatch) in enumerate(zip(cost_fields, cost_styles)):
            x = index + (j - 1) * .24
            height = costs[key]
            d.bar(x, height, width=.22, color=COLORS[mode], hatch=hatch, edgecolor='white', linewidth=.4)
            d.text(x, height + 1, f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    d.set(title='d  Parameters and nested measured wall scopes', ylabel='Seconds (nested scopes; do not sum)', ylim=(0, 78))
    d.set_xticks((0, 1), ('POSTERIOR\n13,397 parameters', 'RAW\n13,397 parameters'))
    d.legend(handles=[plt.Rectangle((0, 0), 1, 1, facecolor='.6', hatch=h, label=l)
                      for h, l in zip(cost_styles, ('2400 updates', 'Whole science', 'Local coordinator'))],
             loc='upper left', ncol=1)
    for ax in (a, b, c, d):
        ax.grid(axis='y', color='.9', lw=.6)
        ax.set_axisbelow(True)
    fig.suptitle('Posterior value projection: acquisition, old-task behavior and complete-cost scopes', fontsize=14)
    fig.text(.5, .029, 'Single development seed0; no Primary or architecture-advantage claim. Old long TEST is not pure forgetting.\n'
             'All four learning stages frozen before TEST. Common CPU admission 10.647296s is additional; NN/Adam 53,588/107,284B per arm.\n'
             'Replay 55,296B plus metadata. Full training FLOPs, driver/context memory and provider-effective compute are unknown.',
             ha='center', va='bottom', fontsize=8.5)
    fig.subplots_adjust(left=.07, right=.985, top=.9, bottom=.16, hspace=.38, wspace=.24)
    for extension in ('pdf', 'svg', 'png'):
        fig.savefig(out / ('posterior-value-pair-evidence-v1.' + extension), dpi=180)
    plt.close(fig)
    return plotted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    started, cpu = time.perf_counter(), time.process_time()
    # Refuse to overwrite a prior rendered figure/result.
    args.out.mkdir(parents=True, exist_ok=False)
    data = load_exact(args.inputs)
    plotted = render(data, args.out)
    record = dict(status='RENDERED_FROZEN_POSTERIOR_RAW_EVIDENCE_FIGURE', source_sha256=sha(Path(__file__)),
                  input_sha256=PINS, plotted=plotted, SGD_updates=0, NN_queries=0,
                  figure_CPU_seconds=time.process_time() - cpu, figure_wall_seconds=time.perf_counter() - started,
                  figure_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)),
                  full_FLOPs=None, provider_effective_compute=None,
                  limits='Saved producer TEST aggregates: original pair has no per-example TEST payload. VALID curves equal JSONL. '
                         'Single development seed; no Primary/advantage. Historical training and diagnostic costs remain separately charged. '
                         'Graph setup, replay, validation, TEST, serialization included in science/local scopes; nested times not additive.')
    record['artifacts'] = {f.name: dict(bytes=f.stat().st_size, sha256=sha(f)) for f in sorted(args.out.iterdir()) if f.is_file()}
    cache = args.out / '.matplotlib-cache'
    cachefiles = [f for f in cache.rglob('*') if f.is_file()] if cache.exists() else []
    record['matplotlib_cache'] = dict(files=len(cachefiles), bytes=sum(f.stat().st_size for f in cachefiles),
                                      scope='render-only per-run cache payload, separate from figure artifacts')
    (args.out / 'figure-result.json').write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'status': record['status'], 'out': str(args.out), 'NN_queries': 0, 'SGD_updates': 0}))


if __name__ == '__main__':
    main()
