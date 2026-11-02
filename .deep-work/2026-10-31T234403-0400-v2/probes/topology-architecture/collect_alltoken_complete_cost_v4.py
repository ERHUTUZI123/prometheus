"""Read-only raw-result reconciliation; no model execution/config selection."""
import datetime, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = [
    'parameter-char-alltoken-dynamic-300-v1',
    'parameter-char-alltoken-static-300-v1',
    'parameter-char-alltoken-detached-300-v1',
    'parameter-char-alltoken-transformer-300-v1',
    'parameter-char-alltoken-gru-300-v1',
    'parameter-char-alltoken-mamba2-300-v1',
    'parameter-char-alltoken-factorized-300-v1',
    'parameter-char-alltoken-gru34-300-v1',
    'parameter-char-alltoken-transformer24-300-v1',
]
META = 'f157d58e9246f59f78dbd883b4ccc80284278b9eb270365cb13556dc8703e792'
BATCH = '538c1157cacea82a51bf5be950d451768f013ee7178f5aabc100e339b966f615'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    previous_path = ROOT / 'alltoken-parameter-complete-cost-evidence-v3.json'
    previous = json.loads(previous_path.read_text())
    historical = {r['run']: r for r in previous['rows']}
    rows = []
    for name in RUNS:
        folder = ROOT / 'runs' / name / 'learn'
        source = folder / 'summary.json'
        value = json.loads(source.read_text())
        assert value['completed_updates'] == 300
        assert value['metadata_sha256'] == META
        assert value['batch_indices_sha256'] == BATCH
        costs, resources = value['costs'], value['resources']
        assert costs['training_label_observations'] == 614400
        assert costs['optimizer_updates'] == 300
        assert resources['parameter_bytes'] == resources['parameters'] * 4
        tests = {}
        for length in (64, 128, 256):
            modes = value['final']['test' + str(length)]
            assert modes['soft']['last64_paired_observations'] == 32768
            tests[str(length)] = modes
        # Old compile/priming evidence is retained exactly, not counted as zero
        # for a method whose summary reports only the subsequent strict stage.
        old = historical.get(name, {})
        cold = {k: v for k, v in old.items()
                if k.startswith('cold_') or k in ('compilation', 'whole_launch_SSH_coordinator_seconds')}
        rows.append(dict(
            run=name, summary_sha256=sha(source), source_sha256=value['source_sha256'],
            parameters=resources['parameters'], resources=resources,
            actual_updates=300, training_label_observations=614400,
            paired_last64_PPL={str(l): tests[str(l)]['soft']['last64_paired_PPL']
                               for l in (64, 128, 256)},
            VALID_PPL=value['final']['VALID']['soft']['all_token_PPL'],
            costs=costs, driver_wall_seconds=value['driver_wall_seconds'],
            tests=tests, checkpoints=value['checkpoints'],
            reload_audit=value['reload_audit'], cold_setup_evidence=cold,
            configuration=value['model_configuration'],
            resource_overlap_note=value.get('resource_overlap_note'),
            formal_Primary=False, full_training_FLOPs=None, hardware_energy_joules=None,
            cached_generation_latency=None,
        ))
    output = dict(
        schema='alltoken-joint-parameter-and-complete-cost-v4',
        generated_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        previous_ledger_sha256=sha(previous_path),
        rows=rows, primary_frozen=False,
        interpretation='Nine fixed development configurations, not pooled seeds. '
          'Small/large widths are separately named runs. Baseline validity has no accuracy gate. '
          'P-matched GRU34/TF24 chosen before model execution from the parameter formula; '
          'not selected by TEST. No broad advantage established.',
        cost_semantics='All raw time scopes retained; nested times are not additive. '
          'Forward MAC is not full training FLOPs. Cold Mamba compilation/cache is separate '
          'from strict training. Prior concurrency and CPU admission/setup costs remain in '
          'their own original receipts. Unknown fields are not zero. '
          'Evaluation modes differ (Block soft+native vs baselines soft); compare matching '
          'per-mode work, not whole drivers as equal-work latency. '
          'No replay/maintenance advantage inferred from a single-phase language run.',
    )
    target = ROOT / 'alltoken-parameter-complete-cost-evidence-v4.json'
    with target.open('x') as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(rows=len(rows), file=str(target), sha256=sha(target))))


if __name__ == '__main__':
    main()
