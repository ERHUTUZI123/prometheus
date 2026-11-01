"""Execute inspected run-local Python: no network/credential env, bounded process.

The existing numerical environment is read-only; no old research result is input.
This is not isolation of agent contexts or hidden evaluation data.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('script', type=Path)
parser.add_argument('--timeout', type=float, default=120)
parser.add_argument('--log', type=Path, required=True)
parser.add_argument('--write-dir', type=Path)
args = parser.parse_args()
run = Path(__file__).resolve().parents[2]
script, log = args.script.resolve(), args.log.resolve()
write_dir = (args.write_dir or script.parent).resolve()
if (run not in script.parents or run not in log.parents
        or (run != write_dir and run not in write_dir.parents)):
    raise SystemExit('script, log and write directory must stay in this run')
if args.timeout <= 0:
    raise SystemExit('positive timeout required')
tmp = write_dir / 'tmp'
tmp.mkdir(parents=True, exist_ok=True)
project = run.parents[1]
python = project / '.deep-work/2026-10-31T215613-0400-campaign/probes/infrastructure/venv/bin/python'
if not python.exists():
    raise SystemExit('existing numerical Python unavailable; no installation attempted')
profile = ('(version 1)(allow default)(deny network*)'
           '(deny file-write* (require-not (subpath ' + json.dumps(str(write_dir)) + ')))')
for name in ('.ssh', '.aws', '.codex', '.agents', '.config'):
    profile += '(deny file-read* (subpath ' + json.dumps('/Users/xiaoyangliu/' + name) + '))'
env = {'PATH': '/usr/bin:/bin', 'TMPDIR': str(tmp), 'PYTHONDONTWRITEBYTECODE': '1',
       'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
cpu = max(1, int(args.timeout))
def limits():
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
command = ['/usr/bin/sandbox-exec', '-p', profile, str(python), '-B', str(script)]
with log.open('x') as output:
    try:
        result = subprocess.run(command, cwd=write_dir, env=env, stdout=output,
                                stderr=subprocess.STDOUT, timeout=args.timeout,
                                preexec_fn=limits)
    except subprocess.TimeoutExpired:
        print('WALL_TIMEOUT', file=output)
        raise SystemExit(124)
raise SystemExit(result.returncode)
