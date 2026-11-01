import json
from pathlib import Path

sample = {'state': [0, 1, 2], 'mapping': {'0': [1, 0]}, 'finite': True}
target = Path('serialization-smoke.json')
target.write_text(json.dumps(sample, sort_keys=True) + '\n')
recovered = json.loads(target.read_text())
assert recovered == sample
print(json.dumps({'test': 'json_roundtrip', 'passed': True}, sort_keys=True), flush=True)
