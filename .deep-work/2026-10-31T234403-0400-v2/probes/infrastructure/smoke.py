import json
import os
from pathlib import Path
import numpy
import scipy
import sympy
import pypdf

assert not any('KEY' in x or 'TOKEN' in x or x in ('HOME', 'CODEX_HOME') for x in os.environ)
result = {'python_libraries': {m.__name__: m.__version__ for m in (numpy, scipy, sympy, pypdf)},
          'serialization_round_trip': True, 'scientific_result': False,
          'scope': 'read-only existing libraries, sanitized run-local process'}
assert json.loads(json.dumps(result)) == result
Path('smoke.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
