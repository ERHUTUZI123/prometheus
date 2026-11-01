"""Serialization smoke before the R v2 mechanism probe."""
import json
from pathlib import Path
from v2_core import Span, IDENTITY, ZERO, unit, mul

data = {"matrix": [int(x) for x in unit(0, 1)], "metadata": "R_v2_smoke"}
path = Path("v2-smoke-roundtrip.json")
path.write_text(json.dumps(data, sort_keys=True) + "\n")
assert json.loads(path.read_text()) == data
space = Span()
for value in (IDENTITY, unit(0, 1), unit(1, 2), unit(0, 2)):
    space.append(value)
for value in (IDENTITY, ZERO, unit(0, 2)):
    coordinates = space.coordinates(value)
    assert coordinates is not None
    assert space.reconstruct(coordinates) == value
assert mul(unit(0, 1), unit(1, 2)) == unit(0, 2)
print(json.dumps({"status": "SMOKE_PASS", "serialization_roundtrip": True,
                  "exact_basis_roundtrip": True}), flush=True)
