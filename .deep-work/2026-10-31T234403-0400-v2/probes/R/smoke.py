"""Serialization and executor smoke; must precede substantive probe."""
import json
from pathlib import Path
from algebra_core import normal_form, stack_rules

sample = {"input": ["Da", "Db", "Cb", "Ca", "E"], "expected": ["E"]}
path = Path("smoke-roundtrip.json")
path.write_text(json.dumps(sample, sort_keys=True) + "\n")
restored = json.loads(path.read_text())
assert restored == sample
result, steps = normal_form(restored["input"], stack_rules())
assert list(result) == restored["expected"]
print(json.dumps({"status": "SMOKE_PASS", "roundtrip": True,
                  "normal_form": result, "rewrite_steps": steps}), flush=True)
