"""Stdlib-only, read-only statistics from the actual frozen VALID traces."""
import array
import ast
import hashlib
import json
import math
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
RUN = "parameter-memory-live-stage-n8-seed0-dynamic-stage4-v1-valid-memory-path-v2"


def main():
    folder = ROOT / "runs" / RUN / "audit"
    report = json.loads((folder / "result.json").read_text())
    record = report["results"][0]
    assert record["mode"] == "full" and report["checkpoint_and_NN_unchanged"]
    path = folder / record["raw_trace_file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == record["raw_trace_sha256"]
    with zipfile.ZipFile(path) as stored:
        def load(name):
            b = stored.read(name + ".npy")
            assert b[:6] == b"\x93NUMPY" and b[6] in (1, 2, 3)
            end = 10 if b[6] == 1 else 12
            length = int.from_bytes(b[8:end], "little")
            meta = ast.literal_eval(b[end:end + length].decode().strip())
            assert not meta["fortran_order"]
            values = array.array({"<i8": "q", "<f4": "f", "<f8": "d"}[meta["descr"]])
            values.frombytes(b[end + length:])
            assert len(values) == math.prod(meta["shape"])
            return values, meta["shape"]
        addresses, shape = load("write_addresses")
        gates, _ = load("write_gates")
        lengths, _ = load("lengths")
        tokens, token_shape = load("tokens")
        observed = {"WRITE_token9": [], "other_active_tokens": []}
        for row in range(shape[0]):
            for step in range(lengths[row]):
                offset = (row * shape[1] + step) * shape[2]
                weights = addresses[offset:offset + shape[2]]
                assert abs(sum(weights) - 1.) < 1e-5 and min(weights) >= 0
                category = "WRITE_token9" if tokens[row * token_shape[1] + step] == 9 else "other_active_tokens"
                observed[category].append((
                    -sum(v * math.log(v) for v in weights if v > 0),
                    max(weights), gates[row * shape[1] + step]))
    result = dict(schema="frozen-valid-write-address-description-v1", parent=RUN,
        raw_trace_sha256=digest, parameters=report["parameters"],
        observed={key: dict(count=len(rows), mean_address_entropy=sum(r[0] for r in rows) / len(rows),
            mean_maximum_address_probability=sum(r[1] for r in rows) / len(rows),
            mean_write_gate=sum(r[2] for r in rows) / len(rows)) for key, rows in observed.items()},
        uniform8_entropy=math.log(8), extra_label_reads=0, model_executions=0,
        interpretation="WRITE is diffuse; other writes are more concentrated. Descriptive evidence motivates a controlled delta-write test, not proof of a learning bottleneck.",
        costs="local derived statistics only; original GPU/trace/2048 VALID evaluations charged in parent audit; no TRAIN/TEST/model execution")
    target = ROOT / "memory-live-write-address-evidence-v1.json"
    with target.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True);stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
