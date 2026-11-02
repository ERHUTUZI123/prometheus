"""ROOT-RUN fixed-prefix character data; no work/network at import.

Uses only an already prepared TRAIN-only-vocabulary corpus. Outputs the exact
initial/tokens/lengths/targets NPZ terminal contract. TEST target positions are
shared across context32/64/128, not three independent test samples. Sampled
next-character CE/PPL is not full-stream or word-level perplexity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x") as stream:json.dump(value, stream, indent=2, allow_nan=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True, help=".../prepared/shakespeare")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    source = json.loads((a.prepared / "manifest.json").read_text())
    vocab = json.loads((a.prepared / "vocab.json").read_text())
    v = int(vocab["vocabulary_size"])
    if not 2 <= v <= 512 or vocab.get("unk_id") != 0 or len(vocab["characters_by_id_from1"]) != v - 1:
        raise ValueError("expected the frozen TRAIN-only character vocabulary with UNK0 and V<=512")
    receipts = {}
    for name in ("train.npy", "valid.npy", "test.npy", "vocab.json"):
        if (a.prepared / name).stat().st_size > 40 << 20:
            raise ValueError("prepared token/vocabulary source exceeds40MiB development bound")
        actual = sha(a.prepared / name)
        if source["files"].get(name) != actual:
            raise ValueError(f"prepared source hash differs: {name}")
        receipts[name] = actual
    streams = {name: np.load(a.prepared / f"{name}.npy", mmap_mode="r", allow_pickle=False)
               for name in ("train", "valid", "test")}
    for name, ids in streams.items():
        if ids.ndim != 1 or ids.dtype != np.uint16 or np.any(ids >= v):
            raise ValueError(f"invalid frozen token source: {name}")
    specification = {"train": ("train", 8192, 32), "valid": ("valid", 512, 32),
                     "test_iid": ("test", 512, 32), "test_long": ("test", 512, 128),
                     "test_unseen": ("test", 512, 64)}
    positions = {}
    for code, (name, count, minimum) in enumerate((("train", 8192, 32), ("valid", 512, 32), ("test", 512, 128))):
        if len(streams[name]) - minimum < count:
            raise ValueError("source split too short for unique fixed target positions")
        rng = np.random.default_rng(np.random.SeedSequence([a.seed, code, 0x4c4d]))
        positions[name] = np.sort(rng.choice(len(streams[name]) - minimum, count, replace=False) + minimum)
    files = {}
    for split, (origin, count, context) in specification.items():
        ids = streams[origin];pos = positions[origin]
        tokens = np.stack([ids[int(i) - context:int(i)] for i in pos]).astype(np.int64)
        targets = np.asarray(ids[pos], dtype=np.int64)
        filename = f"stage0-{split}.npz";position_file = f"stage0-{split}-positions.npy"
        np.savez_compressed(a.out / filename, initial=np.zeros(count, dtype=np.int64), tokens=tokens,
                            lengths=np.full(count, context, dtype=np.int64), targets=targets)
        np.save(a.out / position_file, pos.astype(np.int64), allow_pickle=False)
        files[split] = {"file": filename, "sha256": sha(a.out / filename),
                        "positions_file": position_file, "positions_sha256": sha(a.out / position_file),
                        "source_split": origin, "rows": count, "context_length": context,
                        "target_position_min": int(pos.min()), "target_position_max": int(pos.max()),
                        "unknown_context_tokens": int((tokens == 0).sum()),
                        "unknown_targets": int((targets == 0).sum())}
    metadata = {"schema": "sampled-char-context-terminal-v1", "development_only": True,
        "n_states": v, "n_tokens": v, "max_length": 128, "stages": [{"stage": 0, "files": files}],
        "seed": a.seed, "numpy_version": np.__version__, "source_prepared": str(a.prepared.resolve()),
        "source_manifest_sha256": sha(a.prepared / "manifest.json"), "source_files_sha256": receipts,
        "raw_corpus_sha256": source["source"]["sha256"], "vocabulary_policy": "source TRAIN-only characters; UNK0",
        "input_contract": "initial=0 is an uninformative shared initializer; prefix is all available input; target is its following character",
        "source_position_contract": "target at p; input ids[p-context:p], all inside own original split",
        "test_pairing": "same512 original TEST target positions across32/64/128; dependent paired measurements",
        "test_unseen_name_scope": "context64 untrained length, not a semantic novelty or separate corpus claim",
        "metric": "sampled conditional next-character NLL and exp(meanNLL); NOT full-corpus/word PPL",
        "target_count_per_example": 1, "training_information": "no hidden state, route, taskID or future token in input",
        "wall_seconds": time.perf_counter() - started,
        "source_sha256": sha(Path(__file__))}
    write_json(a.out / "metadata.json", metadata)
    print(json.dumps({"status": "fixed_context_data_prepared", "out": str(a.out), "vocabulary": v,
                      "rows": {name: value[1] for name, value in specification.items()}}))


if __name__ == "__main__":main()
