"""Six-task raw-terminal benchmark for the parameter-growth experiment.

Public learner interface: tokens, starts, task_ids, targets ONLY. Private teacher
tables, internal states and the crossing-symbol choice never enter returned data
or serialized metadata. Models should consume the generated NPZ files rather
than import private generation code. This is an interface boundary, not an OS
security sandbox; the coordinator retains external execution isolation.

Tasks0..2 independently sample unknown N-state reversible operations. Tasks3..5
independently sample unknown2N-state reversible operations with hidden memory.
All use the SAME public alphabet1..6. Cross-layer symbol is independently private
for each memory task; no fixed g token or correct calling program is supplied.
MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 unpassed; M3 ultimate.
No import-time generation, training, SSH, network, subprocess or installation.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np


SCHEMA = "parameter-efficiency-terminal-stream-v1"
ARRAY_KEYS = ("tokens", "starts", "targets", "task_ids")
ALPHABET = (1, 2, 3, 4, 5, 6)
TRAIN_LENGTH_CAPS = (8, 8, 8, 8, 8, 8)
SPLITS = ("train", "valid", "iid", "ood", "length16")
SPLIT_SEED = {"train": 11, "valid": 23, "iid": 37, "ood": 53, "length16": 71}


def _check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("finite task-stream generation wall limit")


class _PrivateTeacher:
    """Internal label oracle: no transition/state API is returned to learners."""
    __slots__ = ("_transitions", "_n")

    def __init__(self, transitions, observed_states):
        self._transitions, self._n = transitions, observed_states

    def _terminal_labels(self, starts, tokens):
        state = np.asarray(starts, dtype=np.int64).copy()
        for column in range(tokens.shape[1]):
            for symbol in np.unique(tokens[:, column]):
                if symbol:
                    selected = tokens[:, column] == symbol
                    state[selected] = self._transitions[int(symbol)-1, state[selected]]
        return state % self._n


def _private_teacher(n, task_id, seed, deadline):
    """One deterministic draw, never resampled to repair learner/profile errors."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), n, task_id, 70931]))
    _check(deadline)
    if task_id < 3:
        table = np.stack([rng.permutation(n) for _ in ALPHABET])
    else:
        crossing = int(rng.integers(0, len(ALPHABET)))
        table = np.empty((len(ALPHABET), 2*n), dtype=np.int64)
        for a in range(len(ALPHABET)):
            first, second = rng.permutation(n), rng.permutation(n)
            if a == crossing:
                table[a, :n], table[a, n:] = n+first, second
            else:
                table[a, :n], table[a, n:] = first, n+second
    noncommuting = any(not np.array_equal(table[b, table[a, :n]] % n,
                                          table[a, table[b, :n]] % n)
                      for a in range(6) for b in range(a+1, 6))
    if not noncommuting:
        raise RuntimeError("one private draw lacks visible noncommutativity; not resampled")
    return _PrivateTeacher(table, n)


def data_sha256(data):
    digest = hashlib.sha256()
    for key in ARRAY_KEYS:
        array = np.ascontiguousarray(data[key], dtype="<i8")
        digest.update(key.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _raw_episodes(n, task, count, split, seed, max_len, coverage_depth, deadline):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), n, task, SPLIT_SEED[split], 19207]))
    starts, tokens = np.empty(count, dtype=np.int64), np.zeros((count, max_len), dtype=np.int64)
    mandatory = []
    if split == "train":
        # Identical PUBLIC coverage for every task; never depends on a private
        # crossing symbol, teacher transition or teacher's internal state count.
        for length in range(0, coverage_depth+1):
            for word in itertools.product(ALPHABET, repeat=length):
                for x in range(n):
                    mandatory.append((x, word))
    if len(mandatory) > count:
        raise ValueError("requested train count cannot contain the public coverage plan")
    for row, (x, word) in enumerate(mandatory):
        starts[row] = x
        tokens[row, :len(word)] = word
    begin = len(mandatory)
    if split == "ood":
        lo, hi = 9, 12
    elif split == "length16":
        lo = hi = 16
    elif split == "train":
        lo, hi = coverage_depth+1, TRAIN_LENGTH_CAPS[task]
    else:
        lo, hi = 1, TRAIN_LENGTH_CAPS[task]
    lengths = rng.integers(lo, hi+1, size=count-begin)
    starts[begin:] = rng.integers(0, n, size=count-begin)
    for row, length in enumerate(lengths, begin):
        _check(deadline)
        tokens[row, :int(length)] = rng.integers(1, len(ALPHABET)+1, size=int(length))
    if split == "train":
        order = rng.permutation(count)
        starts, tokens = starts[order], tokens[order]
    return starts, tokens, dict(public_coverage_depth=coverage_depth if split == "train" else 0,
        public_coverage_rows=len(mandatory), random_part_length_range=[lo, hi])


def _overlap(train, evaluation):
    keys = {(int(x), tuple(int(a) for a in word if a)) for x, word in zip(train["starts"], train["tokens"])}
    count = sum((int(x), tuple(int(a) for a in word if a)) in keys
                for x, word in zip(evaluation["starts"], evaluation["tokens"]))
    return dict(rows_on_exact_train_inputs=int(count), rows_not_on_exact_train_inputs=len(evaluation["starts"])-int(count))


def generate_stream(n_states, seed=0, train_count=None, eval_count=512,
                    max_len=18, coverage_depth=3, max_seconds=110.):
    """Return six split dictionaries + PUBLIC manifest; no teacher state escapes.

    Each split contains the four int64 arrays plus a public metadata dictionary.
    In-range `iid` means a fresh random short-word test, not equality to the
    deliberately coverage-stratified training distribution. Exact overlap is
    reported. OOD and length16 have strictly longer programs than ANY training
    word and hence cannot be exact training-input memorization tests.
    """
    # The N8/N16 draw schedule is unchanged. N32 requires 259*N=8288
    # public coverage rows, so its explicitly accounted training pool is16384.
    required_train_count = 16384 if n_states == 32 else 8192
    if train_count is None:
        train_count = required_train_count
    if (n_states not in (8, 16, 32) or not isinstance(seed, int) or not 0 <= seed < 2**31
            or train_count != required_train_count or eval_count != 512 or max_len != 18
            or coverage_depth != 3 or not 0 < max_seconds <= 120):
        raise ValueError("requires N8/N16 with8192train or N32 with16384train,512eval,L18,coverage0..3 and finite time")
    deadline = time.monotonic()+max_seconds
    stream, tasks = [], []
    for task in range(6):
        _check(deadline)
        teacher = _private_teacher(n_states, task, seed, deadline)
        splits = {}
        for split in SPLITS:
            count = train_count if split == "train" else eval_count
            starts, tokens, sampling = _raw_episodes(n_states, task, count, split, seed,
                                                   max_len, coverage_depth, deadline)
            labels = teacher._terminal_labels(starts, tokens)
            metadata = dict(schema=SCHEMA, stage=task, task_id=task, split=split,
                observed_state_count=n_states, master_seed=seed, rows=count,
                max_len=max_len, public_alphabet=list(ALPHABET), sampling=sampling,
                learning_interface="tokens,starts,task_ids,targets; final-output supervision only",
                teacher_tables_or_hidden_states_serialized=False)
            data = dict(tokens=tokens, starts=starts, targets=labels,
                        task_ids=np.full(count, task, dtype=np.int64), metadata=metadata)
            metadata["array_sha256"] = data_sha256(data)
            splits[split] = data
        split_info = {}
        for split, data in splits.items():
            overlap = _overlap(splits["train"], data)
            if split in ("ood", "length16") and overlap["rows_on_exact_train_inputs"]:
                raise RuntimeError("OOD program length disjointness invariant failed")
            data["metadata"]["exact_input_overlap_with_train"] = overlap
            split_info[split] = dict(rows=len(data["starts"]), array_sha256=data_sha256(data),
                actual_program_length_min=int(np.count_nonzero(data["tokens"], axis=1).min()),
                actual_program_length_max=int(np.count_nonzero(data["tokens"], axis=1).max()),
                exact_input_overlap_with_train=overlap)
        stream.append(splits)
        tasks.append(dict(task_id=task, train_length_cap=TRAIN_LENGTH_CAPS[task], splits=split_info))
        # Drop the private label oracle; neither it nor its tables is returned.
        del teacher
    manifest = dict(schema=SCHEMA, n_tasks=6, observed_state_count=n_states, master_seed=seed,
        public_alphabet=list(ALPHABET), array_keys=list(ARRAY_KEYS), max_len=max_len,
        train_count=train_count, evaluation_count_per_split=eval_count,
        public_coverage_depth=coverage_depth, tasks=tasks,
        generator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy_version=np.__version__, hidden_state_or_operation_tables_in_outputs=False,
        private_teacher_resampling=False, learner_profile_sufficiency_checked_by_teacher=False,
        heldout_policy="fresh seed streams; exact short-input overlap recorded; OOD9..12 and length16 disjoint from train<=8",
        MAIN="TOPOLOGY_USER_LOCKED", primary_baseline="Transformer", M0="NOT_PASSED")
    return stream, manifest


def save_stream(out, n_states, seed=0, coverage_depth=3, max_seconds=110.):
    """Write NEW directory of30 NPZ files and a public seed/hash manifest."""
    out = Path(out)
    if out.exists():
        raise ValueError("new output directory required; no existing evidence overwritten")
    began = time.monotonic()
    stream, manifest = generate_stream(n_states, seed=seed, coverage_depth=coverage_depth,
                                       max_seconds=max_seconds)
    out.mkdir(parents=True, exist_ok=False)
    data_dir = out/"datasets"
    data_dir.mkdir()
    files = []
    for task, splits in enumerate(stream):
        for split, data in splits.items():
            _check(began+max_seconds)
            path = data_dir/f"task{task}-{split}.npz"
            np.savez_compressed(path, **{key: data[key] for key in ARRAY_KEYS},
                metadata=np.asarray(json.dumps(data["metadata"], sort_keys=True, allow_nan=False)))
            files.append(dict(task_id=task, split=split, file=str(path.relative_to(out)),
                bytes=path.stat().st_size, npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                array_sha256=data_sha256(data)))
    manifest.update(files=files, generation_and_write_seconds=time.monotonic()-began)
    with (out/"manifest.json").open("x", encoding="utf8") as stream_file:
        json.dump(manifest, stream_file, indent=2, sort_keys=True, allow_nan=False)
        stream_file.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=int, choices=(8, 16, 32), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--coverage-depth", type=int, choices=(3,), default=3)
    parser.add_argument("--max-seconds", type=float, default=110.)
    args = parser.parse_args()
    manifest = save_stream(args.out, args.scale, seed=args.seed,
                           coverage_depth=args.coverage_depth, max_seconds=args.max_seconds)
    print(json.dumps(dict(status="SIX_TASK_RAW_DATA_SAVED", out=str(args.out),
        scale=args.scale, seed=args.seed, files=len(manifest["files"]),
        generator_source_sha256=manifest["generator_source_sha256"]), sort_keys=True))


if __name__ == "__main__":
    main()
