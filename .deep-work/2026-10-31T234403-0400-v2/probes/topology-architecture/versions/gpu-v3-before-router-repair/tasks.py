"""UNEXECUTED terminal-supervision benchmark; NumPy only.

Same V2: MAIN=TOPOLOGY / USER_LOCKED / HUMAN_ONLY_CHANGE; M0 not passed,
M3 final ambition; Online PARK and Persistent State standalone PAUSE.

Teacher computation is private to dataset generation. Learners receive only
tokens, observed starting operands, terminal targets, and stage IDs. No hidden
mode, intermediate state, trace, operation matrix, or correct program is returned.
This interface is not a security boundary against inspection of Python source.

The harness selects training max_len=3, 5, 7 at stages 0, 1, 2. Explicit length
bounds support evaluation while max_len always fixes the padded array width.
Nothing in this module runs at import except definitions and constants.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np


SCHEMA = "terminal-recurrent-tasks-v1"
ARRAY_KEYS = ("tokens", "starts", "targets", "task_ids")


def _integer(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _teacher_terminal(tokens: np.ndarray, starts: np.ndarray, n_states: int) -> np.ndarray:
    """Teacher-only execution: the only returned quantity is terminal operand."""
    state = starts.copy()
    mode = np.zeros(starts.shape, dtype=np.bool_)
    bits = n_states.bit_length() - 1
    mask = n_states - 1
    xor_mask = n_states // 2 + 1
    for column in range(tokens.shape[1]):
        token = tokens[:, column]
        # Token 6 toggles mode; it does not itself alter the observed operand.
        mode ^= token == 6
        use_u = ((token == 1) & ~mode) | ((token == 2) & mode)
        use_v = ((token == 2) & ~mode) | ((token == 1) & mode)
        old = state.copy()
        state[use_u] = (old[use_u] + 1) & mask
        value = old[use_v]
        state[use_v] = ((value << 1) & mask) | (value >> (bits - 1))
        use_w = token == 4
        state[use_w] = old[use_w] ^ xor_mask
        use_r = token == 5
        state[use_r] = (3 * old[use_r] + 1) & mask
        # PAD=0, n=3, and g=6 leave the operand unchanged.
    return state


def data_sha256(data: Mapping[str, Any]) -> str:
    """Hash the four learner arrays canonically, without mutable metadata."""
    digest = hashlib.sha256()
    digest.update(SCHEMA.encode("ascii"))
    for name in ARRAY_KEYS:
        array = np.ascontiguousarray(np.asarray(data[name]), dtype="<i8")
        header = json.dumps(
            {"name": name, "shape": list(array.shape), "dtype": "<i8"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def dataset_summary(data: Mapping[str, Any]) -> dict[str, Any]:
    """Counts and encoding-bit upper bounds, not mutual information estimates."""
    tokens = np.asarray(data["tokens"])
    targets = np.asarray(data["targets"])
    metadata = data.get("metadata", {})
    n_states = int(metadata["n_states"])
    stage = int(metadata["stage"])
    lengths = np.count_nonzero(tokens, axis=1)
    operand_bits = n_states.bit_length() - 1
    highest_token = (3, 5, 6)[stage]
    token_bits = highest_token.bit_length()  # Includes PAD in dense encoding.
    return {
        "n_samples": int(tokens.shape[0]),
        "padded_width": int(tokens.shape[1]),
        "nonpad_tokens": int(lengths.sum()),
        "length_histogram": {
            str(int(length)): int(np.count_nonzero(lengths == length))
            for length in np.unique(lengths)
        },
        "token_histogram": {
            str(token): int(np.count_nonzero(tokens == token))
            for token in range(highest_token + 1)
        },
        "target_histogram": np.bincount(targets, minlength=n_states).tolist(),
        "operand_bits_per_example": operand_bits,
        "terminal_target_bits_upper": int(tokens.shape[0]) * operand_bits,
        "observed_start_bits_upper": int(tokens.shape[0]) * operand_bits,
        "dense_token_encoding_bits_upper": int(tokens.size) * token_bits,
        "nonpad_token_encoding_bits_upper": int(lengths.sum()) * token_bits,
    }


def make_data(
    n_states: int,
    seed: int,
    stage: int,
    n_samples: int,
    max_len: int,
    *,
    length_lo: int = 1,
    length_hi: int | None = None,
) -> dict[str, Any]:
    """Generate deterministic terminal examples using an isolated seeded RNG.

    Stage 0 alphabet: u, v, n. Stage 1 adds w, r. Stage 2 adds g, which
    toggles a hidden mode swapping subsequent u/v meanings. Other meanings
    are mode independent. Every example begins in hidden mode zero.

    At stage 2, ceil(B/2) selected examples are constructively forced to
    contain an adjacent g then u/v, with mode zero before that g. Their lengths
    are sampled uniformly from [max(2,length_lo), length_hi]; other lengths
    are uniform on the requested range. This disclosed mixture ensures at
    least half actually exercise u/v in mode one; it is not an IID uniform-word
    distribution. There are no intermediate supervision labels.
    """
    n_states = _integer("n_states", n_states, 4)
    if n_states & (n_states - 1):
        raise ValueError("n_states must be a power of two >= 4")
    seed = _integer("seed", seed, 0)
    stage = _integer("stage", stage, 0)
    if stage > 2:
        raise ValueError("stage must be 0, 1, or 2")
    n_samples = _integer("n_samples", n_samples, 0)
    max_len = _integer("max_len", max_len, 1)
    length_lo = _integer("length_lo", length_lo, 1)
    length_hi = max_len if length_hi is None else _integer("length_hi", length_hi, 1)
    if not length_lo <= length_hi <= max_len:
        raise ValueError("require 1 <= length_lo <= length_hi <= max_len")
    if stage == 2 and n_samples and length_hi < 2:
        raise ValueError("stage 2 requires length_hi >= 2 for the forced mode examples")

    rng = np.random.default_rng(seed)
    highest_token = (3, 5, 6)[stage]
    lengths = rng.integers(length_lo, length_hi + 1, size=n_samples, dtype=np.int64)
    forced = np.empty(0, dtype=np.int64)
    if stage == 2 and n_samples:
        forced = rng.permutation(n_samples)[: (n_samples + 1) // 2]
        lengths[forced] = rng.integers(
            max(2, length_lo), length_hi + 1, size=forced.size, dtype=np.int64
        )
    tokens = rng.integers(1, highest_token + 1, size=(n_samples, max_len), dtype=np.int64)
    tokens[np.arange(max_len)[None, :] >= lengths[:, None]] = 0
    for row in forced:
        position = int(rng.integers(0, int(lengths[row]) - 1))
        prefix_g = np.flatnonzero(tokens[row, :position] == 6)
        if prefix_g.size % 2:
            tokens[row, int(prefix_g[0])] = 3
        tokens[row, position] = 6
        tokens[row, position + 1] = int(rng.integers(1, 3))
    starts = rng.integers(0, n_states, size=n_samples, dtype=np.int64)
    data: dict[str, Any] = {
        "tokens": tokens,
        "starts": starts,
        "targets": _teacher_terminal(tokens, starts, n_states),
        "task_ids": np.full(n_samples, stage, dtype=np.int64),
        "metadata": {
            "schema": SCHEMA,
            "n_states": n_states,
            "seed": seed,
            "stage": stage,
            "max_len": max_len,
            "length_lo": length_lo,
            "length_hi": length_hi,
            "forced_mode_examples": int(forced.size),
            "supervision": "terminal_operand_only",
            "sampling": "forced_mode_mixture" if stage == 2 else "uniform_length_and_tokens",
        },
    }
    data["metadata"]["array_sha256"] = data_sha256(data)
    data["metadata"]["counts"] = dataset_summary(data)
    return data


def collision_report(train: Mapping[str, Any], evaluation: Mapping[str, Any]) -> dict[str, int]:
    """Report real exact (start, non-PAD word) collisions; do not promise uniqueness."""
    train_keys = {
        (int(start), tuple(int(token) for token in row if token != 0))
        for start, row in zip(train["starts"], train["tokens"])
    }
    evaluation_keys = [
        (int(start), tuple(int(token) for token in row if token != 0))
        for start, row in zip(evaluation["starts"], evaluation["tokens"])
    ]
    unique_evaluation = set(evaluation_keys)
    return {
        "train_unique_examples": len(train_keys),
        "evaluation_unique_examples": len(unique_evaluation),
        "evaluation_duplicate_rows": len(evaluation_keys) - len(unique_evaluation),
        "evaluation_rows_colliding_with_train": sum(key in train_keys for key in evaluation_keys),
        "evaluation_unique_collisions_with_train": len(unique_evaluation & train_keys),
    }


def evaluate_batches(
    n_states: int,
    seed: int,
    stage: int,
    n_samples: int,
    max_len: int,
    *,
    train_data: Mapping[str, Any] | None = None,
    length_lo: int = 1,
    length_hi: int | None = None,
    ood_length_lo: int = 7,
    ood_length_hi: int = 10,
) -> dict[str, dict[str, Any]]:
    """Build evaluation data, not model predictions, with distinct reproducible seeds.

    seed denotes the training seed. IID and requested length-shift batches use
    new derived seeds. Exact train overlap is reported when train_data is given.
    Length 7 overlaps a stage-2 training cap of 7, so the requested 7..10 range
    is not described as wholly unseen-length or guaranteed unseen-word data.
    No rejection loop hides exhaustion of a small finite word/start universe.
    """
    seed = _integer("seed", seed, 0)
    stage = _integer("stage", stage, 0)
    ood_length_lo = _integer("ood_length_lo", ood_length_lo, 1)
    ood_length_hi = _integer("ood_length_hi", ood_length_hi, ood_length_lo)
    forbidden = {seed}
    if train_data is not None:
        forbidden.add(int(train_data["metadata"]["seed"]))
    iid_seed = seed + 10_000_019 + stage * 1_000_003
    while iid_seed in forbidden:
        iid_seed += 1
    forbidden.add(iid_seed)
    ood_seed = seed + 20_000_033 + stage * 1_000_003
    while ood_seed in forbidden:
        ood_seed += 1
    batches = {
        "iid": make_data(
            n_states, iid_seed, stage, n_samples, max_len,
            length_lo=length_lo, length_hi=length_hi,
        ),
        "ood_composition": make_data(
            n_states, ood_seed, stage, n_samples, max(max_len, ood_length_hi),
            length_lo=ood_length_lo, length_hi=ood_length_hi,
        ),
    }
    training_hi = max_len if length_hi is None else int(length_hi)
    for name, batch in batches.items():
        batch["metadata"]["split"] = name
        batch["metadata"]["training_seed"] = seed
        if train_data is not None:
            batch["metadata"]["collision_report"] = collision_report(train_data, batch)
    batches["ood_composition"]["metadata"]["requested_lengths_overlapping_training_range"] = [
        value for value in range(ood_length_lo, ood_length_hi + 1)
        if length_lo <= value <= training_hi
    ]
    return batches
