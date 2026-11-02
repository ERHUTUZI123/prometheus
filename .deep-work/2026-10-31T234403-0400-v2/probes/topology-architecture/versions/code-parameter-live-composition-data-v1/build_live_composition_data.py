"""ROOT-RUN offline fixed closure diagnostic from ORIGINAL stage4 TRAIN only.

Select FIRST complete contiguous256 grid block whose max length<=46 using
lengths ONLY, before any model query. For each original prefix use last MIX15
or replace only that last token by known READ10, then four fixed suffixes.
2048 rows. Original oracle/truth stay in this CPU-only generator namespace;
only data/metadata are copied to model evaluators. No fitting or NN imports.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

ORIGINAL_GENERATOR_SHA = "8943cd6dd4d638224dcc7be62df82e505b871b32bbcf1be5de3be3afadb92726"
ORIGINAL_TRAIN_SHA = "b950377e180c06fc1ae2137c47637f1327a1a9b0d3dfebd54260cec45b037609"
ORIGINAL_TRUTH_SHA = "7f08f29f671a483dd6b979723dc81359408fade781cd72a7ee2d6ef6afb08418"
SUFFIXES = ((), (1, 3), (2, 3), (14,))
CONDITIONS = ("MIX15", "READ10_control")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def suffix_result(q, suffix):
    if suffix == 0:return q
    if suffix == 1:return (q + 1) % 8
    if suffix == 2:return (3 * q + 2) % 8
    if suffix == 3:return 0
    raise ValueError("fixed suffix only")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--truth", type=Path, required=True)
    p.add_argument("--original-metadata", type=Path, required=True)
    p.add_argument("--original-generator", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-seconds", type=float, default=60)
    a = p.parse_args()
    if a.out.exists() or not 0 < a.max_seconds <= 60:
        raise ValueError("new output and finite<=60s CPU generation required")
    import numpy as np
    began = time.perf_counter(); deadline = began + a.max_seconds
    a.out.mkdir(parents=True); data_out = a.out / "data"; data_out.mkdir()
    def check():
        if time.perf_counter() > deadline:raise TimeoutError("finite closure generation bound")
    try:
        manifest = json.loads(a.manifest.read_text())
        if manifest.get("schema") != "live-composition-original-inputs-v1":
            raise ValueError("frozen original-input/initial-six-allowlist manifest required")
        for path, expected in ((a.train, ORIGINAL_TRAIN_SHA), (a.truth, ORIGINAL_TRUTH_SHA),
                               (a.original_generator, ORIGINAL_GENERATOR_SHA)):
            if path.is_symlink() or sha(path) != expected:raise ValueError("exact original owned source required")
        if sha(a.original_metadata) != manifest["original_metadata_sha256"]:
            raise ValueError("original metadata changed")
        original_meta = json.loads(a.original_metadata.read_text())
        if (original_meta["source_sha256"] != ORIGINAL_GENERATOR_SHA
                or original_meta["files"]["train"]["sha256"] != ORIGINAL_TRAIN_SHA
                or original_meta["counterfactual_truth_sha256"] != ORIGINAL_TRUTH_SHA):
            raise ValueError("original generator/TRAIN/truth provenance differs")
        with np.load(a.train, allow_pickle=False) as z:
            if set(z.files) != {"initial", "tokens", "lengths", "targets"}:
                raise ValueError("exact original terminal observation archive required")
            lengths = z["lengths"]
            if lengths.dtype != np.int64 or lengths.shape != (4096,):raise ValueError("original4096 lengths required")
            candidates = [{"block": i, "maximum_length": int(lengths[256*i:256*(i+1)].max()),
                           "minimum_length": int(lengths[256*i:256*(i+1)].min())} for i in range(16)]
            selected = next((r for r in candidates if r["maximum_length"] <= 46), None)
            if selected is None:raise ValueError("no complete256-row block fits frozen <=46 rule; no resampling")
            # Selection is complete before targets/truth/oracle are accessed.
            start = 256 * selected["block"]; stop = start + 256
            initial = z["initial"][start:stop].copy()
            tokens = z["tokens"][start:stop].copy()
            lens = lengths[start:stop].copy()
            source_targets = z["targets"][start:stop].copy()
        check()
        with np.load(a.truth, allow_pickle=False) as z:
            truth = z["train_values_group_order"][start:stop].copy()
        expected_grid = {(v0, v1, q, w) for v0 in range(8) for v1 in range(8) for q in range(2) for w in range(2)}
        if (truth.shape != (256, 5) or truth.dtype != np.int64 or set(truth[:, 0]) != {selected["block"]}
                or {tuple(int(v) for v in row[1:]) for row in truth} != expected_grid):
            raise ValueError("selected block is not the complete original256 grid")
        spec = importlib.util.spec_from_file_location("_owned_original_closure_oracle", a.original_generator)
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
        rows = []; original_checks = 0; oracle_calls = 0; oracle_token_steps = 0; suffix_checks = 0
        for i in range(256):
            word = tokens[i, :int(lens[i])].tolist()
            if len(word) > 46 or word[-1] != 15:raise ValueError("original eligible MIX-ending prefix required")
            target = int(module.terminal(int(initial[i]), word)); oracle_calls += 1; oracle_token_steps += len(word)
            if target != int(source_targets[i]):raise ValueError("original prefix label fails original oracle; no repair/resample")
            original_checks += 1
        for condition in range(2):
            for i in range(256):
                word = tokens[i, :int(lens[i])].tolist()
                if condition == 1:word[-1] = 10
                base_target = None
                for suffix, extra in enumerate(SUFFIXES):
                    expanded = word + list(extra)
                    target = int(module.terminal(int(initial[i]), expanded)); oracle_calls += 1; oracle_token_steps += len(expanded)
                    if suffix == 0:base_target = target
                    if target != suffix_result(base_target, suffix):raise AssertionError("offline closure label rule failed")
                    suffix_checks += 1
                    rows.append((int(initial[i]), expanded, target, start+i, i, condition, suffix))
        arrays = dict(initial=np.asarray([r[0] for r in rows], dtype=np.int64),
                      tokens=np.zeros((2048, 48), dtype=np.int64),
                      lengths=np.asarray([len(r[1]) for r in rows], dtype=np.int64),
                      targets=np.asarray([r[2] for r in rows], dtype=np.int64),
                      source_train_index=np.asarray([r[3] for r in rows], dtype=np.int64),
                      base_local_index=np.asarray([r[4] for r in rows], dtype=np.int64),
                      condition=np.asarray([r[5] for r in rows], dtype=np.int64),
                      suffix=np.asarray([r[6] for r in rows], dtype=np.int64))
        for row, (_, word, *_rest) in enumerate(rows):arrays["tokens"][row, :len(word)] = word
        assert len(rows) == 2048 and int(arrays["lengths"].max()) <= 48
        assert oracle_calls == 2304 and original_checks == 256 and suffix_checks == 2048
        for condition in range(2):
            for i in range(256):
                base = (condition*256+i)*4
                for suffix, extra in enumerate(SUFFIXES):
                    index = base + suffix
                    assert arrays["condition"][index] == condition and arrays["suffix"][index] == suffix
                    assert arrays["lengths"][index] == lens[i] + len(extra)
        check()
        file = data_out / "closure.npz"
        with file.open("xb") as stream:np.savez_compressed(stream, **arrays)
        target_counts = {}
        for condition in range(2):
            target_counts[CONDITIONS[condition]] = {
                str(s):np.bincount(arrays["targets"][(arrays["condition"] == condition) & (arrays["suffix"] == s)], minlength=8).tolist()
                for s in range(4)}
        metadata = dict(schema="live-composition-closure-data-v1", development_postfreeze=True, formal_Primary=False,
            no_old_dataset_or_result_change=True, selected_original_TRAIN_block=selected, candidates_by_length_only=candidates,
            source_TRAIN_rows=list(range(start, stop)), selected_original_prefix_label_checks=256,
            selection="FIRST contiguous complete256 grid with max originalTRAIN length<=46, using length only before any NN query",
            n_states=8, n_tokens=16, max_length=48, rows=2048, base_examples_per_condition=256,
            conditions=list(CONDITIONS), suffix_order=["empty", "SET0_STEP", "SET1_STEP", "CLEAR"],
            row_layout="((condition*256)+base_local_index)*4+suffix; conditions/suffix IDs are evaluator-only",
            neural_inputs_only=["initial", "tokens", "lengths"], evaluation_labels_loaded_only_after_frozen_NN_restore=True,
            source_sha256=dict(original_generator=ORIGINAL_GENERATOR_SHA, original_TRAIN=ORIGINAL_TRAIN_SHA,
                original_truth=ORIGINAL_TRUTH_SHA, original_metadata=sha(a.original_metadata), generator=sha(Path(__file__))),
            file=dict(name=file.name, sha256=sha(file), bytes=file.stat().st_size, array_bytes=sum(v.nbytes for v in arrays.values())),
            teacher_costs=dict(new_materialized_label_observations=2048, oracle_calls_for_new_rows=2048,
                original_TRAIN_label_reads=256, original_prefix_oracle_calls=256, total_oracle_calls=2304,
                original_TRAIN_target_array_rows_decompressed=4096,
                original_TRAIN_target_values_used_for_validation=256,
                oracle_terminal_token_steps=oracle_token_steps,
                analytic_suffix_label_checks=2048, repeated_original_MIX_empty_rows=256,
                no_model_queries=True, generation_paid_once_shared_by_all_six_evaluators=True),
            initial_model_allowlist_protocol=manifest["initial_model_allowlist_protocol"],
            initial_six_model_allowlist=manifest["initial_six_model_allowlist"],
            target_counts_by_condition_suffix=target_counts, generation_wall_seconds=time.perf_counter()-began,
            cpu_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)),
            scope="fixed TRAIN-prefix postfreeze semantic-closure diagnostic; not oldTEST replacement, newtrainingdata or general-computation proof")
        with (data_out / "metadata.json").open("x") as stream:json.dump(metadata, stream, indent=2, sort_keys=True);stream.write("\n")
        with (a.out / "generation-receipt.json").open("x") as stream:json.dump(metadata, stream, indent=2, sort_keys=True);stream.write("\n")
        print(json.dumps(dict(status="LIVE_COMPOSITION_CLOSURE_DATA_CREATED", rows=2048, selected_block=selected,
                              file_sha256=metadata["file"]["sha256"], oracle_calls=2304, wall_seconds=metadata["generation_wall_seconds"])))
    except BaseException as error:
        with (a.out / "failure.json").open("x") as stream:
            json.dump(dict(error_type=type(error).__name__, error=str(error), wall_seconds=time.perf_counter()-began), stream, indent=2)
        raise


if __name__ == "__main__":main()
