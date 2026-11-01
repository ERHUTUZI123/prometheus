"""Replay saved native graphs using ONLY records.jsonl and observed NPZ arrays.

No Torch, checkpoint/model loading, training, teacher functions, or inferred
correct intermediate states. Every emitted state path is explicitly MODEL.
Usage: python analyze_native_errors.py --records RUN/learn/records.jsonl --out NEW_DIR
Optional --datasets DIRECTORY and repeated --model NAME select input records.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np


SPLITS = ("train", "valid", "iid", "ood", "length16")


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def csv_write(path, rows):
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: compact(value) if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def read_records(path):
    rows, warnings, digest = [], [], hashlib.sha256()
    before = path.stat()
    with path.open("rb") as stream:
        for line_number, line in enumerate(stream, 1):
            digest.update(line)
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    row["source_line"] = line_number
                    rows.append(row)
            except (ValueError, UnicodeError) as error:
                warnings.append(dict(source_line=line_number, reason=str(error)))
    after = path.stat()
    return rows, warnings, dict(records_path=str(path), records_sha256=digest.hexdigest(),
        stable_during_read=(before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns))


def graph_arrays(graph):
    enc = np.asarray(graph.get("encoder", graph.get("enc")), dtype=np.int64)
    dec = np.asarray(graph.get("decoder", graph.get("dec")), dtype=np.int64)
    raw = graph.get("learned_transitions", graph.get("trans", {}))
    transitions = {int(token): np.asarray(values, dtype=np.int64) for token, values in raw.items()}
    if enc.ndim != 1 or dec.ndim != 1 or not len(enc) or not len(dec):
        raise ValueError("invalid saved encoder/decoder")
    d, n = len(dec), len(enc)
    if np.any(enc < 0) or np.any(enc >= d) or np.any(dec < 0) or np.any(dec >= n):
        raise ValueError("saved code indices outside latent/observable dimensions")
    for token, transition in transitions.items():
        if token <= 0 or transition.shape != (d,) or np.any(transition < 0) or np.any(transition >= d):
            raise ValueError("invalid saved native transition")
    return enc, dec, transitions


def word(row):
    return tuple(int(token) for token in row if int(token) != 0)


def native_path(enc, dec, transitions, start, sequence):
    state = int(enc[int(start)])
    path = [state]
    for token in sequence:
        if int(token) == 0:
            continue
        if int(token) not in transitions:
            raise ValueError("token absent from saved native graph: " + str(token))
        state = int(transitions[int(token)][state])
        path.append(state)
    return dict(MODEL_states=path, MODEL_decoded_states=[int(dec[q]) for q in path],
                MODEL_terminal=int(dec[state]), final_MODEL_state=state)


def replay(enc, dec, transitions, arrays):
    tokens, starts, targets = (arrays[name] for name in ("tokens", "starts", "targets"))
    if tokens.ndim != 2 or starts.shape != (len(tokens),) or targets.shape != starts.shape:
        raise ValueError("invalid learner NPZ shapes")
    if np.any(starts < 0) or np.any(starts >= len(enc)) or np.any(targets < 0) or np.any(targets >= len(enc)):
        raise ValueError("NPZ observed class outside saved code")
    unknown = set(int(t) for t in np.unique(tokens) if int(t)) - set(transitions)
    if unknown:
        raise ValueError("tokens absent from saved native graph: " + str(sorted(unknown)))
    states = np.empty((len(tokens), tokens.shape[1] + 1), dtype=np.int64)
    states[:, 0] = enc[starts]
    for column in range(tokens.shape[1]):
        states[:, column + 1] = states[:, column]
        for token in np.unique(tokens[:, column]):
            if token:
                active = tokens[:, column] == token
                states[active, column + 1] = transitions[int(token)][states[active, column]]
    predictions = dec[states[:, -1]]
    correct = predictions == targets
    confusion = np.zeros((len(enc), len(enc)), dtype=np.int64)
    np.add.at(confusion, (targets, predictions), 1)
    visits = np.bincount(states[:, 0], minlength=len(dec))
    transition_counts, error_counts = {}, {}
    for token in transitions:
        transition_counts[token] = np.zeros((len(dec), len(dec)), dtype=np.int64)
        error_counts[token] = np.zeros((len(dec), len(dec)), dtype=np.int64)
    for column in range(tokens.shape[1]):
        active = tokens[:, column] != 0
        visits += np.bincount(states[active, column + 1], minlength=len(dec))
        for token in np.unique(tokens[active, column]):
            pick = tokens[:, column] == token
            np.add.at(transition_counts[int(token)], (states[pick, column], states[pick, column + 1]), 1)
            pick &= ~correct
            np.add.at(error_counts[int(token)], (states[pick, column], states[pick, column + 1]), 1)
    return dict(predictions=predictions, states=states, correct=correct, confusion=confusion,
                visits=visits, transition_counts=transition_counts, error_counts=error_counts)


def structural_reachability(enc, transitions):
    reached = set(int(q) for q in enc)
    pending = deque(reached)
    while pending:
        state = pending.popleft()
        for transition in transitions.values():
            following = int(transition[state])
            if following not in reached:
                reached.add(following)
                pending.append(following)
    return sorted(reached)


def witness_hash(arrays, task):
    digest = hashlib.sha256()
    for name in ("tokens", "starts", "targets"):
        array = np.ascontiguousarray(arrays[name], dtype="<i8")
        digest.update(name.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    digest.update(str(task).encode())
    return digest.hexdigest()


def load_dataset(path):
    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name], dtype=np.int64)
                  for name in ("tokens", "starts", "targets", "task_ids")}
        metadata = json.loads(str(source["metadata"].item())) if "metadata" in source else {}
    return arrays, metadata


def training_index(arrays):
    index = defaultdict(Counter)
    for row in range(len(arrays["starts"])):
        index[int(arrays["starts"][row]), word(arrays["tokens"][row])][int(arrays["targets"][row])] += 1
    return index


def analyze_witness(event, graph, training, origin):
    enc, dec, transitions = graph_arrays(graph)
    raw = event["report"]
    task = int(raw.get("fixed_task_id", graph["task"]))
    nodes = {int(node["id"]): node for node in raw.get("clique_nodes", [])}
    mapped, groups = {}, defaultdict(list)
    for identifier, node in nodes.items():
        path = native_path(enc, dec, transitions, node["start"], node["prefix"])
        mapped[identifier] = dict(id=identifier, start=node["start"], prefix=node["prefix"], **path)
        groups[path["final_MODEL_state"]].append(identifier)
        matches = [row for row in range(len(training["starts"]))
                   if int(training["starts"][row]) == int(node["start"])
                   and word(training["tokens"][row])[:len(node["prefix"])] == tuple(node["prefix"])]
        suffixes = {word(training["tokens"][row])[len(node["prefix"]):] for row in matches}
        mapped[identifier].update(actual_matching_train_rows=len(matches), observed_unique_suffixes=len(suffixes))
    pairs = []
    for proof in raw.get("pair_witnesses", []):
        left, right = int(proof["left"]), int(proof["right"])
        if left not in mapped or right not in mapped:
            continue
        suffix = tuple(int(t) for t in proof["suffix"])
        labels, source_rows = list(proof["terminal_labels"]), list(proof["source_rows"])
        verified = labels[0] != labels[1]
        for side, identifier in enumerate((left, right)):
            source_row = int(source_rows[side])
            node = nodes[identifier]
            verified = verified and 0 <= source_row < len(training["starts"])
            if verified:
                verified = (int(training["starts"][source_row]) == int(node["start"])
                    and word(training["tokens"][source_row]) == tuple(node["prefix"]) + suffix
                    and int(training["targets"][source_row]) == int(labels[side])
                    and int(training["task_ids"][source_row]) == task)
        continuations = []
        for identifier in (left, right):
            q = mapped[identifier]["final_MODEL_state"]
            path = [q]
            for token in suffix:
                q = int(transitions[token][q])
                path.append(q)
            continuations.append(dict(MODEL_states=path, MODEL_terminal=int(dec[q])))
        merged = mapped[left]["final_MODEL_state"] == mapped[right]["final_MODEL_state"]
        wrong = [continuations[i]["MODEL_terminal"] != labels[i] for i in (0, 1)]
        category = ("unverified_source_pair" if not verified else
                    "UNRESOLVED_MODEL_STATE_ALIAS" if merged else
                    "NONMERGED_OBSERVED_TERMINAL_CONTRADICTION" if any(wrong) else
                    "OBSERVED_PAIR_RESOLVED_BY_MODEL")
        pairs.append(dict(**origin, witness_source_line=event["source_line"], left=left, right=right,
            left_history=dict(start=nodes[left]["start"], prefix=nodes[left]["prefix"]),
            right_history=dict(start=nodes[right]["start"], prefix=nodes[right]["prefix"]),
            source_rows=source_rows, suffix=list(suffix), observed_terminal_labels=labels,
            observed_pair_verified=bool(verified), MODEL_histories_merged=bool(merged),
            MODEL_prefix_states=[mapped[left]["final_MODEL_state"], mapped[right]["final_MODEL_state"]],
            MODEL_continuations=continuations, observed_rows_wrong_under_MODEL=wrong,
            classification=category,
            localization_limit="no correct intermediate state/transition supplied; nonmerged contradiction may involve prefix, continuation or readout"))
    actual_hash = witness_hash(training, task)
    return dict(**origin, witness_source_line=event["source_line"],
        observed_lower_bound=raw.get("lower_bound"), current_MODEL_dim=len(dec),
        original_witness_dataset_sha256=raw.get("dataset_sha256"), actual_dataset_sha256=actual_hash,
        witness_dataset_hash_matches=raw.get("dataset_sha256") == actual_hash,
        current_history_states=list(mapped.values()),
        merged_history_groups=[dict(MODEL_state=q, clique_history_ids=ids) for q, ids in groups.items() if len(ids) > 1],
        pair_count=len(pairs), verified_pair_count=sum(p["observed_pair_verified"] for p in pairs),
        unresolved_alias_pairs=sum(p["classification"] == "UNRESOLVED_MODEL_STATE_ALIAS" for p in pairs),
        nonmerged_terminal_contradictions=sum(p["classification"] == "NONMERGED_OBSERVED_TERMINAL_CONTRADICTION" for p in pairs),
        interpretation="verified merged clique histories rule out insufficient distinguishing terminal data for THOSE pairs; other unseen histories remain open"), pairs


def ce_rows(rows, models):
    promotions = defaultdict(list)
    for row in rows:
        if row.get("kind") == "operation_materialization" and row.get("report", {}).get("success"):
            promotions[row.get("model"), int(row.get("stage", 0))].append(row)
    previous, result = {}, []
    for row in rows:
        if row.get("kind") != "curve" or row.get("model") not in models:
            continue
        key = row["model"], int(row["stage"])
        step = int(row["step"])
        begin = previous.get(key, 0)
        events = promotions[key]
        earlier = [int(event["step"]) for event in events if int(event["step"]) < step]
        ending = [int(event["step"]) for event in events if int(event["step"]) == step]
        crossing = [s for s in earlier if begin < s < step]
        phase = ("no_training_updates" if step == 0 else
                 "mixed_window_crosses_promotion" if crossing else
                 "pre_promotion_window_ending_at_promotion" if ending and begin < step else
                 "post_promotion_training" if earlier and begin >= max(earlier) else
                 "pre_promotion_or_no_promotion")
        current = row.get("current", {})
        result.append(dict(model=key[0], stage=key[1], source_line=row["source_line"],
            step=step, CE_update_window_begin_exclusive=begin, CE_update_window_end_inclusive=step,
            CE_window_phase=phase, train_CE_mean=row.get("train_CE_mean"),
            gradient_norm_mean=row.get("gradient_norm_mean"), router_negative_MI_mean=row.get("router_negative_MI_mean"),
            router_MI_weight=row.get("router_MI_weight"), temperature=row.get("temperature"),
            actual_temperature=row.get("actual_temperature"),
            current_valid_soft=current.get("valid", {}).get("soft_acc"),
            current_valid_native_hard=current.get("valid", {}).get("hard_acc"),
            current_iid_soft=current.get("iid", {}).get("soft_acc"),
            current_iid_native_hard=current.get("iid", {}).get("hard_acc"),
            successful_promotion_steps=[int(event["step"]) for event in events],
            actual_operation_bank_size=row.get("actual_operation_bank_size")))
        previous[key] = step
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--datasets", "--data", dest="datasets", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", action="append")
    parser.add_argument("--stage", type=int)
    parser.add_argument("--max-examples", type=int, default=100)
    args = parser.parse_args()
    if not 0 <= args.max_examples <= 100:
        parser.error("max-examples must be0..100 per model/task/split")
    records = args.records.resolve(strict=True)
    out = args.out.resolve()
    if out.exists():
        parser.error("--out must be a NEW directory; no evidence is overwritten")
    rows, warnings, source = read_records(records)
    config = next((row for row in rows if row.get("kind") == "configuration"), {})
    original_hashes = [{key: row[key] for key in ("source_sha256", "source_hashes", "script_sha256") if key in row}
                       for row in rows if any(key in row for key in ("source_sha256", "source_hashes", "script_sha256"))]
    selected = {}
    for row in rows:
        if row.get("kind") != "curve" or not row.get("learned_structure", {}).get("available"):
            continue
        if args.model and row.get("model") not in args.model:
            continue
        if args.stage is not None and int(row["stage"]) != args.stage:
            continue
        key = row["model"]
        if key not in selected or (row["stage"], row["step"], row["source_line"]) > (
                selected[key]["stage"], selected[key]["step"], selected[key]["source_line"]):
            selected[key] = row
    if not selected:
        parser.error("no saved learned native graph curve matches selection")
    dataset_records = {(int(row["task"]), row["split"]): row for row in rows if row.get("kind") == "dataset"}
    data_cache, data_sources = {}, []

    def dataset(task, split):
        key = task, split
        if key in data_cache:
            return data_cache[key]
        record = dataset_records.get(key, {})
        filename = Path(record.get("file", f"datasets/task{task}-{split}.npz"))
        path = ((args.datasets / filename.name) if args.datasets else records.parent / filename)
        if not path.is_file() and args.datasets is None:
            path = records.parent / "datasets" / filename.name
        path = path.resolve(strict=True)
        if path.suffix != ".npz":
            raise ValueError("only NPZ dataset inputs are allowed")
        arrays, metadata = load_dataset(path)
        if np.any(arrays["task_ids"] != task):
            raise ValueError("NPZ task IDs disagree with requested saved graph")
        details = dict(task=task, split=split, npz_path=str(path), npz_sha256=file_sha(path),
            recorded_array_sha256=record.get("sha256"), recorded_dataset_source_line=record.get("source_line"),
            metadata=metadata)
        data_sources.append(details)
        data_cache[key] = arrays, details
        return data_cache[key]

    summaries, examples, confusions, reachability, exposures = [], [], [], [], []
    witness_results, pair_rows, graph_reports = [], [], []
    for model, curve in selected.items():
        native_status = ("NATIVE_COMPONENT_PROMOTED" if curve.get("native_promotion", config.get("args", {}).get("native_promotion"))
                         else "COMPONENT_RECORDED" if config.get("args", {}).get("compilation") == "component"
                         else "RECORDED_GRAPH_NATIVE_STATUS_UNVERIFIED")
        graphs = {int(g["task"]): g for g in curve["learned_structure"].get("graphs", [])}
        for task, graph in sorted(graphs.items()):
            enc, dec, transitions = graph_arrays(graph)
            origin = dict(model=model, task=task, final_training_stage=int(curve["stage"]),
                          final_step=int(curve["step"]), graph_source_line=curve["source_line"],
                          graph_execution_status=native_status)
            structural = structural_reachability(enc, transitions)
            graph_reports.append(dict(**origin, latent_dim=len(dec), observable_dim=len(enc),
                structural_reachable_MODEL_states=structural,
                structural_unreachable_MODEL_states=sorted(set(range(len(dec))) - set(structural)),
                saved_reachable_count=graph.get("reachable"), recorded_routes=graph.get("routes"),
                graph=graph, state_paths_are="MODEL only; no correct intermediate states"))
            try:
                train, _ = dataset(task, "train")
                train_seen = training_index(train)
            except (OSError, ValueError, KeyError) as error:
                warnings.append(dict(**origin, split="train", reason=str(error)))
                train, train_seen = None, {}
            for split in SPLITS:
                try:
                    arrays, data_source = dataset(task, split)
                    result = replay(enc, dec, transitions, arrays)
                except (OSError, ValueError, KeyError) as error:
                    warnings.append(dict(**origin, split=split, reason=str(error)))
                    continue
                n = len(arrays["starts"])
                bad = np.flatnonzero(~result["correct"])
                seen_error_count = sum((int(arrays["starts"][row]), word(arrays["tokens"][row])) in train_seen for row in bad)
                logged = curve.get("terminal_metrics", {}).get(str(task), {}).get(split, {}).get("hard_acc")
                accuracy = float(result["correct"].mean()) if n else None
                summary = dict(**origin, split=split, n=n, correct=int(result["correct"].sum()),
                    error_count=int(len(bad)), native_replay_accuracy=accuracy,
                    recorded_hard_accuracy=logged,
                    replay_matches_recorded=None if logged is None or accuracy is None else abs(logged-accuracy) < 1e-12,
                    errors_on_exact_inputs_seen_in_train=int(seen_error_count),
                    errors_on_exact_inputs_not_seen_in_train=int(len(bad)-seen_error_count),
                    exported_error_examples=min(len(bad), args.max_examples),
                    npz_path=data_source["npz_path"], npz_sha256=data_source["npz_sha256"])
                summaries.append(summary)
                for row in bad[:args.max_examples]:
                    start, tokens = int(arrays["starts"][row]), word(arrays["tokens"][row])
                    path = native_path(enc, dec, transitions, start, tokens)
                    labels = train_seen.get((start, tokens), {})
                    examples.append(dict(**origin, split=split, source_npz_row=int(row), start=start,
                        tokens=list(tokens), observed_terminal_target=int(arrays["targets"][row]),
                        MODEL_prediction=int(result["predictions"][row]), **path,
                        exact_input_seen_in_train=bool(labels),
                        training_terminal_label_counts={str(label): int(count) for label, count in labels.items()},
                        npz_path=data_source["npz_path"], npz_sha256=data_source["npz_sha256"],
                        path_resolution="one native composed transition per non-PAD token; no inferred teacher states"))
                for target, prediction in np.argwhere(result["confusion"] > 0):
                    confusions.append(dict(**origin, split=split, observed_terminal_target=int(target),
                        MODEL_prediction=int(prediction), count=int(result["confusion"][target, prediction])))
                for state in range(len(dec)):
                    reachability.append(dict(**origin, split=split, MODEL_state=state,
                        decoded_MODEL_value=int(dec[state]), visits=int(result["visits"][state]),
                        structurally_reachable=state in structural))
                for token, counts in result["transition_counts"].items():
                    for before, after in np.argwhere(counts > 0):
                        exposures.append(dict(**origin, split=split, token=token,
                            MODEL_before=int(before), MODEL_after=int(after), uses=int(counts[before, after]),
                            uses_in_terminal_error_episodes=int(result["error_counts"][token][before, after]),
                            localization="cooccurrence only, not a correct/incorrect intermediate transition label"))
            if train is not None:
                for event in rows:
                    if (event.get("kind") == "observed_history_state_witness"
                            and event.get("model") == model and int(event.get("stage", -1)) == task):
                        try:
                            analysis, pairs = analyze_witness(event, graph, train, origin)
                            witness_results.append(analysis)
                            pair_rows.extend(pairs)
                        except (ValueError, KeyError, IndexError) as error:
                            warnings.append(dict(**origin, witness_source_line=event["source_line"], reason=str(error)))
    curves = ce_rows(rows, selected)
    for table in (summaries, examples, confusions, reachability, exposures, pair_rows, curves):
        for row in table:
            row.update(source)
    payload = dict(schema="native-error-replay-v1", inputs=source,
        original_source_hashes=original_hashes or None,
        original_source_hashes_limitation=None if original_hashes else
            "not present in allowed records input; no receipt/current-source hashes substituted",
        selected_final_curves={model: dict(stage=row["stage"], step=row["step"], source_line=row["source_line"])
                               for model, row in selected.items()},
        dataset_sources=data_sources, split_summaries=summaries, error_examples=examples,
        MODEL_graphs=graph_reports, confusion=confusions, MODEL_state_reachability=reachability,
        MODEL_transition_exposure=exposures, observed_history_witness_analysis=witness_results,
        observed_history_pair_analysis=pair_rows, CE_curve_windows=curves,
        warnings=warnings, M0="NOT_ASSESSED_BY_ERROR_REPLAY",
        limitations=["Only saved graph semantics and observed terminal labels are used",
            "Every intermediate path is MODEL, never a teacher-correct path",
            "At most100 errors per model/task/split are exported; all rows contribute to counters",
            "Verified clique histories merged in one MODEL state identify an unresolved representation alias",
            "Nonmerged terminal contradictions do not identify which prefix/transition/readout is incorrect",
            "Unseen exact input is a coverage fact, not proof of insufficient learning data",
            "CE windows ending at a promotion include training BEFORE that promotion, despite event ordering"])
    out.mkdir(parents=True, exist_ok=False)
    for name, table in (("split-summary", summaries), ("error-examples", examples),
                        ("confusion", confusions), ("MODEL-reachability", reachability),
                        ("MODEL-transition-exposure", exposures), ("history-pairs", pair_rows),
                        ("CE-curve-windows", curves)):
        csv_write(out / (name + ".csv"), table)
    with (out / "analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(out), models=list(selected), split_rows=len(summaries),
                          error_examples=len(examples), history_witnesses=len(witness_results),
                          warnings=len(warnings))))


if __name__ == "__main__":
    main()
