"""Bounded terminal-data repair of a saved native permutation executor.

Only saved learned graphs, observed training episodes and their recorded history
witness are read during search. No teacher operations, hidden state labels,
Torch, checkpoints, SSH or network are used. Search edits a NATIVE expert class,
propagating that edit through every saved call of the shared expert. Old-task
closed behavior is checked before scoring. Validation is consulted once, AFTER
training-only selection; IID/OOD/length16 are opened only after that decision.

This is a discrete learning/selection diagnostic, not an implementation of a
continuous braid trajectory or a claim of topological advantage. The output is
an executable graph update and explicit bank proposal; installing that proposal
in a neural checkpoint is a separate transaction, not silently claimed here.

MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 NOT PASSED; M3 ultimate.
"""
import argparse
import copy
import hashlib
import json
import time
from collections import deque
from pathlib import Path

import numpy as np


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(part)
    return result.hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def as_graph(raw):
    enc = np.asarray(raw.get("encoder", raw.get("enc")), dtype=np.int64)
    dec = np.asarray(raw.get("decoder", raw.get("dec")), dtype=np.int64)
    trans = {int(a): np.asarray(p, dtype=np.int64)
             for a, p in raw.get("learned_transitions", raw.get("trans", {})).items()}
    if enc.ndim != 1 or dec.ndim != 1 or not len(enc) or not len(dec):
        raise ValueError("invalid saved native code")
    d = len(dec)
    if np.any(enc < 0) or np.any(enc >= d) or np.any(dec < 0) or np.any(dec >= len(enc)):
        raise ValueError("saved code indices outside dimensions")
    tokens = sorted(trans)
    routes = raw.get("routes")
    if routes is None or len(routes) != len(tokens):
        raise ValueError("saved native routes required for a shared-bank update")
    expert_of = {}
    for a, route in zip(tokens, routes):
        p = trans[a]
        if p.shape != (d,) or not np.array_equal(np.sort(p), np.arange(d)):
            raise ValueError("saved native operation is not a full permutation")
        active = [int(k) for k in route if int(k) != 0]
        if len(active) > 1:
            raise ValueError("diagnostic requires already materialized one-expert calls")
        k = active[0] if active else 0
        if k == 0 and not np.array_equal(p, np.arange(d)):
            raise ValueError("identity route disagrees with saved native operation")
        expert_of[a] = k
    return dict(task=int(raw["task"]), enc=enc, dec=dec, trans=trans,
                routes={a: [int(k) for k in r] for a, r in zip(tokens, routes)},
                expert_of=expert_of)


def json_graph(graph):
    return dict(task=graph["task"], encoder=graph["enc"].tolist(),
                decoder=graph["dec"].tolist(),
                routes=[graph["routes"][a] for a in sorted(graph["trans"])],
                learned_transitions={str(a): p.tolist() for a, p in graph["trans"].items()})


def path(graph, start, tokens):
    q = int(graph["enc"][int(start)])
    states = [q]
    for a in tokens:
        q = int(graph["trans"][int(a)][q])
        states.append(q)
    return states


def word(values):
    return tuple(int(a) for a in values if int(a))


def data_hash(data, task):
    digest = hashlib.sha256()
    for name in ("tokens", "starts", "targets"):
        array = np.ascontiguousarray(data[name], dtype="<i8")
        digest.update(name.encode())
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    digest.update(str(task).encode())
    return digest.hexdigest()


def load_data(pathname, task):
    with np.load(pathname, allow_pickle=False) as archive:
        arrays = {k: np.asarray(archive[k], dtype=np.int64)
                  for k in ("tokens", "starts", "targets", "task_ids")}
    n = len(arrays["starts"])
    if (arrays["tokens"].ndim != 2 or arrays["targets"].shape != (n,)
            or arrays["task_ids"].shape != (n,)
            or arrays["tokens"].shape[0] != n
            or arrays["tokens"].shape[1] > 18 or n > 16384
            or np.any(arrays["task_ids"] != task)):
        raise ValueError("dataset shape, task ID or resource cap violation")
    return arrays


def predict(graph, data, costs):
    tokens, starts = data["tokens"], data["starts"]
    if (np.any(starts < 0) or np.any(starts >= len(graph["enc"]))
            or np.any(data["targets"] < 0)
            or np.any(data["targets"] >= len(graph["enc"]))):
        raise ValueError("observed input/output outside code")
    q = graph["enc"][starts].copy()
    for column in range(tokens.shape[1]):
        for a in np.unique(tokens[:, column]):
            if a:
                selected = tokens[:, column] == a
                q[selected] = graph["trans"][int(a)][q[selected]]
    costs["terminal_episode_evaluations"] += len(starts)
    costs["native_transition_evaluations"] += int(np.count_nonzero(tokens))
    return graph["dec"][q]


def score(graph, data, costs):
    predictions = predict(graph, data, costs)
    errors = int(np.count_nonzero(predictions != data["targets"]))
    return dict(examples=len(predictions), errors=errors,
                accuracy=float(1 - errors / len(predictions)))


def related(old, new):
    """Finite closed relation, hence equality of learned old outputs for ALL words."""
    if set(old["trans"]) != set(new["trans"]) or len(old["enc"]) != len(new["enc"]):
        return False, 0
    pending = deque(zip(old["enc"].tolist(), new["enc"].tolist()))
    visited = set()
    while pending:
        pair = pending.popleft()
        if pair in visited:
            continue
        visited.add(pair)
        q, r = pair
        if int(old["dec"][q]) != int(new["dec"][r]):
            return False, len(visited)
        for a in old["trans"]:
            pending.append((int(old["trans"][a][q]), int(new["trans"][a][r])))
    return True, len(visited)


def verify_witness(event, data, task):
    report = event["report"]
    if (not report.get("certificate_valid")
            or int(report.get("fixed_task_id", -1)) != task
            or report.get("dataset_sha256") != data_hash(data, task)):
        raise ValueError("witness validity, fixed task or actual train hash failed")
    nodes = {int(n["id"]): n for n in report["clique_nodes"]}
    pairs = report["pair_witnesses"]
    if not nodes or len(nodes) > 64:
        raise ValueError("history-node resource cap")
    for pair in pairs:
        ids, labels = [int(pair[k]) for k in ("left", "right")], pair["terminal_labels"]
        if labels[0] == labels[1]:
            raise ValueError("pair is not a distinguishability witness")
        for side, identifier in enumerate(ids):
            row, node = int(pair["source_rows"][side]), nodes[identifier]
            if not 0 <= row < len(data["starts"]):
                raise ValueError("witness source row outside train data")
            if (int(data["starts"][row]) != int(node["start"])
                    or word(data["tokens"][row]) != tuple(node["prefix"]) + tuple(pair["suffix"])
                    or int(data["targets"][row]) != int(labels[side])):
                raise ValueError("actual terminal train record does not verify witness")
    return nodes, pairs


def aliases(graph, nodes, pairs):
    states = {i: path(graph, n["start"], n["prefix"])[-1] for i, n in nodes.items()}
    return [p for p in pairs if states[int(p["left"])] == states[int(p["right"])]]


def bank_from_graphs(graphs):
    bank = {0: np.arange(len(next(iter(graphs.values()))["dec"]), dtype=np.int64)}
    for graph in graphs.values():
        for a, k in graph["expert_of"].items():
            if k in bank and not np.array_equal(bank[k], graph["trans"][a]):
                raise ValueError("same shared expert has inconsistent saved semantics")
            bank[k] = graph["trans"][a].copy()
    return bank


def apply_bank_edit(graphs, expert, permutation):
    result = {}
    for task, graph in graphs.items():
        changed = dict(graph)
        changed["trans"] = {a: permutation if graph["expert_of"][a] == expert else p
                            for a, p in graph["trans"].items()}
        result[task] = changed
    return result


def candidates(graphs, task, nodes, unresolved):
    """Redirect an observed prefix edge, compensating inverse image to stay bijective."""
    graph, bank = graphs[task], bank_from_graphs(graphs)
    result, seen = [], set()
    for proof in unresolved:
        for identifier in (int(proof["left"]), int(proof["right"])):
            node = nodes[identifier]
            states = path(graph, node["start"], node["prefix"])
            for position, token in enumerate(node["prefix"]):
                expert, q = graph["expert_of"][int(token)], int(states[position])
                if expert == 0:
                    continue
                old_p = bank[expert]
                inverse = np.argsort(old_p)
                targets = sorted(range(len(old_p)), key=lambda r:
                                 (graph["dec"][r] != graph["dec"][old_p[q]], r))
                for target in targets:
                    other = int(inverse[target])
                    if other == q:
                        continue
                    signature = expert, min(q, other), max(q, other)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    candidate = old_p.copy()
                    candidate[q], candidate[other] = candidate[other], candidate[q]
                    result.append(dict(expert=expert, source_rows=[q, other],
                        before_images=[int(old_p[q]), int(old_p[other])],
                        after_images=[int(candidate[q]), int(candidate[other])],
                        permutation=candidate, witness_pair=[int(proof["left"]), int(proof["right"])],
                        witness_history=identifier, observed_prefix_position=position,
                        preserve_current_prefix_output=bool(graph["dec"][old_p[q]] == graph["dec"][target])))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--datasets", type=Path)
    parser.add_argument("--model", default="topo")
    parser.add_argument("--task", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=30.)
    args = parser.parse_args()
    if not (1 <= args.max_candidates <= 2048 and 1 <= args.max_rounds <= 8
            and 0 < args.time_limit <= 110 and 0 <= args.task <= 2):
        parser.error("bounded candidate/round/time/task limits required")
    if args.out.exists():
        parser.error("--out must be new; existing evidence is not overwritten")
    records = args.records.resolve(strict=True)
    before_stat = records.stat()
    rows = []
    with records.open(encoding="utf8") as stream:
        for line, text in enumerate(stream, 1):
            row = json.loads(text)
            row["source_line"] = line
            rows.append(row)
    if before_stat.st_size != records.stat().st_size:
        raise ValueError("records changed while reading")
    curves = [r for r in rows if r.get("kind") == "curve" and r.get("model") == args.model
              and int(r.get("stage", -1)) == args.task and r.get("learned_structure", {}).get("available")]
    if not curves:
        raise ValueError("no matching current-stage saved native graph")
    selected = max(curves, key=lambda r: (int(r["step"]), int(r["source_line"])))
    graphs = {int(g["task"]): as_graph(g) for g in selected["learned_structure"]["graphs"]}
    if args.task not in graphs or max(len(g["dec"]) for g in graphs.values()) > 64:
        raise ValueError("selected task missing or latent cap exceeded")
    witnesses = [r for r in rows if r.get("kind") == "observed_history_state_witness"
                 and r.get("model") == args.model and int(r.get("stage", -1)) == args.task]
    if not witnesses:
        raise ValueError("no recorded observed-history witness")
    data_dir = args.datasets.resolve(strict=True) if args.datasets else records.parent / "datasets"
    inputs = []

    def dataset(task, split):
        pathname = data_dir / f"task{task}-{split}.npz"
        arrays = load_data(pathname, task)
        inputs.append(dict(task=task, split=split, path=str(pathname), sha256=sha256(pathname)))
        return arrays

    # Search has access ONLY to these current training labels and prior graphs.
    training = dataset(args.task, "train")
    nodes, pairs = verify_witness(witnesses[-1], training, args.task)
    original = copy.deepcopy(graphs)
    costs = dict(candidate_attempts=0, old_relation_checks=0, old_relation_pairs=0,
                 training_scores=0, terminal_episode_evaluations=0,
                 native_transition_evaluations=0, distinct_training_labels=len(training["starts"]),
                 additional_external_label_queries=0)
    started = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "config.json", dict(schema="native-witness-repair-v1",
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_sha256=sha256(Path(__file__)), records_sha256=sha256(records),
        selected_graph_source_line=selected["source_line"], selected_training_step=selected["step"],
        witness_source_line=witnesses[-1]["source_line"],
        information="observed current training terminal labels; prior learned graphs; no teacher states",
        old_protection_scope="tasks strictly BEFORE current task; current provisional semantics may be revised",
        validation_policy="one final no-degradation admission check after training-only selection",
        heldout_policy="open IID/OOD/length16 only after admission decision",
        mechanism="witness-prefix-conditioned two-image permutation edits; shared native bank semantics"))
    initial_score = score(graphs[args.task], training, costs)
    costs["training_scores"] += 1
    original_aliases = aliases(graphs[args.task], nodes, pairs)
    accepted_edits, stop_reason, status = [], None, "NO_ACCEPTED_EDIT"
    with (args.out / "events.jsonl").open("x", encoding="utf8") as log:
        def emit(kind, **fields):
            log.write(json.dumps(dict(kind=kind, elapsed_seconds=time.monotonic()-started,
                                     **fields), sort_keys=True, allow_nan=False)+"\n")
            log.flush()

        emit("baseline", train=initial_score, unresolved_witness_alias_pairs=len(original_aliases))
        current_score = initial_score
        for round_id in range(args.max_rounds):
            unresolved = aliases(graphs[args.task], nodes, pairs)
            if not unresolved:
                stop_reason = "all recorded distinguishability aliases resolved"
                break
            pool = candidates(graphs, args.task, nodes, unresolved)
            best = None
            for proposal in pool:
                if costs["candidate_attempts"] >= args.max_candidates:
                    stop_reason = "candidate cap"
                    break
                if time.monotonic() - started >= args.time_limit:
                    stop_reason = "search wall cap"
                    break
                costs["candidate_attempts"] += 1
                public = {k: v for k, v in proposal.items() if k != "permutation"}
                proposed = apply_bank_edit(graphs, proposal["expert"], proposal["permutation"])
                old_checks = []
                for task in sorted(original):
                    if task >= args.task:
                        continue
                    okay, count = related(original[task], proposed[task])
                    costs["old_relation_checks"] += 1
                    costs["old_relation_pairs"] += count
                    old_checks.append(dict(task=task, related=okay, relation_pairs=count))
                if not all(c["related"] for c in old_checks):
                    emit("candidate", round=round_id, proposal=public, old_checks=old_checks,
                         result="REJECT_OLD_BEHAVIOR")
                    continue
                remaining = aliases(proposed[args.task], nodes, pairs)
                if len(remaining) >= len(unresolved):
                    emit("candidate", round=round_id, proposal=public, old_checks=old_checks,
                         unresolved_alias_pairs=len(remaining), result="REJECT_NO_ALIAS_REDUCTION")
                    continue
                trial_score = score(proposed[args.task], training, costs)
                costs["training_scores"] += 1
                improving = trial_score["errors"] < current_score["errors"]
                emit("candidate", round=round_id, proposal=public, old_checks=old_checks,
                     train=trial_score, unresolved_alias_pairs=len(remaining),
                     result="ELIGIBLE_TRAIN_IMPROVEMENT" if improving else "REJECT_TRAIN_NONIMPROVEMENT")
                if improving and (best is None or trial_score["errors"] < best[0]["errors"]):
                    best = trial_score, proposed, proposal, len(remaining), old_checks
            if best is None:
                stop_reason = stop_reason or "no witnessed train-improving protected edit"
                break
            current_score, graphs, proposal, remaining, old_checks = best
            edit = dict(**{k: v for k, v in proposal.items() if k != "permutation"},
                        native_permutation=proposal["permutation"].tolist(), round=round_id,
                        train=current_score, unresolved_alias_pairs=remaining, old_checks=old_checks)
            accepted_edits.append(edit)
            emit("training_selected_edit", **edit)
            if stop_reason in ("candidate cap", "search wall cap"):
                break
        else:
            stop_reason = "round cap"

        # Single validation gate; no candidate is selected or tuned using valid.
        validation = dataset(args.task, "valid")
        valid_before = score(original[args.task], validation, costs)
        valid_after = score(graphs[args.task], validation, costs)
        admitted = bool(accepted_edits and valid_after["errors"] <= valid_before["errors"])
        if not admitted:
            graphs = original
            status = "VALIDATION_REJECTED" if accepted_edits else "NO_ACCEPTED_EDIT"
        else:
            status = "ADMITTED_NATIVE_GRAPH_REPAIR"
        emit("validation_admission", admitted=admitted, before=valid_before,
             selected_candidate=valid_after, status=status)
        # Final operation is frozen before any heldout data is read.
        write_json(args.out / "frozen-native-graphs.json", dict(
            schema="native-executable-graph-update-v1", status=status,
            source_graphs=[json_graph(g) for _, g in sorted(original.items())],
            graphs=[json_graph(g) for _, g in sorted(graphs.items())],
            admitted_bank_updates=accepted_edits if admitted else [],
            checkpoint_parameters_updated=False,
            execution="apply learned native transition per observed token, then learned decoder"))
        metrics = []
        for task in sorted(graphs):
            for split in ("train", "valid", "iid", "ood", "length16"):
                arrays = training if (task, split) == (args.task, "train") else (
                         validation if (task, split) == (args.task, "valid") else dataset(task, split))
                before = score(original[task], arrays, costs)
                after = score(graphs[task], arrays, costs)
                metric = dict(task=task, split=split, before=before, after=after)
                metrics.append(metric)
                emit("frozen_evaluation", **metric)
        costs["search_and_evaluation_seconds"] = time.monotonic()-started
        summary = dict(schema="native-witness-repair-result-v1", status=status,
            stop_reason=stop_reason, selected_edits=accepted_edits,
            admitted=admitted, verified_training_pair_witnesses=len(pairs),
            original_unresolved_alias_pairs=len(original_aliases),
            final_unresolved_alias_pairs=len(aliases(graphs[args.task], nodes, pairs)),
            costs=costs, metrics=metrics, input_datasets=inputs,
            strongest_conventional_interpretation="terminal-supervised discrete permutation program repair",
            evidence_boundary="executable saved-graph repair; neural checkpoint installation and retraining are separate",
            M0="NOT_ASSESSED")
        write_json(args.out / "summary.json", summary)
        print(json.dumps(dict(status=status, output=str(args.out), edits=len(accepted_edits),
                              admitted=admitted, costs=costs), sort_keys=True))


if __name__ == "__main__":
    main()
