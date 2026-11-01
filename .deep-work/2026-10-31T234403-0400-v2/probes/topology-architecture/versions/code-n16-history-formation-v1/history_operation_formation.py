"""Bounded terminal-history state/operation formation for the SAME native model.

No import-time experiment or teacher functions. The coordinator owns execution.
MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 unpassed; M3 ultimate.
"""
import copy
import hashlib
import itertools
import math
import time

import numpy as np
import torch

from topology_model import linear_sum_assignment


def _word(row):
    return tuple(int(a) for a in row if int(a))


def _hash(data, stage):
    h = hashlib.sha256()
    for key in ("tokens", "starts", "targets"):
        a = np.ascontiguousarray(data[key], dtype="<i8")
        h.update(key.encode())
        h.update(np.asarray(a.shape, dtype="<i8").tobytes())
        h.update(a.tobytes())
    h.update(str(stage).encode())
    return h.hexdigest()


def _assignment(mask, scores):
    r, c = linear_sum_assignment(-np.where(mask, scores, -1e30))
    if not np.all(mask[r, c]):
        return None
    p = np.empty(len(mask), dtype=np.int64)
    p[r] = c
    return p


def _supported(mask, matching):
    """Edges in some perfect matching, using alternating-cycle reachability."""
    n, inverse = len(mask), np.argsort(matching)
    graph = [inverse[np.flatnonzero(row)].tolist() for row in mask]
    reach = []
    for i in range(n):
        seen, pending = {i}, [i]
        while pending:
            for j in graph[pending.pop()]:
                if j not in seen:
                    seen.add(j)
                    pending.append(j)
        reach.append(seen)
    support = np.zeros_like(mask)
    for i in range(n):
        for j in np.flatnonzero(mask[i]):
            owner = int(inverse[j])
            support[i, j] = owner in reach[i] and i in reach[owner]
    return support


def _two_matchings(mask, check):
    """Enumerate at most the stated two remaining g matchings, never D! search."""
    answers, n = [], len(mask)
    def visit(partial, used):
        check()
        if len(partial) == n:
            answers.append(np.asarray([partial[i] for i in range(n)], dtype=np.int64))
            if len(answers) > 2:
                raise ValueError("g has more than two matchings; scoped formation budget exceeded")
            return
        remaining = [i for i in range(n) if i not in partial]
        i = min(remaining, key=lambda k: sum(j not in used for j in np.flatnonzero(mask[k])))
        for j in np.flatnonzero(mask[i]):
            if int(j) not in used:
                visit(dict(partial) | {i: int(j)}, used | {int(j)})
    visit({}, set())
    return answers


@torch.no_grad()
def candidate(model, stage, training, witness_record, recorder, max_candidates=64, seconds=110.):
    """Return candidate_graph,audit; selection uses ONLY existing TRAIN feedback."""
    if stage != 2 or not 1 <= max_candidates <= 64 or not 0 < seconds <= 110:
        raise ValueError("bounded stage2 history-formation diagnostic required")
    began, deadline = time.monotonic(), time.monotonic()+seconds
    costs = dict(pair_witnesses_verified=0, observed_profile_entries=0,
        compatibility_comparisons=0, alignment_candidates=0, feasible_alignments=0,
        assignment_calls=0, g_matchings_enumerated=0, old_reuse_source_choices=0,
        old_reuse_feasible_choices=0, terminal_candidate_evaluations=0,
        terminal_episode_evaluations=0, native_transition_uses=0,
        new_external_label_queries=0, source_graph_bank_assignments=0)
    def check():
        recorder.check()
        if time.monotonic() >= deadline:
            raise TimeoutError("110second history-formation bound")
    report = witness_record.get("report", witness_record)
    if (not report.get("certificate_valid") or report.get("fixed_task_id") != stage
            or report.get("dataset_sha256") != _hash(training, stage)
            or np.any(training["task_ids"] != stage)):
        raise ValueError("actual training hash/task/witness verification failed")
    nodes = sorted(report["clique_nodes"], key=lambda n: int(n["id"]))
    n = len(nodes)
    if n != model.d or n > 64 or len(report["pair_witnesses"]) != n*(n-1)//2:
        raise ValueError("this construction needs an actually witnessed full-D clique")
    ids = {int(node["id"]): i for i, node in enumerate(nodes)}
    histories = [(int(node["start"]), tuple(int(a) for a in node["prefix"])) for node in nodes]
    if len(set(histories)) != n:
        raise ValueError("duplicate histories")
    words = [_word(row) for row in training["tokens"]]
    verified_pairs = set()
    for proof in report["pair_witnesses"]:
        check()
        pair = tuple(sorted((int(proof["left"]), int(proof["right"]))))
        if pair[0] == pair[1] or pair in verified_pairs or any(key not in ids for key in pair):
            raise ValueError("duplicate/invalid clique witness pair")
        verified_pairs.add(pair)
        labels = proof["terminal_labels"]
        if labels[0] == labels[1]:
            raise ValueError("non-distinguishing pair witness")
        for side, key in enumerate(("left", "right")):
            h = histories[ids[int(proof[key])]]
            row = int(proof["source_rows"][side])
            if (not 0 <= row < len(words) or int(training["starts"][row]) != h[0]
                    or words[row] != h[1]+tuple(proof["suffix"])
                    or int(training["targets"][row]) != int(labels[side])):
                raise ValueError("pair source row does not verify against original data")
        costs["pair_witnesses_verified"] += 1
    layers = sorted(set(p for _, p in histories), key=lambda p: (len(p), p))
    if len(layers) != 2 or layers[0] != () or len(layers[1]) != 1:
        raise ValueError("scoped constructor requires observed empty and one-new-symbol history layers")
    g = layers[1][0]
    lookup = {h: i for i, h in enumerate(histories)}
    if any((x, p) not in lookup for x in range(model.n) for p in layers):
        raise ValueError("history layers do not cover the observed input domain")
    alphabet = tuple(sorted(int(a) for a in model.graph(stage, range(1, model.vocab+1))["tokens"]))
    native = model.graph(stage, alphabet)
    old = model.graph(stage-1, tuple(a for a in alphabet if a != g))
    costs["source_graph_bank_assignments"] += 3*model.n_ops
    enc, dec = native["enc"], native["dec"]
    observations = [dict() for _ in histories]
    for row, (x, word, y) in enumerate(zip(training["starts"], words, training["targets"])):
        for prefix in layers:
            key = int(x), prefix
            if key in lookup and word[:len(prefix)] == prefix:
                table, suffix = observations[lookup[key]], word[len(prefix):]
                if suffix in table and table[suffix][0] != int(y):
                    raise ValueError("conflicting terminal records under one history/suffix")
                table.setdefault(suffix, (int(y), row))
    costs["observed_profile_entries"] = sum(map(len, observations))
    successors = {a: [] for a in alphabet}
    compatibility = {}
    for a in alphabet:
        for table in observations:
            successors[a].append({s[1:]: value for s, value in table.items() if s and s[0] == a})
        c = np.ones((n, n), dtype=bool)
        for i, table in enumerate(successors[a]):
            for j, target in enumerate(observations):
                for suffix, (y, _) in table.items():
                    costs["compatibility_comparisons"] += 1
                    if suffix in target and target[suffix][0] != y:
                        c[i, j] = False
                        break
        compatibility[a] = c
    # Construct anchored history masks; missing outputs remain unrestricted.
    masks = np.ones((n, model.d), dtype=bool)
    for i, (x, prefix) in enumerate(histories):
        if not prefix:
            masks[i] = False
            masks[i, int(enc[x])] = True
        if () in observations[i]:
            masks[i] &= dec == observations[i][()][0]
    fixed, used = {}, set()
    while True:
        options = {i: [int(q) for q in np.flatnonzero(masks[i]) if int(q) not in used]
                   for i in range(n) if i not in fixed}
        if any(not qs for qs in options.values()):
            raise ValueError("history output/anchor constraints infeasible")
        singles = [(i, qs[0]) for i, qs in options.items() if len(qs) == 1]
        if not singles:
            break
        for i, q in singles:
            if q in used:
                raise ValueError("history clique conflicts with fixed learned code")
            fixed[i], used = q, used | {q}
    remaining = [i for i in range(n) if i not in fixed]
    available = sorted(set(range(model.d))-used)
    if len(remaining) > 2:
        raise ValueError("more than two unaligned histories; this bounded constructor does not enumerate them")
    alignments = []
    for values in itertools.permutations(available):
        if all(masks[i, q] for i, q in zip(remaining, values)):
            mu = np.asarray([fixed.get(i, -1) for i in range(n)], dtype=np.int64)
            for i, q in zip(remaining, values):
                mu[i] = q
            alignments.append(mu)
    if not alignments or len(alignments) > 2:
        raise ValueError("scoped history alignment count invalid")
    ops = model.soft_ops()
    soft = {}
    for a in alphabet:
        tok = torch.as_tensor([[a]], device=model.encoder.device, dtype=torch.long)
        tid = torch.as_tensor([stage], device=model.encoder.device, dtype=torch.long)
        probabilities = torch.softmax(model.calls(tok, tid)[0, 0]/model.temperature, -1)
        matrix = torch.eye(model.d, device=model.encoder.device)
        for slot in range(model.slots):
            matrix = matrix @ torch.einsum("k,kij->ij", probabilities[slot], ops)
        soft[a] = matrix.clamp_min(1e-30).log().cpu().double().numpy()
    old_actions = {}
    for b, transition in old["trans"].items():
        action = old["dec"][transition[old["enc"]]]
        if np.array_equal(np.sort(action), np.arange(model.n)):
            old_actions[int(b)] = [int(x) for x in action]
    best, alignment_audits, total_combinations = None, [], 0
    for alignment_id, mu in enumerate(alignments):
        check()
        costs["alignment_candidates"] += 1
        pools, alignment_info = {}, dict(id=alignment_id, mu=mu.tolist(), feasible=True, operators={})
        for a in alphabet:
            check()
            mask = compatibility[a].copy()
            for i, table in enumerate(successors[a]):
                if () in table:
                    mask[i] &= dec[mu] == table[()][0]
            scores = soft[a][mu[:, None], mu[None, :]]
            costs["assignment_calls"] += 1
            matched = _assignment(mask, scores)
            if matched is None:
                alignment_info.update(feasible=False, failed_operator=a,
                                      zero_rows=np.flatnonzero(~mask.any(axis=1)).tolist())
                break
            support = _supported(mask, matched)
            pool = {}
            def add(sigma, origin):
                key = tuple(int(q) for q in sigma)
                if key not in pool:
                    pool[key] = dict(sigma=np.asarray(sigma), origins=[], score=float(scores[np.arange(n), sigma].sum()))
                pool[key]["origins"].append(origin)
            add(matched, dict(kind="learned_soft_score_constrained_matching"))
            if a == g:
                for sigma in _two_matchings(support, check):
                    costs["g_matchings_enumerated"] += 1
                    add(sigma, dict(kind="remaining_globally_feasible_g_matching"))
            else:
                for choices in itertools.product(sorted(old_actions), repeat=len(layers)):
                    check()
                    costs["old_reuse_source_choices"] += 1
                    by_layer = dict(zip(layers, choices))
                    sigma = np.asarray([lookup[old_actions[by_layer[p]][x], p] for x, p in histories])
                    if np.all(mask[np.arange(n), sigma]):
                        costs["old_reuse_feasible_choices"] += 1
                        add(sigma, dict(kind="old_learned_action_reused_per_observed_history_layer",
                            source_task=stage-1, source_symbols_by_prefix=[dict(prefix=list(p), old_symbol=by_layer[p]) for p in layers]))
            pools[a] = sorted(pool.values(), key=lambda item: (-item["score"], tuple(item["sigma"])))
            alignment_info["operators"][str(a)] = dict(allowed_mask=mask.tolist(),
                supported_mask=support.tolist(), allowed_edges=int(mask.sum()),
                globally_forced_rows=int(np.count_nonzero(support.sum(axis=1) == 1)),
                candidate_count=len(pools[a]), candidate_origins=[item["origins"] for item in pools[a]])
        alignment_audits.append(alignment_info)
        recorder.emit("history_alignment_feasibility", **alignment_info)
        if not alignment_info["feasible"]:
            continue
        costs["feasible_alignments"] += 1
        total_combinations += math.prod(len(pools[a]) for a in alphabet)
        for combination in itertools.product(*(pools[a] for a in alphabet)):
            check()
            if costs["terminal_candidate_evaluations"] >= max_candidates:
                break
            transitions = {}
            for a, item in zip(alphabet, combination):
                permutation = np.empty(model.d, dtype=np.int64)
                permutation[mu] = mu[item["sigma"]]
                transitions[a] = permutation
            state = enc[training["starts"]].copy()
            for column in range(training["tokens"].shape[1]):
                for a in np.unique(training["tokens"][:, column]):
                    if a:
                        selected = training["tokens"][:, column] == a
                        state[selected] = transitions[int(a)][state[selected]]
            errors = int(np.count_nonzero(dec[state] != training["targets"]))
            rank = errors, -sum(item["score"] for item in combination)
            index = costs["terminal_candidate_evaluations"]
            costs["terminal_candidate_evaluations"] += 1
            costs["terminal_episode_evaluations"] += len(state)
            costs["native_transition_uses"] += int(np.count_nonzero(training["tokens"]))
            origin = {str(a): item["origins"] for a, item in zip(alphabet, combination)}
            recorder.emit("history_operation_train_candidate", candidate=index,
                alignment_id=alignment_id, terminal_train_errors=errors,
                terminal_train_accuracy=1-errors/len(state), sources=origin,
                tie_break="learned operator score only; no validation or heldout selection")
            if best is None or rank < best[0]:
                best = rank, transitions, mu.copy(), origin, index
    if best is None:
        raise ValueError("no feasible bounded history/operation candidate")
    selected = copy.deepcopy(native)
    selected["trans"] = best[1]
    reached, pending = set(int(q) for q in enc), list(int(q) for q in enc)
    while pending:
        q = pending.pop()
        for transition in selected["trans"].values():
            following = int(transition[q])
            if following not in reached:
                reached.add(following)
                pending.append(following)
    selected["reachable"] = sorted(reached)
    costs["seconds"] = time.monotonic()-began
    audit = dict(rule="witnessed history separation, observed continuation closure, learned-score and old-action reuse proposals",
        training_only=True, fixed_task_id=stage, dataset_sha256=_hash(training, stage),
        history_nodes=nodes, observed_profile_sizes=[len(o) for o in observations],
        observed_profiles=[dict(history=i, observations=[dict(suffix=list(s), terminal_label=y, source_row=row)
                              for s, (y, row) in o.items()]) for i, o in enumerate(observations)],
        old_learned_actions={str(a): values for a, values in old_actions.items()},
        anchor_states=enc.tolist(), decoder=dec.tolist(),
        initial_alignment_completions=len(alignments), alignment_audits=alignment_audits,
        selected_mu=best[2].tolist(), selected_candidate=best[4],
        selected_train_errors=best[0][0], selected_operation_origins=best[3],
        total_candidate_combinations=total_combinations,
        candidate_search_censored=costs["terminal_candidate_evaluations"] < total_combinations,
        costs=costs, teacher_operations_used=False, teacher_routes_used=False,
        unobserved_history_outputs_not_invented=True,
        strongest_conventional_explanation="passive automaton/state identification and constrained program learning")
    selected["formation_audit"] = audit
    return selected, audit
