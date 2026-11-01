#!/usr/bin/env python3
"""Inspected, bounded finite schema/circuit construction; no imports with network APIs.

Learner functions receive only opaque raw transition rows and outputs. The oracle
generator and final evaluator are code-separated, not access-control separated.
"""
import collections
import itertools
import json
import math
from pathlib import Path
import random
import time

HERE = Path(__file__).resolve().parent
N = 8
ALL = (1 << N) - 1
MAX_EXPR_NODES = 7


def make_world(seed):
    raw_for_hidden = list(range(N))
    random.Random(seed).shuffle(raw_for_hidden)
    hidden_for_raw = {raw: h for h, raw in enumerate(raw_for_hidden)}

    def transition(raw, event, rival=False):
        h = hidden_for_raw[raw]
        m, p, n = (h >> 2) & 1, (h >> 1) & 1, h & 1
        if event == 0:
            m = 0
        elif event == 1:
            m = 1
        elif event == 2:
            p ^= 1
        elif event == 3:
            n ^= 1
        elif event == 4:
            m ^= p ^ (n if rival else 0)
        else:
            raise ValueError(event)
        return raw_for_hidden[(m << 2) | (p << 1) | n]

    def output(raw):
        h = hidden_for_raw[raw]
        return ((h >> 2) ^ (h >> 1)) & 1

    old = tuple(tuple(transition(s, a) for s in range(N)) for a in range(4))
    new = tuple(transition(s, 4) for s in range(N))
    obs = tuple(output(s) for s in range(N))
    return old, new, obs, transition, output, raw_for_hidden


def schema_search(raw_rows):
    """Products of finite enums; no semantic variable or action names are used."""
    results = []
    total_count = 0
    for shape, groups in [((8,), (7,)), ((2, 4), (4, 3)), ((2, 2, 2), (4, 2, 1))]:
        hist = collections.Counter()
        best_score, best_book, best_count = math.inf, None, 0
        books = [tuple(range(N))] if len(shape) == 1 else itertools.permutations(range(N))
        for book in books:
            score = 0
            for row in raw_rows:
                changed_bits = 0
                for src, dst in enumerate(row):
                    changed_bits |= book[src] ^ book[dst]
                score += sum(g.bit_count() for g in groups if g & changed_bits)
            hist[score] += 1
            total_count += 1
            if score < best_score:
                best_score, best_book, best_count = score, book, 1
            elif score == best_score:
                best_count += 1
        results.append({"shape": shape, "score": best_score, "book": best_book,
                        "minimizers": best_count, "histogram": dict(sorted(hist.items()))})
    best = min(results, key=lambda r: (r["score"], len(r["shape"]), r["book"]))
    return best, results, total_count


def expression_library():
    """Shortest Boolean expression trees up to the frozen seven-node cap."""
    by_mask, by_size = {}, collections.defaultdict(list)

    def put(mask, expr, size):
        if mask not in by_mask:
            by_mask[mask] = (expr, size)
            by_size[size].append(mask)

    put(0, ("const", 0), 1)
    put(ALL, ("const", 1), 1)
    for j in range(3):
        mask = sum(((z >> (2-j)) & 1) << z for z in range(N))
        put(mask, ("var", j), 1)
    attempts = 5
    for size in range(2, MAX_EXPR_NODES + 1):
        for m in tuple(by_size[size-1]):
            attempts += 1
            put(m ^ ALL, ("not", by_mask[m][0]), size)
        for left_size in range(1, size-1):
            right_size = size-1-left_size
            for l in tuple(by_size[left_size]):
                for r in tuple(by_size[right_size]):
                    le, re = by_mask[l][0], by_mask[r][0]
                    for op, value in (("and", l & r), ("or", l | r), ("xor", l ^ r)):
                        attempts += 1
                        put(value, (op, le, re), size)
    return by_mask, attempts


def evaluate(expr, fields):
    op = expr[0]
    if op == "const":
        return expr[1]
    if op == "var":
        return fields[expr[1]]
    if op == "not":
        return 1 ^ evaluate(expr[1], fields)
    a, b = evaluate(expr[1], fields), evaluate(expr[2], fields)
    if op == "and":
        return a & b
    if op == "or":
        return a | b
    if op == "xor":
        return a ^ b
    raise ValueError(op)


def fields(code):
    return tuple((code >> (2-j)) & 1 for j in range(3))


def pack(z):
    return (z[0] << 2) | (z[1] << 1) | z[2]


def compile_row(raw_targets, codebook, exprs):
    masks = [0, 0, 0]
    for raw, target in enumerate(raw_targets):
        code, target_code = codebook[raw], codebook[target]
        for j in range(3):
            masks[j] |= ((target_code >> (2-j)) & 1) << code
    return tuple(exprs[m][0] for m in masks)


def compile_output(outputs, codebook, exprs):
    mask = sum(outputs[raw] << codebook[raw] for raw in range(N))
    return exprs[mask][0]


def step(program, state):
    return tuple(evaluate(expr, state) for expr in program)


def gates(expr):
    return 0 if expr[0] in ("const", "var") else 1 + sum(gates(x) for x in expr[1:])


def semantic_axis_audit(codebook, raw_for_hidden):
    """Evaluation only: match learned coordinates modulo sign/permutation."""
    matches = []
    for j in range(3):
        signature = tuple(fields(codebook[raw_for_hidden[h]])[j] for h in range(N))
        options = []
        for k in range(3):
            target = tuple((h >> (2-k)) & 1 for h in range(N))
            if signature == target:
                options.append([k, 0])
            if signature == tuple(1-x for x in target):
                options.append([k, 1])
        matches.append(options)
    return matches


def main():
    started = time.perf_counter()
    exprs, expr_attempts = expression_library()
    runs = []
    for seed in (11, 29):
        old, new, obs, oracle_step, oracle_output, hidden_map = make_world(seed)
        search_start = time.perf_counter()
        best, scores, n_books = schema_search(old)
        search_seconds = time.perf_counter() - search_start
        assert best["shape"] == (2, 2, 2)
        book = best["book"]
        inverse = {code: raw for raw, code in enumerate(book)}
        old_programs = tuple(compile_row(row, book, exprs) for row in old)
        new_program = compile_row(new, book, exprs)
        readout = compile_output(obs, book, exprs)
        candidate = old_programs + (new_program,)
        # Baseline freezes every old function. Its new-event adapter has the same
        # state interface, grammar and data as the candidate, and is compiled fresh.
        baseline_old = tuple(tuple(p) for p in old_programs)
        baseline_new = compile_row(new, book, exprs)
        row_checks = 0
        for a, row in enumerate(old + (new,)):
            for s in range(N):
                out = inverse[pack(step(candidate[a], fields(book[s])))]
                assert out == row[s]
                row_checks += 1
        lens_pairs = 0
        for s in range(N):
            assert inverse[book[s]] == s
            for target_code in range(N):
                put = inverse[target_code]
                assert book[put] == target_code
                lens_pairs += 1
        sequence_checks, transition_checks = 0, 0
        for length in range(5):
            for word in itertools.product(range(5), repeat=length):
                for s in range(N):
                    raw = s
                    cz = bz = fields(book[s])
                    for a in word:
                        raw = oracle_step(raw, a)
                        cz = step(candidate[a], cz)
                        bp = baseline_old[a] if a < 4 else baseline_new
                        bz = step(bp, bz)
                        assert inverse[pack(cz)] == raw == inverse[pack(bz)]
                        assert evaluate(readout, cz) == oracle_output(raw)
                        transition_checks += 1
                    sequence_checks += 1
        # Constructive local repair: a new XOR edge corrupts one old output
        # expression. Causal state targets localize it without a new task oracle.
        damaged = list(old_programs[0])
        damaged[0] = ("xor", damaged[0], ("var", 1))
        residuals = []
        for s in range(N):
            got = step(damaged, fields(book[s]))
            want = fields(book[old[0][s]])
            residuals.append([got[j] != want[j] for j in range(3)])
        faulty_fields = [j for j in range(3) if any(row[j] for row in residuals)]
        repaired = list(damaged)
        inferred = compile_row(old[0], book, exprs)
        for j in faulty_fields:
            repaired[j] = inferred[j]
        assert tuple(repaired) == old_programs[0]
        # Ambiguity of fifth-action discovery from only the four old actions.
        rival_difference = [s for s in range(N) if oracle_step(s, 4) != oracle_step(s, 4, rival=True)]
        assert len(rival_difference) == 4
        runs.append({
            "seed": seed,
            "teacher_transition_queries": 32,
            "teacher_output_queries": 8,
            "new_action_raw_supervision_rows": 8,
            "schema_codebooks_checked": n_books,
            "schema_search_seconds": search_seconds,
            "schema_scores": scores,
            "selected_codebook": book,
            "semantic_axes_evaluation_only": semantic_axis_audit(book, hidden_map),
            "old_programs": old_programs,
            "new_program": new_program,
            "readout_program": readout,
            "all_transition_row_checks": row_checks,
            "lens_put_pair_checks": lens_pairs,
            "post_training_sequence_checks": sequence_checks,
            "post_training_step_checks": transition_checks,
            "candidate_gate_count_sum": sum(gates(e) for p in candidate for e in p) + gates(readout),
            "baseline_gate_count_sum": sum(gates(e) for p in baseline_old + (baseline_new,) for e in p) + gates(readout),
            "candidate_and_adapter_programs_identical": candidate == baseline_old + (baseline_new,),
            "damaged_fields_identified": faulty_fields,
            "damage_violated_rows": sum(any(r) for r in residuals),
            "local_repair_restores_old_program_exactly": tuple(repaired) == old_programs[0],
            "rival_extension_disagrees_on_raw_states": rival_difference,
            "worked_start_raw": hidden_map[3],
            "worked_after_new_event_raw": oracle_step(hidden_map[3], 4),
            "worked_start_fields": fields(book[hidden_map[3]]),
            "worked_after_new_event_fields": step(new_program, fields(book[hidden_map[3]])),
        })
    result = {"status": "completed", "evaluation": "exploratory full finite-state supervision",
              "expression_tree_node_cap": MAX_EXPR_NODES,
              "boolean_functions_in_capped_library": len(exprs),
              "expression_candidates_examined_once": expr_attempts,
              "total_seconds": time.perf_counter() - started,
              "runs": runs}
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
