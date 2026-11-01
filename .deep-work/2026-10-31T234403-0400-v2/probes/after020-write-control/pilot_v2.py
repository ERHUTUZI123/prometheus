#!/usr/bin/env python3
"""One root-contract exploratory control pass; author has not executed this file."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time


HOLD = 64
ITERATIONS = 512
LR = 0.05
MARGIN = 0.4
CLIP = 10.0
BOUNDS = ((0.0, 2.5), (-4.0, 4.0), (-3.0, 3.0))
RANK_TOL = 1e-12
FD_STEP, FD_ATOL, FD_RTOL = 1e-6, 1e-7, 2e-4
CHECKPOINT_EVERY = 16
INTERNAL_WALL_SECONDS = 25.0
MAX_RAW_BYTES = 3 * 1024 * 1024
FAILURE_RESERVE = 8192
NAMES = ("A", "beta", "v")
HERE = Path(__file__).resolve().parent
SOURCE_RAW = HERE.parent / "after020-local-contrast" / "pilot-v1.raw.jsonl"
EXPECTED_SOURCE_SHA256 = "ce2bc2777c261e56bfd877b170ded8195d0cac6a62954a1108fa780100e6d1fc"
SOURCE_ARM = "endpoint_plus_prefix_plus_local_carry"
RAW_PATH = HERE / "pilot-v2.raw.jsonl"
START_WALL, START_CPU = time.monotonic(), time.process_time()


class BudgetExceeded(RuntimeError):
    pass


def guard_time():
    if time.monotonic() - START_WALL > INTERNAL_WALL_SECONDS:
        raise BudgetExceeded("25 s internal backup wall guard")


class RawWriter:
    def __init__(self):
        self.handle = RAW_PATH.open("xb")
        self.bytes = self.rows = 0

    def write(self, record, failure=False):
        record = dict(record)
        record.update(wall_seconds=time.monotonic() - START_WALL,
                      cpu_seconds=time.process_time() - START_CPU)
        data = (json.dumps(record, allow_nan=False, separators=(",", ":"))
                + "\n").encode()
        limit = MAX_RAW_BYTES if failure else MAX_RAW_BYTES - FAILURE_RESERVE
        if self.bytes + len(data) >= limit:
            raise BudgetExceeded("raw output cap would be exceeded")
        self.handle.write(data)
        self.handle.flush()
        self.bytes += len(data)
        self.rows += 1

    def close(self):
        self.handle.close()


def counter():
    return dict.fromkeys((
        "mode_steps", "rtrl_steps", "episode_replays", "loss_evaluations",
        "cached_training_label_uses", "local_write_steps", "write_cut_uses",
        "local_carry_steps", "carry_pair_uses", "hold_transition_scans",
        "hold_jacobian_rows", "projection_dot_products",
        "dot_scalar_multiplications", "projection_vector_subtractions",
        "isolated_hold_evaluations"), 0)


def object_bytes(obj, seen=None):
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    result = sys.getsizeof(obj)
    if isinstance(obj, dict):
        result += sum(object_bytes(k, seen) + object_bytes(v, seen)
                      for k, v in obj.items())
    elif isinstance(obj, (tuple, list)):
        result += sum(object_bytes(v, seen) for v in obj)
    return result


def load_source_once():
    data = SOURCE_RAW.read_bytes()  # One read; no old optimizer/training is run.
    digest = hashlib.sha256(data).hexdigest()
    if digest != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("v1 raw hash differs from pre-execution contract source")
    records = [json.loads(line) for line in data.splitlines()]
    starts = [r for r in records if r.get("kind") == "initial"]
    ends = [r for r in records if r.get("kind") == "complete"]
    checkpoints = [r for r in records if r.get("kind") == "checkpoint"
                   and r.get("arm") == SOURCE_ARM and r.get("step") == 512]
    if len(starts) != 1 or len(ends) != 1 or len(checkpoints) != 1:
        raise RuntimeError("source run does not contain one complete selected checkpoint")
    theta = [checkpoints[0]["theta"][key] for key in NAMES]
    originals = []
    for e in starts[0]["training_bank"]:
        originals.append({"name": e["name"], "inputs": list(e["inputs"]),
                          "label": e["label"], "context": e["context"],
                          "label_source": "two cached v1 researcher labels"})
    if len(originals) != 2:
        raise RuntimeError("expected two original episodes")
    for e in originals:
        if (len(e["inputs"]) != HOLD + 1 or e["inputs"][0] not in (-1.0, 1.0)
                or any(x != 0.0 for x in e["inputs"][1:])):
            raise RuntimeError("original raw inputs differ from contracted fixture")
    provenance = {
        "sha256": digest, "bytes_read_once": len(data), "json_rows_parsed": len(records),
        "selected_arm": SOURCE_ARM, "selected_step": 512,
        "actual_initial_theta": dict(zip(NAMES, theta)),
        "old_acquisition_separate_from_new_pass": {
            "whole_v1_parameter_updates": ends[0]["total_parameter_updates"],
            "selected_arm_costs": ends[0]["arm_costs"][SOURCE_ARM]},
    }
    return theta, originals, provenance


def acquire_researcher_fixture(originals):
    """Four NEW designed endpoint labels; no intermediate mode-state labels."""
    bank = list(originals)
    for first in (-1.0, 1.0):
        for last in (-1.0, 1.0):
            bank.append({
                "name": f"two_writes_{first:+g}_{last:+g}",
                "inputs": [first] + [0.0] * HOLD + [last] + [0.0] * HOLD,
                "label": last, "context": originals[0]["context"],
                "label_source": "NEW researcher-generated endpoint; last-write task extension",
            })
    return bank


def mine(bank):
    costs = {"incumbent_steps": 0, "candidate_episode_pairs": 0,
             "raw_token_comparisons": 0, "prefix_comparisons": 0,
             "future_comparisons": 0, "merger_state_checks": 0,
             "merger_future_comparisons": 0, "old_training_labels": 2,
             "new_training_labels": 4, "total_training_labels": 6,
             "stored_raw_tokens": sum(len(e["inputs"]) for e in bank)}
    old = []
    for e in bank:
        states = [0.0]
        for x in e["inputs"]:
            states.append(x)
            costs["incumbent_steps"] += 1
        old.append(states)
    pairs, cuts = [], []
    for i in range(len(bank)):
        for j in range(i + 1, len(bank)):
            costs["candidate_episode_pairs"] += 1
            a, b = bank[i], bank[j]
            if (a["context"] != b["context"] or a["label"] == b["label"]
                    or len(a["inputs"]) != len(b["inputs"])):
                continue
            differences = []
            for t, (xa, xb) in enumerate(zip(a["inputs"], b["inputs"])):
                costs["raw_token_comparisons"] += 1
                if xa != xb:
                    differences.append(t)
            if len(differences) != 1:
                continue  # Actual same-past single-input contrasts, not prefix correlations.
            k = differences[-1] + 1  # State index immediately AFTER differing input.
            for t in range(k - 1):
                costs["prefix_comparisons"] += 1
                if a["inputs"][t] != b["inputs"][t]:
                    raise RuntimeError("mined pair has different pasts")
            for t in range(k, len(a["inputs"])):
                costs["future_comparisons"] += 1
                if a["inputs"][t] != b["inputs"][t]:
                    raise RuntimeError("mined future is not identical")
            merger = None
            for t in range(len(a["inputs"]) + 1):
                costs["merger_state_checks"] += 1
                if old[i][t] != old[j][t]:
                    continue
                same = True
                for u in range(t, len(a["inputs"])):
                    costs["merger_future_comparisons"] += 1
                    if a["inputs"][u] != b["inputs"][u]:
                        same = False
                        break
                if same:
                    merger = t
                    break
            if merger is None:
                raise RuntimeError("no actual incumbent merger with the matched future")
            plus, minus = (i, j) if a["label"] > b["label"] else (j, i)
            pair = {"positive": plus, "negative": minus, "cut": k,
                    "first_exact_incumbent_merger": merger,
                    "total_inputs": len(a["inputs"]),
                    "carry_indices": list(range(k + 1, len(a["inputs"]) + 1)),
                    "orientation": "endpoint-label sign; designed gauge with positive readout"}
            pairs.append(pair)
            cuts.extend([{"episode": i, "cut": k}, {"episode": j, "cut": k}])
    if len(pairs) != 3 or len(cuts) != 6 or len({c["episode"] for c in cuts}) != 6:
        raise RuntimeError("contract requires three mined pairs and one cut per episode")
    cuts.sort(key=lambda x: x["episode"])
    costs["stored_python_object_bytes_estimate"] = object_bytes((bank, old, pairs, cuts))
    return pairs, cuts, old, costs


def forward(theta, inputs, cost, derivatives=True):
    A, beta, _ = theta
    r = dA = dbeta = 0.0
    rs, das, dbs = [0.0], [0.0], [0.0]
    cost["episode_replays"] += 1
    for x in inputs:
        previous = r
        r = math.tanh(A * previous + beta * x)
        cost["mode_steps"] += 1
        if derivatives:
            slope = 1.0 - r * r
            dA = slope * (previous + A * dA)
            dbeta = slope * (x + A * dbeta)
            das.append(dA)
            dbs.append(dbeta)
            cost["rtrl_steps"] += 1
        rs.append(r)
    return {"r": rs, "dA": das, "dbeta": dbs}


def replay(theta, bank, cost):
    return [forward(theta, e["inputs"], cost) for e in bank]


def task(theta, bank, traces, cost):
    loss, grad, per_episode = 0.0, [0.0] * 3, []
    v = theta[2]
    for e, tr in zip(bank, traces):
        error = v * tr["r"][-1] - e["label"]
        value = 0.5 * error * error
        per_episode.append(value)
        loss += value / len(bank)
        grad[0] += error * v * tr["dA"][-1] / len(bank)
        grad[1] += error * v * tr["dbeta"][-1] / len(bank)
        grad[2] += error * tr["r"][-1] / len(bank)
        cost["cached_training_label_uses"] += 1
    cost["loss_evaluations"] += 1
    return loss, grad, per_episode


def freeze_write(bank, traces, cuts):
    return [(traces[c["episode"]]["r"][c["cut"] - 1],
             bank[c["episode"]]["inputs"][c["cut"] - 1],
             bank[c["episode"]]["label"]) for c in cuts]


def write_loss(theta, fixed, cost):
    A, beta, _ = theta
    loss, grad = 0.0, [0.0] * 3
    for previous, x, y in fixed:
        r = math.tanh(A * previous + beta * x)
        cost["mode_steps"] += 1
        cost["local_write_steps"] += 1
        gap = MARGIN - y * r
        if gap > 0.0:
            loss += 0.5 * gap * gap / len(fixed)
            dz = -y * gap * (1.0 - r * r) / len(fixed)
            grad[0] += dz * previous
            grad[1] += dz * x
        cost["write_cut_uses"] += 1
    cost["loss_evaluations"] += 1
    return loss, grad


def freeze_carry(bank, traces, pairs):
    result = []
    for pair in pairs:
        p, n = pair["positive"], pair["negative"]
        for t in pair["carry_indices"]:
            result.append((traces[p]["r"][t - 1], traces[n]["r"][t - 1],
                           bank[p]["inputs"][t - 1], bank[n]["inputs"][t - 1]))
    return result


def carry_loss(theta, fixed, cost):
    A, beta, _ = theta
    loss, grad = 0.0, [0.0] * 3
    for ap, an, xp, xn in fixed:
        rp = math.tanh(A * ap + beta * xp)
        rn = math.tanh(A * an + beta * xn)
        cost["mode_steps"] += 2
        cost["local_carry_steps"] += 2
        delta = rp - rn
        gap = 2.0 * MARGIN - abs(delta)
        if gap > 0.0:
            loss += 0.25 * gap * gap / len(fixed)
            sign = 1.0 if delta > 0 else -1.0 if delta < 0 else 0.0
            dd = -0.5 * gap * sign / len(fixed)
            gp, gn = 1.0 - rp * rp, 1.0 - rn * rn
            grad[0] += dd * (gp * ap - gn * an)
            grad[1] += dd * (gp * xp - gn * xn)
        cost["carry_pair_uses"] += 1
    cost["loss_evaluations"] += 1
    return loss, grad


def dot(a, b, cost):
    cost["projection_dot_products"] += 1
    cost["dot_scalar_multiplications"] += len(a)
    return sum(x * y for x, y in zip(a, b))


def orthonormal_rows(rows, cost):
    basis = []
    for row in rows:
        residual = list(row)
        # Actual modified Gram-Schmidt with reorthogonalization, absolute rank tolerance.
        for _ in range(2):
            for b in basis:
                coefficient = dot(residual, b, cost)
                residual = [residual[j] - coefficient * b[j] for j in range(3)]
                cost["projection_vector_subtractions"] += 3
        norm = math.sqrt(dot(residual, residual, cost))
        if norm > RANK_TOL:
            basis.append([x / norm for x in residual])
    return basis


def project(vector, basis, cost):
    result = list(vector)
    for b in basis:
        coefficient = dot(result, b, cost)
        result = [result[j] - coefficient * b[j] for j in range(3)]
        cost["projection_vector_subtractions"] += 3
    return result


def hold_rows(theta, bank, traces, cost):
    """PARAMETER Jacobian, not a loss gradient; actual original hold predecessors."""
    A, beta, _ = theta
    rows, samples = [], []
    for index in (0, 1):
        for t, x in enumerate(bank[index]["inputs"], start=1):
            cost["hold_transition_scans"] += 1
            if x != 0.0:
                continue
            previous = traces[index]["r"][t - 1]
            value = math.tanh(A * previous + beta * x)
            slope = 1.0 - value * value
            rows.append([slope * previous, slope * x, 0.0])
            samples.append({"episode": index, "t": t, "previous": previous, "input": x})
            cost["mode_steps"] += 1
            cost["hold_jacobian_rows"] += 1
    return rows, samples


def projection_details(theta, bank, traces, write_gradient, cost):
    rows, samples = hold_rows(theta, bank, traces, cost)
    basis = orthonormal_rows(rows, cost)
    projected = project(write_gradient, basis, cost)
    annihilation = max((abs(dot(row, projected, cost)) for row in rows), default=0.0)
    orthogonality = max((abs(dot(a, b, cost) - (1.0 if i == j else 0.0))
                         for i, a in enumerate(basis) for j, b in enumerate(basis)), default=0.0)
    norms = [math.sqrt(sum(x * x for x in row)) for row in rows]
    report = {"rank": len(basis), "basis": basis, "row_count": len(rows),
              "absolute_rank_tolerance": RANK_TOL,
              "min_row_norm": min(norms) if norms else 0.0,
              "max_row_norm": max(norms) if norms else 0.0,
              "annihilation_residual_max": annihilation,
              "basis_orthogonality_residual": orthogonality,
              "rank_zero_not_replaced_by_manual_mask": len(basis) == 0}
    return projected, report, rows, samples


def components(theta, bank, pairs, cuts, cost, with_write, with_carry, with_projection):
    traces = replay(theta, bank, cost)
    lt, gt, per_episode = task(theta, bank, traces, cost)
    lw, gw = write_loss(theta, freeze_write(bank, traces, cuts), cost) if with_write else (None, None)
    lc, gc = carry_loss(theta, freeze_carry(bank, traces, pairs), cost) if with_carry else (None, None)
    projected, projection_report = gw, None
    if with_projection:
        projected, projection_report, _, _ = projection_details(theta, bank, traces, gw, cost)
    gradient = [gt[j] + (projected[j] if with_write else 0.0)
                + (gc[j] if with_carry else 0.0) for j in range(3)]
    # A projected semigradient is NOT asserted to be the gradient of this scalar sum.
    loss_sum = lt + (lw if lw is not None else 0.0) + (lc if lc is not None else 0.0)
    return {"traces": traces, "task": (lt, gt), "write": (lw, gw), "carry": (lc, gc),
            "per_episode_task": per_episode, "projected_write": projected,
            "projection": projection_report, "active_gradient": gradient,
            "reported_active_loss_sum_not_gradient_certificate": loss_sum}


def step(theta, gradient):
    if not all(math.isfinite(x) for x in gradient):
        raise RuntimeError("nonfinite update gradient")
    norm = math.sqrt(sum(g * g for g in gradient))
    scale = min(1.0, CLIP / norm) if norm else 1.0
    proposed = [theta[j] - LR * scale * gradient[j] for j in range(3)]
    result = [min(BOUNDS[j][1], max(BOUNDS[j][0], proposed[j])) for j in range(3)]
    return result, {"pre_theta": list(theta), "gradient": list(gradient),
                    "norm_before_clip": norm, "clip_scale": scale,
                    "post_theta": result, "bound_clipped": [NAMES[j] for j in range(3)
                                                            if proposed[j] != result[j]]}


def derivative_checks(bank, pairs, cuts, initial, writer, cost):
    point = [1.05, 0.12, 0.8]
    traces = replay(point, bank, cost)
    fixed_write = freeze_write(bank, traces, cuts)
    fixed_carry = freeze_carry(bank, traces, pairs)
    fixed_rows, hold_samples = hold_rows(point, bank, traces, cost)
    checks = []

    def evaluate(name, theta):
        if name == "task_true_RTRL":
            tr = replay(theta, bank, cost)
            value, gradient, _ = task(theta, bank, tr, cost)
            return value, gradient
        if name == "write_conditional":
            return write_loss(theta, fixed_write, cost)
        return carry_loss(theta, fixed_carry, cost)

    def check(name, parameter, analytic, numeric, extra=None):
        tolerance = FD_ATOL + FD_RTOL * max(abs(analytic), abs(numeric))
        record = {"kind": "gradient_check", "component": name,
                  "parameter": NAMES[parameter], "analytic": analytic,
                  "central_difference": numeric, "absolute_error": abs(analytic - numeric),
                  "tolerance": tolerance, "passed": abs(analytic - numeric) <= tolerance,
                  "fd_point": dict(zip(NAMES, point))}
        if extra:
            record.update(extra)
        checks.append(record)
        writer.write(record)

    for name in ("task_true_RTRL", "write_conditional", "carry_conditional"):
        _, gradient = evaluate(name, point)
        for j in range(3):
            plus, minus = list(point), list(point)
            plus[j] += FD_STEP
            minus[j] -= FD_STEP
            numeric = (evaluate(name, plus)[0] - evaluate(name, minus)[0]) / (2 * FD_STEP)
            check(name, j, gradient[j], numeric,
                  {"predecessors_frozen_on_both_sides": name != "task_true_RTRL"})
    # First and last observed hold in each original episode; all predecessor values frozen.
    chosen = [(row, sample) for row, sample in zip(fixed_rows, hold_samples)
              if sample["t"] in (2, len(bank[sample["episode"]]["inputs"]))]
    for row, sample in chosen:
        for j in range(3):
            plus, minus = list(point), list(point)
            plus[j] += FD_STEP
            minus[j] -= FD_STEP
            previous, x = sample["previous"], sample["input"]
            numerical = (math.tanh(plus[0] * previous + plus[1] * x)
                         - math.tanh(minus[0] * previous + minus[1] * x)) / (2 * FD_STEP)
            cost["mode_steps"] += 2
            cost["isolated_hold_evaluations"] += 2
            check("HOLD_transition_parameter_Jacobian", j, row[j], numerical,
                  {"predecessor_frozen_on_both_sides": True, "sample": sample})
    if not all(r["passed"] for r in checks):
        raise RuntimeError("finite-difference preflight failed")

    # Test projection at the ACTUAL acquired checkpoint, not a loss-weighted Jacobian.
    actual = replay(initial, bank, cost)
    _, raw_write = write_loss(initial, freeze_write(bank, actual, cuts), cost)
    projected, report, rows, samples = projection_details(initial, bank, actual, raw_write, cost)
    isolated_theta, isolated_update = step(initial, projected)  # Diagnostic only; not retained/trained.
    differences = []
    for sample in samples:
        previous, x = sample["previous"], sample["input"]
        before = math.tanh(initial[0] * previous + initial[1] * x)
        after = math.tanh(isolated_theta[0] * previous + isolated_theta[1] * x)
        differences.append(abs(after - before))
        cost["mode_steps"] += 2
        cost["isolated_hold_evaluations"] += 2
    _, _, per_episode = task(initial, bank, actual, cost)
    maximum_change = max(differences, default=0.0)
    passed = (report["rank"] > 0 and report["annihilation_residual_max"] <= 1e-10
              and maximum_change <= 1e-12)
    writer.write({"kind": "projection_preflight", "projection": report,
                  "unprojected_write_gradient": raw_write, "projected_write_gradient": projected,
                  "isolated_write_update_not_training": isolated_update,
                  "old_two_episode_task_loss": sum(per_episode[:2]) / 2,
                  "hold_preservation_max_abs_on_actual_frozen_predecessors": maximum_change,
                  "model_Jacobian_rows_not_multiplied_by_task_error": True,
                  "passed": passed,
                  "scope": "isolated projected write only; task/carry updates remain unprojected"})
    if not passed:
        raise RuntimeError("projection/isolated-HOLD preflight failed")
    return {"fd_checks": len(checks), "all_passed": True, "projection_preflight": report}


def snapshot(arm, iteration, bank, pairs, cuts, writer):
    report = components(arm["theta"], bank, pairs, cuts, arm["audit_cost"], True, True, True)
    gt, gw, gc = report["task"][1], report["write"][1], report["carry"][1]
    used_write = report["projected_write"] if arm["project"] else gw
    gradient = [gt[j] + (used_write[j] if arm["write"] else 0.0)
                + (gc[j] if arm["carry"] else 0.0) for j in range(3)]
    per_episode = report["per_episode_task"]
    cut_states = []
    for c in cuts:
        i, k = c["episode"], c["cut"]
        tr, y = report["traces"][i]["r"], bank[i]["label"]
        cut_states.append({"episode": i, "cut": k, "previous": tr[k - 1],
                           "input": bank[i]["inputs"][k - 1], "after": tr[k],
                           "endpoint_label_orientation": y, "signed_cut_margin": y * tr[k],
                           "final_state": tr[-1], "final_prediction": arm["theta"][2] * tr[-1]})
    pair_states = []
    for pair in pairs:
        rp, rn = report["traces"][pair["positive"]]["r"], report["traces"][pair["negative"]]["r"]
        indices = [pair["cut"]] + pair["carry_indices"]
        pair_states.append({"positive": pair["positive"], "negative": pair["negative"],
                            "cut": pair["cut"], "cut_separation": abs(rp[indices[0]] - rn[indices[0]]),
                            "minimum_cut_through_end_separation": min(abs(rp[t] - rn[t]) for t in indices),
                            "final_separation": abs(rp[-1] - rn[-1])})
    row = {
        "kind": "checkpoint", "arm": arm["name"], "step": iteration,
        "theta": dict(zip(NAMES, arm["theta"])),
        "losses": {key: report[key][0] for key in ("task", "write", "carry")},
        "component_gradients": {key: report[key][1] for key in ("task", "write", "carry")},
        "projected_write_gradient": report["projected_write"], "hold_projection": report["projection"],
        "active_gradient": gradient, "active_gradient_norm": math.sqrt(sum(g * g for g in gradient)),
        "active_flags": {"write": arm["write"], "carry": arm["carry"], "project_write_only": arm["project"]},
        "old_task_mean_half_MSE": sum(per_episode[:2]) / 2,
        "new_write_task_mean_half_MSE": sum(per_episode[2:]) / 4,
        "per_episode_task_loss": per_episode, "cut_states": cut_states, "pair_states": pair_states,
        "readout_orientation_positive": arm["theta"][2] > 0.0,
        "nonpositive_v_update_count": arm["nonpositive_v_count"],
        "first_nonpositive_v_step": arm["first_nonpositive_v_step"],
        "last_update": arm["last_update"], "optimization_cost": dict(arm["optimization_cost"]),
        "checkpoint_audit_cost": dict(arm["audit_cost"]),
        "inactive_components_audited_not_used_in_updates": True,
    }
    if iteration in (0, ITERATIONS):
        row["full_training_state_traces"] = [tr["r"] for tr in report["traces"]]
    writer.write(row)
    return {"theta": list(arm["theta"]), "old_task_loss": sum(per_episode[:2]) / 2,
            "new_write_task_loss": sum(per_episode[2:]) / 4,
            "all_task_loss": report["task"][0], "cut_states": cut_states,
            "hold_projection": report["projection"], "pair_states": pair_states,
            "readout_orientation_positive": arm["theta"][2] > 0.0}


def program_baseline(bank, writer):
    """Supplied three-program grammar; affects no recurrent input or update."""
    descriptions = {
        "first_nonzero": "return first nonzero input (or zero)",
        "last_nonzero": "scan; replace register on nonzero; return register",
        "sign_product": "multiply all nonzero inputs (or zero if none)",
    }
    costs = {"candidate_tests": 0, "token_reads": 0, "cached_training_label_uses": 0,
             "literal_description_bytes": sum(len(t.encode()) for t in descriptions.values())}

    def run(name, inputs):
        result, seen = 0.0, False
        for x in inputs:
            costs["token_reads"] += 1
            if x == 0.0:
                continue
            if name == "first_nonzero":
                return x
            if name == "last_nonzero":
                result = x
            else:
                result = result * x if seen else x
                seen = True
        return result

    summaries = {}
    for phase, episodes in (("original_two_only", bank[:2]), ("six_endpoint_labels", bank)):
        phase_records = {}
        for name in descriptions:
            predictions = []
            for e in episodes:
                predictions.append(run(name, e["inputs"]))
                costs["candidate_tests"] += 1
                costs["cached_training_label_uses"] += 1
            wrong = sum(prediction != e["label"] for prediction, e in zip(predictions, episodes))
            phase_records[name] = {"predictions": predictions, "wrong_labels": wrong}
        consistent = [name for name, r in phase_records.items() if r["wrong_labels"] == 0]
        summaries[phase] = {"tests": phase_records, "consistent_programs": consistent,
                            "selected_ambiguity_count": len(consistent)}
    writer.write({"kind": "costed_program_grammar_baseline", "supplied_grammar": descriptions,
                  "phase_fits": summaries, "cost": costs,
                  "no_hidden_teacher_or_program_input_to_recurrent_arms": True,
                  "description_bytes_are_not_a_Kolmogorov_or_full_interpreter_cost": True})
    return costs


def make_evaluation_bank():
    bank = []
    for hold in (32, 128, 512):
        for writes in (1, 2, 4):
            for first in (-1.0, 1.0):
                inputs, cues = [], []
                sign = first
                for _ in range(writes):
                    inputs.append(sign)
                    cues.append({"state_index": len(inputs), "cue": sign})
                    inputs.extend([0.0] * hold)
                    sign = -sign
                bank.append({"inputs": inputs, "cues": cues, "hold": hold,
                             "write_count": writes, "first": first,
                             "label": cues[-1]["cue"]})
    return bank


def evaluate_final(arm, evaluation_bank, writer):
    groups = {str(w): [] for w in (1, 2, 4)}
    for episode in evaluation_bank:
        tr = forward(arm["theta"], episode["inputs"], arm["evaluation_cost"], False)["r"]
        answer = arm["theta"][2] * tr[-1]
        latest, margins = None, []
        for x, value in zip(episode["inputs"], tr[1:]):
            if x != 0:
                latest = x
            margins.append(latest * value)  # Evaluation rubric only, never a training state target.
        transitions = [{"cue": c["cue"], "state_index": c["state_index"],
                        "before": tr[c["state_index"] - 1], "after": tr[c["state_index"]]}
                       for c in episode["cues"]]
        sign = 1 if answer > 0 else -1 if answer < 0 else 0
        loss = 0.5 * (answer - episode["label"]) ** 2
        record = {"kind": "evaluation", "arm": arm["name"], "hold": episode["hold"],
                  "write_count": episode["write_count"], "first_cue": episode["first"],
                  "input_count": len(episode["inputs"]), "endpoint_label_last_write": episode["label"],
                  "final_state": tr[-1], "prediction": answer, "half_squared_loss": loss,
                  "noiseless_prediction_sign": sign, "sign_correct": sign == episode["label"],
                  "worst_signed_state_margin_after_latest_cue": min(margins),
                  "final_signed_state_margin": episode["label"] * tr[-1],
                  "maximum_abs_state": max(abs(x) for x in tr),
                  "near_saturation_state_count": sum(abs(x) >= 1.0 - 1e-12 for x in tr[1:]),
                  "cue_boundary_states": transitions, "no_updates": True,
                  "intermediate_signed_margin_is_evaluation_only": True}
        writer.write(record)
        groups[str(episode["write_count"])].append(record)
        guard_time()
    return {group: {"episodes": len(rows),
                    "mean_half_squared_loss": sum(r["half_squared_loss"] for r in rows) / len(rows),
                    "sign_correct": sum(r["sign_correct"] for r in rows),
                    "worst_signed_margin": min(r["worst_signed_state_margin_after_latest_cue"] for r in rows)}
            for group, rows in groups.items()}


def main():
    writer, arms = None, []
    try:
        writer = RawWriter()
        initial, original, source = load_source_once()
        bank = acquire_researcher_fixture(original)
        json.dumps({"bank": bank, "theta": initial}, allow_nan=False)  # Serialization smoke.
        writer.write({"kind": "initial", "source": source,
                      "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "contract_sha256": hashlib.sha256((HERE / "contract.md").read_bytes()).hexdigest(),
                      "training_bank": bank,
                      "settings": {"hold": HOLD, "iterations": ITERATIONS, "lr": LR,
                                   "margin": MARGIN, "global_norm_clip": CLIP, "bounds": BOUNDS,
                                   "rank_tolerance": RANK_TOL, "all_loss_coefficients": 1.0,
                                   "write_loss": "half mean individual oriented hinge squared over 6 cuts",
                                   "carry_loss": "quarter hinge squared mean over 3x64 conditional pairs",
                                   "no_additional_v1_exact_seed_loss": True},
                      "new_training_labels": 4, "old_cached_training_labels": 2,
                      "exploratory_not_confirmation": True})
        pairs, cuts, old, mining_cost = mine(bank)
        writer.write({"kind": "mined_cuts_and_witnesses", "pairs": pairs, "cuts": cuts,
                      "actual_old_h_traces": old, "cost": mining_cost,
                      "carry_term_count": sum(len(p["carry_indices"]) for p in pairs)})
        fd_cost = counter()
        check_summary = derivative_checks(bank, pairs, cuts, initial, writer, fd_cost)
        writer.write({"kind": "preflight_complete", "checks": check_summary, "cost": fd_cost})
        baseline_cost = program_baseline(bank, writer)
        for name, write_flag, carry_flag, project_flag in (
            ("endpoint", False, False, False),
            ("endpoint_local_write", True, False, False),
            ("endpoint_joint_write_carry", True, True, False),
            ("endpoint_projected_write_carry", True, True, True),
        ):
            arm = {"name": name, "theta": list(initial), "write": write_flag,
                   "carry": carry_flag, "project": project_flag, "last_update": None,
                   "optimization_cost": counter(), "audit_cost": counter(), "evaluation_cost": counter(),
                   "nonpositive_v_count": int(initial[2] <= 0.0),
                   "first_nonpositive_v_step": 0 if initial[2] <= 0.0 else None,
                   "rank_zero_training_steps": 0}
            arms.append(arm)
            snapshot(arm, 0, bank, pairs, cuts, writer)
        final_training = {}
        for iteration in range(1, ITERATIONS + 1):
            guard_time()
            for arm in arms:
                report = components(arm["theta"], bank, pairs, cuts, arm["optimization_cost"],
                                    arm["write"], arm["carry"], arm["project"])
                arm["theta"], arm["last_update"] = step(arm["theta"], report["active_gradient"])
                arm["last_update"]["reported_pre_update_loss_sum"] = report[
                    "reported_active_loss_sum_not_gradient_certificate"]
                if arm["project"] and report["projection"]["rank"] == 0:
                    arm["rank_zero_training_steps"] += 1
                if arm["theta"][2] <= 0.0:
                    arm["nonpositive_v_count"] += 1
                    if arm["first_nonpositive_v_step"] is None:
                        arm["first_nonpositive_v_step"] = iteration
            if iteration % CHECKPOINT_EVERY == 0 or iteration == ITERATIONS:
                for arm in arms:
                    final_training[arm["name"]] = snapshot(arm, iteration, bank, pairs, cuts, writer)
        evaluation_bank = make_evaluation_bank()
        if len(evaluation_bank) != 18:
            raise RuntimeError("contracted final evaluation count changed")
        evaluation_summaries = {}
        for arm in arms:
            evaluation_summaries[arm["name"]] = evaluate_final(arm, evaluation_bank, writer)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        writer.write({"kind": "complete", "total_training_parameter_updates": ITERATIONS * len(arms),
                      "updates_per_arm": ITERATIONS, "source": source,
                      "training_label_account": {"old_cached": 2, "new_researcher_generated": 4, "total": 6},
                      "evaluation_label_account": {"shared_evaluation_inputs": 18,
                                                   "targets_not_used_in_training": True},
                      "training_summaries": final_training, "final_evaluation": evaluation_summaries,
                      "mining_cost": mining_cost, "finite_difference_and_projection_check_cost": fd_cost,
                      "program_grammar_cost": baseline_cost,
                      "arm_costs": {a["name"]: {"optimization": a["optimization_cost"],
                                                "checkpoint_audit": a["audit_cost"],
                                                "evaluation": a["evaluation_cost"],
                                                "rank_zero_training_steps": a["rank_zero_training_steps"],
                                                "nonpositive_v_update_count": a["nonpositive_v_count"]}
                                    for a in arms},
                      "raw_bytes_before_complete": writer.bytes, "raw_rows_before_complete": writer.rows,
                      "max_rss_native": usage.ru_maxrss,
                      "max_rss_units": "KiB on Linux; bytes on macOS",
                      "claim_limit": "same-lineage exploratory control diagnostic, not general acquisition/clear/noise result"})
        print(json.dumps({"status": "complete", "raw_name": RAW_PATH.name,
                          "raw_bytes": writer.bytes, "raw_rows": writer.rows}, allow_nan=False))
        return 0
    except Exception as error:
        failure = {"kind": "failure", "error_type": type(error).__name__, "message": str(error),
                   "partial_parameters": {a["name"]: a["theta"] for a in arms}}
        if writer is not None:
            try:
                writer.write(failure, failure=True)
            except Exception:
                pass
        print(json.dumps({"status": "failed", **failure}, allow_nan=False))
        return 1
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    sys.exit(main())
