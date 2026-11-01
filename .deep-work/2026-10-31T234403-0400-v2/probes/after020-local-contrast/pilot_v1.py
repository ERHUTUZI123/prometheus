#!/usr/bin/env python3
"""One contract-bound exploratory run. Developer wrote this file without running it.

No CLI/default selection, network, subprocess, random search, or hidden-state labels.
The external runner owns CPU/wall/filesystem/network isolation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time


# Fixed by root's contract BEFORE execution. A hold counts zero inputs after cue.
TRAIN_HOLD = 64
EVAL_HOLDS = (32, 128, 512)
ITERATIONS = 512
LEARNING_RATE = 0.05
MARGIN = 0.4
GLOBAL_NORM_CAP = 10.0
CHECKPOINT_EVERY = 16
BOUNDS = ((0.0, 2.5), (-4.0, 4.0), (-3.0, 3.0))
MAX_RAW_BYTES = 2 * 1024 * 1024
FAILURE_RESERVE_BYTES = 4096
INTERNAL_WALL_SECONDS = 25.0  # Backup guard; external contract cap is 30 s.
FD_STEP = 1e-6
FD_ATOL = 1e-7
FD_RTOL = 2e-4
PARAMETER_NAMES = ("A", "beta", "v")
HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "pilot-v1.raw.jsonl"
CONTRACT_PATH = HERE / "contract.md"
START_WALL = time.monotonic()
START_CPU = time.process_time()


class BudgetExceeded(RuntimeError):
    pass


def guard_time():
    if time.monotonic() - START_WALL > INTERNAL_WALL_SECONDS:
        raise BudgetExceeded("internal wall guard reached before external 30 s cap")


def object_bytes(value, seen=None):
    """Python object-graph estimate, not process peak RSS or a theoretical bit bound."""
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    result = sys.getsizeof(value)
    if isinstance(value, dict):
        result += sum(object_bytes(k, seen) + object_bytes(v, seen)
                      for k, v in value.items())
    elif isinstance(value, (tuple, list)):
        result += sum(object_bytes(v, seen) for v in value)
    return result


class RawWriter:
    def __init__(self):
        # Refuse an accidental second invocation/overwrite of a partial run.
        self.handle = RAW_PATH.open("xb")
        self.bytes = 0
        self.rows = 0

    def write(self, record, failure=False):
        record = dict(record)
        record["wall_seconds"] = time.monotonic() - START_WALL
        record["cpu_seconds"] = time.process_time() - START_CPU
        payload = (json.dumps(record, allow_nan=False, separators=(",", ":"))
                   + "\n").encode("utf-8")
        limit = MAX_RAW_BYTES if failure else MAX_RAW_BYTES - FAILURE_RESERVE_BYTES
        if self.bytes + len(payload) >= limit:
            raise BudgetExceeded("raw JSONL byte cap would be exceeded")
        self.handle.write(payload)
        self.handle.flush()  # Initial, every checkpoint, and every other row too.
        self.bytes += len(payload)
        self.rows += 1

    def close(self):
        self.handle.close()


def new_counter():
    return {
        "mode_steps": 0,
        "rtrl_sensitivity_steps": 0,
        "local_detached_steps": 0,
        "episode_replays": 0,
        "cached_endpoint_label_uses": 0,
        "seed_pair_uses": 0,
        "carry_pair_uses": 0,
        "objective_evaluations": 0,
    }


def make_bank():
    # Endpoint targets are the two labels specified by the contract. No r_t label.
    return [
        {"name": "positive_cue", "inputs": [1.0] + [0.0] * TRAIN_HOLD,
         "label": 1.0, "context": "cue_endpoint"},
        {"name": "negative_cue", "inputs": [-1.0] + [0.0] * TRAIN_HOLD,
         "label": -1.0, "context": "cue_endpoint"},
    ]


def mine_witness(bank):
    """Actually evaluate h'=x, compare complete remaining words, find first merger."""
    cost = {
        "incumbent_transitions": 0,
        "aligned_checkpoint_checks": 0,
        "full_state_equality_checks": 0,
        "future_token_comparisons": 0,
        "endpoint_labels_received": len(bank),
        "retained_input_tokens": sum(len(e["inputs"]) for e in bank),
    }
    states = []
    for episode in bank:
        trace = [0.0]
        for token in episode["inputs"]:
            trace.append(token)  # Full incumbent state in this contract is h=x.
            cost["incumbent_transitions"] += 1
        states.append(trace)
    left, right = bank
    if left["context"] != right["context"] or left["label"] == right["label"]:
        raise RuntimeError("fixture does not supply a valid same-context label contrast")
    if len(left["inputs"]) != len(right["inputs"]):
        raise RuntimeError("this pilot requires aligned equal-length episodes")
    total = len(left["inputs"])
    tau = None
    for checkpoint in range(total + 1):
        cost["aligned_checkpoint_checks"] += 1
        cost["full_state_equality_checks"] += 1
        if states[0][checkpoint] != states[1][checkpoint]:
            continue
        equal_future = True
        for index in range(checkpoint, total):
            cost["future_token_comparisons"] += 1
            if left["inputs"][index] != right["inputs"][index]:
                equal_future = False
                break
        if equal_future:
            tau = checkpoint
            break
    if tau is None or tau >= total:
        raise RuntimeError("no nonterminal exact merger with a complete common future")
    positive = 0 if left["label"] > right["label"] else 1
    negative = 1 - positive
    witness = {
        "tau": tau,
        "total_inputs": total,
        "positive_index": positive,
        "negative_index": negative,
        "old_state_at_merger": states[0][tau],
        "remaining_word": list(left["inputs"][tau:]),
        "carry_transition_indices": list(range(tau + 1, total + 1)),
        "carry_terms": total - tau,
        "orientation_origin": "ordered already-observed endpoint labels; design prior",
        "full_future_and_context_checked": True,
    }
    cost["retained_python_object_bytes_estimate"] = object_bytes((bank, states, witness))
    return witness, states, cost


def forward(theta, inputs, counter, sensitivities=True):
    """Analytic RTRL for r'=tanh(A*r+beta*x); v is only in the readout."""
    A, beta, _ = theta
    states, sensitivity_A, sensitivity_beta = [0.0], [0.0], [0.0]
    r = dA = dbeta = 0.0
    counter["episode_replays"] += 1
    for token in inputs:
        previous = r
        r = math.tanh(A * previous + beta * token)
        counter["mode_steps"] += 1
        if sensitivities:
            slope = 1.0 - r * r
            dA = slope * (previous + A * dA)
            dbeta = slope * (token + A * dbeta)
            counter["rtrl_sensitivity_steps"] += 1
        states.append(r)
        if sensitivities:
            sensitivity_A.append(dA)
            sensitivity_beta.append(dbeta)
    return {"r": states, "dA": sensitivity_A, "dbeta": sensitivity_beta}


def paired_forward(theta, bank, counter):
    return [forward(theta, e["inputs"], counter) for e in bank]


def task_component(theta, bank, traces, counter):
    v = theta[2]
    loss, gradient = 0.0, [0.0, 0.0, 0.0]
    for episode, trace in zip(bank, traces):
        final = trace["r"][-1]
        error = v * final - episode["label"]
        loss += 0.5 * error * error / len(bank)
        gradient[0] += error * v * trace["dA"][-1] / len(bank)
        gradient[1] += error * v * trace["dbeta"][-1] / len(bank)
        gradient[2] += error * final / len(bank)
        counter["cached_endpoint_label_uses"] += 1
    counter["objective_evaluations"] += 1
    return loss, gradient


def seed_component(traces, witness, counter):
    tau = witness["tau"]
    loss, gradient = 0.0, [0.0, 0.0, 0.0]
    for index, target in ((witness["positive_index"], MARGIN),
                          (witness["negative_index"], -MARGIN)):
        trace = traces[index]
        error = trace["r"][tau] - target
        # Equation (2): mean SQUARED loss, not mean HALF-squared loss.
        loss += 0.5 * error * error
        gradient[0] += error * trace["dA"][tau]
        gradient[1] += error * trace["dbeta"][tau]
    counter["seed_pair_uses"] += 1
    counter["objective_evaluations"] += 1
    return loss, gradient


def freeze_carry_pairs(bank, traces, witness):
    positive, negative = witness["positive_index"], witness["negative_index"]
    # Strictly AFTER the merger, through terminal T. For this fixture: 63 terms.
    return [
        (traces[positive]["r"][t - 1], traces[negative]["r"][t - 1],
         bank[positive]["inputs"][t - 1], bank[negative]["inputs"][t - 1])
        for t in witness["carry_transition_indices"]
    ]


def carry_from_frozen(theta, frozen_pairs, counter):
    """Conditional gradient: predecessors are constants, including during FD."""
    if not frozen_pairs:
        raise RuntimeError("carry requires at least one observed future transition")
    A, beta, _ = theta
    loss, gradient = 0.0, [0.0, 0.0, 0.0]
    count = len(frozen_pairs)
    for previous_p, previous_n, token_p, token_n in frozen_pairs:
        rp = math.tanh(A * previous_p + beta * token_p)
        rn = math.tanh(A * previous_n + beta * token_n)
        counter["mode_steps"] += 2
        counter["local_detached_steps"] += 2
        delta = rp - rn
        gap = 2.0 * MARGIN - abs(delta)
        if gap > 0.0:
            # Quarter-normalized Eq(3), averaged over the observed common future.
            loss += 0.25 * gap * gap / count
            # At exact delta=0 use subgradient 0; the seed supplies symmetry breaking.
            direction = 1.0 if delta > 0.0 else -1.0 if delta < 0.0 else 0.0
            derivative_delta = -0.5 * gap * direction / count
            slope_p, slope_n = 1.0 - rp * rp, 1.0 - rn * rn
            gradient[0] += derivative_delta * (
                slope_p * previous_p - slope_n * previous_n)
            gradient[1] += derivative_delta * (
                slope_p * token_p - slope_n * token_n)
        counter["carry_pair_uses"] += 1
    counter["objective_evaluations"] += 1
    return loss, gradient


def components(theta, bank, witness, counter, include_seed, include_carry):
    traces = paired_forward(theta, bank, counter)
    task = task_component(theta, bank, traces, counter)
    seed = seed_component(traces, witness, counter) if include_seed else None
    carry = (carry_from_frozen(theta, freeze_carry_pairs(bank, traces, witness), counter)
             if include_carry else None)
    included = [task] + ([seed] if seed is not None else []) + (
        [carry] if carry is not None else [])
    total_loss = sum(item[0] for item in included)
    total_gradient = [sum(item[1][j] for item in included) for j in range(3)]
    return {"task": task, "seed": seed, "carry": carry,
            "total_loss": total_loss, "total_gradient": total_gradient,
            "traces": traces}


def gradient_checks(bank, witness, writer, counter):
    # Interior point, independent of all trained outcomes. No clipping/projection in FD.
    theta = [1.05, 0.12, 0.8]
    traces = paired_forward(theta, bank, counter)
    frozen_pairs = freeze_carry_pairs(bank, traces, witness)

    def value_gradient(name, point):
        if name == "carry_conditional_frozen_predecessors":
            return carry_from_frozen(point, frozen_pairs, counter)
        actual = paired_forward(point, bank, counter)
        if name == "task_true":
            return task_component(point, bank, actual, counter)
        return seed_component(actual, witness, counter)

    records = []
    for name in ("task_true", "seed_true", "carry_conditional_frozen_predecessors"):
        value, analytic = value_gradient(name, theta)
        for index, parameter in enumerate(PARAMETER_NAMES):
            plus, minus = list(theta), list(theta)
            plus[index] += FD_STEP
            minus[index] -= FD_STEP
            numeric = (value_gradient(name, plus)[0] -
                       value_gradient(name, minus)[0]) / (2.0 * FD_STEP)
            difference = abs(analytic[index] - numeric)
            tolerance = FD_ATOL + FD_RTOL * max(abs(analytic[index]), abs(numeric))
            passed = difference <= tolerance
            record = {"component": name, "parameter": parameter,
                      "theta": theta, "value": value, "analytic": analytic[index],
                      "central_difference": numeric, "absolute_error": difference,
                      "tolerance": tolerance, "passed": passed,
                      "predecessors_frozen_for_both_fd_sides": name.startswith("carry")}
            records.append(record)
            writer.write({"kind": "gradient_check", **record})
    if not all(record["passed"] for record in records):
        raise RuntimeError("pre-training analytic/finite-difference check failed")
    return records


def update(theta, gradient):
    if not all(math.isfinite(value) for value in gradient):
        raise RuntimeError("nonfinite training gradient")
    norm = math.sqrt(sum(value * value for value in gradient))
    scale = min(1.0, GLOBAL_NORM_CAP / norm) if norm else 1.0
    unclipped = [theta[j] - LEARNING_RATE * scale * gradient[j] for j in range(3)]
    projected = [min(BOUNDS[j][1], max(BOUNDS[j][0], unclipped[j]))
                 for j in range(3)]
    return projected, {"unclipped_gradient_norm": norm, "clip_scale": scale,
                       "gradient": list(gradient), "pre_update_theta": list(theta),
                       "post_update_theta": projected,
                       "projected_parameters": [PARAMETER_NAMES[j] for j in range(3)
                                                if projected[j] != unclipped[j]]}


def checkpoint(arm, step, bank, witness, writer):
    # Full diagnostic components at checkpoints, even for inactive auxiliary losses.
    # They NEVER enter an arm's update unless its arm flags include that component.
    report = components(arm["theta"], bank, witness, arm["audit_cost"], True, True)
    enabled = ["task"] + (["seed"] if arm["seed"] else []) + (
        ["carry"] if arm["carry"] else [])
    total_gradient = [sum(report[key][1][j] for key in enabled) for j in range(3)]
    tau = witness["tau"]
    positive, negative = witness["positive_index"], witness["negative_index"]
    rp, rn = report["traces"][positive]["r"], report["traces"][negative]["r"]
    separations = [abs(rp[t] - rn[t]) for t in range(tau, len(rp))]
    writer.write({
        "kind": "checkpoint", "arm": arm["name"], "step": step,
        "theta": dict(zip(PARAMETER_NAMES, arm["theta"])),
        "enabled_components": enabled,
        "losses": {key: report[key][0] for key in ("task", "seed", "carry")},
        "component_gradients": {key: dict(zip(PARAMETER_NAMES, report[key][1]))
                                for key in ("task", "seed", "carry")},
        "active_total_loss": sum(report[key][0] for key in enabled),
        "active_gradient": total_gradient,
        "active_gradient_norm": math.sqrt(sum(g * g for g in total_gradient)),
        "training_mode_traces_including_reset": [t["r"] for t in report["traces"]],
        "merger_separation": separations[0], "endpoint_separation": separations[-1],
        "minimum_separation_from_merger": min(separations),
        "last_update": arm["last_update"],
        "optimization_cost": dict(arm["optimization_cost"]),
        "checkpoint_audit_cost": dict(arm["audit_cost"]),
        "note": "inactive auxiliary diagnostics do not alter updates; charged separately",
    })
    return {"theta": list(arm["theta"]), "task_loss": report["task"][0],
            "endpoint_separation": separations[-1],
            "minimum_separation_from_merger": min(separations)}


def final_evaluation(arm, writer):
    """Exactly 12 no-update episodes per arm; no endpoint labels acquired here."""
    results = []
    theta = arm["theta"]
    for hold in EVAL_HOLDS:
        for first in (1.0, -1.0):
            for family in ("single_cue", "changed_write"):
                if family == "single_cue":
                    inputs = [first] + [0.0] * hold
                    last = first
                    write_index = None
                else:
                    last = -first
                    inputs = [first] + [0.0] * hold + [last] + [0.0] * hold
                    write_index = hold + 2  # One-based state index after second cue.
                trace = forward(theta, inputs, arm["evaluation_cost"], False)["r"]
                answer = theta[2] * trace[-1]
                record = {
                    "kind": "evaluation", "arm": arm["name"], "family": family,
                    "neutral_hold": hold, "input_count": len(inputs),
                    "first_cue": first, "last_nonzero_input": last,
                    "r_final": trace[-1], "prediction": answer,
                    "prediction_sign": 1 if answer > 0 else -1 if answer < 0 else 0,
                    "first_cue_target": first, "last_write_target": last,
                    "half_squared_error_first_cue": 0.5 * (answer - first) ** 2,
                    "half_squared_error_last_write": 0.5 * (answer - last) ** 2,
                    "r_before_changed_write": (trace[write_index - 1]
                                               if write_index is not None else None),
                    "r_after_changed_write": (trace[write_index]
                                              if write_index is not None else None),
                    "no_updates": True,
                    "interpretation": ("single-cue training does not identify first-vs-last "
                                       "write rule; both are reported without selecting one"),
                }
                results.append(record)
                writer.write(record)
                guard_time()
    if len(results) != 12:
        raise RuntimeError("declared evaluation-episode count changed")
    return results


def main():
    writer = None
    arms = []
    try:
        writer = RawWriter()
        bank = make_bank()
        # Serialization smoke occurs before gradients/learning; no hidden temporary file.
        json.dumps({"bank": bank, "test_numbers": [0.0, 0.4, 1.0]}, allow_nan=False)
        writer.write({
            "kind": "initial", "exploratory_only": True,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "contract_sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
            "settings": {"train_hold": TRAIN_HOLD, "iterations_per_arm": ITERATIONS,
                         "learning_rate": LEARNING_RATE, "margin": MARGIN,
                         "gradient_norm_cap": GLOBAL_NORM_CAP, "bounds": BOUNDS,
                         "eval_holds": EVAL_HOLDS, "checkpoint_every": CHECKPOINT_EVERY,
                         "carry_normalization": "mean quarter-normalized Eq3",
                         "auxiliary_coefficients": {"seed": 1.0, "carry": 1.0}},
            "training_bank": bank,
            "information": "two cached endpoint labels; no supplied intermediate mode labels",
        })
        witness, old_states, mining_cost = mine_witness(bank)
        writer.write({"kind": "mined_witness", "witness": witness,
                      "incumbent_state_traces": old_states, "mining_cost": mining_cost})
        fd_cost = new_counter()
        checks = gradient_checks(bank, witness, writer, fd_cost)
        writer.write({"kind": "gradient_checks_complete", "all_passed": True,
                      "checks": len(checks), "finite_difference_cost": fd_cost})
        for name, start_A, seed, carry in (
            ("endpoint_only", 0.5, False, False),
            ("endpoint_plus_prefix", 0.5, True, False),
            ("endpoint_plus_prefix_plus_local_carry", 0.5, True, True),
            ("endpoint_persistent_initialization_control", 1.2, False, False),
        ):
            arm = {"name": name, "theta": [start_A, 0.01, 1.0],
                   "seed": seed, "carry": carry, "last_update": None,
                   "optimization_cost": new_counter(), "audit_cost": new_counter(),
                   "evaluation_cost": new_counter()}
            arms.append(arm)
            checkpoint(arm, 0, bank, witness, writer)
        latest = {}
        for step in range(1, ITERATIONS + 1):
            guard_time()
            for arm in arms:
                result = components(arm["theta"], bank, witness,
                                    arm["optimization_cost"], arm["seed"], arm["carry"])
                arm["theta"], arm["last_update"] = update(
                    arm["theta"], result["total_gradient"])
                arm["last_update"]["pre_update_active_loss"] = result["total_loss"]
            if step % CHECKPOINT_EVERY == 0 or step == ITERATIONS:
                for arm in arms:
                    latest[arm["name"]] = checkpoint(
                        arm, step, bank, witness, writer)
        for arm in arms:
            guard_time()
            final_evaluation(arm, writer)
        usage = resource.getrusage(resource.RUSAGE_SELF)
        writer.write({
            "kind": "complete", "updates_per_arm": ITERATIONS,
            "total_parameter_updates": ITERATIONS * len(arms),
            "final_training_summaries": latest, "mining_cost": mining_cost,
            "finite_difference_cost": fd_cost,
            "arm_costs": {arm["name"]: {
                "optimization": arm["optimization_cost"],
                "checkpoint_audit": arm["audit_cost"],
                "evaluation": arm["evaluation_cost"],
                "evaluation_episodes": arm["evaluation_cost"]["episode_replays"],
            } for arm in arms},
            "raw_bytes_before_complete_row": writer.bytes,
            "raw_rows_before_complete_row": writer.rows,
            "max_rss_native": usage.ru_maxrss,
            "max_rss_unit_note": "platform-native: KiB on Linux, bytes on macOS",
            "claim_limit": "exploratory credit diagnostic; no learned-overwrite/reset claim",
        })
        print(json.dumps({"status": "complete", "raw_name": RAW_PATH.name,
                          "raw_bytes": writer.bytes, "raw_rows": writer.rows},
                         allow_nan=False))
        return 0
    except Exception as error:
        failure = {"kind": "failure", "error_type": type(error).__name__,
                   "message": str(error), "partial_rows_retained": writer is not None,
                   "partial_arm_parameters": {a["name"]: a["theta"] for a in arms}}
        if writer is not None:
            try:
                writer.write(failure, failure=True)
            except Exception:
                pass  # External runner must preserve any earlier flushed rows.
        print(json.dumps({"status": "failed", **failure}, allow_nan=False))
        return 1
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    sys.exit(main())
