#!/usr/bin/env python3
"""Frozen BF discrimination: positive learned mobility vs unsafe potential.

Inspected, offline, no subprocesses, no network, no credential access. Writes only
next to this script. No adaptive hyperparameter or channel selection.
"""
import json
import math
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
CONFIG = dict(seed=20261031, n=4, train_count=16, test_count=64,
              step=0.025, train_horizon=60.0, test_horizon=180.0,
              sample_times=[25., 30., 35., 40., 45., 50., 55., 60.],
              eta=0.2, beta=0.5, tolerance=1e-6,
              refinement_indices=[0, 1, 2, 3], outer_timeout_seconds=90)


def wrap(x):
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def residual(y):
    e = y.copy()
    e[:, 1:] -= y[:, :-1]
    return wrap(e)


def parent_force(y):
    # Four sines; six scalar subtractions per trajectory. Negation is a sign bit.
    e = y.copy()
    e[:, 1:] -= y[:, :-1]
    s = np.sin(e)
    f = -s
    f[:, :-1] += s[:, 1:]
    return f


def field(y, kind, v):
    f = parent_force(y)
    if kind == "mobility":
        # Written in precisely the low-rank form charged in v1.
        dot = np.sum(f * v[None, :], axis=1)
        f = 0.5 * f + dot[:, None] * field.twice_v[None, :]
    elif kind == "shortcut":
        f[:, -1] -= np.sin(y[:, -1])
        f *= 0.8
    return f


def rk4(y, dt, kind, v):
    a = field(y, kind, v)
    b = field(y + (dt / 2) * a, kind, v)
    c = field(y + (dt / 2) * b, kind, v)
    d = field(y + dt * c, kind, v)
    return y + (dt / 6) * (a + 2 * b + 2 * c + d)


def learn_channel(rng):
    starts = rng.uniform(-np.pi, np.pi, (16, 4))
    v = rng.normal(size=4)
    v /= np.linalg.norm(v)
    v_initial = v.copy()
    y = starts.copy()
    snapshots = []
    times_by_step = {round(t / CONFIG["step"]): t
                     for t in CONFIG["sample_times"]}
    for k in range(1, round(60 / CONFIG["step"]) + 1):
        y = rk4(y, CONFIG["step"], "tree", v)
        if k in times_by_step:
            snapshots.append(parent_force(y))
    # Trajectory-major, then chronological. No best-of selection.
    sample_array = np.stack(snapshots, axis=1)
    nonzero = 0
    for f in sample_array.reshape((-1, 4)):
        norm = np.linalg.norm(f)
        if norm > 0:
            u = f / norm
            v += 0.2 * u * np.dot(u, v)
            v /= np.linalg.norm(v)
            nonzero += 1
    return v, dict(starts=starts.tolist(), initial_v=v_initial.tolist(),
                   force_samples=sample_array.tolist(), updates=nonzero,
                   rk4_steps_per_trajectory=2400,
                   integration_force_evaluations=16 * 2400 * 4,
                   diagnostic_force_evaluations=16 * 8,
                   final_train_max_residual=np.max(np.abs(residual(y)), axis=1).tolist())


def solve(starts, kind, v, dt=0.025):
    y = np.array(starts, dtype=float).copy()
    count = len(y)
    steps = np.zeros(count, dtype=int)
    settled = np.zeros(count, dtype=bool)
    times = np.full(count, np.nan)
    max_steps = round(CONFIG["test_horizon"] / dt)
    for k in range(max_steps + 1):
        err = np.max(np.abs(residual(y)), axis=1)
        newly = (~settled) & (err < CONFIG["tolerance"])
        settled[newly] = True
        times[newly] = k * dt
        active = ~settled
        if not np.any(active) or k == max_steps:
            break
        y[active] = rk4(y[active], dt, kind, v)
        steps[active] += 1
    return dict(kind=kind, step=dt, starts=np.array(starts).tolist(),
                final_y=y.tolist(), final_wrapped_y=wrap(y).tolist(),
                final_max_parent_residual=np.max(np.abs(residual(y)), axis=1).tolist(),
                final_output_error=np.abs(wrap(y[:, -1])).tolist(),
                success=settled.tolist(),
                settling_times=[float(t) if np.isfinite(t) else None for t in times],
                rk4_steps=steps.tolist(), force_evaluations=(4 * steps).tolist(),
                total_force_evaluations=int(4 * np.sum(steps)),
                total_rk4_steps=int(np.sum(steps)),
                censored_count=int(np.sum(~settled)))


def save(name, obj):
    with (OUT / name).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def main():
    began = time.monotonic()
    save("config.json", CONFIG)
    rng = np.random.default_rng(CONFIG["seed"])
    v, training = learn_channel(rng)
    field.twice_v = 2.0 * v  # Four charged one-time scalar multiplies, 4 cache scalars.
    starts = rng.uniform(-np.pi, np.pi, (64, 4))
    # All learned quantities are now fixed. No spectral result entered learning.
    tree = solve(starts, "tree", v)
    mobile = solve(starts, "mobility", v)
    alpha = 2 * np.pi / 5
    winding_start = (np.arange(1, 5) * alpha
                     + np.array([0.02, -0.01, 0.015, -0.005]))[None, :]
    winding = {kind: solve(winding_start, kind, v)
               for kind in ("tree", "mobility", "shortcut")}
    refine_starts = np.concatenate([winding_start, starts[[0, 1, 2, 3]]])
    coarse_ref = {kind: solve(refine_starts, kind, v)
                  for kind in ("tree", "mobility")}
    fine_ref = {kind: solve(refine_starts, kind, v, 0.0125)
                for kind in ("tree", "mobility")}
    refinement = {}
    for kind in coarse_ref:
        a, b = coarse_ref[kind], fine_ref[kind]
        refinement[kind] = dict(
            coarse_times=a["settling_times"], fine_times=b["settling_times"],
            max_time_difference=float(np.max(np.abs(np.array(a["settling_times"])
                                                      - np.array(b["settling_times"])))),
            max_final_wrapped_difference=float(np.max(np.abs(wrap(
                np.array(a["final_y"]) - np.array(b["final_y"]))))))

    D = np.eye(4)
    D[np.arange(1, 4), np.arange(3)] = -1
    H = D.T @ D
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    M = 0.5 * np.eye(4) + 2 * np.outer(v, v)
    local_mobile_rates = np.sort(np.linalg.eigvals(M @ H).real)
    tree_times = np.array(tree["settling_times"])
    mobile_times = np.array(mobile["settling_times"])
    base_calls = tree["total_force_evaluations"]
    mobile_calls = mobile["total_force_evaluations"]
    train_calls = (training["integration_force_evaluations"]
                   + training["diagnostic_force_evaluations"])
    # Each RK4 trajectory-step additionally uses 24 scalar multiplications and
    # 28 additions in this implementation (coefficient setup and
    # diagnostic wrapping excluded equally). Force accounting is also exposed
    # separately so conclusions need not depend on this implementational model.
    force_costs = {"tree": dict(sin=4, mul=0, add=6),
                   "mobility": dict(sin=4, mul=12, add=13),
                   "shortcut": dict(sin=5, mul=4, add=7)}
    weighted = {}
    for sine_weight in (1, 4, 16):
        base_one = 6 + 4 * sine_weight
        mobile_one = 25 + 4 * sine_weight
        actual_base = base_calls * base_one
        actual_mobile = mobile_calls * mobile_one
        weighted[str(sine_weight)] = dict(
            force_only_tree_work=actual_base,
            force_only_mobility_inference_work=actual_mobile,
            force_only_mobility_plus_training_work=actual_mobile + train_calls * base_one,
            inference_work_ratio=actual_mobile / actual_base,
            including_training_work_ratio=(actual_mobile + train_calls * base_one) / actual_base,
            integrator_included_inference_ratio=(actual_mobile + 52 * mobile["total_rk4_steps"])
                                               / (actual_base + 52 * tree["total_rk4_steps"]),
            force_only_query_break_even_lower_bound=(64 * train_calls * base_one / (actual_base - actual_mobile)
                                                     if actual_base > actual_mobile else None))

    summary = dict(
        actual_execution="isolated by run_isolated.py; no external services in numerical code",
        versions=dict(numpy=np.__version__), learned_v=v.tolist(), mobility=M.tolist(),
        trace=float(np.trace(M)), min_mobility_eigenvalue=float(np.linalg.eigvalsh(M)[0]),
        absolute_alignment_to_slow_parent_mode=float(abs(np.dot(v, eigenvectors[:, 0]))),
        parent_local_rates=eigenvalues.tolist(), mobile_local_rates=local_mobile_rates.tolist(),
        tree_success_count=int(sum(tree["success"])),
        mobility_success_count=int(sum(mobile["success"])),
        tree_median_settling_time=float(np.median(tree_times)),
        mobility_median_settling_time=float(np.median(mobile_times)),
        tree_total_force_evaluations=base_calls,
        mobility_total_force_evaluations=mobile_calls,
        training_total_force_evaluations=train_calls,
        inference_force_evaluation_ratio=mobile_calls/base_calls,
        improved_query_count=int(np.sum(mobile_times < tree_times)),
        winding={k: dict(time=x["settling_times"][0],
                         output_error=x["final_output_error"][0],
                         max_parent_residual=x["final_max_parent_residual"][0],
                         final_wrapped_y=x["final_wrapped_y"][0])
                 for k, x in winding.items()},
        refinement=refinement, force_costs=force_costs,
        weighted_cost_sensitivity=weighted,
        oja_arithmetic=dict(updates=training["updates"], per_update=dict(mul=20, add=13, divide=8, sqrt=2)),
        other_accounting="Rank-one cache construction: 4 scalar multiplications once. RK4 stage-combination arithmetic: 24 multiplications + 28 additions per trajectory-step. Every settling diagnostic needs residual wrapping (4 modulo operations and 11 adds/subtracts), 4 absolute values and 3 maximum comparisons. Force-only cost sensitivity excludes these diagnostics, Oja arithmetic, coefficient setup and all source/probe overhead; it is not wall-clock or hardware energy.",
        communication="Every mobility force: 4-to-1 reduction and 1-to-4 broadcast; two logical rounds; 8 scalar carrier/hub links. Oja updates: two norm reductions plus one dot reduction per sample, and broadcasts; input force samples require 4 scalars each.",
        storage="4 persistent v scalars plus 4 cached twice-v scalars, 128 four-component samples in this offline audit (512 scalars); streaming Oja can avoid the audit sample buffer.",
        elapsed_seconds=time.monotonic()-began)
    save("training-records.json", training)
    save("query-records.json", dict(tree=tree, mobility=mobile, winding=winding,
                                    refinement_coarse=coarse_ref, refinement_fine=fine_ref))
    save("results.json", summary)
    # Acceptance touches advertised functional computation and central attack;
    # no positive speed/work result is forced by an assertion.
    assert all(tree["success"]), "original tree failed its finite solver check"
    assert all(mobile["success"]), "mobility failed its finite solver check"
    assert winding["shortcut"]["censored_count"] == 1
    assert winding["shortcut"]["final_output_error"][0] > 1.0
    assert winding["tree"]["success"][0] and winding["mobility"]["success"][0]
    assert max(x["max_time_difference"] for x in refinement.values()) < 0.1
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
