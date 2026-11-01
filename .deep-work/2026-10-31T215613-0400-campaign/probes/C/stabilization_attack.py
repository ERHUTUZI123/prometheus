"""Bounded algebra/numerical attacks on candidate prerequisites, not a learning run.

No network, randomness, scientific dependencies, or host interaction.
Run with python3; output is complete and deterministic.
"""
import json
import math


def mean_homeostasis():
    inputs = [-1.0, 1.0]
    target = inputs
    corrupted = [-x for x in inputs]
    return {
        "inputs": inputs,
        "target_outputs": target,
        "corrupted_outputs": corrupted,
        "target_mean": sum(target) / len(target),
        "corrupted_mean": sum(corrupted) / len(corrupted),
        "mean_activity_error": abs(sum(target) - sum(corrupted)) / len(target),
        "functional_mse": sum((a-b)**2 for a, b in zip(target, corrupted)) / len(target),
        "interpretation": "Zero mean-activity error does not distinguish an identity computation from its sign reversal.",
    }


def scalar_accommodation(a, d, duration=50.0, dt=0.001, bound=None):
    # y=a*m+d; dm/dt=(1-y).  Euler discretization; analytical stable rate a>0.
    m = 0.0
    for _ in range(round(duration/dt)):
        y = a*m+d
        m += dt*(1.0-y)
        if bound is not None:
            m = max(-bound, min(bound, m))
    y = a*m+d
    return {"a": a, "d": d, "bound": bound, "m": m, "y": y,
            "final_error": 1.0-y,
            "analytical_required_m": (1.0-d)/a,
            "unbounded_linear_stable": a > 0}


def finite_trace_counterexample():
    # One write of v=1 at t=0, then a hold and a read. Unit retention ideal.
    a = 0.999
    trained_holds = [1, 2, 4, 8, 16]
    heldout_holds = [100, 1000, 10000]
    relative_tolerance = 0.02
    train = [{"hold": n, "leaky_output": a**n, "abs_error": 1-a**n}
             for n in trained_holds]
    unseen = [{"hold": n, "leaky_output": a**n, "abs_error": 1-a**n}
              for n in heldout_holds]
    return {"leak": a, "relative_tolerance": relative_tolerance,
            "development": train, "heldout": unseen,
            "accepted_on_development": all(x["abs_error"] <= relative_tolerance for x in train),
            "passes_all_heldout": all(x["abs_error"] <= relative_tolerance for x in unseen),
            "interpretation": "A finite short-trace contract can accept an implementation that catastrophically loses longer-term memory."}


result = {
    "scope": "Prerequisite attack only; no candidate learning, neural training, graph construction or scientific performance comparison executed.",
    "mean_activity": mean_homeostasis(),
    "control_authority": [scalar_accommodation(0.2, 0.4),
                           scalar_accommodation(-0.2, 0.4),
                           scalar_accommodation(0.2, 0.0, bound=2.0)],
    "finite_trace": finite_trace_counterexample(),
}
assert result["mean_activity"]["mean_activity_error"] == 0.0
assert result["mean_activity"]["functional_mse"] == 4.0
assert abs(result["control_authority"][0]["final_error"]) < 0.0001
assert abs(result["control_authority"][1]["final_error"]) > 1000
assert abs(result["control_authority"][2]["final_error"]-0.6) < 1e-12
assert result["finite_trace"]["accepted_on_development"]
assert not result["finite_trace"]["passes_all_heldout"]
print(json.dumps(result, indent=2))
