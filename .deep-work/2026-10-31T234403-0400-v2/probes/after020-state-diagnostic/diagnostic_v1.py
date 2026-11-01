"""Bounded exploratory state/credit assay. Supplied dynamics, no training."""
from fractions import Fraction
import json
import math
from pathlib import Path
import time

OUT = Path(__file__).resolve().parent
DELAYS = (0, 1, 4, 16, 64, 256, 1024)
KAPPA = 2.0
SIGMA = 0.1
DELTA = 0.001
started = time.monotonic()
updates = 0

a = 0.9
fixed_iterations = 0
for fixed_iterations in range(1, 201):
    nxt = math.tanh(KAPPA * a)
    if abs(nxt - a) < 1e-15:
        a = nxt
        break
    a = nxt
assert abs(a - math.tanh(KAPPA * a)) < 1e-14
r = KAPPA * (1.0 - a * a)
assert 0.0 < r < 1.0


def bayes_error(mean):
    # Exact formula evaluated numerically: final-cut observation noise ONLY.
    return 0.5 * math.erfc(abs(mean) / (math.sqrt(2.0) * SIGMA))


def evolve(kind, sign, delay):
    global updates
    h = sign * a
    product = 1.0
    log_product = 0.0
    for _ in range(delay):
        if kind == "bistable":
            h = math.tanh(KAPPA * h)
            derivative = KAPPA * (1.0 - h * h)
        elif kind == "matched_fading":
            h *= r
            derivative = r
        else:
            raise ValueError(kind)
        product *= derivative
        log_product += math.log(derivative)
        updates += 1
    return h, product, log_product


records = []
with (OUT / "raw.jsonl").open("x") as raw:
    def emit(row):
        assert len(records) < 100
        raw.write(json.dumps(row, allow_nan=False) + "\n")
        raw.flush()
        records.append(row)

    for delay in DELAYS:
        for kind in ("bistable", "matched_fading"):
            pos, jp, ljp = evolve(kind, +1, delay)
            neg, jn, ljn = evolve(kind, -1, delay)
            qp = DELTA * round(pos / DELTA)
            qn = DELTA * round(neg / DELTA)
            err = bayes_error(pos)
            emit({
                "kind": kind, "delay": delay,
                "state_plus": pos, "state_minus": neg,
                "separation": abs(pos - neg),
                "runtime_float_alias": pos == neg,
                "jacobian_plus": jp, "jacobian_minus": jn,
                "log_abs_jacobian_plus": ljp,
                "log_abs_jacobian_minus": ljn,
                "jacobian_underflow": jp == 0.0 and delay > 0,
                "computed_query_noise_bayes_error": err,
                "computed_inverted_readout_error": 1.0 - err,
                "query_cut_rounded_plus": qp,
                "query_cut_rounded_minus": qn,
                "query_cut_rounding_alias": qp == qn,
                "tiny_tangent_trigger": ljp < math.log(1e-8),
                "noise_information_deficit_trigger": err > 0.25,
                "interpretation": "supplied frozen dynamics; no learned capability",
            })
        integer_pos = 1000
        integer_neg = -1000
        slow_float = 1.0
        for _ in range(delay):
            integer_pos = round(Fraction(9 * integer_pos, 10))
            integer_neg = round(Fraction(9 * integer_neg, 10))
            slow_float *= 0.9
            updates += 3
        emit({
            "kind": "per_update_rounding_control", "delay": delay,
            "rho_exact": "9/10", "grid_delta": DELTA,
            "integer_grid_state_plus": integer_pos,
            "integer_grid_state_minus": integer_neg,
            "rounded_state_plus": DELTA * integer_pos,
            "rounded_state_minus": DELTA * integer_neg,
            "unrounded_slow_state_plus": slow_float,
            "only_query_rounded_slow_state": DELTA * round(slow_float / DELTA),
            "per_update_grid_alias": integer_pos == integer_neg,
            "rounded_map_next_integer": round(Fraction(9 * integer_pos, 10)),
            "interpretation": "rounding is part of this different runtime map; no smooth-J claim",
        })

assert updates <= 10000, updates
assert len(records) == 21
by_key = {(x["kind"], x["delay"]): x for x in records}
at16 = {kind: by_key[kind, 16] for kind in ("bistable", "matched_fading")}
# These check the implementations against the derived diagnostic, not novelty.
for delay in DELAYS:
    s = by_key["bistable", delay]
    f = by_key["matched_fading", delay]
    assert abs(s["separation"] - 2 * a) < 1e-12
    assert abs(s["log_abs_jacobian_plus"] - f["log_abs_jacobian_plus"]) < 1e-9
assert at16["bistable"]["tiny_tangent_trigger"]
assert at16["matched_fading"]["tiny_tangent_trigger"]
summary = {
    "status": "exploratory diagnostic computation, not learning/advantage confirmation",
    "kappa": KAPPA, "a": a, "r": r,
    "fixed_point_iterations": fixed_iterations,
    "fixed_point_residual": abs(a - math.tanh(KAPPA * a)),
    "query_cut_noise_sigma": SIGMA,
    "matched_tangent_at_delay16": at16,
    "rounding_at_delay1024": by_key["per_update_rounding_control", 1024],
    "state_update_count": updates,
    "raw_rows": len(records),
    "training_steps": 0,
    "old_experiment_inputs_read": 0,
    "seconds_internal": time.monotonic() - started,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(json.dumps({"raw_rows": len(records), "state_updates": updates,
                  "seconds": summary["seconds_internal"], "training_steps": 0}))
