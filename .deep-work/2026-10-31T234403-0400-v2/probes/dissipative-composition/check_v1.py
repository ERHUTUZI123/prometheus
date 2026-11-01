"""Fixed necessary-condition test; no training or external input."""
import json
from pathlib import Path
import time

import numpy as np

START = time.monotonic()
BETA = 0.1
STEP = 0.1
MAX_STEPS = 5000
GRAD_TOL = 1e-12
OUT = Path(__file__).resolve().parent
CALLS = {"energy": 0, "gradient": 0, "hessian": 0}


def energy(x):
    CALLS["energy"] += 1
    a, b, c = x
    return float(((a*b-1)**2 + (b*c-1)**2 + (c*a-1)**2
                  + BETA*(a*b*c-1)**2) / 2)


def gradient(x):
    CALLS["gradient"] += 1
    a, b, c = x
    triple = BETA * (a*b*c-1)
    return np.array([(a*b-1)*b + (c*a-1)*c + triple*b*c,
                     (a*b-1)*a + (b*c-1)*c + triple*a*c,
                     (b*c-1)*b + (c*a-1)*a + triple*a*b])


def hessian(x):
    CALLS["hessian"] += 1
    a, b, c = x
    return np.array([
        [b*b+c*c+BETA*b*b*c*c,
         2*a*b-1+BETA*(2*a*b*c*c-c),
         2*a*c-1+BETA*(2*a*c*b*b-b)],
        [2*a*b-1+BETA*(2*a*b*c*c-c),
         a*a+c*c+BETA*a*a*c*c,
         2*b*c-1+BETA*(2*b*c*a*a-a)],
        [2*a*c-1+BETA*(2*a*c*b*b-b),
         2*b*c-1+BETA*(2*b*c*a*a-a),
         a*a+b*b+BETA*a*a*b*b]])


def clauses(signs):
    a, b, c = signs
    return [int(a*b == 1), int(b*c == 1),
            int(c*a == 1), int(a*b*c == 1)]


def signs_of(x):
    return [1 if v > 1e-10 else -1 if v < -1e-10 else 0 for v in x]


def finite_difference_checks():
    x = np.array([0.37, -0.51, 0.22])
    eps = 1e-6
    basis = np.eye(3)
    dg = np.array([(energy(x+eps*e)-energy(x-eps*e))/(2*eps)
                   for e in basis])
    dh = np.column_stack([(gradient(x+eps*e)-gradient(x-eps*e))/(2*eps)
                          for e in basis])
    result = {"point": x.tolist(), "epsilon": eps,
              "gradient_max_abs": float(np.max(np.abs(dg-gradient(x)))),
              "hessian_max_abs": float(np.max(np.abs(dh-hessian(x))))}
    if result["gradient_max_abs"] > 1e-8 or result["hessian_max_abs"] > 1e-8:
        raise AssertionError(result)
    return result


def negative_stationary():
    lo, hi = -1.0, -0.9
    def q(t):
        return 2*(t*t-1)+BETA*t*(t*t*t-1)
    if not q(lo) > 0 > q(hi):
        raise AssertionError("invalid predeclared bracket")
    for _ in range(60):
        mid = (lo+hi)/2
        if q(mid) > 0:
            lo = mid
        else:
            hi = mid
    x = np.full(3, (lo+hi)/2)
    eig = np.linalg.eigvalsh(hessian(x))
    return {"root": x.tolist(), "bracket": [lo, hi],
            "gradient_inf": float(np.max(np.abs(gradient(x)))),
            "energy": energy(x), "hessian_eigenvalues": eig.tolist(),
            "discrete_linearization_eigenvalues": (1-STEP*eig).tolist(),
            "signs": signs_of(x), "clauses_satisfied": clauses(signs_of(x)),
            "status": "numeric diagnostic; analytic interval proof is separate",
            "diagnostic_bisection_steps": 60}


def gf2_solver():
    # sign=(-1)^bit; the four input sign products are all +1.
    mat = [[1, 1, 0, 0], [0, 1, 1, 0],
           [1, 0, 1, 0], [1, 1, 1, 0]]
    original = [r[:] for r in mat]
    pivots, xors, scans = [], 0, 0
    for col in range(3):
        pivot = None
        for row in range(len(pivots), len(mat)):
            scans += 1
            if mat[row][col]:
                pivot = row
                break
        if pivot is None:
            continue
        row = len(pivots)
        mat[row], mat[pivot] = mat[pivot], mat[row]
        for target in range(len(mat)):
            scans += 1
            if target != row and mat[target][col]:
                mat[target] = [a ^ b for a, b in zip(mat[target], mat[row])]
                xors += 1
        pivots.append(col)
    inconsistent = any(not any(r[:3]) and r[3] for r in mat)
    if inconsistent or pivots != [0, 1, 2]:
        raise AssertionError("predeclared system not uniquely solvable")
    bits = [mat[row][3] for row in range(3)]
    signs = [(-1)**bit for bit in bits]
    return {"input_rows": original, "reduced_rows": mat,
            "rank": len(pivots), "unique": True, "bits": bits,
            "signs": signs, "clauses_satisfied": clauses(signs),
            "row_xors_of_width4": xors, "pivot_bit_checks": scans}


def individual_completions():
    # Each factor with all other arguments clamped +1 is a scalar quadratic.
    # Other sign clamps are conjugate by changing the free coordinate's sign.
    results = []
    for name, weight in [("ab", 1.), ("bc", 1.), ("ca", 1.), ("abc", BETA)]:
        value = -.99
        for updates in range(MAX_STEPS+1):
            grad = weight*(value-1)
            if abs(grad) <= GRAD_TOL or updates == MAX_STEPS:
                break
            value = float(np.clip(value-STEP*grad, -1., 1.))
        results.append({"factor": name, "weight": weight,
                        "clamped_other_signs": "+1", "initial_free": -.99,
                        "final_free": value, "updates": updates,
                        "gradient_evaluations": updates+1,
                        "gradient_abs": abs(weight*(value-1)),
                        "correct": abs(value-1) < 1e-8})
    return results


with (OUT / "raw.jsonl").open("x") as raw:
    def emit(kind, **data):
        raw.write(json.dumps({"kind": kind, **data}, allow_nan=False)+"\n")
        raw.flush()

    emit("config", beta=BETA, step=STEP, max_steps=MAX_STEPS,
         gradient_tolerance=GRAD_TOL, internal_timeout_seconds=50,
         scope="exact factors supplied equally; no learning test")
    fd = finite_difference_checks()
    emit("derivative_check", **fd)
    stationary = negative_stationary()
    emit("negative_stationary", **stationary)
    rival = gf2_solver()
    emit("same_information_gf2", **rival)
    individuals = individual_completions()
    for item in individuals:
        emit("individual_completion", **item)
    starts = {"symmetric_negative": [-.99, -.99, -.99],
              "asymmetric_negative": [-.99, -.97, -.95],
              "positive": [.9, .9, .9], "zero": [0., 0., 0.]}
    trajectories = []
    for name, initial in starts.items():
        x = np.array(initial, dtype=float)
        previous = energy(x)
        status = "max_steps"
        updates = 0
        for k in range(MAX_STEPS+1):
            if time.monotonic()-START > 50:
                emit("failure", reason="internal timeout", trajectory=name, k=k)
                raise TimeoutError("internal 50s cap")
            grad = gradient(x)
            norm = float(np.max(np.abs(grad)))
            current = energy(x)
            emit("state", trajectory=name, update=k, x=x.tolist(),
                 energy=current, gradient_inf=norm)
            if not np.isfinite(x).all() or current > previous+1e-12:
                raise AssertionError("nonfinite or nonmonotone execution")
            if norm <= GRAD_TOL:
                status = "stationary"
                break
            if k == MAX_STEPS:
                break
            previous = current
            x = np.clip(x-STEP*grad, -1., 1.)
            updates += 1
        eig = np.linalg.eigvalsh(hessian(x))
        signs = signs_of(x)
        result = {"name": name, "initial": initial, "updates": updates,
                  "gradient_evaluations": updates+1, "status": status,
                  "final": x.tolist(), "energy": energy(x),
                  "gradient_inf": float(np.max(np.abs(gradient(x)))),
                  "hessian_eigenvalues": eig.tolist(), "signs": signs,
                  "clauses_satisfied": clauses(signs),
                  "correct": all(clauses(signs)),
                  "local_minimum_numeric": bool(min(eig)>0)}
        emit("trajectory_result", **result)
        trajectories.append(result)
    summary = {"scope": "necessary execution test, not formation or novelty",
               "derivative_check": fd, "stationary": stationary,
               "rival": rival, "individual_completions": individuals,
               "trajectories": trajectories,
               "seconds": time.monotonic()-START,
               "inference_gradient_evaluations": sum(t["gradient_evaluations"]
                                                       for t in trajectories),
               "individual_gradient_evaluations": sum(t["gradient_evaluations"]
                                                         for t in individuals),
               "total_vector_function_calls_including_diagnostics": dict(CALLS),
               "cost_boundary": "call and row-operation counts, not full FLOPs/RSS/energy"}
    emit("summary", **summary)
    with (OUT / "summary.json").open("x") as out:
        json.dump(summary, out, indent=2, allow_nan=False)
        out.write("\n")
