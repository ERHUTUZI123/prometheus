"""Read-only reconstruction of a conventional full-quadratic path controller.

Input is this campaign's SAVED action moments and learned omega. No model,
teacher, task labels, path simulator or training is invoked. The quadratic is
the recorded midpoint-quadrature objective, not an exact physical integral.
The zero and analytic controllers are reconstructed alternatives, not newly
executed trajectories or new continual-learning experiments.

MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 NOT PASSED; M3 ultimate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


MAX_BYTES = 100 * 1024 * 1024


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dot(a, b):
    return math.fsum(x*y for x, y in zip(a, b))


def multiply(matrix, vector):
    return [dot(row, vector) for row in matrix]


def action(c, b, h, omega):
    return math.fsum((c, 2*dot(b, omega), dot(omega, multiply(h, omega))))


def solve_positive_definite(h, rhs):
    """Ordinary full 4x4 Cholesky solve; no diagonal approximation."""
    n = len(rhs)
    lower = [[0.] * n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            remaining = h[i][j]-math.fsum(lower[i][k]*lower[j][k] for k in range(j))
            if i == j:
                if remaining <= 0:
                    raise ValueError("saved full quadratic is not numerically positive definite")
                lower[i][j] = math.sqrt(remaining)
            else:
                lower[i][j] = remaining/lower[j][j]
    y = [0.] * n
    for i in range(n):
        y[i] = (rhs[i]-math.fsum(lower[i][j]*y[j] for j in range(i)))/lower[i][i]
    result = [0.] * n
    for i in reversed(range(n)):
        result[i] = (y[i]-math.fsum(lower[j][i]*result[j] for j in range(i+1, n)))/lower[i][i]
    return result, [lower[i][i] for i in range(n)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.)
    args = parser.parse_args()
    if not 0 < args.seconds <= 30 or len(args.records) > 8 or args.out.exists():
        parser.error("new output, <=8 saved files, and internal cap<=30s required")
    started, rows, cache, sources = time.perf_counter(), [], {}, []
    for path in args.records:
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("saved input exceeds100MiB")
        sources.append(dict(path=str(path), sha256=digest(path), bytes=path.stat().st_size))
        initial = None
        with path.open(encoding="utf8") as stream:
            for line_number, line in enumerate(stream, 1):
                if time.perf_counter()-started >= args.seconds:
                    raise TimeoutError("bounded reconstruction time limit")
                record = json.loads(line)
                if record.get("kind") == "configuration":
                    initial = record.get("path_omega_initial", initial)
                if record.get("kind") != "curve" or "action_moments" not in record:
                    continue
                moments = record["action_moments"]
                c, b = float(moments["constant"]), [float(x) for x in moments["linear"]]
                h = [[float(x) for x in row] for row in moments["quadratic"]]
                learned = [float(x) for x in record["path_omega"]]
                if initial is None:
                    initial = learned[:]
                if (len(b) != 4 or len(h) != 4 or any(len(r) != 4 for r in h)
                        or len(learned) != 4 or len(initial) != 4
                        or not all(math.isfinite(v) for v in [c, *b, *learned, *initial, *sum(h, [])])):
                    raise ValueError("finite recorded four-dimensional objective required")
                if max(abs(h[i][j]-h[j][i]) for i in range(4) for j in range(4)) > 1e-9:
                    raise ValueError("saved quadratic is not symmetric")
                signature = hashlib.sha256(json.dumps(dict(c=c, b=b, h=h), sort_keys=True).encode()).hexdigest()
                solve_reused = signature in cache
                if not solve_reused:
                    began = time.perf_counter()
                    optimum, diagonal = solve_positive_definite(h, [-x for x in b])
                    cache[signature] = dict(omega=optimum, cholesky_diagonal=diagonal,
                                           reconstruction_solve_seconds=time.perf_counter()-began)
                solved = cache[signature]
                optimum = solved["omega"]
                if max(abs(x) for x in optimum) > 8:
                    raise ValueError("unconstrained optimum exceeds existing |omega|<=8 path cap; box solve needed")
                zero = [0.] * 4
                a_initial, a_learned, a_zero, a_opt = [action(c, b, h, w)
                    for w in (initial, learned, zero, optimum)]
                residual = [x+y for x, y in zip(multiply(h, optimum), b)]
                gap_vector = [x-y for x, y in zip(learned, optimum)]
                source_costs = record.get("costs", {})
                rows.append(dict(source=str(path), source_line=line_number, step=record.get("step"),
                    moments_sha256=signature, objective="A(omega)=c+2*b^T*omega+omega^T*H*omega",
                    objective_scope="saved midpoint quadrature only", normalizer=moments["normalizer"],
                    omega_initial=initial, omega_learned=learned, omega_zero=zero, omega_full_quadratic_optimum=optimum,
                    action_initial_double_reconstruction=a_initial,
                    action_learned_double_reconstruction=a_learned,
                    action_zero=a_zero, action_optimum=a_opt,
                    initial_to_learned_reduction=a_initial-a_learned,
                    initial_to_zero_reduction=a_initial-a_zero,
                    zero_to_optimum_reduction=a_zero-a_opt,
                    learned_to_optimum_reduction=a_learned-a_opt,
                    learned_to_optimum_positive_quadratic_gap=dot(gap_vector, multiply(h, gap_vector)),
                    analytic_stationarity_residual_Linf=max(abs(v) for v in residual),
                    within_existing_omega_cap=True, cholesky_diagonal=solved["cholesky_diagonal"],
                    saved_reported_action=record.get("kinetic_action"),
                    saved_float_action_minus_double_reconstruction=record.get("kinetic_action", a_learned)-a_learned,
                    full_coupled_solve_cached_from_identical_moments=solve_reused,
                    active_classes=moments.get("operation_classes"),
                    common_paid_moment_information=dict(
                        quadrature_point_strands=moments.get("quadrature_point_strands"),
                        quadrature_per_segment=moments.get("quadrature_per_segment"),
                        original_cumulative_moment_rebuilds=source_costs.get("geometry_moment_rebuilds"),
                        original_cumulative_moment_seconds=source_costs.get("geometry_moment_seconds"),
                        original_cumulative_class_check_seconds=source_costs.get("geometry_class_check_seconds")),
                    direct_controller_additional_external_labels=0,
                    direct_controller_numeric_state_bytes_float64=(1+4+16+4)*8,
                    direct_controller_new_path_execution=False))
    if not rows:
        raise ValueError("no actual saved action curves found")
    result = dict(schema="saved-full-quadratic-control-reconstruction-v1", STATUS="RECONSTRUCTION",
        MAIN="TOPOLOGY_USER_LOCKED", M0="NOT_ASSESSED", primary_baseline="Transformer",
        source_files=sources, rows=rows,
        costs=dict(new_training_steps=0, new_simulated_path_executions=0,
                   new_external_labels=0, unique_four_by_four_full_solves=len(cache),
                   reconstruction_solve_seconds=sum(v["reconstruction_solve_seconds"] for v in cache.values()),
                   analysis_wall_seconds=time.perf_counter()-started),
        boundaries=[
            "Uses the same already-paid full c,b,H as the learned geometric action objective; no free calibration or task information.",
            "omega=0 is a specified direct control, not a fitted teacher answer; the full coupled optimizer is -H^{-1}b.",
            "Stationarity and Cholesky reconstruct the numeric quadratic optimum inside the existing omega cap.",
            "Original moment construction, shared model acquisition, semantic checking and path execution costs remain payable by applicable alternatives.",
            "Local analysis solve times are not a controlled runtime comparison with GPU training.",
            "No counterfactual optimizer path was executed and no hardware action/energy advantage follows.",
            "All reconstruction values concern saved quadrature; reported float32 training values can differ through rounding."])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n"
    if len(text.encode()) > MAX_BYTES:
        raise RuntimeError("bounded analysis output cap")
    with args.out.open("x", encoding="utf8") as stream:
        stream.write(text)
    print(json.dumps(dict(output=str(args.out), rows=len(rows), **result["costs"]), sort_keys=True))


if __name__ == "__main__":
    main()
