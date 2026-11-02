"""Continuous configuration-cover execution of MODEL-learned native operations.

This module has no Torch dependency, teacher semantics, training, or import-time
work. A permutation is compiled to an adjacent-exchange configuration loop, but
execution follows all marked roots through UNLABELED geometric snapshots. The
expected permutation is read only after lifting, to audit faithfulness.

At time t, C(t) is an unordered set of distinct complex points; equivalently
F(z,t)=prod_{c in C(t)}(z-c)=0 is a D-sheeted polynomial covering along this
path. The implementation evaluates the factored configuration directly, NOT a
polynomial-coefficient root solver or a physical quantum device.

Shared deformation: Phi_{omega,t}(x+iy)=x+i(y+a(t)h_omega(x)), where a vanishes
at operation endpoints and h=sum_l omega_l tanh(v_l*x+b_l). Phi is a global
diffeomorphism for every finite omega. A linear parameter homotopy therefore
preserves primitive monodromy while changing its continuous implementation.
This diagnostic supplies a real continuous execution layer; it does not by
itself supply a new acquisition law or establish a resource advantage.

MAIN=TOPOLOGY USER_LOCKED. M0 NOT PASSED; M3 ultimate; Transformer primary.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


SPACING = 4.0
FREQUENCIES = np.asarray([0.03125, -0.0625, 0.125, -0.25])
BIASES = np.asarray([-0.3, 0.2, 0.1, -0.2])


def canonical_word(permutation):
    """Same positive adjacent-exchange selector as native topology_model.py."""
    p = np.asarray(permutation, dtype=np.int64)
    if p.ndim != 1 or not 1 <= len(p) <= 64 or not np.array_equal(np.sort(p), np.arange(len(p))):
        raise ValueError("require a permutation of1..64 latent states")
    order, word = list(range(len(p))), []
    for i, item in enumerate(np.argsort(p).tolist()):
        j = order.index(item)
        while j > i:
            order[j - 1], order[j] = order[j], order[j - 1]
            word.append(j - 1)
            j -= 1
    if len(word) > 512:
        raise ValueError("512 adjacent-exchange process cap exceeded")
    return tuple(word)


class ConfigurationLoop:
    def __init__(self, permutation, omega):
        self.word = canonical_word(permutation)
        self.d = len(permutation)
        self.base = SPACING * (np.arange(self.d) - (self.d - 1) / 2)
        self.omega = np.asarray(omega, dtype=np.float64)
        if self.omega.shape != FREQUENCIES.shape or not np.isfinite(self.omega).all():
            raise ValueError("omega requires four finite shared deformation weights")
        if np.max(np.abs(self.omega)) > 8:
            raise ValueError("diagnostic shared-weight magnitude cap8")
        self.duration = max(1, len(self.word))
        self.shear_lipschitz = float(np.sum(np.abs(self.omega * FREQUENCIES)))
        self.shear_amplitude = float(np.sum(np.abs(self.omega)))
        # For a shear, both Phi and Phi^{-1} are <=(1+B)-Lipschitz.
        self.clearance_bound = SPACING / (1 + self.shear_lipschitz)
        base_speed = 3 * math.pi if self.word else 0.0  # radius2 * pi * max h'=1.5
        self.speed_bound = ((1 + self.shear_lipschitz) * base_speed
                            + math.pi * self.shear_amplitude / self.duration)
        # Strict displacement <delta/4 and exact inclusion of EVERY segment
        # boundary. At the next sample the true root is uniquely closest.
        self.steps_per_segment = max(1, int(math.floor(
            4 * self.speed_bound / self.clearance_bound)) + 1)
        self.total_steps = self.duration * self.steps_per_segment
        if self.total_steps > 200000:
            raise ValueError("200000 snapshot per-operation process cap exceeded")

    def configuration(self, t):
        """Return only an UNLABELED point set; ordering is discarded by caller."""
        t = float(t)
        if not 0 <= t <= self.duration:
            raise ValueError("path time out of domain")
        points = self.base.astype(np.complex128)
        if self.word and t < self.duration:
            segment = min(int(math.floor(t)), len(self.word) - 1)
            u = t - segment
            h = u * u * (3 - 2 * u)
            j = self.word[segment]
            midpoint = (self.base[j] + self.base[j + 1]) / 2
            radius = 2 * np.exp(1j * math.pi * h)
            points[j], points[j + 1] = midpoint - radius, midpoint + radius
        # The exact endpoint is returned as the fixed unordered basepoint;
        # smoothstep gives a continuously agreeing set at every boundary.
        envelope = 0.0 if t in (0.0, float(self.duration)) else math.sin(math.pi * t / self.duration) ** 2
        shear = np.tanh(points.real[:, None] * FREQUENCIES + BIASES) @ self.omega
        return points + 1j * envelope * shear

    def certificate(self):
        return dict(latent_states=self.d, adjacent_exchanges=len(self.word),
                    duration=self.duration, shared_omega=self.omega.tolist(),
                    shear_lipschitz_upper=self.shear_lipschitz,
                    uniform_pair_clearance_lower=self.clearance_bound,
                    uniform_point_speed_upper=self.speed_bound,
                    steps_per_segment=self.steps_per_segment,
                    snapshots=self.total_steps,
                    per_step_displacement_upper=self.speed_bound / self.steps_per_segment,
                    nearest_lift_admission_strict=(self.speed_bound / self.steps_per_segment
                                                  < self.clearance_bound / 4),
                    analytic_scope="real-arithmetic sufficient bound; numerical diagnostics also recorded")


def lift_sequence(loops, deadline, trace_path=None):
    """Lift through geometry, without reading stored permutations or word labels.

    Adjacent words are consumed solely inside configuration(t). This executor
    receives only unordered coordinates plus analytic clearance/speed bounds.
    Marked roots are initialized at the fixed basepoint and are continued over
    concatenated operation loops without resetting their sheet identities.
    """
    if not loops or len(loops) > 8 or any(loop.d != loops[0].d for loop in loops):
        raise ValueError("require1..8 loops in a common marked covering")
    d = loops[0].d
    tracked = loops[0].base.astype(np.complex128)
    rng = np.random.default_rng(1729)
    snapshots = 0
    minimum_sampled_clearance = math.inf
    maximum_observed_step = 0.0
    nearest_margin_minimum = math.inf
    squared_distance_evaluations = 0
    trace = trace_path.open("x", encoding="utf-8") if trace_path else None
    try:
        for operation, loop in enumerate(loops):
            if np.max(np.abs(np.sort(tracked.real) - loop.base)) > 1e-8 or np.max(np.abs(tracked.imag)) > 1e-8:
                raise ValueError("operation interface is not the common fixed fiber")
            for step in range(1, loop.total_steps + 1):
                if time.monotonic() >= deadline:
                    raise TimeoutError("internal continuous-lifting wall limit")
                t = step / loop.steps_per_segment
                points = loop.configuration(t)
                # Enumeration is intentionally unrelated to marked sheets and
                # adjacent exchange instructions. No labels are passed along.
                points = points[rng.permutation(d)]
                differences = np.abs(tracked[:, None] - points[None, :])
                closest = differences.argmin(axis=1)
                selected = differences[np.arange(d), closest]
                if len(np.unique(closest)) != d:
                    raise RuntimeError("nearest continuation was not bijective")
                if d > 1:
                    nearest_two = np.partition(differences, 1, axis=1)[:, :2]
                    margin = float(np.min(np.max(nearest_two, axis=1) - np.min(nearest_two, axis=1)))
                    nearest_margin_minimum = min(nearest_margin_minimum, margin)
                    pair = np.abs(points[:, None] - points[None, :])
                    np.fill_diagonal(pair, math.inf)
                    clearance = float(np.min(pair))
                    minimum_sampled_clearance = min(minimum_sampled_clearance, clearance)
                    if clearance < loop.clearance_bound - 1e-8:
                        raise RuntimeError("observed clearance violates analytic sufficient bound")
                if np.max(selected) > loop.speed_bound / loop.steps_per_segment + 1e-8:
                    raise RuntimeError("observed step violates analytic speed bound")
                maximum_observed_step = max(maximum_observed_step, float(np.max(selected)))
                tracked = points[closest]
                snapshots += 1
                squared_distance_evaluations += d * d * (2 if d > 1 else 1)
                if trace:
                    trace.write(json.dumps(dict(operation=operation, step=step, t=t,
                        marked_root_real=tracked.real.tolist(), marked_root_imag=tracked.imag.tolist(),
                        max_marked_step=float(np.max(selected))), separators=(",", ":")) + "\n")
        distances = np.abs(tracked[:, None] - loops[0].base[None, :])
        squared_distance_evaluations += d * d
        result = distances.argmin(axis=1)
        if len(np.unique(result)) != d or np.max(distances[np.arange(d), result]) > 1e-8:
            raise RuntimeError("lift endpoint did not return to the fixed fiber")
        return dict(monodromy=result.tolist(), snapshots=snapshots,
                    min_sampled_pair_clearance=None if d == 1 else minimum_sampled_clearance,
                    max_observed_marked_step=maximum_observed_step,
                    minimum_nearest_continuation_margin=None if d == 1 else nearest_margin_minimum,
                    distance_matrix_entries_evaluated=squared_distance_evaluations,
                    trace_path=str(trace_path) if trace_path else None)
    finally:
        if trace:
            trace.close()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def learned_operations(records, model, max_operations):
    last = None
    with records.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("kind") != "curve" or row.get("model") != model:
                continue
            if row.get("learned_structure", {}).get("available"):
                last = row, line_number
    if last is None:
        raise ValueError("no actual saved learned graph for requested model")
    row, line_number = last
    unique = {}
    # Prioritize the most recently learned task; earlier contexts still add
    # reusable classes when they fit the six-operation diagnostic cap.
    for graph in sorted(row["learned_structure"]["graphs"], key=lambda g: -int(g["task"])):
        for token, values in graph["learned_transitions"].items():
            key = tuple(int(q) for q in values)
            unique.setdefault(key, []).append(dict(task=int(graph["task"]), token=int(token)))
    if len(unique) > 24:
        raise ValueError("saved learned graph exceeds24 unique operation cap")
    # Keep saved order; no task correctness or teacher arithmetic is inspected.
    operations = [dict(id=i, permutation=list(p), saved_call_sites=sites)
                  for i, (p, sites) in enumerate(unique.items())][:max_operations]
    return operations, dict(source_line=line_number, stage=int(row["stage"]), step=int(row["step"]),
                            recorded_unique_operations=len(unique))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="topo")
    parser.add_argument("--max-operations", type=int, default=6)
    parser.add_argument("--seconds", type=float, default=110)
    args = parser.parse_args()
    if not 1 <= args.max_operations <= 6 or not 0 < args.seconds <= 120:
        parser.error("require1..6 operations and internal wall cap<=120seconds")
    if args.out.exists():
        parser.error("--out must be a NEW directory")
    records = args.records.resolve(strict=True)
    operations, origin = learned_operations(records, args.model, args.max_operations)
    if not operations:
        parser.error("saved learned graph contains no operations")
    args.out.mkdir(parents=True)
    started, deadline = time.monotonic(), time.monotonic() + args.seconds
    deformations = {"original": [0., 0., 0., 0.], "shared_changed": [0.7, -0.5, 0.4, -0.3]}
    reports, composition = [], []
    summary = dict(status="STARTED", MAIN="TOPOLOGY", M0="NOT_PASSED", M3="ultimate",
                   Transformer="primary architecture baseline; this is an execution diagnostic",
                   source_records=str(records), source_records_sha256=sha(records),
                   code_sha256=sha(Path(__file__)), source_graph=origin,
                   operations=operations, shared_deformations=deformations,
                   shared_parameter_homotopy=dict(path="omega(lambda)=(1-lambda)omega_old+lambda*omega_new",
                       parameter_domain="0<=lambda<=1", collision_avoidance="global vertical shear diffeomorphism for everylambda",
                       uniform_pair_clearance_lower=SPACING / (1 + max(float(np.sum(np.abs(np.asarray(w) * FREQUENCIES)))
                                                                      for w in deformations.values()))),
                   scope="continuous configuration-cover lift of learned native operators; no training or physical experiment",
                   training_performed=False, coefficient_root_solver_used=False,
                   reports=reports, compositions=composition)
    with (args.out / "raw.jsonl").open("x", encoding="utf-8") as raw:
        def emit(row):
            raw.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            raw.flush()
        emit(dict(kind="configuration", **{k: v for k, v in summary.items() if k not in ("reports", "compositions")}))
        try:
            for operation in operations:
                for name, omega in deformations.items():
                    loop = ConfigurationLoop(operation["permutation"], omega)
                    result = lift_sequence([loop], deadline,
                        args.out / f"trace-op{operation['id']}-{name}.jsonl")
                    row = dict(kind="primitive_lift", operation=operation["id"], deformation=name,
                               certificate=loop.certificate(), **result)
                    row["matches_saved_native_operation"] = result["monodromy"] == operation["permutation"]
                    reports.append(row)
                    emit(row)
            pair = None
            for first in operations:
                for second in operations:
                    p, q = np.asarray(first["permutation"]), np.asarray(second["permutation"])
                    if not np.array_equal(q[p], p[q]):
                        pair = first, second
                        break
                if pair:
                    break
            if pair:
                for name, omega in deformations.items():
                    for first, second in (pair, pair[::-1]):
                        result = lift_sequence([ConfigurationLoop(first["permutation"], omega),
                                                ConfigurationLoop(second["permutation"], omega)], deadline,
                            args.out / f"trace-compose{first['id']}-{second['id']}-{name}.jsonl")
                        expected = np.asarray(second["permutation"])[np.asarray(first["permutation"])].tolist()
                        row = dict(kind="noncommutative_composition_lift", operations=[first["id"], second["id"]],
                                   deformation=name, matches_saved_native_composition=result["monodromy"] == expected,
                                   **result)
                        composition.append(row)
                        emit(row)
            summary.update(status="CONTINUOUS_LIFT_DIAGNOSTIC_COMPLETED",
                           all_primitive_faithful=all(r["matches_saved_native_operation"] for r in reports),
                           noncommuting_pair_found=pair is not None,
                           all_compositions_faithful=all(r["matches_saved_native_composition"] for r in composition),
                           wall_seconds=time.monotonic() - started)
        except Exception as error:
            summary.update(status="FAILED_WITH_RAW_RECORDS", error_type=type(error).__name__,
                           error=str(error), wall_seconds=time.monotonic() - started)
            emit(dict(kind="failure", error_type=type(error).__name__, error=str(error)))
            raise
        finally:
            with (args.out / "summary.json").open("x", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
