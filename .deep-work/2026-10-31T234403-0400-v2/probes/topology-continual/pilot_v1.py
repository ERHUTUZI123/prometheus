"""Inspected-before-execution finite diagnostic; importing performs no work.

MAIN=TOPOLOGY / USER_LOCKED; M0 active NOT PASSED; M3 ultimate.
Online USER-PARK; standalone Persistent State USER-PAUSED.
No network, installations, subprocesses, arbitrary execution, or model routes.
The existing run_isolated.py must provide external isolation and 120 s limits.
Only the FIRST frozen stream is implemented for this invocation. No results are
claimed by this file. Floating midpoint checks are not rigorous enclosures.
"""
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np


OUT = Path(__file__).resolve().parent
SEED = 20261101
MAX_SECONDS = 100.0
MAX_PROPAGATIONS = 3000             # one gate / hypothesis / resolution
MAX_SEGMENT_BATCHES = 15000         # one physical segment, <=16 hypotheses
MAX_ROWS = 5000
MAX_BYTES = 25_000_000
MAX_FINAL_PAIRS = 36
SAMPLES, ORACLE, FINAL = 192, 384, 768
DELTA_GLOBAL, MAX_CONFIDENCE_BATCHES = .03, 72
DELTA_BATCH = DELTA_GLOBAL / MAX_CONFIDENCE_BATCHES
NUM_ALLOWANCE = .002                # heuristic allowance, NOT an enclosure
ADMISSION = .08
RHOS = (.45, .475, .5, .525, .55)
LEGS = (2., 4., 8.)
PRIOR = tuple(itertools.product((-.1, .1), (-.1, .1), (.9, 1.1), (0., .2)))
NEUTRAL = (0., 0., 1., 0.)
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], complex)
SY = np.array([[0, -1j], [1j, 0]], complex)
SZ = np.diag([1., -1.]).astype(complex)


def safe(x):
    if isinstance(x, np.ndarray):
        return safe(x.tolist())
    if isinstance(x, (complex, np.complexfloating)):
        return [float(x.real), float(x.imag)]
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


def encoded(x):
    return json.dumps(safe(x), sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode()


class Stop(RuntimeError):
    pass


class Run:
    def __init__(self):
        self.start = time.monotonic()
        self.raw = (OUT / "pilot-v1.raw.jsonl").open("xb")
        self.bits = (OUT / "pilot-v1.bits").open("xb")
        self.rows = self.bytes = 0
        self.cost = dict(modeled_gate_hypotheses=0, segment_batches=0,
                         modeled_segment_hypotheses=0,
                         matrix_exponentials_4d=0, midpoint_matrix_products=0,
                         hermitian_eigendecompositions_4d=0,
                         inverse_component_evaluations=0, controller_samples=0,
                         cache_hits=0, target_bits=0, calibration_bits=0,
                         preparations=0, readouts=0, observation_evolution=0.,
                         observation_gap_action=0., observation_applied_action=0.,
                         calibration_routing_configurations=0,
                         candidate_proposals=0, rejected_proposals=0,
                         final_gate_plant_pairs=0, word_input_episodes=0,
                         word_simulated_evolution=0.)
        self.progress = dict(stream=SEED, stage=0, phase="created")
        self.final_checks = []
        self.emit("configuration", seed=SEED, only_first_stream=True,
                  remaining_frozen_seeds_unexecuted=[20261102, 20261103],
                  prior=PRIOR, delta_global=DELTA_GLOBAL,
                  delta_batch=DELTA_BATCH, numerical_allowance=NUM_ALLOWANCE,
                  admission_score="worst modeled error + angle halfwidth/2 + .002 <= .08",
                  counts={"full_gate": "each hypothesis and resolution",
                          "segment_batch": "one physical segment, <=16 hypotheses; time samples streamed",
                          "physical_queries": "only sampled observation batches; proposals are simulations"},
                  main="TOPOLOGY_USER_LOCKED", M0="active_NOT_PASSED", M3="ultimate",
                  online="USER_PARK", persistent_state="standalone_USER_PAUSED")

    def check(self, gates=0, segments=0):
        if time.monotonic() - self.start >= MAX_SECONDS:
            raise Stop("inner 100 s wall limit")
        if self.cost["modeled_gate_hypotheses"] + gates > MAX_PROPAGATIONS:
            raise Stop("3000 gate/hypothesis propagation cap")
        if self.cost["segment_batches"] + segments > MAX_SEGMENT_BATCHES:
            raise Stop("15000 batched physical-segment cap")
        if self.rows >= MAX_ROWS - 3 or self.bytes >= MAX_BYTES - 100000:
            raise Stop("raw event/output cap; finish reserve retained")

    def emit(self, kind, **fields):
        row = encoded(dict(kind=kind, elapsed=time.monotonic() - self.start,
                           progress=self.progress, **fields)) + b"\n"
        if self.rows >= MAX_ROWS or self.bytes + len(row) > MAX_BYTES:
            raise Stop("output hard cap")
        if json.loads(row)["kind"] != kind:
            raise Stop("JSON serialization failure")
        self.raw.write(row)
        self.raw.flush()
        self.rows += 1
        self.bytes += len(row)

    def sample(self, rng, p, n, category, **metadata):
        self.check()
        bits = (rng.random(n) < p).astype(np.uint8)
        packed = np.packbits(bits, bitorder="little").tobytes()
        unpacked = np.unpackbits(np.frombuffer(packed, dtype=np.uint8),
                                 bitorder="little")[:n]
        if not np.array_equal(bits, unpacked):
            raise Stop("packed Born-bit serialization round-trip failed")
        if self.bytes + len(packed) + 10000 >= MAX_BYTES:
            raise Stop("packed-bit output cap")
        offset = self.bits.tell()
        self.bits.write(packed)
        self.bits.flush()
        self.bytes += len(packed)
        successes = int(bits.sum())
        self.cost[category + "_bits"] += n
        self.cost["preparations"] += n
        self.cost["readouts"] += n
        self.emit("sampled_simulated_Born_batch", category=category, n=n,
                  successes=successes, probability_evaluator_only=p,
                  packed_file="pilot-v1.bits", offset_bytes=offset,
                  packed_bytes=len(packed), bitorder="little",
                  sha256=hashlib.sha256(packed).hexdigest(), **metadata)
        return n, successes

    def finish(self, status, **fields):
        result = dict(status=status, elapsed=time.monotonic() - self.start,
                      progress=self.progress, cost=self.cost, rows=self.rows,
                      bytes=self.bytes, final_checks=self.final_checks,
                      M0="active_NOT_PASSED", main="TOPOLOGY_USER_LOCKED", **fields)
        self.emit("finished", result=result)
        payload = encoded(result) + b"\n"
        if self.bytes + len(payload) <= MAX_BYTES:
            with (OUT / "pilot-v1.summary.json").open("xb") as f:
                f.write(payload)
        self.raw.close()
        self.bits.close()


def kron3(a, b, c):
    return np.kron(np.kron(a, b), c)


def algebra():
    gammas = []
    for k in range(3):
        for p in (SX, SY):
            gammas.append(kron3(*([SZ] * k + [p] + [I2] * (2 - k))))
    parity = kron3(SZ, SZ, SZ)
    select = np.eye(8, dtype=complex)[:, np.where(np.diag(parity).real > .5)[0]]
    # Restrict EVEN bilinears, never single parity-changing Majoranas.
    bil = {(i, j): select.conj().T @ (1j * gammas[i] @ gammas[j]) @ select
           for i in range(6) for j in range(i + 1, 6)}
    idle = bil[0, 3]
    val, vec = np.linalg.eigh(idle)
    ground = vec[:, val < 0]
    zv, zvec = np.linalg.eigh(ground.conj().T @ bil[1, 2] @ ground)
    code = ground @ zvec[:, np.argsort(zv)[::-1]]
    cross = (code.conj().T @ bil[1, 4] @ code)[0, 1]
    code[:, 1] *= np.conj(cross) / abs(cross)
    return gammas, parity, select, bil, idle, code


def rz(beta):
    return np.diag(np.exp(-.5j * beta * np.array([1., -1.])))


def rx(beta):
    return np.cos(beta / 2) * I2 - 1j * np.sin(beta / 2) * SX


def aligned_error(actual, desired):
    """One declared trace-aligned global phase; conservative, not exact infimum."""
    overlap = np.trace(desired.conj().T @ actual)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-14 else 1.
    return float(np.linalg.norm(actual - phase * desired, 2))


def chart(a, rho):
    if not (.1 <= a <= .4 and .45 <= rho <= .55):
        raise Stop("chart domain violation")
    d = np.sqrt(a * a / 4 + (1 - a * a) / (12 * rho * (1 - rho)))
    m1, m2 = .5 - (1 - rho) * d, .5 + rho * d
    q = np.array([m1 - a * rho / 2, m1 + a * rho / 2,
                  m2 - a * (1 - rho) / 2, m2 + a * (1 - rho) / 2])
    if not (0 < q[0] < q[1] < q[2] < q[3] < 1):
        raise Stop("chart ordering failure")
    return q


def polynomial(x):
    return x * (1 - x), x * (1 - x) * (2 * x - 1)


def warp(x, c1, c2):
    p1, p2 = polynomial(x)
    return x + c1 * p1 + c2 * p2


def warp_derivative(x, c1, c2):
    return 1 + c1 * (1 - 2 * x) + c2 * (-6 * x * x + 6 * x - 1)


def ramp(s):
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def ramp_derivative(s):
    return 30 * s * s * (1 - s) ** 2


def inverse_response(d, nominal):
    k0, k1 = nominal[2:]
    if k1 == 0:
        return d / k0
    scale = np.sqrt(k0 / (3 * k1))
    return 2 * scale * np.sinh(np.arcsinh(3 * d / (2 * k0 * scale)) / 3)


def response(q, plant):
    return plant[2] * q + plant[3] * q ** 3


def loop_segments(left, right, pair, reverse=False):
    """Positive loop: down xR, equator xR->xL, up xL (root-derived sign)."""
    legs = [(0., np.pi / 2, right, right),
            (np.pi / 2, np.pi / 2, right, left),
            (np.pi / 2, 0., left, left)]
    if reverse:
        legs = [(b, a, d, c) for a, b, c, d in legs[::-1]]
    return [("geometry", pair, *leg) for leg in legs]


def program(op, theta, family):
    rho, leg, nominal = theta
    if family == "direct":
        return [("direct", op["axis"], op["beta"] / 2)], leg
    if op["kind"] == "old":
        return loop_segments(0., 1., (1, 4) if op["axis"] == "X" else (1, 2)), leg
    a = op["beta"] / np.pi
    if family == "no_moment":
        intervals = [((1 - a) / 2, (1 + a) / 2)]
    else:
        q = chart(a, rho)
        intervals = [(q[0], q[1]), (q[2], q[3])]
    forward = sum((loop_segments(l, r, (1, 2)) for l, r in intervals), [])
    backward = sum((loop_segments(l, r, (1, 2), True)
                    for l, r in intervals[::-1]), [])
    # Each X is TWO actual routed positive exchanges; no Pauli is injected.
    physical_x = 2 * loop_segments(0., 1., (1, 4))
    return forward + physical_x + backward + physical_x, leg


@dataclass
class Result:
    U: np.ndarray
    metrics: dict


class Model:
    def __init__(self, run, bil, idle, code):
        self.run, self.bil, self.idle, self.code = run, bil, idle, code
        self.cache = {}

    def key(self, op, theta, family, plant, samples):
        # rho never changes a full boundary exchange or a direct pulse.
        effective_rho = .5 if op["kind"] == "old" or family in ("direct", "no_moment") else theta[0]
        return (op["kind"], op["axis"], float(op["beta"]),
                (effective_rho, theta[1]), tuple(theta[2]), family, tuple(plant), samples)

    def gate(self, op, theta, family, plants, samples=SAMPLES):
        if len(plants) > 16 or samples > FINAL:
            raise Stop("fixed array dimension cap")
        keys = [self.key(op, theta, family, p, samples) for p in plants]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        self.run.cost["cache_hits"] += len(keys) - len(missing)
        if missing:
            segments, leg = program(op, theta, family)
            self.run.check(gates=len(missing), segments=len(segments))
            self.run.cost["modeled_gate_hypotheses"] += len(missing)
            results = self.propagate(segments, leg, theta[2],
                                     [plants[i] for i in missing], samples)
            for i, result in zip(missing, results):
                self.cache[keys[i]] = result
        return [self.cache[k] for k in keys]

    def geometry_query(self, width, samples):
        op = dict(kind="query", axis="Z", beta=width)
        theta = (.5, 4., NEUTRAL)
        keys = [("wedge", width, p, samples) for p in PRIOR]
        missing = [i for i, k in enumerate(keys) if k not in self.cache]
        if missing:
            self.run.check(gates=len(missing), segments=3)
            self.run.cost["modeled_gate_hypotheses"] += len(missing)
            results = self.propagate(loop_segments(0., width, (1, 2)),
                                     4., NEUTRAL, [PRIOR[i] for i in missing], samples)
            for i, result in zip(missing, results):
                self.cache[keys[i]] = result
        return [self.cache[k] for k in keys]

    def propagate(self, segments, duration, nominal, plants, samples):
        # Segmentwise streaming: the largest H array is 16*768*4*4, NEVER
        # a concatenated 24*768 waveform. No array has >6144 time slots.
        nh = len(plants)
        carried = np.broadcast_to(np.eye(4, dtype=complex), (nh, 4, 4)).copy()
        metrics = [dict(duration=0., gap_action=0., command_action=0.,
                        applied_action=0., full_H_action=0., peak_requested=0.,
                        peak_applied=0., sampled_requested_slew=0.,
                        sampled_applied_slew=0., segments=len(segments), gap=1.)
                   for _ in plants]
        s = (np.arange(samples) + .5) / samples
        dt = duration / samples
        smooth, dsmooth = ramp(s), ramp_derivative(s) / duration
        for segment in segments:
            self.run.check(segments=1)
            self.run.cost["segment_batches"] += 1
            self.run.cost["modeled_segment_hypotheses"] += nh
            self.run.cost["controller_samples"] += samples
            if segment[0] == "direct":
                _, axis, area = segment
                # Smooth C2 pulse, integral_0^1 64*s^3*(1-s)^3 ds=16/35.
                desired = area / (duration * (16 / 35)) * 64 * s ** 3 * (1 - s) ** 3
                request = inverse_response(desired, nominal)
                pair = (1, 4) if axis == "X" else (1, 2)
                basis = self.bil[pair]
                applied = np.array([response(request, p) for p in plants])
                self.run.cost["inverse_component_evaluations"] += samples
                # These commuting finite Hamiltonians permit an exact scalar
                # reduction; midpoint quadrature of the inverse pulse remains.
                integrated = self.idle[None] * duration + basis[None] * (
                    applied.sum(axis=1) * dt)[:, None, None]
                vals, vecs = np.linalg.eigh(integrated)
                self.run.cost["hermitian_eigendecompositions_4d"] += nh
                exps = (vecs * np.exp(-1j * vals)[:, None, :]) @ vecs.conj().swapaxes(-1, -2)
                carried = exps @ carried
                self.run.cost["matrix_exponentials_4d"] += nh
                hs = self.idle[None, None] + applied[:, :, None, None] * basis
                q = request[:, None]
                applied_components = applied[:, :, None]
                hnorm = np.max(np.abs(np.linalg.eigvalsh(hs)), axis=-1)
                self.run.cost["hermitian_eigendecompositions_4d"] += nh * samples
            else:
                _, pair, t0, t1, x0, x1 = segment
                polar = t0 + (t1 - t0) * smooth
                polar_dot = (t1 - t0) * dsmooth
                x = x0 + (x1 - x0) * smooth
                x_dot = (x1 - x0) * dsmooth
                phi_nom = np.pi / 2 * warp(x, nominal[0], nominal[1])
                phi_dot = np.pi / 2 * warp_derivative(x, nominal[0], nominal[1]) * x_dot
                sn, cs = np.sin(polar), np.cos(polar)
                n_nom = np.array([sn * np.cos(phi_nom), sn * np.sin(phi_nom), cs]).T
                dn_nom = np.array([
                    polar_dot * cs * np.cos(phi_nom) - sn * np.sin(phi_nom) * phi_dot,
                    polar_dot * cs * np.sin(phi_nom) + sn * np.cos(phi_nom) * phi_dot,
                    -polar_dot * sn]).T
                local_pairs = ((0, 1), (0, 2), (1, 2))
                desired = np.stack([.5 * (n_nom[:, j] * dn_nom[:, i]
                                                   - n_nom[:, i] * dn_nom[:, j])
                                    for i, j in local_pairs], axis=1)
                q = inverse_response(desired, nominal)
                self.run.cost["inverse_component_evaluations"] += samples * 3
                applied_components = np.array([response(q, p) for p in plants])
                active = (pair[0], pair[1], 3)
                gap_basis = np.array([self.bil[0, j] for j in active])
                cd_basis = []
                for i, j in local_pairs:
                    a, b = active[i], active[j]
                    cd_basis.append(self.bil[min(a, b), max(a, b)] * (1 if a < b else -1))
                cd_basis = np.array(cd_basis)
                phi_actual = np.array([np.pi / 2 * warp(x, p[0], p[1]) for p in plants])
                n_actual = np.stack([sn * np.cos(phi_actual), sn * np.sin(phi_actual),
                                     np.broadcast_to(cs, (nh, samples))], axis=2)
                hs = np.einsum("htc,cij->htij", n_actual, gap_basis)
                hs += np.einsum("htc,cij->htij", applied_components, cd_basis)
                vals, vecs = np.linalg.eigh(hs)
                self.run.cost["hermitian_eigendecompositions_4d"] += nh * samples
                hnorm = np.max(np.abs(vals), axis=-1)
                exps = (vecs * np.exp(-1j * vals * dt)[:, :, None, :]) @ vecs.conj().swapaxes(-1, -2)
                self.run.cost["matrix_exponentials_4d"] += nh * samples
                for k in range(samples):
                    if k % 64 == 0:
                        self.run.check()
                    carried = exps[:, k] @ carried
                self.run.cost["midpoint_matrix_products"] += nh * samples
            if float(np.max(np.abs(q))) > 1 + 1e-12:
                raise Stop("requested correction exceeds stipulated cubic domain")
            if not np.all(np.isfinite(carried)):
                raise Stop("nonfinite modeled propagator")
            # These are sampled diagnostics, not certified continuous extrema.
            req_slew = np.max(np.abs(np.diff(np.vstack([np.zeros((1, q.shape[1])), q,
                                                       np.zeros((1, q.shape[1]))]), axis=0))) / dt
            for h in range(nh):
                app = applied_components[h]
                app_slew = np.max(np.abs(np.diff(np.vstack([np.zeros((1, app.shape[1])), app,
                                                           np.zeros((1, app.shape[1]))]), axis=0))) / dt
                m = metrics[h]
                m["duration"] += duration
                m["gap_action"] += duration
                m["command_action"] += float(np.sum(q * q) * dt)
                m["applied_action"] += float(np.sum(app * app) * dt)
                m["full_H_action"] += float(np.sum(hnorm[h] ** 2) * dt)
                m["peak_requested"] = max(m["peak_requested"], float(np.max(np.abs(q))))
                m["peak_applied"] = max(m["peak_applied"], float(np.max(np.abs(app))))
                m["sampled_requested_slew"] = max(m["sampled_requested_slew"], float(req_slew))
                m["sampled_applied_slew"] = max(m["sampled_applied_slew"], float(app_slew))
        return [Result(carried[h], metrics[h]) for h in range(nh)]

    def measure(self, result, basis):
        state = result.U @ (self.code @ (np.array([1., 1.]) / np.sqrt(2)))
        operator = self.code @ (SX if basis == "X" else SY) @ self.code.conj().T
        # Leakage yields a fair bit; there is no uncharged postselection.
        return float(np.clip((1 + np.vdot(state, operator @ state).real) / 2, 0, 1))


class Evaluator:
    """The only holder of the hidden plant index and teacher angles.

    Isolation is code-interface discipline, not OS blindness. Raw evaluator
    audit records contain the truth, but learner selection never reads them.
    """
    def __init__(self, run):
        self.run = run
        streams = np.random.SeedSequence(SEED).spawn(3)
        self._plant_rng, self._task_rng, self._cal_rng = [np.random.default_rng(s) for s in streams]
        self._true = int(self._plant_rng.integers(16))
        self._targets = (np.pi / 4, np.pi / 6, np.pi / 5)
        run.emit("evaluator_only_truth", true_index=self._true,
                 true_plant=PRIOR[self._true], teacher_angles=self._targets,
                 learner_access="counts only; raw truth is not read by learner")

    def task(self, stage):
        beta = self._targets[stage]
        answers = {}
        for basis, p in (("X", (1 + np.cos(beta)) / 2),
                         ("Y", (1 + np.sin(beta)) / 2)):
            answers[basis] = self.run.sample(self._task_rng, float(p), 65536,
                                            "target", stage=stage + 1, basis=basis)
        return answers

    def calibration(self, query, n, oracle_probabilities, duration, action):
        p = float(oracle_probabilities[self._true])
        answer = self.run.sample(self._cal_rng, p, n, "calibration", query=query)
        self.run.cost["observation_evolution"] += n * duration
        self.run.cost["observation_gap_action"] += n * duration
        self.run.cost["observation_applied_action"] += n * action[self._true]
        self.run.cost["calibration_routing_configurations"] += 1
        return answer

    def actual_result(self, model, op, theta, family, samples=ORACLE):
        return model.gate(op, theta, family, [PRIOR[self._true]], samples)[0]

    def true_pair(self):
        return self._true, PRIOR[self._true]

    def target_for_audit(self, op):
        return rz(self._targets[op["stage"] - 1]) if op["kind"] == "new" else ideal(op)


def confidence(n, successes):
    eps = np.sqrt(np.log(2 / DELTA_BATCH) / (2 * n))
    return max(0., successes / n - eps), min(1., successes / n + eps)


def task_operation(stage, counts):
    px, py = [confidence(*counts[b]) for b in ("X", "Y")]
    prior_lo, prior_hi = np.pi / 10, 2 * np.pi / 5
    # cos decreases, sin increases on this frozen interval.
    lo = max(prior_lo, np.arccos(np.clip(2 * px[1] - 1, -1, 1)),
             np.arcsin(np.clip(2 * py[0] - 1, -1, 1)))
    hi = min(prior_hi, np.arccos(np.clip(2 * px[0] - 1, -1, 1)),
             np.arcsin(np.clip(2 * py[1] - 1, -1, 1)))
    if lo > hi:
        raise Stop("empty target-angle confidence set")
    return dict(name="new" + str(stage + 1), kind="new", axis="Z", stage=stage + 1,
                beta=float((lo + hi) / 2), interval=[float(lo), float(hi)],
                halfwidth=float((hi - lo) / 2))


def ideal(op):
    return (rx if op["axis"] == "X" else rz)(op["beta"])


def assess(model, op, results):
    desired = model.code @ ideal(op)
    errors = [aligned_error(r.U @ model.code, desired) for r in results]
    leakage = [float(np.linalg.norm((np.eye(4) - model.code @ model.code.conj().T)
                                    @ r.U @ model.code, 2)) for r in results]
    worst = int(np.argmax(errors))
    score = max(errors) + op["halfwidth"] / 2 + NUM_ALLOWANCE
    return dict(errors=errors, worst_position=worst, worst_error=max(errors),
                max_leakage=max(leakage), score=score,
                feasible=bool(score <= ADMISSION and
                              max(r.metrics["duration"] for r in results) <=
                              (24. if op["kind"] == "old" else 192.)),
                metrics_worst_error=results[worst].metrics,
                maximum_duration=max(r.metrics["duration"] for r in results),
                maximum_applied_action=max(r.metrics["applied_action"] for r in results))


def query_menu(model, run):
    menu = []
    for q in (.25, .75):
        probabilities = np.array([np.cos(response(q, p)) ** 2 for p in PRIOR])
        menu.append(dict(name="rabi_" + str(q), probabilities=probabilities,
                         oracle=probabilities, duration=1.,
                         action=np.array([response(q, p) ** 2 for p in PRIOR]),
                         numerical_difference=0., formula="full commuting idle-pair Hamiltonian"))
    for width in (1 / 3, 2 / 3):
        coarse = model.geometry_query(width, SAMPLES)
        oracle = model.geometry_query(width, ORACLE)
        for basis in ("X", "Y"):
            p = np.array([model.measure(r, basis) for r in coarse])
            po = np.array([model.measure(r, basis) for r in oracle])
            mismatch = float(np.max(np.abs(p - po)))
            run.emit("calibration_likelihood_numerical_check", width=width,
                     basis=basis, midpoint_samples=SAMPLES, oracle_samples=ORACLE,
                     maximum_difference=mismatch, allowance=NUM_ALLOWANCE,
                     rigorous_enclosure=False)
            if mismatch > NUM_ALLOWANCE:
                raise Stop("geometry-query likelihood numerical allowance exceeded")
            menu.append(dict(name="wedge_" + str(width) + "_" + basis,
                             probabilities=p, oracle=po, duration=12.,
                             action=np.array([r.metrics["applied_action"] for r in oracle]),
                             numerical_difference=mismatch,
                             formula="full finite four-dimensional parity evolution"))
    return menu


def calibrate(stage, U, records, menu, evaluator, run):
    available = list(range(6))
    n = (2048, 8192, 32768)[stage]
    for item in range(6):
        if len(U) <= 1:
            break
        # Declared information-per-evolution-cost proxy, not forced six queries.
        choice = max(available, key=lambda k: (
            float(np.var(menu[k]["probabilities"][U])) / menu[k]["duration"], -k))
        available.remove(choice)
        query = menu[choice]
        counts = evaluator.calibration(query["name"], n, query["oracle"],
                                       query["duration"], query["action"])
        low, high = confidence(*counts)
        p = query["probabilities"]
        U = [i for i in U if low - NUM_ALLOWANCE <= p[i] <= high + NUM_ALLOWANCE]
        records.append((choice, *counts))
        run.emit("calibration_update", stage=stage + 1, menu_item=query["name"],
                 counts=counts, statistical_probability_interval=[low, high],
                 numerical_likelihood_allowance=NUM_ALLOWANCE,
                 retained_hypotheses=U, modeled_probabilities=p,
                 no_desired_result_narrowing=True)
        if not U:
            raise Stop("empty plant uncertainty set")
    loglike = np.zeros(16)
    for choice, shots, wins in records:
        p = np.clip(menu[choice]["probabilities"], 1e-12, 1 - 1e-12)
        loglike += wins * np.log(p) + (shots - wins) * np.log1p(-p)
    nominal_index = max(U, key=lambda i: (loglike[i], -i))
    run.emit("paid_data_nominal_fit", stage=stage + 1,
             retained_hypotheses=U, nominal_index=nominal_index,
             nominal_plant=PRIOR[nominal_index], log_likelihoods=loglike,
             observations_used=len(records), all_arms_share_same_data=True,
             per_arm_minimum_needed_observations="not measured; common diagnostic purchase only")
    return U, PRIOR[nominal_index]


def interface_checks(model, run, gammas, parity, select):
    ident = np.eye(8)
    anti = max(float(np.linalg.norm(a @ b + b @ a - (2 * ident if i == j else 0)))
               for i, a in enumerate(gammas) for j, b in enumerate(gammas))
    comm = max(float(np.linalg.norm(parity @ a @ b - a @ b @ parity))
               for a in gammas for b in gammas)
    code = model.code
    code_checks = dict(rank=int(np.linalg.matrix_rank(code)),
                       orthogonality=float(np.linalg.norm(code.conj().T @ code - I2)),
                       idle_ground=float(np.linalg.norm(model.idle @ code + code)),
                       logical_z=float(np.linalg.norm(code.conj().T @ model.bil[1, 2] @ code - SZ)),
                       logical_x=float(np.linalg.norm(code.conj().T @ model.bil[1, 4] @ code - SX)))
    if anti > 1e-12 or comm > 1e-12 or code_checks["rank"] != 2 or max(
            v for k, v in code_checks.items() if k != "rank") > 1e-12:
        raise Stop("Majorana/parity/common-code algebra interface failure")
    run.emit("algebra_interface_check", anticommutator=anti,
             even_bilinear_parity_commutator=comm, full_dimension=8,
             fixed_parity_dimension=select.shape[1], code=code_checks,
             spectator_modes=[4, 5], phase_convention="B12=Z, B14=X; one global phase per uncontrolled call")
    oldz = dict(name="Rz", kind="old", axis="Z", beta=np.pi / 2, halfwidth=0.)
    oldx = dict(name="Rx", kind="old", axis="X", beta=np.pi / 2, halfwidth=0.)
    checked = {}
    for samples in (SAMPLES, ORACLE):
        for op in (oldz, oldx):
            result = model.gate(op, (.5, 2., NEUTRAL), "moment", [NEUTRAL], samples)[0]
            error = aligned_error(result.U @ code, code @ ideal(op))
            unitary = float(np.linalg.norm(result.U.conj().T @ result.U - np.eye(4)))
            run.emit("known_geometry_CD_sign_check", operation=op["name"], samples=samples,
                     path="north -> xR=1 equator -> xL=0 equator -> north",
                     declared_positive_exchange=True, error=error, unitarity=unitary,
                     sign_selected_from_root_derivation=True, hidden_target_used=False)
            if error > NUM_ALLOWANCE or unitary > 1e-10:
                raise Stop("known-geometry CD orientation/unitarity failed")
            checked[op["name"]] = result.U
    physical_x = checked["Rx"] @ checked["Rx"]
    x_error = aligned_error(physical_x @ code, code @ (-1j * SX))
    noncommutation = float(np.linalg.norm((physical_x @ checked["Rz"]
                                          - checked["Rz"] @ physical_x) @ code, 2))
    if x_error > 2 * NUM_ALLOWANCE or noncommutation < 1:
        raise Stop("physical echoed X/noncommuting old-library interface failed")
    for a in (.1, .25, .4):
        for rho in RHOS:
            q = chart(a, rho)
            p1, p2 = polynomial(q)
            moments = [q[1] - q[0] + q[3] - q[2] - a,
                       p1[1] - p1[0] + p1[3] - p1[2],
                       p2[1] - p2[0] + p2[3] - p2[2]]
            if max(abs(np.array(moments))) > 1e-12:
                raise Stop("chart moment interface failure")
    run.emit("physical_X_and_chart_check", physical_X_two_exchanges=True,
             physical_X_error=x_error, X_Z_noncommutator=noncommutation,
             finite_grid_moment_smoke=True, not_a_continuous_numeric_certificate=True)
    return [oldz, oldx]


def candidates(nominal, incumbent):
    choices = [(rho, leg, tuple(nominal)) for rho in RHOS for leg in LEGS]
    if incumbent is not None and incumbent not in choices:
        choices.append(incumbent)
    return choices


def search(stage, ops, U, nominal, incumbent, family, model, run):
    plants = [PRIOR[i] for i in U]
    table = []
    if family == "no_moment":
        # rho does not affect a centered contiguous interval. Do not charge
        # five duplicate physical proposals to this ablation.
        choices = [(.5, leg, tuple(nominal)) for leg in LEGS]
        normalized_incumbent = ((.5, incumbent[1], incumbent[2])
                                if incumbent is not None else None)
        if normalized_incumbent is not None and normalized_incumbent not in choices:
            choices.append(normalized_incumbent)
    else:
        choices = candidates(nominal, incumbent)
    for theta in choices:
        run.check()
        run.cost["candidate_proposals"] += 1
        assessments = []
        for op in ops:
            results = model.gate(op, theta, family, plants)
            assessments.append(assess(model, op, results))
        feasible = all(a["feasible"] for a in assessments)
        if not feasible:
            run.cost["rejected_proposals"] += 1
        row = dict(theta=theta, assessments=assessments, feasible=feasible)
        table.append(row)
        run.emit("candidate_model_proposal", stage=stage + 1, family=family,
                 theta=theta, contexts=[op["name"] for op in ops],
                 retained_hypotheses=U, feasible_all_retained=feasible,
                 assessments=assessments, query_shots=0)
    return table


def choice_key(row, index):
    a = row["assessments"][index]
    return (a["maximum_duration"], a["maximum_applied_action"],
            a["score"], row["theta"][0], row["theta"][1], row["theta"][2])


def best_for_context(table, index):
    feasible = [row for row in table if row["assessments"][index]["feasible"]]
    return min(feasible, key=lambda row: choice_key(row, index)) if feasible else None


def serialized_arm_map_bytes(arm):
    return len(encoded(arm))


def initial_old_library(old, U, nominal, model, run):
    """Establish actual-model old contracts BEFORE the first target batch."""
    table = []
    for leg in LEGS:
        theta = (.5, leg, tuple(nominal))
        assessments = [assess(model, op, model.gate(op, theta, "moment",
                        [PRIOR[i] for i in U])) for op in old]
        feasible = all(a["feasible"] for a in assessments)
        table.append(dict(theta=theta, assessments=assessments, feasible=feasible))
        run.cost["candidate_proposals"] += 1
        run.emit("initial_old_library_model_proposal", theta=theta,
                 retained_hypotheses=U, assessments=assessments,
                 all_old_feasible=feasible, no_task_targets_yet=True)
    feasible = [row for row in table if row["feasible"]]
    if not feasible:
        raise Stop("initial routed old library fails actual-model contracts; stream not started")
    selected = min(feasible, key=lambda row: (row["theta"][1],
                   sum(a["maximum_applied_action"] for a in row["assessments"]),
                   max(a["score"] for a in row["assessments"])))
    entry_checks = []
    for op in old:
        coarse = model.gate(op, selected["theta"], "moment", [PRIOR[i] for i in U], SAMPLES)
        refined = model.gate(op, selected["theta"], "moment", [PRIOR[i] for i in U], ORACLE)
        differences = [aligned_error(a.U @ model.code, b.U @ model.code)
                       for a, b in zip(coarse, refined)]
        refined_assessment = assess(model, op, refined)
        entry_checks.append(dict(operation=op["name"], numerical_differences=differences,
                                 refined_assessment=refined_assessment))
        run.emit("initial_old_library_refined_check", operation=op["name"],
                 retained_hypotheses=U, numerical_differences=differences,
                 refined_assessment=refined_assessment, before_first_target_copy=True)
        if max(differences) > NUM_ALLOWANCE or not refined_assessment["feasible"]:
            raise Stop("initial old-library refined numerical/model admission failed; stream not started")
        if op["axis"] == "X":
            physical_errors = [aligned_error(r.U @ r.U @ model.code,
                                             model.code @ (-1j * SX)) for r in refined]
            run.emit("initial_actual_physical_X", component="two routed Rx exchanges",
                     modeled_errors=physical_errors, retained_hypotheses=U,
                     derived_two_call_error_budget=2 * ADMISSION,
                     duration=2 * refined[0].metrics["duration"],
                     no_ideal_Pauli_injected=True)
    run.emit("initial_old_library_established", theta=selected["theta"],
             assessments=selected["assessments"],
             entry_refined_checks=entry_checks, before_first_target_copy=True,
             provisional_numerical_status=True)
    return selected["theta"]


def waveform_descriptor(op, theta, family):
    """Finite equal-phase-position samples of EXECUTED controls; no target oracle.

    Include actual H0 geometry only via the nominal controller's path command;
    evaluator separately compares the resulting physical operator. Duration is
    a distinct coordinate, so timing changes remain visible at fixed shape.
    """
    segments, leg = program(op, theta, family)
    s = (np.arange(33) + .5) / 33
    values = []
    for segment in segments:
        if segment[0] == "direct":
            desired = segment[2] / (leg * (16 / 35)) * 64 * s ** 3 * (1 - s) ** 3
            values.append(inverse_response(desired, theta[2]).ravel())
            continue
        _, pair, t0, t1, x0, x1 = segment
        polar = t0 + (t1 - t0) * ramp(s)
        dp = (t1 - t0) * ramp_derivative(s) / leg
        x = x0 + (x1 - x0) * ramp(s)
        dx = (x1 - x0) * ramp_derivative(s) / leg
        phi = np.pi / 2 * warp(x, theta[2][0], theta[2][1])
        dphi = np.pi / 2 * warp_derivative(x, theta[2][0], theta[2][1]) * dx
        n = np.array([np.sin(polar) * np.cos(phi), np.sin(polar) * np.sin(phi),
                      np.cos(polar)]).T
        dn = np.array([dp * np.cos(polar) * np.cos(phi) - np.sin(polar) * np.sin(phi) * dphi,
                       dp * np.cos(polar) * np.sin(phi) + np.sin(polar) * np.cos(phi) * dphi,
                       -dp * np.sin(polar)]).T
        desired = np.stack([.5 * (n[:, j] * dn[:, i] - n[:, i] * dn[:, j])
                            for i, j in ((0, 1), (0, 2), (1, 2))], axis=1)
        # H0 commanded direction is unwarped; nominal warp informs CD only.
        commanded_n = np.stack([np.sin(polar) * np.cos(np.pi / 2 * x),
                                np.sin(polar) * np.sin(np.pi / 2 * x),
                                np.cos(polar)], axis=1)
        values.extend([commanded_n.ravel(), inverse_response(desired, theta[2]).ravel()])
    return np.concatenate(values)


def record_shared_change(stage, old_ops, before, after, evaluator, model, run):
    changes = []
    for op in old_ops:
        before_wave = waveform_descriptor(op, before, "moment")
        after_wave = waveform_descriptor(op, after, "moment")
        run.cost["controller_samples"] += 66 * len(program(op, after, "moment")[0])
        run.cost["inverse_component_evaluations"] += 198 * len(program(op, after, "moment")[0])
        before_result = evaluator.actual_result(model, op, before, "moment")
        after_result = evaluator.actual_result(model, op, after, "moment")
        changes.append(dict(operation=op["name"],
                            sampled_command_waveform_L2=float(np.linalg.norm(after_wave - before_wave)),
                            sampled_command_waveform_Linf=float(np.max(np.abs(after_wave - before_wave))),
                            duration_before=before_result.metrics["duration"],
                            duration_after=after_result.metrics["duration"],
                            physical_code_operator_change=aligned_error(
                                after_result.U @ model.code, before_result.U @ model.code)))
    run.emit("actual_shared_controller_before_after", stage=stage + 1,
             theta_before=before, theta_after=after,
             rho_before=before[0], rho_after=after[0],
             nominal_compensation_before=before[2], nominal_compensation_after=after[2],
             leg_before=before[1], leg_after=after[1], old_context_changes=changes,
             zero_change_is_recorded_not_forced=True,
             task_scalar_growth_alone_is_not_shared_continual_learning=True)


def execute_words(stage, ops, arms, evaluator, model, run):
    latest = ops[-1]["name"]
    words = [("Rx", latest, "Rz"), (latest, "Rx", latest)]
    lookup = {op["name"]: op for op in ops}
    for arm_name, arm in arms.items():
        if latest not in arm["controllers"]:
            continue
        for word in words:
            if any(name not in arm["controllers"] for name in word):
                run.emit("word_unavailable", arm=arm_name, word=word)
                continue
            carried = model.code.copy()  # one preparation; NEVER reset between calls
            wanted = I2.copy()
            hidden_wanted = I2.copy()
            duration = action = target_allowance = 0.
            for name in word:
                op = lookup[name]
                theta = arm["controllers"][name]
                result = evaluator.actual_result(model, op, theta, arm["family"])
                carried = result.U @ carried
                wanted = ideal(op) @ wanted
                hidden_wanted = evaluator.target_for_audit(op) @ hidden_wanted
                duration += result.metrics["duration"]
                action += result.metrics["applied_action"]
                target_allowance += op["halfwidth"] / 2
            leakage = np.linalg.norm((np.eye(4) - model.code @ model.code.conj().T) @ carried, 2)
            run.cost["word_input_episodes"] += 2
            run.cost["word_simulated_evolution"] += 2 * duration
            run.emit("common_code_noncommuting_word_execution", stage=stage + 1,
                     arm=arm_name, chronological_word=word, carried_isometry=carried,
                     no_intermediate_repreparation=True, independent_input_columns=2,
                     estimated_target_error=aligned_error(carried, model.code @ wanted),
                     evaluator_only_true_target_error=aligned_error(carried, model.code @ hidden_wanted),
                     target_uncertainty_allowance=target_allowance,
                     leakage=float(leakage), duration=duration, applied_action=action,
                     deterministic_simulator_diagnostic=True, readout_shots=0)


def final_checks(ops, arms, U, evaluator, model, run):
    """Final library, rather than every transient search proposal, gets 384/768.

    Check COARSE-grid worst retained plant and true plant, deduplicating
    identical selections. The fine-grid worst plant could differ and is not
    exhaustively scanned. This is an exploratory check, not an enclosure.
    Stop honestly at36 selected pairs.
    """
    pending = {}
    true_index, true_plant = evaluator.true_pair()
    for arm_name, arm in arms.items():
        for op in ops:
            if op["name"] not in arm["controllers"]:
                continue
            theta, family = arm["controllers"][op["name"]], arm["family"]
            results = model.gate(op, theta, family, [PRIOR[i] for i in U])
            assessment = assess(model, op, results)
            worst_id = U[assessment["worst_position"]]
            for plant_id in dict.fromkeys((worst_id, true_index)):
                key = model.key(op, theta, family, PRIOR[plant_id], SAMPLES)
                pending.setdefault(key, dict(op=op, theta=theta, family=family,
                                             plant_id=plant_id, arms=[]))["arms"].append(arm_name)
    if len(pending) > MAX_FINAL_PAIRS:
        run.emit("final_check_cap_exceeded", required_pairs=len(pending), cap=MAX_FINAL_PAIRS)
        raise Stop("more than 36 selected final gate/plant pairs; admissions remain provisional")
    for record in pending.values():
        run.check()
        run.cost["final_gate_plant_pairs"] += 1
        op, theta, family, pid = (record[k] for k in ("op", "theta", "family", "plant_id"))
        coarse = model.gate(op, theta, family, [PRIOR[pid]], SAMPLES)[0]
        mid = model.gate(op, theta, family, [PRIOR[pid]], ORACLE)[0]
        fine = model.gate(op, theta, family, [PRIOR[pid]], FINAL)[0]
        difference_192_384 = aligned_error(coarse.U @ model.code, mid.U @ model.code)
        difference_384_768 = aligned_error(mid.U @ model.code, fine.U @ model.code)
        error_fine = aligned_error(fine.U @ model.code, model.code @ ideal(op))
        valid = max(difference_192_384, difference_384_768) <= NUM_ALLOWANCE
        check = dict(operation=op["name"], family=family, theta=theta,
                     plant_id=pid, arms=record["arms"],
                     differences=[difference_192_384, difference_384_768],
                     fine_error=error_fine, numerical_allowance=NUM_ALLOWANCE,
                     score_fine=error_fine + op["halfwidth"] / 2 + NUM_ALLOWANCE,
                     valid=bool(valid), rigorous_enclosure=False)
        check["plant_selection"] = "coarse192-worst plus true; fine-grid worst may differ"
        run.final_checks.append(check)
        run.emit("final_selected_gate_numerical_check", **check)
        if not valid or check["score_fine"] > ADMISSION:
            raise Stop("selected final admission invalidated by numerical/refined-error check")


def main():
    run = Run()
    try:
        gammas, parity, select, bil, idle, code = algebra()
        model = Model(run, bil, idle, code)
        run.progress["phase"] = "interface_checks"
        old = interface_checks(model, run, gammas, parity, select)
        evaluator = Evaluator(run)
        menu = query_menu(model, run)
        U, observation_records, ops = list(range(16)), [], list(old)
        target_count_records = []
        incumbent = no_moment_incumbent = None
        frozen = {}
        modular_shapes = {}
        arms = {}
        for stage in range(3):
            run.progress.update(stage=stage + 1, phase="paid_calibration")
            U, nominal = calibrate(stage, U, observation_records, menu, evaluator, run)
            if stage == 0:
                incumbent = initial_old_library(old, U, nominal, model, run)
                frozen = {o["name"]: incumbent for o in old}
                modular_shapes = {o["name"]: incumbent[:2] for o in old}
                arms["shared"] = dict(family="moment", controllers=dict(frozen))
            run.progress["phase"] = "target_observations"
            counts = evaluator.task(stage)
            target_count_records.append(counts)
            op = task_operation(stage, counts)
            run.emit("task_learned_from_counts", operation=op, counts=counts,
                     beta_not_supplied_to_learner=True,
                     operation_status="requested_pending_admission")
            previous_shared_ops = [o for o in ops if o["name"] in arms["shared"]["controllers"]]
            ops.append(op)
            run.progress["phase"] = "shared_and_UGD_search"
            table = search(stage, ops, U, nominal, incumbent, "moment", model, run)
            shared_feasible = [row for row in table if row["feasible"]]
            selected = min(shared_feasible, key=lambda row: choice_key(row, len(ops) - 1)) if shared_feasible else None
            previous_incumbent = incumbent
            if selected is not None:
                incumbent = selected["theta"]
                arms["shared"] = dict(family="moment", controllers={o["name"]: incumbent for o in ops})
                record_shared_change(stage, previous_shared_ops, previous_incumbent,
                                     incumbent, evaluator, model, run)
            else:
                # Preserve the last actual library and do not install a rejected theta.
                run.emit("shared_new_operation_rejected", stage=stage + 1,
                         reason="no one shared proposal retains all contracts",
                         previous_incumbent=incumbent)
            # UGD can pick a separate scalar chart/time for EVERY old/new context.
            # This weakly contains the shared feasible choices, by construction.
            ugd = {o["name"]: best_for_context(table, i)["theta"]
                   for i, o in enumerate(ops) if best_for_context(table, i) is not None}
            arms["UGD"] = dict(family="moment", controllers=ugd)
            for i, o in enumerate(ops):
                if o["name"] not in frozen:
                    choice = best_for_context(table, i)
                    if choice is not None:
                        frozen[o["name"]] = choice["theta"]
            arms["frozen_waveform"] = dict(family="moment", controllers=dict(frozen))
            frozen_assessments = {}
            for o in ops:
                if o["name"] in frozen:
                    results = model.gate(o, frozen[o["name"]], "moment", [PRIOR[i] for i in U])
                    frozen_assessments[o["name"]] = assess(model, o, results)
            # A distinct, stronger modular policy freezes shape/timing scalars
            # while legitimately refreshing its shared nominal calibration.
            # Every refreshed operation is actually repropagated/recertified.
            modular_controllers, modular_assessments, modular_failed = {}, {}, []
            for i, o in enumerate(ops):
                if o["name"] not in modular_shapes:
                    choice = best_for_context(table, i)
                    if choice is not None:
                        modular_shapes[o["name"]] = choice["theta"][:2]
                if o["name"] not in modular_shapes:
                    modular_failed.append(o["name"])
                    continue
                shape = modular_shapes[o["name"]]
                theta = (shape[0], shape[1], tuple(nominal))
                result = model.gate(o, theta, "moment", [PRIOR[j] for j in U])
                assessment = assess(model, o, result)
                modular_assessments[o["name"]] = assessment
                if assessment["feasible"]:
                    modular_controllers[o["name"]] = theta
                else:
                    modular_failed.append(o["name"])
            arms["modular_shape_shared_calibration"] = dict(
                family="moment", controllers=modular_controllers,
                frozen_shapes=dict(modular_shapes), failed_contexts=modular_failed)
            run.emit("modular_shape_shared_calibration_recertification", stage=stage + 1,
                     frozen_shapes=modular_shapes, shared_nominal_compensation=nominal,
                     assessments=modular_assessments, failed_contexts=modular_failed,
                     old_feasibility_not_assumed=True, extra_observation_shots=0)
            run.emit("moment_controller_choices", stage=stage + 1,
                     shared_selected=incumbent if selected is not None else None,
                     previous_shared=previous_incumbent,
                     shared_parameters_changed=bool(selected is not None and incumbent != previous_incumbent),
                     UGD_operation_specific_choices=ugd,
                     UGD_allowed_identical_chart_and_independent_scalar_caching=True,
                     frozen_choices=frozen, frozen_assessments=frozen_assessments,
                     provisional_until_final_convergence=True,
                     shared_acquired_contexts=list(arms["shared"]["controllers"]),
                     requested_contexts=[o["name"] for o in ops],
                     finite_grid_only_not_continuous_safe_interval=True)
            # Direct has no rho dependence. Three bounded duration choices suffice.
            run.progress["phase"] = "direct_control"
            direct_table = []
            for leg in LEGS:
                theta = (.5, leg, tuple(nominal))
                run.cost["candidate_proposals"] += 1
                assessments = [assess(model, o, model.gate(o, theta, "direct", [PRIOR[i] for i in U])) for o in ops]
                row = dict(theta=theta, assessments=assessments)
                direct_table.append(row)
                run.emit("direct_model_proposal", stage=stage + 1, theta=theta,
                         assessments=assessments, legal_idle_pair_actuator=True,
                         exact_commuting_reduction=True, query_shots=0)
            arms["direct"] = dict(family="direct", controllers={o["name"]: best_for_context(direct_table, i)["theta"]
                                  for i, o in enumerate(ops) if best_for_context(direct_table, i) is not None})
            # Strong controls and words are checkpointed before the more expensive
            # ablation, preserving an informative first-stage comparison on timeout.
            execute_words(stage, ops, arms, evaluator, model, run)
            run.progress["phase"] = "no_moment_ablation"
            ablated_table = search(stage, ops, U, nominal, no_moment_incumbent,
                                   "no_moment", model, run)
            ablated_feasible = [row for row in ablated_table if row["feasible"]]
            ablated_selected = min(ablated_feasible, key=lambda row: choice_key(row, len(ops) - 1)) if ablated_feasible else None
            if ablated_selected is not None:
                no_moment_incumbent = ablated_selected["theta"]
                arms["no_moment"] = dict(family="no_moment", controllers={o["name"]: no_moment_incumbent for o in ops})
                execute_words(stage, ops, {"no_moment": arms["no_moment"]}, evaluator, model, run)
            else:
                run.emit("no_moment_new_operation_rejected", stage=stage + 1,
                         previous_incumbent=no_moment_incumbent)
            run.emit("stage_checkpoint", stage=stage + 1, learned_operation=op,
                     retained_hypotheses=U, arms=arms,
                     requested_contexts=[o["name"] for o in ops],
                     acquired_contexts_by_arm={name: list(arm["controllers"]) for name, arm in arms.items()},
                     serialized_arm_map_bytes={name: serialized_arm_map_bytes(arm) for name, arm in arms.items()},
                     common_semantic_calibration_state_bytes=len(encoded(dict(
                         ops=ops, U=U, calibration_counts=observation_records,
                         target_counts=target_count_records))),
                     audit_file_bytes_before_this_checkpoint=dict(
                         packed_bits=run.bits.tell(), raw_jsonl=run.raw.tell(),
                         combined=run.bytes),
                     byte_measure="canonical serialized records; not runtime RAM, full waveform storage or FLOPs",
                     target_task_oracle_evolution_cost="unspecified; target copies/preparations/readouts counted, no zero-time assumption",
                     common_observation_cost=run.cost,
                     model_cache_entries=len(model.cache),
                     approximate_cached_matrix_bytes=sum(r.U.nbytes for r in model.cache.values()),
                     angle_intervals_retained=[o.get("interval") for o in ops],
                     admissions="PROVISIONAL_PENDING_FINAL_NUMERICAL_CHECKS",
                     success_requires_actual_shared_profile_changes_and_full_cost_review=True)
            if selected is None:
                # A rejected task does not silently become an old retained
                # obligation or get reacquired retrospectively at the next stage.
                run.emit("stream_stopped_after_rejected_shared_task", stage=stage + 1,
                         controls_checkpointed=True, broader_topology_hypothesis="OPEN")
                raise Stop("shared acquisition failed; finite version stopped after controls")
        run.progress["phase"] = "final_selected_numerical_checks"
        final_checks(ops, arms, U, evaluator, model, run)
        run.progress["phase"] = "complete_first_stream"
        run.finish("FIRST_STREAM_FINISHED_EXPLORATORY", arms=arms,
                   retained_hypotheses=U, operations=ops,
                   evidence_boundary="finite simulated Born outcomes and numerical checks; no rigorous enclosure, hardware, novelty or M0 passage",
                   further_streams="not executed; require scientific justification and remaining budget")
    except Exception as error:
        run.finish("STOPPED_WITH_PARTIAL_EVIDENCE", error_type=type(error).__name__,
                   reason=str(error), admissions="unconfirmed unless explicit completed final checks exist")
        raise


if __name__ == "__main__":
    main()
