"""ROOT-RUN loop-class acquisition from terminal callbacks, not supplied maps.

MAIN=TOPOLOGY, USER_LOCKED, human-only change. Same Prometheus V2.
No import-time experiment, subprocess, network, installation, or remote access.
This NEW standalone diagnostic does not modify the current native prototype.

The executor really continues all three roots of z**3-3*z-u=0 along polygon
loops. Its full-fiber endpoint observation is exposed, charged, and available
to EVERY arm; no transposition table is an execution oracle. Terminal labels
are ordinary synthetic diagnostic data, not claims about the six-stage stream.

The event arm is intentionally an exact same-mechanism correspondence with an
ordinary discontinuous-program boundary optimizer. No superiority or historical
novelty follows from that identity. The score and discrete-search arms receive
the same source, event geometry, continuation, and terminal-feedback interface.

Numerical continuation checks are diagnostic checks, NOT interval certificates.
Failures are explicit exceptions/censoring, never fabricated losses or gradients.

ROOT may inspect and run, with an OUTSIDE timeout, for example:
  timeout 120s python loop_boundary_learning.py --out NEW_DIRECTORY --seconds 90
The internal shared deadline and per-arm query/step/update caps also terminate.
Nothing in this module authorizes that run or capacity growth.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn


class BudgetStop(RuntimeError):
    """A real externally configured diagnostic limit has been reached."""


class NumericalFailure(RuntimeError):
    """Unresolved execution/event geometry; must not become a task loss."""


@dataclass(frozen=True)
class Limits:
    max_feedback: int = 1600
    max_programs: int = 2000
    max_continuation_steps: int = 160000
    max_step_attempts: int = 240000
    max_events: int = 30000
    max_updates: int = 24
    max_word_length: int = 16
    max_vertices: int = 16
    min_control_clearance: float = 0.015
    max_control_coordinate: float = 8.0
    max_control_step: float = 0.12
    max_newton: int = 16
    residual_tolerance: float = 1e-10
    endpoint_tolerance: float = 1e-7


@dataclass
class CostLedger:
    limits: Limits
    deadline: float
    counts: dict[str, int] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    writer: Callable[[dict[str, Any]], None] | None = None

    _bounded = {
        "feedback_calls": "max_feedback",
        "program_calls": "max_programs",
        "continuation_steps": "max_continuation_steps",
        "step_attempts": "max_step_attempts",
        "event_candidates": "max_events",
        "updates": "max_updates",
    }

    def check(self) -> None:
        if time.monotonic() >= self.deadline:
            raise BudgetStop("shared monotonic wall deadline reached")

    def add(self, key: str, count: int = 1) -> None:
        self.check()
        value = self.counts.get(key, 0) + int(count)
        bound = self._bounded.get(key)
        if bound is not None and value > getattr(self.limits, bound):
            raise BudgetStop(f"{key} would exceed {getattr(self.limits, bound)}")
        self.counts[key] = value

    def elapsed(self, key: str, started: float) -> None:
        self.seconds[key] = self.seconds.get(key, 0.0) + time.perf_counter() - started

    def emit(self, kind: str, **values: Any) -> None:
        if self.writer is not None:
            self.writer({"kind": kind, **values})

    def snapshot(self) -> dict[str, Any]:
        return {"counts": dict(self.counts), "seconds": dict(self.seconds),
                "limits": asdict(self.limits)}


def _cross(a: complex, b: complex) -> float:
    return a.real * b.imag - a.imag * b.real


def segment_distance(point: complex, left: complex, right: complex) -> float:
    delta = right - left
    if abs(delta) == 0:
        return abs(point - left)
    fraction = ((point - left).real * delta.real
                + (point - left).imag * delta.imag) / abs(delta) ** 2
    fraction = min(1.0, max(0.0, fraction))
    return abs(point - (left + fraction * delta))


def loop_clearance(vertices: Sequence[complex], critical: Sequence[complex]) -> float:
    points = (0j, *vertices, 0j)
    return min(segment_distance(c, p, q)
               for p, q in zip(points, points[1:]) for c in critical)


class AffineLoopFamily:
    """Vertices = template + sum(action[j] * basis[j]); basepoint is always 0.

    General affine coefficient slices may move several vertices. Event detection
    uses the actual quadratic segment equation and counts every calculation.
    Only a raw single-vertex-coordinate basis has the sharper <=2B event bound.
    There is no clipping or renormalized/truncated exploration distribution.
    """

    def __init__(self, template: Sequence[complex], basis: Sequence[Sequence[complex]],
                 action_box: tuple[float, float] = (-6.0, 6.0)):
        self.template = tuple(complex(v) for v in template)
        self.basis = tuple(tuple(complex(v) for v in row) for row in basis)
        self.action_box = tuple(float(v) for v in action_box)
        if not self.template or not self.basis:
            raise ValueError("nonempty vertex template and deformation basis required")
        if any(len(row) != len(self.template) for row in self.basis):
            raise ValueError("basis row/vertex dimension mismatch")
        if not self.action_box[0] < self.action_box[1]:
            raise ValueError("nonempty declared action box required")

    @property
    def dimension(self) -> int:
        return len(self.basis)

    def vertices(self, action: Sequence[float]) -> tuple[complex, ...]:
        if len(action) != self.dimension:
            raise ValueError("action dimension mismatch")
        if any(not math.isfinite(float(a)) or not self.action_box[0] <= a <= self.action_box[1]
               for a in action):
            raise NumericalFailure("action outside its declared finite box")
        return tuple(v + sum(float(a) * row[k] for a, row in zip(action, self.basis))
                     for k, v in enumerate(self.template))

    def require_support(self, mean: Sequence[float], sigma: float) -> None:
        lo, hi = self.action_box
        if sigma <= 0 or any(not math.isfinite(float(a))
                             or a - sigma < lo or a + sigma > hi for a in mean):
            raise NumericalFailure("mean plus UNTRUNCATED smoothing support leaves action box")

    def configuration(self) -> dict[str, Any]:
        encode = lambda values: [[v.real, v.imag] for v in values]
        return {"template": encode(self.template),
                "basis": [encode(row) for row in self.basis],
                "action_box": list(self.action_box)}

    @classmethod
    def from_configuration(cls, config: Mapping[str, Any]) -> "AffineLoopFamily":
        decode = lambda values: [complex(*v) for v in values]
        return cls(decode(config["template"]), [decode(row) for row in config["basis"]],
                   tuple(config["action_box"]))


class SharedVertexController(nn.Module):
    """A real NN whose shared weights produce loop vertices via an affine family.

    Features are public operation inputs, not target maps. All supplied/frozen
    features and all allocated trainable parameters are reported in checkpoints.
    The first diagnostic uses a two-weight linear controller. A nonlinear hidden
    layer is available to callers, without a claim that the small check tests it.
    """

    def __init__(self, features: Sequence[Sequence[float]], action_dim: int,
                 hidden: int = 0):
        super().__init__()
        features_tensor = torch.as_tensor(features, dtype=torch.float64)
        if features_tensor.ndim != 2 or features_tensor.shape[0] == 0:
            raise ValueError("require operation-by-feature matrix")
        self.register_buffer("features", features_tensor)
        self.action_dim, self.hidden = int(action_dim), int(hidden)
        if hidden:
            self.network = nn.Sequential(nn.Linear(features_tensor.shape[1], hidden),
                                         nn.Tanh(), nn.Linear(hidden, action_dim))
        else:
            self.network = nn.Linear(features_tensor.shape[1], action_dim, bias=False)
        self.double()

    def forward(self, operation: int) -> torch.Tensor:
        return self.network(self.features[int(operation)])

    def action(self, operation: int) -> list[float]:
        return self(operation).detach().cpu().tolist()

    def configuration(self) -> dict[str, Any]:
        return {"features": self.features.detach().cpu().tolist(),
                "action_dim": self.action_dim, "hidden": self.hidden}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_hash(self) -> str:
        # JSON is an audit signature, never a substitute for the saved tensors.
        values = [p.detach().cpu().reshape(-1).tolist() for p in self.parameters()]
        return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


class CubicCoveringExecutor:
    """Actual safeguarded predictor/corrector path continuation of all 3 roots.

    Full-fiber computation is deliberately disclosed. The terminal receipt gives
    the entire ACTUALLY CONTINUED endpoint map to all rivals at the same cost.
    It is not cached here. A task callback still labels only its requested case.
    The fixed source/critical controls are known to all arms; no table of their
    sheet-transposition effects appears in this executor.
    """

    critical_controls = (-2 + 0j, 2 + 0j)
    base_roots = (-math.sqrt(3) + 0j, 0j, math.sqrt(3) + 0j)

    def __init__(self, ledger: CostLedger):
        self.ledger = ledger

    @staticmethod
    def polynomial(z: complex, u: complex) -> complex:
        return z * z * z - 3 * z - u

    def _correct(self, prediction: complex, u: complex) -> tuple[complex, float]:
        z = prediction
        for _ in range(self.ledger.limits.max_newton):
            self.ledger.add("newton_iterations")
            residual = self.polynomial(z, u)
            self.ledger.add("polynomial_evaluations")
            if abs(residual) <= self.ledger.limits.residual_tolerance:
                return z, abs(residual)
            derivative = 3 * z * z - 3
            self.ledger.add("derivative_evaluations")
            if abs(derivative) < 1e-12:
                raise NumericalFailure("small derivative in Newton correction")
            z -= residual / derivative
            if not math.isfinite(z.real) or not math.isfinite(z.imag):
                raise NumericalFailure("nonfinite Newton iterate")
        raise NumericalFailure("Newton iteration cap")

    def execute(self, loops: Sequence[Sequence[complex]]) -> dict[str, Any]:
        self.ledger.add("program_calls")
        started = time.perf_counter()
        limits = self.ledger.limits
        if not 1 <= len(loops) <= limits.max_word_length:
            raise ValueError("word length outside external process bounds")
        roots = list(self.base_roots)  # Marked initial roots, carried THROUGH calls.
        local_steps = 0
        minimum_separation = math.inf
        minimum_clearance = math.inf
        max_residual = 0.0
        try:
            for vertices in loops:
                if not 1 <= len(vertices) <= limits.max_vertices:
                    raise ValueError("polygon vertex count outside process bound")
                if any(not math.isfinite(v.real) or not math.isfinite(v.imag)
                       or max(abs(v.real), abs(v.imag)) > limits.max_control_coordinate
                       for v in vertices):
                    raise NumericalFailure("control vertex outside declared finite box")
                clearance = loop_clearance(vertices, self.critical_controls)
                self.ledger.add("execution_clearance_segment_checks",
                                (len(vertices) + 1) * len(self.critical_controls))
                minimum_clearance = min(minimum_clearance, clearance)
                if clearance < limits.min_control_clearance:
                    raise NumericalFailure("control loop below executable clearance")
                points = (0j, *vertices, 0j)
                for left, right in zip(points, points[1:]):
                    segment = right - left
                    length = abs(segment)
                    if length == 0:
                        continue
                    fraction = 0.0
                    max_du = limits.max_control_step * min(1.0, math.sqrt(clearance))
                    nominal = min(1.0, max_du / length)
                    while fraction < 1.0:
                        self.ledger.check()
                        step = min(nominal, 1.0 - fraction)
                        accepted = False
                        for _ in range(25):
                            self.ledger.add("step_attempts")
                            delta_u = step * segment
                            u_new = left + (fraction + step) * segment
                            separation = min(abs(a - b) for i, a in enumerate(roots)
                                             for b in roots[i + 1:])
                            try:
                                predictions = []
                                for root in roots:
                                    derivative = 3 * root * root - 3
                                    self.ledger.add("derivative_evaluations")
                                    if abs(derivative) < 1e-12:
                                        raise NumericalFailure("small continuation derivative")
                                    predictions.append(root + delta_u / derivative)
                                corrected_and_residual = [self._correct(p, u_new)
                                                          for p in predictions]
                                corrected = [pair[0] for pair in corrected_and_residual]
                                residual = max(pair[1] for pair in corrected_and_residual)
                                movement = max(abs(a - b) for a, b in zip(roots, corrected))
                                correction = max(abs(a - b) for a, b in zip(predictions, corrected))
                                new_separation = min(abs(a - b) for i, a in enumerate(corrected)
                                                     for b in corrected[i + 1:])
                                if (movement > 0.18 * separation
                                        or correction > 0.06 * separation
                                        or new_separation < 1e-6):
                                    raise NumericalFailure("continuation locality/separation check")
                            except NumericalFailure:
                                self.ledger.add("step_halvings")
                                step *= 0.5
                                if step < 1e-12:
                                    break
                                continue
                            roots = corrected
                            fraction = min(1.0, fraction + step)
                            minimum_separation = min(minimum_separation, new_separation)
                            max_residual = max(max_residual, residual)
                            self.ledger.add("continuation_steps")
                            self.ledger.add("root_strand_steps", 3)
                            local_steps += 1
                            accepted = True
                            break
                        if not accepted:
                            raise NumericalFailure("continuation step refinement exhausted")
            endpoint_map = []
            for root in roots:
                distances = [abs(root - base) for base in self.base_roots]
                self.ledger.add("endpoint_distance_evaluations", 3)
                closest = min(range(3), key=distances.__getitem__)
                if distances[closest] > limits.endpoint_tolerance:
                    raise NumericalFailure("endpoint does not return to resting fiber")
                endpoint_map.append(closest)
            if len(set(endpoint_map)) != 3:
                raise NumericalFailure("continued fiber endpoints are not bijective")
            return {"endpoint_map": endpoint_map, "continuation_steps": local_steps,
                    "minimum_root_separation": minimum_separation,
                    "minimum_control_clearance": minimum_clearance,
                    "maximum_polynomial_residual": max_residual,
                    "observation": "all three continued endpoints, available to every arm",
                    "certificate_scope": "numerical checks, not interval certification"}
        except (BudgetStop, NumericalFailure):
            self.ledger.add("execution_failures")
            raise
        finally:
            self.ledger.elapsed("continuation", started)


@dataclass(frozen=True)
class ProgramExample:
    name: str
    initial_sheet: int
    word: tuple[int, ...]
    target: int


Feedback = Callable[[ProgramExample, int, Mapping[str, Any]], float]


def zero_one_terminal_feedback(example: ProgramExample, observed_sheet: int,
                               receipt: Mapping[str, Any]) -> float:
    del receipt
    return float(observed_sheet != example.target)


class TerminalObjective:
    """No desired map: execute programs, then call the supplied terminal callback.

    The caller chooses exactly which examples a gradient/evaluation may access.
    Full endpoint receipts are available, not silently withheld from competitors.
    No memoization is performed; feedback reuse or map caches would need an
    explicit storage/cost policy. Failure remains an exception.
    """

    def __init__(self, controller: SharedVertexController, family: AffineLoopFamily,
                 executor: CubicCoveringExecutor, examples: Sequence[ProgramExample],
                 feedback: Feedback = zero_one_terminal_feedback):
        if not examples:
            raise ValueError("nonempty terminal-example objective required")
        self.controller, self.family, self.executor = controller, family, executor
        self.examples, self.feedback = tuple(examples), feedback

    def __call__(self, overrides: Mapping[int, Sequence[float]] | None = None) -> float:
        overrides = overrides or {}
        ledger = self.executor.ledger
        total = 0.0
        for example in self.examples:
            ledger.check()
            ledger.add("feedback_calls")  # Reserve BEFORE the costly execution.
            actions = {op: list(overrides[op]) if op in overrides else self.controller.action(op)
                       for op in set(example.word)}
            ledger.add("controller_operation_outputs", len(set(example.word) - set(overrides)))
            loops = [self.family.vertices(actions[op]) for op in example.word]
            receipt = self.executor.execute(loops)
            observed = receipt["endpoint_map"][example.initial_sheet]
            started = time.perf_counter()
            loss = float(self.feedback(example, observed, receipt))
            ledger.elapsed("terminal_callback", started)
            if not math.isfinite(loss) or not 0.0 <= loss <= 1.0:
                raise ValueError("this bounded diagnostic requires callback loss in [0,1]")
            total += loss
            ledger.emit("terminal_feedback", example=asdict(example), actions=actions,
                        observed_sheet=observed, loss=loss, receipt=receipt)
        return total / len(self.examples)


class RaisedCosineKernel:
    """Exactly normalized compact kernel; rejection draws and proposals counted."""

    def __init__(self, sigma: float):
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("positive finite smoothing scale required")
        self.sigma = float(sigma)

    def density(self, displacement: float) -> float:
        if abs(displacement) >= self.sigma:
            return 0.0
        return (1 + math.cos(math.pi * displacement / self.sigma)) / (2 * self.sigma)

    def score(self, displacement: float) -> float:
        if not abs(displacement) < self.sigma:
            raise ValueError("location score requires an interior kernel sample")
        return math.pi / self.sigma * math.tan(math.pi * displacement / (2 * self.sigma))

    def sample(self, rng: random.Random, ledger: CostLedger) -> float:
        while True:
            ledger.add("smoothing_proposals")
            ledger.add("random_uniform_draws", 2)
            value = self.sigma * (2 * rng.random() - 1)
            if rng.random() < (1 + math.cos(math.pi * value / self.sigma)) / 2:
                ledger.add("smoothing_samples")
                return value
            ledger.add("smoothing_rejections")


def _real_quadratic_roots(a: float, b: float, c: float) -> list[float]:
    scale = max(abs(a), abs(b), abs(c), 1.0)
    tolerance = 1e-12 * scale
    if abs(a) <= tolerance:
        if abs(b) <= tolerance:
            if abs(c) <= tolerance:
                raise NumericalFailure("nongeneric slice segment persistently collinear with critical value")
            return []
        return [-c / b]
    discriminant = b * b - 4 * a * c
    if discriminant < -tolerance * scale:
        return []
    if abs(discriminant) <= tolerance * scale:
        raise NumericalFailure("unresolved tangent/coincident event on chosen slice")
    root = math.sqrt(discriminant)
    q = -0.5 * (b + math.copysign(root, b))
    return [q / a, c / q]


def slice_events(family: AffineLoopFamily, fixed_action: Sequence[float], coordinate: int,
                 left: float, right: float, critical: Sequence[complex],
                 ledger: CostLedger) -> list[float]:
    """Find actual discriminant intersections on a scalar affine action slice.

    No monodromy/permutation labels are inferred from event positions. Coincident
    or tangent events are unresolved rather than assigned arbitrary jump signs.
    Boundary locations at support endpoints have zero kernel weight and need no
    representative outside the declared smoothing support.
    """
    started = time.perf_counter()
    fixed = list(float(v) for v in fixed_action)
    fixed[coordinate] = 0.0
    # Zero is in the declared first family's box. For a caller with another box,
    # evaluate the affine expression directly, since this is an algebraic slice
    # origin, not an executed action or exploration sample.
    origins = tuple(v + sum(a * row[k] for a, row in zip(fixed, family.basis))
                    for k, v in enumerate(family.template))
    directions = family.basis[coordinate]
    positions = (0j, *origins, 0j)
    motions = (0j, *directions, 0j)
    events: list[float] = []
    try:
        for u, v, du, dv in zip(positions, positions[1:], motions, motions[1:]):
            if abs(du) == 0 and abs(dv) == 0:
                continue
            for c in critical:
                ledger.add("event_polynomial_calculations")
                aa = _cross(du, dv)
                bb = _cross(du, v - c) + _cross(u - c, dv)
                cc = _cross(u - c, v - c)
                for event in _real_quadratic_roots(aa, bb, cc):
                    ledger.add("event_candidates")
                    if not left < event < right:
                        continue
                    endpoint_u, endpoint_v = u + event * du, v + event * dv
                    if abs(endpoint_u - c) < 1e-9 or abs(endpoint_v - c) < 1e-9:
                        raise NumericalFailure("event meets polygon vertex; nongeneric slice")
                    product = ((c - endpoint_u).real * (c - endpoint_v).real
                               + (c - endpoint_u).imag * (c - endpoint_v).imag)
                    if product <= 1e-10:
                        events.append(event)
        events.sort()
        for a, b in zip(events, events[1:]):
            if abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)):
                raise NumericalFailure("coincident discriminant events on slice")
        ledger.add("accepted_events", len(events))
        return events
    finally:
        ledger.elapsed("event_detection", started)


def ordinary_boundary_gradient(mean: Sequence[float], kernel: RaisedCosineKernel,
                               objective: Callable[[Sequence[float]], float],
                               event_provider: Callable[[Sequence[float], int, float, float], list[float]],
                               rng: random.Random, ledger: CostLedger,
                               transverse_samples: int = 1) -> tuple[list[float], dict[str, Any]]:
    """Generic discontinuous-program optimizer with the SAME event information.

    This function has no topology/cover knowledge. The topology event arm below
    deliberately delegates to it: their exact ordinary correspondence is exposed.
    All coordinates are estimated; sampling a subset would need 1/q weighting.
    """
    if transverse_samples < 1:
        raise ValueError("at least one transverse sample required")
    gradient = [0.0] * len(mean)
    details: list[dict[str, Any]] = []
    for coordinate in range(len(mean)):
        for _ in range(transverse_samples):
            action = list(float(value) for value in mean)
            for other in range(len(mean)):
                if other != coordinate:
                    action[other] += kernel.sample(rng, ledger)
            left, right = mean[coordinate] - kernel.sigma, mean[coordinate] + kernel.sigma
            events = event_provider(action, coordinate, left, right)
            if not events:
                details.append({"coordinate": coordinate, "events": [], "gradient": 0.0})
                continue
            boundaries = [left, *events, right]
            losses, representatives = [], []
            for low, high in zip(boundaries, boundaries[1:]):
                if high - low < 1e-9:
                    raise NumericalFailure("unresolved narrow event interval")
                representative = (low + high) / 2
                action[coordinate] = representative
                # Any failed representative remains unresolved; never drop it or
                # fabricate a zero loss while continuing the jump estimator.
                losses.append(float(objective(action)))
                representatives.append(representative)
                ledger.add("chamber_representative_evaluations")
            jumps = [b - a for a, b in zip(losses, losses[1:])]
            contribution = sum(jump * kernel.density(event - mean[coordinate])
                               for jump, event in zip(jumps, events))
            gradient[coordinate] += contribution / transverse_samples
            details.append({"coordinate": coordinate, "events": events,
                            "representatives": representatives, "losses": losses,
                            "jumps": jumps, "gradient": contribution})
    ledger.emit("boundary_estimate", mean=list(mean), gradient=gradient, slices=details,
                ordinary_correspondence="same generic boundary optimizer and event information")
    return gradient, {"slices": details}


def loop_boundary_gradient(mean: Sequence[float], family: AffineLoopFamily,
                           executor: CubicCoveringExecutor, kernel: RaisedCosineKernel,
                           objective: Callable[[Sequence[float]], float], rng: random.Random,
                           transverse_samples: int = 1) -> tuple[list[float], dict[str, Any]]:
    family.require_support(mean, kernel.sigma)
    provider = lambda action, j, left, right: slice_events(
        family, action, j, left, right, executor.critical_controls, executor.ledger)
    return ordinary_boundary_gradient(mean, kernel, objective, provider, rng,
                                      executor.ledger, transverse_samples)


def score_gradient(mean: Sequence[float], family: AffineLoopFamily,
                   kernel: RaisedCosineKernel, objective: Callable[[Sequence[float]], float],
                   rng: random.Random, ledger: CostLedger, samples: int = 2,
                   baseline: float = 0.0) -> tuple[list[float], dict[str, Any]]:
    family.require_support(mean, kernel.sigma)
    if samples < 1 or not math.isfinite(baseline):
        raise ValueError("positive sample count and fixed finite baseline required")
    gradient = [0.0] * len(mean)
    records = []
    for _ in range(samples):
        noise = [kernel.sample(rng, ledger) for _ in mean]
        action = [a + delta for a, delta in zip(mean, noise)]
        loss = objective(action)
        scores = [kernel.score(delta) for delta in noise]
        for j, score in enumerate(scores):
            gradient[j] += (loss - baseline) * score / samples
        records.append({"noise": noise, "action": action, "loss": loss, "scores": scores})
    ledger.emit("score_estimate", mean=list(mean), gradient=gradient, draws=records,
                fixed_baseline=baseline, normalized_density="(1+cos(pi*t/sigma))/(2*sigma)")
    return gradient, {"draws": records}


class ProtectedLoopSet:
    """Protect admitted controller inputs, including their actual input binding.

    Fixed sheet encoder/readout are a declared first-diagnostic limitation.
    The predicate is a real-arithmetic sufficient homotopy guard evaluated in
    doubles; it is not an interval certificate. No old terminal table is needed
    for the guard. Current old vertices are recomputed and transiently stored.
    """

    def __init__(self, admitted: Iterable[int] = ()):
        self.admitted = set(int(op) for op in admitted)
        self.receipts: dict[int, dict[str, Any]] = {}

    def before(self, controller: SharedVertexController, family: AffineLoopFamily,
               executor: CubicCoveringExecutor) -> dict[int, tuple[tuple[complex, ...], float]]:
        snapshot = {}
        for operation in sorted(self.admitted):
            executor.ledger.add("protection_controller_outputs")
            vertices = family.vertices(controller.action(operation))
            clearance = loop_clearance(vertices, executor.critical_controls)
            executor.ledger.add("protection_segment_checks",
                                (len(vertices) + 1) * len(executor.critical_controls))
            if clearance < executor.ledger.limits.min_control_clearance:
                raise NumericalFailure("already admitted loop lacks executable clearance")
            snapshot[operation] = vertices, clearance
        return snapshot

    def accepts(self, controller: SharedVertexController, family: AffineLoopFamily,
                snapshot: Mapping[int, tuple[tuple[complex, ...], float]],
                ledger: CostLedger) -> bool:
        for operation, (old_vertices, clearance) in snapshot.items():
            ledger.add("protection_controller_outputs")
            current = family.vertices(controller.action(operation))
            displacement = max(abs(a - b) for a, b in zip(old_vertices, current))
            ledger.add("protection_vertex_comparisons", len(current))
            if not displacement < clearance:
                return False
        return True

    def admit(self, operation: int, controller: SharedVertexController,
              family: AffineLoopFamily, executor: CubicCoveringExecutor) -> dict[str, Any]:
        # Admission's semantic receipt comes ONLY from actual continuation.
        vertices = family.vertices(controller.action(operation))
        receipt = executor.execute([vertices])
        self.admitted.add(int(operation))
        self.receipts[int(operation)] = {"action": controller.action(operation),
                                         "actual_lift_receipt": receipt}
        executor.ledger.emit("primitive_admission", operation=int(operation),
                             receipt=self.receipts[int(operation)])
        return receipt


def protected_parameter_step(controller: SharedVertexController, family: AffineLoopFamily,
                             executor: CubicCoveringExecutor, protected: ProtectedLoopSet,
                             parameter_directions: Sequence[torch.Tensor],
                             current_operation: int, sigma: float,
                             scales: Sequence[float] = (1., .5, .25, .125, .0625, .03125, 0.)) -> float:
    """Apply an explicit proposal, backtracking; restore exactly on every failure.

    No optimizer state is hidden/reset: this module uses explicit SGD proposals.
    A future Adam integration must transact its optimizer state explicitly.
    """
    ledger = executor.ledger
    started = time.perf_counter()
    snapshot = protected.before(controller, family, executor)
    parameters = list(controller.parameters())
    if len(parameters) != len(parameter_directions):
        raise ValueError("proposal/parameter list mismatch")
    old = [p.detach().clone() for p in parameters]
    committed = False
    try:
        for scale in scales:
            ledger.add("protection_proposals")
            with torch.no_grad():
                for parameter, original, direction in zip(parameters, old, parameter_directions):
                    parameter.copy_(original + float(scale) * direction)
            try:
                family.require_support(controller.action(current_operation), sigma)
                new_vertices = family.vertices(controller.action(current_operation))
                ledger.add("current_clearance_checks")
                current_regular = (loop_clearance(new_vertices, executor.critical_controls)
                                   >= ledger.limits.min_control_clearance)
                allowed = current_regular and protected.accepts(controller, family, snapshot, ledger)
            except NumericalFailure:
                allowed = False
            if allowed:
                ledger.add("updates")
                ledger.add("parameter_scalar_assignments", sum(p.numel() for p in parameters))
                ledger.emit("protected_update", accepted_scale=scale,
                            parameter_hash=controller.parameter_hash(),
                            current_action=controller.action(current_operation))
                committed = True
                return float(scale)
            ledger.add("protection_rejections")
        raise NumericalFailure("even restored parameters violate the declared transaction")
    finally:
        if not committed:
            with torch.no_grad():
                for parameter, original in zip(parameters, old):
                    parameter.copy_(original)
        ledger.elapsed("protection_transaction", started)


def neural_gradient_step(controller: SharedVertexController, family: AffineLoopFamily,
                         executor: CubicCoveringExecutor, protected: ProtectedLoopSet,
                         operation: int, gradient: Sequence[float], learning_rate: float,
                         sigma: float) -> float:
    controller.zero_grad(set_to_none=True)
    output = controller(operation)
    output.backward(output.new_tensor(gradient))
    executor.ledger.add("neural_backward_calls")
    directions = [(-learning_rate * p.grad.detach()) if p.grad is not None else torch.zeros_like(p)
                  for p in controller.parameters()]
    return protected_parameter_step(controller, family, executor, protected, directions,
                                    operation, sigma)


def discrete_class_step(controller: SharedVertexController, family: AffineLoopFamily,
                        executor: CubicCoveringExecutor, protected: ProtectedLoopSet,
                        operation: int, kernel: RaisedCosineKernel,
                        objective: Callable[[Sequence[float]], float]) -> dict[str, Any]:
    """Affordable exact 1D chamber search; same events and ACTUAL executions.

    Search the complete allowed mean-action interval and fit its selected path
    representative into the SAME two-weight NN while preserving old actions.
    This is a strong trivial rival for the scalar witness, not a hidden oracle.
    No extrapolation to a complete high-dimensional class enumerator is made.
    """
    if family.dimension != 1 or controller.hidden:
        raise ValueError("this bounded discrete control supports a 1D linear controller")
    ledger = executor.ledger
    mean = controller.action(operation)
    low, high = (family.action_box[0] + kernel.sigma,
                 family.action_box[1] - kernel.sigma)
    events = slice_events(family, mean, 0, low, high,
                          executor.critical_controls, ledger)
    boundaries = [low, *events, high]
    candidates = []
    for left, right in zip(boundaries, boundaries[1:]):
        action = [(left + right) / 2]
        candidates.append((objective(action), action))
        ledger.add("discrete_chamber_evaluations")
    loss, selected = min(candidates, key=lambda item: item[0])
    # Geometry targets are this arm's CURRENT old actions and a paid, selected
    # new path representative. No desired semantic map enters this fit.
    operations = [*sorted(protected.admitted), operation]
    features = controller.features[operations]
    targets = controller.network(features).detach().clone()
    targets[-1] = targets.new_tensor(selected)
    ledger.add("neural_geometric_least_squares_calls")
    solution = torch.linalg.lstsq(features, targets).solution.T
    parameter = controller.network.weight
    directions = [solution - parameter.detach()]
    scale = protected_parameter_step(controller, family, executor, protected, directions,
                                     operation, kernel.sigma)
    ledger.emit("discrete_search", events=events, candidates=candidates,
                selected_action=selected, selected_loss=loss, accepted_scale=scale,
                information="same complete 1D event geometry and terminal feedback")
    return {"selected_loss": loss, "selected_action": selected, "accepted_scale": scale}


def checkpoint_payload(controller: SharedVertexController, family: AffineLoopFamily,
                       protected: ProtectedLoopSet, ledger: CostLedger) -> dict[str, Any]:
    """Actual NN lifecycle boundary; no supplied permutations in the primitive bank.

    Existing Method integration: bank slots reference controller inputs and real
    loop execution; call slots concatenate them carrying the same sheet. D=3 and
    fixed E/R are explicit here. Current promotion/growth/Adam code is untouched.
    Replacing that lifecycle requires a separately reviewed adapter and costs.
    """
    bank = {}
    for operation in sorted(protected.admitted):
        vertices = family.vertices(controller.action(operation))
        bank[operation] = {"controller_input": operation,
                           "vertices": [[v.real, v.imag] for v in vertices],
                           "origin": "controller parameters; admission by actual continuation",
                           "admission_receipt": copy.deepcopy(protected.receipts.get(operation))}
    return {"version": "loop-boundary-v1", "main": "TOPOLOGY_USER_LOCKED",
            "controller_configuration": controller.configuration(),
            "state_dict": {k: v.detach().cpu().clone() for k, v in controller.state_dict().items()},
            "family_configuration": family.configuration(), "primitive_bank": bank,
            "state_capacity": 3, "encoding_readout": "fixed resting cubic sheets",
            "parameters_total": controller.parameter_count(),
            "feature_buffer_scalars": controller.features.numel(),
            "optimizer": "explicit SGD; no Adam state in this diagnostic",
            "costs": ledger.snapshot(), "parameter_hash": controller.parameter_hash()}


def restore_checkpoint(payload: Mapping[str, Any]) -> tuple[SharedVertexController, AffineLoopFamily, ProtectedLoopSet]:
    controller = SharedVertexController(**payload["controller_configuration"])
    controller.load_state_dict(payload["state_dict"], strict=True)
    family = AffineLoopFamily.from_configuration(payload["family_configuration"])
    protected = ProtectedLoopSet(int(op) for op in payload["primitive_bank"])
    protected.receipts = {int(op): values["admission_receipt"]
                          for op, values in payload["primitive_bank"].items()}
    if controller.parameter_hash() != payload["parameter_hash"]:
        raise NumericalFailure("actual NN checkpoint parameter hash mismatch")
    return controller, family, protected


def _diagnostic_data() -> tuple[tuple[ProgramExample, ...], tuple[ProgramExample, ...]]:
    # Author-specified TERMINAL records for the diagnostic only. There is no
    # target-permutation object consumed by the controller, solver, or gradient.
    training = (
        ProgramExample("new-start0", 0, (1,), 1),
        ProgramExample("new-then-old", 0, (1, 0), 2),
        ProgramExample("old-then-new", 1, (0, 1), 2),
        ProgramExample("repeated-new", 2, (1, 0, 1), 0),
    )
    heldout = (
        ProgramExample("heldout-order-a", 0, (1, 0, 1), 2),
        ProgramExample("heldout-order-b", 1, (1, 0), 0),
        ProgramExample("heldout-initial2", 2, (0, 1), 0),
        ProgramExample("heldout-suffix", 0, (0, 1, 0), 2),
    )
    return training, heldout


def _sample_variance(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def run_arm(mode: str, output: Path, limits: Limits, deadline: float,
            seed: int, sigma: float, learning_rate: float,
            gradient_draws: int, score_samples: int) -> dict[str, Any]:
    output.mkdir(exist_ok=False)
    stream = (output / "records.jsonl").open("x", encoding="utf-8")
    def write(record: dict[str, Any]) -> None:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
    ledger = CostLedger(limits, deadline, writer=write)
    rng = random.Random(seed)
    controller = SharedVertexController(((1., -1.), (1., 1.)), action_dim=1)
    with torch.no_grad():
        controller.network.weight.copy_(torch.tensor([[-1., 2.]], dtype=torch.float64))
    family = AffineLoopFamily((1j, -1j), ((1 + 0j, 1 + 0j),))
    executor = CubicCoveringExecutor(ledger)
    protected = ProtectedLoopSet()
    kernel = RaisedCosineKernel(sigma)
    training, heldout = _diagnostic_data()
    objective = TerminalObjective(controller, family, executor, training)
    current_objective = lambda action: objective({1: action})
    results: dict[str, Any] = {"arm": mode, "status": "started", "seed": seed,
        "parameters_total": controller.parameter_count(), "sigma": sigma,
        "feedback_contract": "same terminal records and full paid continuation observations",
        "ordinary_identity": "event and ordinary_event invoke the same generic boundary optimizer",
        "source": "fixed internally authored z^3-3z-u; no desired map supplied"}
    try:
        old_receipt = protected.admit(0, controller, family, executor)
        old_targets = old_receipt["endpoint_map"]
        old_examples = tuple(ProgramExample(f"old-self-native-{s}", s, (0,), target)
                             for s, target in enumerate(old_targets))
        old_objective = TerminalObjective(controller, family, executor, old_examples)
        initial_hash = controller.parameter_hash()
        results["initial_old_native_map"] = old_targets
        results["initial_new_action"] = controller.action(1)
        ledger.emit("frozen_arm", configuration=results, limits=asdict(limits),
                    training=[asdict(e) for e in training],
                    heldout_record_count=len(heldout), family=family.configuration(),
                    controller=controller.configuration(), initial_parameter_hash=initial_hash)

        # Credit/cost calibration, followed by genuine shared-weight acquisition.
        # Event is deterministic in this 1D slice; score gets the SAME objective
        # and its own honestly counted samples. This is not the acquisition test.
        if mode != "discrete":
            estimates, calibration_start = [], dict(ledger.counts)
            for _ in range(gradient_draws):
                mean = controller.action(1)
                if mode == "score":
                    gradient, _ = score_gradient(mean, family, kernel, current_objective,
                                                  rng, ledger, score_samples)
                else:
                    # Explicitly identical for event and ordinary_event, with
                    # actual evaluations repeated/counted in each separate arm.
                    gradient, _ = loop_boundary_gradient(mean, family, executor, kernel,
                                                          current_objective, rng)
                estimates.append(gradient[0])
            results["credit_calibration"] = {
                "estimates": estimates,
                "sample_variance": _sample_variance(estimates),
                "additional_feedback": ledger.counts.get("feedback_calls", 0)
                    - calibration_start.get("feedback_calls", 0),
                "does_not_establish": "equal-cost acquisition superiority"}

        trajectory = []
        acquired = False
        for iteration in range(limits.max_updates):
            ledger.check()
            before_parameters = [p.detach().clone() for p in controller.parameters()]
            mean = controller.action(1)
            if mode == "discrete":
                result = discrete_class_step(controller, family, executor, protected, 1,
                                              kernel, current_objective)
                gradient = None
                accepted_scale = result["accepted_scale"]
            else:
                if mode == "score":
                    gradient, _ = score_gradient(mean, family, kernel, current_objective,
                                                  rng, ledger, score_samples)
                else:
                    gradient, _ = loop_boundary_gradient(mean, family, executor, kernel,
                                                          current_objective, rng)
                accepted_scale = neural_gradient_step(controller, family, executor, protected,
                                                       1, gradient, learning_rate, sigma)
            displacement = math.sqrt(sum(float(torch.sum((p.detach() - old) ** 2))
                                         for p, old in zip(controller.parameters(), before_parameters)))
            training_loss = objective()  # Deterministic mean-path admission test.
            old_loss = old_objective()
            row = {"iteration": iteration, "action_before": mean,
                   "new_action": controller.action(1), "old_action": controller.action(0),
                   "gradient": gradient, "accepted_scale": accepted_scale,
                   "shared_parameter_displacement": displacement,
                   "training_loss": training_loss, "old_loss": old_loss,
                   "parameter_hash": controller.parameter_hash(), "costs": ledger.snapshot()}
            trajectory.append(row)
            ledger.emit("learning_curve", **row)
            if old_loss != 0.0:
                raise NumericalFailure("old semantics failed despite the numeric homotopy guard")
            if training_loss == 0.0:
                protected.admit(1, controller, family, executor)
                acquired = True
                break
        results["trajectory"] = trajectory
        results["acquired"] = acquired
        results["parameter_count_unchanged"] = controller.parameter_count() == 2
        results["actual_shared_parameters_changed"] = controller.parameter_hash() != initial_hash
        results["heldout_loss"] = TerminalObjective(controller, family, executor, heldout)()
        results["final_old_loss"] = old_objective()
        results["final_actions"] = {str(op): controller.action(op) for op in (0, 1)}
        results["final_parameter_hash"] = controller.parameter_hash()

        # Genuine current NN tensors, controller-input bank, and actual admission
        # receipts are saved; no fabricated native maps or generated target maps.
        payload = checkpoint_payload(controller, family, protected, ledger)
        checkpoint = output / "checkpoint.pt"
        torch.save(payload, checkpoint)
        ledger.add("checkpoint_writes")
        restored_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        restored, restored_family, restored_protected = restore_checkpoint(restored_payload)
        ledger.add("checkpoint_reloads")
        restored_loss = TerminalObjective(restored, restored_family, executor, heldout)()
        results["checkpoint_reload"] = {
            "path": str(checkpoint), "parameter_hash_equal": restored.parameter_hash() == controller.parameter_hash(),
            "heldout_loss": restored_loss, "admitted_operations": sorted(restored_protected.admitted),
            "bytes": checkpoint.stat().st_size}
        results["status"] = "completed_finite_diagnostic"
    except (BudgetStop, NumericalFailure) as error:
        results["status"] = "censored" if isinstance(error, BudgetStop) else "numerical_unresolved"
        results["reason"] = str(error)
        # Current neural state is still checkpointed at a safe transaction
        # boundary when possible. No missing outcome is converted to a success.
        torch.save(checkpoint_payload(controller, family, protected, ledger),
                   output / "partial-checkpoint.pt")
    finally:
        results["costs"] = ledger.snapshot()
        write({"kind": "arm_end", **results})
        stream.close()
        (output / "result.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--max-feedback", type=int, default=1600)
    parser.add_argument("--max-updates", type=int, default=24)
    parser.add_argument("--gradient-draws", type=int, default=3)
    parser.add_argument("--score-samples", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", nargs="+", default=["event", "ordinary_event", "score", "discrete"],
                        choices=["event", "ordinary_event", "score", "discrete"])
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a NEW directory")
    if (not 0 < args.seconds <= 110 or not 1 <= args.max_updates <= 32
            or not 1 <= args.gradient_draws <= 16 or not 1 <= args.score_samples <= 16
            or not 20 <= args.max_feedback <= 4000 or not 1.1 <= args.sigma <= 2.0
            or not 0 < args.learning_rate <= 2.0):
        parser.error("require bounded time<=110, updates<=32, draws/samples<=16, feedback<=4000, 1.1<=sigma<=2")
    if len(set(args.arms)) != len(args.arms):
        parser.error("duplicate arms are not meaningful")
    args.out.mkdir(parents=True, exist_ok=False)
    configuration = vars(args).copy()
    configuration["out"] = str(args.out)
    configuration.update({"main": "TOPOLOGY_USER_LOCKED", "execution": "ROOT-invoked bounded local diagnostic",
                          "effective_mode": "unknown", "new_originality_claim": False,
                          "outside_timeout_required": True,
                          "fairness": "same per-arm query/update caps; shared wall deadline can censor later arms"})
    (args.out / "protocol.json").write_text(json.dumps(configuration, indent=2) + "\n")
    deadline = time.monotonic() + args.seconds
    limits = Limits(max_feedback=args.max_feedback, max_updates=args.max_updates)
    results = []
    for arm in args.arms:
        if time.monotonic() >= deadline:
            results.append({"arm": arm, "status": "not_started_shared_deadline"})
            continue
        results.append(run_arm(arm, args.out / arm, limits, deadline, args.seed,
                               args.sigma, args.learning_rate, args.gradient_draws, args.score_samples))
    indexed = {result["arm"]: result for result in results}
    correspondence = {"available": False}
    if all(indexed.get(arm, {}).get("status") == "completed_finite_diagnostic"
           for arm in ("event", "ordinary_event")):
        event, ordinary = indexed["event"], indexed["ordinary_event"]
        correspondence = {"available": True,
            "final_parameter_hash_equal": event["final_parameter_hash"] == ordinary["final_parameter_hash"],
            "curve_parameter_hashes_equal": [r["parameter_hash"] for r in event["trajectory"]]
                == [r["parameter_hash"] for r in ordinary["trajectory"]],
            "meaning": "same-event ordinary implementation correspondence, no topology-exclusive win"}
    final = {"results": results, "ordinary_correspondence": correspondence,
             "scope": "small synthetic acquisition/retention diagnostic, not M0, scaling, novelty, or superiority"}
    (args.out / "result.json").write_text(json.dumps(final, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "status": [(r["arm"], r["status"]) for r in results],
                      "ordinary_correspondence": correspondence}, indent=2))


if __name__ == "__main__":
    main()
