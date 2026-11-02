"""Unexecuted, frame-frozen local irreversible-interface development variant.

Same Prometheus V2; MAIN=TOPOLOGY USER_LOCKED, human-only change. This NEW file
does not modify the core, Method, paper, old results, or execution permissions.
No import-time experiment, subprocess, SSH, networking, or installation occurs.

Intended increment: restore the actual acquired D3 hybrid core, freeze its
physical frame, and learn a new operation on core x fresh ternary ancilla.
Its nine-state reversible lift is obtained from ACTUAL root continuation of
z**9-9*z-u. No SWAP, reset permutation, garbage code, or target loop is provided
to the neural learner. A program of raw scalar-controlled triangle loops is
learned from logical terminal labels; new call logits are initially uniform.
The forward soft interface uses the derived nonbijective logical channel;
native execution retains typed garbage and supports charged inverse cleanup.

This explicitly does NOT test shared continual-backbone adaptation: the restored
core/E/R/calls are frozen while a history frame is live. Only reset_loop_head
and reset_call_head remain learnable. All old allocated parameters still count.
It is a software typed-port interface between the D3 and D9 local executors,
not an unimplemented physical teleportation/state-preparation claim.

The finite polynomial, marks, triangle grammar and fresh-cell source are supplied
priors. The ordinary channel control learns its input specification from the
same singleton terminal records; the canonical reversible compiler consumes
that LEARNED channel, never a teacher table. Source-aware loop synthesis is
separately charged when actual D9 actuation is requested. This is not a new
reversible-computing or polynomial-compiler novelty claim.

Root entry APIs (no run is started here):
  limits = proposed_limits()
  ledger = CostLedger(limits, externally_bounded_monotonic_deadline, writer=...)
  session = ResetInterfaceSession.from_core_checkpoint(path, ledger)
  training, heldout, manifest = session.diagnostic_specification()
  session.step(training, random.Random(0))  # finite caller-owned update loop
  session.evaluate(training); session.admit_if_successful(training)
  # Freeze actual weights BEFORE heldout; save/restore and cold audits explicitly.
The returned manifest is a prospective diagnostic, not an execution receipt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import cmath
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from loop_boundary_learning import (
    AffineLoopFamily, BudgetStop, CostLedger, Limits, NumericalFailure,
    ProgramExample, RaisedCosineKernel, loop_clearance, slice_events,
    vertices_in_execution_box,
)
from hybrid_loop_block import (
    HybridLearningSession, SOFT_NATIVE_MIN_MASS, native_output,
    native_relation_preserved,
)


def proposed_limits() -> Limits:
    """Prospective finite ROOT-RUN bounds; calling this grants no permission."""
    return Limits(max_feedback=12000, max_programs=4000,
                  max_continuation_steps=160000, max_updates=100,
                  max_word_length=16, max_vertices=16, max_control_coordinate=16.)


def permutation_hash(image: Sequence[int]) -> str:
    if sorted(image) != list(range(len(image))):
        raise NumericalFailure("expected an actually obtained permutation")
    return hashlib.sha256(bytes(image)).hexdigest()


def inverse_permutation(image: Sequence[int], ledger: CostLedger) -> list[int]:
    permutation_hash(image)
    inverse = [0] * len(image)
    for source, target in enumerate(image):
        inverse[target] = source
        ledger.add("local_inverse_entry_assignments")
    return inverse


class PolynomialPairExecutor:
    """Real predictor/corrector continuation, no supplied endpoint-action table.

    Base mark 0 is root 0; marks 1..8 follow the circular roots in angular order.
    Pair ports encode (core,ancilla) as 3*core+ancilla. Full nine-root observations
    are charged and exposed equally to controls. Checks are numerical diagnostics,
    not interval certification. An explicit external deadline is still required.
    """

    degree = 9

    def __init__(self, ledger: CostLedger):
        self.ledger = ledger
        radius = 9. ** (1. / 8.)
        self.base_roots = (0j, *(radius * cmath.exp(2j * math.pi * j / 8) for j in range(8)))
        self.critical = tuple(-8 * cmath.exp(2j * math.pi * j / 8) for j in range(8))

    def _correct(self, prediction: complex, control: complex) -> tuple[complex, float]:
        root = prediction
        for _ in range(self.ledger.limits.max_newton):
            self.ledger.add("newton_iterations")
            residual = root ** 9 - 9 * root - control
            self.ledger.add("polynomial_evaluations")
            if abs(residual) <= self.ledger.limits.residual_tolerance:
                return root, abs(residual)
            derivative = 9 * (root ** 8 - 1)
            self.ledger.add("derivative_evaluations")
            if abs(derivative) < 1e-12:
                raise NumericalFailure("small D9 correction derivative")
            root -= residual / derivative
            if not math.isfinite(root.real) or not math.isfinite(root.imag):
                raise NumericalFailure("nonfinite D9 correction")
        raise NumericalFailure("D9 Newton iteration limit")

    def execute(self, loops: Sequence[Sequence[complex]]) -> dict[str, Any]:
        self.ledger.add("program_calls")
        if not 1 <= len(loops) <= 32:
            raise ValueError("local D9 source permits at most32 based generator loops per cold call")
        started = time.perf_counter()
        roots = list(self.base_roots)
        minimum_separation = min(abs(a - b) for i, a in enumerate(roots) for b in roots[i + 1:])
        minimum_clearance, maximum_residual, steps = math.inf, 0., 0
        try:
            for vertices in loops:
                if not vertices_in_execution_box(vertices, self.ledger.limits):
                    raise NumericalFailure("D9 loop outside declared numeric control box")
                clearance = loop_clearance(vertices, self.critical)
                self.ledger.add("pair_execution_clearance_checks", (len(vertices) + 1) * 8)
                if clearance < self.ledger.limits.min_control_clearance:
                    raise NumericalFailure("D9 loop below numeric clearance")
                minimum_clearance = min(minimum_clearance, clearance)
                points = (0j, *vertices, 0j)
                for left, right in zip(points, points[1:]):
                    segment = right - left
                    if abs(segment) == 0:
                        continue
                    fraction = 0.
                    nominal = min(1., self.ledger.limits.max_control_step
                                  * min(1., math.sqrt(clearance)) / abs(segment))
                    while fraction < 1.:
                        step, accepted = min(nominal, 1. - fraction), False
                        for _ in range(25):
                            self.ledger.add("step_attempts")
                            delta = step * segment
                            control = left + (fraction + step) * segment
                            separation = min(abs(a - b) for i, a in enumerate(roots)
                                             for b in roots[i + 1:])
                            try:
                                predictions = []
                                for root in roots:
                                    derivative = 9 * (root ** 8 - 1)
                                    self.ledger.add("derivative_evaluations")
                                    if abs(derivative) < 1e-12:
                                        raise NumericalFailure("small D9 continuation derivative")
                                    predictions.append(root + delta / derivative)
                                corrected = [self._correct(prediction, control) for prediction in predictions]
                                next_roots = [value[0] for value in corrected]
                                residual = max(value[1] for value in corrected)
                                movement = max(abs(a - b) for a, b in zip(roots, next_roots))
                                correction = max(abs(a - b) for a, b in zip(predictions, next_roots))
                                next_separation = min(abs(a - b) for i, a in enumerate(next_roots)
                                                      for b in next_roots[i + 1:])
                                if (movement > .18 * separation or correction > .06 * separation
                                        or next_separation < 1e-6):
                                    raise NumericalFailure("D9 locality/separation check")
                            except NumericalFailure:
                                self.ledger.add("step_halvings")
                                step *= .5
                                if step < 1e-12:
                                    break
                                continue
                            roots = next_roots
                            fraction = min(1., fraction + step)
                            minimum_separation = min(minimum_separation, next_separation)
                            maximum_residual = max(maximum_residual, residual)
                            self.ledger.add("continuation_steps")
                            self.ledger.add("root_strand_steps", 9)
                            steps += 1
                            accepted = True
                            break
                        if not accepted:
                            raise NumericalFailure("D9 step refinement exhausted")
            image = []
            for root in roots:
                distances = [abs(root - base) for base in self.base_roots]
                self.ledger.add("endpoint_distance_evaluations", 9)
                target = min(range(9), key=distances.__getitem__)
                if distances[target] > self.ledger.limits.endpoint_tolerance:
                    raise NumericalFailure("D9 endpoint outside marked resting fiber")
                image.append(target)
            permutation_hash(image)
            return {"endpoint_map": image, "degree": 9, "continuation_steps": steps,
                    "minimum_root_separation": minimum_separation,
                    "minimum_control_clearance": minimum_clearance,
                    "maximum_polynomial_residual": maximum_residual,
                    "observation": "all9 continued endpoints, available to every control",
                    "certificate_scope": "numerical checks, not interval proof"}
        finally:
            self.ledger.elapsed("pair_continuation", started)


class LocalLoopProgramBank:
    """A bounded loop grammar, not a supplied reset or SWAP operation.

    Three passes through8 radial triangle families give24 scalar gates. Each
    triangle can be inside/outside its actual discriminant event. Full maps are
    observed by continuation at paid safe representatives and composed on one
    marked nine-state fiber. Source-aware ordinary program synthesis has the same
    observations. The grammar is an explicit, possibly inefficient inductive bias.
    """

    def __init__(self, ledger: CostLedger, passes: int = 3):
        if not 1 <= passes <= 3:
            raise ValueError("bounded1..3 passes required")
        self.ledger, self.executor = ledger, PolynomialPairExecutor(ledger)
        self.gates = 8 * int(passes)
        self.families = []
        for critical in self.executor.critical:
            direction = critical / abs(critical)
            self.families.append(AffineLoopFamily((1j * direction, -1j * direction),
                                                  ((direction, direction),), (4., 12.)))
        self.events_cache: dict[int, list[float]] = {}
        self.maps_cache: dict[tuple[int, int], dict[str, Any]] = {}
        self.namespace = hashlib.sha256(json.dumps({"cover": "z9-9z-u", "base": 0,
            "marks": [[z.real, z.imag] for z in self.executor.base_roots],
            "families": [family.configuration() for family in self.families]}, sort_keys=True).encode()).hexdigest()

    def events(self, family: int) -> list[float]:
        family %= 8
        if family not in self.events_cache:
            self.events_cache[family] = slice_events(self.families[family], [7.], 0, 4., 12.,
                                                      self.executor.critical, self.ledger)
            self.ledger.add("pair_event_family_constructions")
        else:
            self.ledger.add("pair_event_family_cache_hits")
        return self.events_cache[family]

    def mean_domain(self, actions: Sequence[float], sigma: float = 2.) -> None:
        if len(actions) != self.gates:
            raise ValueError("pair program coordinate count mismatch")
        for slot, action in enumerate(actions):
            family = self.families[slot % 8]
            family.require_support([action], sigma)
            vertices = family.vertices([action])
            self.ledger.add("pair_mean_domain_segment_checks", 3 * 8)
            if (not vertices_in_execution_box(vertices, self.ledger.limits)
                    or loop_clearance(vertices, self.executor.critical)
                       < self.ledger.limits.min_control_clearance):
                raise NumericalFailure("pair mean program lacks finite executable clearance")

    def generator_map(self, family: int, action: float) -> list[int]:
        family %= 8
        if not 4. <= action <= 12. or not math.isfinite(action):
            raise NumericalFailure("pair program sample outside untruncated action box")
        events = self.events(family)
        if any(abs(action - event) < 1e-12 for event in events):
            raise NumericalFailure("pair sample on unresolved discriminant")
        chamber = sum(action > event for event in events)
        key = family, chamber
        self.ledger.add("pair_chamber_membership_checks", len(events))
        if key not in self.maps_cache:
            boundaries = [4., *events, 12.]
            representative = .5 * (boundaries[chamber] + boundaries[chamber + 1])
            vertices = self.families[family].vertices([representative])
            receipt = self.executor.execute([vertices])
            self.maps_cache[key] = {"representative": representative, "receipt": receipt}
            self.ledger.add("pair_cold_generator_lifts")
            self.ledger.emit("pair_generator_observation", family=family, chamber=chamber,
                             representative=representative, receipt=receipt,
                             common_information=True)
        else:
            self.ledger.add("pair_certified_class_cache_hits")
        # The request is an ideal semantic sample. Its actual near-boundary
        # trajectory is not claimed integrated; its chamber representative was.
        return list(self.maps_cache[key]["receipt"]["endpoint_map"])

    def program_map(self, actions: Sequence[float], *, mean: bool = False) -> list[int]:
        if len(actions) != self.gates:
            raise ValueError("pair program length mismatch")
        if mean:
            self.mean_domain(actions)
        image = list(range(9))
        for slot, action in enumerate(actions):
            gate = self.generator_map(slot % 8, float(action))
            image = [gate[state] for state in image]
            self.ledger.add("pair_program_image_applications", 9)
        permutation_hash(image)
        return image

    def cold_program(self, actions: Sequence[float], *, inverse: bool = False) -> dict[str, Any]:
        self.mean_domain(actions)
        loops = [self.families[slot % 8].vertices([action]) for slot, action in enumerate(actions)]
        if inverse:
            loops = [tuple(reversed(vertices)) for vertices in reversed(loops)]
        return self.executor.execute(loops)

    def storage(self) -> dict[str, int]:
        serializable = [{"key": list(key), **value} for key, value in self.maps_cache.items()]
        return {"observed_generator_maps": len(self.maps_cache),
                "map_int64_payload_bytes": 8 * 9 * len(self.maps_cache),
                "serialized_receipt_bytes": len(json.dumps(serializable).encode()),
                "event_metadata_bytes": len(json.dumps(self.events_cache).encode())}


@dataclass(frozen=True)
class HistoryEntry:
    kind: str
    operation: int
    semantic_hash: str
    source_namespace: str
    cell: int | None = None


class TypedHistoryArena:
    """Factorized native data/history; no enumeration of3**capacity states."""

    def __init__(self, initial_core: int, frame_hash: str, ledger: CostLedger,
                 capacity: int = 4, max_records: int = 64):
        if not 0 <= initial_core < 3 or not 1 <= capacity <= 8 or not 1 <= max_records <= 128:
            raise ValueError("bounded ternary history arena required")
        self.initial_core = self.core = int(initial_core)
        self.frame_hash, self.ledger = frame_hash, ledger
        self.cells: list[int | None] = [None] * capacity
        self.entries: list[HistoryEntry] = []
        self.max_records = int(max_records)
        self.output_copy: int | None = None
        self.peak_records = self.peak_dirty = 0
        ledger.add("history_preallocated_cells", capacity)
        ledger.add("history_frame_registrations")

    def _record(self, entry: HistoryEntry) -> None:
        if len(self.entries) >= self.max_records:
            raise BudgetStop("typed native gate log exhausted")
        self.ledger.add("typed_history_records")
        self.entries.append(entry)
        self.peak_records = max(self.peak_records, len(self.entries))
        self.peak_dirty = max(self.peak_dirty, sum(value is not None for value in self.cells))

    def core_gate(self, operation: int, image: Sequence[int], namespace: str) -> None:
        if len(image) != 3:
            raise ValueError("core port requires a three-state permutation")
        digest = permutation_hash(image)
        if len(self.entries) >= self.max_records:
            raise BudgetStop("typed native gate log exhausted before operation")
        next_core = image[self.core]
        self._record(HistoryEntry("core", operation, digest, namespace))
        self.core = next_core
        self.ledger.counts["native_core_map_applications"] = self.ledger.counts.get("native_core_map_applications", 0) + 1

    def reset_gate(self, image: Sequence[int], namespace: str) -> None:
        if len(image) != 9:
            raise ValueError("pair port requires a nine-state permutation")
        digest = permutation_hash(image)
        if len(self.entries) >= self.max_records:
            raise BudgetStop("typed native gate log exhausted before allocation")
        try:
            cell = self.cells.index(None)
        except ValueError as error:
            raise BudgetStop("fresh ternary ancilla capacity exhausted") from error
        self.ledger.add("pair_port_encodings")
        result = image[3 * self.core]  # explicitly fresh ancilla=0
        next_core, garbage = divmod(result, 3)
        self.ledger.add("pair_port_decodings")
        self._record(HistoryEntry("pair", 0, digest, namespace, cell))
        # No budget check between writing a native state and its history entry.
        self.core, self.cells[cell] = next_core, garbage
        self.peak_dirty = max(self.peak_dirty, sum(value is not None for value in self.cells))
        for key in ("fresh_ancilla_preparations", "native_pair_map_applications", "garbage_cell_writes"):
            self.ledger.counts[key] = self.ledger.counts.get(key, 0) + 1

    def copy_result(self) -> int:
        if self.output_copy is not None:
            raise NumericalFailure("output copy register is not a fresh blank")
        # Supplied classical reversible add on(core, blank), charged. This is
        # not cloning an unknown quantum state and not a learned reset oracle.
        self.ledger.check()
        self.output_copy = (0 + self.core) % 3
        for key in ("fresh_output_register_preparations", "reversible_classical_copy_calls"):
            self.ledger.counts[key] = self.ledger.counts.get(key, 0) + 1
        return self.output_copy

    def cleanup(self, core_maps: Mapping[int, Sequence[int]], pair_map: Sequence[int],
                core_namespace: str, pair_namespace: str, current_frame_hash: str) -> dict[str, Any]:
        if current_frame_hash != self.frame_hash:
            raise NumericalFailure("live history frame changed")
        if self.output_copy is None:
            raise NumericalFailure("copy result before undoing the computation")
        before = self.storage()
        while self.entries:
            entry = self.entries[-1]
            image = core_maps[entry.operation] if entry.kind == "core" else pair_map
            namespace = core_namespace if entry.kind == "core" else pair_namespace
            if namespace != entry.source_namespace or permutation_hash(image) != entry.semantic_hash:
                raise NumericalFailure("current lift changed old garbage semantics; inverse cleanup refused")
            inverse = inverse_permutation(image, self.ledger)
            if entry.kind == "core":
                next_core = inverse[self.core]
                self.ledger.check()
                self.core = next_core
                self.entries.pop()
                self.ledger.counts["cleanup_core_inverse_calls"] = self.ledger.counts.get("cleanup_core_inverse_calls", 0) + 1
            else:
                assert entry.cell is not None
                garbage = self.cells[entry.cell]
                if garbage is None:
                    raise NumericalFailure("history cell was reused or lost")
                self.ledger.add("pair_port_encodings")
                next_core, blank = divmod(inverse[3 * self.core + garbage], 3)
                self.ledger.add("pair_port_decodings")
                if blank != 0:
                    raise NumericalFailure("inverse did not restore prescribed blank")
                # Complete one history entry before the next budget check, so
                # an interrupted cleanup retains an accurate remaining log.
                self.core = next_core
                self.cells[entry.cell] = None
                self.entries.pop()
                self.ledger.counts["cleanup_pair_inverse_calls"] = self.ledger.counts.get("cleanup_pair_inverse_calls", 0) + 1
                self.ledger.counts["garbage_cells_reclaimed"] = self.ledger.counts.get("garbage_cells_reclaimed", 0) + 1
        restored = self.core == self.initial_core and all(value is None for value in self.cells)
        if not restored:
            raise NumericalFailure("cleanup did not recover input and blank arena")
        return {"input_restored": True, "blank_cells_restored": True,
                "copied_physical_output": self.output_copy, "before_cleanup_storage": before,
                "retained_input_and_output_registers": 2,
                "execution": "digital inverses of paid observed maps; not fresh cold trajectories"}

    def storage(self) -> dict[str, Any]:
        return {"ternary_core_registers": 1, "preallocated_ternary_cells": len(self.cells),
                "dirty_cells": sum(value is not None for value in self.cells),
                "peak_dirty_cells": self.peak_dirty, "gate_records": len(self.entries),
                "peak_gate_records": self.peak_records,
                "fixed_data_payload_bits": 2 * (1 + len(self.cells) + int(self.output_copy is not None)),
                "serialized_live_history_bytes": len(json.dumps([asdict(entry) for entry in self.entries]).encode()),
                "frame_and_address_metadata_bytes": len(json.dumps({"frame": self.frame_hash,
                    "capacity": len(self.cells), "max_records": self.max_records}).encode()),
                "whole_history_configuration_table_allocated": False}

    def checkpoint(self) -> dict[str, Any]:
        return {"initial_core": self.initial_core, "current_core": self.core,
                "frame_hash": self.frame_hash, "cells": list(self.cells),
                "entries": [asdict(entry) for entry in self.entries], "max_records": self.max_records,
                "output_copy": self.output_copy, "peak_records": self.peak_records,
                "peak_dirty": self.peak_dirty, "storage": self.storage()}

    @classmethod
    def restore(cls, payload: Mapping[str, Any], ledger: CostLedger) -> "TypedHistoryArena":
        cells = list(payload["cells"])
        if any(value is not None and value not in (0, 1, 2) for value in cells):
            raise NumericalFailure("invalid encoded ternary history cell")
        arena = cls(int(payload["initial_core"]), payload["frame_hash"], ledger,
                    len(cells), int(payload["max_records"]))
        arena.core = int(payload["current_core"])
        if not 0 <= arena.core < 3:
            raise NumericalFailure("invalid restored physical core")
        arena.cells = cells
        arena.entries = [HistoryEntry(**entry) for entry in payload["entries"]]
        if len(arena.entries) > arena.max_records:
            raise NumericalFailure("restored gate log exceeds declared capacity")
        arena.output_copy = payload["output_copy"]
        arena.peak_records, arena.peak_dirty = int(payload["peak_records"]), int(payload["peak_dirty"])
        ledger.add("typed_history_restore_bytes", len(json.dumps(payload).encode()))
        return arena


class FactorizedResetBlock(nn.Module):
    """Restored learned core plus two NEW learnable heads; frame is frozen."""

    def __init__(self, core_payload: Mapping[str, Any], ledger: CostLedger, passes: int = 3):
        super().__init__()
        self.core_session = HybridLearningSession.restore(core_payload, ledger)
        self.core = self.core_session.model
        if (self.core.slots != 1 or not {0, 1}.issubset(self.core_session.protected.admitted)
                or not {0, 1}.issubset(self.core_session.old_tokens)):
            raise ValueError("this increment requires the acquired one-slot two-operation V2 core")
        for parameter in self.core.parameters():
            parameter.requires_grad_(False)
        self.ledger = ledger
        self.bank = LocalLoopProgramBank(ledger, passes)
        self.register_buffer("reset_feature", torch.tensor([1., .5], dtype=torch.float64))
        self.register_buffer("program_offset", torch.full((self.bank.gates,), 7., dtype=torch.float64))
        self.reset_loop_head = nn.Linear(2, self.bank.gates, bias=False).double()
        self.reset_call_head = nn.Linear(2, 4, bias=False).double()
        with torch.no_grad():
            self.reset_loop_head.weight.zero_()
            self.reset_call_head.weight.zero_()
        self.frozen_core_hash = self.core.parameter_hash()
        self.source_core_payload = copy.deepcopy(core_payload)
        self.admitted_pair: list[int] | None = None
        self.admitted_logical: list[int] | None = None
        self.history_capacity = 4
        self.kernel = RaisedCosineKernel(2.)
        initial_maps = self.core_maps()
        self.reference = self.core.native_signature(initial_maps, ledger)
        encoder, readout = self.reference["encoder"], self.reference["readout"]
        if sorted(encoder) != list(range(3)) or any(readout[encoder[x]] != x for x in range(3)):
            raise ValueError("requires a bijective old physical register frame")
        token_images = []
        for token in (0, 1):
            image = []
            for state in range(3):
                current = state
                for route in self.reference["routes"][token]:
                    if route:
                        current = initial_maps[route - 1][current]
                    ledger.add("old_transitive_frame_checks")
                image.append(current)
            token_images.append(image)
        reached = {0}
        for _ in range(3):
            reached |= {image[state] for image in token_images for state in tuple(reached)}
        if len(reached) != 3:
            raise ValueError("this diagnostic specifies a transitive acquired old algebra")

    def assert_frame(self) -> None:
        self.ledger.add("frozen_core_frame_hash_checks")
        if self.core.parameter_hash() != self.frozen_core_hash:
            raise NumericalFailure("frame-frozen variant cannot update the restored core")

    def trainable_named(self) -> list[tuple[str, nn.Parameter]]:
        return [(name, parameter) for name, parameter in self.named_parameters() if parameter.requires_grad]

    def actions_tensor(self) -> torch.Tensor:
        self.ledger.add("reset_loop_head_forwards")
        return self.program_offset + self.reset_loop_head(self.core.shared(self.reset_feature))

    def actions(self) -> list[float]:
        return self.actions_tensor().detach().cpu().tolist()

    def reset_calls(self) -> torch.Tensor:
        self.ledger.add("reset_call_head_forwards")
        return torch.softmax(self.reset_call_head(self.core.shared(self.reset_feature)), -1)

    def core_maps(self) -> dict[int, list[int]]:
        self.assert_frame()
        return self.core_session.bank.maps()

    @staticmethod
    def logical_channel(pair_map: Sequence[int]) -> list[int]:
        permutation_hash(pair_map)
        return [pair_map[3 * q] // 3 for q in range(3)]

    def soft_probability(self, example: ProgramExample, pair_map: Sequence[int]) -> torch.Tensor:
        self.assert_frame()
        if not 0 <= example.initial_sheet < 3 or any(token not in (0, 1, 2) for token in example.word):
            raise ValueError("example outside the declared typed core alphabet")
        if len(example.word) > self.ledger.limits.max_word_length or example.word.count(2) > self.history_capacity:
            raise BudgetStop("logical word/history capacity outside fixed diagnostic")
        alpha = torch.softmax(self.core.encoder_logits[example.initial_sheet], -1)
        readout = torch.softmax(self.core.readout_logits, -1)
        core_maps = self.core_maps()
        images = [list(range(3)), core_maps[0], core_maps[1], self.logical_channel(pair_map)]
        matrices = torch.stack([torch.nn.functional.one_hot(torch.tensor(image), 3).to(alpha)
                                for image in images]).detach()
        for token in example.word:
            if token == 2:
                alpha = alpha @ torch.einsum("k,kij->ij", self.reset_calls(), matrices)
                self.ledger.add("soft_fresh_cell_channel_transitions")
            elif token in (0, 1):
                probabilities = self.core.call_probabilities(token)[0]
                alpha = alpha @ torch.einsum("k,kij->ij", probabilities, matrices[:3])
                self.ledger.add("soft_old_core_transitions")
            else:
                raise ValueError("unknown typed diagnostic token")
        return alpha @ readout

    def terminal_loss(self, examples: Sequence[ProgramExample], pair_map: Sequence[int]) -> torch.Tensor:
        if not examples:
            raise ValueError("nonempty terminal batch required")
        losses = []
        for example in examples:
            self.ledger.add("feedback_calls")
            self.ledger.add("reset_terminal_label_accesses")
            probability = self.soft_probability(example, pair_map)
            loss = -torch.log(probability[example.target].clamp_min(1e-12))
            losses.append(loss)
            self.ledger.emit("reset_terminal_feedback", example=asdict(example),
                             probability=probability.detach().cpu().tolist(), loss=float(loss.detach()),
                             desired_joint_lift_or_garbage_code_supplied=False)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def native(self, example: ProgramExample, pair_map: Sequence[int], cleanup: bool = True) -> dict[str, Any]:
        self.assert_frame()
        if not 0 <= example.initial_sheet < 3 or any(token not in (0, 1, 2) for token in example.word):
            raise ValueError("example outside the declared typed core alphabet")
        if (len(example.word) > self.ledger.limits.max_word_length
                or example.word.count(2) > self.history_capacity):
            raise BudgetStop("native logical word/history capacity outside fixed diagnostic")
        maps = self.core_maps()
        signature = self.core.native_signature(maps, self.ledger)
        arena = TypedHistoryArena(signature["encoder"][example.initial_sheet],
                                   self.frozen_core_hash, self.ledger, self.history_capacity)
        for token in example.word:
            choice = int(self.reset_calls().argmax()) if token == 2 else signature["routes"][token][0]
            self.ledger.add("native_call_choices")
            if choice == 3:
                arena.reset_gate(pair_map, self.bank.namespace)
            elif choice:
                arena.core_gate(choice - 1, maps[choice - 1], self.frozen_core_hash)
        physical_output = arena.copy_result()
        output = signature["readout"][physical_output]
        storage = arena.storage()
        cleanup_receipt = arena.cleanup(maps, pair_map, self.frozen_core_hash,
                                         self.bank.namespace, self.core.parameter_hash()) if cleanup else None
        self.ledger.emit("typed_native_execution", example=asdict(example), output=output,
                         storage=storage, cleanup=cleanup_receipt)
        return {"output": output, "storage": storage, "cleanup": cleanup_receipt, "arena": arena}

    def gradient(self, examples: Sequence[ProgramExample], rng: random.Random) -> dict[str, Any]:
        """Coupled credit, with ONE uniformly sampled event coordinate (1/q).

        This bounds counterfactual work without pretending to compute the full
        event sum deterministically. Transverse noise and coordinate selection
        are logged; no invalid-sample rejection or density renormalization occurs.
        """
        self.assert_frame()
        mean_tensor, named = self.actions_tensor(), self.trainable_named()
        parameters = [parameter for _, parameter in named]
        mean = mean_tensor.detach().cpu().tolist()
        self.bank.mean_domain(mean)
        noise = [self.kernel.sample(rng, self.ledger) for _ in mean]
        sample = [a + delta for a, delta in zip(mean, noise)]
        within_loss = self.terminal_loss(examples, self.bank.program_map(sample))
        within_raw = torch.autograd.grad(within_loss, parameters, retain_graph=True, allow_unused=True)
        within = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, within_raw)]
        self.ledger.add("reset_within_vjps")
        coordinate = rng.randrange(len(mean))
        self.ledger.add("random_event_coordinate_draws")
        transverse = [a + self.kernel.sample(rng, self.ledger) if j != coordinate else a
                      for j, a in enumerate(mean)]
        left, right = mean[coordinate] - 2., mean[coordinate] + 2.
        events = [event for event in self.bank.events(coordinate % 8) if left < event < right]
        action_gradient = [0.] * len(mean)
        losses, representatives = [], []
        if events:
            for lo, hi in zip([left, *events], [*events, right]):
                transverse[coordinate] = .5 * (lo + hi)
                with torch.no_grad():
                    loss = float(self.terminal_loss(examples, self.bank.program_map(transverse)))
                losses.append(loss)
                representatives.append(transverse[coordinate])
            jumps = [b - a for a, b in zip(losses, losses[1:])]
            action_gradient[coordinate] = len(mean) * sum(
                jump * self.kernel.density(event - mean[coordinate]) for jump, event in zip(jumps, events))
        surrogate = torch.dot(mean_tensor, mean_tensor.new_tensor(action_gradient).detach())
        boundary_raw = torch.autograd.grad(surrogate, parameters, allow_unused=True)
        boundary = [torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, boundary_raw)]
        self.ledger.add("reset_boundary_vjps")
        total = [a + b for a, b in zip(within, boundary)]
        if any(not bool(torch.isfinite(g).all()) for g in total):
            raise NumericalFailure("nonfinite reset acquisition gradient")
        record = {"coordinate": coordinate, "coordinate_probability": 1. / len(mean),
                  "inverse_probability_weight": len(mean), "mean": mean, "within_noise": noise,
                  "transverse_actions": transverse, "events": events, "representatives": representatives,
                  "event_losses": losses, "action_gradient": action_gradient,
                  "within_parameter_norms": {name: float(g.norm()) for (name, _), g in zip(named, within)},
                  "class_parameter_norms": {name: float(g.norm()) for (name, _), g in zip(named, boundary)}}
        self.ledger.emit("reset_coupled_gradient", **record)
        return {"total": total, "within": within, "class": boundary, "record": record}

    def transaction(self, gradients: Sequence[torch.Tensor], learning_rate: float = .05) -> dict[str, Any]:
        if not 0 < learning_rate <= .05:
            raise ValueError("bounded reset SGD rate required")
        named = self.trainable_named()
        parameters = [parameter for _, parameter in named]
        if len(parameters) != len(gradients):
            raise ValueError("reset gradient/parameter count mismatch")
        originals = [parameter.detach().clone() for parameter in parameters]
        plans = [("joint", gradients, scale) for scale in (1., .5, .25, .125, .0625, .03125, .015625)]
        call_only = [g if name.startswith("reset_call_head") else torch.zeros_like(g)
                     for (name, _), g in zip(named, gradients)]
        if any(bool(g.abs().max() > 0) for g in call_only):
            plans += [("call_fallback", call_only, scale) for scale in (1., .5, .25, .125, .0625, .03125, .015625)]
        plans.append(("zero", gradients, 0.))
        committed, trials = False, []
        try:
            for block, direction, scale in plans:
                self.ledger.add("reset_parameter_proposals")
                with torch.no_grad():
                    for parameter, original, gradient in zip(parameters, originals, direction):
                        parameter.copy_(original - learning_rate * scale * gradient)
                reason = None
                try:
                    self.assert_frame()
                    if any(not bool(torch.isfinite(p).all()) or float(p.abs().max()) > 16 for p in parameters):
                        raise NumericalFailure("reset parameter box exceeded")
                    mean = self.actions()
                    self.bank.mean_domain(mean)
                    candidate = self.core.native_signature(self.core_maps(), self.ledger)
                    if not native_relation_preserved(self.reference, candidate, (0, 1), self.ledger):
                        raise NumericalFailure("old E/R/call relation changed")
                    if self.core.soft_native_path_bound((0, 1), 16, self.ledger) <= SOFT_NATIVE_MIN_MASS:
                        raise NumericalFailure("old soft/native certificate failed")
                    if self.admitted_pair is not None:
                        pair_map = self.bank.program_map(mean, mean=True)
                        if pair_map != self.admitted_pair:
                            raise NumericalFailure("admitted full reversible lift changed")
                        channels = [list(range(3)), self.core_maps()[0], self.core_maps()[1], self.logical_channel(pair_map)]
                        if channels[int(self.reset_calls().argmax())] != self.admitted_logical:
                            raise NumericalFailure("admitted new calling semantics changed")
                        if self.soft_path_mass() <= SOFT_NATIVE_MIN_MASS:
                            raise NumericalFailure("admitted reset soft/native confidence changed")
                except NumericalFailure as error:
                    reason = str(error)
                trials.append({"block": block, "scale": scale, "reason": reason})
                if reason:
                    self.ledger.add("reset_proposal_rejections")
                    continue
                self.ledger.add("updates")
                result = {"accepted_block": block, "accepted_scale": scale, "trials": trials,
                          "core_frozen": True, "full_lift_guard": self.admitted_pair is not None}
                self.ledger.emit("reset_committed_transaction", **result)
                committed = True
                return result
            raise NumericalFailure("restored reset state did not pass its contract")
        finally:
            if not committed:
                with torch.no_grad():
                    for parameter, original in zip(parameters, originals):
                        parameter.copy_(original)

    @torch.no_grad()
    def soft_path_mass(self, max_length: int = 16) -> float:
        """Own-native probability for admissible mixed words, not singleton only.

        Scope: length<=max_length and reset-token count<=history_capacity. Every
        soft branch then has enough fresh cells. Pair channels are deterministic
        row-one-hot (not necessarily bijective), and no forward gate reads old
        garbage; the selected native path contributes this mass to its output.
        """
        encoder = torch.softmax(self.core.encoder_logits, -1)
        readout = torch.softmax(self.core.readout_logits, -1)
        own_encoder = float(encoder.max(-1).values.min())
        own_readout = float(readout.max(-1).values.min())
        own_call = min(float(self.core.call_probabilities(token).max()) for token in (0, 1))
        own_call = min(own_call, float(self.reset_calls().max()))
        self.ledger.add("reset_soft_path_mass_bounds")
        return own_encoder * own_readout * own_call ** max_length


class ResetInterfaceSession:
    def __init__(self, core_payload: Mapping[str, Any], ledger: CostLedger, passes: int = 3):
        self.ledger, self.model = ledger, FactorizedResetBlock(core_payload, ledger, passes)
        self.source_checkpoint_bytes: int | None = None
        self.admission_cold_receipts: dict[str, Any] | None = None

    @classmethod
    def from_core_checkpoint(cls, path: Path, ledger: CostLedger) -> "ResetInterfaceSession":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        ledger.add("inherited_checkpoint_loads")
        session = cls(payload, ledger)
        session.source_checkpoint_bytes = path.stat().st_size
        return session

    def step(self, examples: Sequence[ProgramExample], rng: random.Random,
             component: str = "total", learning_rate: float = .05) -> dict[str, Any]:
        if component not in ("total", "within", "class"):
            raise ValueError("unknown gradient ablation")
        gradients = self.model.gradient(examples, rng)
        commit = self.model.transaction(gradients[component], learning_rate)
        return {"gradient": gradients["record"], "transaction": commit,
                "unused_components_still_charged": True}

    @torch.no_grad()
    def evaluate(self, examples: Sequence[ProgramExample], cleanup: bool = True) -> dict[str, Any]:
        if not examples:
            raise ValueError("nonempty evaluation set required")
        pair_map = self.model.bank.program_map(self.model.actions(), mean=True)
        soft_correct = native_correct = 0
        rows = []
        for example in examples:
            self.ledger.add("feedback_calls")
            self.ledger.add("reset_terminal_label_accesses")
            probability = self.model.soft_probability(example, pair_map)
            native = self.model.native(example, pair_map, cleanup)
            soft_correct += int(int(probability.argmax()) == example.target)
            native_correct += int(native["output"] == example.target)
            rows.append({"example": asdict(example), "soft_probability": probability.cpu().tolist(),
                         "native_output": native["output"], "storage": native["storage"],
                         "cleanup": native["cleanup"]})
        result = {"soft_accuracy": soft_correct / len(examples), "native_accuracy": native_correct / len(examples),
                  "pair_map": pair_map, "logical_channel": self.model.logical_channel(pair_map),
                  "rows": rows, "parameter_storage": self.parameter_storage()}
        self.ledger.emit("reset_evaluation", **result)
        return result

    def parameter_storage(self) -> dict[str, Any]:
        return {"total_parameters": sum(p.numel() for p in self.model.parameters()),
                "learnable_parameters": sum(p.numel() for _, p in self.model.trainable_named()),
                "frozen_parameters": sum(p.numel() for p in self.model.parameters() if not p.requires_grad),
                "learnable_parameter_names": [name for name, _ in self.model.trainable_named()],
                "registered_buffer_scalars": sum(b.numel() for b in self.model.buffers()),
                "pair_receipts": self.model.bank.storage(),
                "inherited_source_checkpoint_bytes": self.source_checkpoint_bytes,
                "unregistered_source_metadata_bytes": len(json.dumps({"critical": [[c.real, c.imag] for c in self.model.bank.executor.critical],
                    "roots": [[z.real, z.imag] for z in self.model.bank.executor.base_roots],
                    "families": [f.configuration() for f in self.model.bank.families]}).encode()),
                "configuration_count_is_not_parameter_count": True}

    def admit_if_successful(self, examples: Sequence[ProgramExample]) -> bool:
        result = self.evaluate(examples)
        if result["native_accuracy"] != 1. or result["soft_accuracy"] != 1.:
            return False
        if len(set(result["logical_channel"])) == 3 or int(self.model.reset_calls().argmax()) != 3:
            self.ledger.emit("reset_admission_pending", reason="a noninvertible called local lift has not formed")
            return False
        mass = self.model.soft_path_mass()
        if mass <= SOFT_NATIVE_MIN_MASS:
            self.ledger.emit("reset_admission_pending", reason="new soft/native confidence contract incomplete", path_mass=mass)
            return False
        self.admission_cold_receipts = self.audit_cold_local_maps()
        self.model.admitted_pair = list(result["pair_map"])
        chosen = int(self.model.reset_calls().argmax())
        channels = [list(range(3)), self.model.core_maps()[0], self.model.core_maps()[1], result["logical_channel"]]
        self.model.admitted_logical = list(channels[chosen])
        self.ledger.emit("reset_lift_admission", pair_map=self.model.admitted_pair,
                         logical_channel=self.model.admitted_logical,
                         garbage_targets_supplied=False)
        return True

    def audit_cold_local_maps(self) -> dict[str, Any]:
        """Actually lift the full learned mean program and its reversed program.

        This is distinct from composing paid representative maps. No acquisition
        or physical-execution claim may silently substitute that composition for
        this audit. Root may call it on frozen partial outcomes too.
        """
        actions = self.model.actions()
        expected = self.model.bank.program_map(actions, mean=True)
        forward = self.model.bank.cold_program(actions)
        backward = self.model.bank.cold_program(actions, inverse=True)
        inverse = inverse_permutation(expected, self.ledger)
        if forward["endpoint_map"] != expected or backward["endpoint_map"] != inverse:
            raise NumericalFailure("full learned D9 cold program disagrees with composed source receipts")
        result = {"actions": actions, "forward": forward, "inverse": backward,
                  "cached_composition_equal": True, "actual_full_mean_program_lifted": True}
        self.ledger.emit("full_pair_cold_audit", **result)
        return result

    def maintain_own_call_margin(self) -> dict[str, Any]:
        """Optional ONCE-per-protocol own-class repair; no correct route supplied.

        The caller must freeze its prospective trigger (e.g. TRAIN dual success
        with pending confidence). Targets use only current own argmax. This costs
        one normal transaction inside the same update cap and changes no P.
        """
        model = self.model
        with torch.no_grad():
            hidden = model.core.shared(model.reset_feature)
            norm = torch.dot(hidden, hidden)
            if float(norm) <= 1e-12:
                raise NumericalFailure("own call features do not support a margin fit")
            choice = int(model.reset_calls().argmax())
            logits = torch.full((4,), -6., dtype=torch.float64)
            logits[choice] = 6.
            target = torch.outer(logits, hidden) / norm
            if float(target.abs().max()) > 16:
                raise NumericalFailure("own call margin leaves the declared parameter box")
            gradients = [torch.zeros_like(parameter) if name != "reset_call_head.weight"
                         else (parameter.detach() - target) / .05
                         for name, parameter in model.trainable_named()]
        self.ledger.add("reset_own_call_margin_fits")
        commit = model.transaction(gradients, .05)
        return {"own_choice": choice, "external_target_information": 0,
                "parameter_count_unchanged": True, "transaction": commit}

    def checkpoint(self, open_arenas: Sequence[TypedHistoryArena] = ()) -> dict[str, Any]:
        self.model.assert_frame()
        if any(arena.frame_hash != self.model.frozen_core_hash for arena in open_arenas):
            raise NumericalFailure("cannot encode a live arena under a different frame")
        return {"version": "factorized-reset-interface-v1", "core_payload": self.model.source_core_payload,
                "passes": self.model.bank.gates // 8,
                "state_dict": {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()},
                "frozen_core_hash": self.model.frozen_core_hash,
                "pair_namespace": self.model.bank.namespace,
                "pair_maps_cache": [{"key": list(key), "value": copy.deepcopy(value)}
                                    for key, value in self.model.bank.maps_cache.items()],
                "pair_events_cache": copy.deepcopy(self.model.bank.events_cache),
                "admitted_pair": self.model.admitted_pair, "admitted_logical": self.model.admitted_logical,
                "admission_cold_receipts": self.admission_cold_receipts,
                "source_checkpoint_bytes": self.source_checkpoint_bytes,
                "open_typed_histories": [arena.checkpoint() for arena in open_arenas],
                "encoded_live_history_bytes": sum(len(json.dumps(arena.checkpoint()).encode()) for arena in open_arenas),
                "costs": self.ledger.snapshot(), "parameter_storage": self.parameter_storage(),
                "scope": "frame-frozen local-interface variant; no shared continual-adaptation claim"}

    @classmethod
    def restore(cls, payload: Mapping[str, Any], ledger: CostLedger) -> "ResetInterfaceSession":
        session = cls(payload["core_payload"], ledger, int(payload["passes"]))
        session.model.load_state_dict(payload["state_dict"], strict=True)
        if (session.model.core.parameter_hash() != payload["frozen_core_hash"]
                or session.model.bank.namespace != payload["pair_namespace"]):
            raise NumericalFailure("reset actual NN/frame/source checkpoint mismatch")
        session.model.bank.maps_cache = {tuple(entry["key"]): copy.deepcopy(entry["value"])
                                        for entry in payload["pair_maps_cache"]}
        session.model.bank.events_cache = {int(key): list(value) for key, value in payload["pair_events_cache"].items()}
        session.model.admitted_pair = copy.deepcopy(payload["admitted_pair"])
        session.model.admitted_logical = copy.deepcopy(payload["admitted_logical"])
        session.source_checkpoint_bytes = payload["source_checkpoint_bytes"]
        session.admission_cold_receipts = copy.deepcopy(payload.get("admission_cold_receipts"))
        session.restored_open_arenas = [TypedHistoryArena.restore(arena, ledger)
                                        for arena in payload.get("open_typed_histories", ())]
        ledger.add("reset_checkpoint_restores")
        return session

    def diagnostic_specification(self) -> tuple[tuple[ProgramExample, ...], tuple[ProgramExample, ...], dict[str, Any]]:
        """Exact prospective data; methods receive records, not the generator rule."""
        signature = self.model.core.native_signature(self.model.core_maps(), self.ledger)
        def target(initial: int, word: Sequence[int]) -> int:
            current = initial
            for token in word:
                current = 0 if token == 2 else native_output(signature, ProgramExample("source", current, (token,), 0), self.ledger)
            return current
        training = tuple(ProgramExample(f"train-reset-{x}-{suffix}", x, (2,) if not suffix else (2, 1),
                                         target(x, (2,) if not suffix else (2, 1)))
                         for x in range(3) for suffix in (0, 1))
        words = ((1, 2, 1), (0, 2, 1, 0), (2, 1, 2, 0),
                 (0, 2, 1, 2, 1), (2, 1, 0, 2, 1, 0), (2, 1, 2, 1, 2, 1))
        heldout = tuple(ProgramExample(f"heldout-{i}", i % 3, word, target(i % 3, word))
                        for i, word in enumerate(words))
        manifest = {"status": "PROPOSED_NOT_EXECUTED", "seed": 0, "learning_rate": .05,
            "max_updates_per_arm": 100, "suggested_internal_seconds": 90, "outside_timeout_seconds": 120,
            "training": [asdict(e) for e in training], "heldout_count": len(heldout),
            "heldout_policy": "evaluate only after weights/selection frozen",
            "native_history_capacity": 4, "local_joint_states": 9,
            "soft_certificate_scope": "length<=16 and at most4 reset tokens; fresh cell on every selected pair transition",
            "new_native_function": "acquired derived channel must be many-to-one in the middle of mixed programs",
            "acquisition_input": "terminal examples; no target pair permutation, garbage coding or correct call",
            "gradient": "within call-head plus unbiased one-coordinate boundary term, uniform1/gates sampling",
            "maintenance": "at most one own-current-call margin fit after TRAIN dual success with pending path mass; charged transaction",
            "controls": ["same-event ordinary loop optimizer", "learned direct logical channel",
                         "canonical reversible compiler FROM that learned channel, then optional charged source synthesis"],
            "audits": ["old ability", "native and soft new function", "fresh cells and cleanup",
                       "actual NN checkpoint reload", "cold D9 forward/inverse map agreement", "all storage/costs"],
            "frozen_core": True, "does_not_test": "shared continual-backbone adaptation or competitive superiority"}
        return training, heldout, manifest


class LearnedClassicalChannel:
    """Strong direct control: learn a logical table from the SAME terminal records."""

    def __init__(self, session: ResetInterfaceSession):
        self.session, self.learned = session, None

    def fit(self, examples: Sequence[ProgramExample]) -> list[int]:
        observed: dict[int, int] = {}
        for example in examples:
            if example.word != (2,):
                continue
            self.session.ledger.add("feedback_calls")
            self.session.ledger.add("classical_terminal_label_accesses")
            if example.target not in (0, 1, 2):
                raise ValueError("terminal label outside declared logical alphabet")
            if example.initial_sheet in observed and observed[example.initial_sheet] != example.target:
                raise ValueError("inconsistent singleton terminal observations")
            observed[example.initial_sheet] = example.target
        if set(observed) != {0, 1, 2}:
            raise NumericalFailure("classical channel is not identified on all three starts")
        self.learned = [observed[x] for x in range(3)]
        self.session.ledger.add("classical_channel_table_writes", 3)
        return list(self.learned)

    def physical_channel(self) -> list[int]:
        if self.learned is None:
            raise RuntimeError("fit the channel from observed terminal records first")
        encoder = self.session.model.reference["encoder"]
        result = [0] * 3
        for raw, target in enumerate(self.learned):
            result[encoder[raw]] = encoder[target]
            self.session.ledger.add("classical_frame_conversion_entries")
        return result

    def evaluate(self, examples: Sequence[ProgramExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("nonempty classical evaluation set required")
        channel, model = self.physical_channel(), self.session.model
        signature = model.core.native_signature(model.core_maps(), self.session.ledger)
        correct = 0
        for example in examples:
            state = signature["encoder"][example.initial_sheet]
            for token in example.word:
                if token == 2:
                    state = channel[state]
                    self.session.ledger.add("classical_irreversible_channel_overwrites")
                else:
                    for route in signature["routes"][token]:
                        if route:
                            state = signature["maps"][route - 1][state]
                        self.session.ledger.add("classical_old_gate_applications")
            self.session.ledger.add("feedback_calls")
            correct += int(signature["readout"][state] == example.target)
        return {"accuracy": correct / len(examples), "channel_int64_payload_bytes": 24,
                "retained_garbage": 0, "cleanup": "not required for this irreversible logical control",
                "physical_energy": "not measured or inferred"}


class CanonicalFactorizedLiftControl:
    """Compile only a learned channel; no teacher permutation enters this object."""

    def __init__(self, channel: LearnedClassicalChannel):
        self.channel = channel
        self.pair_map: list[int] | None = None

    def compile(self) -> list[int]:
        physical = self.channel.physical_channel()
        ledger = self.channel.session.ledger
        counts, image = [0] * 3, [None] * 9
        for source, target in enumerate(physical):
            image[3 * source] = 3 * target + counts[target]
            counts[target] += 1
            ledger.add("canonical_preimage_slot_assignments")
        used = {value for value in image if value is not None}
        # Preserve available identity entries, then biject the remaining sets.
        for source in range(9):
            if image[source] is None and source not in used:
                image[source] = source
                used.add(source)
                ledger.add("canonical_identity_completions")
        remaining = iter(sorted(set(range(9)) - used))
        for source in range(9):
            if image[source] is None:
                image[source] = next(remaining)
                ledger.add("canonical_remaining_assignments")
        result = [int(value) for value in image]
        permutation_hash(result)
        ledger.add("canonical_pair_map_validation_entries", 9)
        self.pair_map = result
        return list(result)

    def evaluate(self, examples: Sequence[ProgramExample], cleanup: bool = True) -> dict[str, Any]:
        """Digital factorized reversible control, with the SAME typed arena cost."""
        if self.pair_map is None or not examples:
            raise ValueError("compile a learned channel and supply explicit evaluation records")
        session, model = self.channel.session, self.channel.session.model
        ledger, core_maps = session.ledger, model.core_maps()
        signature = model.core.native_signature(core_maps, ledger)
        namespace = "canonical-lift-from-learned-channel-v1"
        rows, correct = [], 0
        for example in examples:
            arena = TypedHistoryArena(signature["encoder"][example.initial_sheet],
                                       model.frozen_core_hash, ledger, model.history_capacity)
            for token in example.word:
                if token == 2:
                    arena.reset_gate(self.pair_map, namespace)
                elif token in (0, 1):
                    for route in signature["routes"][token]:
                        if route:
                            arena.core_gate(route - 1, core_maps[route - 1], model.frozen_core_hash)
                else:
                    raise ValueError("classical control received unknown token")
                ledger.add("classical_factorized_call_decisions")
            physical = arena.copy_result()
            output = signature["readout"][physical]
            storage = arena.storage()
            receipt = arena.cleanup(core_maps, self.pair_map, model.frozen_core_hash,
                                      namespace, model.core.parameter_hash()) if cleanup else None
            ledger.add("feedback_calls")
            correct += int(output == example.target)
            rows.append({"example": asdict(example), "output": output, "storage": storage, "cleanup": receipt})
        return {"accuracy": correct / len(examples), "rows": rows,
                "pair_map_int64_payload_bytes": 72, "learned_channel_int64_payload_bytes": 24,
                "execution": "digital table compiled from learned observations; D9 cold actuation separate"}

    def cold_actuation(self) -> dict[str, Any]:
        synthesis = self.synthesize_paid_loops()
        word, bank = synthesis["generator_word"], self.channel.session.model.bank
        if not word:
            # An empty word is the declared identity source; no nontrivial cold
            # actuation is fabricated in this edge case.
            return {"synthesis": synthesis, "identity_word": True}
        if len(word) > 32:
            raise BudgetStop("compiled actuation exceeds this executor's32-loop source cap")
        loops = [bank.families[family].vertices([.5 * (bank.events(family)[0] + 12.)]) for family in word]
        receipt = bank.executor.execute(loops)
        if receipt["endpoint_map"] != self.pair_map:
            raise NumericalFailure("cold classical compiled word disagrees with learned lift")
        return {"synthesis": synthesis, "cold_receipt": receipt,
                "source_actuation_paid": True, "target_specification": "learned channel only"}

    def synthesize_paid_loops(self, max_gates: int = 128) -> dict[str, Any]:
        """Source-aware conventional synthesis using ACTUALLY observed generators.

        Digital-table use does not need this. Same-substrate actuation does, and
        all cold generator observations, graph work and emitted gates are counted.
        The synthesized word is not passed to the neural learner.
        """
        if self.pair_map is None:
            raise RuntimeError("compile the learned channel first")
        session, ledger = self.channel.session, self.channel.session.ledger
        graph: dict[int, list[tuple[int, int]]] = {q: [] for q in range(9)}
        generators = []
        for family in range(8):
            events = session.model.bank.events(family)
            if len(events) != 1:
                raise NumericalFailure("this bounded compiler requires an observed simple one-event family")
            action = .5 * (events[0] + 12.)
            image = session.model.bank.generator_map(family, action)
            changed = [q for q, target in enumerate(image) if target != q]
            ledger.add("classical_generator_support_checks", 9)
            if len(changed) != 2 or image[changed[0]] != changed[1] or image[changed[1]] != changed[0]:
                raise NumericalFailure("observed generator is not a transposition")
            a, b = changed
            graph[a].append((b, family)); graph[b].append((a, family))
            generators.append(image)
        def transposition_word(a: int, b: int) -> list[int]:
            queue, previous = [a], {a: None}
            for vertex in queue:
                if vertex == b:
                    break
                for neighbor, family in graph[vertex]:
                    ledger.add("classical_synthesis_graph_edges")
                    if neighbor not in previous:
                        previous[neighbor] = vertex, family
                        queue.append(neighbor)
            if b not in previous:
                raise NumericalFailure("observed transposition graph is disconnected")
            path, current = [], b
            while current != a:
                parent, family = previous[current]
                path.append(family); current = parent
            path.reverse()
            return path + list(reversed(path[:-1]))
        visited, word = set(), []
        for start in range(9):
            if start in visited:
                continue
            cycle, current = [], start
            while current not in visited:
                visited.add(current); cycle.append(current)
                current = self.pair_map[current]
                ledger.add("classical_synthesis_cycle_entries")
            for target in cycle[1:]:
                word.extend(transposition_word(cycle[0], target))
        if len(word) > max_gates:
            raise BudgetStop("classical synthesized word exceeds its declared gate cap")
        actual = list(range(9))
        for family in word:
            actual = [generators[family][value] for value in actual]
            ledger.add("classical_synthesis_verification_entries", 9)
        if actual != self.pair_map:
            raise NumericalFailure("classical source-aware synthesis failed actual-map verification")
        return {"generator_word": word, "gate_count": len(word), "verified_pair_map": actual,
                "pair_map_int64_payload_bytes": 72, "channel_payload_bytes": 24,
                "compiled_from": "channel learned only from common terminal observations",
                "physical_execution": "generator maps paid; complete cold word execution separate"}
