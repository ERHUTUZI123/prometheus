"""Unexecuted constructive adapter: coupled neural credit + learned loop classes.

Same Prometheus V2. MAIN=TOPOLOGY USER_LOCKED, human-only change. Transformer
remains the primary architecture baseline. This NEW module does not modify the
frozen loop diagnostic, native prototype, or canonical water/method.md.

Exact learning interface in the stated finite, fixed-cover scope:

  J(theta) = E_V[L(V; theta)], V ~ p_sigma(V-a_theta).
  grad J = E[partial_theta L(V;theta)]
           + (D_theta a_theta)^T E[boundary-jump gradient].

The first derivative holds all executed maps FIXED and learns encoder, readout,
calls and their shared backbone. The second holds theta fixed while measuring
terminal-loss jumps between loop classes, then chains through the SAME backbone.
Dropping the first term loses real call/E/R credit; differentiating a supplied
soft permutation table would revert to the old map-first renderer. There is NO
trainable W_k/permutation score tensor in this model.

This first adapter has D=3, two primitive bank slots, one forming primitive, a
fixed affine loop family, and a bounded old input alphabet. Capacity growth,
continuous input-domain certificates, and competitive generalization are OPEN.
It can be called by Root inside an existing bounded local execution envelope;
there is no import-time experiment, network, subprocess, or remote execution.

All event geometry, terminal labels, actual root observations, caches and cost
rules are also available to conventional controls. `same_certificate` is exact
ordinary implementation correspondence. `map_reexecute` is a cost/plasticity
ablation, not the strongest generic rival; it too may use the analytic guard.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from loop_boundary_learning import (
    AffineLoopFamily, BudgetStop, CostLedger, CubicCoveringExecutor,
    NumericalFailure, ProgramExample, ProtectedLoopSet, RaisedCosineKernel,
    loop_clearance, ordinary_boundary_gradient, slice_events,
    vertices_in_execution_box,
)


TerminalLoss = Callable[[torch.Tensor, ProgramExample], torch.Tensor]
SOFT_NATIVE_MIN_MASS = 0.5 + 1e-6


def terminal_negative_log_likelihood(probability: torch.Tensor,
                                     example: ProgramExample) -> torch.Tensor:
    """Terminal label only; clipped probability is the declared finite objective."""
    return -torch.log(probability[example.target].clamp_min(1e-12))


class HybridLoopBlock(nn.Module):
    """Learned E/R/calls and learned loop vertices share actual neural weights.

    Slots use choices [skip, bank0, bank1]. Skip is the supplied identity program;
    bank maps are detached results of paid continuation, never learned matrices.
    The two public feature vectors are inductive biases, not target operations.
    All initial/frozen buffers and allocated neural parameters are disclosed.
    """

    def __init__(self, slots: int = 1, new_route_seeded: bool = False):
        super().__init__()
        if not 1 <= slots <= 2:
            raise ValueError("bounded adapter supports one or two call slots")
        self.slots = int(slots)
        self.new_route_seeded = bool(new_route_seeded)
        self.states, self.bank_size = 3, 2
        self.register_buffer("operation_features", torch.tensor([[1., 0.], [1., 1.]],
                                                                dtype=torch.float64))
        self.register_buffer("token_features", self.operation_features.detach().clone())
        self.shared = nn.Linear(2, 2, bias=False).double()
        self.loop_head = nn.Linear(2, 1, bias=False).double()
        self.calls = nn.ModuleList(nn.Linear(2, 3, bias=False).double() for _ in range(slots))
        self.encoder_logits = nn.Parameter(6 * torch.eye(3, dtype=torch.float64))
        self.readout_logits = nn.Parameter(6 * torch.eye(3, dtype=torch.float64))
        self.forward_counts = {"loop": 0, "call": 0, "soft_program": 0}
        with torch.no_grad():
            self.shared.weight.copy_(torch.eye(2, dtype=torch.float64))
            self.loop_head.weight.copy_(torch.tensor([[-3., 4.]], dtype=torch.float64))
            for slot, head in enumerate(self.calls):
                if slot == 0:
                    if new_route_seeded:
                        initialization = [[-8., 0.], [8., -8.], [0., 8.]]
                    else:
                        # Old token binding is a declared acquired starting
                        # mechanism. NEW token starts uniform across all three
                        # choices, so successful native admission also requires
                        # actual terminal-trained call formation.
                        initialization = [[-8., 8.], [8., -8.], [0., 0.]]
                    head.weight.copy_(torch.tensor(initialization, dtype=torch.float64))
                else:
                    head.weight.copy_(torch.tensor([[8., 0.], [-8., 0.], [-8., 0.]],
                                                   dtype=torch.float64))

    def action_tensor(self, operation: int) -> torch.Tensor:
        self.forward_counts["loop"] += 1
        return self.loop_head(self.shared(self.operation_features[int(operation)]))

    def action(self, operation: int) -> list[float]:
        return self.action_tensor(operation).detach().cpu().tolist()

    def call_probabilities(self, token: int) -> torch.Tensor:
        self.forward_counts["call"] += 1
        hidden = self.shared(self.token_features[int(token)])
        return torch.stack([torch.softmax(head(hidden), dim=-1) for head in self.calls])

    def parameter_hash(self) -> str:
        # Include fixed feature buffers: changing the public input binding can
        # change old loops even when trainable weight values are identical.
        arrays = {name: tensor.detach().cpu().reshape(-1).tolist()
                  for name, tensor in self.state_dict().items()}
        return hashlib.sha256(json.dumps(arrays, sort_keys=True).encode()).hexdigest()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def soft_probability(self, example: ProgramExample, maps: Mapping[int, Sequence[int]],
                         ledger: CostLedger) -> torch.Tensor:
        """Within-chamber differentiable path; executed maps are CONSTANT tensors."""
        self.forward_counts["soft_program"] += 1
        alpha = torch.softmax(self.encoder_logits[example.initial_sheet], dim=-1)
        readout = torch.softmax(self.readout_logits, dim=-1)
        identity = torch.eye(3, dtype=alpha.dtype, device=alpha.device)
        matrices = [identity]
        for operation in range(self.bank_size):
            image = torch.as_tensor(maps[operation], dtype=torch.long, device=alpha.device)
            matrices.append(torch.nn.functional.one_hot(image, num_classes=3).to(alpha.dtype))
        bank = torch.stack(matrices).detach()
        for token in example.word:
            for probabilities in self.call_probabilities(token):
                transition = torch.einsum("k,kij->ij", probabilities, bank)
                alpha = alpha @ transition
                ledger.add("soft_slot_matrix_applications")
        ledger.add("soft_terminal_readouts")
        return alpha @ readout

    @torch.no_grad()
    def native_signature(self, maps: Mapping[int, Sequence[int]], ledger: CostLedger) -> dict[str, Any]:
        ledger.add("native_signature_extractions")
        return {"encoder": self.encoder_logits.argmax(-1).cpu().tolist(),
                "readout": self.readout_logits.argmax(-1).cpu().tolist(),
                "routes": [self.call_probabilities(token).argmax(-1).cpu().tolist()
                           for token in range(len(self.token_features))],
                "maps": {int(op): list(values) for op, values in maps.items()}}

    @torch.no_grad()
    def soft_native_path_bound(self, old_tokens: Sequence[int], max_length: int,
                               ledger: CostLedger) -> float:
        """Nonnegative own-native path mass; bank maps are exact one-hot here.

        Above 1/2, soft argmax agrees with the current hard native program for
        all starts and words in old_tokens up to max_length. This is an analytic
        real-arithmetic bound evaluated in doubles, not an interval certificate.
        """
        encoder = torch.softmax(self.encoder_logits, -1)
        readout = torch.softmax(self.readout_logits, -1)
        p_encoder = float(encoder.max(-1).values.min())
        p_readout = float(readout.max(-1).values.min())
        route_min = min(float(self.call_probabilities(token).max(-1).values.min())
                        for token in old_tokens) if old_tokens else 1.0
        ledger.add("soft_path_mass_bounds")
        return p_encoder * p_readout * route_min ** (self.slots * max_length)


def native_output(signature: Mapping[str, Any], example: ProgramExample,
                  ledger: CostLedger) -> int:
    state = signature["encoder"][example.initial_sheet]
    for token in example.word:
        for route in signature["routes"][token]:
            if route:
                state = signature["maps"][route - 1][state]
            ledger.add("native_slot_applications")
    return int(signature["readout"][state])


def native_relation_preserved(reference: Mapping[str, Any], candidate: Mapping[str, Any],
                              old_tokens: Sequence[int], ledger: CostLedger) -> bool:
    """Finite exact decoded relation closure, preserving every finite old word."""
    queue = list(zip(reference["encoder"], candidate["encoder"]))
    seen: set[tuple[int, int]] = set()
    while queue:
        left, right = queue.pop()
        if (left, right) in seen:
            continue
        seen.add((left, right))
        ledger.add("native_relation_pairs")
        if reference["readout"][left] != candidate["readout"][right]:
            return False
        for token in old_tokens:
            next_left, next_right = left, right
            for route in reference["routes"][token]:
                if route:
                    next_left = reference["maps"][route - 1][next_left]
            for route in candidate["routes"][token]:
                if route:
                    next_right = candidate["maps"][route - 1][next_right]
            ledger.add("native_relation_token_checks")
            queue.append((next_left, next_right))
    return True


class ActualLoopMapBank:
    """Derived, charged maps and receipts; there is no trainable permutation W.

    Exact geometry keys cache actual executions. Admitted semantic reuse relies
    on accepted topology/map transactions and its receipt, NOT on a claim that
    each changed trajectory was numerically executed. Both controls can use it.
    Cache payload and every source observation are exposed in the checkpoint.
    """

    def __init__(self, model: HybridLoopBlock, family: AffineLoopFamily,
                 executor: CubicCoveringExecutor, protected: ProtectedLoopSet,
                 capacity: int = 128):
        if not 1 <= capacity <= 128:
            raise ValueError("explicit bounded derived-map capacity 1..128 required")
        self.model, self.family, self.executor, self.protected = model, family, executor, protected
        self.capacity = int(capacity)
        self.cache: dict[tuple[int, str], dict[str, Any]] = {}
        self.namespace = hashlib.sha256(("fixed-cubic-z3-3z-u-base0-v1:"
                                         + json.dumps(family.configuration(), sort_keys=True)).encode()).hexdigest()
        self.semantic_license_hash: str | None = None

    def _check_namespace(self) -> None:
        namespace = hashlib.sha256(("fixed-cubic-z3-3z-u-base0-v1:"
                                   + json.dumps(self.family.configuration(), sort_keys=True)).encode()).hexdigest()
        self.executor.ledger.add("map_family_namespace_checks")
        if namespace != self.namespace:
            raise NumericalFailure("cover/family mutation requires a new certified namespace")

    def _safe_semantic_representative(self, action: Sequence[float]) -> list[float]:
        """Same class for an otherwise unresolved noisy scalar sample.

        This uses the SAME event geometry all arms receive. It does not reject
        or reweight smoothing samples: the ideal class loss is evaluated at a
        safe representative. Mean-path admission NEVER uses this substitution.
        Higher-dimensional/no-safe-interior cases remain unresolved.
        """
        if self.family.dimension != 1:
            raise NumericalFailure("no generic high-dimensional interiorization assumed")
        ledger = self.executor.ledger
        self._check_namespace()
        lo, hi = self.family.action_box
        events = slice_events(self.family, action, 0, lo, hi,
                              self.executor.critical_controls, ledger)
        boundaries = [lo, *events, hi]
        for left, right in zip(boundaries, boundaries[1:]):
            if left < action[0] < right:
                representative = [(left + right) / 2]
                vertices = self.family.vertices(representative)
                if (not vertices_in_execution_box(vertices, ledger.limits)
                        or loop_clearance(vertices, self.executor.critical_controls)
                           < ledger.limits.min_control_clearance):
                    raise NumericalFailure("sample class has no checked safe scalar representative")
                ledger.add("ideal_class_representative_substitutions")
                return representative
        raise NumericalFailure("sample exactly on a discriminant or action-box boundary")

    def observed_map(self, operation: int, action: Sequence[float], *,
                     ideal_semantic_sample: bool = False) -> list[int]:
        self._check_namespace()
        ledger = self.executor.ledger
        executed_action = list(float(v) for v in action)
        vertices = self.family.vertices(executed_action)
        ledger.add("map_request_domain_checks")
        if (not vertices_in_execution_box(vertices, ledger.limits)
                or loop_clearance(vertices, self.executor.critical_controls)
                   < ledger.limits.min_control_clearance):
            if not ideal_semantic_sample:
                raise NumericalFailure("deterministic mean loop not executable")
            executed_action = self._safe_semantic_representative(action)
            vertices = self.family.vertices(executed_action)
        key = int(operation), self.namespace + ":" + ",".join(float(value).hex() for value in executed_action)
        if key in self.cache:
            ledger.add("derived_map_exact_cache_hits")
            return list(self.cache[key]["receipt"]["endpoint_map"])
        receipt = self.executor.execute([vertices])
        entry = {"action": executed_action, "receipt": receipt,
                 "source": "actual cubic continuation, not target map"}
        if len(self.cache) >= self.capacity:
            del self.cache[next(iter(self.cache))]
            ledger.add("derived_map_evictions")
        self.cache[key] = entry
        ledger.add("derived_map_cold_lifts")
        ledger.emit("hybrid_actual_map", operation=operation, requested_action=list(action),
                    executed_action=executed_action, receipt=receipt,
                    full_observation_available_to_conventional_control=True)
        return list(receipt["endpoint_map"])

    def maps(self, overrides: Mapping[int, Sequence[float]] | None = None, *,
             ideal_semantic_sample: bool = False) -> dict[int, list[int]]:
        self._check_namespace()
        overrides = overrides or {}
        result = {}
        if self.protected.admitted and self.semantic_license_hash != self.model.parameter_hash():
            raise NumericalFailure("old map reuse lacks an accepted current-NN transaction license")
        for operation in range(self.model.bank_size):
            if operation in self.protected.admitted and operation not in overrides:
                self.executor.ledger.add("admitted_semantic_receipt_reuses")
                result[operation] = list(self.protected.receipts[operation]
                                         ["actual_lift_receipt"]["endpoint_map"])
            else:
                action = overrides.get(operation)
                if action is None:
                    action = self.model.action(operation)
                result[operation] = self.observed_map(operation, action,
                                                       ideal_semantic_sample=ideal_semantic_sample)
        return result

    def storage(self) -> dict[str, int]:
        serializable = [{"key": key, **entry} for key, entry in self.cache.items()]
        return {"entries": len(self.cache), "capacity": self.capacity,
                "map_int64_payload_bytes": 8 * 3 * len(self.cache),
                "serialized_cache_bytes": len(json.dumps(serializable).encode())}


class HybridTerminalObjective:
    def __init__(self, model: HybridLoopBlock, examples: Sequence[ProgramExample],
                 ledger: CostLedger, terminal_loss: TerminalLoss = terminal_negative_log_likelihood):
        if not examples:
            raise ValueError("nonempty current terminal batch required")
        self.model, self.examples, self.ledger = model, tuple(examples), ledger
        self.terminal_loss = terminal_loss

    def tensor(self, maps: Mapping[int, Sequence[int]]) -> torch.Tensor:
        losses = []
        for example in self.examples:
            self.ledger.add("feedback_calls")
            self.ledger.add("terminal_label_accesses")
            probability = self.model.soft_probability(example, maps, self.ledger)
            loss = self.terminal_loss(probability, example)
            if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
                raise TypeError("hybrid terminal callback must return a scalar torch loss")
            if not bool(torch.isfinite(loss.detach())):
                raise NumericalFailure("nonfinite terminal loss")
            losses.append(loss)
            self.ledger.emit("hybrid_terminal_feedback", example=asdict(example),
                             probability=probability.detach().cpu().tolist(),
                             loss=float(loss.detach()),
                             information="terminal label only; actual lifted maps are common observations")
        return torch.stack(losses).mean()


def _zeros_for_missing(parameters: Sequence[torch.Tensor],
                       gradients: Sequence[torch.Tensor | None]) -> list[torch.Tensor]:
    return [torch.zeros_like(parameter) if gradient is None else gradient.detach()
            for parameter, gradient in zip(parameters, gradients)]


def _norm(gradients: Sequence[torch.Tensor]) -> float:
    return math.sqrt(sum(float((gradient * gradient).sum()) for gradient in gradients))


def coupled_gradient(model: HybridLoopBlock, bank: ActualLoopMapBank,
                     objective: HybridTerminalObjective, operation: int,
                     kernel: RaisedCosineKernel, rng: random.Random,
                     *, path_estimator: str = "event", within_samples: int = 1) -> dict[str, Any]:
    """Correct coupled theta derivative; all map tensors detached, no W surrogate.

    `event`/`ordinary_event` deliberately share the generic boundary estimator.
    `score` uses identical noise density/observations and the same BPTT term.
    Samples near the finite-clearance exclusion tube may be evaluated at the
    same class's safe representative, with geometry work charged to every arm.
    """
    if path_estimator not in ("event", "ordinary_event", "score") or within_samples < 1:
        raise ValueError("unknown estimator or invalid bounded sample count")
    if operation in bank.protected.admitted:
        raise ValueError("class-changing exploration requires an unadmitted/versioned bank slot")
    ledger = bank.executor.ledger
    action_tensor = model.action_tensor(operation)
    mean = action_tensor.detach().cpu().tolist()
    bank.family.require_support(mean, kernel.sigma)
    losses, score_terms, draws = [], [], []
    for _ in range(within_samples):
        noise = [kernel.sample(rng, ledger) for _ in mean]
        action = [a + delta for a, delta in zip(mean, noise)]
        maps = bank.maps({operation: action}, ideal_semantic_sample=True)
        loss = objective.tensor(maps)
        losses.append(loss)
        score_terms.append([float(loss.detach()) * kernel.score(delta) for delta in noise])
        draws.append({"noise": noise, "action": action, "maps": maps,
                      "terminal_loss": float(loss.detach())})
    within_loss = torch.stack(losses).mean()
    parameters = list(model.parameters())
    within_gradients = _zeros_for_missing(parameters, torch.autograd.grad(
        within_loss, parameters, retain_graph=True, allow_unused=True))
    ledger.add("within_chamber_neural_vjps")

    if path_estimator == "score":
        action_gradient = [sum(row[j] for row in score_terms) / within_samples
                           for j in range(len(mean))]
        event_details: dict[str, Any] = {"estimator": "normalized location score", "draws": draws}
    else:
        # Theta is held fixed while measuring these terminal jumps. Only later
        # does the resulting coefficient chain through the shared NN loop head.
        @torch.no_grad()
        def event_loss(action: Sequence[float]) -> float:
            maps = bank.maps({operation: action}, ideal_semantic_sample=True)
            return float(objective.tensor(maps))
        provider = lambda action, j, left, right: slice_events(
            bank.family, action, j, left, right, bank.executor.critical_controls, ledger)
        action_gradient, event_details = ordinary_boundary_gradient(
            mean, kernel, event_loss, provider, rng, ledger)

    # This is a gradient surrogate ONLY. Its scalar value is not the objective.
    path_surrogate = torch.dot(action_tensor, action_tensor.new_tensor(action_gradient).detach())
    class_gradients = _zeros_for_missing(parameters, torch.autograd.grad(
        path_surrogate, parameters, allow_unused=True))
    ledger.add("class_boundary_neural_vjps")
    total = [left + right for left, right in zip(within_gradients, class_gradients)]
    if any(not bool(torch.isfinite(gradient).all()) for gradient in total):
        raise NumericalFailure("nonfinite coupled neural gradient")
    groups = {}
    for (name, _), left, right in zip(model.named_parameters(), within_gradients, class_gradients):
        groups[name] = {"within_norm": float(torch.linalg.vector_norm(left)),
                        "class_norm": float(torch.linalg.vector_norm(right))}
    record = {"mean_action": mean, "within_loss_sample_mean": float(within_loss.detach()),
              "action_gradient": action_gradient, "within_gradient_norm": _norm(within_gradients),
              "class_gradient_norm": _norm(class_gradients), "total_gradient_norm": _norm(total),
              "parameter_groups": groups, "draws": draws, "event_details": event_details,
              "ordinary_correspondence": "event and ordinary_event are exactly the same rule"}
    ledger.emit("coupled_gradient", **record)
    # Ablation runners may select a component AFTER both were computed and
    # charged. Returning these tensors changes no derivative or acquisition rule.
    return {"parameter_gradients": total, "within_parameter_gradients": within_gradients,
            "class_parameter_gradients": class_gradients, "record": record}


class HybridLearningSession:
    """One actual NN/checkpoint lifecycle with complete safe-update transaction."""

    def __init__(self, ledger: CostLedger, *, slots: int = 1,
                 protection: str = "same_certificate", sigma: float = 2.0,
                 old_tokens: Sequence[int] = (0,), max_old_length: int = 16,
                 new_route_seeded: bool = False):
        if protection not in ("same_certificate", "map_reexecute"):
            raise ValueError("unknown common-information protection control")
        self.ledger = ledger
        self.model = HybridLoopBlock(slots, new_route_seeded=new_route_seeded)
        self.family = AffineLoopFamily((1j, -1j), ((1 + 0j, 1 + 0j),))
        self.executor = CubicCoveringExecutor(ledger)
        self.protected = ProtectedLoopSet(policy=("clearance" if protection == "same_certificate"
                                                  else "executed_map"))
        self.kernel = RaisedCosineKernel(sigma)
        self.bank = ActualLoopMapBank(self.model, self.family, self.executor, self.protected)
        self.old_tokens, self.max_old_length = tuple(old_tokens), int(max_old_length)
        self.protection = protection
        self.reference: dict[str, Any] | None = None
        self.transactions = 0
        self.metadata = {"main": "TOPOLOGY_USER_LOCKED", "effective_mode": "unknown",
                         "scope": "fixed D3 loop/E/R/call adapter; state growth OPEN",
                         "trainable_permutation_W": False,
                         "encoding_readout_initialization": "near-identity supplied starting frame, trainable",
                         "new_call_initialization": "seeded bank1 binding" if new_route_seeded else "uniform; terminal learning required",
                         "old_call_initialization": "supplied acquired old bank0 binding",
                         "conventional_control": "same feedback/source/events/certificates/caches/costs"}

    def initialize_old_contract(self) -> dict[str, Any]:
        """Paid actual old operation, then native binding contract and soft bound."""
        if self.reference is not None:
            raise RuntimeError("do not silently replace an existing old-task contract")
        self.protected.admit(0, self.model, self.family, self.executor)
        self.bank.semantic_license_hash = self.model.parameter_hash()
        maps = self.bank.maps()
        self.reference = self.model.native_signature(maps, self.ledger)
        bound = self.model.soft_native_path_bound(self.old_tokens, self.max_old_length, self.ledger)
        if bound <= SOFT_NATIVE_MIN_MASS:
            raise NumericalFailure("initial old soft/native agreement certificate unavailable")
        self.ledger.emit("hybrid_old_contract", reference=self.reference,
                         soft_native_path_lower_bound=bound, max_length=self.max_old_length)
        return {"reference": copy.deepcopy(self.reference), "soft_path_bound": bound}

    def transaction(self, gradients: Sequence[torch.Tensor], *, operation: int = 1,
                    learning_rate: float = .01) -> dict[str, Any]:
        """Atomic NN proposal, loop guard, binding closure, confidence, commit.

        The source and F3 fiber are fixed. Candidate maps are derived/charged.
        On any failure including budget exhaustion, actual Parameter objects are
        restored in place. No Adam or other hidden optimizer state exists here.
        Geometry observations from rejected proposals remain honestly paid.
        """
        if self.reference is None:
            raise RuntimeError("initialize an actual old contract first")
        if not 0 < learning_rate <= .1:
            raise ValueError("explicit bounded hybrid SGD step required")
        started = time.perf_counter()
        parameters = list(self.model.parameters())
        if len(parameters) != len(gradients):
            raise ValueError("gradient/parameter count mismatch")
        originals = [parameter.detach().clone() for parameter in parameters]
        old_license = self.bank.semantic_license_hash
        old_loop_snapshot = self.protected.before(self.model, self.family, self.executor)
        accepted = False
        rejected = []
        try:
            for scale in (1., .5, .25, .125, .0625, .03125, .015625, 0.):
                self.ledger.add("hybrid_proposals")
                with torch.no_grad():
                    for parameter, original, gradient in zip(parameters, originals, gradients):
                        parameter.copy_(original - learning_rate * scale * gradient)
                reason = None
                try:
                    if any(not bool(torch.isfinite(parameter).all())
                           or float(parameter.detach().abs().max()) > 16 for parameter in parameters):
                        raise NumericalFailure("finite NN parameter box exceeded")
                    self.family.require_support(self.model.action(operation), self.kernel.sigma)
                    current_vertices = self.family.vertices(self.model.action(operation))
                    if (not vertices_in_execution_box(current_vertices, self.ledger.limits)
                            or loop_clearance(current_vertices, self.executor.critical_controls)
                               < self.ledger.limits.min_control_clearance):
                        raise NumericalFailure("current mean path outside execution domain")
                    if not self.protected.accepts(self.model, self.family, old_loop_snapshot, self.executor):
                        raise NumericalFailure("old primitive geometry/map guard rejected")
                    # Temporary license follows the just-checked old loop/map
                    # proof. Rejected/aborted proposals restore the old license.
                    self.bank.semantic_license_hash = self.model.parameter_hash()
                    maps = self.bank.maps()
                    candidate = self.model.native_signature(maps, self.ledger)
                    if not native_relation_preserved(self.reference, candidate, self.old_tokens, self.ledger):
                        raise NumericalFailure("old encoder/call/readout closure rejected")
                    bound = self.model.soft_native_path_bound(self.old_tokens, self.max_old_length, self.ledger)
                    if bound <= SOFT_NATIVE_MIN_MASS:
                        raise NumericalFailure("old soft/native path-mass certificate rejected")
                except NumericalFailure as error:
                    reason = str(error)
                if reason is not None:
                    rejected.append({"scale": scale, "reason": reason})
                    self.ledger.add("hybrid_proposal_rejections")
                    continue
                self.ledger.add("updates")
                record = {"accepted_scale": scale, "rejected": rejected,
                          "soft_native_path_bound": bound,
                          "parameter_hash": self.model.parameter_hash(),
                          "forming_action": self.model.action(operation),
                          "old_loop_max_vertex_displacement": max(
                              max(abs(a - b) for a, b in zip(vertices, self.family.vertices(self.model.action(op))))
                              for op, (vertices, _) in old_loop_snapshot.items()) if old_loop_snapshot else 0.0}
                self.transactions += 1
                accepted = True
                self.ledger.emit("hybrid_committed_transaction", **record)
                return record
            raise NumericalFailure("even restored hybrid state failed its original contract")
        finally:
            if not accepted:
                with torch.no_grad():
                    for parameter, original in zip(parameters, originals):
                        parameter.copy_(original)
                self.bank.semantic_license_hash = old_license
            self.ledger.elapsed("hybrid_transaction", started)

    def step(self, examples: Sequence[ProgramExample], rng: random.Random, *,
             operation: int = 1, path_estimator: str = "event",
             learning_rate: float = .01, within_samples: int = 1,
             terminal_loss: TerminalLoss = terminal_negative_log_likelihood) -> dict[str, Any]:
        objective = HybridTerminalObjective(self.model, examples, self.ledger, terminal_loss)
        result = coupled_gradient(self.model, self.bank, objective, operation, self.kernel, rng,
                                  path_estimator=path_estimator, within_samples=within_samples)
        commit = self.transaction(result["parameter_gradients"], operation=operation,
                                   learning_rate=learning_rate)
        return {"gradient": result["record"], "transaction": commit}

    @torch.no_grad()
    def evaluate(self, examples: Sequence[ProgramExample]) -> dict[str, Any]:
        if not examples:
            raise ValueError("nonempty explicit evaluation set required")
        maps = self.bank.maps()
        signature = self.model.native_signature(maps, self.ledger)
        native_correct, soft_correct, rows = 0, 0, []
        for example in examples:
            self.ledger.add("feedback_calls")
            self.ledger.add("terminal_label_accesses")
            native = native_output(signature, example, self.ledger)
            probability = self.model.soft_probability(example, maps, self.ledger)
            soft = int(probability.argmax())
            native_correct += int(native == example.target)
            soft_correct += int(soft == example.target)
            rows.append({"example": asdict(example), "native": native, "soft": soft,
                         "probability": probability.cpu().tolist()})
        result = {"native_accuracy": native_correct / len(examples),
                  "soft_accuracy": soft_correct / len(examples), "rows": rows,
                  "maps_obtained_from_actual_lifts": maps,
                  "parameters_total": self.model.parameter_count(),
                  "soft_old_path_bound": self.model.soft_native_path_bound(
                      self.old_tokens, self.max_old_length, self.ledger)}
        self.ledger.emit("hybrid_evaluation", **result)
        return result

    def admit_if_successful(self, examples: Sequence[ProgramExample], operation: int = 1,
                            new_tokens: Sequence[int] = (1,)) -> bool:
        """Admit the primitive AND extend the protected task-binding alphabet.

        The new reference is the current acquired native mechanism, not a teacher
        table. Existing old-token behavior already passed closure against its
        earlier reference. New external generalization still requires held-out
        evidence; this contract preserves the actually acquired native program.
        """
        result = self.evaluate(examples)
        if result["native_accuracy"] != 1.0 or result["soft_accuracy"] != 1.0:
            return False
        expanded_tokens = tuple(sorted(set(self.old_tokens) | set(int(token) for token in new_tokens)))
        bound = self.model.soft_native_path_bound(expanded_tokens, self.max_old_length, self.ledger)
        if bound <= SOFT_NATIVE_MIN_MASS:
            self.ledger.emit("hybrid_admission_pending", reason="new native/soft confidence contract incomplete",
                             expanded_token_path_bound=bound)
            return False
        old_admitted = set(self.protected.admitted)
        old_receipts = copy.deepcopy(self.protected.receipts)
        old_reference, old_tokens = self.reference, self.old_tokens
        committed = False
        try:
            self.protected.admit(operation, self.model, self.family, self.executor)
            self.reference = self.model.native_signature(self.bank.maps(), self.ledger)
            self.old_tokens = expanded_tokens
            self.ledger.emit("hybrid_task_admission", operation=operation,
                             protected_tokens=expanded_tokens, path_bound=bound,
                             native_reference=self.reference)
            committed = True
            return True
        finally:
            if not committed:
                self.protected.admitted = old_admitted
                self.protected.receipts = old_receipts
                self.reference, self.old_tokens = old_reference, old_tokens

    def checkpoint(self) -> dict[str, Any]:
        """Actual parameters, binding reference, receipts and derived storage."""
        if self.protected.admitted and self.bank.semantic_license_hash != self.model.parameter_hash():
            raise NumericalFailure("cannot checkpoint untransacted old semantic-map reuse")
        cache = [{"key": list(key), "value": copy.deepcopy(value)}
                 for key, value in self.bank.cache.items()]
        reference_bytes = len(json.dumps(self.reference).encode())
        return {"version": "hybrid-loop-v1", "metadata": self.metadata,
                "configuration": {"slots": self.model.slots, "protection": self.protection,
                                  "sigma": self.kernel.sigma, "old_tokens": self.old_tokens,
                                  "max_old_length": self.max_old_length,
                                  "new_route_seeded": self.model.new_route_seeded},
                "state_dict": {key: value.detach().cpu().clone()
                               for key, value in self.model.state_dict().items()},
                "family": self.family.configuration(), "reference_native_contract": copy.deepcopy(self.reference),
                "protected_operations": sorted(self.protected.admitted),
                "admission_receipts": copy.deepcopy(self.protected.receipts),
                "derived_map_cache": cache, "cache_storage": self.bank.storage(),
                "map_namespace": self.bank.namespace,
                "semantic_license_hash": self.bank.semantic_license_hash,
                "reference_contract_bytes": reference_bytes,
                "parameters_total": self.model.parameter_count(),
                "buffer_scalars": sum(buffer.numel() for buffer in self.model.buffers()),
                "optimizer": "explicit SGD; no hidden optimizer tensors",
                "transactions": self.transactions, "costs": self.ledger.snapshot(),
                "forward_counts": dict(self.model.forward_counts),
                "parameter_hash": self.model.parameter_hash()}

    def save(self, path: Path) -> dict[str, Any]:
        if path.exists():
            raise FileExistsError("hybrid checkpoints require a NEW path")
        payload = self.checkpoint()
        torch.save(payload, path)
        self.ledger.add("hybrid_checkpoint_writes")
        return {"path": str(path), "bytes": path.stat().st_size,
                "parameter_hash": payload["parameter_hash"]}

    @classmethod
    def restore(cls, payload: Mapping[str, Any], ledger: CostLedger) -> "HybridLearningSession":
        session = cls(ledger, **payload["configuration"])
        session.model.load_state_dict(payload["state_dict"], strict=True)
        restored_family = AffineLoopFamily.from_configuration(payload["family"])
        session.family = restored_family
        session.bank.family = restored_family
        restored_namespace = hashlib.sha256(("fixed-cubic-z3-3z-u-base0-v1:"
                                             + json.dumps(restored_family.configuration(), sort_keys=True)).encode()).hexdigest()
        if restored_namespace != payload["map_namespace"]:
            raise NumericalFailure("restored map family namespace mismatch")
        session.bank.namespace = restored_namespace
        session.reference = copy.deepcopy(payload["reference_native_contract"])
        session.protected.admitted = set(int(value) for value in payload["protected_operations"])
        session.protected.receipts = {int(op): copy.deepcopy(value)
                                     for op, value in payload["admission_receipts"].items()}
        session.bank.cache = {tuple(entry["key"]): copy.deepcopy(entry["value"])
                              for entry in payload["derived_map_cache"]}
        session.transactions = int(payload["transactions"])
        session.bank.semantic_license_hash = payload["semantic_license_hash"]
        if session.model.parameter_hash() != payload["parameter_hash"]:
            raise NumericalFailure("hybrid actual NN checkpoint hash mismatch")
        if session.protected.admitted and session.bank.semantic_license_hash != session.model.parameter_hash():
            raise NumericalFailure("restored semantic license is not for these actual NN tensors")
        ledger.add("hybrid_checkpoint_restores")
        return session

