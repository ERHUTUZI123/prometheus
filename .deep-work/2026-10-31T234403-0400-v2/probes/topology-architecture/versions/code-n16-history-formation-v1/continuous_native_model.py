"""Primary native learner with a trainable continuous monodromy execution layer.

Instantiate from stage0. Soft terminal learning, materialization, workspace growth
and learned old-behavior guards are inherited unchanged. Native hard evaluation
ACTUALLY lifts unordered configurations to obtain its maps, then executes the
learned encoder, call slots and decoder using those lift-derived maps. Exact
cache reuse is explicit and charged separately from cold geometric realization.

The four shared path parameters receive the gradient of continuous kinetic
action; the terminal likelihood is invariant to them in this protected family.
MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 NOT PASSED; M3 ultimate.
No import-time experiment, SSH, subprocess or network access.
"""
import copy
import time

import numpy as np
import torch
from torch import nn

from continuous_monodromy import (BIASES, FREQUENCIES, SPACING,
                                  ConfigurationLoop, lift_sequence)
from semantic_promotion_model import SemanticPromotionModel
from topology_model import assignment
from train_monodromy_extension import (PATH_WEIGHT,
                                       QUADRATURE_PER_SEGMENT, kinetic_moments)


ALPHABETS = {0: (1, 2, 3), 1: (1, 2, 3, 4, 5), 2: (1, 2, 3, 4, 5, 6)}


def _new_costs():
    return dict(geometry_regularizer_calls=0, geometry_class_graph_extractions=0,
        geometry_class_bank_assignments=0, geometry_class_check_seconds=0.,
        geometry_moment_rebuilds=0, geometry_quadrature_point_strands=0,
        geometry_moment_seconds=0., geometry_torch_objective_seconds=0.,
        hard_eval_calls=0, hard_eval_total_seconds=0., hard_bank_assignment_batches=0,
        hard_bank_hungarian_assignments=0, hard_bank_assignment_seconds=0.,
        hard_cache_signature_checks=0, hard_cache_signature_seconds=0.,
        hard_cache_generations=0, hard_cache_class_hits=0,
        cold_lift_attempts=0, cold_lift_failures=0, cold_realized_classes=0,
        cold_configuration_snapshots=0, cold_distance_matrix_entries=0,
        cold_realization_seconds=0., warm_native_execution_seconds=0.,
        warm_native_primitive_calls=0, map_cache_bytes_current=0,
        map_cache_bytes_peak=0, cache_restore_invalidations=0)


class ContinuousNativeModel(SemanticPromotionModel):
    geometry_loss_weight = PATH_WEIGHT

    def __init__(self, *args, initial_omega=(0., 0., 0., 0.),
                 continuous_lift_seconds=110., **kwargs):
        super().__init__(*args, **kwargs)
        value = self.encoder.new_tensor(initial_omega)
        if value.shape != (4,) or not torch.isfinite(value).all() or value.abs().max().item() > 8:
            raise ValueError("four finite shared path weights in[-8,8] required")
        if not 0 < float(continuous_lift_seconds) <= 120:
            raise ValueError("continuous hard-call cold realization cap must be<=120seconds")
        self.path_omega = nn.Parameter(value)
        self._initial_path_omega = tuple(float(w) for w in initial_omega)
        self.continuous_lift_seconds = float(continuous_lift_seconds)
        self._geometry_deadline = None
        self._geometry_costs = _new_costs()
        self._moment_signature = self._moments = None
        self._moment_trace_signature = None
        self._moment_linear = self._moment_quadratic = None
        self._bank_parameter_signature = self._hard_bank = None
        self._lift_cache_signature = None
        self._lifted_maps = {}
        self._last_realizations = []

    def set_stage(self, stage, new_token_ids=None):
        super().set_stage(stage, new_token_ids=new_token_ids)
        if int(stage) not in ALPHABETS:
            raise ValueError("continuous learner uses the same three public task alphabets")

    def set_geometry_deadline(self, monotonic_deadline):
        """Optional parent wall budget, expressed in time.monotonic() coordinates."""
        self._geometry_deadline = None if monotonic_deadline is None else float(monotonic_deadline)

    def _invalidate_derived_caches(self):
        self._moment_signature = self._moments = None
        self._moment_linear = self._moment_quadratic = None
        self._bank_parameter_signature = self._hard_bank = None
        self._lift_cache_signature = None
        self._lifted_maps = {}
        self._last_realizations = []
        self._geometry_costs["map_cache_bytes_current"] = 0

    def geometry_regularizer(self):
        """Normalized quadratic action, differentiable through shared path_omega.

        Moment quadrature is refreshed only when CURRENT learned active classes
        change. Their hard class extraction is charged at every call. No target
        or teacher arithmetic selects the active classes.
        """
        self._geometry_costs["geometry_regularizer_calls"] += 1
        if not torch.isfinite(self.path_omega).all() or self.path_omega.abs().max().item() > 8:
            raise ValueError("shared path parameters exceed finite numerical execution envelope")
        start = time.perf_counter()
        graph = self.graph(int(self.stage), ALPHABETS[int(self.stage)])
        active = sorted(set(int(k) for k in graph["routes"].ravel()))
        classes = tuple(sorted(set(tuple(int(q) for q in graph["permutations"][k]) for k in active)))
        signature = int(self.d), classes
        self._geometry_costs["geometry_class_graph_extractions"] += 1
        self._geometry_costs["geometry_class_bank_assignments"] += self.n_ops
        self._geometry_costs["geometry_class_check_seconds"] += time.perf_counter()-start
        if signature != self._moment_signature:
            start = time.perf_counter()
            self._moments = kinetic_moments(classes)
            self._moment_signature = signature
            self._moment_linear = self.path_omega.new_tensor(self._moments["linear"])
            self._moment_quadratic = self.path_omega.new_tensor(self._moments["quadratic"])
            self._geometry_costs["geometry_moment_rebuilds"] += 1
            self._geometry_costs["geometry_quadrature_point_strands"] += self._moments["quadrature_point_strands"]
            self._geometry_costs["geometry_moment_seconds"] += time.perf_counter()-start
        elif (self._moment_linear.device != self.path_omega.device
              or self._moment_linear.dtype != self.path_omega.dtype):
            self._moment_linear = self.path_omega.new_tensor(self._moments["linear"])
            self._moment_quadratic = self.path_omega.new_tensor(self._moments["quadratic"])
        start = time.perf_counter()
        omega = self.path_omega
        raw = (self._moments["constant"] + 2*torch.dot(self._moment_linear, omega)
               + omega @ self._moment_quadratic @ omega)
        normalized = raw/self._moments["normalizer"]
        self._geometry_costs["geometry_torch_objective_seconds"] += time.perf_counter()-start
        return normalized

    @torch.no_grad()
    def _actual_bank(self):
        # Tensor mutation versions invalidate this fast check. Actual matching
        # classes, rather than versions, are part of the subsequent lift key.
        parameter = self.operations
        signature = (id(parameter), parameter.data_ptr(), int(parameter._version),
                     tuple(parameter.shape), str(parameter.device), str(parameter.dtype), int(self.d))
        if signature != self._bank_parameter_signature:
            started = time.perf_counter()
            self._hard_bank = [np.arange(self.d, dtype=np.int64)] + [assignment(w) for w in parameter]
            self._bank_parameter_signature = signature
            self._geometry_costs["hard_bank_assignment_batches"] += 1
            self._geometry_costs["hard_bank_hungarian_assignments"] += self.n_ops
            self._geometry_costs["hard_bank_assignment_seconds"] += time.perf_counter()-started
        return self._hard_bank

    @torch.no_grad()
    def _prepare_lift_cache(self):
        start = time.perf_counter()
        bank = self._actual_bank()
        omega = self.path_omega.detach().cpu().double().numpy()
        signature = (int(self.d), tuple(tuple(int(q) for q in p) for p in bank), omega.tobytes())
        self._geometry_costs["hard_cache_signature_checks"] += 1
        if signature != self._lift_cache_signature:
            self._lift_cache_signature = signature
            self._lifted_maps, self._last_realizations = {}, []
            self._geometry_costs["map_cache_bytes_current"] = 0
            self._geometry_costs["hard_cache_generations"] += 1
        self._geometry_costs["hard_cache_signature_seconds"] += time.perf_counter()-start
        return bank, omega

    @torch.no_grad()
    def _lift_map(self, permutation, omega, deadline):
        key = tuple(int(q) for q in permutation)
        if key in self._lifted_maps:
            self._geometry_costs["hard_cache_class_hits"] += 1
            return self._lifted_maps[key]
        started = time.perf_counter()
        self._geometry_costs["cold_lift_attempts"] += 1
        try:
            loop = ConfigurationLoop(permutation, omega)
            result = lift_sequence([loop], deadline)
            actual = np.asarray(result["monodromy"], dtype=np.int64)
            if not np.array_equal(actual, permutation):
                raise RuntimeError("actual continuous lift differs from learned native operation")
            self._lifted_maps[key] = actual
            self._geometry_costs["cold_realized_classes"] += 1
            self._geometry_costs["cold_configuration_snapshots"] += int(result["snapshots"])
            self._geometry_costs["cold_distance_matrix_entries"] += int(result["distance_matrix_entries_evaluated"])
            self._last_realizations.append(dict(permutation=list(key), certificate=loop.certificate(), **result))
            count = sum(value.nbytes for value in self._lifted_maps.values())
            self._geometry_costs["map_cache_bytes_current"] = count
            self._geometry_costs["map_cache_bytes_peak"] = max(self._geometry_costs["map_cache_bytes_peak"], count)
            return actual
        except BaseException:
            self._geometry_costs["cold_lift_failures"] += 1
            raise
        finally:
            self._geometry_costs["cold_realization_seconds"] += time.perf_counter()-started

    @torch.no_grad()
    def hard_eval(self, tokens, starts, task_ids):
        hard_started = time.perf_counter()
        if tokens.ndim != 2 or tokens.shape[0] > 128 or tokens.shape[1] > 18:
            raise ValueError("native evaluation shape/process bound exceeded")
        self._geometry_costs["hard_eval_calls"] += 1
        deadline = time.monotonic()+self.continuous_lift_seconds
        if self._geometry_deadline is not None:
            deadline = min(deadline, self._geometry_deadline)
        bank, omega = self._prepare_lift_cache()
        enc = self.encoder.argmax(-1).cpu().numpy()
        dec = self.decoder.argmax(-1).cpu().numpy()
        predictions = torch.empty_like(starts)
        for task in task_ids.unique().tolist():
            selected = task_ids == task
            words = tokens[selected].cpu().numpy()
            symbols = sorted(int(a) for a in np.unique(words) if a)
            if any(not 1 <= a <= self.vocab for a in symbols):
                raise ValueError("evaluation symbol outside learned public input alphabet")
            if symbols:
                tok = torch.as_tensor([symbols], dtype=torch.long, device=self.encoder.device)
                tid = torch.as_tensor([task], dtype=torch.long, device=self.encoder.device)
                routes = self.calls(tok, tid)[0].argmax(-1).cpu().numpy()
                calls = {a: [int(k) for k in route] for a, route in zip(symbols, routes)}
                used = sorted(set(k for route in calls.values() for k in route))
                maps = {k: self._lift_map(bank[k], omega, deadline) for k in used}
            else:
                calls, maps = {}, {}
            # Geometry cold work is excluded from warm cached inference time.
            warm_start = time.perf_counter()
            q = enc[starts[selected].cpu().numpy()].copy()
            uses = 0
            for column in range(words.shape[1]):
                for a in np.unique(words[:, column]):
                    if a:
                        mask = words[:, column] == a
                        for k in calls[int(a)]:
                            q[mask] = maps[k][q[mask]]
                            uses += int(np.count_nonzero(mask))
            predictions[selected] = torch.as_tensor(dec[q], device=starts.device)
            self._geometry_costs["warm_native_execution_seconds"] += time.perf_counter()-warm_start
            self._geometry_costs["warm_native_primitive_calls"] += uses
        logits = self.encoder.new_full((len(starts), self.n), -30.)
        result = logits.scatter_(1, predictions[:, None], 0.)
        self._geometry_costs["hard_eval_total_seconds"] += time.perf_counter()-hard_started
        return result

    def geometry_resource_report(self):
        moments = None if self._moments is None else dict(
            constant=float(self._moments["constant"]), linear=self._moments["linear"].tolist(),
            quadratic=self._moments["quadratic"].tolist(),
            normalizer=float(self._moments["normalizer"]),
            quadrature_point_strands=int(self._moments["quadrature_point_strands"]),
            active_learned_classes=[list(p) for p in self._moment_signature[1]])
        return dict(costs=copy.deepcopy(self._geometry_costs),
            current_omega=self.path_omega.detach().cpu().tolist(),
            current_path_omega_gradient=(None if self.path_omega.grad is None
                                         else self.path_omega.grad.detach().cpu().tolist()),
            current_latent_states=int(self.d), bank_size=int(self.n_ops),
            moments=moments, current_cache_realizations=copy.deepcopy(self._last_realizations),
            execution="cold unordered-configuration lifts; exact cached maps for warm learned calls",
            cache_signature="D + ALL actual assigned bank permutations + exact omega values",
            partial_failed_lift_snapshot_work_known=False,
            geometry_gradient_source="normalized kinetic action; terminal likelihood is invariant to this shear",
            timings="perf_counter; torch objective launch time may exclude asynchronous GPU completion")

    def geometry_moment_trace(self):
        """Log every actual moment-class change for same-input cost replay."""
        if self._moment_signature is None or self._moment_signature == self._moment_trace_signature:
            return None
        self._moment_trace_signature = self._moment_signature
        return dict(latent_states=int(self.d), stage=int(self.stage),
                    active_learned_classes=[list(p) for p in self._moment_signature[1]],
                    constant=float(self._moments['constant']),
                    linear=self._moments['linear'].tolist(),
                    quadratic=self._moments['quadratic'].tolist(),
                    normalizer=float(self._moments['normalizer']),
                    quadrature_point_strands=int(self._moments['quadrature_point_strands']),
                    source='actual pre-update learned native classes; no teacher data',
                    cumulative_moment_rebuilds=self._geometry_costs['geometry_moment_rebuilds'])

    def extra_state_dict(self):
        extra = super().extra_state_dict()
        extra["continuous_monodromy"] = dict(
            kind="shared_endpoint_fixed_nonlinear_vertical_shear", state_key="path_omega",
            dimensions=4, frequencies=FREQUENCIES.tolist(), biases=BIASES.tolist(), spacing=SPACING,
            initial_omega=list(self._initial_path_omega),
            kinetic_weight=PATH_WEIGHT, quadrature_per_segment=QUADRATURE_PER_SEGMENT,
            continuous_lift_seconds=self.continuous_lift_seconds,
            checkpoint_loader="continuous_native_model.ContinuousNativeModel",
            native_execution="actual lift-derived cached maps from stage0",
            resource_report=self.geometry_resource_report())
        return extra

    def load_extra_state_dict(self, extra):
        super().load_extra_state_dict(extra)
        metadata = extra.get("continuous_monodromy")
        if metadata:
            if (int(metadata.get("dimensions", -1)) != 4
                    or not np.array_equal(np.asarray(metadata.get("frequencies")), FREQUENCIES)
                    or not np.array_equal(np.asarray(metadata.get("biases")), BIASES)
                    or float(metadata.get("spacing", -1)) != SPACING):
                raise ValueError("checkpoint continuous substrate differs from current model")
            cap = float(metadata.get("continuous_lift_seconds", self.continuous_lift_seconds))
            if not 0 < cap <= 120:
                raise ValueError("saved continuous execution wall cap invalid")
            self.continuous_lift_seconds = cap
            self._initial_path_omega = tuple(float(w) for w in metadata.get("initial_omega", self._initial_path_omega))
            saved_costs = metadata.get("resource_report", {}).get("costs", {})
            for name in self._geometry_costs:
                if name in saved_costs:
                    self._geometry_costs[name] = saved_costs[name]
        self._invalidate_derived_caches()
        self._geometry_costs["cache_restore_invalidations"] += 1

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Dynamic bank resizing remains inherited. path_omega is an ordinary
        # Parameter and is restored by the same real neural state_dict.
        if "path_omega" in state_dict:
            value = state_dict["path_omega"]
            if tuple(value.shape) != (4,) or not torch.isfinite(value).all() or value.abs().max().item() > 8:
                raise ValueError("saved shared geometry weights invalid")
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self._invalidate_derived_caches()
        return result


ContinuousNativeExecutor = ContinuousNativeModel
