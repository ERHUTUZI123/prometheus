"""Bounded, ordinary LRU maintenance optimization for path-action moments.

The existing kinetic_moments sum is additive over unique active permutations.
Cache those individual contributions, then sum the CURRENT active set. This
changes neither the action, feedback, learned classes nor topological rule.
It stores no task labels and never expands a space of possible operations.

Root owns integration, source freeze and execution. Import runs no experiment.
MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 NOT PASSED; M3 ultimate.
"""
from collections import OrderedDict
import json
import time

import numpy as np


MAX_CAPACITY = 128
MAX_LATENT_STATES = 64
MAX_ACTIVE_CLASSES = 25  # bank<=24 plus identity skip
MAX_INPUT_CLASSES = 128
COEFFICIENT_SCALARS = 21  # c + b[4] + H[4,4]
COEFFICIENT_BYTES = COEFFICIENT_SCALARS * np.dtype(np.float64).itemsize


class PathMomentCache:
    """Per-class coefficients under the exact existing quadrature convention.

    ``get(classes)`` returns the old kinetic_moments coefficient interface.
    ``statistics()`` reports costs separately, including overhead and audits.
    Per cached entry: one read-only 21-float64 array and its permutation key.
    Keys also carry D and the fixed quadrature/geometry namespace. Memory
    reports count ndarray payload plus serialized key bytes, not Python RSS.
    """

    def __init__(self, capacity=MAX_CAPACITY):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= MAX_CAPACITY:
            raise ValueError("LRU capacity must be an integer in1..128")
        started = time.perf_counter()
        # Import the actual original mechanism; no alternate approximation.
        from train_monodromy_extension import kinetic_moments, QUADRATURE_PER_SEGMENT
        from continuous_monodromy import FREQUENCIES, BIASES, SPACING
        self.capacity = capacity
        self._original = kinetic_moments
        self.quadrature = int(QUADRATURE_PER_SEGMENT)
        if self.quadrature <= 0:
            raise ValueError("positive existing quadrature size required")
        self.constants = (
            "existing-smoothstep-exchange-sin2-shear-moments-v1",
            self.quadrature, float(SPACING),
            tuple(float(x) for x in FREQUENCIES),
            tuple(float(x) for x in BIASES))
        if len(self.constants[3]) != 4 or len(self.constants[4]) != 4:
            raise ValueError("this cache stores the existing four-coordinate objective")
        self._entries = OrderedDict()
        self._costs = dict(
            requests=0, unique_active_class_requests=0, hits=0, misses=0,
            evictions=0, original_single_class_calls=0,
            original_full_rebuild_equivalents=0,
            point_strands_reference_without_cache=0,
            point_strands_actually_computed_on_misses=0,
            point_strands_avoided_by_hits=0,
            moment_calc_seconds=0., lookup_and_validation_seconds=0.,
            aggregation_seconds=0., total_request_seconds=0.,
            audit_calls=0, audit_original_full_rebuild_calls=0,
            audit_original_point_strands=0, audit_original_seconds=0.,
            initialization_seconds=time.perf_counter()-started)
        # Every permutation of fixed D has the same serialized integer lengths;
        # D=64 bounds all supported keys for this fixed constants namespace.
        self._max_key_bytes = len(self._key_bytes((MAX_LATENT_STATES,
                                                  tuple(range(MAX_LATENT_STATES)),
                                                  self.constants)))

    @staticmethod
    def _key_bytes(key):
        return json.dumps(key, separators=(",", ":"), allow_nan=False).encode("utf8")

    @staticmethod
    def _normalize(classes):
        unique = OrderedDict()
        dimension = None
        for index, permutation in enumerate(classes):
            if index >= MAX_INPUT_CLASSES:
                raise ValueError("at most128 supplied class descriptions per request")
            array = np.asarray(permutation)
            if (array.ndim != 1 or array.dtype.kind not in "iu"
                    or not 1 <= len(array) <= MAX_LATENT_STATES):
                raise ValueError("require integer permutations on1..64 states")
            values = tuple(int(x) for x in array)
            if sorted(values) != list(range(len(values))):
                raise ValueError("active class is not a complete permutation")
            if dimension is None:
                dimension = len(values)
            if len(values) != dimension:
                raise ValueError("one active bank must use one common D")
            unique.setdefault(values, None)
            if len(unique) > MAX_ACTIVE_CLASSES:
                raise ValueError("active classes exceed bank24 plus identity")
        return tuple(unique)

    def _point_strands(self, permutation):
        # The original canonical adjacent-exchange selector has exactly the
        # permutation inversion count. Identity still has one time segment.
        inversions = sum(permutation[i] > permutation[j]
                         for i in range(len(permutation))
                         for j in range(i+1, len(permutation)))
        return self.quadrature * len(permutation) * max(1, inversions)

    def get(self, classes):
        """Aggregate CURRENT unique classes; no old inactive contribution leaks."""
        began = time.perf_counter()
        validation_began = time.perf_counter()
        classes = self._normalize(classes)
        self._costs["lookup_and_validation_seconds"] += time.perf_counter()-validation_began
        self._costs["requests"] += 1
        self._costs["original_full_rebuild_equivalents"] += 1
        self._costs["unique_active_class_requests"] += len(classes)
        aggregate = np.zeros(COEFFICIENT_SCALARS, dtype=np.float64)
        active_points = 0
        for permutation in classes:
            lookup_began = time.perf_counter()
            key = (len(permutation), permutation, self.constants)
            expected_points = self._point_strands(permutation)
            active_points += expected_points
            self._costs["point_strands_reference_without_cache"] += expected_points
            coefficients = self._entries.get(key)
            if coefficients is not None:
                self._costs["hits"] += 1
                self._costs["point_strands_avoided_by_hits"] += expected_points
                self._entries.move_to_end(key)
                self._costs["lookup_and_validation_seconds"] += time.perf_counter()-lookup_began
            else:
                self._costs["misses"] += 1
                self._costs["lookup_and_validation_seconds"] += time.perf_counter()-lookup_began
                compute_began = time.perf_counter()
                original = self._original([permutation])
                self._costs["moment_calc_seconds"] += time.perf_counter()-compute_began
                self._costs["original_single_class_calls"] += 1
                actual_points = int(original["quadrature_point_strands"])
                self._costs["point_strands_actually_computed_on_misses"] += actual_points
                if actual_points != expected_points:
                    raise RuntimeError("existing quadrature convention differs from cache point accounting")
                linear = np.asarray(original["linear"], dtype=np.float64)
                quadratic = np.asarray(original["quadratic"], dtype=np.float64)
                if linear.shape != (4,) or quadratic.shape != (4,4):
                    raise ValueError("existing moment shapes changed")
                coefficients = np.concatenate((np.asarray([original["constant"]], dtype=np.float64),
                                                linear, quadratic.reshape(-1)))
                if coefficients.shape != (COEFFICIENT_SCALARS,) or not np.isfinite(coefficients).all():
                    raise ValueError("finite c,b4,H4x4 coefficients required")
                coefficients.setflags(write=False)
                if len(self._entries) >= self.capacity:
                    self._entries.popitem(last=False)
                    self._costs["evictions"] += 1
                self._entries[key] = coefficients
            aggregation_began = time.perf_counter()
            aggregate += coefficients
            self._costs["aggregation_seconds"] += time.perf_counter()-aggregation_began
        c = float(aggregate[0])
        result = dict(constant=c, linear=aggregate[1:5].copy(),
                      quadratic=aggregate[5:].reshape(4,4).copy(),
                      normalizer=max(1., c), operation_classes=len(classes),
                      quadrature_point_strands=active_points,
                      quadrature_per_segment=self.quadrature)
        self._costs["total_request_seconds"] += time.perf_counter()-began
        return result

    def statistics(self):
        payload = sum(value.nbytes for value in self._entries.values())
        keys = sum(len(self._key_bytes(key)) for key in self._entries)
        return dict(**self._costs,
                    entries=len(self._entries), capacity=self.capacity,
                    coefficients_per_entry=COEFFICIENT_SCALARS,
                    numpy_arrays=len(self._entries), numpy_dtype="float64",
                    numpy_payload_bytes=payload, serialized_key_bytes=keys,
                    accounted_payload_and_keys_bytes=payload+keys,
                    accounted_payload_and_keys_upper_bound=self.capacity*(COEFFICIENT_BYTES+self._max_key_bytes),
                    maximum_serialized_key_bytes=self._max_key_bytes,
                    bounded_entries=True,
                    memory_scope="ndarray numeric payload plus serialized key bytes; excludes Python objects, ndarray headers, allocator overhead and RSS",
                    audit_work_separate_from_cache_savings=True,
                    counter_scope="cumulative since this cache instance; request sub-timers overlap total_request_seconds")

    def audit(self, classes, atol=1.e-9, rtol=2.e-12, raise_on_fail=True):
        """Pay for an original full rebuild and compare the complete objective.

        Call on the same ordered, deduplicated active classes as the learner.
        All audit work is reported separately; it is not free verification.
        """
        if not 0 < atol <= 1.e-8 or not 0 < rtol <= 1.e-10:
            raise ValueError("audit tolerances must remain strict and positive")
        classes = self._normalize(classes)
        cached = self.get(classes)
        began = time.perf_counter()
        original = self._original(classes)
        seconds = time.perf_counter()-began
        self._costs["audit_calls"] += 1
        self._costs["audit_original_full_rebuild_calls"] += 1
        self._costs["audit_original_point_strands"] += int(original["quadrature_point_strands"])
        self._costs["audit_original_seconds"] += seconds
        checks, differences = {}, {}
        for name in ("constant", "linear", "quadratic", "normalizer"):
            actual, expected = np.asarray(cached[name]), np.asarray(original[name])
            checks[name] = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
            differences[name] = float(np.max(np.abs(actual-expected)))
        checks["operation_classes"] = cached["operation_classes"] == original["operation_classes"]
        checks["quadrature_point_strands"] = cached["quadrature_point_strands"] == original["quadrature_point_strands"]
        report = dict(passed=all(checks.values()), checks=checks,
                      absolute_differences=differences, atol=atol, rtol=rtol,
                      original_audit_seconds=seconds,
                      original_audit_point_strands=int(original["quadrature_point_strands"]),
                      cached_coefficients=cached, original_coefficients=original,
                      scope="same original numeric quadrature; floating summation order can differ")
        if not report["passed"] and raise_on_fail:
            raise RuntimeError("cached action moments failed full original audit: "+str(differences))
        return report

