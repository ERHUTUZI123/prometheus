"""Bounded factor compiler + actual first-lift cache for D<=64 permutations.

The original continuous family, per-loop512 exchange cap, per-loop200000
snapshot cap and sequence8-loop cap are unchanged. A reduced adjacent word of
length<=2016 is split into at most4 factors, each with a legal canonical word
of length<=512. One actual lift_sequence executes ALL factors without resetting
marked sheets. Only its audited final map may be cached. Hits are ideal semantic
reuse under the endpoint-fixed shear homotopy, not fresh geometric execution.

This extends execution coverage; it is not a new acquisition law, topology
advantage, or permission/resource expansion. The caller supplies the original
external deadline. No training, labels, network, installation or import-time work.
MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 unpassed; M3 ultimate.
"""
import hashlib
import inspect
import json
import operator
from pathlib import Path
import time

import numpy as np

import continuous_monodromy as _family
from homotopy_lift_cache import HomotopyLiftCache, OMEGA_ABS_LIMIT


FACTOR_CACHE_VERSION = "bounded-actual-factored-monodromy-v1"
MAX_DIMENSION = 64
MAX_ADJACENT_PER_LOOP = 512
MAX_FACTORS = 4
MAX_TOTAL_ADJACENT = 2016


def _canonical_uncapped(permutation):
    """Original positive selector, without allocating beyond D(D-1)/2<=2016."""
    order = list(range(len(permutation)))
    word = []
    for i, item in enumerate(sorted(order, key=lambda j: permutation[j])):
        j = order.index(item)
        while j > i:
            order[j-1], order[j] = order[j], order[j-1]
            word.append(j-1)
            j -= 1
    return tuple(word)


def _chunk_permutation(dimension, word):
    order = list(range(dimension))
    for j in word:
        order[j], order[j+1] = order[j+1], order[j]
    destination = [0] * dimension
    for position, marked_identity in enumerate(order):
        destination[marked_identity] = position
    return tuple(destination)


def factor_permutation(permutation):
    """Return legal factor permutations in EXECUTION order, no numerical lift.

    A factor maps a currently occupied base position to its next base position;
    composition is therefore p_next[p_previous], not the reverse convention.
    Each contiguous subword of the original reduced word is reduced. Its own
    canonical selector consequently has at most512 exchanges; checked explicitly.
    """
    try:
        if any(isinstance(v, (bool, np.bool_)) for v in permutation):
            raise ValueError("Boolean permutation entries are not accepted")
        p = tuple(operator.index(v) for v in permutation)
    except TypeError as exc:
        raise ValueError("require an integer permutation") from exc
    d = len(p)
    if not 1 <= d <= MAX_DIMENSION or sorted(p) != list(range(d)):
        raise ValueError("require a permutation of1..64 latent positions")
    word = _canonical_uncapped(p)
    if len(word) > MAX_TOTAL_ADJACENT:
        raise RuntimeError("full reduced-word bound violated")
    chunks = tuple(word[i:i+MAX_ADJACENT_PER_LOOP]
                   for i in range(0, len(word), MAX_ADJACENT_PER_LOOP)) or ((),)
    factors = tuple(_chunk_permutation(d, w) for w in chunks)
    canonical_lengths = tuple(len(_canonical_uncapped(q)) for q in factors)
    if len(factors) > MAX_FACTORS or any(n > MAX_ADJACENT_PER_LOOP for n in canonical_lengths):
        raise RuntimeError("factor compiler exceeded unchanged per-loop cap")
    composed = tuple(range(d))
    for q in factors:
        composed = tuple(q[i] for i in composed)
    if composed != p:
        raise RuntimeError("factor composition does not equal requested permutation")
    return dict(dimension=d, permutations=factors, full_adjacent_exchanges=len(word),
                factor_count=len(factors), word_chunk_lengths=tuple(map(len, chunks)),
                canonical_chunk_lengths=canonical_lengths,
                composition_verified=True,
                composition_convention="last_factor[...second_factor[first_factor[position]]]",
                limits=dict(dimension=MAX_DIMENSION, per_loop_adjacent=MAX_ADJACENT_PER_LOOP,
                            factors=MAX_FACTORS, total_adjacent=MAX_TOTAL_ADJACENT))


def _factor_runtime_token():
    return tuple((id(f), id(f.__code__)) for f in
                 (factor_permutation, _canonical_uncapped, _chunk_permutation))


def factored_kinetic_moments(permutations):
    """Original path quadrature over EVERY factor, with no hidden free action.

    Each factor uses its own original endpoint-fixed envelope, exactly as in
    the actual concatenated lift. Integrals add; no averaging by factor count
    or deduplication of repeated factors is performed. The original normalizer
    max(1,c) is retained. This is a NEW path compiler, not an assertion that its
    continuous kinetic objective equals the unfactored path's objective.
    The root may wire this callback for the explicit factored-model flag only.
    """
    from train_monodromy_extension import kinetic_moments
    classes = []
    for permutation in permutations:
        if len(classes) >= 48:
            raise ValueError("factored active-class bank cap48 exceeded")
        classes.append(tuple(permutation))
    plans = [factor_permutation(p) for p in classes]
    if len({p['dimension'] for p in plans}) > 1:
        raise ValueError("factored kinetic classes require a common dimension")
    factors = [q for p in plans for q in p['permutations']]
    result = kinetic_moments(factors)
    result.update(operation_classes=len(classes), factor_loops=len(factors),
        full_adjacent_exchanges=sum(p['full_adjacent_exchanges'] for p in plans),
        factor_compiler=FACTOR_CACHE_VERSION,
        accounting='original midpoint quadrature charged for every executed factor; no factor deduplication')
    return result


class FactoredMonodromyCache(HomotopyLiftCache):
    """Original-cache subclass: identical get/statistics/capacity API, new namespace."""
    def __init__(self, capacity=128, max_payload_bytes=1024*1024):
        began = time.perf_counter()
        super().__init__(capacity=capacity, max_payload_bytes=max_payload_bytes)
        self._factor_namespace_token = _factor_runtime_token()
        descriptor = json.loads(self._family_json)
        descriptor.update(version=FACTOR_CACHE_VERSION,
            unfactored_family_fingerprint=self.family_fingerprint,
            factor_compiler_module_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            factor_compiler_function_source_sha256=hashlib.sha256(
                inspect.getsource(factor_permutation).encode()).hexdigest(),
            factorization=dict(max_dimension=MAX_DIMENSION,
                max_total_adjacent=MAX_TOTAL_ADJACENT,
                unchanged_per_loop_adjacent_cap=MAX_ADJACENT_PER_LOOP,
                max_factors=MAX_FACTORS,
                execution="one original lift_sequence over all factor loops; marked sheets never reset",
                first_lift_audit="actual composed monodromy must equal requested native permutation"))
        self._family_json = json.dumps(descriptor, sort_keys=True, separators=(',', ':'), allow_nan=False)
        self._family_fingerprint = hashlib.sha256(self._family_json.encode()).hexdigest()
        self._counts.update(factorizations=0, factorization_seconds=0.,
            actual_factor_loops=0, actual_full_adjacent_exchanges=0,
            max_factor_count=0, max_full_adjacent_exchanges=0,
            failed_lift_partial_snapshot_count_known=False)
        self._counts['namespace_setup_seconds'] = time.perf_counter()-began

    def _validate(self, permutation, omega, deadline):
        p, weights, key = super()._validate(permutation, omega, deadline)
        if _factor_runtime_token() != self._factor_namespace_token:
            raise ValueError("factor compiler runtime changed; require a new namespace")
        return p, weights, key

    def get(self, permutation, omega, deadline):
        began = time.perf_counter()
        self._counts['get_calls'] += 1
        try:
            validation_started = time.perf_counter()
            try:
                p, weights, key = self._validate(permutation, omega, deadline)
            except Exception:
                self._counts['validation_failures'] += 1
                raise
            finally:
                self._counts['validation_and_key_seconds'] += time.perf_counter()-validation_started
            management_started = time.perf_counter()
            if key in self._entries:
                entry = self._entries.pop(key)
                self._entries[key] = entry
                self._counts['hits'] += 1
                changed = tuple(float(w) for w in weights) != entry['origin_omega']
                self._counts['cross_omega_hits'] += int(changed)
                certificate = json.loads(entry['certificate_json'])
                value = self._map(entry)
                self._counts['cache_management_seconds'] += time.perf_counter()-management_started
                return value, dict(newactual_lift=False, cache_hit=True, cross_omega_reuse=changed,
                    family_fingerprint=self.family_fingerprint, latent_states=len(p),
                    current_omega=weights.tolist(), certificate=certificate,
                    actual_lift_report=None, actual_lift_seconds=0., seconds=time.perf_counter()-began,
                    new_omega_trajectory_actually_executed=False,
                    interpretation="ideal factored monodromy reused by checked endpoint-fixed homotopy")
            self._counts['cache_management_seconds'] += time.perf_counter()-management_started
            self._counts['misses'] += 1
            factor_started = time.perf_counter()
            plan = factor_permutation(p)
            self._counts['factorization_seconds'] += time.perf_counter()-factor_started
            self._counts['factorizations'] += 1
            self._counts['max_factor_count'] = max(self._counts['max_factor_count'], plan['factor_count'])
            self._counts['max_full_adjacent_exchanges'] = max(
                self._counts['max_full_adjacent_exchanges'], plan['full_adjacent_exchanges'])
            if time.monotonic() >= float(deadline):
                raise TimeoutError("factorization exhausted original external deadline")
            self._counts['actual_lift_attempts'] += 1
            lifting_started = time.perf_counter()
            try:
                loops = [_family.ConfigurationLoop(q, weights) for q in plan['permutations']]
                if any(len(loop.word) > MAX_ADJACENT_PER_LOOP for loop in loops):
                    raise RuntimeError("original canonical compiler exceeded factor bound")
                actual = _family.lift_sequence(loops, float(deadline))
                observed = np.asarray(actual['monodromy'], dtype=np.int64)
                if not np.array_equal(observed, p):
                    raise RuntimeError("actual multi-loop lift failed requested-permutation audit")
            except BaseException:
                self._counts['actual_lift_failures'] += 1
                raise
            finally:
                lift_seconds = time.perf_counter()-lifting_started
                self._counts['actual_lift_seconds'] += lift_seconds
            self._counts['actual_lifts'] += 1
            self._counts['actual_factor_loops'] += len(loops)
            self._counts['actual_full_adjacent_exchanges'] += plan['full_adjacent_exchanges']
            self._counts['actual_lift_snapshots'] += int(actual['snapshots'])
            self._counts['actual_distance_matrix_entries'] += int(actual['distance_matrix_entries_evaluated'])
            management_started = time.perf_counter()
            max_lipschitz = OMEGA_ABS_LIMIT*sum(abs(v) for v in self._constant_signature[1])
            certificate = dict(family_fingerprint=self.family_fingerprint,
                dimensions=len(p), fixed_base_anchors=list(key[2]), native_permutation=p.tolist(),
                actual_origin_omega=weights.tolist(), factorization=plan,
                original_lift_certificates=[loop.certificate() for loop in loops],
                origin_actual_lift_report=actual,
                uniform_box_shear_lipschitz_upper=max_lipschitz,
                uniform_box_pair_clearance_lower=self._constant_signature[0]/(1+max_lipschitz),
                homotopy="omega(lambda)=(1-lambda)*origin_omega+lambda*current_omega,0<=lambda<=1",
                fixed_basepoint=True, same_canonical_braid_skeleton=True,
                canonical_skeleton="fixed execution-ordered factors, each original canonical selector",
                marked_sheets_reset_between_factors=False,
                scope="ideal semantic cache; hit does not execute the new trajectory; all first-lift factors charged")
            text = json.dumps(certificate, sort_keys=True, separators=(',', ':'), allow_nan=False)
            encoded = np.asarray(observed, dtype='<i8').tobytes()
            payload = len(encoded)+len(text.encode())+16*len(p)+len(self.family_fingerprint.encode())+40
            entry = dict(map_bytes=encoded, certificate_json=text,
                         origin_omega=tuple(float(w) for w in weights), payload_bytes=payload)
            stored = payload <= self.max_payload_bytes
            if stored:
                while self._entries and (len(self._entries) >= self.capacity
                                         or self._payload_bytes+payload > self.max_payload_bytes):
                    _, evicted = self._entries.popitem(last=False)
                    self._payload_bytes -= evicted['payload_bytes']
                    self._counts['evictions'] += 1
                self._entries[key] = entry
                self._payload_bytes += payload
                self._counts['maximum_entries'] = max(self._counts['maximum_entries'], len(self._entries))
                self._counts['maximum_payload_bytes'] = max(self._counts['maximum_payload_bytes'], self._payload_bytes)
            else:
                self._counts['oversize_uncached_results'] += 1
            value = self._map(entry)
            self._counts['cache_management_seconds'] += time.perf_counter()-management_started
            return value, dict(newactual_lift=True, cache_hit=False, cross_omega_reuse=False,
                stored=stored, family_fingerprint=self.family_fingerprint, latent_states=len(p),
                current_omega=weights.tolist(), certificate=certificate, actual_lift_report=actual,
                actual_lift_seconds=lift_seconds, seconds=time.perf_counter()-began,
                new_omega_trajectory_actually_executed=True,
                interpretation="one actual configuration lift over1..4 legal factors, without sheet reset")
        finally:
            self._counts['get_seconds'] += time.perf_counter()-began

    def policy(self):
        policy = super().policy()
        policy['kind'] = 'factored_homotopy_certified_in_memory_lift_cache'
        policy['factorization'] = dict(max_dimension=MAX_DIMENSION,
            per_loop_adjacent_cap=MAX_ADJACENT_PER_LOOP,
            max_factors=MAX_FACTORS, max_total_adjacent=MAX_TOTAL_ADJACENT)
        return policy
