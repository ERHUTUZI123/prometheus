"""Bounded, in-memory homotopy-certified monodromy cache; no model changes.

For the ACTUAL endpoint-fixed vertical-shear family in continuous_monodromy,
Phi_(omega,t)(x,y)=(x,y+a(t)sum_l omega_l*tanh(v_l*x+b_l)) is a global
diffeomorphism, with inverse obtained by subtracting the same shear. Convex
interpolation between any two omega in[-8,8]^4 fixes both path endpoints and
never creates a collision. A fixed canonical braid skeleton therefore has the
same marked-cover monodromy throughout this family.

A MISS performs the original actual unordered-configuration lift. A HIT checks
the frozen family namespace and four bounded finite weights, then reuses that
measured map by the homotopy argument. It does NOT run the new trajectory and
does NOT make physical control, kinetic action or hardware energy free. Ordinary
direct computation may cache the same known permutation with NO geometric lift;
savings relative to repeated self-lifts are not an advantage over that control.

API: HomotopyLiftCache(capacity=128).get(permutation,omega,deadline)->(map,event).
Maps are backed by immutable bytes. Cache is reusable across model instances,
bounded by entry count AND packed payload bytes, and never persisted here.
MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 unpassed; M3 ultimate.
"""
import argparse
from collections import OrderedDict
import hashlib
import inspect
import json
import math
from pathlib import Path
import time

import numpy as np

import continuous_monodromy as _family


FAMILY_VERSION = "endpoint-fixed-global-vertical-shear-v1"
OMEGA_ABS_LIMIT = 8.0


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _constants():
    frequencies = np.asarray(_family.FREQUENCIES, dtype=np.float64)
    biases = np.asarray(_family.BIASES, dtype=np.float64)
    spacing = float(_family.SPACING)
    if (frequencies.shape != (4,) or biases.shape != (4,)
            or not np.isfinite(frequencies).all() or not np.isfinite(biases).all()
            or not math.isfinite(spacing) or spacing <= 0):
        raise ValueError("unsupported continuous family constants")
    return spacing, tuple(float(v) for v in frequencies), tuple(float(v) for v in biases)


def _runtime_token():
    return (id(_family.canonical_word), id(_family.canonical_word.__code__),
            id(_family.ConfigurationLoop), id(_family.ConfigurationLoop.__init__.__code__),
            id(_family.ConfigurationLoop.configuration.__code__),
            id(_family.ConfigurationLoop.certificate.__code__),
            id(_family.lift_sequence), id(_family.lift_sequence.__code__))


class HomotopyLiftCache:
    def __init__(self, capacity=128, max_payload_bytes=1024*1024):
        started = time.perf_counter()
        if not isinstance(capacity, int) or not 1 <= capacity <= 1024:
            raise ValueError("cache capacity must be1..1024 entries")
        if not isinstance(max_payload_bytes, int) or not 1024 <= max_payload_bytes <= 16*1024*1024:
            raise ValueError("packed payload limit must be1KiB..16MiB")
        self._capacity, self._max_payload_bytes = capacity, max_payload_bytes
        self._constant_signature, self._namespace_token = _constants(), _runtime_token()
        spacing, frequencies, biases = self._constant_signature
        descriptor = dict(version=FAMILY_VERSION, module="continuous_monodromy",
            compiler_source_sha256=_sha(Path(_family.__file__).read_bytes()),
            canonical_word_source_sha256=_sha(inspect.getsource(_family.canonical_word).encode()),
            ConfigurationLoop_source_sha256=_sha(inspect.getsource(_family.ConfigurationLoop).encode()),
            lift_sequence_source_sha256=_sha(inspect.getsource(_family.lift_sequence).encode()),
            cache_proof_source_sha256=_sha(Path(__file__).read_bytes()),
            spacing=spacing, frequencies=list(frequencies), biases=list(biases),
            basepoint_rule="spacing*(arange(D)-(D-1)/2)",
            omega_domain=dict(dimensions=4, finite=True, component_abs_max=OMEGA_ABS_LIMIT),
            proof="global vertical shear diffeomorphism, endpoint-fixed convex parameter homotopy",
            proof_scope="ideal real-arithmetic monodromy; first lift also has original floating diagnostics")
        self._family_json = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False)
        self._family_fingerprint = _sha(self._family_json.encode())
        self._entries = OrderedDict()
        self._payload_bytes = 0
        self._counts = dict(get_calls=0, hits=0, misses=0, cross_omega_hits=0,
            validation_failures=0, actual_lift_attempts=0, actual_lift_failures=0,
            actual_lifts=0, actual_lift_snapshots=0, actual_distance_matrix_entries=0,
            evictions=0, oversize_uncached_results=0,
            namespace_setup_seconds=time.perf_counter()-started,
            validation_and_key_seconds=0., actual_lift_seconds=0.,
            cache_management_seconds=0., get_seconds=0.,
            maximum_payload_bytes=0, maximum_entries=0)

    @property
    def capacity(self):
        return self._capacity

    @property
    def max_payload_bytes(self):
        return self._max_payload_bytes

    @property
    def family_fingerprint(self):
        return self._family_fingerprint

    def _validate(self, permutation, omega, deadline):
        if (_runtime_token() != self._namespace_token or _constants() != self._constant_signature):
            raise ValueError("continuous family namespace changed; a new cache namespace is required")
        weights = np.asarray(omega, dtype=np.float64)
        if (weights.shape != (4,) or not np.isfinite(weights).all()
                or np.max(np.abs(weights)) > OMEGA_ABS_LIMIT):
            raise ValueError("homotopy reuse requires four finite omega components in[-8,8]")
        raw = np.asarray(permutation)
        if raw.dtype.kind not in ("i", "u") or raw.ndim != 1 or not 1 <= len(raw) <= 64:
            raise ValueError("integer permutation dimension must be1..64")
        p = np.asarray(raw, dtype=np.int64)
        if not np.array_equal(np.sort(p), np.arange(len(p))):
            raise ValueError("invalid native permutation")
        if not math.isfinite(float(deadline)) or time.monotonic() >= float(deadline):
            raise TimeoutError("continuous cache external deadline already exhausted")
        spacing = self._constant_signature[0]
        anchors = tuple(float(x) for x in spacing*(np.arange(len(p))-(len(p)-1)/2))
        key = (self.family_fingerprint, len(p), anchors, tuple(int(q) for q in p))
        return p, weights, key

    @staticmethod
    def _map(entry):
        # A fresh ndarray header backed by immutable bytes prevents callers
        # changing the cached shape or making its underlying data writable.
        return np.frombuffer(entry["map_bytes"], dtype="<i8")

    def get(self, permutation, omega, deadline):
        began = time.perf_counter()
        self._counts["get_calls"] += 1
        try:
            validation_started = time.perf_counter()
            try:
                p, weights, key = self._validate(permutation, omega, deadline)
            except Exception:
                self._counts["validation_failures"] += 1
                raise
            finally:
                self._counts["validation_and_key_seconds"] += time.perf_counter()-validation_started
            management_started = time.perf_counter()
            if key in self._entries:
                entry = self._entries.pop(key)
                self._entries[key] = entry
                self._counts["hits"] += 1
                changed = tuple(float(w) for w in weights) != entry["origin_omega"]
                self._counts["cross_omega_hits"] += int(changed)
                metadata = json.loads(entry["certificate_json"])
                value = self._map(entry)
                self._counts["cache_management_seconds"] += time.perf_counter()-management_started
                return value, dict(newactual_lift=False, cache_hit=True, cross_omega_reuse=changed,
                    family_fingerprint=self.family_fingerprint, latent_states=len(p),
                    current_omega=weights.tolist(), certificate=metadata,
                    actual_lift_report=None, actual_lift_seconds=0., seconds=time.perf_counter()-began,
                    new_omega_trajectory_actually_executed=False,
                    interpretation="ideal monodromy reused by the checked endpoint-fixed homotopy family")
            self._counts["cache_management_seconds"] += time.perf_counter()-management_started
            self._counts["misses"] += 1
            self._counts["actual_lift_attempts"] += 1
            lifting_started = time.perf_counter()
            try:
                loop = _family.ConfigurationLoop(p, weights)
                actual = _family.lift_sequence([loop], float(deadline))
                observed = np.asarray(actual["monodromy"], dtype=np.int64)
                if not np.array_equal(observed, p):
                    raise RuntimeError("first actual lift failed learned-permutation faithfulness audit")
            except BaseException:
                self._counts["actual_lift_failures"] += 1
                raise
            finally:
                lift_seconds = time.perf_counter()-lifting_started
                self._counts["actual_lift_seconds"] += lift_seconds
            self._counts["actual_lifts"] += 1
            self._counts["actual_lift_snapshots"] += int(actual["snapshots"])
            self._counts["actual_distance_matrix_entries"] += int(actual["distance_matrix_entries_evaluated"])
            management_started = time.perf_counter()
            max_lipschitz = OMEGA_ABS_LIMIT*sum(abs(v) for v in self._constant_signature[1])
            certificate = dict(family_fingerprint=self.family_fingerprint,
                dimensions=len(p), fixed_base_anchors=list(key[2]), native_permutation=p.tolist(),
                actual_origin_omega=weights.tolist(), original_lift_certificate=loop.certificate(),
                origin_actual_lift_report=actual,
                uniform_box_shear_lipschitz_upper=max_lipschitz,
                uniform_box_pair_clearance_lower=self._constant_signature[0]/(1+max_lipschitz),
                homotopy="omega(lambda)=(1-lambda)*origin_omega+lambda*current_omega,0<=lambda<=1",
                fixed_basepoint=True, same_canonical_braid_skeleton=True,
                scope="ideal semantic cache; a cache hit does not numerically execute the new path")
            text = json.dumps(certificate, sort_keys=True, separators=(",", ":"), allow_nan=False)
            encoded = np.asarray(observed, dtype="<i8").tobytes()
            # Packed key + ndarray data + serialized certificate; Python object
            # overhead is separately bounded by capacity and D<=64.
            payload = len(encoded)+len(text.encode())+16*len(p)+len(self.family_fingerprint.encode())+40
            entry = dict(map_bytes=encoded, certificate_json=text,
                         origin_omega=tuple(float(w) for w in weights), payload_bytes=payload)
            stored = payload <= self.max_payload_bytes
            if stored:
                while self._entries and (len(self._entries) >= self.capacity
                                         or self._payload_bytes+payload > self.max_payload_bytes):
                    _, evicted = self._entries.popitem(last=False)
                    self._payload_bytes -= evicted["payload_bytes"]
                    self._counts["evictions"] += 1
                self._entries[key] = entry
                self._payload_bytes += payload
                self._counts["maximum_entries"] = max(self._counts["maximum_entries"], len(self._entries))
                self._counts["maximum_payload_bytes"] = max(self._counts["maximum_payload_bytes"], self._payload_bytes)
            else:
                self._counts["oversize_uncached_results"] += 1
            value = self._map(entry)
            self._counts["cache_management_seconds"] += time.perf_counter()-management_started
            return value, dict(newactual_lift=True, cache_hit=False, cross_omega_reuse=False,
                stored=stored, family_fingerprint=self.family_fingerprint, latent_states=len(p),
                current_omega=weights.tolist(), certificate=certificate, actual_lift_report=actual,
                actual_lift_seconds=lift_seconds, seconds=time.perf_counter()-began,
                new_omega_trajectory_actually_executed=True,
                interpretation="original actual configuration lift, then bounded in-memory receipt")
        finally:
            self._counts["get_seconds"] += time.perf_counter()-began

    def policy(self):
        """Serializable policy only: neither map payload nor warmness is restored."""
        return dict(kind="homotopy_certified_in_memory_lift_cache", capacity=self.capacity,
                    max_payload_bytes=self.max_payload_bytes, family_fingerprint=self.family_fingerprint,
                    family=json.loads(self._family_json), maps_persisted=False)

    def statistics(self):
        keys = [dict(family_fingerprint=key[0], dimensions=key[1],
                     base_anchors=list(key[2]), native_permutation=list(key[3]),
                     origin_omega=list(entry["origin_omega"]), payload_bytes=entry["payload_bytes"])
                for key, entry in self._entries.items()]
        return dict(policy=self.policy(), counts=dict(self._counts), entries=len(self._entries),
                    keys=keys, numdata=sum(len(entry["map_bytes"])//8 for entry in self._entries.values()),
                    map_data_bytes=sum(len(entry["map_bytes"]) for entry in self._entries.values()),
                    certificate_bytes=sum(len(entry["certificate_json"].encode()) for entry in self._entries.values()),
                    packed_payload_bytes=self._payload_bytes,
                    payload_accounting="packed numeric keys/maps and certificate JSON; Python overhead bounded separately by entries<=capacity,D<=64",
                    maps_persisted=False)


def main():
    """Optional bounded implementation diagnostic; the coordinator runs it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=110.)
    args = parser.parse_args()
    if args.out.exists() or not 0 < args.seconds <= 120:
        parser.error("new output and wall bound<=120seconds required")
    args.out.mkdir(parents=True)
    deadline = time.monotonic()+args.seconds
    cache = HomotopyLiftCache(capacity=2)
    records = []
    examples = [([1, 2, 3, 0], [0., 0., 0., 0.]),
                ([1, 2, 3, 0], [.4, -.2, .3, -.1]),
                ([1, 0, 2, 3], [0., 0., 0., 0.]),
                ([1, 2, 3, 4, 5, 6, 7, 0], [.2, .1, -.2, .1])]
    with (args.out/"raw.jsonl").open("x", encoding="utf8") as stream:
        for permutation, omega in examples:
            value, event = cache.get(permutation, omega, deadline)
            event["returned_map"] = value.tolist()
            event["returned_map_writeable"] = bool(value.flags.writeable)
            stream.write(json.dumps(event, sort_keys=True)+"\n")
            stream.flush()
            records.append(event)
        # Paid independent fresh trajectory at a previously cached new omega.
        permutation, omega = examples[1]
        fresh = _family.lift_sequence([_family.ConfigurationLoop(permutation, omega)], deadline)
        if fresh["monodromy"] != records[1]["returned_map"]:
            raise RuntimeError("independent fresh trajectory disagrees with homotopy cache")
        stream.write(json.dumps(dict(kind="paid_independent_new_omega_lift", result=fresh))+"\n")
    with (args.out/"summary.json").open("x", encoding="utf8") as stream:
        json.dump(dict(status="FINITE_CACHE_IMPLEMENTATION_DIAGNOSTIC_COMPLETED", statistics=cache.statistics(),
            independent_fresh_lift=fresh, family_hit_new_trajectory_executed=False,
            scientific_scope="cache implementation only, no learning/physical-energy/direct-control advantage"),
            stream, indent=2, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
