"""Bounded paired cost replay of ACTUALLY LOGGED geometry moment states.

No training, labels, path execution or reconstruction of missing old events.
For each geometry_moment_state event, independently execute the existing full
quadrature and a bounded per-class LRU. Alternate timing order by event index.
This is one diagnostic pass, not a statistical or whole-training benchmark.

Root owns source inspection/freeze and execution in the existing offline CPU
runner (external timeout<=120s). Import starts no experiment.
MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 NOT PASSED; M3 ultimate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


MAX_BYTES = 100 * 1024 * 1024
ATOL, RTOL = 1.e-9, 2.e-12


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite evidence field")
    return value


class Recorder:
    def __init__(self, out, seconds):
        out.mkdir(parents=True, exist_ok=False)
        self.out, self.started = out, time.perf_counter()
        self.deadline = self.started+seconds
        self.stream = (out/"records.jsonl").open("x", encoding="utf8")
        self.written = 0

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("bounded moment replay deadline reached")

    def emit(self, kind, **fields):
        row = json.dumps(plain(dict(kind=kind, wall=time.perf_counter()-self.started,
                                    **fields)), sort_keys=True, allow_nan=False)+"\n"
        self.written += len(row.encode("utf8"))
        if self.written > MAX_BYTES:
            raise RuntimeError("100MiB replay record cap")
        self.stream.write(row)
        self.stream.flush()

    def summary(self, result):
        text = json.dumps(plain(result), sort_keys=True, indent=2, allow_nan=False)+"\n"
        if len(text.encode("utf8")) > MAX_BYTES:
            raise RuntimeError("100MiB replay summary cap")
        with (self.out/"summary.json").open("x", encoding="utf8") as stream:
            stream.write(text)


def read_actual_schedule(path, max_events, recorder):
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("one completed saved records file<=100MiB required")
    before = sha(path)
    events, total_lines = [], 0
    with path.open(encoding="utf8") as stream:
        for line_number, line in enumerate(stream, 1):
            recorder.check()
            total_lines = line_number
            if len(line.encode("utf8")) > 2*1024*1024:
                raise ValueError("individual input event exceeds2MiB")
            record = json.loads(line)
            if record.get("kind") != "geometry_moment_state":
                continue
            if len(events) >= max_events:
                raise ValueError("actual schedule exceeds declared max-events; no silent truncation")
            state = record["moment_state"]
            classes = state["active_learned_classes"]
            if not isinstance(classes, list) or not 1 <= len(classes) <= 25:
                raise ValueError("actual active class list must contain1..25 permutations")
            dimension = int(state["latent_states"])
            if not 1 <= dimension <= 64:
                raise ValueError("actual latent state count exceeds64")
            normalized = []
            for permutation in classes:
                if (not isinstance(permutation, list) or len(permutation) != dimension
                        or any(isinstance(q, bool) or not isinstance(q, int) for q in permutation)
                        or sorted(permutation) != list(range(dimension))):
                    raise ValueError("logged actual class is not a common-D integer permutation")
                normalized.append(tuple(permutation))
            if len(set(normalized)) != len(normalized):
                raise ValueError("logged active classes must already be deduplicated")
            # Preserve line and class order exactly. No old curve state is used
            # to manufacture an event missing from the source trace.
            events.append(dict(source_line=line_number, model=record.get("model"),
                               stage=record.get("stage"), step=record.get("step"),
                               moment_state=state, classes=tuple(normalized)))
    if not events:
        raise ValueError("no geometry_moment_state records; old missing schedules are not reconstructed")
    if sha(path) != before:
        raise RuntimeError("source changed while reading actual schedule")
    return events, dict(path=str(path), sha256=before, bytes=path.stat().st_size,
                        total_lines=total_lines, actual_moment_events=len(events),
                        schedule_sha256=hashlib.sha256(json.dumps(
                            [dict(line=e["source_line"], model=e["model"], stage=e["stage"],
                                  step=e["step"], classes=e["classes"]) for e in events],
                            sort_keys=True, separators=(",", ":")).encode()).hexdigest())


def checked_coefficients(value, count, np):
    c, normalizer = float(value["constant"]), float(value["normalizer"])
    b = np.asarray(value["linear"], dtype=np.float64)
    h = np.asarray(value["quadratic"], dtype=np.float64)
    if (not math.isfinite(c) or not math.isfinite(normalizer) or normalizer < 1
            or b.shape != (4,) or h.shape != (4,4)
            or not np.isfinite(b).all() or not np.isfinite(h).all()
            or int(value.get("operation_classes", count)) != count
            or int(value["quadrature_point_strands"]) < 0):
        raise ValueError("finite saved c,b4,H4x4 and matching class count required")
    return dict(constant=c, linear=b, quadratic=h, normalizer=normalizer,
                operation_classes=count,
                quadrature_point_strands=int(value["quadrature_point_strands"]))


def compare(actual, reference, np):
    checks, errors = {}, {}
    for key in ("constant", "linear", "quadratic", "normalizer"):
        a, b = np.asarray(actual[key]), np.asarray(reference[key])
        checks[key] = bool(np.allclose(a, b, atol=ATOL, rtol=RTOL))
        errors[key] = float(np.max(np.abs(a-b)))
    for key in ("operation_classes", "quadrature_point_strands"):
        checks[key] = actual[key] == reference[key]

    # For |omega_i|<=8: delta f=dc+2 db^T omega+omega^T dH omega.
    # This row-sum bound permits nonsymmetric floating dH as well.
    def box_bound(ca, ba, ha, cb, bb, hb):
        dc, db, dh = ca-cb, ba-bb, ha-hb
        gradient_rows = 2*np.abs(db)+8*np.sum(np.abs(dh)+np.abs(dh.T), axis=1)
        return dict(omega_Linf_max=8.,
                    action_absolute_error_upper=float(abs(dc)+16*np.abs(db).sum()+64*np.abs(dh).sum()),
                    gradient_component_error_upper=gradient_rows,
                    gradient_Linf_error_upper=float(gradient_rows.max()))
    raw_bound = box_bound(actual["constant"], actual["linear"], actual["quadratic"],
                          reference["constant"], reference["linear"], reference["quadratic"])
    na, nr = actual["normalizer"], reference["normalizer"]
    normalized_bound = box_bound(actual["constant"]/na, actual["linear"]/na, actual["quadratic"]/na,
                                 reference["constant"]/nr, reference["linear"]/nr, reference["quadratic"]/nr)
    return dict(passed=all(checks.values()), checks=checks, absolute_errors=errors,
                raw_action_and_gradient_box_bound=raw_bound,
                normalized_action_and_gradient_box_bound=normalized_bound,
                bounds_scope="coefficient perturbation inequality evaluated in float64; not outward-rounded interval proof")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=100.)
    parser.add_argument("--max-events", type=int, default=512)
    args = parser.parse_args()
    if not 0 < args.seconds <= 110 or not 1 <= args.max_events <= 512 or args.out.exists():
        parser.error("new output, internal<=110s/external<=120s and1..512 actual events required")
    return args


def main():
    args = arguments()
    recorder = Recorder(args.out, args.seconds)
    status, failure, source, cache = "FAILED_WITH_PARTIAL_RECORDS", None, None, None
    totals = dict(events_compared=0, full_original_kernel_seconds=0.,
                  cached_kernel_seconds=0., coefficient_comparison_seconds=0.,
                  cache_statistics_report_seconds=0.,
                  full_original_point_strands=0, all_comparisons_passed=True,
                  max_normalized_gradient_box_error=0.,
                  max_raw_gradient_box_error=0., import_seconds=0., cache_init_seconds=0.)
    try:
        began = time.perf_counter()
        import numpy as np
        import path_moment_cache
        from path_moment_cache import PathMomentCache
        from train_monodromy_extension import kinetic_moments
        import continuous_monodromy
        totals["import_seconds"] = time.perf_counter()-began
        recorder.check()
        began = time.perf_counter()
        cache = PathMomentCache(capacity=128)
        totals["cache_init_seconds"] = time.perf_counter()-began
        events, source = read_actual_schedule(args.records, args.max_events, recorder)
        source_files = [Path(__file__), Path(path_moment_cache.__file__),
                        Path(kinetic_moments.__code__.co_filename), Path(continuous_monodromy.__file__)]
        recorder.emit("configuration", source=source, max_events=args.max_events,
                      code_sha256={str(p): sha(p) for p in source_files},
                      atol=ATOL, rtol=RTOL, cache_capacity=128,
                      new_training=False, labels_loaded=False, continuous_paths_executed=False,
                      reconstructed_missing_events=False,
                      order="even event original then cache; odd event cache then original",
                      cache_audit_not_called=True, import_seconds=totals["import_seconds"],
                      cache_init_seconds=totals["cache_init_seconds"],
                      PRIMARY_BASELINE="Transformer", M0="NOT_ASSESSED")
        for index, event in enumerate(events):
            recorder.check()
            classes = event["classes"]
            results, timings = {}, {}
            order = ("original", "cache") if index%2 == 0 else ("cache", "original")
            for name in order:
                recorder.check()
                began = time.perf_counter()
                results[name] = kinetic_moments(classes) if name == "original" else cache.get(classes)
                timings[name] = time.perf_counter()-began
                recorder.check()
            began = time.perf_counter()
            original = checked_coefficients(results["original"], len(classes), np)
            cached = checked_coefficients(results["cache"], len(classes), np)
            logged = checked_coefficients(event["moment_state"], len(classes), np)
            comparisons = dict(cache_vs_original=compare(cached, original, np),
                               original_vs_logged=compare(original, logged, np),
                               cache_vs_logged=compare(cached, logged, np))
            comparison_seconds = time.perf_counter()-began
            passed = all(x["passed"] for x in comparisons.values())
            began = time.perf_counter()
            stats = cache.statistics()
            stats_seconds = time.perf_counter()-began
            totals["events_compared"] += 1
            totals["full_original_kernel_seconds"] += timings["original"]
            totals["cached_kernel_seconds"] += timings["cache"]
            totals["coefficient_comparison_seconds"] += comparison_seconds
            totals["cache_statistics_report_seconds"] += stats_seconds
            totals["full_original_point_strands"] += original["quadrature_point_strands"]
            totals["all_comparisons_passed"] &= passed
            for comparison in comparisons.values():
                totals["max_normalized_gradient_box_error"] = max(totals["max_normalized_gradient_box_error"],
                    comparison["normalized_action_and_gradient_box_bound"]["gradient_Linf_error_upper"])
                totals["max_raw_gradient_box_error"] = max(totals["max_raw_gradient_box_error"],
                    comparison["raw_action_and_gradient_box_bound"]["gradient_Linf_error_upper"])
            recorder.emit("paired_actual_moment_event", event_index=index,
                          source_line=event["source_line"], model=event["model"],
                          stage=event["stage"], step=event["step"],
                          cumulative_logged_rebuilds=event["moment_state"].get("cumulative_moment_rebuilds"),
                          active_learned_classes=classes, execution_order=order,
                          kernel_seconds=timings, comparison_seconds=comparison_seconds,
                          cache_statistics_report_seconds=stats_seconds,
                          logged=logged, original=original, cached=cached,
                          comparisons=comparisons, cache_statistics_cumulative=stats)
            if not passed:
                raise RuntimeError(f"numerical coefficient audit failed at source line{event['source_line']}")
        if sha(args.records) != source["sha256"]:
            raise RuntimeError("actual completed source changed during replay")
        final_stats = cache.statistics()
        if (final_stats["point_strands_reference_without_cache"] != totals["full_original_point_strands"]
                or final_stats["point_strands_actually_computed_on_misses"]
                   + final_stats["point_strands_avoided_by_hits"] != totals["full_original_point_strands"]):
            raise RuntimeError("full-reference/cache/avoided point accounting differs")
        status = "COMPLETE_MATCHED_ACTUAL_SCHEDULE_REPLAY"
    except BaseException as error:
        failure = dict(exception=type(error).__name__, reason=str(error), traceback=traceback.format_exc())
        recorder.emit("failure", **failure)
        raise
    finally:
        stats = None if cache is None else cache.statistics()
        recorder.summary(dict(
            schema="actual-moment-schedule-paired-cost-replay-v1", status=status,
            MAIN="TOPOLOGY_USER_LOCKED", M0="NOT_ASSESSED", source=source,
            totals=totals, cache_statistics=stats, failure=failure,
            observed_kernel_ratio_original_over_cache=(None if not totals["cached_kernel_seconds"] else
                totals["full_original_kernel_seconds"]/totals["cached_kernel_seconds"]),
            actual_total_analysis_wall_seconds=time.perf_counter()-recorder.started,
            boundaries=[
                "Every event is an actual logged geometry_moment_state in original file order; missing old schedules are not reconstructed.",
                "Reference and cache execute independently on identical active classes; main cache is not warmed by an audit or original results.",
                "Moment coefficients, normalization and all-box gradient perturbation are checked against both original and logged values.",
                "Both reference and cached kernels were paid during this analysis; their ratio is a one-pass diagnostic, not measured end-to-end training acceleration.",
                "Imports/init, comparison, reporting, storage and cache eviction costs are separately recorded.",
                "No model, task labels, trajectory simulation or new learning evidence is produced.",
                "This is ordinary bounded maintenance caching, not a new topological mechanism or topology advantage claim."]))


if __name__ == "__main__":
    main()
