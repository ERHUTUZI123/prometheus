"""Measure inference costs of OWN saved architectures; never trains a model.

Root must inspect/freeze this file and launch it in the existing bounded CUDA
runner. This file is not a sandbox. Trusted campaign checkpoints alone may be
loaded with weights_only=False. Only tokens/starts/task_ids are read from NPZ;
no target labels, teacher functions or new observations enter the benchmark.

MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 NOT PASSED; M3 ultimate.
Compiled CPU graph execution and actual neural execution are separate modes.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from train import NEW_TOKENS, make_model


MAX_FILE_BYTES = 100 * 1024 * 1024
INPUT_KEYS = ("tokens", "starts", "task_ids")
MODELS = ("topo", "transformer", "replay", "adapter")
SPECS = ((7, "iid"), (12, "ood"), (16, "length16"))


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite evidence value")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_hash(arrays):
    digest = hashlib.sha256()
    for key in INPUT_KEYS:
        array = np.asarray(arrays[key], dtype="<i8", order="C")
        digest.update(key.encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape)).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def tensor_bytes(value):
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def numeric_payload_bytes(value, seen=None):
    """Numeric payload only; neither Python-object nor allocator overhead."""
    seen = set() if seen is None else seen
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        if id(value) in seen:
            return 0
        seen.add(id(value))
        return tensor_bytes(value) if torch.is_tensor(value) else int(value.nbytes)
    if isinstance(value, dict):
        return sum(numeric_payload_bytes(v, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(numeric_payload_bytes(v, seen) for v in value)
    return 0


def state_hash(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


class Recorder:
    def __init__(self, out, seconds):
        out.mkdir(parents=True, exist_ok=False)
        self.out, self.started = out, time.perf_counter()
        self.deadline = self.started + seconds
        self.stream = (out / "records.jsonl").open("x", encoding="utf8")
        self.written = 0

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise TimeoutError("bounded inference benchmark deadline reached")

    def emit(self, kind, **fields):
        row = json.dumps(plain(dict(kind=kind, wall=time.perf_counter()-self.started,
                                    **fields)), sort_keys=True, allow_nan=False) + "\n"
        self.written += len(row.encode("utf8"))
        if self.written > MAX_FILE_BYTES:
            raise RuntimeError("100MiB raw record cap")
        self.stream.write(row)
        self.stream.flush()

    def summary(self, result):
        encoded = json.dumps(plain(result), sort_keys=True, indent=2,
                             allow_nan=False) + "\n"
        if len(encoded.encode("utf8")) > MAX_FILE_BYTES:
            raise RuntimeError("100MiB summary cap")
        with (self.out / "summary.json").open("x", encoding="utf8") as stream:
            stream.write(encoded)


def synchronize(device):
    torch.cuda.synchronize(device)


def load_inputs(folder, recorder):
    result = {}
    for cap, split in SPECS:
        recorder.check()
        path = folder / f"task2-{split}.npz"
        if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("missing or oversized saved input NPZ")
        # Deliberately never access archive['targets'] or dataset generators.
        with np.load(path, allow_pickle=False) as archive:
            raw = {k: np.asarray(archive[k], dtype=np.int64) for k in INPUT_KEYS}
        count = len(raw["starts"])
        if (not 64 <= count <= 16384 or raw["tokens"].shape != (count, 18)
                or raw["starts"].shape != (count,)
                or raw["task_ids"].shape != (count,)
                or np.any(raw["task_ids"] != 2)
                or np.any(raw["tokens"] < 0) or np.any(raw["tokens"] > 6)
                or np.any(raw["starts"] < 0) or np.any(raw["starts"] >= 8)):
            raise ValueError("input differs from saved N8 task2 dimensions")
        lengths = np.count_nonzero(raw["tokens"], axis=1)
        expected_padding = np.arange(18)[None, :] >= lengths[:, None]
        if not np.array_equal(raw["tokens"] == 0, expected_padding):
            raise ValueError("only saved left-packed PAD convention is supported")
        eligible = np.flatnonzero(lengths >= cap)
        if len(eligible) < 64:
            raise ValueError(f"need 64 actual rows with length >= {cap}; got {len(eligible)}")
        for batch in (1, 64):
            selected = eligible[:batch]
            arrays = dict(tokens=np.ascontiguousarray(raw["tokens"][selected, :cap]),
                          starts=np.ascontiguousarray(raw["starts"][selected]),
                          task_ids=np.ascontiguousarray(raw["task_ids"][selected]))
            meta = dict(batch=batch, length_cap=cap, source_split=split,
                        source_path=str(path), source_file_sha256=sha256(path),
                        row_indices=selected.tolist(), source_lengths=lengths[selected].tolist(),
                        token_shape=list(arrays["tokens"].shape),
                        labels_loaded=False, input_sha256=input_hash(arrays),
                        selection="first actual rows with enough non-PAD tokens; common prefix truncated to cap",
                        actual_nonpad_tokens=int(np.count_nonzero(arrays["tokens"])))
            result[(batch, cap)] = (arrays, meta)
            recorder.emit("benchmark_input", **meta)
    return result


def restore_model(name, path, device, recorder, resource_rows):
    recorder.check()
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("missing or oversized TRUSTED campaign checkpoint")
    began = time.perf_counter()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    read_seconds = time.perf_counter() - began
    if checkpoint.get("schema") != "terminal-architecture-checkpoint-v1":
        raise ValueError("checkpoint schema mismatch")
    if checkpoint.get("model_name") != name or int(checkpoint.get("stage", -1)) != 2:
        raise ValueError("need final task2 checkpoint of the declared arm")
    saved_args = checkpoint["args"]
    if int(saved_args["scale"]) != 8 or int(saved_args["seed"]) != 0:
        raise ValueError("this fixed comparison uses actual N8 seed0 checkpoints")
    args = SimpleNamespace(**saved_args)
    state, extra = checkpoint["model"], checkpoint.get("model_extra", {})
    expected_hash = state_hash(state)
    began = time.perf_counter()
    latent_factor = int(state["encoder"].shape[1] // args.scale) if name == "topo" else None
    continuous = name == "topo" and "path_omega" in state
    if continuous:
        from train_monodromy_extension import extended_model
        model = extended_model(checkpoint, torch.device("cpu"))
        restoration = "train_monodromy_extension.extended_model"
    else:
        model = make_model(name, args, restored_latent_factor=latent_factor)
        restoration = "train.make_model"
    # The saved initialized-task set prevents reinitializing trained adapters.
    masks = checkpoint.get("masks_and_initialized_tasks", {})
    if "_initialized_tasks" in masks:
        model._initialized_tasks = masks["_initialized_tasks"]
    if callable(getattr(model, "set_stage", None)):
        model.set_stage(int(checkpoint["stage"]),
                        new_token_ids=NEW_TOKENS[int(checkpoint["stage"])])
    model.load_state_dict(state, strict=True)
    loader = getattr(model, "load_extra_state_dict", None)
    if callable(loader):
        loader(extra)
    else:
        for key, value in extra.items():
            setattr(model, key, value)
    for key, value in masks.items():
        setattr(model, key, value)
    if state_hash(model.state_dict()) != expected_hash:
        raise RuntimeError("restoration changed saved learned state")
    construct_seconds = time.perf_counter()-began
    began = time.perf_counter()
    model = model.to(device).eval()
    synchronize(device)
    transfer_seconds = time.perf_counter()-began
    counts = model.parameter_counts() if callable(getattr(model, "parameter_counts", None)) else dict(
        total=sum(p.numel() for p in model.parameters()),
        requires_grad=sum(p.numel() for p in model.parameters() if p.requires_grad))
    row = next((r for r in resource_rows if r["model"] == name and r["stage"] == 2), None)
    resources = dict(
        checkpoint=str(path), checkpoint_sha256=sha256(path),
        checkpoint_bytes=path.stat().st_size, checkpoint_step=int(checkpoint["step"]),
        schema=checkpoint["schema"], original_saved_args=saved_args,
        restoration=restoration, continuous_path_state_restored=continuous,
        path_omega=None if not continuous else model.path_omega.detach().cpu().tolist(),
        parameters=counts,
        model_parameter_tensor_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        model_state_dict_tensor_bytes=tensor_bytes(model.state_dict()),
        saved_optimizer_tensor_bytes=tensor_bytes(checkpoint.get("optimizer", {})),
        optimizer_instantiated_for_inference=False,
        reference_array_bytes=numeric_payload_bytes(extra.get("reference", [])),
        total_extra_numeric_payload_bytes=numeric_payload_bytes(extra),
        saved_mask_and_snapshot_tensor_bytes=tensor_bytes(masks),
        replay_memory_tensor_bytes=None if row is None else row.get("replay_memory_tensor_bytes"),
        replay_memory_source=None if row is None else row["summary_path"],
        state_sha256_before=expected_hash,
        checkpoint_read_seconds=read_seconds,
        cpu_construct_restore_seconds=construct_seconds,
        model_to_cuda_seconds=transfer_seconds,
        gpu_allocated_after_load=int(torch.cuda.memory_allocated(device)),
        gpu_reserved_after_load=int(torch.cuda.memory_reserved(device)),
        memory_scope="tensor payloads/torch allocator; excludes Python, CPU RSS, CUDA context and driver",
        replay_memory_note="saved training resource only; not loaded or needed for inference")
    recorder.emit("model_loaded", model=name, resources=resources)
    return model, resources


def cached_native_execution(graph, arrays):
    """Same CPU graph transition semantics as actual hard_eval, no labels."""
    words = arrays["tokens"]
    state = graph["enc"][arrays["starts"]].copy()
    for column in range(words.shape[1]):
        for symbol, transition in graph["trans"].items():
            selected = words[:, column] == symbol
            state[selected] = transition[state[selected]]
    return graph["dec"][state]


def realize_continuous_bank(model, recorder):
    """Rebuild every unique learned bank class through actual geometric lifts.

    The lift receives unlabeled configurations from ConfigurationLoop. Its
    result, not the expected stored permutation, becomes the executable map.
    Expected permutations are used only for the post-lift faithfulness check.
    """
    from continuous_monodromy import ConfigurationLoop, lift_sequence
    recorder.check()
    graph = model.graph(2, range(1, model.vocab+1))
    omega = model.path_omega.detach().cpu().double().numpy()
    cache, classes = {}, []
    snapshots = distance_entries = 0
    began = time.perf_counter()
    for permutation in graph["permutations"]:
        key = tuple(int(q) for q in permutation)
        if key in cache:
            continue
        recorder.check()
        loop = ConfigurationLoop(permutation, omega)
        remaining = recorder.deadline-time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("continuous bank realization deadline reached")
        lifted = lift_sequence([loop], time.monotonic()+remaining)
        actual = np.asarray(lifted["monodromy"], dtype=np.int64)
        if not np.array_equal(actual, permutation):
            raise RuntimeError("actual continuous lift differs from saved learned native class")
        cache[key] = actual
        snapshots += int(lifted["snapshots"])
        distance_entries += int(lifted["distance_matrix_entries_evaluated"])
        classes.append(dict(certificate=loop.certificate(), snapshots=lifted["snapshots"],
                            distance_matrix_entries=lifted["distance_matrix_entries_evaluated"]))
    maps = [cache[tuple(int(q) for q in permutation)] for permutation in graph["permutations"]]
    return graph, maps, dict(
        unique_realized_classes=len(cache), original_bank_entries=len(maps),
        configuration_snapshots=snapshots, distance_matrix_entries=distance_entries,
        continuous_lift_seconds=time.perf_counter()-began,
        native_map_cache_bytes=sum(value.nbytes for value in cache.values()),
        graph_numeric_payload_bytes=numeric_payload_bytes(graph),
        actual_path_omega=omega.tolist(), classes=classes,
        trace_files_written=False, no_expected_map_used_as_executor=True)


def cached_continuous_execution(graph, maps, arrays):
    """Actual learned routes invoke maps obtained from continuous path lifts."""
    routes = {int(a): calls for a, calls in zip(graph["tokens"], graph["routes"])}
    state = graph["enc"][arrays["starts"]].copy()
    words = arrays["tokens"]
    for column in range(words.shape[1]):
        symbols = words[:, column]
        for token in np.unique(symbols):
            if not token:
                continue
            selected = symbols == token
            for expert in routes[int(token)]:
                state[selected] = maps[int(expert)][state[selected]]
    return graph["dec"][state]


def output_predictions(value):
    if torch.is_tensor(value):
        if not torch.isfinite(value).all():
            raise FloatingPointError("nonfinite neural benchmark output")
        return value.argmax(-1).detach().cpu().numpy()
    return np.asarray(value, dtype=np.int64)


def latency(function, batch, length, device, recorder, warmup=5, repeats=30):
    for _ in range(warmup):
        recorder.check()
        function()
        synchronize(device)
    samples, output = [], None
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(repeats):
        recorder.check()
        synchronize(device)
        began = time.perf_counter_ns()
        output = function()
        synchronize(device)
        samples.append((time.perf_counter_ns()-began)/1e6)
    array = np.asarray(samples, dtype=np.float64)
    mean = float(array.mean())
    return dict(warmup_repeats=warmup, timed_repeats=repeats, samples_ms=samples,
                mean_ms=mean, p50_ms=float(np.quantile(array, .5)),
                p95_ms=float(np.quantile(array, .95)),
                min_ms=float(array.min()), max_ms=float(array.max()),
                episodes_per_second=batch*1000./mean,
                nonpad_tokens_per_second=batch*length*1000./mean,
                gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated(device))), output


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topo-checkpoint", type=Path, required=True)
    parser.add_argument("--transformer-checkpoint", type=Path, required=True)
    parser.add_argument("--replay-checkpoint", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--resources-json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--seconds", type=float, default=100.)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 110:
        parser.error("internal deadline must be <=110s; root external runner <=120s")
    return args


@torch.inference_mode()
def main():
    args = arguments()
    recorder = Recorder(args.out, args.seconds)
    rows, resources, status = [], [], "FAILED_WITH_PARTIAL_RECORDS"
    try:
        torch.set_num_threads(1)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU fallback authorized for this comparison")
        device = torch.device(args.device)
        synchronize(device)
        inputs = load_inputs(args.datasets, recorder)
        resource_rows = []
        if args.resources_json:
            if args.resources_json.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("resource extraction exceeds size bound")
            resource_rows = json.loads(args.resources_json.read_text())["stages"]
            recorder.emit("training_resource_source", path=args.resources_json,
                          sha256=sha256(args.resources_json), used_only_for_saved_cost_metadata=True)
        recorder.emit("configuration", args=vars(args), torch_version=torch.__version__,
                      cuda_version=torch.version.cuda, gpu=torch.cuda.get_device_name(device),
                      inference_mode=True, no_training=True, no_labels_loaded=True,
                      fixed_models=list(MODELS), PRIMARY_BASELINE="Transformer",
                      ordinary_warmup=5, ordinary_repeats=30,
                      full_continuous_cold_warmup=1, full_continuous_cold_repeats=5,
                      batch_sizes=[1, 64], lengths=[7, 12, 16],
                      internal_deadline=args.seconds, required_external_timeout_max=120,
                      source_sha256=sha256(Path(__file__)))
        for name in MODELS:
            recorder.check()
            path = getattr(args, name+"_checkpoint")
            model, resource = restore_model(name, path, device, recorder, resource_rows)
            resources.append(dict(model=name, **resource))
            graph = continuous_graph = continuous_maps = None
            if name == "topo":
                synchronize(device)
                began = time.perf_counter()
                graph = model.graph(2, range(1, model.vocab+1))
                synchronize(device)
                recorder.emit("native_graph_initial_materialization", model=name,
                              seconds=time.perf_counter()-began,
                              numeric_payload_bytes=numeric_payload_bytes(graph),
                              backend="actual graph(): neural routes + assignments + reachable closure",
                              labels_used=False)
                if hasattr(model, "path_omega"):
                    synchronize(device)
                    began = time.perf_counter()
                    continuous_graph, continuous_maps, realization = realize_continuous_bank(model, recorder)
                    synchronize(device)
                    recorder.emit("continuous_bank_initial_realization", model=name,
                                  total_seconds=time.perf_counter()-began, **realization)
            for (batch, cap), (arrays, input_meta) in inputs.items():
                recorder.check()
                synchronize(device)
                began = time.perf_counter()
                gpu = {k: torch.as_tensor(v, dtype=torch.long, device=device)
                       for k, v in arrays.items()}
                synchronize(device)
                transfer = time.perf_counter()-began
                neural = lambda: model(gpu["tokens"], gpu["starts"], gpu["task_ids"])
                modes = [("soft_neural_forward" if name == "topo" else "neural_forward",
                          "CUDA learned network, resident input tensors, output logits", neural, 5, 30)]
                if name == "topo":
                    cold = lambda: model.hard_eval(gpu["tokens"], gpu["starts"], gpu["task_ids"])
                    warm = lambda: cached_native_execution(graph, arrays)
                    # Agreement checks compare implementations, never task labels.
                    cold_predictions = output_predictions(cold())
                    cached_predictions = output_predictions(warm())
                    if not np.array_equal(cold_predictions, cached_predictions):
                        raise RuntimeError("cached executor differs from actual hard_eval")
                    modes += [
                        ("native_cold_materialize_and_execute",
                         "actual hard_eval: CPU graph extraction/matching and execution + CUDA input/output copies", cold, 5, 30),
                        ("native_warm_cached_cpu_graph_execute",
                         "CPU NumPy learned graph lookup with cached graph and resident CPU inputs; no neural routing cost", warm, 5, 30)]
                    if continuous_maps is not None:
                        warm_continuous = lambda: cached_continuous_execution(continuous_graph, continuous_maps, arrays)
                        if not np.array_equal(output_predictions(warm_continuous()), cold_predictions):
                            raise RuntimeError("actual continuous-derived bank differs from neural native executor")
                        def cold_continuous():
                            local_graph, local_maps, details = realize_continuous_bank(model, recorder)
                            predicted = cached_continuous_execution(local_graph, local_maps, arrays)
                            return predicted, details
                        modes += [
                            ("continuous_cold_realize_bank_and_execute",
                             "CPU actual unlabeled-configuration lifts of ALL unique learned bank classes + learned route-slot execution; neural graph extraction included",
                             cold_continuous, 1, 5),
                            ("continuous_warm_lift_map_cache_execute",
                             "CPU learned encoder/routes/decoder invoke cached ACTUALLY lifted primitive maps; cold geometry/neural extraction excluded",
                             warm_continuous, 5, 30)]
                for mode, scope, function, warmup, repeats in modes:
                    realization_samples = []
                    if mode == "continuous_cold_realize_bank_and_execute":
                        def timed_continuous_call():
                            prediction, details = function()
                            realization_samples.append(details)
                            return prediction
                        run = timed_continuous_call
                    else:
                        run = function
                    timing, output = latency(run, batch, cap, device, recorder, warmup, repeats)
                    predictions = output_predictions(output)
                    row = dict(model=name, mode=mode, timing_scope=scope,
                               batch=batch, length_cap=cap, input_sha256=input_meta["input_sha256"],
                               source_split=input_meta["source_split"], input_transfer_seconds=transfer,
                               input_transfer_inside_timing=(mode == "native_cold_materialize_and_execute"),
                               external_input_transfer_excluded=True,
                               predicted_output_sha256=hashlib.sha256(np.asarray(predictions, dtype="<i8").tobytes()).hexdigest(),
                               accuracy_not_evaluated=True, **timing)
                    if realization_samples:
                        row["cold_realization_per_call"] = realization_samples
                        row["cold_realization_scope"] = "includes declared warmup then timed calls, in order"
                        row["total_cold_configuration_snapshots_including_warmup"] = sum(r["configuration_snapshots"] for r in realization_samples)
                        row["total_cold_distance_entries_including_warmup"] = sum(r["distance_matrix_entries"] for r in realization_samples)
                    if mode.startswith("continuous_"):
                        row["warm_map_cache_bytes"] = sum({id(m): m.nbytes for m in continuous_maps}.values())
                        row["native_primitive_calls_per_episode"] = cap*model.slots
                    rows.append(row)
                    recorder.emit("inference_latency", **row)
                del gpu
            final_hash = state_hash(model.state_dict())
            if final_hash != resource["state_sha256_before"]:
                raise RuntimeError("inference mutated learned parameters/buffers")
            recorder.emit("model_inference_complete", model=name, state_sha256_after=final_hash,
                          learned_state_unchanged=True)
            del model, graph, continuous_graph, continuous_maps
            gc.collect()
            torch.cuda.empty_cache()
        status = "COMPLETED_NO_TRAINING"
    except BaseException as error:
        recorder.emit("failure", exception=type(error).__name__, reason=str(error),
                      traceback=traceback.format_exc())
        raise
    finally:
        recorder.summary(dict(
            schema="saved-architecture-inference-cost-v1", status=status,
            M0="NOT_ASSESSED", PRIMARY_BASELINE="Transformer",
            rows=rows, model_resources=resources,
            wall_seconds=time.perf_counter()-recorder.started,
            comparison_boundaries=[
                "All arms use byte-identical saved input arrays and no labels; no parameters are updated.",
                "Soft neural execution, cold native extraction/execution and cached CPU graph execution are separate interfaces.",
                "Cached CPU graph timing excludes graph construction, neural routing and host/device transfers; it is not a neural speedup measurement.",
                "Continuous cold mode rebuilds every unique learned bank class by geometric lifting; five timed repeats and one warmup are explicitly separated from the thirty-repeat warm modes.",
                "Continuous warm mode caches actual lift-derived primitive maps and executes the learned call slots; it is not a fresh continuous lift for every request.",
                "Cold hard_eval includes its actual extraction and GPU/CPU copies; setup/checkpoint loading is separately recorded.",
                "GPU peaks are PyTorch allocator statistics, not complete VRAM or CPU/process memory.",
                "Stored optimizer/replay costs describe training artifacts and are not deployed inference requirements.",
                "Thirty same-process timings are timing repeats, not independent training seeds or a scalable advantage certificate.",
                "This benchmark does not establish an equal-accuracy lifecycle cost frontier, physical energy advantage or M0 pass."]))


if __name__ == "__main__":
    main()
