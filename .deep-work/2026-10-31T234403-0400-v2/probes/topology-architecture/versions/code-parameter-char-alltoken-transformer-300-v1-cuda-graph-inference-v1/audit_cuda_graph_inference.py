"""ROOT-RUN frozen all-token LM CUDA Graph cost diagnostic, no learning.

Exact owned 58feb69 core + 567e372 all-token wrapper only. All timed inputs are
the first 32 original TRAIN windows, tiled to L64/128/256; no labels are loaded.
Original eager includes original validation/active indexing. Tensor-only eager
and CUDA Graph replay call the SAME function. The latter two remove validation
and the all-active gather/scatter scaffolding, never change arithmetic modules.

No compile, Triton, subprocess, network, install, optimizer or fallback. Root
uses the unchanged external 900s/GPU .3 runner, and schedules GPU work serially.
API references: https://docs.pytorch.org/docs/2.8/notes/cuda.html#cuda-graphs
https://docs.pytorch.org/docs/2.8/generated/torch.cuda.CUDAGraph.html
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback


CORE_SHA = "58feb69e5d628745f64b6a4d546858aaba528169ae0d3cdae0530de61055d497"
DRIVER_SHA = "567e372187e4cee3c8a5ca3d49f5f7d60a8664a07619c3d57dc1c799cbbd9f02"
OWNED = {
    "dynamic": (41593, 549107, "1e0fc8ec0e063aa1c86f4c7b0d09d91d65c32338273323a4c6d10c2752993f80"),
    "static": (41593, 549107, "4f320f342dacd733f8c4c10701bd481ed43be0c3511a26d49fbf8228eb35b19c"),
    "detached": (41593, 549107, "f6ac70ed8af68280a74533ef672c0e75459c7217e3b85a3bf4a8fb87f16d4711"),
    "gru": (32660, 423214, "a5ce103d93dd023df2437d7b9fac1463f6ad65b504e81045e0558d5f9a70954e"),
    "transformer": (39664, 646703, "7a8ee603b1244281d268ffcd5c56b9bd9abe63abac9cab24e98591d1d9c2b33a"),
}
CONTEXTS = (64, 128, 256)
BATCHES = (1, 32)
REPEATS = 12
WARMUPS = 3
ATOL = 2e-5
RTOL = 2e-6


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def percentile(values, quantile):
    """Linear percentile, explicitly fixed rather than nearest-rank p95."""
    if not values or not 0 <= quantile <= 1:
        raise ValueError("nonempty sample and quantile in [0,1] required")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lo = math.floor(position)
    hi = math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def break_even(setup_seconds, eager_seconds, graph_seconds):
    saving = eager_seconds - graph_seconds
    return dict(per_call_saving_seconds=saving,
                calls=math.ceil(setup_seconds / saving) if saving > 0 else None,
                reached_only_if_positive_saving=saving > 0)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def owned_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("owned source import unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def restore(args, payload, device):
    if payload.get("schema") != "char-alltoken-ar-neural-v1" or payload.get("completed_updates") != 300:
        raise ValueError("owned completed300 all-token checkpoint required")
    if payload.get("formal_Primary") is not False or payload["args"]["arm"] != args.arm:
        raise ValueError("owned development arm differs")
    required = {"continuous_primitive_block.py": CORE_SHA, "train_char_alltoken.py": DRIVER_SHA}
    if payload["source_sha256"] != required:
        raise ValueError("checkpoint requires the exact owned original source pair")
    for leaf, digest in required.items():
        path = args.frozen_source / leaf
        if path.is_symlink() or sha(path) != digest:
            raise ValueError("original frozen source changed: " + leaf)
    owned_module(args.frozen_source / "continuous_primitive_block.py", "continuous_primitive_block")
    module = owned_module(args.frozen_source / "train_char_alltoken.py", "_owned_cuda_graph_alltoken")
    model = module.AllTokenLM(args.arm, payload["metadata"]["n_tokens"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    layouts = []
    for name, tensor in model.state_dict(keep_vars=True).items():
        saved = payload["model_state"][name]
        if tensor.shape != saved.shape or tensor.dtype != saved.dtype:
            raise ValueError("actual saved tensor shape/dtype differs")
        original_stride = tuple(tensor.stride())
        if original_stride != tuple(saved.stride()):
            if any(s < 0 for s in saved.stride()):
                raise ValueError("negative saved stride")
            tensor.data = torch.empty_strided(saved.shape, saved.stride(), dtype=saved.dtype, device=device).copy_(saved)
        if not torch.equal(tensor.detach().cpu(), saved):
            raise RuntimeError("actual restored tensor values differ")
        layouts.append(dict(name=name, shape=list(saved.shape), saved_stride=list(saved.stride()),
                            initial_factory_stride=list(original_stride), restored_stride=list(tensor.stride())))
    if model.configuration() != payload["model_configuration"]:
        raise ValueError("actual wrapper configuration differs from checkpoint")
    if sum(p.numel() for p in model.parameters()) != OWNED[args.arm][0]:
        raise ValueError("actual total P differs from fixed owned architecture")
    model.eval()
    return model, layouts


def tensor_initial_carry(core, initial):
    """Exact tensor statements from original initial_carry, no range checks.

    Original initial_carry is separately called OUTSIDE capture and checked
    against this initialization. Encoder/expand/zero still execute every call;
    the learned initial h is never treated as a free cached inference result.
    """
    h = core.encoder(initial)
    return {"h": h,
            "keys": core.initial_memory_keys[None].expand(len(h), -1, -1),
            "values": h.new_zeros(len(h), core.memory_slots, core.internal_states)}


def tensor_forward(model, arm, initial, tokens, mode):
    """All-active tensor body; called unchanged for eager and graph capture."""
    if arm in ("dynamic", "static", "detached"):
        core = model.body
        carry = tensor_initial_carry(core, initial)
        outputs = []
        for step in range(tokens.shape[1]):
            raw_x = core.token_embedding(tokens[:, step])
            carry = core._advance_embedded(carry, raw_x, mode)
            outputs.append(F.log_softmax(core.readout(carry["h"]), -1))
        return torch.stack(outputs, 1)
    h = model.embedding(tokens)
    if arm == "gru":
        h, _ = model.body(h)
    else:
        h = h + model.positions[:tokens.shape[1]][None]
        for layer in model.body:
            h = layer(h, src_mask=model.causal_mask[:tokens.shape[1], :tokens.shape[1]], is_causal=True)
        h = model.norm(h)
    return F.log_softmax(model.readout(h), -1)


def compare(reference, output):
    if reference.shape != output.shape or not bool(torch.isfinite(output).all()):
        raise RuntimeError("invalid full all-token output")
    absolute = (reference - output).abs()
    normalizer = float(torch.logsumexp(output, -1).abs().max())
    row = dict(bitwise_equal=bool(torch.equal(reference, output)),
               all_elements_close=bool(torch.allclose(reference, output, atol=ATOL, rtol=RTOL)),
               maximum_absolute_logprob_difference=float(absolute.max()),
               mean_absolute_logprob_difference=float(absolute.double().mean()),
               prediction_disagreements=int((reference.argmax(-1) != output.argmax(-1)).sum()),
               maximum_log_normalizer_error=normalizer, atol=ATOL, rtol=RTOL,
               checked_output_elements=output.numel())
    if not row["all_elements_close"] or row["prediction_disagreements"] or normalizer > 1e-5:
        raise RuntimeError("fixed per-output numerical equivalence contract failed: " + json.dumps(row))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frozen-source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(OWNED), required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=850)
    parser.add_argument("--resource-overlap-note", required=True)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 850 or args.out.exists():
        parser.error("new output and internal <=850s bound required")
    global torch, np, F
    import numpy as np
    import torch
    from torch.nn import functional as F

    args.out.mkdir(parents=True)
    began = time.perf_counter()
    deadline = began + args.max_seconds
    stream = (args.out / "records.jsonl").open("x", buffering=1)
    device = torch.device(args.device)
    results = []
    costs = {}
    model = None
    checkpoint_sha = None
    before_hash = None
    phase = "preflight"
    phase_tick = began
    active_case = None

    def emit(kind, **fields):
        stream.write(json.dumps(dict(kind=kind, session_wall_seconds=time.perf_counter() - began, **fields),
                                sort_keys=True, allow_nan=False) + "\n")

    def check():
        if time.perf_counter() > deadline:
            raise TimeoutError("finite CUDA Graph inference audit bound reached")

    def sync():
        torch.cuda.synchronize(device)

    def memory():
        free, total = torch.cuda.mem_get_info(device)
        return dict(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
                    cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    device_global_free_bytes=free, device_global_total_bytes=total,
                    cpu_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)))

    def charge(label, batch, length, mode, calls=1):
        row = costs.setdefault(label, dict(successful_forward_calls=0, examples=0, output_positions=0, forward_MAC_proxy=0))
        row["successful_forward_calls"] += calls
        row["examples"] += batch * calls
        row["output_positions"] += batch * length * calls
        row["forward_MAC_proxy"] += model.proxy(batch, length, mode) * calls

    def timed(label, function, batch, length, mode):
        # Timing includes these identical CUDA event record + synchronize calls.
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        sync()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = memory()
        draws = []
        for repeat in range(REPEATS):
            check()
            sync()
            tick = time.perf_counter()
            start.record()
            value = function()
            end.record()
            sync()
            elapsed = time.perf_counter() - tick
            gpu_elapsed = start.elapsed_time(end) / 1000
            del value
            charge(label + "_timed", batch, length, mode)
            row = dict(backend=label, batch=batch, context=length, mode=mode, repeat=repeat,
                       wall_seconds=elapsed, CUDA_stream_elapsed_seconds=gpu_elapsed)
            draws.append(row)
            emit("actual_latency_draw", **row)
        return dict(draws=draws, p50_wall_seconds=percentile([x["wall_seconds"] for x in draws], .5),
                    p95_wall_seconds=percentile([x["wall_seconds"] for x in draws], .95),
                    p50_CUDA_stream_seconds=percentile([x["CUDA_stream_elapsed_seconds"] for x in draws], .5),
                    p95_CUDA_stream_seconds=percentile([x["CUDA_stream_elapsed_seconds"] for x in draws], .95),
                    memory_before=baseline, memory_after=memory(),
                    scope="synchronized resident-input fullwindow forward; CUDA stream elapsed is not a sum of kernel-only durations")

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("authorized CUDA unavailable; no fallback")
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.3, device)
        expected_p, expected_bytes, expected_sha = OWNED[args.arm]
        if args.checkpoint.is_symlink() or args.checkpoint.stat().st_size != expected_bytes:
            raise ValueError("exact owned checkpoint file required")
        checkpoint_sha = sha(args.checkpoint)
        if checkpoint_sha != expected_sha:
            raise ValueError("checkpoint outside exact owned allowlist")
        check()
        load_tick = time.perf_counter()
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        metadata_file = args.data / "metadata.json"
        metadata = json.loads(metadata_file.read_text())
        if metadata != payload["metadata"] or metadata.get("schema") != "char-alltoken-ar-development-v1":
            raise ValueError("exact original all-token metadata required")
        entry = metadata["files"]["train"]
        train_file = args.data / entry["file"]
        if train_file.is_symlink() or sha(train_file) != entry["sha256"]:
            raise ValueError("original TRAIN window file changed")
        with np.load(train_file, allow_pickle=False) as stored:
            if set(stored.files) != {"initial", "tokens", "lengths", "targets"}:
                raise ValueError("unexpected TRAIN archive fields")
            # Deliberately do not decompress/access targets: label reads stay 0.
            train = {key: stored[key] for key in ("initial", "tokens", "lengths")}
        if (any(x.dtype != np.int64 for x in train.values()) or train["tokens"].shape != (8192, 64)
                or train["initial"].shape != (8192,) or train["lengths"].shape != (8192,)
                or np.any(train["initial"]) or np.any(train["lengths"] != 64)
                or np.any((train["tokens"] < 0) | (train["tokens"] >= metadata["n_tokens"]))):
            raise ValueError("fixed original all-active TRAIN inputs required")
        model, layouts = restore(args, payload, device)
        before_hash = state_hash(model.state_dict())
        if before_hash != state_hash(payload["model_state"]):
            raise RuntimeError("restored all-parameter/buffer values differ")
        saved_resources = dict(total_parameters=expected_p,
            saved_trainable_parameters=sum(p.numel() for name, p in model.named_parameters() if payload["parameter_flags"][name]),
            parameter_payload_bytes=tensor_bytes(dict(model.named_parameters())),
            buffer_payload_bytes=tensor_bytes(dict(model.named_buffers())),
            checkpoint_bytes=expected_bytes, checkpoint_sha256=checkpoint_sha,
            saved_training_optimizer_tensor_bytes=tensor_bytes(payload["optimizer"]), live_inference_optimizer_tensor_bytes=0,
            saved_training_data_cache_array_bytes=metadata["uncompressed_array_bytes"],
            live_loaded_TRAIN_input_array_bytes=sum(x.nbytes for x in train.values()),
            source_checkpoint_costs=payload["costs"], source_checkpoint_cost_scope="historical training costs retained, not added to inference latency",
            labels_loaded=0, validation_or_TEST_inputs_loaded=0)
        properties = torch.cuda.get_device_properties(device)
        emit("configuration", arm=args.arm, checkpoint_sha256=checkpoint_sha,
            source_sha256=payload["source_sha256"], audit_source_sha256=sha(Path(__file__)),
            model_state_before_sha256=before_hash, model_configuration=model.configuration(),
            original_metadata_sha256=sha(metadata_file), TRAIN_file_sha256=sha(train_file),
            layouts=layouts, resources=saved_resources,
            load_and_restore_wall_seconds=time.perf_counter() - load_tick,
            torch_version=str(torch.__version__), CUDA_version=torch.version.cuda,
            cuDNN_version=torch.backends.cudnn.version(), GPU_name=properties.name,
            precision=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                           cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                           float32_matmul_precision=torch.get_float32_matmul_precision()),
            cases=dict(contexts=list(CONTEXTS), batches=list(BATCHES), modes=["soft", "native"] if args.arm in ("dynamic", "static", "detached") else ["soft"],
                       warmups=WARMUPS, resident_draws=REPEATS, fixed_TRAIN_rows=list(range(32))),
            no_training=True, no_test_selection=True, formal_Primary=False,
            transform="remove CPU validation and redundant all-active index operations only; unchanged core/expert/read/write/attention/modules/readouts",
            cost_scope="all setup/capture/instantiate/warmup/replay/sync/pool cleanup paid; counters overlap and must not be summed blindly",
            resource_overlap_note=args.resource_overlap_note)
        del payload
        gc.collect()
        modes = ("soft", "native") if args.arm in ("dynamic", "static", "detached") else ("soft",)
        with torch.inference_mode():
            for length in CONTEXTS:
                for batch in BATCHES:
                    for mode in modes:
                        check()
                        case_tick = time.perf_counter()
                        phase = "input_setup"
                        phase_tick = setup_tick = time.perf_counter()
                        host_tokens = np.tile(train["tokens"][:batch], (1, length // 64))
                        host_initial = train["initial"][:batch].copy()
                        host_lengths = np.full(batch, length, dtype=np.int64)
                        inputs = {"initial": torch.as_tensor(host_initial, device=device),
                                  "tokens": torch.as_tensor(host_tokens, device=device),
                                  "lengths": torch.as_tensor(host_lengths, device=device)}
                        sync()
                        input_setup_seconds = time.perf_counter() - setup_tick
                        original = lambda: model(inputs["initial"], inputs["tokens"], inputs["lengths"], mode)
                        pure = lambda: tensor_forward(model, args.arm, inputs["initial"], inputs["tokens"], mode)
                        initial_contract = None
                        if args.arm in ("dynamic", "static", "detached"):
                            init_tick = time.perf_counter()
                            left = model.body.initial_carry(inputs["initial"])
                            right = tensor_initial_carry(model.body, inputs["initial"])
                            initial_contract = all(torch.equal(left[k], right[k]) for k in left)
                            if not initial_contract:
                                raise RuntimeError("original initial_carry tensor initialization differs")
                            del left, right
                            sync()
                            initialization_contract_wall_seconds = time.perf_counter() - init_tick
                        else:
                            initialization_contract_wall_seconds = 0.
                        case = dict(arm=args.arm, batch=batch, context=length, mode=mode,
                            total_parameters=expected_p, forward_MAC_proxy_per_call=model.proxy(batch, length, mode),
                            input_setup_wall_seconds=input_setup_seconds,
                            resident_input_tensor_bytes=sum(tensor_bytes(x) for x in inputs.values()),
                            host_case_input_array_bytes=host_tokens.nbytes + host_initial.nbytes + host_lengths.nbytes,
                            original_initial_carry_tensor_contract=initial_contract,
                            initialization_contract_wall_seconds=initialization_contract_wall_seconds)
                        active_case = case
                        snapshots = {}
                        for backend, function in (("original_eager", original), ("tensor_eager", pure)):
                            phase = backend + "_warmup"
                            phase_tick = tick = time.perf_counter()
                            for _ in range(WARMUPS):
                                check()
                                value = function()
                                sync()
                                del value
                                charge(backend + "_warmup", batch, length, mode)
                            warm_seconds = time.perf_counter() - tick
                            phase = backend + "_output_validation"
                            phase_tick = tick = time.perf_counter()
                            value = function()
                            sync()
                            snapshots[backend] = value.detach().cpu().contiguous()
                            del value
                            charge(backend + "_output_validation", batch, length, mode)
                            contract = compare(snapshots["original_eager"], snapshots[backend])
                            validation_seconds = time.perf_counter() - tick
                            phase = backend + "_timing"
                            phase_tick = time.perf_counter()
                            case[backend] = timed(backend, function, batch, length, mode)
                            case[backend].update(warmup_wall_seconds=warm_seconds,
                                output_validation_wall_seconds=validation_seconds, output_contract=contract)
                            emit("actual_eager_backend_complete", arm=args.arm, batch=batch, context=length,
                                 mode=mode, backend=backend, measurement=case[backend])

                        phase = "graph_side_stream_warmup"
                        phase_tick = tick = time.perf_counter()
                        case["cuda_graph_progress"] = dict(phase=phase)
                        side = torch.cuda.Stream(device=device)
                        side.wait_stream(torch.cuda.current_stream(device))
                        with torch.cuda.stream(side):
                            for _ in range(WARMUPS):
                                check()
                                value = pure()
                                del value
                                charge("graph_side_stream_warmup", batch, length, mode)
                        torch.cuda.current_stream(device).wait_stream(side)
                        sync()
                        side_warmup_seconds = time.perf_counter() - tick
                        case["cuda_graph_progress"]["side_stream_warmup_wall_seconds"] = side_warmup_seconds
                        phase = "graph_construction"
                        phase_tick = tick = time.perf_counter()
                        case["cuda_graph_progress"]["phase"] = phase
                        pool = torch.cuda.graph_pool_handle()
                        graph = torch.cuda.CUDAGraph(keep_graph=True)
                        construction_seconds = time.perf_counter() - tick
                        case["cuda_graph_progress"]["construction_wall_seconds"] = construction_seconds
                        sync()
                        torch.cuda.reset_peak_memory_stats(device)
                        memory_before_capture = memory()
                        case["cuda_graph_progress"]["memory_before_capture"] = memory_before_capture
                        phase = "graph_capture"
                        phase_tick = tick = time.perf_counter()
                        case["cuda_graph_progress"]["phase"] = phase
                        emit("actual_graph_capture_begin", batch=batch, context=length, mode=mode,
                             memory_before_capture=memory_before_capture)
                        with torch.cuda.graph(graph, pool=pool, stream=side, capture_error_mode="global"):
                            graph_output = pure()
                        sync()
                        capture_seconds = time.perf_counter() - tick
                        case["cuda_graph_progress"]["capture_wall_seconds"] = capture_seconds
                        # Capturing describes arithmetic nodes; it is not a paid
                        # executed forward. Its CPU/setup/memory costs are paid.
                        phase = "graph_instantiation"
                        phase_tick = tick = time.perf_counter()
                        case["cuda_graph_progress"]["phase"] = phase
                        graph.instantiate()
                        sync()
                        instantiate_seconds = time.perf_counter() - tick
                        memory_after_instantiation = memory()
                        case["cuda_graph_progress"].update(instantiation_wall_seconds=instantiate_seconds,
                            memory_after_instantiation=memory_after_instantiation, phase="instantiated")
                        emit("actual_graph_instantiated", batch=batch, context=length, mode=mode,
                             progress=case["cuda_graph_progress"])

                        def replay():
                            graph.replay()
                            return graph_output

                        phase = "graph_output_validation"
                        phase_tick = tick = time.perf_counter()
                        graph.replay()
                        sync()
                        snapshots["cuda_graph"] = graph_output.detach().cpu().contiguous()
                        charge("graph_output_validation", batch, length, mode)
                        contract = compare(snapshots["original_eager"], snapshots["cuda_graph"])
                        graph_validation_seconds = time.perf_counter() - tick
                        phase = "graph_timing"
                        phase_tick = time.perf_counter()
                        case["cuda_graph"] = timed("cuda_graph", replay, batch, length, mode)
                        # The last timed replay is checked again, excluding this
                        # CPU copy/comparison from latency and preventing drift.
                        phase = "graph_postdraw_output_validation"
                        phase_tick = tick = time.perf_counter()
                        postdraw = graph_output.detach().cpu().contiguous()
                        postdraw_contract = compare(snapshots["original_eager"], postdraw)
                        postdraw_seconds = time.perf_counter() - tick
                        del postdraw
                        graph_setup = side_warmup_seconds + construction_seconds + capture_seconds + instantiate_seconds + graph_validation_seconds
                        case["cuda_graph"].update(side_stream_warmup_wall_seconds=side_warmup_seconds,
                            graph_construction_wall_seconds=construction_seconds, capture_wall_seconds=capture_seconds,
                            instantiation_wall_seconds=instantiate_seconds, output_validation_wall_seconds=graph_validation_seconds,
                            postdraw_validation_wall_seconds=postdraw_seconds, output_contract=contract,
                            postdraw_output_contract=postdraw_contract, graph_pool_token=list(graph.pool()),
                            memory_before_capture=memory_before_capture, memory_after_instantiation=memory_after_instantiation,
                            persistent_graph_setup_wall_seconds=graph_setup,
                            persistent_setup_excludes="postdraw verification and teardown; both included in complete lifecycle below",
                            captured_forward_descriptions=1,
                            capture_is_not_an_executed_forward=True,
                            memory_scope="Torch allocator totals/deltas plus device-global free bytes; exact CUDA graph executable/driver bytes unavailable",
                            private_pool_policy="one graph/private pool live per case; no free multi-shape persistent cache assumed")
                        case["speed_ratios"] = dict(
                            original_eager_over_tensor_eager=case["original_eager"]["p50_wall_seconds"] / case["tensor_eager"]["p50_wall_seconds"],
                            tensor_eager_over_cuda_graph=case["tensor_eager"]["p50_wall_seconds"] / case["cuda_graph"]["p50_wall_seconds"])
                        output_path = args.out / f"B{batch}-L{length}-{mode}-outputs.npz"
                        with output_path.open("xb") as dest:
                            np.savez_compressed(dest, **{name: value.numpy() for name, value in snapshots.items()})
                        case.update(output_file=output_path.name, output_bytes=output_path.stat().st_size,
                                    output_sha256=sha(output_path), labels_read=0)
                        phase = "graph_pool_cleanup"
                        phase_tick = tick = time.perf_counter()
                        graph.reset()
                        del graph_output, graph, pool, replay, side, inputs, original, pure, snapshots
                        gc.collect()
                        torch.cuda.empty_cache()
                        sync()
                        cleanup_seconds = time.perf_counter() - tick
                        lifecycle_seconds = graph_setup + postdraw_seconds + cleanup_seconds
                        case["cuda_graph"]["complete_graph_lifecycle_wall_seconds"] = lifecycle_seconds
                        case["amortization"] = dict(
                            primary_complete_graph_lifecycle={backend: break_even(lifecycle_seconds, case[backend]["p50_wall_seconds"], case["cuda_graph"]["p50_wall_seconds"])
                                for backend in ("original_eager", "tensor_eager")},
                            persistent_setup_excluding_postdraw_and_cleanup={backend: break_even(graph_setup, case[backend]["p50_wall_seconds"], case["cuda_graph"]["p50_wall_seconds"])
                                for backend in ("original_eager", "tensor_eager")},
                            shared_input_load_and_correctness_audit_costs="separately charged in full case/driver time; not recurring replay latency")
                        case.update(pool_cleanup_wall_seconds=cleanup_seconds,
                                    memory_after_pool_cleanup=memory(), case_total_wall_seconds=time.perf_counter() - case_tick)
                        results.append(case)
                        emit("actual_case_complete", **case)
                        active_case = None

        phase = "final_invariance"
        phase_tick = time.perf_counter()
        after_hash = state_hash(model.state_dict())
        if before_hash != after_hash or checkpoint_sha != sha(args.checkpoint):
            raise RuntimeError("actual parameters/buffers or source checkpoint changed")
        for leaf, expected in (("continuous_primitive_block.py", CORE_SHA), ("train_char_alltoken.py", DRIVER_SHA)):
            if sha(args.frozen_source / leaf) != expected:
                raise RuntimeError("original frozen source changed during audit")
        result = dict(status="FROZEN_CUDA_GRAPH_INFERENCE_AUDIT_COMPLETED", arm=args.arm, cases=results,
            resources=saved_resources, successful_forward_costs_by_phase=costs,
            model_state_before_sha256=before_hash, model_state_after_sha256=after_hash,
            all_parameters_buffers_and_checkpoint_unchanged=True, final_memory=memory(),
            driver_wall_seconds=time.perf_counter() - began, training_updates=0, label_reads=0,
            inputs="first32 originalTRAIN windows; first1/32 rows tiled to64/128/256; resident buffers unchanged across12draws",
            interpretation="original→tensor removes CPU checks AND redundant all-active indexing; tensor→graph removes repeated Python/launch scheduling for the same tensor function",
            limitations="no new mathematical/FLOP advantage, training change, H2D/serving/KV-cache generation claim, energy claim, or free graph setup/pool",
            resource_overlap_note=args.resource_overlap_note, formal_Primary=False,
            not_full_resource_certificate="CUDA executable driver memory and exact backward FLOPs unavailable; recorded fields retain their stated scopes")
        write_json(args.out / "summary.json", result)
        emit("completed", cases=len(results), training_updates=0, label_reads=0,
             all_parameters_buffers_and_checkpoint_unchanged=True)
        print(json.dumps(dict(status=result["status"], cases=len(results), out=str(args.out))))
    except BaseException as error:
        invariance = None
        failure_memory = None
        try:
            invariance = dict(checkpoint_unchanged=checkpoint_sha == sha(args.checkpoint),
                              model_state_unchanged=before_hash == state_hash(model.state_dict()) if model is not None and before_hash else None)
            failure_memory = memory() if torch.cuda.is_available() else None
        except BaseException as audit_error:
            invariance = dict(check_error=type(audit_error).__name__ + ": " + str(audit_error))
        failure = dict(status="FAILED_OR_BOUNDED_INCOMPLETE_NO_FALLBACK", phase=phase,
            error_type=type(error).__name__, error=str(error), completed_cases=results,
            active_case_partial=active_case, active_phase_wall_seconds=time.perf_counter() - phase_tick,
            failure_memory_if_checkable=failure_memory,
            successful_forward_costs_by_phase=costs, invariance_if_checkable=invariance,
            driver_wall_seconds=time.perf_counter() - began, training_updates=0, label_reads=0)
        emit("actual_failure_no_fallback", **{k: v for k, v in failure.items() if k != "completed_cases"})
        (args.out / "exception.txt").write_text(traceback.format_exc())
        write_json(args.out / "failure.json", failure)
        raise
    finally:
        stream.close()


if __name__ == "__main__":
    main()
