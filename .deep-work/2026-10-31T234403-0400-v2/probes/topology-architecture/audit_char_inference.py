"""ROOT-RUN frozen character inference audit; no training or model selection.

Use the exact owned frozen source directory accompanying each checkpoint.
Continuous native execution hardens CALL only and keeps a normalized continuous
readout: its NLL/PPL is well-defined, unlike categorical one-hot readout PPL.
All three contexts use the same512 target positions. Latencies are resident-input
forward measurements, not end-to-end serving or exclusive-GPU claims.
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

import numpy as np
import torch
from torch.nn import functional as F


FROZEN_CONTINUOUS_SHA = "b8a51a6ca9026b2ebff877493bf9031972a319d097cb3ae2605cc6ad3ea3083f"
SPLITS = (("test_iid", 32), ("test_unseen", 64), ("test_long", 128))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def tensor_bytes(value):
    if torch.is_tensor(value): return value.numel() * value.element_size()
    if isinstance(value, dict): return sum(tensor_bytes(x) for x in value.values())
    if isinstance(value, (list, tuple)): return sum(tensor_bytes(x) for x in value)
    return 0


def state_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        x = value.detach().cpu().contiguous()
        h.update(json.dumps([name, str(x.dtype), list(x.shape)], separators=(",", ":")).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()


def owned_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise RuntimeError("owned source import failed")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_model(args, payload, device):
    if payload.get("schema") != "char-context-development-v1" or payload.get("completed_updates") != 300:
        raise ValueError("owned completed300-update character checkpoint required")
    sources = payload["source_sha256"]
    for name, expected in sources.items():
        path = args.frozen_source / name
        if Path(name).name != name or not path.is_file() or sha(path) != expected:
            raise ValueError(f"frozen source differs from checkpoint: {name}")
    arm = payload["args"]["arm"]
    if arm == "block":
        path = args.frozen_source / "continuous_primitive_block.py"
        if sources.get(path.name) != FROZEN_CONTINUOUS_SHA: raise ValueError("exact original b8a51 continuous source required")
        module = owned_module(path, "_owned_continuous_char_audit")
        if payload.get("block_config") is None: raise ValueError("continuous configuration absent")
        model = module.ContinuousPrimitiveBlock(**payload["block_config"]).to(device)
    elif arm in ("gru", "transformer"):
        # This checked original helper imports its own categorical dependency;
        # keep the entire frozen directory first, never a current live module.
        sys.path.insert(0, str(args.frozen_source.resolve()))
        module = owned_module(args.frozen_source / "train_contextual_primitives.py", "_owned_baseline_char_audit")
        cls = module.TerminalGRU if arm == "gru" else module.TerminalTransformer
        model = cls(**payload["baseline_spec"]).to(device)
    else: raise ValueError("only continuous Block/GRU/Transformer owned models supported")
    model.load_state_dict(payload["model_state"], strict=True)
    # Restore the exact owned parameter/buffer storage strides where needed.
    layouts = []
    for name, value in model.state_dict(keep_vars=True).items():
        source = payload["model_state"][name]
        if value.dtype != source.dtype or value.shape != source.shape: raise ValueError("owned tensor layout mismatch")
        target_stride = tuple(value.stride()); saved_stride = tuple(source.stride())
        if target_stride != saved_stride:
            if any(s < 0 for s in saved_stride): raise ValueError("negative saved tensor stride")
            value.data = torch.empty_strided(source.shape, saved_stride, dtype=source.dtype, device=device).copy_(source)
        if not torch.equal(value.detach().cpu(), source): raise RuntimeError("actual source tensor values not restored")
        layouts.append(dict(name=name, shape=list(source.shape), saved_stride=list(saved_stride),
            initial_factory_stride=list(target_stride), restored_stride=list(value.stride()), values_equal=True))
    model.eval()
    return model, arm, layouts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--frozen-source", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-seconds", type=float, default=850)
    p.add_argument("--latency-repeats", type=int, default=12)
    p.add_argument("--resource-overlap-note", default="coordinator must record actual occupancy; no isolated-throughput claim")
    args = p.parse_args()
    if not 1 <= args.max_seconds <= 850 or not 4 <= args.latency_repeats <= 32: p.error("bounded time/repeat protocol required")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); deadline = started + args.max_seconds
    stream = (args.out / "records.jsonl").open("x", buffering=1)
    def emit(kind, **value):
        stream.write(json.dumps(dict(kind=kind, wall=time.perf_counter() - started, **value), sort_keys=True, allow_nan=False) + "\n")
    def check():
        if time.perf_counter() > deadline: raise TimeoutError("frozen character audit bound")
    device = torch.device(args.device)
    def sync():
        if device.type == "cuda": torch.cuda.synchronize(device)
    def memory():
        result = dict(cpu_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)))
        if device.type == "cuda": result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(device), cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        return result
    def batch(data, indices):
        return {k: torch.as_tensor(data[k][indices], device=device) for k in ("initial", "tokens", "lengths", "targets")}
    def forward(inputs, mode):
        if arm == "block":
            return model(inputs["initial"], inputs["tokens"], inputs["lengths"], execution=mode, output="logits")
        return F.log_softmax(model(inputs["initial"], inputs["tokens"], inputs["lengths"]), -1)
    def proxy(count, length, mode):
        if arm == "block": return model.resource_report(batch_size=count, sequence_length=length,
            active_steps=count * length, execution=mode)["proxy_forward_macs"]
        return model.proxy(count, length, count * length)["proxy_forward_macs"]
    costs = dict(evaluation_forward_examples=0, evaluation_forward_MAC_proxy=0, evaluation_wall_seconds=0.,
        timed_forward_calls=0, timed_forward_examples=0, timed_forward_MAC_proxy=0,
        warmup_forward_calls=0, warmup_forward_examples=0, warmup_forward_MAC_proxy=0,
        batch_assembly_and_transfer_wall_seconds=0.)
    result = {}; model = None
    try:
        if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("requested GPU unavailable; no fallback")
        if args.checkpoint.stat().st_size > 100 << 20: raise ValueError("owned checkpoint cap")
        checkpoint_sha = sha(args.checkpoint)
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        metadata = json.loads((args.data / "metadata.json").read_text())
        if metadata != payload["metadata"]: raise ValueError("exact original prepared metadata required")
        if metadata.get("schema") != "sampled-char-context-terminal-v1":
            raise ValueError("owned character preparation schema required")
        model, arm, layouts = load_model(args, payload, device)
        before_hash = state_hash(model)
        modes = ("soft", "native") if arm == "block" else ("soft",)
        optimizer_bytes = tensor_bytes(payload.get("optimizer", {}))
        source_checkpoint_resources = dict(parameters=sum(x.numel() for x in model.parameters()),
            parameter_bytes=tensor_bytes(dict(model.named_parameters())), buffer_bytes=tensor_bytes(dict(model.named_buffers())),
            checkpoint_bytes=args.checkpoint.stat().st_size, saved_training_optimizer_tensor_bytes=optimizer_bytes,
            live_inference_optimizer_tensor_bytes=0)
        emit("configuration", args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            checkpoint_sha256=checkpoint_sha, completed_training_updates=300, arm=arm,
            source_sha256=payload["source_sha256"], audit_driver_sha256=sha(Path(__file__)),
            data_metadata_sha256=sha(args.data / "metadata.json"), model_state_before_sha256=before_hash,
            restored_layouts=layouts, resources=source_checkpoint_resources,
            no_training=True, no_test_selection=True, formal_Primary=False,
            native_scope="hardCALL only; continuous state/memory/operations and normalizedreadout; NLLisvalid" if arm == "block" else None)
        del payload; gc.collect()
        results = {}; latency = []; receipts = []; paired_positions = paired_targets = None
        with torch.inference_mode():
            for split, extent in SPLITS:
                check(); entry = metadata["stages"][0]["files"][split]
                source = args.data / entry["file"]; positions_path = args.data / entry["positions_file"]
                if sha(source) != entry["sha256"] or sha(positions_path) != entry["positions_sha256"]: raise ValueError("frozen input/position hash mismatch")
                with np.load(source, allow_pickle=False) as stored:
                    if set(stored.files) != {"initial", "tokens", "lengths", "targets"}: raise ValueError("unexpected data keys")
                    data = {k: stored[k] for k in stored.files}
                positions = np.load(positions_path, allow_pickle=False)
                if (data["tokens"].shape != (512, extent) or any(x.dtype != np.int64 for x in data.values())
                        or data["initial"].shape != (512,) or np.any(data["initial"])
                        or data["targets"].shape != (512,) or data["lengths"].shape != (512,)
                        or np.any(data["lengths"] != extent) or positions.dtype != np.int64 or positions.shape != (512,)):
                    raise ValueError("fixed512-target/context contract")
                if paired_positions is None: paired_positions, paired_targets = positions.copy(), data["targets"].copy()
                elif not np.array_equal(positions, paired_positions) or not np.array_equal(data["targets"], paired_targets): raise ValueError("TEST contexts are not the original paired targets")
                receipts.append(dict(split=split, source_sha256=sha(source), positions_sha256=sha(positions_path), rows=512, context=extent))
                results[split] = {}
                for mode in modes:
                    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
                    sync(); t = time.perf_counter(); losses = []; predicted = []; norm_error = 0.
                    for offset in range(0, 512, 64):
                        check(); bt = time.perf_counter(); inputs = batch(data, np.arange(offset, offset + 64)); sync()
                        costs["batch_assembly_and_transfer_wall_seconds"] += time.perf_counter() - bt
                        output = forward(inputs, mode)
                        if not bool(torch.isfinite(output).all()): raise FloatingPointError("nonfinite probability model")
                        norm_error = max(norm_error, float(torch.logsumexp(output, -1).abs().max()))
                        losses.append(F.nll_loss(output, inputs["targets"], reduction="none").cpu().numpy().astype(np.float64))
                        predicted.append(output.argmax(-1).cpu().numpy())
                        costs["evaluation_forward_examples"] += 64; costs["evaluation_forward_MAC_proxy"] += proxy(64, extent, mode)
                    sync(); elapsed = time.perf_counter() - t; costs["evaluation_wall_seconds"] += elapsed
                    nll = np.concatenate(losses); prediction = np.concatenate(predicted); mean = float(nll.mean())
                    if norm_error > 1e-5: raise RuntimeError("normalized log-readout contract failed")
                    vectors = args.out / f"{split}-{mode}-per-target.npz"
                    with vectors.open("xb") as dest: np.savez_compressed(dest, nll=nll, prediction=prediction,
                        targets=data["targets"], positions=positions)
                    result_row = dict(rows=512, context=extent, next_character_NLL=mean,
                        sampled_character_PPL=math.exp(mean) if mean < 700 else None,
                        accuracy=float(np.mean(prediction == data["targets"])), maximum_log_normalizer_error=norm_error,
                        per_target_file=vectors.name, per_target_sha256=sha(vectors), wall_seconds=elapsed,
                        forward_MAC_proxy=proxy(512, extent, mode), memory=memory())
                    results[split][mode] = result_row; emit("actual_probability_evaluation", split=split, mode=mode, **result_row)
                    del inputs, output; gc.collect()
                    for size in (1, 64):
                        check(); bt = time.perf_counter(); inputs = batch(data, np.arange(size)); sync()
                        costs["batch_assembly_and_transfer_wall_seconds"] += time.perf_counter() - bt
                        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
                        allocated_before = memory()
                        for _ in range(2):
                            check(); value = forward(inputs, mode); sync(); del value
                            costs["warmup_forward_calls"] += 1; costs["warmup_forward_examples"] += size
                            costs["warmup_forward_MAC_proxy"] += proxy(size, extent, mode)
                        durations = []
                        for repeat in range(args.latency_repeats):
                            check(); sync(); tick = time.perf_counter(); value = forward(inputs, mode); sync()
                            elapsed = time.perf_counter() - tick; del value; durations.append(elapsed)
                            costs["timed_forward_calls"] += 1; costs["timed_forward_examples"] += size
                            costs["timed_forward_MAC_proxy"] += proxy(size, extent, mode)
                            emit("actual_latency_draw", split=split, mode=mode, batch_size=size, repeat=repeat, wall_seconds=elapsed)
                        row = dict(split=split, context=extent, mode=mode, batch_size=size, repetitions=len(durations),
                            warmup_repetitions=2, resident_forward_latency_seconds=durations,
                            p50_seconds=float(np.percentile(durations, 50)), p95_seconds=float(np.percentile(durations, 95)),
                            forward_MAC_proxy_per_call=proxy(size, extent, mode), memory_before=allocated_before, memory_after=memory(),
                            scope="synchronizedwall resident-input no-gradforward inclPython/kernel/gather; excludesdata/H2D/loading; noautoregressiveKVcache")
                        latency.append(row); emit("latency_case_complete", **row)
                        del inputs; gc.collect()
        after_hash = state_hash(model); final_checkpoint_sha = sha(args.checkpoint)
        if before_hash != after_hash or checkpoint_sha != final_checkpoint_sha: raise RuntimeError("frozen weights/checkpoint changed during audit")
        result = dict(status="completed_frozen_inference_audit", arm=arm, results=results, latency=latency,
            input_receipts=receipts, checkpoint_sha256=checkpoint_sha, model_state_before_sha256=before_hash,
            model_state_after_sha256=after_hash, frozen_weights_unchanged=True,
            resources=source_checkpoint_resources, costs=costs, final_memory=memory(),
            paired_TEST_targets=True, no_training=True, no_selection=True, formal_Primary=False,
            driver_wall_seconds=time.perf_counter() - started, resource_overlap_note=args.resource_overlap_note,
            metric_scope="512paired sampledcharactertargets, notfullcorpusPPL; hardCALLPPL normalizedcontinuousreadout",
            cost_scope="evaluation/batchtimingsnested; warmup+timeddrawsfullycounted; forwardMACproxiesonly; nohardwareenergyorexclusiveGPUclaim")
        with (args.out / "summary.json").open("x") as dest: json.dump(result, dest, indent=2, sort_keys=True, allow_nan=False); dest.write("\n")
        emit("completed", frozen_weights_unchanged=True)
        print(json.dumps(dict(status=result["status"], arm=arm, out=str(args.out))))
    except Exception as error:
        emit("failure", error=f"{type(error).__name__}: {error}", costs=costs)
        (args.out / "exception.txt").write_text(traceback.format_exc())
        with (args.out / "failure.json").open("x") as dest: json.dump(dict(error=f"{type(error).__name__}: {error}", costs=costs, partial_result=result), dest, indent=2, allow_nan=False)
        raise
    finally: stream.close()


if __name__ == "__main__": main()
