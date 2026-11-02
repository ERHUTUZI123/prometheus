"""ROOT-RUN300-update sampled next-character development; no import-time run.

Existing Contextual Block (SOFT only), GRU64, Transformer64 consume identical
prefixes and300x64 common minibatch indices at Adam LR.003. Bidirectional
attention INSIDE the fully observed prefix is legitimate next-token conditioning;
the target character is never an input. This is terminal-prefix LM development,
not a full-stream LM benchmark or publication Primary. Requires Root's external
resource watchdog/isolation. No network, SSH, installation, HPO or silent device
fallback. Native Block accuracy is separate; no native one-hot PPL is reported.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import resource
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from contextual_primitive_block import ContextualPrimitiveBlock
from train_contextual_primitives import (Recorder, TerminalGRU, TerminalTransformer,
    batch, load_split, plain, sha256, synchronize, tensor_bytes, tensor_state_equal)


SPLITS = ("train", "valid", "test_iid", "test_long", "test_unseen")


def source_hashes():
    return {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("contextual_primitive_block.py"),
                                      Path(__file__).with_name("train_contextual_primitives.py"))}


def frozen_split(args, metadata, name):
    data, receipt = load_split(args, metadata, 0, name)
    entry = metadata["stages"][0]["files"][name]
    expected_length = {"train": 32, "valid": 32, "test_iid": 32, "test_unseen": 64, "test_long": 128}[name]
    if receipt["file_sha256"] != entry["sha256"]:
        raise ValueError("sampled input differs from the frozen preparation hash")
    if (len(data["initial"]) != (8192 if name == "train" else 512) or np.any(data["initial"])
            or np.any(data["lengths"] != expected_length)):
        raise ValueError("fixed character sample/initializer/context protocol differs")
    return data, receipt


def probability_forward(model, inputs, arm, native=False):
    if arm == "block":
        return model(inputs["initial"], inputs["tokens"], inputs["lengths"],
                     execution="native" if native else "soft", output="logits")
    return model(inputs["initial"], inputs["tokens"], inputs["lengths"])


def objective(output, targets, arm, reduction="mean"):
    return (F.nll_loss(output, targets, reduction=reduction) if arm == "block"
            else F.cross_entropy(output, targets, reduction=reduction))


def build(arm, metadata, internal_states, device, block_initialization="iid"):
    v = metadata["n_states"]
    if arm == "block":
        return ContextualPrimitiveBlock(n_states=v, n_tokens=v, internal_states=internal_states,
            n_primitives=4, token_dim=16, state_dim=16, memory_dim=16,
            caller_hidden=32, n_slots=4, routing="dynamic", initialization=block_initialization).to(device)
    cls = TerminalGRU if arm == "gru" else TerminalTransformer
    return cls(n_states=v, n_tokens=v, max_length=128, width=64).to(device)


def resource_state(model, optimizer, device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report = {"parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
        "parameter_bytes": tensor_bytes(dict(model.named_parameters())),
        "buffer_bytes": tensor_bytes(dict(model.named_buffers())),
        "optimizer_tensor_bytes": tensor_bytes(optimizer.state_dict()),
        "cpu_peak_RSS_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
        "replay_bytes": 0}
    if device.type == "cuda":
        report.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return report


def proxy(model, args, count, length, active, native=False):
    if args.arm == "block":
        return model.resource_report(batch_size=count, sequence_length=length, active_steps=active,
            execution="native" if native else "soft")["proxy_forward_macs"]
    return model.proxy(count, length, active)["proxy_forward_macs"]


@torch.no_grad()
def evaluate(model, data, args, device, recorder, native_accuracy=True):
    was = model.training;model.eval();correct = native_correct = 0;nll = 0.;work = native_work = 0
    synchronize(device);started = time.perf_counter()
    try:
        for offset in range(0, len(data["initial"]), 64):
            recorder.check();indices = np.arange(offset, min(offset + 64, len(data["initial"])))
            inputs, length, active = batch(data, indices, device)
            output = probability_forward(model, inputs, args.arm)
            if not bool(torch.isfinite(output).all()):raise FloatingPointError("nonfinite LM evaluation output")
            nll += float(objective(output, inputs["targets"], args.arm, "sum"))
            correct += int((output.argmax(-1) == inputs["targets"]).sum())
            work += proxy(model, args, len(indices), length, active)
            if args.arm == "block" and native_accuracy:
                native = probability_forward(model, inputs, args.arm, True)
                native_correct += int((native.argmax(-1) == inputs["targets"]).sum())
                native_work += proxy(model, args, len(indices), length, active, True)
        synchronize(device)
    finally:model.train(was)
    count = len(data["initial"]);mean = nll / count
    return {"sampled_target_count": count, "NLL_sum": nll, "next_character_NLL": mean,
        "sampled_character_PPL": math.exp(mean) if mean < 700 else None,
        "PPL_overflow": mean >= 700, "neural_next_character_accuracy": correct / count,
        "native_argmax_accuracy": native_correct / count if args.arm == "block" and native_accuracy else None,
        "native_PPL": None, "wall_seconds": time.perf_counter() - started,
        "forward_MAC_proxy": work, "native_forward_MAC_proxy": native_work,
        "scope": "fixed sampled next-character positions, not all corpus tokens; no native one-hot PPL"}


def rng_snapshot(device):
    npstate = np.random.get_state()
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
        "numpy": {"algorithm": str(npstate[0]), "keys": npstate[1].tolist(), "position": int(npstate[2]),
                  "has_gauss": int(npstate[3]), "cached_gaussian": float(npstate[4])}}


def save_model(model, optimizer, args, metadata, recorder, completed, receipts, costs, indices_sha, name):
    payload = {"schema": "char-context-development-v1", "args": plain(vars(args)),
        "metadata": metadata, "completed_updates": completed, "fixed_update_budget": 300,
        "model_state": model.state_dict(), "block_config": model.configuration() if args.arm == "block" else None,
        "baseline_spec": None if args.arm == "block" else model.spec,
        "parameter_flags": {n: p.requires_grad for n, p in model.named_parameters()},
        "optimizer": optimizer.state_dict(), "rng": rng_snapshot(next(model.parameters()).device), "input_receipts": receipts,
        "common_batch_indices_sha256": indices_sha, "costs": copy.deepcopy(costs),
        "source_sha256": source_hashes(),
        "torch_version": str(torch.__version__), "exact_training_resume_tested": False,
        "scope": "sampled-prefix LM development; no full-corpus PPL or frozen Primary claim"}
    path = recorder.out / name;started = time.perf_counter()
    with path.open("xb") as stream:torch.save(payload, stream)
    if path.stat().st_size > 100 << 20:raise RuntimeError("100MiB owned checkpoint cap; file retained")
    receipt = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256(path),
               "wall_seconds": time.perf_counter() - started, "completed_updates": completed}
    recorder.emit("actual_LM_checkpoint", **receipt)
    return path, receipt


def reload_audit(path, model, args, valid, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    other = build(args.arm, payload["metadata"], args.internal_states, device, args.block_initialization)
    other.load_state_dict(payload["model_state"], strict=True)
    if not tensor_state_equal(payload["model_state"], other.state_dict()):raise RuntimeError("LM tensor reload mismatch")
    indices = np.arange(min(64, len(valid["initial"])));inputs, length, active = batch(valid, indices, device)
    was = model.training;model.eval();other.eval()
    try:
        with torch.no_grad():
            left, right = probability_forward(model, inputs, args.arm), probability_forward(other, inputs, args.arm)
            if not torch.equal(left, right):raise RuntimeError("same-weight LM forward changed after actual disk reload")
    finally:model.train(was)
    return {"tensor_values_equal": True, "same_input_neural_output_equal": True,
            "rows": len(indices), "actual_forward_examples": 2 * len(indices),
            "forward_MAC_proxy": 2 * proxy(model, args, len(indices), length, active),
            "exact_training_resume_tested": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True);p.add_argument("--out", type=Path, required=True)
    p.add_argument("--arm", choices=("block", "gru", "transformer"), required=True)
    p.add_argument("--seed", type=int, default=0);p.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    p.add_argument("--max-seconds", type=float, default=850.)
    p.add_argument("--internal-states", type=int, default=None)
    p.add_argument("--block-initialization", choices=("iid", "carry"), default="iid")
    p.add_argument("--resource-overlap-note", default="coordinator must record concurrent jobs; not isolated throughput")
    args = p.parse_args()
    if args.out.exists() or not 1 <= args.max_seconds <= 850:raise ValueError("new output and <=850s bound required")
    rec = Recorder(args.out, args.max_seconds);device = torch.device(args.device)
    model = optimizer = None;completed = 0;inputs = [];curves = [];checkpoints = [];status = "started"
    costs = {"training_examples": 0, "training_input_characters": 0, "training_target_characters": 0,
             "optimizer_updates": 0, "training_wall_seconds": 0., "training_CPU_seconds": 0.,
             "batch_assembly_wall_seconds": 0.,
             "training_forward_MAC_proxy": 0, "evaluation_wall_seconds": 0.,
             "evaluation_forward_MAC_proxy": 0, "native_evaluation_forward_MAC_proxy": 0,
             "evaluation_neural_examples": 0, "evaluation_native_examples": 0,
             "checkpoint_wall_seconds": 0., "reload_wall_seconds": 0.,
             "reload_forward_examples": 0, "reload_forward_MAC_proxy": 0}
    final_path = None
    def count_evaluation(report):
        costs["evaluation_wall_seconds"] += report["wall_seconds"]
        costs["evaluation_forward_MAC_proxy"] += report["forward_MAC_proxy"]
        costs["native_evaluation_forward_MAC_proxy"] += report["native_forward_MAC_proxy"]
        costs["evaluation_neural_examples"] += report["sampled_target_count"]
        costs["evaluation_native_examples"] += report["sampled_target_count"] if args.arm == "block" else 0
    try:
        if device.type == "cuda" and not torch.cuda.is_available():raise RuntimeError("requested CUDA unavailable; no silent CPU fallback")
        torch.set_num_threads(1)
        if device.type == "cuda":torch.cuda.set_per_process_memory_fraction(.3, device)
        torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
        metadata = json.loads((args.data / "metadata.json").read_text());v = metadata["n_states"]
        if metadata.get("schema") != "sampled-char-context-terminal-v1" or not 2 <= v <= 512 or metadata["n_tokens"] != v or metadata["max_length"] != 128 or len(metadata["stages"]) != 1:
            raise ValueError("frozen sampled character development metadata required")
        args.states = v
        if args.internal_states is None:args.internal_states = v
        if not 1 <= args.internal_states <= 512:raise ValueError("bounded Block width required")
        train, receipt = frozen_split(args, metadata, "train");inputs.append(receipt)
        valid, receipt = frozen_split(args, metadata, "valid");inputs.append(receipt)
        indices = np.random.default_rng(np.random.SeedSequence([args.seed, v, 300, 0x4c4d])).integers(0, 8192, size=(300, 64), dtype=np.int64)
        np.savez_compressed(rec.out / "common-batch-indices.npz", indices=indices)
        indices_sha = sha256(rec.out / "common-batch-indices.npz")
        model = build(args.arm, metadata, args.internal_states, device, args.block_initialization)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        if device.type == "cuda":torch.cuda.reset_peak_memory_stats(device)
        rec.emit("configuration", args=vars(args), vocabulary_size=v, metadata_sha256=sha256(args.data / "metadata.json"),
            common_batch_indices_sha256=indices_sha, input_receipts=inputs, resources=resource_state(model, optimizer, device),
            fixed_budget={"updates": 300, "batch": 64, "lr": .003, "optimizer": "Adam", "clip_norm": 1.},
            training_execution="SOFT only; native Block is accuracy-only evaluation", no_test_selection=True,
            block_initialization=args.block_initialization if args.arm == "block" else None,
            formal_Primary=False, validity="checked implementation/information/prospective protocol, independent of achieved accuracy",
            source_sha256=source_hashes(), torch_version=str(torch.__version__), CUDA_version=torch.version.cuda,
            device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "explicit CPU request",
            metric_scope="sampled next-character conditional PPL; not full-corpus/word PPL")
        for step in range(301):
            rec.check()
            if step % 50 == 0:
                report = evaluate(model, valid, args, device, rec)
                count_evaluation(report)
                curves.append({"step": step, "VALID": report});rec.emit("VALID_curve", step=step, metrics=report, costs=costs)
                if step and step < 300:
                    _, receipt = save_model(model, optimizer, args, metadata, rec, step, inputs, costs, indices_sha, f"{args.arm}-step-{step:03d}.pt")
                    checkpoints.append(receipt)
                    costs["checkpoint_wall_seconds"] += receipt["wall_seconds"]
            if step == 300:break
            batch_started = time.perf_counter();data, length, active = batch(train, indices[step], device)
            costs["batch_assembly_wall_seconds"] += time.perf_counter() - batch_started
            model.train();synchronize(device);started = time.perf_counter();cpu = time.process_time()
            optimizer.zero_grad(set_to_none=True)
            output = probability_forward(model, data, args.arm);loss = objective(output, data["targets"], args.arm)
            if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite sampled-char training loss")
            loss.backward();grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if any(not bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):raise FloatingPointError("nonfinite LM parameter after update")
            synchronize(device);elapsed = time.perf_counter() - started
            completed = step + 1;costs["training_examples"] += 64;costs["training_input_characters"] += active
            costs["training_target_characters"] += 64;costs["optimizer_updates"] += 1
            costs["training_wall_seconds"] += elapsed;costs["training_CPU_seconds"] += time.process_time() - cpu
            costs["training_forward_MAC_proxy"] += proxy(model, args, 64, length, active)
            rec.emit("actual_LM_update", step=completed, loss=float(loss.detach()), preclip_grad_norm=float(grad_norm),
                     training_next_char_accuracy=float((output.argmax(-1) == data["targets"]).float().mean()),
                     wall_seconds=elapsed, cumulative_costs=costs)
        final_path, receipt = save_model(model, optimizer, args, metadata, rec, completed, inputs, costs, indices_sha, f"{args.arm}-frozen-final.pt")
        checkpoints.append(receipt)
        costs["checkpoint_wall_seconds"] += receipt["wall_seconds"]
        final = {"VALID": copy.deepcopy(curves[-1]["VALID"])}
        # First opening of any TEST file follows all300 fixed updates and disk freeze.
        for split in ("test_iid", "test_unseen", "test_long"):
            data, receipt = frozen_split(args, metadata, split);inputs.append(receipt)
            final[split] = evaluate(model, data, args, device, rec)
            count_evaluation(final[split])
        audit_started = time.perf_counter();audit = reload_audit(final_path, model, args, valid, device)
        synchronize(device);costs["reload_wall_seconds"] += time.perf_counter() - audit_started
        costs["reload_forward_examples"] += audit["actual_forward_examples"]
        costs["reload_forward_MAC_proxy"] += audit["forward_MAC_proxy"]
        rec.json("summary.json", {"status": "completed_300_update_LM_development", "arm": args.arm,
            "completed_updates": completed, "curves": curves, "final": final, "reload_audit": audit,
            "costs": costs, "resources": resource_state(model, optimizer, device), "checkpoints": checkpoints,
            "inputs": inputs, "common_batch_indices_sha256": indices_sha, "formal_Primary": False,
            "driver_wall_seconds": time.perf_counter() - rec.started,
            "cost_scope": "nested phase timings and forward-only MAC proxies; external runner includes process/import overhead",
            "metric_scope": "512 sampled paired TEST positions, context32/64/128; not full-corpus PPL",
            "resource_overlap_note": args.resource_overlap_note})
        status = "completed"
    except Exception as error:
        status = "failed_or_bounded_incomplete"
        rec.emit("actual_LM_failure", error=f"{type(error).__name__}: {error}", completed_updates=completed)
        (rec.out / "exception.txt").write_text(traceback.format_exc())
        if model is not None:
            with (rec.out / "partial-actual-neural-state.pt").open("xb") as stream:
                torch.save({"model_state": model.state_dict(), "completed_updates": completed,
                            "scope": "actual partial tensors; not successful training or exact resume"}, stream)
        rec.json("failure.json", {"status": status, "completed_updates": completed, "error": f"{type(error).__name__}: {error}",
            "costs": costs, "checkpoints": checkpoints, "inputs": inputs, "curves": curves})
    finally:
        rec.stream.close()
        print(json.dumps({"status": status, "out": str(args.out), "completed_updates": completed}))
    if status != "completed":raise SystemExit(1)


if __name__ == "__main__":main()
