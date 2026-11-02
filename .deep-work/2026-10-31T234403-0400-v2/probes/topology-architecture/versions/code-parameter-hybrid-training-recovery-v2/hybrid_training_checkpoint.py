"""Full CPU training checkpoints for the owned, explicit-SGD Hybrid session.

No run at import. This adds external sampling state to the existing inference
checkpoint without changing the model, frozen V1/V2 artifacts, or Method.
Only checkpoints produced from checked, owned inputs are supported. Loading is
weights_only=True; no saved Python model code or callback is deserialized.

The restored training ledger retains its saved counters/limits. Physical load
overhead is returned separately, never hidden by that logical state restore.
An old inference checkpoint has no original sampling state: this helper cannot
recover it and rejects it as a full training checkpoint.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import torch

import hybrid_loop_block
import loop_boundary_learning
from hybrid_loop_block import HybridLearningSession
from loop_boundary_learning import CostLedger


SCHEMA = "owned-hybrid-full-training-checkpoint-v1"
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def rng_fingerprints(rng: random.Random) -> dict[str, str]:
    return {
        "external_python": hashlib.sha256(json.dumps(rng.getstate()).encode()).hexdigest(),
        "torch_cpu": hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest(),
    }


def runtime_configuration() -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "default_dtype": str(torch.get_default_dtype()),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "device_scope": "CPU only; no CUDA RNG/generator state is covered",
    }


def source_fingerprints() -> dict[str, str]:
    paths = (Path(__file__), Path(hybrid_loop_block.__file__),
             Path(loop_boundary_learning.__file__))
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("source leaf names must be unique")
    return {path.name: file_sha256(path) for path in paths}


def tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    if tensor.device.type != "cpu":
        raise ValueError("this checkpoint helper supports the actual CPU session only")
    raw = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
            "sha256": hashlib.sha256(bytes(raw.tolist())).hexdigest()}


def capture_training_state(session: HybridLearningSession, rng: random.Random, *,
                           next_iteration: int, training_control: Mapping[str, Any],
                           provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Capture at a committed step boundary, with no pending gradient graph.

    training_control must identify the externally owned iteration/algorithm
    policy. Objective callbacks themselves are not saved or executed on load.
    The caller supplies and verifies the terminal-example digest on resumption.
    """
    if not isinstance(rng, random.Random) or next_iteration < 0:
        raise ValueError("an explicit external Random and nonnegative next iteration are required")
    state = session.checkpoint()
    if state.get("optimizer") != "explicit SGD; no hidden optimizer tensors":
        raise ValueError("unsupported optimizer: its state must not be silently omitted")
    named = dict(session.model.named_parameters())
    buffers = dict(session.model.named_buffers())
    tensors = {**named, **buffers}
    if set(tensors) != set(state["state_dict"]):
        raise ValueError("aliased or extra model state needs an explicit recovery implementation")
    layouts = {name: tensor_fingerprint(tensor) for name, tensor in tensors.items()}
    result = {
        "schema": SCHEMA,
        "session": state,
        "external_python_rng_state": copy.deepcopy(rng.getstate()),
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "rng_fingerprints": rng_fingerprints(rng),
        "parameter_requires_grad": {name: p.requires_grad for name, p in named.items()},
        "parameter_gradients": {name: None if p.grad is None else p.grad.detach().cpu().clone()
                                for name, p in named.items()},
        "module_training_flags": {name: module.training for name, module in session.model.named_modules()},
        "tensor_layouts": layouts,
        "next_iteration": int(next_iteration),
        "training_control": _json_copy(training_control),
        "source_sha256": source_fingerprints(),
        "runtime": runtime_configuration(),
        "logical_training_costs": copy.deepcopy(session.ledger.snapshot()),
        "remaining_wall_seconds_at_capture": max(0.0, session.ledger.deadline - time.monotonic()),
        "provenance": _json_copy(provenance),
        "scope": "committed CPU explicit-SGD session plus external sampling/control state; no original missing history reconstructed",
    }
    return result


def save_training_checkpoint(path: Path, session: HybridLearningSession, rng: random.Random, *,
                             next_iteration: int, training_control: Mapping[str, Any],
                             provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Write a NEW path; IO is separate from logical training-state counters."""
    started = time.perf_counter()
    payload = capture_training_state(session, rng, next_iteration=next_iteration,
                                     training_control=training_control, provenance=provenance)
    path = Path(path)
    with path.open("xb") as stream:
        torch.save(payload, stream)
    if path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("new owned training checkpoint exceeds the bounded format")
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": file_sha256(path), "wall_seconds": time.perf_counter() - started,
            "model_parameter_hash": payload["session"]["parameter_hash"],
            "rng_fingerprints": payload["rng_fingerprints"],
            "next_iteration": next_iteration, "logical_costs": payload["logical_training_costs"],
            "accounting": "new audit IO work; not an additional optimizer/parameter update"}


def _restore_layout(tensor: torch.Tensor, saved: torch.Tensor, layout: Mapping[str, Any]) -> None:
    shape, strides = tuple(layout["shape"]), tuple(layout["stride"])
    if tensor.device.type != "cpu" or tuple(saved.shape) != shape or tuple(tensor.shape) != shape:
        raise ValueError("owned tensor shape/device mismatch")
    if str(saved.dtype) != layout["dtype"] or tensor.dtype != saved.dtype:
        raise ValueError("owned tensor dtype mismatch")
    if not 1 <= len(shape) <= 2 or tensor.numel() > 1_000_000 or any(s <= 0 for s in strides):
        raise ValueError("only bounded positive-stride owned vectors/matrices are supported")
    offsets = [0]
    for size, stride in zip(shape, strides):
        offsets = [offset + i * stride for offset in offsets for i in range(size)]
    if len(set(offsets)) != tensor.numel() or max(offsets, default=0) > 4_000_000:
        raise ValueError("overlapping or oversized tensor layout")
    with torch.no_grad():
        if tuple(tensor.stride()) != strides:
            tensor.data = torch.empty_strided(shape, strides, dtype=saved.dtype, device="cpu")
        tensor.copy_(saved)
    if tensor_fingerprint(tensor) != dict(layout):
        raise ValueError("restored tensor bits/strides do not match the saved layout")


def restore_training_checkpoint(path: Path, ledger: CostLedger, *,
                                source_directory: Path | None = None
                                ) -> tuple[HybridLearningSession, random.Random, dict[str, Any], dict[str, Any]]:
    """Restore all supported training state; return new physical-load overhead.

    The caller's deadline remains an external bound. Saved remaining wall time
    can tighten it, but cannot enlarge it. Logical counters are restored so
    remaining count budgets are preserved. Constructor/load work is reported
    separately even though logical training counters return to their old values.
    """
    started = time.perf_counter()
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("bounded owned full-training checkpoint required")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("inference-only checkpoint lacks training RNG/control state")
    source_directory = Path(source_directory or Path(__file__).parent)
    for name, expected in payload["source_sha256"].items():
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError("source provenance must use trusted code-directory leaf names")
        if file_sha256(source_directory / name) != expected:
            raise ValueError(f"training code changed: {name}")
    if runtime_configuration() != payload["runtime"]:
        raise ValueError("exact training recovery requires the same recorded CPU runtime/configuration")
    saved_costs = payload["logical_training_costs"]
    if asdict(ledger.limits) != saved_costs["limits"]:
        raise ValueError("count/resource limits differ from the saved training state")
    counts_before, seconds_before = dict(ledger.counts), dict(ledger.seconds)
    session = HybridLearningSession.restore(payload["session"], ledger)
    session.metadata = copy.deepcopy(payload["session"]["metadata"])
    capacity = int(payload["session"]["cache_storage"]["capacity"])
    if not 1 <= capacity <= 128 or len(session.bank.cache) > capacity:
        raise ValueError("saved derived-map cache capacity is inconsistent")
    session.bank.capacity = capacity
    tensors = {**dict(session.model.named_parameters()), **dict(session.model.named_buffers())}
    if set(tensors) != set(payload["tensor_layouts"]):
        raise ValueError("restored state names differ")
    for name, tensor in tensors.items():
        _restore_layout(tensor, payload["session"]["state_dict"][name], payload["tensor_layouts"][name])
    named = dict(session.model.named_parameters())
    if set(named) != set(payload["parameter_requires_grad"]) or set(named) != set(payload["parameter_gradients"]):
        raise ValueError("parameter flag/gradient metadata incomplete")
    for name, parameter in named.items():
        parameter.requires_grad_(bool(payload["parameter_requires_grad"][name]))
        gradient = payload["parameter_gradients"][name]
        if gradient is not None and (gradient.shape != parameter.shape or gradient.dtype != parameter.dtype):
            raise ValueError("saved pending gradient shape/dtype mismatch")
        parameter.grad = None if gradient is None else gradient.clone()
    modules = dict(session.model.named_modules())
    if set(modules) != set(payload["module_training_flags"]):
        raise ValueError("module training flags incomplete")
    for name, module in modules.items():
        module.training = bool(payload["module_training_flags"][name])
    session.model.forward_counts = copy.deepcopy(payload["session"]["forward_counts"])
    overhead = {"counts": {key: value - counts_before.get(key, 0)
                           for key, value in ledger.counts.items()
                           if value != counts_before.get(key, 0)},
                "seconds": {key: value - seconds_before.get(key, 0.0)
                            for key, value in ledger.seconds.items()
                            if value != seconds_before.get(key, 0.0)}}
    ledger.counts = copy.deepcopy(saved_costs["counts"])
    ledger.seconds = copy.deepcopy(saved_costs["seconds"])
    ledger.deadline = min(ledger.deadline, time.monotonic() + payload["remaining_wall_seconds_at_capture"])
    rng = random.Random()
    rng.setstate(payload["external_python_rng_state"])
    torch.set_rng_state(payload["torch_cpu_rng_state"])
    if rng_fingerprints(rng) != payload["rng_fingerprints"]:
        raise ValueError("restored random-generator state mismatch")
    receipt = {"path": str(path), "sha256": file_sha256(path),
               "bytes": path.stat().st_size, "wall_seconds": time.perf_counter() - started,
               "physical_restore_overhead": overhead,
               "logical_costs_restored": copy.deepcopy(saved_costs),
               "rng_fingerprints": rng_fingerprints(rng),
               "model_parameter_hash": session.model.parameter_hash(),
               "next_iteration": payload["next_iteration"],
               "provenance": payload["provenance"],
               "scope": "training-state restoration; trajectory equality requires the actual next-step diagnostic"}
    control = {"next_iteration": payload["next_iteration"],
               "training_control": copy.deepcopy(payload["training_control"])}
    return session, rng, control, receipt
