"""ROOT-RUN exact-next-step recovery diagnostic; no execution at import.

Use an owned pre-admission V2 step-030.pt (or a recorded earlier sibling) and
a NEW output directory. This creates a NEW audited fork with fresh explicit
RNG seeds. It does NOT recover/reconstruct the missing original V2 RNG history.
Exactly three new terminal-gradient calculations are allowed: warm step, next
continuous step, and the same next step after actual disk reload. CPU only;
<=90s internally, caller MUST impose an outside120s bound and isolation.

No TEST selection, training curriculum change, added model parameters, network,
SSH, downloads, subprocesses, or modification of an existing checkpoint.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time
import traceback
from typing import Any

import torch

from hybrid_loop_block import (HybridLearningSession, HybridTerminalObjective,
                               coupled_gradient, native_output)
from hybrid_training_checkpoint import (file_sha256,
                                        restore_training_checkpoint, rng_fingerprints,
                                        save_training_checkpoint, tensor_fingerprint)
from loop_boundary_learning import CostLedger, Limits, ProgramExample, _diagnostic_data


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def tensor_groups(session: HybridLearningSession) -> dict[str, Any]:
    return {name: tensor_fingerprint(tensor) for name, tensor in session.model.state_dict().items()}


def semantic_state(session: HybridLearningSession) -> dict[str, Any]:
    checkpoint = session.checkpoint()
    return {key: checkpoint[key] for key in (
        "family", "reference_native_contract", "protected_operations", "admission_receipts",
        "derived_map_cache", "map_namespace", "semantic_license_hash", "transactions",
        "configuration", "forward_counts", "parameters_total", "buffer_scalars",
    )}


def owned_input(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("bounded existing owned hybrid checkpoint required")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != "hybrid-loop-v1" or payload.get("parameters_total") != 30:
        raise ValueError("this audit requires the actual fixed-D3 30-P hybrid format")
    if payload.get("configuration", {}).get("transaction_policy") != "block_fallback":
        raise ValueError("this audit requires the already executed V2 policy")
    return payload


def choose_source(requested: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    """Only admission eligibility selects a source, never accuracy/test output."""
    requested = requested.resolve(strict=True)
    candidates = [requested]
    candidates += sorted(requested.parent.glob("step-*.pt"), reverse=True)[:12]
    candidates += [requested.parent / "initial.pt"]
    notes = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        payload = owned_input(path)
        admitted = payload["protected_operations"]
        if 1 in admitted:
            notes.append({"path": str(path), "sha256": file_sha256(path),
                          "decision": "not eligible: operation1 is already admitted; no class step permitted"})
            continue
        if 0 not in admitted or payload.get("reference_native_contract") is None:
            raise ValueError("source lacks an actual admitted old operation/binding")
        notes.append({"path": str(path), "sha256": file_sha256(path),
                      "decision": "selected actual pre-admission state; no history/RNG reconstruction"})
        return path, payload, notes
    raise ValueError("no existing pre-admission sibling; no new checkpoint/history will be invented")


def delta_cost(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    return {kind: {key: value - before[kind].get(key, 0)
                   for key, value in after[kind].items()
                   if value != before[kind].get(key, 0)} for kind in ("counts", "seconds")}


def run_one_step(session: HybridLearningSession, rng: random.Random,
                 training: tuple[ProgramExample, ...], old: tuple[ProgramExample, ...],
                 iteration: int, learning_rate: float) -> dict[str, Any]:
    before = session.ledger.snapshot()
    started = time.perf_counter()
    objective = HybridTerminalObjective(session.model, training, session.ledger)
    gradients = coupled_gradient(session.model, session.bank, objective, 1,
                                  session.kernel, rng, path_estimator="event", within_samples=1)
    gradient_bits = {kind: {name: tensor_fingerprint(tensor) for (name, _), tensor in
                            zip(session.model.named_parameters(), gradients[kind])}
                     for kind in ("parameter_gradients", "within_parameter_gradients", "class_parameter_gradients")}
    commit = session.transaction(gradients["parameter_gradients"], operation=1,
                                  learning_rate=learning_rate)
    train_eval, old_eval = session.evaluate(training), session.evaluate(old)
    if old_eval["native_accuracy"] != 1.0 or old_eval["soft_accuracy"] != 1.0:
        raise RuntimeError("a committed recovery step violated the old observed contract")
    result = {"new_audit_iteration": iteration, "gradient_record": gradients["record"],
              "gradient_tensor_bits": gradient_bits,
              "transaction": commit, "training_evaluation": train_eval,
              "old_evaluation": old_eval, "model_tensor_bits": tensor_groups(session),
              "execution_flags": {
                  "parameter_requires_grad": {name: p.requires_grad for name, p in session.model.named_parameters()},
                  "module_training": {name: m.training for name, m in session.model.named_modules()},
              },
              "rng_fingerprints": rng_fingerprints(rng),
              "semantic_state": semantic_state(session),
              "cost_delta": delta_cost(before, session.ledger.snapshot()),
              "wall_seconds": time.perf_counter() - started}
    session.ledger.emit("new_recovery_audit_step", **result)
    return result


def comparable(step: dict[str, Any]) -> dict[str, Any]:
    # Real-time measurements must differ; they are separately preserved.
    transaction = step["transaction"]
    return {key: step[key] for key in (
        "gradient_record", "gradient_tensor_bits", "training_evaluation", "old_evaluation",
        "model_tensor_bits", "execution_flags", "rng_fingerprints", "semantic_state",
    )} | {"chosen_transaction": {key: transaction[key] for key in (
        "accepted_scale", "accepted_block", "parameter_hash", "forming_action",
        "soft_native_path_bound", "old_loop_max_vertex_displacement",
    )}, "step_work_counts": step["cost_delta"]["counts"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3901)
    parser.add_argument("--learning-rate", type=float, default=.05)
    parser.add_argument("--seconds", type=float, default=90.)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 90 or not 0 < args.learning_rate <= .1:
        raise ValueError("bounded CPU seconds and actual existing SGD range required")
    args.out.mkdir(parents=True, exist_ok=False)
    audit_started = time.perf_counter()
    deadline = time.monotonic() + args.seconds
    result: dict[str, Any] = {"status": "started", "MAIN": "TOPOLOGY_USER_LOCKED",
        "purpose": "training-state integrity check; no architecture advantage or benchmark tuning",
        "source_requested": str(args.checkpoint), "fresh_audit_seed": args.seed,
        "original_missing_rng_recovered": False, "new_gradient_calculation_limit": 3,
        "internal_seconds": args.seconds, "external_timeout_seconds_required": 120,
        "device": "CPU only", "new_work": {}}
    streams = []
    def new_ledger(name: str) -> CostLedger:
        stream = (args.out / f"{name}.jsonl").open("x", encoding="utf-8")
        streams.append(stream)
        def write(row: dict[str, Any]) -> None:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
        return CostLedger(limits, deadline, writer=write)
    try:
        source, original, choices = choose_source(args.checkpoint)
        result.update(source_selected=str(source), source_sha256=file_sha256(source),
                      source_selection=choices,
                      source_transactions=int(original["transactions"]),
                      source_format="inference only; original external RNG and flags were not saved",
                      fork_flags="this new audited fork explicitly enables every model gradient")
        limits = Limits(max_feedback=3000, max_programs=600, max_continuation_steps=60000,
                        max_step_attempts=100000, max_events=1000, max_updates=4)
        ledger = new_ledger("continuous")
        initialization_started = time.perf_counter()
        session = HybridLearningSession.restore(original, ledger)
        for parameter in session.model.parameters():
            parameter.requires_grad_(True)
        session.model.train(True)
        rng = random.Random(args.seed)
        torch.manual_seed(args.seed)
        training, _unused_heldout = _diagnostic_data()
        del _unused_heldout
        training = tuple(training)
        training_sha = hashlib.sha256(canonical([asdict(e) for e in training]).encode()).hexdigest()
        old = []
        for initial in range(3):
            for name, word in (("empty", ()), ("old", (0,))):
                item = ProgramExample(f"owned-old-{name}-{initial}", initial, word, 0)
                label = native_output(session.reference, item, ledger)
                old.append(ProgramExample(item.name, initial, word, label))
        old = tuple(old)
        result["new_fork_initialization"] = {
            "wall_seconds": time.perf_counter() - initialization_started,
            "costs": ledger.snapshot(),
            "scope": "new source restore and own-old example derivation; excludes historical source training",
        }
        result["protocol"] = {"training_rows": [asdict(e) for e in training],
                              "training_sha256": training_sha, "heldout_accesses": 0,
                              "limits": asdict(limits), "learning_rate": args.learning_rate,
                              "source_model_parameters": session.model.parameter_count(),
                              "seed_provenance": "fresh independent audit, not original V2 continuation"}
        result["warm_step"] = run_one_step(session, rng, training, old, 0, args.learning_rate)
        control = {"operation": 1, "path_estimator": "event", "within_samples": 1,
                   "selected_gradient": "parameter_gradients", "learning_rate": args.learning_rate,
                   "training_sha256": training_sha, "admission_or_maintenance": "none in this integrity check",
                   "driver_sha256": file_sha256(Path(__file__))}
        checkpoint_path = args.out / "full-training-after-one-new-step.pt"
        result["save_receipt"] = save_training_checkpoint(
            checkpoint_path, session, rng, next_iteration=1, training_control=control,
            provenance={"kind": "new independently seeded recovery audit", "source": str(source),
                        "source_sha256": file_sha256(source), "fresh_seed": args.seed,
                        "original_missing_rng_recovered": False,
                        "historic_source_transactions": int(original["transactions"])})
        result["continuous_next"] = run_one_step(session, rng, training, old, 1, args.learning_rate)
        restored_ledger = new_ledger("restored")
        restored, restored_rng, restored_control, restore_receipt = restore_training_checkpoint(
            checkpoint_path, restored_ledger)
        result["restore_receipt"] = restore_receipt
        if restored_control != {"next_iteration": 1, "training_control": control}:
            raise RuntimeError("restored iteration/control does not match the actually saved boundary")
        result["restored_next"] = run_one_step(restored, restored_rng, training, old, 1, args.learning_rate)
        left, right = comparable(result["continuous_next"]), comparable(result["restored_next"])
        checks = {key: canonical(left[key]) == canonical(right[key]) for key in left}
        result["checks"] = checks
        result["no_new_architecture_parameters"] = session.model.parameter_count() == restored.model.parameter_count() == 30
        if not all(checks.values()) or not result["no_new_architecture_parameters"]:
            raise RuntimeError("actual continuous-versus-reloaded next step differs; preserve all raw evidence")
        result["new_work"] = {
            "gradient_calculations": 3,
            "warm_step": result["warm_step"]["cost_delta"],
            "continuous_next": result["continuous_next"]["cost_delta"],
            "restored_next": result["restored_next"]["cost_delta"],
            "save_wall_seconds": result["save_receipt"]["wall_seconds"],
            "restore_wall_seconds": restore_receipt["wall_seconds"],
            "restore_constructor_counts": restore_receipt["physical_restore_overhead"],
            "accounting": "three genuinely new gradient calculations; saved logical history is not new work; timing scopes may overlap",
        }
        # Preserve both actually trained states, separately from their equality claim.
        for name, actual, actual_rng in (("continuous-final", session, rng),
                                          ("restored-final", restored, restored_rng)):
            result[f"{name}-checkpoint"] = save_training_checkpoint(
                args.out / f"{name}.pt", actual, actual_rng, next_iteration=2,
                training_control=control, provenance={"kind": "new recovery audit endpoint",
                                                      "source_checkpoint": str(checkpoint_path)})
        result["status"] = "actual_next_step_recovery_passed"
    except Exception as error:
        result["status"] = "recovery_check_failed_or_blocked"
        result["error"] = f"{type(error).__name__}: {error}"
        (args.out / "exception.txt").write_text(traceback.format_exc())
    finally:
        for stream in streams:
            stream.close()
        result["wall_seconds"] = time.perf_counter() - audit_started
        if not result["new_work"]:
            result["new_work"] = {
                "completed_gradient_steps": [name for name in ("warm_step", "continuous_next", "restored_next")
                                             if name in result],
                "costs_of_completed_steps": {name: result[name]["cost_delta"]
                                            for name in ("warm_step", "continuous_next", "restored_next")
                                            if name in result},
                "scope": "partial failures remain in raw ledgers; no missing step or cost is fabricated",
            }
        result["wall_scope"] = "new audit only; source V2 training is historical and not relabeled"
        (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": result["status"], "out": str(args.out),
                          "checks": result.get("checks"), "error": result.get("error")}))
    if result["status"] != "actual_next_step_recovery_passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
