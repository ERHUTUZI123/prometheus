"""ROOT-RUN bounded integrated mechanism diagnostic; no import-time execution.

Same V2, MAIN=TOPOLOGY USER_LOCKED. No network, SSH, subprocess or installation.
Root owns execution/freeze. Use an external 120-second process timeout. This
driver has a shared <=90-second internal deadline, <=100 real parameter updates
per arm, and explicit feedback/program/continuation bounds. It does not change
the six-stage study, create a new campaign, or establish M0/general superiority.

Arms: full coupled, within-only, class-only. BOTH gradient components are always
computed and charged before selecting the applied tensor. Their work counts
must not be used as an efficient implementation comparison. NEW routing starts
uniform; old ability and near-identity E/R are disclosed supplied starting state.
Held-out records are evaluated only AFTER the actual final weights are frozen
and saved; no held-out outcome controls training, retry, or model selection.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
import traceback
from typing import Any, Sequence

import torch

from loop_boundary_learning import (
    BudgetStop, CostLedger, Limits, NumericalFailure, ProgramExample, _diagnostic_data,
)
from hybrid_loop_block import (
    HybridLearningSession, HybridTerminalObjective, SOFT_NATIVE_MIN_MASS,
    coupled_gradient, native_output,
)


def source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_parameters(session: HybridLearningSession) -> dict[str, list[float]]:
    return {name: value.detach().cpu().reshape(-1).tolist()
            for name, value in session.model.named_parameters()}


def parameter_displacements(before: dict[str, list[float]],
                            after: dict[str, list[float]]) -> dict[str, float]:
    return {name: math.sqrt(sum((a - b) ** 2 for a, b in zip(values, after[name])))
            for name, values in before.items()}


def own_native_confidence_maintenance(session: HybridLearningSession,
                                     forming_operation: int = 1) -> dict[str, Any]:
    """ONE own-class repair; no task labels enter its targets and P is unchanged.

    Current hard call choices generate targets; current hard E/R generate their
    own sharpened targets. Fit call logits by LS on CURRENT shared features.
    Shared/loop-head weights stay exactly fixed. Install all changes through the
    SAME primitive/binding/confidence transaction, so no correct route is donated.
    Both ablation arms are allowed the same repair under the same TRAIN trigger.
    """
    ledger, model = session.ledger, session.model
    started = time.perf_counter()
    before_hash, before_count = model.parameter_hash(), model.parameter_count()
    ledger.add("own_confidence_maintenance_attempts")
    with torch.no_grad():
        hidden = model.shared(model.token_features)
        ledger.add("maintenance_shared_feature_matrices")
        targets: dict[str, torch.Tensor] = {}
        fits = []
        for slot, head in enumerate(model.calls):
            current_logits = head(hidden)
            own_choices = current_logits.argmax(-1)
            # A logit gap of 12 is enough for the length16 path-mass contract;
            # using +/-12 would require +/-24 differences in the two-feature
            # linear fit even before learning, outside the existing NN box.
            desired = torch.full_like(current_logits, -6.0)
            desired.scatter_(1, own_choices[:, None], 6.0)
            ledger.add("maintenance_least_squares_solves")
            solution = torch.linalg.lstsq(hidden, desired).solution.T
            fitted = hidden @ solution.T
            residual = float((fitted - desired).abs().max())
            preserved = bool(torch.equal(fitted.argmax(-1), own_choices))
            fits.append({"slot": slot, "own_choices": own_choices.cpu().tolist(),
                         "maximum_fit_residual": residual, "choices_preserved": preserved})
            if not preserved or not bool(torch.isfinite(solution).all()) or residual > 1e-6:
                ledger.elapsed("own_confidence_maintenance", started)
                return {"attempted": True, "installed": False,
                        "reason": "current shared features cannot fit own hard calls", "fits": fits}
            if float(solution.abs().max()) > 16.0:
                ledger.elapsed("own_confidence_maintenance", started)
                return {"attempted": True, "installed": False,
                        "reason": "own LS target leaves existing NN parameter box", "fits": fits}
            targets[f"calls.{slot}.weight"] = solution
        for name, logits in (("encoder_logits", model.encoder_logits),
                             ("readout_logits", model.readout_logits)):
            own_choices = logits.argmax(-1)
            desired = torch.full_like(logits, -6.0)
            desired.scatter_(1, own_choices[:, None], 6.0)
            targets[name] = desired
        # No teacher labels or desired semantic permutation is read here.
        gradients = []
        for name, parameter in model.named_parameters():
            target = targets.get(name)
            gradients.append(torch.zeros_like(parameter) if target is None
                             else (parameter.detach() - target) / 0.05)
    commit = session.transaction(gradients, operation=forming_operation, learning_rate=0.05)
    if model.parameter_count() != before_count:
        raise NumericalFailure("own-class confidence repair changed allocated parameter count")
    ledger.add("own_confidence_maintenance_installs", int(commit["accepted_scale"] > 0))
    ledger.elapsed("own_confidence_maintenance", started)
    result = {"attempted": True, "installed": commit["accepted_scale"] > 0,
              "fits": fits, "transaction": commit, "parameters_before": before_count,
              "parameters_after": model.parameter_count(), "hash_before": before_hash,
              "hash_after": model.parameter_hash(), "external_target_information": 0,
              "targets": "own current hard calls and E/R; shared and loop head fixed"}
    ledger.emit("own_confidence_maintenance", **result)
    return result


def save_actual_checkpoint(session: HybridLearningSession, path: Path,
                           tag: str) -> dict[str, Any]:
    """Small bounded delivery operation also survives exhausted learning budget."""
    if path.exists():
        raise FileExistsError("checkpoint paths must be new within this new run")
    started = time.perf_counter()
    payload = session.checkpoint()
    torch.save(payload, path)
    # Preservation is allowed after a learning deadline. Record directly rather
    # than invoking a budget check that could conceal a successfully saved file.
    session.ledger.counts["actual_checkpoint_writes"] = (
        session.ledger.counts.get("actual_checkpoint_writes", 0) + 1)
    session.ledger.seconds["checkpoint_delivery"] = (
        session.ledger.seconds.get("checkpoint_delivery", 0.0) + time.perf_counter() - started)
    return {"tag": tag, "path": str(path), "bytes": path.stat().st_size,
            "parameter_hash": payload["parameter_hash"],
            "parameters_total": payload["parameters_total"],
            "buffer_scalars": payload["buffer_scalars"], "cache_storage": payload["cache_storage"]}


def run_arm(arm: str, directory: Path, limits: Limits, arm_deadline: float,
            *, seed: int, learning_rate: float, checkpoint_every: int = 10) -> dict[str, Any]:
    directory.mkdir(exist_ok=False)
    raw = (directory / "records.jsonl").open("x", encoding="utf-8")
    def write(record: dict[str, Any]) -> None:
        raw.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        raw.flush()
    ledger = CostLedger(limits, arm_deadline, writer=write)
    session = HybridLearningSession(ledger, slots=1, sigma=2.0, new_route_seeded=False)
    rng = random.Random(seed)
    training, _withheld_unused = _diagnostic_data()
    # The training phase does not inspect, evaluate, select on, or log any
    # held-out target. The private helper's second tuple is discarded here.
    del _withheld_unused
    selected_key = {"full_coupled": "parameter_gradients",
                    "within_only": "within_parameter_gradients",
                    "class_only": "class_parameter_gradients"}[arm]
    started = time.perf_counter()
    results: dict[str, Any] = {"arm": arm, "status": "started", "seed": seed,
        "learning_rate": learning_rate, "parameters_total": session.model.parameter_count(),
        "buffer_scalars": sum(buffer.numel() for buffer in session.model.buffers()),
        "new_call_initialization": "uniform", "encoding_readout_initialization": "near identity, trainable",
        "old_ability_initialization": "supplied starting loop and call; measured by actual continuation",
        "unused_gradient_component_computed_and_charged": True,
        "update_counter_scope": "accepted parameter transactions, including zero-step/no-op commits; displacements reported",
        "efficiency_claim_from_ablation_work": False, "training": [asdict(e) for e in training],
        "heldout_selection": False, "checkpoints": [], "curves": [], "maintenance": {"attempted": False}}
    old_examples: tuple[ProgramExample, ...] = ()
    initial_parameters = snapshot_parameters(session)
    maintenance_attempted = False
    admitted = False
    # Reserve a small bounded slice of this arm's allocation for final actual
    # checkpoint/reload inference. This is a run limit, not an extra retry.
    learning_deadline = arm_deadline - 2.0
    try:
        contract = session.initialize_old_contract()
        reference = contract["reference"]
        derived = []
        for initial in range(3):
            for name, word in (("empty", ()), ("old", (0,))):
                placeholder = ProgramExample(f"old-derived-{name}-{initial}", initial, word, 0)
                target = native_output(reference, placeholder, ledger)
                derived.append(ProgramExample(placeholder.name, initial, word, target))
        old_examples = tuple(derived)
        ledger.emit("frozen_hybrid_arm", arm=arm, limits=asdict(limits), metadata=session.metadata,
                    initial_parameters=initial_parameters, old_contract=contract,
                    old_examples=[asdict(e) for e in old_examples],
                    training=[asdict(e) for e in training], parameters=session.model.parameter_count(),
                    selected_gradient_tensor=selected_key, heldout_targets_used=False)
        results["initial_old_evaluation"] = session.evaluate(old_examples)
        results["initial_training_evaluation"] = session.evaluate(training)
        results["checkpoints"].append(save_actual_checkpoint(session, directory / "initial.pt", "initial"))
        for iteration in range(limits.max_updates):
            if time.monotonic() >= learning_deadline:
                raise BudgetStop("learning stopped to reserve final checkpoint/reload time")
            if ledger.counts.get("updates", 0) >= limits.max_updates:
                break
            before = snapshot_parameters(session)
            objective = HybridTerminalObjective(session.model, training, ledger)
            gradients = coupled_gradient(session.model, session.bank, objective, 1,
                                         session.kernel, rng, path_estimator="event", within_samples=1)
            # Both exact components have already been computed and charged.
            applied = gradients[selected_key]
            commit = session.transaction(applied, operation=1, learning_rate=learning_rate)
            train_result = session.evaluate(training)
            old_result = session.evaluate(old_examples)
            if old_result["native_accuracy"] != 1.0 or old_result["soft_accuracy"] != 1.0:
                raise NumericalFailure("observed old capability failed its accepted contract")
            after = snapshot_parameters(session)
            row = {"iteration": iteration, "actual_parameter_updates": ledger.counts.get("updates", 0),
                   "gradient": gradients["record"], "selected_component": selected_key,
                   "within_parameter_gradients": {name: tensor.cpu().reshape(-1).tolist()
                       for (name, _), tensor in zip(session.model.named_parameters(), gradients["within_parameter_gradients"])},
                   "class_parameter_gradients": {name: tensor.cpu().reshape(-1).tolist()
                       for (name, _), tensor in zip(session.model.named_parameters(), gradients["class_parameter_gradients"])},
                   "transaction": commit, "parameters": after,
                   "parameter_displacements": parameter_displacements(before, after),
                   "train": train_result, "old": old_result,
                   "current_costs": ledger.snapshot(), "derived_storage": session.bank.storage()}
            results["curves"].append(row)
            ledger.emit("hybrid_learning_curve", **row)

            # Trigger uses TRAIN only. Its targets come only from the model's
            # own current hard choices, not the TRAIN labels or desired route.
            if train_result["native_accuracy"] == 1.0 and train_result["soft_accuracy"] == 1.0:
                all_token_bound = session.model.soft_native_path_bound((0, 1), 16, ledger)
                if all_token_bound <= SOFT_NATIVE_MIN_MASS and not maintenance_attempted:
                    maintenance_attempted = True
                    if ledger.counts.get("updates", 0) < limits.max_updates:
                        results["maintenance"] = own_native_confidence_maintenance(session)
                    else:
                        results["maintenance"] = {"attempted": False,
                                                  "reason": "no remaining update in fixed 100-update envelope"}
                admitted = session.admit_if_successful(training, operation=1, new_tokens=(1,))
                if admitted:
                    results["training_stop"] = "TRAIN acquisition and own-native confidence admission passed"
                    break
            if (iteration + 1) % checkpoint_every == 0:
                results["checkpoints"].append(save_actual_checkpoint(
                    session, directory / f"step-{iteration + 1:03d}.pt", f"step-{iteration + 1}"))
        if not admitted:
            results["training_stop"] = "fixed update budget reached or admission remains pending"
        results["admitted_new_operation_and_binding"] = admitted
        results["status"] = "finite_training_complete"
    except BudgetStop as error:
        results["status"], results["reason"] = "censored", str(error)
    except NumericalFailure as error:
        results["status"], results["reason"] = "numerical_unresolved", str(error)
    except Exception as error:
        results["status"], results["reason"] = "implementation_failure", f"{type(error).__name__}: {error}"
        (directory / "exception.txt").write_text(traceback.format_exc())

    # No parameter update, maintenance, selection or retry occurs after here.
    # Save the actual final/partial tensors BEFORE accessing held-out records.
    try:
        frozen_hash = session.model.parameter_hash()
        results["frozen_before_heldout_hash"] = frozen_hash
        for parameter in session.model.parameters():
            parameter.requires_grad_(False)
        results["checkpoints"].append(save_actual_checkpoint(session, directory / "final.pt", "frozen-final-or-partial"))
        results["final_parameters"] = snapshot_parameters(session)
        results["parameter_displacements_from_initial"] = parameter_displacements(initial_parameters, results["final_parameters"])
        results["admitted_operations"] = sorted(session.protected.admitted)
        results["protected_token_alphabet"] = list(session.old_tokens)
        results["actual_parameter_updates"] = ledger.counts.get("updates", 0)
        results["admitted_new_operation_and_binding"] = 1 in session.protected.admitted and 1 in session.old_tokens
        if time.monotonic() < arm_deadline:
            _, heldout = _diagnostic_data()
            results["heldout_after_frozen_weights"] = session.evaluate(heldout)
            if old_examples:
                results["final_old_evaluation"] = session.evaluate(old_examples)
            payload = torch.load(directory / "final.pt", map_location="cpu", weights_only=False)
            restored = HybridLearningSession.restore(payload, ledger)
            for parameter in restored.model.parameters():
                parameter.requires_grad_(False)
            results["reload_heldout"] = restored.evaluate(heldout)
            if old_examples:
                results["reload_old"] = restored.evaluate(old_examples)
            cold_maps = {}
            for operation in range(restored.model.bank_size):
                receipt = restored.executor.execute([restored.family.vertices(restored.model.action(operation))])
                ledger.add("reload_cold_primitive_reexecutions")
                cold_maps[operation] = receipt["endpoint_map"]
            recorded_maps = restored.bank.maps()
            results["reload_check"] = {
                "actual_nn_hash_equal": restored.model.parameter_hash() == frozen_hash,
                "cold_reexecuted_maps": cold_maps, "persisted_derived_maps": recorded_maps,
                "cold_maps_equal": cold_maps == recorded_maps,
                "inference_uses_actual_learned_encoder_call_readout": True,
                "final_original_hash_unchanged": session.model.parameter_hash() == frozen_hash}
        else:
            results["heldout_reload_status"] = "not executed: internal wall budget exhausted"
    except (BudgetStop, NumericalFailure) as error:
        results["heldout_reload_status"] = f"incomplete: {type(error).__name__}: {error}"
    except Exception as error:
        results["delivery_status"] = f"failure: {type(error).__name__}: {error}"
        (directory / "delivery-exception.txt").write_text(traceback.format_exc())
    finally:
        results["wall_seconds"] = time.perf_counter() - started
        results["costs"] = ledger.snapshot()
        results["forward_counts"] = dict(session.model.forward_counts)
        results["final_storage"] = session.bank.storage()
        results["checkpoint_total_bytes"] = sum(entry["bytes"] for entry in results["checkpoints"])
        write({"kind": "hybrid_arm_end", **results})
        raw.close()
        (directory / "result.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--max-updates", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a NEW run directory")
    if not 6 < args.seconds <= 90 or not 1 <= args.max_updates <= 100 or args.learning_rate != 0.05 or args.seed != 0:
        parser.error("frozen envelope: 6<seconds<=90, 1..100 updates, LR .05, seed0")
    args.out.mkdir(parents=True, exist_ok=False)
    source_directory = Path(__file__).resolve().parent
    protocol = {"version": "hybrid-loop-diagnostic-v1", "main": "TOPOLOGY_USER_LOCKED",
        "seed": 0, "learning_rate": .05, "seconds_internal": args.seconds,
        "outside_timeout_seconds_required": 120, "max_actual_updates_per_arm": args.max_updates,
        "limits_per_arm": {"max_feedback": 12000, "max_programs": 4000,
                            "max_continuation_steps": 160000},
        "arms": ["full_coupled", "within_only", "class_only"],
        "gradient_work": "both components always computed/charged; no ablation efficiency claim",
        "heldout_policy": "only after frozen and saved final weights; no selection/retry",
        "maintenance": "at most one own-current-class fit after TRAIN dual success, same transaction, no new P/info",
        "source_sha256": {name: source_sha(source_directory / name) for name in (
            "loop_boundary_learning.py", "hybrid_loop_block.py", Path(__file__).name)},
        "scope": "bounded D3 integrated mechanism diagnostic; no six-stage, generality, M0, or superiority claim",
        "wall_allocation": "equal share of remaining time, reserving 3 seconds global delivery; later budgets may differ"}
    (args.out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    deadline = time.monotonic() + args.seconds
    limits = Limits(max_feedback=12000, max_programs=4000,
                    max_continuation_steps=160000, max_updates=args.max_updates)
    results = []
    for index, arm in enumerate(protocol["arms"]):
        now = time.monotonic()
        remaining_arms = len(protocol["arms"]) - index
        available = deadline - now - 3.0
        if available <= 3.0 * remaining_arms:
            results.append({"arm": arm, "status": "not_started_internal_wall_limit"})
            continue
        arm_deadline = min(deadline - 3.0, now + available / remaining_arms)
        results.append(run_arm(arm, args.out / arm, limits, arm_deadline,
                               seed=args.seed, learning_rate=args.learning_rate))
    summary = {"protocol": protocol, "results": results,
               "evidence_boundary": "actual raw traces/checkpoints only; missing/censored checks stay missing"}
    (args.out / "result.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(args.out), "arms": [(result["arm"], result["status"]) for result in results]}, indent=2))


if __name__ == "__main__":
    main()
