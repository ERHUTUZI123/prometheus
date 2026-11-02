"""ROOT-RUN bounded two-event mean-policy repair of an ACTUAL saved interface.

Same Prometheus V2; MAIN=TOPOLOGY_USER_LOCKED. No execution at import.
This resumes failed interface learning, rather than restarting its old core.
It retains the cover, grammar, labels, E/R, available call routes, and P.

The new rule is ordinary finite class-neighbor policy improvement: a paid scan
of opposite mean chambers, followed by a caller gradient of the SAME mean CE.
If every single flip fails, max-edit-depth2 permits exactly276 unordered pair
auditions. Only strict improvements can be committed; no neutral stepping
state is installed. A complete local obstruction ends class search immediately.
It is not an unbiased estimator of the earlier fully smoothed objective, nor
a claim of topological advantage. An equal-information ordinary optimizer can
make precisely these updates. No desired reset program or pair map is supplied.

ROOT must supply existing external isolation/timeout. Example entry point:
  python mean_policy_reset_pair_repair.py --checkpoint MEAN_V1/topological-frozen.pt \
      --controls-result V1/result.json --out NEW_DIRECTORY --seconds 90 \
      --max-edit-depth 2 --max-pairs 3
"""
from __future__ import annotations

import argparse
import copy
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import time
import traceback
from typing import Any, Sequence

import torch

from factorized_reset_block import ResetInterfaceSession, proposed_limits
from hybrid_loop_block import SOFT_NATIVE_MIN_MASS, native_output
from loop_boundary_learning import BudgetStop, CostLedger, NumericalFailure, ProgramExample


IMPROVEMENT_TOLERANCE = 1e-12
LEARNING_RATE = .05


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_objective(session: ResetInterfaceSession, examples: Sequence[ProgramExample],
                   actions: Sequence[float] | None = None) -> tuple[torch.Tensor, list[int]]:
    """Terminal TRAIN CE using actual mean classes and the CURRENT soft caller.

    Explicit action interventions do not modify NN weights. They are paid
    counterfactual class auditions; terminal_loss counts every accessed label.
    No route is forced. The map consists of paid observed class representatives.
    """
    model = session.model
    model.assert_frame()
    mean = model.actions() if actions is None else list(actions)
    pair_map = model.bank.program_map(mean, mean=True)
    session.ledger.add("mean_policy_objective_batches")
    loss = model.terminal_loss(examples, pair_map)
    if not bool(torch.isfinite(loss)):
        raise NumericalFailure("nonfinite mean-policy TRAIN objective")
    return loss, pair_map


def opposite_mean_representative(session: ResetInterfaceSession, coordinate: int,
                                 current: float) -> dict[str, Any]:
    """Interior opposite chamber derived from actual event and support box.

    For the existing grammar the event is8 and eroded mean box is[6,10],
    yielding7/9. These values are derived, not a supplied operation program.
    """
    bank = session.model.bank
    family = bank.families[coordinate % 8]
    events = bank.events(coordinate % 8)
    if len(events) != 1:
        raise NumericalFailure("mean repair requires the declared single-event family")
    event = events[0]
    sigma = session.model.kernel.sigma
    left, right = family.action_box[0] + sigma, family.action_box[1] - sigma
    if not left < event < right or current == event or not left <= current <= right:
        raise NumericalFailure("mean event or source mean outside legal opposite-chamber domain")
    target = .5 * (event + (right if current < event else left))
    session.ledger.add("mean_policy_opposite_representatives")
    return {"event": event, "current_chamber": int(current > event),
            "opposite_chamber": int(current < event), "target_action": target,
            "legal_mean_box": [left, right]}


def scan_and_fit(session: ResetInterfaceSession, examples: Sequence[ProgramExample],
                 max_edit_depth: int = 2) -> dict[str, Any]:
    """Scan24 singles, then ONLY on obstruction276 unordered double flips.

    Every candidate starts from the same actual mean/current caller. Singles
    have priority when any strictly improves; a pair is not two greedy steps.
    Ties follow lexicographic coordinates. No neutral state is committed.
    """
    if max_edit_depth not in (1, 2):
        raise ValueError("mean repair edit depth must be1 or2")
    model, ledger = session.model, session.ledger
    actions = model.actions()
    if len(actions) != 24:
        raise ValueError("this diagnostic freezes the existing24-gate grammar")
    with torch.no_grad():
        baseline_loss, baseline_map = mean_objective(session, examples, actions)
        baseline = float(baseline_loss)
        candidates: list[dict[str, Any]] = []
        best: dict[str, Any] | None = None
        proposals = [opposite_mean_representative(session, coordinate, current)
                     for coordinate, current in enumerate(actions)]
        def audition(coordinates: tuple[int, ...]) -> dict[str, Any]:
            proposed_actions = list(actions)
            edits = []
            for coordinate in coordinates:
                proposal = proposals[coordinate]
                proposed_actions[coordinate] = proposal["target_action"]
                edits.append({"coordinate": coordinate, "source_action": actions[coordinate], **proposal})
            loss, pair_map = mean_objective(session, examples, proposed_actions)
            value = float(loss)
            ledger.add("mean_policy_candidate_programs")
            ledger.add("mean_policy_candidate_coordinate_edits", len(coordinates))
            ledger.add("mean_policy_single_candidates" if len(coordinates) == 1 else "mean_policy_pair_candidates")
            row = {"coordinates": list(coordinates), "edit_depth": len(coordinates), "edits": edits,
                   "train_mean_ce": value, "ce_change": value - baseline,
                   "pair_map": pair_map, "logical_channel": model.logical_channel(pair_map)}
            candidates.append(row)
            return row
        for coordinate in range(len(actions)):
            row = audition((coordinate,))
            value = row["train_mean_ce"]
            if value < baseline - IMPROVEMENT_TOLERANCE and (
                    best is None or value < best["train_mean_ce"] - IMPROVEMENT_TOLERANCE):
                best = row
        single_obstruction = best is None
        if single_obstruction:
            ledger.add("mean_policy_single_event_obstructions")
        if single_obstruction and max_edit_depth == 2:
            ledger.add("mean_policy_pair_scans")
            for coordinates in combinations(range(len(actions)), 2):
                row = audition(coordinates)
                value = row["train_mean_ce"]
                if value < baseline - IMPROVEMENT_TOLERANCE and (
                        best is None or value < best["train_mean_ce"] - IMPROVEMENT_TOLERANCE):
                    best = row
    record: dict[str, Any] = {"baseline_mean_ce": baseline, "baseline_actions": actions,
        "baseline_pair_map": baseline_map, "baseline_logical_channel": model.logical_channel(baseline_map),
        "candidates": candidates, "candidate_count": len(candidates), "selected": best,
        "max_edit_depth": max_edit_depth, "single_event_obstruction": single_obstruction,
        "single_candidate_count": 24, "pair_candidate_count": len(candidates) - 24,
        "strict_improvement_tolerance": IMPROVEMENT_TOLERANCE,
        "correct_call_or_program_supplied": False, "transaction": None}
    if best is None:
        ledger.add("mean_policy_local_obstructions")
        record["status"] = "local_obstruction_no_strict_improvement_within_declared_depth"
        record["stop_class_search"] = True
        ledger.emit("mean_policy_class_scan", **record)
        return record

    named = model.trainable_named()
    gradients = [torch.zeros_like(parameter) for _, parameter in named]
    with torch.no_grad():
        hidden = model.core.shared(model.reset_feature)
        norm = torch.dot(hidden, hidden)
        ledger.add("mean_policy_frozen_feature_forwards")
        if not bool(torch.isfinite(hidden).all()) or float(norm) <= 1e-12:
            raise NumericalFailure("frozen feature cannot support exact selected-row mean fits")
        fitted_rows = []
        for edit in best["edits"]:
            coordinate = edit["coordinate"]
            delta = edit["target_action"] - actions[coordinate]
            row_change = delta * hidden / norm
            for (name, _), gradient in zip(named, gradients):
                if name != "reset_loop_head.weight":
                    continue
                gradient[coordinate].copy_(-row_change / LEARNING_RATE)
            fitted_rows.append({"coordinate": coordinate, "desired_action_change": delta,
                                "row_parameter_change": row_change.cpu().tolist()})
            ledger.add("mean_policy_rank1_row_fits")
            ledger.add("mean_policy_rank1_fitted_scalars", int(hidden.numel()))
    record["fit"] = {"coordinates": best["coordinates"], "frozen_feature": hidden.cpu().tolist(),
                     "feature_squared_norm": float(norm), "rows": fitted_rows,
                     "only_parameter_block": "reset_loop_head.weight",
                     "other_loop_rows_masked": True, "call_head_masked": True}
    # The existing transaction checks parameter box, mean geometry, old native
    # relation, old soft path mass, and any already-admitted full-lift contract.
    record["transaction"] = model.transaction(gradients, LEARNING_RATE)
    record["actual_actions_after_transaction"] = model.actions()
    record["status"] = "strict_selected_neighbor_fit_transacted_attainment_requires_evaluation"
    record["stop_class_search"] = False
    ledger.emit("mean_policy_class_scan", **record)
    return record


def mean_caller_step(session: ResetInterfaceSession,
                     examples: Sequence[ProgramExample]) -> dict[str, Any]:
    """Existing caller weights receive a gradient of SAME mean-class TRAIN CE."""
    model, ledger = session.model, session.ledger
    loss, pair_map = mean_objective(session, examples)
    gradient, = torch.autograd.grad(loss, (model.reset_call_head.weight,))
    gradient = gradient.detach()
    if not bool(torch.isfinite(gradient).all()):
        raise NumericalFailure("nonfinite mean-policy caller gradient")
    named = model.trainable_named()
    gradients = [gradient if name == "reset_call_head.weight" else torch.zeros_like(parameter)
                 for name, parameter in named]
    ledger.add("mean_policy_call_vjps")
    record = {"train_mean_ce": float(loss.detach()), "pair_map": pair_map,
              "logical_channel": model.logical_channel(pair_map),
              "gradient_norm": float(gradient.norm()),
              "only_parameter_block": "reset_call_head.weight", "loop_head_masked": True,
              "correct_call_supplied": False,
              "transaction": model.transaction(gradients, LEARNING_RATE)}
    ledger.emit("mean_policy_call_update", **record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="actual mean-v1 topological-frozen.pt, NOT an earlier core/restart")
    parser.add_argument("--controls-result", type=Path, required=True,
                        help="completed V1 result; frozen classical outputs reused without rerunning")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=90.)
    parser.add_argument("--max-edit-depth", type=int, choices=(1, 2), default=2)
    parser.add_argument("--max-pairs", type=int, default=3)
    parser.add_argument("--max-transactions", type=int, default=100)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 90 or not 1 <= args.max_pairs <= 3 or not 1 <= args.max_transactions <= 100:
        raise ValueError("bounded pair repair requires <=3 pairs/100 transactions/90s")
    if args.checkpoint.resolve() == (args.out / "topological-frozen.pt").resolve():
        raise ValueError("source checkpoint must remain read-only")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stream = (args.out / "topological.jsonl").open("x")
    def emit(row: dict[str, Any]) -> None:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
    ledger = CostLedger(proposed_limits(), time.monotonic() + args.seconds, writer=emit)
    source_paths = [Path(__file__), *(Path(__file__).with_name(name) for name in (
        "factorized_reset_block.py", "hybrid_loop_block.py", "loop_boundary_learning.py"))]
    result: dict[str, Any] = {"scope": "TODO3 resumed frame-frozen bounded two-event mean-policy repair; not Primary acceptance or advantage",
        "MAIN": "TOPOLOGY_USER_LOCKED", "status": "started", "seed": 0,
        "resume_checkpoint": {"path": str(args.checkpoint), "sha256": digest(args.checkpoint),
                              "bytes": args.checkpoint.stat().st_size},
        "source_sha256": {p.name: digest(p) for p in source_paths},
        "reused_controls": {"path": str(args.controls_result), "sha256": digest(args.controls_result),
                            "new_control_training": False},
        "rule": {"objective": "six TRAIN terminal CE at actual mean loop classes/current soft caller",
                 "class_step": "all24 single flips; only if none improves, all276 unordered double flips; strict best; existing-head selected-row rank1 fits",
                 "call_step": "existing call-head gradient of same mean objective",
                 "max_edit_depth": args.max_edit_depth,
                 "no_strict_neighbor": "record local obstruction and end class-search/learning loop immediately; no neutral move or deeper lookahead",
                 "max_pairs": args.max_pairs, "max_actual_transactions": args.max_transactions,
                 "learning_rate": LEARNING_RATE, "own_call_margin_max": 1,
                 "transaction": "existing reset block transaction, all original certificates",
                 "new_parameter_count": 0, "same_event_ordinary_correspondence": True},
        "heldout_used_for_selection": False, "exact_training_resume_tested": False,
        "arms": {"topological": {"curve": [], "pair_curve": [], "admitted": False,
                                  "maintenance_used": False}}}
    session: ResetInterfaceSession | None = None
    arm = result["arms"]["topological"]

    def update_count() -> int:
        return ledger.counts.get("updates", 0)

    def capacity() -> bool:
        return update_count() < args.max_transactions

    def save(name: str) -> dict[str, Any]:
        assert session is not None
        path = args.out / name
        checkpoint = {"interface": session.checkpoint(), "repair_rule": result["rule"],
                      "resume_checkpoint": result["resume_checkpoint"],
                      "actual_repair_transactions": update_count(),
                      "torch_cpu_rng": torch.get_rng_state(), "exact_training_resume_tested": False,
                      "sources": result["source_sha256"]}
        with path.open("xb") as file:
            torch.save(checkpoint, file)
        ledger.add("mean_policy_checkpoint_writes")
        return {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}

    def evaluation_row(stage: str, pair_index: int, update: dict[str, Any]) -> dict[str, Any]:
        assert session is not None
        evaluation = session.evaluate(train)
        row = {"stage": stage, "pair": pair_index, "transaction_index": update_count(), "update": update,
               "soft_accuracy": evaluation["soft_accuracy"], "native_accuracy": evaluation["native_accuracy"],
               "logical_channel": evaluation["logical_channel"],
               "losses": [-math.log(max(1e-12, r["soft_probability"][r["example"]["target"]]))
                          for r in evaluation["rows"]], "costs": ledger.snapshot()}
        arm["curve"].append(row)
        ledger.emit("mean_policy_training_curve", **row)
        return evaluation

    def train_ready(evaluation: dict[str, Any], pair_index: int) -> bool:
        assert session is not None
        if evaluation["soft_accuracy"] != 1. or evaluation["native_accuracy"] != 1.:
            return False
        if not arm["maintenance_used"] and capacity():
            arm["maintenance_used"] = True
            maintenance = session.maintain_own_call_margin()
            arm["maintenance"] = maintenance
            evaluation = evaluation_row("own_call_margin", pair_index, maintenance)
        mass = session.model.soft_path_mass()
        ready = (evaluation["soft_accuracy"] == evaluation["native_accuracy"] == 1.
                 and len(set(evaluation["logical_channel"])) < 3
                 and int(session.model.reset_calls().argmax()) == 3
                 and mass > SOFT_NATIVE_MIN_MASS)
        ledger.emit("mean_policy_train_readiness", ready=ready, path_mass=mass,
                    maintenance_used=arm["maintenance_used"], actual_transactions=update_count())
        return ready

    try:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        ledger.add("mean_policy_resume_checkpoint_loads")
        if "interface" not in payload or payload["interface"].get("version") != "factorized-reset-interface-v1":
            raise ValueError("resume input is not an actual saved factorized reset interface")
        source_interface = payload["interface"]
        if source_interface.get("open_typed_histories"):
            raise ValueError("pre-admission repair does not support persistent live arenas")
        if source_interface.get("admitted_pair") is not None:
            raise ValueError("this bounded repair expects the failed, unadmitted mean-v1 interface")
        result["historical_resume_costs"] = copy.deepcopy(source_interface["costs"])
        session = ResetInterfaceSession.restore(source_interface, ledger)
        train, heldout, manifest = session.diagnostic_specification()
        result["prospective_manifest_from_source"] = manifest
        result["actual_repair_training"] = [r.__dict__ for r in train]
        arm["parameter_storage"] = session.parameter_storage()
        arm["initial"] = session.evaluate(train)
        ready = False
        for pair_index in range(1, args.max_pairs + 1):
            if not capacity():
                break
            pair_record: dict[str, Any] = {"pair": pair_index, "transactions_before": update_count()}
            arm["pair_curve"].append(pair_record)
            scan = scan_and_fit(session, train, args.max_edit_depth)
            pair_record["class_scan"] = scan
            if scan["stop_class_search"]:
                pair_record["transactions_after"] = update_count()
                arm["class_search_stop"] = {"pair": pair_index, "reason": scan["status"],
                                             "max_edit_depth": args.max_edit_depth,
                                             "no_additional_class_scan_or_caller_update": True}
                break
            if scan["transaction"] is not None:
                evaluation = evaluation_row("mean_class_fit", pair_index, scan)
                if train_ready(evaluation, pair_index):
                    ready = True
                    pair_record["transactions_after"] = update_count()
                    break
            if not capacity():
                pair_record["transactions_after"] = update_count()
                break
            caller = mean_caller_step(session, train)
            pair_record["caller_update"] = caller
            evaluation = evaluation_row("mean_caller_gradient", pair_index, caller)
            ready = train_ready(evaluation, pair_index)
            pair_record["transactions_after"] = update_count()
            if ready:
                break
            if pair_index % 5 == 0:
                pair_record["checkpoint"] = save(f"topological-pair-{pair_index:03d}.pt")
        arm["train_ready_before_cold"] = ready
        arm["actual_transactions"] = update_count()
        arm["completed_pairs"] = len(arm["pair_curve"])
        arm["train"] = session.evaluate(train)
        # No parameter changes follow this disk freeze. Admission can update
        # semantic license metadata, but its cold execution occurs AFTER freeze.
        arm["pre_audit_weight_freeze"] = save("topological-pre-audit-frozen.pt")
        arm["weights_frozen_before_any_full_mean_cold_audit"] = True
        arm["admitted"] = session.admit_if_successful(train)
        arm["cold_audit"] = (session.admission_cold_receipts if arm["admitted"]
                             else session.audit_cold_local_maps())
        arm["frozen_checkpoint"] = save("topological-frozen.pt")
        arm["heldout"] = session.evaluate(heldout)
        old = tuple(ProgramExample(f"own-old-{x}-{i}", x, word,
                    native_output(session.model.reference, ProgramExample("reference", x, word, 0), ledger))
                    for x in range(3) for i, word in enumerate(((), (0,), (1,), (0, 1), (1, 0))))
        arm["old"] = session.evaluate(old)
        reloaded_payload = torch.load(args.out / "topological-frozen.pt", map_location="cpu", weights_only=True)
        ledger.add("mean_policy_final_checkpoint_disk_loads")
        restored = ResetInterfaceSession.restore(reloaded_payload["interface"], ledger)
        reloaded = restored.evaluate(heldout)
        arm["actual_reload"] = {"native_equal": [r["native_output"] for r in reloaded["rows"]]
                                == [r["native_output"] for r in arm["heldout"]["rows"]],
            "soft_equal": all(r["soft_probability"] == s["soft_probability"]
                              for r, s in zip(reloaded["rows"], arm["heldout"]["rows"])),
            "cold_maps": restored.audit_cold_local_maps(), "storage": restored.parameter_storage()}
        # Frozen ordinary outputs are copied only now; no control is retrained,
        # and no heldout control result influences repair selection.
        previous = json.loads(args.controls_result.read_text())
        for name in ("ordinary_direct_channel", "ordinary_learned_factorized_lift"):
            result["arms"][name] = previous["arms"][name]
        result["historical_classical_combined_costs"] = previous["classical_combined_costs"]
        arm["final_parameter_storage"] = session.parameter_storage()
        arm["costs"] = ledger.snapshot()
        result["status"] = "completed_measured_diagnostic"
    except (BudgetStop, NumericalFailure) as error:
        result["status"] = "bounded_diagnostic_censored"
        result["error"] = f"{type(error).__name__}: {error}"
        (args.out / "exception.txt").write_text(traceback.format_exc())
    except Exception as error:
        result["status"] = "implementation_failed"
        result["error"] = f"{type(error).__name__}: {error}"
        (args.out / "exception.txt").write_text(traceback.format_exc())
    finally:
        if session is not None and result["status"] != "completed_measured_diagnostic":
            # Emergency serialization performs no inference/label access and
            # remains possible after a ledger deadline. It is explicitly a raw
            # partial state, not an asserted complete/restored training session.
            path = args.out / "partial-actual-state.pt"
            raw = {"state_dict": {k: v.detach().cpu().clone() for k, v in session.model.state_dict().items()},
                   "pair_maps_cache": copy.deepcopy(session.model.bank.maps_cache),
                   "pair_events_cache": copy.deepcopy(session.model.bank.events_cache),
                   "admitted_pair": session.model.admitted_pair,
                   "admitted_logical": session.model.admitted_logical,
                   "resume_checkpoint": result["resume_checkpoint"],
                   "scope": "actual partial tensors/caches; complete training resume not verified"}
            with path.open("xb") as file:
                torch.save(raw, file)
            result["partial_actual_state"] = {"path": str(path), "sha256": digest(path),
                                              "bytes": path.stat().st_size}
        result["repair_costs"] = ledger.snapshot()
        result["wall_seconds"] = time.perf_counter() - started
        stream.close()
        (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": result["status"], "error": result.get("error"),
                          "admitted": arm["admitted"], "actual_transactions": update_count()}))
    if result["status"] == "implementation_failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
