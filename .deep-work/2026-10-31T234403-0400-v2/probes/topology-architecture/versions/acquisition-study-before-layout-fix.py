"""Five-arm acquisition contribution study from ONE actual neural checkpoint.

Fresh source parameters/RNG/old references in every arm; identical200 current
training batches, terminalCE+.01geometry, and NO current-task consolidation until
the end. Formation candidates use TRAIN only. Validation admission is end-only;
IID/OOD never select candidates or stop training. No import-time experiment,
teacher functions, SSH, subprocess, network or dependency installation.
MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 unpassed; M3 ultimate.
Run only inside the existing bounded external CUDA sandbox.
"""
import argparse
import copy
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from history_operation_formation import candidate as history_candidate
from integrate_native_repair import (MAX_FILE_BYTES, Recorder, evaluate,
    graph_summary, sha256, source_model, synchronize, tensor_bytes)
from observed_operation_formation import ALPHABETS, SPLITS, observed_candidate_graph, read_dataset


ARMS = ("continue_only", "native_recompile", "one_step", "history_full", "history_no_explicit_reuse")


def restore_rng(checkpoint, device):
    torch.set_rng_state(checkpoint["torch_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
    if device.type == "cuda" and checkpoint.get("cuda_rng"):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])


def geometry_delta(before, after):
    return {k: v-before.get(k, 0) for k, v in after["costs"].items()
            if isinstance(v, (int, float)) and k not in ("map_cache_bytes_current", "map_cache_bytes_peak")}


def save_actual_checkpoint(model, optimizer, original, recorder, completed, provenance, suffix):
    synchronize(model.encoder.device)
    began = time.perf_counter()
    path = recorder.out / f"topo-stage2-step{int(original['step'])+completed}-{suffix}.pt"
    payload = dict(schema="terminal-architecture-checkpoint-v1", args=copy.deepcopy(original["args"]),
        model_name="topo", stage=2, step=int(original["step"])+completed,
        model=model.state_dict(), optimizer=optimizer.state_dict(), model_extra=model.extra_state_dict(),
        masks_and_initialized_tasks={}, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
        python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        no_automatic_resume_or_retry=True, M0="NOT_PASSED",
        acquisition_contribution_study=dict(provenance=provenance, completed_additional_updates=completed,
            original_args_preserved=True, extra_budget_not_retroactive=True))
    temporary = path.with_name(path.name+".tmp")
    torch.save(payload, temporary)
    if temporary.stat().st_size > MAX_FILE_BYTES:
        raise RuntimeError("100MiB checkpoint cap")
    temporary.replace(path)
    recorder.emit("checkpoint", file=path.name, bytes=path.stat().st_size, sha256=sha256(path),
                  additional_updates=completed, seconds=time.perf_counter()-began)
    return path


def run_arm(name, original, datasets, witness, batches, device, root, provenance):
    root.check()
    recorder = Recorder(root.out/"arms"/name, max(.001, root.deadline-time.perf_counter()))
    arm_provenance = dict(provenance, arm=name)
    costs = dict(train_seconds=0., guard_seconds=0., evaluation_seconds=0.,
        evaluation_soft_episodes=0, evaluation_native_transition_uses=0,
        training_current_episodes=0, training_old_episodes=0,
        guard_calls=0, guard_full=0, guard_backtracked=0, guard_rejected=0,
        guard_relation_checks=0, guard_relation_pairs=0,
        formation_seconds=0., checkpoint_reload_seconds=0.)
    model = optimizer = restored = None
    completed, curves, formation = 0, [], None
    initial_geometry = None
    try:
        source_started = time.perf_counter()
        model = source_model(original, device)
        model.set_stage(2)
        model.protection = True
        model.set_geometry_deadline(time.monotonic()+max(.001, recorder.deadline-time.perf_counter()))
        if (not callable(getattr(model, "geometry_regularizer", None))
                or {int(r["task"]) for r in model.reference} != {0, 1}
                or any(int(r["task"]) >= 2 for r in model.reference)):
            raise ValueError("actual continuous source with only old0/1 references required")
        if not all(torch.equal(p.detach().cpu(), original["model"][key].cpu())
                   for key, p in model.state_dict().items()):
            raise RuntimeError("fresh source neural state differs from the common checkpoint")
        restore_rng(original, device)
        source_references = copy.deepcopy(model.reference)
        initial_geometry = copy.deepcopy(model.geometry_resource_report()["costs"])
        initial_parameters = sum(p.numel() for p in model.parameters())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        costs["source_restore_seconds"] = time.perf_counter()-source_started
        mechanisms = dict(formation=False, terminal_one_step_constraints=False,
            history_separation=False, continuation_closure=False,
            explicit_old_action_candidate_proposals=False,
            existing_neural_old_knowledge_present=True,
            old_context_protection=True, continuous_monodromy=True)
        recorder.emit("arm_begin", arm=name, provenance=arm_provenance,
            actual_source_parameters=initial_parameters, exact_source_theta_verified=True,
            source_RNG_restored=True, shared_temperature=model.temperature,
            old_reference_tasks=[r["task"] for r in model.reference],
            current_task_protected_during_training=False,
            same200_current_training_batches=True, no_router_MI=True,
            current_task_admission="only after all200 updates and fixed final validation gate")
        began = time.perf_counter()
        candidate_graph = audit = None
        if name == "native_recompile":
            mechanisms["formation"] = True
            candidate_graph = copy.deepcopy(model.graph(2, ALPHABETS[2]))
            audit = dict(training_only=True, fixed_task_id=2,
                rule="sham native recompile: current learned native graph unchanged as target",
                new_terminal_constraints_used=False, history_constraints_used=False,
                source="actual CURRENT native graph; no teacher operation or correct routing")
            candidate_graph["formation_audit"] = audit
        elif name == "one_step":
            mechanisms.update(formation=True, terminal_one_step_constraints=True)
            candidate_graph, audit = observed_candidate_graph(model, 2, datasets[2]["train"], recorder)
        elif name in ("history_full", "history_no_explicit_reuse"):
            reuse = name == "history_full"
            mechanisms.update(formation=True, history_separation=True, continuation_closure=True,
                              explicit_old_action_candidate_proposals=reuse)
            candidate_graph, audit = history_candidate(model, 2, datasets[2]["train"], witness, recorder,
                max_candidates=64, seconds=min(110., max(.001, root.deadline-time.perf_counter())),
                include_explicit_reuse=reuse)
        elif name != "continue_only":
            raise ValueError("unknown frozen study arm")
        if candidate_graph is not None:
            recorder.json("candidate-audit.json", audit)
            formation = model.materialize(2, ALPHABETS[2], ALPHABETS, candidate_graph=candidate_graph)
            recorder.emit("provisional_native_materialization", report=formation,
                current_task_not_consolidated=True, initial_validation_gate_not_used=True)
            if not formation.get("success"):
                raise RuntimeError("arm materialization failed: "+str(formation.get("reason")))
            actual = model.graph(2, ALPHABETS[2])
            if (not np.array_equal(candidate_graph["enc"], actual["enc"])
                    or not np.array_equal(candidate_graph["dec"], actual["dec"])
                    or any(not np.array_equal(candidate_graph["trans"][a], actual["trans"][a])
                           for a in ALPHABETS[2])):
                raise RuntimeError("installed neural graph differs from selected candidate")
        synchronize(device)
        costs["formation_seconds"] = time.perf_counter()-began
        old_checks = []
        for reference in source_references:
            okay, pairs = model.related(reference, model.graph(reference["task"], reference["tokens"]))
            old_checks.append(dict(task=reference["task"], related=bool(okay), relation_pairs=pairs))
            if not okay:
                raise RuntimeError("formation changed an old learned closed behavior")
        if {int(r["task"]) for r in model.reference} != {0, 1}:
            raise RuntimeError("current task was prematurely consolidated by an arm")
        recorder.emit("formation_completed", mechanisms=mechanisms, costs=costs,
            old_checks=old_checks, parameters=sum(p.numel() for p in model.parameters()),
            initial_current_validation_not_used_for_admission=True,
            explicit_reuse_ablation_scope="extra old-action proposals removed; old knowledge in source NN retained")
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        for step in range(201):
            root.check()
            recorder.check()
            if step in (0, 100, 200):
                metrics = evaluate(model, datasets, ALPHABETS, device, 64, recorder, costs)
                resources = model.geometry_resource_report()
                curve = dict(step=step, metrics=metrics, mechanisms=mechanisms,
                    parameters=sum(p.numel() for p in model.parameters()),
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                    optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
                    representation_state_bytes=model.representation_state_bytes(),
                    geometry_resources=resources, geometry_counter_deltas=geometry_delta(initial_geometry, resources),
                    actual_graphs=[graph_summary(model.graph(t, a)) for t, a in ALPHABETS.items()],
                    costs=costs.copy(), current_task_reference_present=False,
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
                curves.append(curve)
                recorder.emit("curve", **curve)
                if step in (0, 100):
                    save_actual_checkpoint(model, optimizer, original, recorder, step,
                                           arm_provenance, "contribution-provisional")
            if step == 200:
                break
            if any(int(r["task"]) >= 2 for r in model.reference):
                raise RuntimeError("current reference timing differs between study arms")
            batch = {key: torch.as_tensor(value[batches[step]], device=device, dtype=torch.long)
                     for key, value in datasets[2]["train"].items()}
            synchronize(device)
            training_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["tokens"], batch["starts"], batch["task_ids"])
            ce = F.cross_entropy(logits, batch["targets"])
            geometry = model.geometry_regularizer()
            loss = ce+.01*geometry
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training objective")
            loss.backward()
            omega_gradient = None if model.path_omega.grad is None else model.path_omega.grad.detach().clone()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gradient):
                raise FloatingPointError("nonfinite gradient")
            before = {key: p.detach().clone() for key, p in model.named_parameters()}
            optimizer.step()
            synchronize(device)
            costs["train_seconds"] += time.perf_counter()-training_started
            began = time.perf_counter()
            guard = model.project_update(before)
            synchronize(device)
            costs["guard_seconds"] += time.perf_counter()-began
            alpha = float(guard.get("alpha", 1.))
            costs["guard_calls"] += 1
            costs["guard_full" if alpha == 1 else "guard_backtracked" if alpha > 0 else "guard_rejected"] += 1
            costs["guard_relation_checks"] += int(guard.get("checks", 0))
            costs["guard_relation_pairs"] += int(guard.get("relation_pairs", 0))
            costs["training_current_episodes"] += 64
            completed = step+1
            recorder.emit("actual_neural_parameter_update", step=completed,
                terminal_CE=float(ce.item()), normalized_geometry_action=float(geometry.item()),
                total_loss=float(loss.item()), preclip_gradient_norm=float(gradient.item()),
                preclip_omega_gradient=omega_gradient, shared_omega=model.path_omega,
                old_only_guard=guard, current_task_reference_present=False)
        old_checks = []
        for reference in source_references:
            okay, pairs = model.related(reference, model.graph(reference["task"], reference["tokens"]))
            old_checks.append(dict(task=reference["task"], related=bool(okay), relation_pairs=pairs))
            if not okay:
                raise RuntimeError("final neural state changed protected old behavior")
        final_valid = curves[-1]["metrics"][2]["valid"]["native_acc"]
        admitted = final_valid >= .95
        admission = model.consolidate(2, ALPHABETS[2]) if admitted else None
        recorder.emit("END_ONLY_validation_admission", validation_accuracy=final_valid, threshold=.95,
            admitted=admitted, admission=admission, old_checks=old_checks,
            validation_never_stopped_or_selected_training=True, no_training_after_consolidation=True)
        final_path = save_actual_checkpoint(model, optimizer, original, recorder, completed,
                                           arm_provenance, "contribution-final")
        began = time.perf_counter()
        restored_payload = torch.load(final_path, map_location="cpu", weights_only=False)
        restored = source_model(restored_payload, device)
        restored.set_geometry_deadline(time.monotonic()+max(.001, root.deadline-time.perf_counter()))
        if not all(torch.equal(p, restored.state_dict()[key]) for key, p in model.state_dict().items()):
            raise RuntimeError("final checkpoint neural parameters did not reload identically")
        validation = datasets[2]["valid"]
        probe = {key: torch.as_tensor(value[:64], device=device, dtype=torch.long)
                 for key, value in validation.items()}
        with torch.no_grad():
            actual = model.hard_eval(probe["tokens"], probe["starts"], probe["task_ids"])
            reloaded = restored.hard_eval(probe["tokens"], probe["starts"], probe["task_ids"])
            if not torch.equal(actual, reloaded):
                raise RuntimeError("actual continuous checkpoint execution changed after reload")
        synchronize(device)
        costs["checkpoint_reload_seconds"] += time.perf_counter()-began
        recorder.emit("actual_checkpoint_reload_verified", checkpoint=final_path.name,
            all_neural_parameters_bitwise_equal=True, actual_continuous_rows=64,
            actual_continuous_outputs_equal=True, restored_geometry=restored.geometry_resource_report())
        result = dict(arm=name, status="COMPLETED200_UPDATES", provenance=arm_provenance,
            mechanisms=mechanisms, initial_parameters=initial_parameters,
            final_parameters=sum(p.numel() for p in model.parameters()),
            final_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
            final_optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
            final_representation_state_bytes=model.representation_state_bytes(),
            final_checkpoint_bytes=final_path.stat().st_size,
            maximum_observed_current_map_cache_bytes=max(
                c["geometry_resources"]["costs"]["map_cache_bytes_current"] for c in curves),
            actual_additional_updates=completed, formation=formation, curves=curves, costs=costs,
            final_valid_admitted=admitted, current_reference_first_added_only_at_end=admitted,
            geometry_resources=model.geometry_resource_report(),
            final_checkpoint=final_path.name, actual_checkpoint_reload_verified=True,
            wall_seconds=time.perf_counter()-recorder.started, M0="NOT_PASSED",
            scope="formation components from one common development checkpoint; not topology advantage versus Transformer")
        recorder.json("summary.json", result)
        return result
    except Exception as error:
        if model is not None and optimizer is not None:
            try:
                save_actual_checkpoint(model, optimizer, original, recorder, completed,
                                       arm_provenance, "contribution-interrupted")
            except Exception as save_error:
                recorder.emit("interrupted_checkpoint_failed", reason=str(save_error))
        result = dict(arm=name, status="FAILED_WITH_PARTIAL_RECORDS", exception=type(error).__name__,
                      reason=str(error), actual_additional_updates=completed, costs=costs,
                      wall_seconds=time.perf_counter()-recorder.started)
        recorder.emit("arm_failure", **result)
        recorder.json("failure.json", result)
        return result
    finally:
        recorder.stream.close()
        del model, optimizer, restored
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seed", type=int, default=9010)
    parser.add_argument("--max-seconds", type=float, default=850.)
    args = parser.parse_args()
    arms = tuple(name.strip() for name in args.arms.split(",") if name.strip())
    if (not arms or len(set(arms)) != len(arms) or set(arms)-set(ARMS) or args.seed != 9010
            or args.out.exists() or not 0 < args.max_seconds <= 1500
            or not args.checkpoint.is_file() or args.checkpoint.stat().st_size > MAX_FILE_BYTES
            or not args.records.is_file()):
        parser.error("known arms,seed9010,newoutput,trusted source and bounded time required")
    root = Recorder(args.out, args.max_seconds)
    device = torch.device(args.device)
    torch.set_num_threads(1)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("requested authorized GPU unavailable; no silent substitute")
        torch.cuda.set_per_process_memory_fraction(.3, device)
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if (original.get("schema") != "terminal-architecture-checkpoint-v1"
            or original.get("model_name") != "topo" or int(original.get("stage", -1)) != 2
            or "path_omega" not in original["model"]):
        raise ValueError("actual V8 continuous stage2 topology checkpoint required")
    witness = None
    with args.records.open(encoding="utf8") as stream:
        for line in stream:
            event = json.loads(line)
            if (event.get("kind") == "observed_history_state_witness"
                    and event.get("model") == "topo" and int(event.get("stage", -1)) == 2):
                witness = event
    if witness is None:
        raise ValueError("actual stage2 observed-history witness missing")
    files = ("acquisition_contribution_study.py", "history_operation_formation.py",
        "observed_operation_formation.py", "integrate_native_repair.py", "topology_model.py",
        "semantic_promotion_model.py", "continuous_native_model.py", "continuous_monodromy.py",
        "train_monodromy_extension.py")
    provenance = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
        records=str(args.records), records_sha256=sha256(args.records),
        source_sha256={name: sha256(Path(__file__).with_name(name)) for name in files})
    datasets, input_sources = {}, []
    for task in sorted(ALPHABETS):
        datasets[task] = {}
        for split in SPLITS:
            data, source = read_dataset(args.datasets, task, split)
            datasets[task][split] = data
            input_sources.append(source)
    generator = np.random.default_rng(args.seed)
    batches = generator.integers(0, len(datasets[2]["train"]["starts"]), size=(200, 64), dtype=np.int64)
    np.savez_compressed(root.out/"common-training-indices.npz", indices=batches, seed=np.asarray(args.seed))
    provenance["batch_indices_sha256"] = sha256(root.out/"common-training-indices.npz")
    root.emit("configuration", args=vars(args), arms=arms, provenance=provenance,
        datasets=input_sources, fresh_source_restore_per_arm=True, same200_indices_per_arm=True,
        batch=64, lr=.01, objective="terminalCE+.01geometry", old_replay_samples=0,
        current_reference_policy="none during ALLarms'200updates; end-only validation admission",
        candidate_ranking="TRAIN only; no future-test selection", old_contracts=[0, 1],
        parameters_may_differ_and_are_recorded=True, formation_search_costs_are_additional=True,
        MAIN="TOPOLOGY_USER_LOCKED", primary_architecture_baseline="Transformer", M0="NOT_PASSED")
    results = []
    try:
        for arm in arms:
            root.check()
            result = run_arm(arm, original, datasets, witness, batches, device, root, provenance)
            folder = root.out/"arms"/arm
            result["actual_saved_bytes_after_arm_closed"] = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            results.append(result)
            root.emit("arm_completed", arm=arm, status=result["status"],
                actual_updates=result["actual_additional_updates"],
                actual_saved_bytes=result["actual_saved_bytes_after_arm_closed"])
        root.json("summary.json", dict(status="BOUNDED_COMPONENT_STUDY_FINISHED", results=results,
            provenance=provenance, input_datasets=input_sources,
            requested_arms=list(arms), all_arms_completed200=all(r["status"] == "COMPLETED200_UPDATES" for r in results),
            wall_seconds=time.perf_counter()-root.started, M0="NOT_PASSED",
            attribution_scope="operation/state formation interventions with common source NN and common end-only admission timing",
            limits="one source checkpoint/seed; actual formation budgets and parameter counts differ and are reported"))
        print(json.dumps(dict(status="BOUNDED_COMPONENT_STUDY_FINISHED",
            arms={r["arm"]: r["status"] for r in results}, output=str(args.out)), sort_keys=True))
    except BaseException as error:
        root.emit("study_interrupted", exception=type(error).__name__, reason=str(error),
                  completed_arms=[r["arm"] for r in results])
        root.json("failure.json", dict(status="INTERRUPTED_WITH_PARTIAL_ARMS", results=results,
            exception=type(error).__name__, reason=str(error), M0="NOT_PASSED"))
        raise
    finally:
        root.stream.close()


if __name__ == "__main__":
    main()
