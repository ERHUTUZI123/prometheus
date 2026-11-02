"""Form native operations from observed terminal episodes in the SAME learner.

For newly observed public symbols, one-step terminal records constrain a latent
permutation through the model's CURRENT learned encoder/decoder. Among feasible
permutations, maximize the CURRENT learned soft joint operator score. Use the
normal native materialization/shared-call/old-contract transaction with its
explicit candidate_graph hook. Nothing supplies teacher operations or routes.

Requires the coordinator's inspected SemanticPromotionModel.materialize(...,
candidate_graph=None) hook. Existing default behavior must remain unchanged.
Run only in the existing external bounded GPU sandbox. No import-time work,
teacher functions, downloaded code, SSH, subprocess or network calls.
MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 NOT PASSED; M3 ultimate.
"""
import argparse
import copy
import inspect
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from integrate_native_repair import (MAX_FILE_BYTES, Recorder, evaluate,
    graph_summary, sha256, source_model, synchronize, tensor_bytes)
from topology_model import linear_sum_assignment


ALPHABETS = {0: (1, 2, 3), 1: (1, 2, 3, 4, 5), 2: (1, 2, 3, 4, 5, 6)}
ARRAY_KEYS = ("tokens", "starts", "targets", "task_ids")
SPLITS = ("train", "valid", "iid", "ood", "length16")


def read_dataset(directory, task, split):
    path = directory / f"task{task}-{split}.npz"
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key], dtype=np.int64) for key in ARRAY_KEYS}
    n = len(data["starts"])
    if (data["tokens"].shape != (n, 18) or n > 16384
            or data["targets"].shape != (n,) or data["task_ids"].shape != (n,)
            or np.any(data["task_ids"] != task)
            or set(int(a) for a in np.unique(data["tokens"]) if a) - set(ALPHABETS[task])):
        raise ValueError("original data task/shape/symbol budget mismatch")
    return data, dict(task=task, split=split, path=str(path), sha256=sha256(path),
                     rows=n, array_bytes=sum(value.nbytes for value in data.values()))


@torch.no_grad()
def observed_candidate_graph(model, stage, training, recorder):
    """Executable acquisition rule; only CURRENT TRAIN terminal labels enter it."""
    began = time.perf_counter()
    alphabet = ALPHABETS[stage]
    new_symbols = sorted(set(alphabet)-set(ALPHABETS.get(stage-1, ())))
    candidate = copy.deepcopy(model.graph(stage, alphabet))
    encoder, decoder = candidate["enc"], candidate["dec"]
    costs = dict(training_record_scans=0, observed_one_step_records=0,
                 constraint_destination_comparisons=0, assignment_calls=0,
                 assignment_score_entries=0, joint_matrix_multiplications=0,
                 source_bank_assignments=model.n_ops, new_external_label_queries=0)
    audits = []
    if any(int(reference["task"]) >= stage for reference in model.reference):
        raise ValueError("current stage must not already have an admitted protection contract")
    ops = model.soft_ops()
    for symbol in new_symbols:
        recorder.check()
        costs["training_record_scans"] += len(training["starts"])
        rows = np.flatnonzero((np.count_nonzero(training["tokens"], axis=1) == 1)
                              & (training["tokens"][:, 0] == symbol))
        if not len(rows):
            raise ValueError("new symbol has no observed one-step terminal records; do not invent them")
        mask = np.ones((model.d, model.d), dtype=bool)
        source_rows, observed_q = [], set()
        for row in rows:
            x, y = int(training["starts"][row]), int(training["targets"][row])
            if not 0 <= x < model.n or not 0 <= y < model.n:
                raise ValueError("observed input/output outside learned code")
            q = int(encoder[x])
            allowed = decoder == y
            mask[q] &= allowed
            observed_q.add(q)
            costs["observed_one_step_records"] += 1
            costs["constraint_destination_comparisons"] += model.d
            source_rows.append(dict(dataset_row=int(row), observed_start=x,
                observed_terminal_output=y, MODEL_encoded_state=q,
                MODEL_destination_states_with_observed_output=np.flatnonzero(allowed).tolist()))
        if np.any(~mask.any(axis=1)):
            raise ValueError("current learned code contradicts terminal constraints; no fabricated state split")
        tokens = torch.as_tensor([[symbol]], device=model.encoder.device, dtype=torch.long)
        task = torch.as_tensor([stage], device=model.encoder.device, dtype=torch.long)
        probabilities = torch.softmax(model.calls(tokens, task)[0, 0]/model.temperature, -1)
        joint = torch.eye(model.d, device=model.encoder.device)
        for slot in range(model.slots):
            joint = joint @ torch.einsum("k,kij->ij", probabilities[slot], ops)
            costs["joint_matrix_multiplications"] += 1
        log_score = joint.clamp_min(1e-30).log().cpu().double().numpy()
        unconstrained_rows, unconstrained_cols = linear_sum_assignment(-log_score)
        unconstrained = np.empty(model.d, dtype=np.int64)
        unconstrained[unconstrained_rows] = unconstrained_cols
        # Finite penalty makes infeasibility explicit via the post-check.
        scored = np.where(mask, log_score, -1e30)
        assigned_rows, assigned_cols = linear_sum_assignment(-scored)
        if not np.all(mask[assigned_rows, assigned_cols]):
            raise ValueError("observed terminal constraints admit no full permutation")
        permutation = np.empty(model.d, dtype=np.int64)
        permutation[assigned_rows] = assigned_cols
        costs["assignment_calls"] += 2
        costs["assignment_score_entries"] += 2*model.d*model.d
        candidate["trans"][symbol] = permutation
        per_row_counts = mask.sum(axis=1)
        audit = dict(public_symbol=symbol, one_step_record_count=len(rows),
            actual_training_records=source_rows,
            MODEL_observed_source_states=sorted(observed_q),
            MODEL_unobserved_source_states=sorted(set(range(model.d))-observed_q),
            allowed_destinations_mask=mask.tolist(), allowed_destination_counts=per_row_counts.tolist(),
            rowwise_ambiguous_states=np.flatnonzero(per_row_counts > 1).tolist(),
            rowwise_unique_states=np.flatnonzero(per_row_counts == 1).tolist(),
            observed_terminal_constraints_satisfied=True,
            learned_joint_log_scores=log_score.tolist(),
            unconstrained_learned_joint_permutation=unconstrained.tolist(),
            constrained_learned_permutation=permutation.tolist(),
            changed_rows_from_unconstrained_matching=int(np.count_nonzero(permutation != unconstrained)),
            selected_log_score=float(log_score[assigned_rows, assigned_cols].sum()),
            uniqueness_scope="rowwise counts only; number of global feasible matchings not enumerated",
            teacher_operations_used=False, teacher_routes_used=False,
            latent_state_names_are="MODEL, not hidden-state labels")
        audits.append(audit)
        recorder.emit("terminal_constrained_operation_candidate", **audit)
    synchronize(model.encoder.device)
    costs["seconds"] = time.perf_counter()-began
    audit = dict(rule="maximize learned full-operator score subject to observed one-step terminal compatibility",
        training_only=True, fixed_task_id=stage, new_public_symbols=new_symbols,
        learned_encoder=encoder.tolist(), learned_decoder=decoder.tolist(),
        symbol_audits=audits, costs=costs,
        strongest_conventional_explanation="terminal-supervised constrained permutation/program learning",
        source="actual task training episodes plus CURRENT model representation and soft operator scores")
    candidate["formation_audit"] = audit
    return candidate, audit


def save_checkpoint(model, optimizer, original, recorder, updates, provenance, suffix):
    stage = int(original["stage"])
    path = recorder.out / f"topo-stage{stage}-step{int(original['step'])+updates}-observed-formation-{suffix}.pt"
    payload = dict(schema="terminal-architecture-checkpoint-v1", args=copy.deepcopy(original["args"]),
        model_name="topo", stage=stage, step=int(original["step"])+updates,
        model=model.state_dict(), optimizer=optimizer.state_dict(), model_extra=model.extra_state_dict(),
        masks_and_initialized_tasks={}, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
        python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        no_automatic_resume_or_retry=True, M0="NOT_PASSED",
        observed_operation_formation=dict(provenance=provenance, additional_updates=updates,
            original_training_args_preserved=True, extra_budget_not_retroactive=True))
    temporary = path.with_name(path.name+".tmp")
    torch.save(payload, temporary)
    if temporary.stat().st_size > MAX_FILE_BYTES:
        raise RuntimeError("checkpoint exceeds100MiB")
    temporary.replace(path)
    recorder.emit("checkpoint", file=path.name, bytes=path.stat().st_size,
                  sha256=sha256(path), actual_neural_weights=True, additional_updates=updates)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1)
    parser.add_argument("--stage2", dest="stage", action="store_const", const=2)
    parser.add_argument("--candidate-kind", choices=("one_step", "history"), default="one_step")
    parser.add_argument("--records", type=Path)
    parser.add_argument("--seed", type=int, default=9003)
    parser.add_argument("--max-seconds", type=float, default=850.)
    args = parser.parse_args()
    if (args.updates != 200 or not 0 <= args.seed < 2**31 or args.out.exists()
            or not 0 < args.max_seconds <= 1500 or not args.checkpoint.is_file()
            or args.checkpoint.stat().st_size > MAX_FILE_BYTES):
        parser.error("frozen200-update diagnostic requires trusted checkpoint, new output and finite bounds")
    if args.candidate_kind == "history" and (args.stage != 2 or args.records is None or not args.records.is_file()):
        parser.error("history candidate requires stage2 and original saved records")
    recorder = Recorder(args.out, args.max_seconds)
    device = torch.device(args.device)
    torch.set_num_threads(1)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("authorized GPU unavailable; no silent CPU substitute")
        torch.cuda.set_per_process_memory_fraction(.3, device)
        torch.cuda.reset_peak_memory_stats(device)
    files = ("observed_operation_formation.py", "integrate_native_repair.py", "topology_model.py",
             "semantic_promotion_model.py", "continuous_native_model.py", "continuous_monodromy.py",
             "train_monodromy_extension.py")
    if args.candidate_kind == "history":
        files += ("history_operation_formation.py",)
    provenance = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                     source_sha256={name: sha256(Path(__file__).with_name(name)) for name in files})
    costs = dict(train_seconds=0., guard_seconds=0., evaluation_seconds=0.,
        evaluation_soft_episodes=0, evaluation_native_transition_uses=0,
        training_current_episodes=0, training_old_episodes=0,
        guard_calls=0, guard_full=0, guard_backtracked=0, guard_rejected=0,
        guard_relation_checks=0, guard_relation_pairs=0)
    model = optimizer = original = None
    completed, stage = 0, args.stage
    try:
        original = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if (original.get("schema") != "terminal-architecture-checkpoint-v1"
                or original.get("model_name") != "topo" or int(original.get("stage", -1)) != stage
                or "path_omega" not in original["model"]):
            raise ValueError("actual PRIMARY continuous checkpoint of the requested stage required")
        model = source_model(original, device)
        if "candidate_graph" not in inspect.signature(model.materialize).parameters:
            raise RuntimeError("explicit inspected materialize(candidate_graph=None) hook is not deployed")
        if not callable(getattr(model, "geometry_regularizer", None)):
            raise ValueError("actual continuous primary model was not restored")
        model.set_geometry_deadline(time.monotonic()+max(.001, recorder.deadline-time.perf_counter()))
        model.stage, model.protection = stage, True
        if {int(r["task"]) for r in model.reference} != set(range(stage)):
            raise ValueError("completed old tasks and an unadmitted current stage required; no old contract revision")
        initial_references = copy.deepcopy(model.reference)
        initial_omega = model.path_omega.detach().clone()
        if "torch_rng" in original:
            torch.set_rng_state(original["torch_rng"])
        if "numpy_rng" in original:
            np.random.set_state(original["numpy_rng"])
        if "python_rng" in original:
            random.setstate(original["python_rng"])
        if device.type == "cuda" and original.get("cuda_rng"):
            torch.cuda.set_rng_state_all(original["cuda_rng"])
        training, training_source = read_dataset(args.datasets, stage, "train")
        recorder.emit("configuration", args=vars(args), provenance=provenance,
            source_model_class=type(model).__module__+"."+type(model).__name__,
            actual_neural_parameters=sum(p.numel() for p in model.parameters()),
            MAIN="TOPOLOGY_USER_LOCKED", primary_baseline="Transformer", M0="NOT_PASSED",
            acquisition_information=f"original stage{stage} terminal TRAIN episodes only",
            formation_gate="one current VALID gate: native accuracy>=.95 and no degradation; old relation exact",
            validation_never_selects_between_candidates=True, heldout_used_only_after_gate=True,
            additional_training="200freshAdam,batch64,LR.01,terminalCE+.01geometry; extra budget explicit",
            training_source=training_source, initial_shared_omega=initial_omega)
        if args.candidate_kind == "history":
            from history_operation_formation import candidate as history_candidate
            witness = None
            with args.records.open(encoding="utf8") as stream:
                for line in stream:
                    event = json.loads(line)
                    if (event.get("kind") == "observed_history_state_witness"
                            and event.get("model") == original["model_name"]
                            and int(event.get("stage", -1)) == stage):
                        witness = event
            if witness is None:
                raise ValueError("actual saved current-stage history witness missing")
            provenance["history_records_sha256"] = sha256(args.records)
            candidate, audit = history_candidate(model, stage, training, witness, recorder)
        else:
            candidate, audit = observed_candidate_graph(model, stage, training, recorder)
        recorder.json("candidate-formation-audit.json", audit)
        validation, validation_source = read_dataset(args.datasets, stage, "valid")
        native_before = model.graph(stage, ALPHABETS[stage])

        def native_accuracy(graph, data):
            q = graph["enc"][data["starts"]].copy()
            for column in range(data["tokens"].shape[1]):
                for a in np.unique(data["tokens"][:, column]):
                    if a:
                        selected = data["tokens"][:, column] == a
                        q[selected] = graph["trans"][int(a)][q[selected]]
            return float(np.mean(graph["dec"][q] == data["targets"]))

        valid_before = native_accuracy(native_before, validation)
        synchronize(device)
        formation_started = time.perf_counter()
        report = model.materialize(stage, ALPHABETS[stage], ALPHABETS, candidate_graph=candidate)
        synchronize(device)
        costs["neural_materialization_seconds"] = time.perf_counter()-formation_started
        recorder.emit("observed_native_materialization", report=report)
        if not report.get("success"):
            raise RuntimeError("data-constrained materialization failed: "+str(report.get("reason")))
        actual = model.graph(stage, ALPHABETS[stage])
        if (not np.array_equal(actual["enc"], candidate["enc"])
                or not np.array_equal(actual["dec"], candidate["dec"])
                or any(not np.array_equal(actual["trans"][a], candidate["trans"][a]) for a in ALPHABETS[stage])):
            raise RuntimeError("actual neural native program differs from selected data-constrained candidate")
        old_checks = []
        for reference in initial_references:
            okay, pairs = model.related(reference, model.graph(reference["task"], reference["tokens"]))
            old_checks.append(dict(task=reference["task"], related=bool(okay), relation_pairs=pairs))
            if not okay:
                raise RuntimeError("formation changed an old learned closed behavior")
        valid_after = native_accuracy(actual, validation)
        admitted = valid_after >= .95 and valid_after >= valid_before
        recorder.emit("single_validation_gate", before=valid_before, after=valid_after,
            admitted=admitted, validation_source=validation_source, old_checks=old_checks,
            source_shared_geometry_unchanged=torch.equal(initial_omega, model.path_omega))
        if not admitted:
            recorder.json("summary.json", dict(status="CANDIDATE_REJECTED_BY_VALIDATION",
                before_valid=valid_before, candidate_valid=valid_after, audit=audit,
                materialization=report, actual_additional_updates=0, M0="NOT_PASSED"))
            return
        admission = model.consolidate(stage, ALPHABETS[stage])
        recorder.emit("new_native_operation_admitted", reference=admission,
                      source="terminal-data-constrained candidate, actual neural installation and single validation")
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        installed = save_checkpoint(model, optimizer, original, recorder, 0, provenance, "installed")
        # Candidate and its admission are frozen before opening heldout splits.
        datasets, input_sources = {}, []
        for task in sorted(ALPHABETS):
            datasets[task] = {}
            for split in SPLITS:
                data, source = read_dataset(args.datasets, task, split)
                datasets[task][split] = data
                input_sources.append(source)
        recorder.emit("frozen_evaluation_inputs", input_datasets=input_sources,
                      only_training_task=stage, no_heldout_selection=True)
        generator = np.random.default_rng(args.seed)
        batches = generator.integers(0, len(training["starts"]), size=(args.updates, 64), dtype=np.int64)
        np.savez_compressed(recorder.out / "additional-training-indices.npz", indices=batches, seed=np.asarray(args.seed))
        recorder.emit("additional_training_begin", updates=args.updates, batch64=True, lr=.01,
            fresh_adam=True, old_replay_samples=0, current_task=stage,
            geometry_weight=.01, geometry_gradient_source="continuous kinetic action",
            batch_indices_sha256=sha256(recorder.out / "additional-training-indices.npz"),
            all_parameters_trainable=True, current_and_old_learned_references_guarded=True)
        curves = []
        final_checkpoint = installed
        for step in range(args.updates+1):
            recorder.check()
            if step in (0, 50, 100, 200):
                metrics = evaluate(model, datasets, ALPHABETS, device, 64, recorder, costs)
                curve = dict(step=step, metrics=metrics, costs=costs.copy(),
                    geometry_resources=model.geometry_resource_report(),
                    actual_graphs=[graph_summary(model.graph(t, a)) for t, a in ALPHABETS.items()],
                    parameters=sum(p.numel() for p in model.parameters()),
                    optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0)
                recorder.emit("curve", **curve)
                curves.append(curve)
                if step:
                    final_checkpoint = save_checkpoint(model, optimizer, original, recorder, step, provenance, "trained")
            if step == args.updates:
                break
            batch = {key: torch.as_tensor(value[batches[step]], device=device, dtype=torch.long)
                     for key, value in training.items()}
            synchronize(device)
            training_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["tokens"], batch["starts"], batch["task_ids"])
            ce = F.cross_entropy(logits, batch["targets"])
            geometry = model.geometry_regularizer()
            loss = ce+.01*geometry
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite terminal/geometry learning objective")
            loss.backward()
            omega_gradient = model.path_omega.grad.detach().clone()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gradient):
                raise FloatingPointError("nonfinite neural gradient")
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            optimizer.step()
            synchronize(device)
            costs["train_seconds"] += time.perf_counter()-training_started
            guard_started = time.perf_counter()
            guard = model.project_update(before)
            synchronize(device)
            costs["guard_seconds"] += time.perf_counter()-guard_started
            alpha = float(guard.get("alpha", 1.))
            costs["guard_calls"] += 1
            costs["guard_full" if alpha == 1 else "guard_backtracked" if alpha > 0 else "guard_rejected"] += 1
            costs["guard_relation_checks"] += int(guard.get("checks", 0))
            costs["guard_relation_pairs"] += int(guard.get("relation_pairs", 0))
            costs["training_current_episodes"] += 64
            completed = step+1
            recorder.emit("actual_neural_parameter_update", step=completed, terminal_CE=float(ce.item()),
                normalized_action=float(geometry.item()), total_loss=float(loss.item()),
                shared_omega=model.path_omega, preclip_omega_gradient=omega_gradient,
                preclip_gradient_norm=float(gradient.item()), old_and_current_guard=guard)
        restored_checkpoint = torch.load(final_checkpoint, map_location="cpu", weights_only=False)
        restored = source_model(restored_checkpoint, device)
        restored.set_geometry_deadline(time.monotonic()+max(.001, recorder.deadline-time.perf_counter()))
        if not all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items()):
            raise RuntimeError("actual saved/reloaded neural parameter mismatch")
        probe = {key: torch.as_tensor(value[:64], device=device, dtype=torch.long) for key, value in validation.items()}
        with torch.no_grad():
            direct = model.hard_eval(probe["tokens"], probe["starts"], probe["task_ids"])
            reloaded = restored.hard_eval(probe["tokens"], probe["starts"], probe["task_ids"])
            if not torch.equal(direct, reloaded):
                raise RuntimeError("reloaded actual continuous neural execution mismatch")
        recorder.emit("actual_checkpoint_reload_verified", checkpoint=final_checkpoint.name,
            all_neural_parameters_bitwise_equal=True, continuous_validation_probe_rows=64,
            actual_continuous_outputs_equal=True, restored_shared_omega=restored.path_omega,
            reload_geometry_resources=restored.geometry_resource_report())
        recorder.json("summary.json", dict(status="OBSERVED_OPERATION_FORMATION_AND200_NEURAL_UPDATES_COMPLETED",
            provenance=provenance, input_datasets=input_sources, formation_audit=audit,
            materialization=report, validation_before=valid_before, validation_after=valid_after,
            installed_checkpoint=installed.name, final_checkpoint=final_checkpoint.name,
            actual_additional_updates=completed, current_training_task=stage, curves=curves, costs=costs,
            geometry_resources=model.geometry_resource_report(),
            actual_neural_parameters=sum(p.numel() for p in model.parameters()),
            checkpoint_reload_actual_continuous_outputs_equal=True,
            wall_seconds=time.perf_counter()-recorder.started, M0="NOT_PASSED",
            strongest_conventional_explanation="constrained terminal-supervised permutation/program learning",
            evidence_boundary="one development checkpoint; actual neural operation formation, protected continuous execution and further training"))
        print(json.dumps(dict(status="OBSERVED_OPERATION_FORMATION_AND200_NEURAL_UPDATES_COMPLETED",
                              output=str(args.out), additional_updates=completed, costs=costs), sort_keys=True))
    except BaseException as error:
        if model is not None and optimizer is not None:
            try:
                save_checkpoint(model, optimizer, original, recorder, completed, provenance, "interrupted")
            except Exception as save_error:
                recorder.emit("interrupted_checkpoint_failed", reason=str(save_error))
        recorder.emit("failure", exception=type(error).__name__, reason=str(error), completed_updates=completed)
        recorder.json("failure.json", dict(status="FAILED_WITH_PARTIAL_RECORDS", exception=type(error).__name__,
            reason=str(error), actual_additional_updates=completed, costs=costs))
        raise
    finally:
        recorder.stream.close()


if __name__ == "__main__":
    main()
