"""Train continuous monodromy implementation INSIDE the current native model.

Loads only this campaign's trusted repaired neural checkpoint. Adds four shared
path-deformation parameters to that same model/optimizer, continues terminal CE,
and trains continuous kinetic action. Hard evaluations build the native bank by
actual unordered-configuration path lifting before executing learned call slots.
No teacher operation, hidden mode, or correct route enters this script.

The discrete learned operation classes remain the semantic acquisition layer.
The shared continuous layer changes their execution paths through a family of
endpoint-fixed diffeomorphisms. This diagnostic does not establish M0 or an
advantage over Transformer / conventional geometric compilation.

MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 NOT PASSED; M3 ultimate.
Run only inside the existing bounded external GPU runner. No sandbox at import.
"""

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from continuous_monodromy import (BIASES, FREQUENCIES, SPACING, ConfigurationLoop,
                                  lift_sequence)
from integrate_native_repair import (MAX_FILE_BYTES, Recorder, compare_graph,
    evaluate, graph_summary, load_datasets, plain, sha256, source_model,
    synchronize, tensor_bytes)


ALPHABETS = {0: (1, 2, 3), 1: (1, 2, 3, 4, 5), 2: (1, 2, 3, 4, 5, 6)}
INITIAL_OMEGA = (0.7, -0.5, 0.4, -0.3)
PATH_WEIGHT = 0.01
QUADRATURE_PER_SEGMENT = 16


def extended_model(checkpoint, device):
    """Reuse the real source loader, retaining the extension as model state."""
    source = dict(checkpoint)
    source["model"] = dict(checkpoint["model"])
    saved_omega = source["model"].pop("path_omega", None)
    model = source_model(source, device)
    value = (torch.as_tensor(INITIAL_OMEGA, device=device, dtype=model.encoder.dtype)
             if saved_omega is None else saved_omega.to(device=device, dtype=model.encoder.dtype))
    if value.shape != (4,) or not torch.isfinite(value).all():
        raise ValueError("trusted path_omega must contain four finite shared weights")
    model.register_parameter("path_omega", nn.Parameter(value.detach().clone()))
    model.stage, model.protection = 2, True
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model


def graph_bank(model):
    graphs = {task: model.graph(task, alphabet) for task, alphabet in ALPHABETS.items()}
    bank = graphs[0]["permutations"]
    for graph in graphs.values():
        if len(graph["permutations"]) != len(bank) or any(
                not np.array_equal(a, b) for a, b in zip(bank, graph["permutations"])):
            raise RuntimeError("shared learned native bank differs between public tasks")
    return graphs, bank


def kinetic_moments(permutations):
    """Midpoint quadrature of actual continuous paths, reduced to a quadratic.

    A(omega)=c+2 b^T omega+omega^T H omega, with c=integral sum_j|z'_j|².
    For z=x+iy, shear y'=y+a(t)h_omega(x) gives vertical velocity
    vy+sum_l omega_l[a' tanh(v_l*x+b_l)+a v_l sech²(v_l*x+b_l) vx].
    The loop skeleton and learned primitive classes are fixed during a single
    update; moments are rebuilt whenever the current active classes change.
    """
    constant, linear, quadratic = 0.0, np.zeros(4), np.zeros((4, 4))
    samples = 0
    for permutation in permutations:
        loop = ConfigurationLoop(permutation, [0.] * 4)
        for segment in range(loop.duration):
            u = (np.arange(QUADRATURE_PER_SEGMENT) + 0.5) / QUADRATURE_PER_SEGMENT
            t = segment + u
            x = np.broadcast_to(loop.base, (QUADRATURE_PER_SEGMENT, loop.d)).copy()
            vx, vy = np.zeros_like(x), np.zeros_like(x)
            if loop.word:
                j = loop.word[segment]
                h, hp = u*u*(3-2*u), 6*u*(1-u)
                midpoint = (loop.base[j] + loop.base[j+1]) / 2
                cosine, sine = np.cos(math.pi*h), np.sin(math.pi*h)
                x[:, j], x[:, j+1] = midpoint-2*cosine, midpoint+2*cosine
                vx[:, j], vx[:, j+1] = 2*math.pi*hp*sine, -2*math.pi*hp*sine
                vy[:, j], vy[:, j+1] = -2*math.pi*hp*cosine, 2*math.pi*hp*cosine
            a = np.sin(math.pi*t/loop.duration)**2
            ap = math.pi/loop.duration*np.sin(2*math.pi*t/loop.duration)
            hidden = np.tanh(x[..., None]*FREQUENCIES+BIASES)
            basis = (ap[:, None, None]*hidden + a[:, None, None]*FREQUENCIES
                     *(1-hidden*hidden)*vx[..., None])
            weight = 1/QUADRATURE_PER_SEGMENT
            constant += float(np.sum(vx*vx+vy*vy))*weight
            linear += np.einsum("qd,qdl->l", vy, basis)*weight
            quadratic += np.einsum("qdl,qdm->lm", basis, basis)*weight
            samples += len(u)*loop.d
    return dict(constant=constant, linear=linear, quadratic=quadratic,
                normalizer=max(1., constant), quadrature_point_strands=samples,
                operation_classes=len(permutations), quadrature_per_segment=QUADRATURE_PER_SEGMENT)


class ActionObjective:
    def __init__(self, model, costs):
        self.model, self.costs, self.signature, self.moments = model, costs, None, None

    def refresh(self):
        started = time.perf_counter()
        graph = self.model.graph(2, ALPHABETS[2])
        active = sorted(set(int(k) for k in graph["routes"].ravel()))
        # Deduplicate semantic classes, including a possibly used identity skip.
        classes = tuple(dict.fromkeys(tuple(int(q) for q in graph["permutations"][k]) for k in active))
        self.costs["geometry_class_graph_extractions"] += 1
        self.costs["geometry_class_bank_assignments"] += self.model.n_ops
        self.costs["geometry_class_check_seconds"] += time.perf_counter()-started
        changed = classes != self.signature
        if changed:
            started = time.perf_counter()
            values = kinetic_moments(classes)
            self.signature, self.moments = classes, values
            self.linear = self.model.path_omega.new_tensor(values["linear"])
            self.quadratic = self.model.path_omega.new_tensor(values["quadratic"])
            self.costs["geometry_moment_rebuilds"] += 1
            self.costs["geometry_quadrature_point_strands"] += values["quadrature_point_strands"]
            self.costs["geometry_moment_seconds"] += time.perf_counter()-started
        return changed

    def __call__(self):
        self.refresh()
        omega = self.model.path_omega
        raw = self.moments["constant"]+2*torch.dot(self.linear, omega)+omega@self.quadratic@omega
        return raw/self.moments["normalizer"], raw

    def report(self):
        return dict(active_learned_classes=[list(p) for p in self.signature], **self.moments)


@torch.no_grad()
def continuous_hard_evaluate(model, datasets, recorder, costs, label):
    """Cold actual path realization, then warm calls of the resulting maps."""
    recorder.check()
    started = time.perf_counter()
    graphs, bank = graph_bank(model)
    costs["hard_bank_graph_extractions"] += len(graphs)
    costs["hard_bank_assignments"] += len(graphs)*model.n_ops
    omega = model.path_omega.detach().cpu().double().numpy()
    cache, descriptions = {}, []
    for expert, permutation in enumerate(bank):
        recorder.check()
        key = tuple(int(q) for q in permutation)
        if key in cache:
            continue
        loop = ConfigurationLoop(permutation, omega)
        trace = recorder.out/f"continuous-{label}-class{len(cache)}.jsonl"
        lifted = lift_sequence([loop], time.monotonic()+max(.001, recorder.deadline-time.perf_counter()), trace)
        actual = np.asarray(lifted["monodromy"], dtype=np.int64)
        if not np.array_equal(actual, permutation):
            raise RuntimeError("continuous primitive lift disagrees with real learned native bank")
        cache[key] = actual
        costs["cold_realized_classes"] += 1
        costs["cold_configuration_snapshots"] += lifted["snapshots"]
        costs["cold_distance_matrix_entries"] += lifted["distance_matrix_entries_evaluated"]
        descriptions.append(dict(class_id=len(cache)-1, first_expert=expert, native_permutation=permutation,
            certificate=loop.certificate(), **lifted))
    cold_seconds = time.perf_counter()-started
    costs["cold_realization_seconds"] += cold_seconds
    maps = [cache[tuple(int(q) for q in permutation)] for permutation in bank]
    cache_bytes = sum(value.nbytes for value in cache.values())
    warm_start, result, uses = time.perf_counter(), {}, 0
    for task, splits in datasets.items():
        graph = graphs[task]
        routes = {int(a): calls for a, calls in zip(graph["tokens"], graph["routes"])}
        result[task] = {}
        for split, arrays in splits.items():
            recorder.check()
            state = graph["enc"][arrays["starts"]].copy()
            for column in range(arrays["tokens"].shape[1]):
                symbols = arrays["tokens"][:, column]
                for token in np.unique(symbols):
                    if not token:
                        continue
                    selected = symbols == token
                    for expert in routes[int(token)]:
                        state[selected] = maps[int(expert)][state[selected]]
                        uses += int(np.count_nonzero(selected))
            correct = int(np.count_nonzero(graph["dec"][state] == arrays["targets"]))
            result[task][split] = dict(n=len(state), correct=correct, continuous_native_acc=correct/len(state))
    warm_seconds = time.perf_counter()-warm_start
    costs["warm_native_execution_seconds"] += warm_seconds
    costs["warm_native_primitive_calls"] += uses
    costs["peak_native_map_cache_bytes"] = max(costs["peak_native_map_cache_bytes"], cache_bytes)
    report = dict(label=label, metrics=result, shared_omega=omega, cold_realization_seconds=cold_seconds,
        cold_unique_classes=len(cache), cold_descriptions=descriptions, warm_execution_seconds=warm_seconds,
        warm_primitive_calls=uses, native_map_cache_bytes=cache_bytes,
        execution="unique continuous cover-lifts cached; actual learned encoder/call-slots/decoder execute those maps",
        initialization_and_cold_realization_counted_separately_from_cached_inference=True)
    recorder.emit("continuous_native_hard_evaluation", **report)
    return report


def save_extended(model, optimizer, original, recorder, completed, provenance, suffix):
    path = recorder.out/f"topo-stage2-step{int(original['step'])+completed}-monodromy-{suffix}.pt"
    extra = model.extra_state_dict()
    extra["continuous_monodromy"] = dict(kind="shared_endpoint_fixed_nonlinear_vertical_shear",
        state_key="path_omega", dimensions=4, frequencies=FREQUENCIES, biases=BIASES,
        spacing=SPACING, kinetic_weight=PATH_WEIGHT, quadrature_per_segment=QUADRATURE_PER_SEGMENT,
        checkpoint_loader="train_monodromy_extension.extended_model", native_execution="actual lift-derived cached maps")
    payload = dict(schema="terminal-architecture-checkpoint-v1", args=copy.deepcopy(original["args"]),
        model_name="topo", stage=2, step=int(original["step"])+completed,
        model=model.state_dict(), optimizer=optimizer.state_dict(), model_extra=extra,
        masks_and_initialized_tasks={}, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
        python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        continuous_training=dict(provenance=provenance, completed_additional_updates=completed,
                                 additional_budget_not_retroactive=True), M0="NOT_PASSED")
    temporary = path.with_name(path.name+".tmp")
    torch.save(payload, temporary)
    if temporary.stat().st_size > MAX_FILE_BYTES:
        raise RuntimeError("checkpoint exceeds100MiB")
    temporary.replace(path)
    recorder.emit("checkpoint", file=path.name, bytes=path.stat().st_size, sha256=sha256(path),
                  actual_extra_neural_parameters=4, additional_updates=completed)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=9002)
    parser.add_argument("--max-seconds", type=float, default=850.)
    args = parser.parse_args()
    if (args.updates != 200 or args.batch != 64 or args.lr != .01 or not 0 < args.max_seconds <= 1500
            or not 0 <= args.seed < 2**31 or args.out.exists()):
        parser.error("declared diagnostic requires200updates,batch64,LR.01,newoutput and bounded wall/seed")
    if not args.checkpoint.is_file() or args.checkpoint.stat().st_size > MAX_FILE_BYTES:
        parser.error("trusted neural checkpoint absent or over100MiB")
    recorder = Recorder(args.out, args.max_seconds)
    device = torch.device(args.device)
    torch.set_num_threads(1)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("authorized GPU unavailable; no silent CPU substitution")
        torch.cuda.set_per_process_memory_fraction(.3, device)
        torch.cuda.reset_peak_memory_stats(device)
    code = [Path(__file__), Path(__file__).with_name("continuous_monodromy.py"),
            Path(__file__).with_name("integrate_native_repair.py"),
            Path(__file__).with_name("topology_model.py"), Path(__file__).with_name("semantic_promotion_model.py")]
    provenance = dict(checkpoint_path=str(args.checkpoint), checkpoint_sha256=sha256(args.checkpoint),
                      source_code_sha256={p.name: sha256(p) for p in code})
    costs = dict(train_seconds=0., guard_seconds=0., evaluation_seconds=0.,
        evaluation_soft_episodes=0, evaluation_native_transition_uses=0,
        training_current_episodes=0, training_old_episodes=0,
        guard_calls=0, guard_full=0, guard_backtracked=0, guard_rejected=0,
        guard_relation_checks=0, guard_relation_pairs=0,
        geometry_class_graph_extractions=0, geometry_class_bank_assignments=0,
        geometry_class_check_seconds=0., geometry_moment_rebuilds=0,
        geometry_quadrature_point_strands=0, geometry_moment_seconds=0.,
        hard_bank_graph_extractions=0, hard_bank_assignments=0,
        cold_realized_classes=0, cold_configuration_snapshots=0, cold_distance_matrix_entries=0,
        cold_realization_seconds=0., warm_native_execution_seconds=0.,
        warm_native_primitive_calls=0, peak_native_map_cache_bytes=0)
    model = optimizer = original = None
    completed, curves = 0, []
    try:
        original = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if (original.get("schema") != "terminal-architecture-checkpoint-v1"
                or original.get("model_name") != "topo" or int(original.get("stage", -1)) != 2):
            raise ValueError("actual current repaired stage2 neural checkpoint required")
        model = extended_model(original, device)
        if {int(r["task"]) for r in model.reference} != {0, 1, 2}:
            raise ValueError("all three learned neural references required")
        initial_parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
        initial_omega = model.path_omega.detach().cpu().clone()
        datasets, input_sources = load_datasets(args.datasets, ALPHABETS, recorder)
        recorder.emit("configuration", args=vars(args), provenance=provenance, MAIN="TOPOLOGY_USER_LOCKED",
            M0="NOT_PASSED", M3="ultimate", primary_baseline="Transformer", actual_new_neural_parameters=4,
            path_omega_initial=model.path_omega, all_native_parameters_continue_training=True,
            objective="terminal_CE+.01*normalized_continuous_kinetic_action",
            terminal_training_task=2, no_intermediate_targets=True, no_new_teacher_information=True,
            cold_path_realization_and_warm_cached_inference_separately_counted=True,
            action_is_numerical_midpoint_quadrature_not_formal_integral_certificate=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        action = ActionObjective(model, costs)
        generator = np.random.default_rng(args.seed)
        indices = generator.integers(0, len(datasets[2]["train"]["starts"]),
                                     size=(args.updates, args.batch), dtype=np.int64)
        np.savez_compressed(recorder.out/"monodromy-training-batch-indices.npz", indices=indices, seed=args.seed)
        for step in range(args.updates+1):
            recorder.check()
            if step in (0, 100, 200):
                objective, raw_action = action()
                neural_metrics = evaluate(model, datasets, ALPHABETS, device, args.batch, recorder, costs)
                continuous = continuous_hard_evaluate(model, datasets, recorder, costs, f"step{step}")
                if any(neural_metrics[t][s]["native_acc"] != continuous["metrics"][t][s]["continuous_native_acc"]
                       for t in datasets for s in datasets[t]):
                    raise RuntimeError("actual path execution disagrees with actual neural native execution")
                curve = dict(step=step, neural_metrics=neural_metrics, continuous_metrics=continuous["metrics"],
                    path_omega=model.path_omega.detach().clone(), normalized_action=float(objective.item()),
                    kinetic_action=float(raw_action.item()), action_moments=action.report(),
                    params=sum(p.numel() for p in model.parameters()),
                    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                    optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
                    representation_state_bytes=model.representation_state_bytes(),
                    gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                    actual_graphs=[graph_summary(model.graph(t,a)) for t,a in ALPHABETS.items()], costs=costs.copy())
                curves.append(plain(curve))
                recorder.emit("curve", **curve)
                save_extended(model, optimizer, original, recorder, step, provenance, "trained")
            if step == args.updates:
                break
            batch = {name: torch.as_tensor(value[indices[step]], dtype=torch.long, device=device)
                     for name, value in datasets[2]["train"].items()}
            synchronize(device)
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["tokens"], batch["starts"], batch["task_ids"])
            terminal = F.cross_entropy(logits, batch["targets"])
            normalized_action, raw_action = action()
            loss = terminal+PATH_WEIGHT*normalized_action
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite fused objective")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(gradient):
                raise FloatingPointError("nonfinite fused training gradient")
            omega_gradient = model.path_omega.grad.detach().clone()
            before = {name: p.detach().clone() for name,p in model.named_parameters()}
            optimizer.step()
            synchronize(device)
            costs["train_seconds"] += time.perf_counter()-began
            guard_begin = time.perf_counter()
            guard = model.project_update(before)
            synchronize(device)
            costs["guard_seconds"] += time.perf_counter()-guard_begin
            alpha = float(guard.get("alpha", 1.))
            costs["guard_calls"] += 1
            costs["guard_full" if alpha == 1 else "guard_backtracked" if alpha > 0 else "guard_rejected"] += 1
            costs["guard_relation_checks"] += int(guard.get("checks",0))
            costs["guard_relation_pairs"] += int(guard.get("relation_pairs",0))
            costs["training_current_episodes"] += args.batch
            if not torch.isfinite(model.path_omega).all() or model.path_omega.abs().max().item() > 8:
                raise ValueError("shared geometric weights exceed numerical diagnostic bound")
            completed = step+1
            recorder.emit("actual_fused_parameter_update", step=completed,
                terminal_CE=float(terminal.item()), normalized_action=float(normalized_action.item()),
                kinetic_action=float(raw_action.item()), objective=float(loss.item()),
                preclip_gradient_norm=float(gradient.item()), path_omega=model.path_omega,
                path_gradient=omega_gradient, guard=guard)
        final = save_extended(model, optimizer, original, recorder, completed, provenance, "final")
        # Actual reload through the extension-aware source loader, then geometry
        # is re-realized before evaluating all15 original data splits again.
        loaded = torch.load(final, map_location="cpu", weights_only=False)
        restored = extended_model(loaded, device)
        exact_tensors = all(torch.equal(model.state_dict()[k], restored.state_dict()[k]) for k in model.state_dict())
        exact_graphs = all(compare_graph(model.graph(t,a), restored.graph(t,a)) for t,a in ALPHABETS.items())
        if not exact_tensors or not exact_graphs:
            raise RuntimeError("saved fused neural state did not reload exactly")
        reload_metrics = continuous_hard_evaluate(restored, datasets, recorder, costs, "reload-final")
        if plain(reload_metrics["metrics"]) != curves[-1]["continuous_metrics"]:
            raise RuntimeError("reloaded continuous native execution changed measured behavior")
        movement = {name: float(torch.linalg.vector_norm(p.detach()-initial_parameters[name]).item())
                    for name,p in model.named_parameters()}
        recorder.emit("actual_checkpoint_reload_verified", checkpoint=final.name,
            exact_neural_tensors=exact_tensors, exact_native_graphs=exact_graphs,
            continuous_metrics_unchanged=True, shared_omega=restored.path_omega)
        recorder.json("summary.json", dict(status="FUSED_NEURAL_AND_CONTINUOUS_MONODROMY_TRAINING_COMPLETED",
            MAIN="TOPOLOGY_USER_LOCKED", M0="NOT_PASSED", M3="ultimate", primary_baseline="Transformer",
            actual_additional_updates=completed, original_checkpoint_step=int(original["step"]),
            provenance=provenance, input_datasets=input_sources, params=sum(p.numel() for p in model.parameters()),
            initial_omega=initial_omega, final_omega=model.path_omega,
            shared_omega_actually_changed=not torch.equal(initial_omega,model.path_omega.detach().cpu()),
            per_parameter_L2_movement=movement, curves=curves, costs=costs, final_checkpoint=final.name,
            actual_checkpoint_reload_verified=True, wall_seconds=time.perf_counter()-recorder.started,
            boundary="one development checkpoint; actual path-trained parameter state and continuous native execution, no comparative topology advantage orM0pass"))
        print(json.dumps(dict(status="FUSED_NEURAL_AND_CONTINUOUS_MONODROMY_TRAINING_COMPLETED",
                              out=str(args.out), updates=completed, costs=costs),sort_keys=True))
    except BaseException as error:
        if model is not None and optimizer is not None and original is not None:
            try:
                save_extended(model,optimizer,original,recorder,completed,provenance,"interrupted")
            except Exception as save_error:
                recorder.emit("interrupted_checkpoint_failed", error=str(save_error))
        recorder.emit("interrupted", error_type=type(error).__name__,error=str(error), completed_updates=completed)
        recorder.json("failure.json",dict(status="FAILED_WITH_PARTIAL_RECORDS",error_type=type(error).__name__,
                                         error=str(error),completed_updates=completed,costs=costs))
        raise
    finally:
        recorder.stream.close()


if __name__ == "__main__":
    main()
