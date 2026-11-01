"""Four-arm causal/cost diagnostic inside the SAME continuous native architecture.

From ONE actual N16 pre-history-formation stage2 checkpoint, train200 additional
steps in each execution_backend={monodromy,direct} x binding_guard={on,off} arm.
Same real source weights/STRIDES, source RNG, fresh Adam, batches and geometric
objective. Direct bypasses only continuous path lifting in native evaluation;
all arms retain path_omega, its protected shear family and geometric learning.
This does NOT remove all topology, supply new correct operations, or compare a
new architecture against a weak Transformer surrogate. Transformer stays the
primary architecture baseline; this is an internal mechanism diagnostic.

No teacher functions/history formation, network, subprocess or import-time run.
Use the existing external GPU isolation runner, outer900s/internal<=850s.
MAIN=TOPOLOGY USER_LOCKED; M0 NOT PASSED; M3 ultimate; Transformer primary.
"""
import argparse
import copy
import hashlib
import json
import random
import time
import types
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from continuous_native_model import ContinuousNativeModel, _new_costs
from diagnose_projection_numeric import restore_parameter_layouts
from integrate_native_repair import (MAX_FILE_BYTES, Recorder, graph_summary,
    plain, sha256, source_model, synchronize, tensor_bytes)
from semantic_promotion_model import SemanticPromotionModel


ALPHABETS = {0: (1, 2, 3), 1: (1, 2, 3, 4, 5), 2: (1, 2, 3, 4, 5, 6)}
SPLITS = ("train", "valid", "iid", "ood", "length16")
ARM_SPECS = (("monodromy", True), ("direct", True),
             ("monodromy", False), ("direct", False))
STEPS, BATCH, SEED, LR, GEOMETRY_WEIGHT = 200, 64, 9011, .01, .01


def _direct_hard_eval(model, tokens, starts, task_ids):
    return SemanticPromotionModel.hard_eval(model, tokens, starts, task_ids)


def configure_arm(model, backend, binding_guard, reset_costs=True):
    if not isinstance(model, ContinuousNativeModel):
        raise TypeError("every arm must keep the actual continuous native architecture")
    model.protection = bool(binding_guard)
    model.stage = 2
    model.compilation = "component"
    if backend == "direct":
        model.hard_eval = types.MethodType(_direct_hard_eval, model)
    elif backend != "monodromy":
        raise ValueError("unknown actual execution backend")
    model._invalidate_derived_caches()
    if reset_costs:
        # Keep the source's full historical resource report in provenance. These
        # are NEW-study-only counters, not a rewrite of old work or free caches.
        model._geometry_costs = _new_costs()
    return model


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def restore_rng(source, device):
    torch.set_rng_state(source["torch_rng"])
    np.random.set_state(source["numpy_rng"])
    random.setstate(source["python_rng"])
    if device.type == "cuda" and source.get("cuda_rng"):
        torch.cuda.set_rng_state_all(source["cuda_rng"])


def load_datasets(folder, recorder):
    datasets, provenance = {}, []
    for task in sorted(ALPHABETS):
        datasets[task] = {}
        for split in SPLITS:
            path = folder/f"task{task}-{split}.npz"
            if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("actual original dataset missing or over100MiB")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: np.asarray(archive[k], dtype=np.int64)
                          for k in ("tokens", "starts", "targets", "task_ids")}
            count = len(arrays["starts"])
            expected = 4096 if split == "train" else 512
            if (count != expected or arrays["tokens"].shape != (count,18)
                    or arrays["targets"].shape != (count,) or arrays["task_ids"].shape != (count,)
                    or np.any(arrays["task_ids"] != task)
                    or np.any(arrays["starts"] < 0) or np.any(arrays["starts"] >= 16)
                    or np.any(arrays["targets"] < 0) or np.any(arrays["targets"] >= 16)
                    or set(int(x) for x in np.unique(arrays["tokens"]))-set((0,)+ALPHABETS[task])):
                raise ValueError("original data task/shape/code bounds differ")
            datasets[task][split] = arrays
            provenance.append(dict(task=task,split=split,file=str(path),sha256=sha256(path),
                                   rows=count,array_bytes=sum(a.nbytes for a in arrays.values())))
    recorder.emit("input_datasets",datasets=provenance,
                  training_feedback="task2 TRAIN only; old/evaluation labels never select updates")
    return datasets, provenance


@torch.no_grad()
def graphs(model):
    return {task:model.graph(task,alphabet) for task,alphabet in ALPHABETS.items()}


def relations(model, references, current, check):
    checks = []
    for old in references:
        check()
        okay, count = model.related(old,current[int(old["task"])])
        checks.append(dict(task=int(old["task"]),related=bool(okay),relation_pairs=int(count)))
    return dict(passed=all(x["related"] for x in checks),
                violated_old_tasks=[x["task"] for x in checks if not x["related"]],checks=checks)


def functional_changes(model, before, after, check):
    result = {}
    for task in sorted(before):
        check()
        a,b=before[task],after[task]
        equivalent,pairs=model.related(a,b)
        result[task]=dict(encoder_entries_changed=int(np.count_nonzero(a["enc"]!=b["enc"])),
            decoder_entries_changed=int(np.count_nonzero(a["dec"]!=b["dec"])),
            transition_entries_changed={int(k):int(np.count_nonzero(a["trans"][k]!=b["trans"][k]))
                                        for k in a["tokens"]},
            decoded_all_word_behavior_equal=bool(equivalent),relation_pairs=int(pairs))
    return result


def graph_batch_error(graph, batch):
    state=graph["enc"][batch["starts"]].copy()
    for column in range(batch["tokens"].shape[1]):
        symbols=batch["tokens"][:,column]
        for a in np.unique(symbols):
            if a:
                mask=symbols==a
                state[mask]=graph["trans"][int(a)][state[mask]]
    return int(np.count_nonzero(graph["dec"][state]!=batch["targets"]))


@torch.no_grad()
def evaluate(model,datasets,device,recorder,costs):
    """All15 splits; backend_native names remain honest for DIRECT arms."""
    model.eval()
    synchronize(device)
    began=time.perf_counter()
    current=graphs(model)
    metrics={}
    for task,splits in datasets.items():
        metrics[task]={}
        for split,data in splits.items():
            recorder.check()
            count=len(data["starts"])
            graph_errors=graph_batch_error(current[task],data)
            loss=0.;soft_correct=native_correct=0
            backend_seconds=0.
            for start in range(0,count,BATCH):
                recorder.check()
                b={k:torch.as_tensor(v[start:start+BATCH],device=device,dtype=torch.long)
                   for k,v in data.items()}
                logits=model(b["tokens"],b["starts"],b["task_ids"])
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("nonfinite actual neural evaluation")
                loss+=float(F.cross_entropy(logits,b["targets"],reduction="sum").item())
                soft_correct+=int((logits.argmax(-1)==b["targets"]).sum().item())
                synchronize(device)
                call_start=time.perf_counter()
                hard=model.hard_eval(b["tokens"],b["starts"],b["task_ids"])
                synchronize(device)
                backend_seconds+=time.perf_counter()-call_start
                native_correct+=int((hard.argmax(-1)==b["targets"]).sum().item())
            graph_accuracy=1-graph_errors/count
            backend_accuracy=native_correct/count
            if backend_accuracy != graph_accuracy:
                raise RuntimeError("actual backend outputs differ from native semantic graph")
            metrics[task][split]=dict(n=count,CE=loss/count,soft_acc=soft_correct/count,
                graph_native_acc=graph_accuracy,backend_native_acc=backend_accuracy,
                backend_matches_graph=True,actual_backend_seconds=backend_seconds)
            costs["evaluation_soft_episodes"]+=count
            costs["evaluation_backend_native_episodes"]+=count
            costs["evaluation_graph_transition_uses"]+=int(np.count_nonzero(data["tokens"]))
            costs["evaluation_backend_seconds"]+=backend_seconds
    synchronize(device)
    costs["evaluation_seconds"]+=time.perf_counter()-began
    model.train()
    return metrics,current


def save_final(model,optimizer,source,recorder,arm,backend,guard,provenance):
    path=recorder.out/f"{arm}-real-step{int(source['step'])+STEPS}.pt"
    payload=dict(schema="terminal-architecture-checkpoint-v1",args=copy.deepcopy(source["args"]),
        model_name="topo",stage=2,step=int(source["step"])+STEPS,
        model=model.state_dict(),optimizer=optimizer.state_dict(),model_extra=model.extra_state_dict(),
        masks_and_initialized_tasks={},torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
        python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        monodromy_contribution_study=dict(provenance=provenance,execution_backend=backend,
            binding_guard=bool(guard),source_step=source["step"],actual_additional_updates=STEPS,
            extra_budget_not_retroactive=True,geometry_present_in_all_arms=True),M0="NOT_PASSED")
    temporary=path.with_name(path.name+".tmp")
    torch.save(payload,temporary)
    if temporary.stat().st_size>MAX_FILE_BYTES:
        temporary.unlink()
        raise RuntimeError("saved real checkpoint exceeds100MiB")
    temporary.replace(path)
    recorder.emit("checkpoint",arm=arm,file=path.name,bytes=path.stat().st_size,
                  sha256=sha256(path),actual_neural_parameters=True)
    return path


def run_arm(backend,guard,source,datasets,indices,device,recorder,provenance):
    arm=f"{backend}-guard-{'on' if guard else 'off'}"
    arm_began=time.perf_counter()
    costs=dict(train_seconds=0.,guard_seconds=0.,snapshot_seconds=0.,
        raw_proposal_diagnostic_seconds=0.,accepted_graph_diagnostic_seconds=0.,
        evaluation_seconds=0.,evaluation_backend_seconds=0.,
        evaluation_soft_episodes=0,evaluation_backend_native_episodes=0,
        evaluation_graph_transition_uses=0,training_current_sample_uses=0,training_old_sample_uses=0,
        gradient_forwards=0,gradient_backwards=0,Adam_proposals=0,
        guard_calls=0,guard_full=0,guard_backtracked=0,guard_rejected=0,
        guard_relation_checks=0,guard_relation_pairs=0,
        raw_diagnostic_old_relation_checks=0,raw_diagnostic_old_relation_pairs=0,
        raw_proposals_violating_old_contract=0,raw_violation_task_counts={0:0,1:0},
        accepted_old_contract_violation_steps=0,parameter_hash_seconds=0.,checkpoint_seconds=0.,
        reload_seconds=0.)
    recorder.check()
    model=source_model(source,device)
    layouts=restore_parameter_layouts(model,source["model"])
    configure_arm(model,backend,guard)
    model.set_geometry_deadline(time.monotonic()+max(.001,recorder.deadline-time.perf_counter()))
    if {int(r["task"]) for r in model.reference}!={0,1}:
        raise ValueError("actual source must retain tasks0/1, not an already admitted task2")
    references=copy.deepcopy(model.reference)
    for p in model.parameters():p.requires_grad_(True)
    optimizer=torch.optim.Adam(model.parameters(),lr=LR)
    restore_rng(source,device)
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    original=graphs(model)
    source_certificate=relations(model,references,original,recorder.check)
    if not source_certificate["passed"]:
        raise ValueError("source already violates its learned old reference contract")
    source_hash=parameter_hash(model)
    recorder.emit("arm_begin",arm=arm,execution_backend=backend,binding_guard=guard,
        source_parameter_hash=source_hash,parameter_layouts=layouts,
        parameters=sum(p.numel() for p in model.parameters()),
        requires_grad_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        shared_path_parameters=model.path_omega,fresh_Adam=True,optimizer_state_from_source_used=False,
        source_RNG_restored=True,source_old_certificate=source_certificate,
        backend_definition=("actual continuous lifts then cached maps" if backend=="monodromy"
                            else "base SemanticPromotionModel.hard_eval; direct learned graph execution"),
        ablation_scope="execution necessity, not deletion of all topology; same4omega+geometryobjective in ALLarms",
        old_proposal_diagnostics_do_NOT_constrain_OFF=True,
        direct_implementation_not_claimed_optimal_cached_program_executor=True)
    curves=[];proposal_hashes=[];violations=[];completed=0
    for step in range(STEPS+1):
        recorder.check()
        if step in (0,100,200):
            metrics,actual=evaluate(model,datasets,device,recorder,costs)
            curve=dict(step=step,metrics=metrics,parameters=sum(p.numel() for p in model.parameters()),
                parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                optimizer_bytes=tensor_bytes(optimizer.state_dict()),
                reference_array_bytes=model.representation_state_bytes(),
                gpu_peak_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0,
                shared_omega=model.path_omega.detach().cpu().tolist(),
                actual_graphs=[graph_summary(g) for g in actual.values()],
                geometry_resources=model.geometry_resource_report(),costs=costs.copy())
            curves.append(plain(curve))
            recorder.emit("curve",arm=arm,execution_backend=backend,binding_guard=guard,**curve)
        if step==STEPS:break
        rows=indices[step]
        cpu_batch={k:v[rows] for k,v in datasets[2]["train"].items()}
        batch={k:torch.as_tensor(v,device=device,dtype=torch.long) for k,v in cpu_batch.items()}
        synchronize(device)
        diagnostic_start=time.perf_counter()
        before_graphs=graphs(model)
        costs["raw_proposal_diagnostic_seconds"]+=time.perf_counter()-diagnostic_start
        snapshot_start=time.perf_counter()
        before={name:p.detach().clone() for name,p in model.named_parameters()}
        synchronize(device)
        costs["snapshot_seconds"]+=time.perf_counter()-snapshot_start
        began=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits=model(batch["tokens"],batch["starts"],batch["task_ids"])
        terminal=F.cross_entropy(logits,batch["targets"])
        geometry=model.geometry_regularizer()
        loss=terminal+GEOMETRY_WEIGHT*geometry
        if not torch.isfinite(loss):raise FloatingPointError("nonfinite actual terminal/geometric objective")
        loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(gradient):raise FloatingPointError("nonfinite actual gradient")
        omega_gradient=model.path_omega.grad.detach().clone()
        optimizer.step()
        synchronize(device)
        costs["train_seconds"]+=time.perf_counter()-began
        costs["training_current_sample_uses"]+=BATCH
        costs["gradient_forwards"]+=1;costs["gradient_backwards"]+=1;costs["Adam_proposals"]+=1
        diagnostic_start=time.perf_counter()
        proposed=graphs(model)
        raw_certificate=relations(model,references,proposed,recorder.check)
        raw_changes=functional_changes(model,before_graphs,proposed,recorder.check)
        raw_old_violation=not raw_certificate["passed"]
        costs["raw_diagnostic_old_relation_checks"]+=len(raw_certificate["checks"])
        costs["raw_diagnostic_old_relation_pairs"]+=sum(x["relation_pairs"] for x in raw_certificate["checks"])
        costs["raw_proposals_violating_old_contract"]+=int(raw_old_violation)
        for task in raw_certificate["violated_old_tasks"]:costs["raw_violation_task_counts"][task]+=1
        before_errors=graph_batch_error(before_graphs[2],cpu_batch)
        raw_errors=graph_batch_error(proposed[2],cpu_batch)
        costs["raw_proposal_diagnostic_seconds"]+=time.perf_counter()-diagnostic_start
        hstart=time.perf_counter()
        raw_hash=parameter_hash(model)
        costs["parameter_hash_seconds"]+=time.perf_counter()-hstart
        proposal_hashes.append(raw_hash);violations.append(raw_old_violation)
        guard_begin=time.perf_counter()
        if guard:
            guarded=model.project_update(before)
            synchronize(device)
            costs["guard_calls"]+=1
            alpha=float(guarded.get("alpha",1.))
            costs["guard_full" if alpha==1 else "guard_backtracked" if alpha>0 else "guard_rejected"]+=1
            costs["guard_relation_checks"]+=int(guarded.get("checks",0))
            costs["guard_relation_pairs"]+=int(guarded.get("relation_pairs",0))
        else:
            guarded=dict(accepted=True,alpha=1.,checks=0,relation_pairs=0,guard_disabled=True)
        costs["guard_seconds"]+=time.perf_counter()-guard_begin
        accepted_begin=time.perf_counter()
        accepted=graphs(model)
        accepted_certificate=relations(model,references,accepted,recorder.check)
        if not accepted_certificate["passed"]:costs["accepted_old_contract_violation_steps"]+=1
        if guard and not accepted_certificate["passed"]:
            raise RuntimeError("actual ON guard did not retain old contract")
        accepted_errors=graph_batch_error(accepted[2],cpu_batch)
        costs["accepted_graph_diagnostic_seconds"]+=time.perf_counter()-accepted_begin
        completed=step+1
        recorder.emit("actual_Adam_proposal_and_application",arm=arm,step=completed,
            terminal_CE=float(terminal.item()),normalized_action=float(geometry.item()),
            total_loss=float(loss.item()),preclip_gradient_norm=float(gradient.item()),
            path_gradient_after_clip=omega_gradient,shared_omega=model.path_omega,
            raw_proposal_parameter_hash=raw_hash,raw_proposal_graphs=[graph_summary(g) for g in proposed.values()],
            proposal_functional_changes=raw_changes,
            old_contract_before_guard=raw_certificate,application=guarded,
            old_contract_after_application=accepted_certificate,
            current_batch_native_errors_before=before_errors,current_batch_native_errors_raw=raw_errors,
            current_batch_native_errors_after=accepted_errors,
            diagnostic_old_checks_do_NOT_modify_OFF_proposal=True,
            optimizer_state_not_rolled_back_on_guard_rejection=True)
    end_hash=parameter_hash(model)
    saved_start=time.perf_counter()
    path=save_final(model,optimizer,source,recorder,arm,backend,guard,provenance)
    costs["checkpoint_seconds"]+=time.perf_counter()-saved_start
    reload_start=time.perf_counter()
    loaded=torch.load(path,map_location="cpu",weights_only=False)
    restored=source_model(loaded,device)
    reload_layouts=restore_parameter_layouts(restored,loaded["model"])
    configure_arm(restored,backend,guard)
    restored.set_geometry_deadline(time.monotonic()+max(.001,recorder.deadline-time.perf_counter()))
    reload_hash=parameter_hash(restored)
    if reload_hash!=end_hash:raise RuntimeError("actual final checkpoint did not reload exact neural parameter bytes")
    costs["reload_seconds"]+=time.perf_counter()-reload_start
    reload_metrics,_=evaluate(restored,datasets,device,recorder,costs)
    for task in datasets:
        for split in datasets[task]:
            expected,actual=curves[-1]["metrics"][str(task)][split],reload_metrics[task][split]
            if (expected["graph_native_acc"]!=actual["graph_native_acc"]
                    or expected["backend_native_acc"]!=actual["backend_native_acc"]
                    or expected["soft_acc"]!=actual["soft_acc"]):
                raise RuntimeError("saved/reloaded actual model/backend results changed")
    recorder.emit("actual_checkpoint_reload_verified",arm=arm,file=path.name,
        exact_neural_parameter_bytes=True,parameter_layouts=reload_layouts,
        all15_split_soft_native_accuracies_unchanged=True,reload_metrics=reload_metrics,
        reloaded_backend=backend,reloaded_binding_guard=guard,
        reload_geometry_resources=restored.geometry_resource_report())
    result=dict(arm=arm,execution_backend=backend,binding_guard=guard,
        actual_additional_updates=completed,source_parameter_hash=source_hash,final_parameter_hash=end_hash,
        raw_proposal_hashes=proposal_hashes,raw_old_violation_sequence=violations,
        curves=curves,costs=costs,final_geometry_resources=model.geometry_resource_report(),
        final_checkpoint=path.name,actual_reload_verified=True,
        model_wall_seconds=time.perf_counter()-arm_began,
        parameters=sum(p.numel() for p in model.parameters()),
        latency_scope="aggregate matched batched evaluation, not single-query p50/p95 or optimal cached direct execution")
    recorder.emit("arm_end",**result)
    return plain(result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--datasets",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",choices=("cpu","cuda:0"),default="cuda:0")
    parser.add_argument("--max-seconds",type=float,default=850.)
    args=parser.parse_args()
    if (not 0<args.max_seconds<=850 or args.out.exists() or not args.checkpoint.is_file()
            or args.checkpoint.stat().st_size>MAX_FILE_BYTES):
        parser.error("trusted source, NEW output and internal<=850s/external<=900s required")
    recorder=Recorder(args.out,args.max_seconds)
    completed=[]
    try:
        device=torch.device(args.device)
        torch.set_num_threads(1)
        if device.type=="cuda":
            if not torch.cuda.is_available():raise RuntimeError("authorized GPU unavailable; no silent CPU fallback")
            torch.cuda.set_per_process_memory_fraction(.3,device)
        source=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        if (source.get("schema")!="terminal-architecture-checkpoint-v1" or source.get("model_name")!="topo"
                or source.get("stage")!=2 or source.get("step")!=800
                or tuple(source["model"]["encoder"].shape)!=(16,32)
                or source["args"].get("scale")!=16 or "path_omega" not in source["model"]
                or source.get("observed_operation_formation")):
            raise ValueError("actual N16 V8 pre-history-formation stage2-step800 source required")
        names=(Path(__file__).name,"continuous_native_model.py","continuous_monodromy.py",
               "topology_model.py","semantic_promotion_model.py","integrate_native_repair.py",
               "train_monodromy_extension.py","diagnose_projection_numeric.py")
        provenance=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=sha256(args.checkpoint),
            source_sha256={name:sha256(Path(__file__).with_name(name)) for name in names},
            source_model_historical_geometry_report=source.get("model_extra",{}).get("continuous_monodromy",{}).get("resource_report"))
        datasets,dataset_sources=load_datasets(args.datasets,recorder)
        indices=np.random.default_rng(SEED).integers(0,len(datasets[2]["train"]["starts"]),size=(STEPS,BATCH),dtype=np.int64)
        np.savez_compressed(recorder.out/"common-training-indices.npz",indices=indices,seed=np.asarray(SEED))
        recorder.emit("configuration",MAIN="TOPOLOGY_USER_LOCKED",M0="NOT_PASSED",M3="ultimate",
            primary_baseline="Transformer",provenance=provenance,arm_specs=ARM_SPECS,
            steps_per_arm=STEPS,batch=BATCH,seed=SEED,lr=LR,
            objective="terminal_CE+.01geometry",router_MI_used=False,
            history_or_observed_operation_formation_used=False,
            common_indices_sha256=sha256(recorder.out/"common-training-indices.npz"),
            information="original terminal task2 TRAIN only in gradients; all15 splits evaluation only",
            scope="2x2 execution-backend x old-binding-guard, NOT all-topology-removed ablation",
            actual_shared_shear_family_present_in_all_arms=True,
            known_semantic_relation="direct graph maps equal continuous monodromy rho(compiled learned permutation)",
            diagnostic_old_graph_checks_charged_separately_in_ALLarms=True,
            direct_backend_is_explicit_base_graph_implementation_not_best_possible_cached_program=True)
        for backend,guard in ARM_SPECS:
            recorder.check()
            result=run_arm(backend,guard,source,datasets,indices,device,recorder,provenance)
            completed.append(result)
        if len({r["source_parameter_hash"] for r in completed})!=1:
            raise RuntimeError("factorial arms did not start from identical source parameter bytes")
        paired=[]
        for guard in (True,False):
            m=next(r for r in completed if r["execution_backend"]=="monodromy" and r["binding_guard"]==guard)
            d=next(r for r in completed if r["execution_backend"]=="direct" and r["binding_guard"]==guard)
            paired.append(dict(binding_guard=guard,
                equal_raw_proposal_hashes=[a==b for a,b in zip(m["raw_proposal_hashes"],d["raw_proposal_hashes"])],
                differing_raw_proposal_steps=[i+1 for i,(a,b) in enumerate(zip(m["raw_proposal_hashes"],d["raw_proposal_hashes"])) if a!=b],
                final_parameter_bytes_identical=m["final_parameter_hash"]==d["final_parameter_hash"],
                final_task_iid_backend_accuracy={"monodromy":[m["curves"][-1]["metrics"][str(t)]["iid"]["backend_native_acc"] for t in range(3)],
                    "direct":[d["curves"][-1]["metrics"][str(t)]["iid"]["backend_native_acc"] for t in range(3)]}))
        summary=dict(status="FOUR_ARM_MONODROMY_EXECUTION_BINDING_CONTRIBUTION_COMPLETED",
            MAIN="TOPOLOGY_USER_LOCKED",M0="NOT_PASSED",M3="ultimate",primary_baseline="Transformer",
            provenance=provenance,dataset_sources=dataset_sources,completed=completed,
            paired_backend_checks=paired,wall_seconds=time.perf_counter()-recorder.started,
            evidence_boundary="one source checkpoint/seed; backend control bypasses path execution only; same protected shear family and4omega in ALLarms; no M0/novelty/topology-advantage assumption")
        recorder.json("summary.json",summary)
        print(json.dumps(plain(dict(status=summary["status"],output=args.out,
            arms=[dict(arm=r["arm"],costs=r["costs"],model_wall_seconds=r["model_wall_seconds"]) for r in completed],
            paired_backend_checks=paired)),sort_keys=True))
    except BaseException as error:
        recorder.emit("failure",error_type=type(error).__name__,error=str(error),completed_arms=[r["arm"] for r in completed])
        recorder.json("failure.json",dict(status="FAILED_WITH_PARTIAL_RECORDS",error_type=type(error).__name__,error=str(error),
            completed=completed,wall_seconds=time.perf_counter()-recorder.started))
        raise
    finally:
        recorder.stream.close()


if __name__=="__main__":
    main()
