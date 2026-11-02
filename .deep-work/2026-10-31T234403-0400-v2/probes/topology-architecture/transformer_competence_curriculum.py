"""Constructive task0 repair of the ACTUAL shared attention Transformer.

One disclosed optimization policy: advance one input length at a time, keep a
permanent shorter-example anchor, and admit the next length only from full
TRAIN plus VALID terminal-label competence. No teacher intermediate state,
transition table, inferred operation compiler or architecture substitution.
The actual GrowingTaskTransformer attention/FFN/position/head computation is
unchanged. Extra optimization, repeated labels, selection screens and cost are
charged; this is not an equal600-step or topology-advantage experiment.

An optional OWN original stage0 checkpoint can preserve the already learned
short-word prefix instead of rerunning it. TRAIN/VALID alone select the final
candidate. Test files are opened only AFTER a concrete candidate is frozen,
and only when --evaluate-tests is explicitly supplied.

Use existing external CUDA isolation and timeout. No experiment at import.
MAIN=TOPOLOGY human-locked; this is bounded TODO5 baseline development.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from diagnose_projection_numeric import restore_parameter_layouts
from integrate_native_repair import Recorder, plain, sha256, synchronize, tensor_bytes
from parameter_transformer_baselines import GrowingTaskTransformer


KEYS=("tokens","starts","task_ids","targets")
MAX_FILE=100*1024*1024


def batch(data,indices,device):
    return {k:torch.as_tensor(v[indices],device=device,dtype=torch.long) for k,v in data.items()}


def read_data(directory,split,n):
    path=directory/f"task0-{split}.npz"
    if path.stat().st_size>20*1024*1024:raise ValueError("raw dataset exceeds20MiB")
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=set(KEYS):raise ValueError("only four original raw arrays accepted")
        data={k:np.asarray(archive[k],dtype=np.int64) for k in KEYS}
    count=len(data["starts"])
    if (not 64<=count<=32768 or data["tokens"].shape!=(count,18)
            or any(data[k].shape!=(count,) for k in KEYS[1:])
            or np.any(data["task_ids"]!=0)
            or np.any((data["tokens"]<0)|(data["tokens"]>6))
            or np.any((data["starts"]<0)|(data["starts"]>=n))
            or np.any((data["targets"]<0)|(data["targets"]>=n))):
        raise ValueError("actual raw task0 schema/domain differs")
    lengths=np.count_nonzero(data["tokens"],axis=1)
    if not np.array_equal(data["tokens"]!=0,np.arange(18)[None,:]<lengths[:,None]):
        raise ValueError("original right-PAD input required")
    if split in ("train","valid") and np.any(lengths>8):
        raise ValueError("selection datasets must use original <=8 domain")
    if split!="train" and count!=512:raise ValueError("original512-row evaluation splits required")
    digest=hashlib.sha256()
    for key in KEYS:digest.update(key.encode()+data[key].astype("<i8").tobytes())
    return data,lengths,dict(file=str(path),split=split,rows=count,file_sha256=sha256(path),
        four_array_sha256=digest.hexdigest(),array_bytes=sum(x.nbytes for x in data.values()),
        length_counts={str(k):int((lengths==k).sum()) for k in range(19) if (lengths==k).any()})


@torch.no_grad()
def measure(model,data,lengths,device,rec):
    synchronize(device);began,cpu=time.perf_counter(),time.process_time()
    was=model.training;model.eval();correct=np.empty(len(lengths),dtype=np.bool_)
    nll=np.empty(len(lengths),dtype=np.float64)
    try:
        for begin in range(0,len(lengths),64):
            rec.check();end=min(begin+64,len(lengths));b=batch(data,np.arange(begin,end),device)
            logits=model(b["tokens"],b["starts"],b["task_ids"])
            if not bool(torch.isfinite(logits).all()):raise FloatingPointError("nonfinite actual measured logits")
            correct[begin:end]=(logits.argmax(-1)==b["targets"]).cpu().numpy()
            nll[begin:end]=F.cross_entropy(logits,b["targets"],reduction="none").cpu().double().numpy()
    finally:
        synchronize(device);model.train(was)
    return dict(rows=len(lengths),accuracy=float(correct.mean()),terminal_CE=float(nll.mean()),
        by_length={str(k):dict(rows=int((lengths==k).sum()),accuracy=float(correct[lengths==k].mean()),
                    terminal_CE=float(nll[lengths==k].mean())) for k in np.unique(lengths)},
        wall_seconds=time.perf_counter()-began,cpu_process_seconds=time.process_time()-cpu)


def gate(train,valid,cap):
    train_groups=[train["by_length"][str(k)] for k in range(cap+1)]
    valid_groups=[valid["by_length"][str(k)] for k in range(cap+1) if str(k) in valid["by_length"]]
    # Empty programs have TRAIN coverage even though VALID draws length>=1.
    train_min=min(x["accuracy"] for x in train_groups)
    train_max_CE=max(x["terminal_CE"] for x in train_groups)
    valid_min=min(x["accuracy"] for x in valid_groups)
    passed=train_min>=.99 and train_max_CE<=.10 and valid_min>=.95
    return dict(passed=passed,cap=cap,min_TRAIN_length_accuracy=train_min,
        max_TRAIN_length_CE=train_max_CE,min_VALID_length_accuracy=valid_min,
        all_shorter_lengths_remain_in_contract=True,VALID_empty_domain_not_sampled=True)


def make_indices(pools,cap,generator):
    # Exactly64 existing terminal examples. Half challenge the frontier; half
    # are balanced by PREVIOUS length, not sampled in proportion to huge bins.
    frontier=generator.choice(pools[cap],size=32,replace=True)
    shorter_lengths=generator.integers(0,cap,size=32)
    anchor=np.asarray([generator.choice(pools[int(k)]) for k in shorter_lengths],dtype=np.int64)
    indices=np.concatenate((frontier,anchor));generator.shuffle(indices)
    return indices


def state_hash(state):
    digest=hashlib.sha256()
    for name,value in state.items():
        item=value.detach().cpu().contiguous()
        digest.update(name.encode()+str(item.dtype).encode()+np.asarray(item.shape,dtype="<i8").tobytes()+item.numpy().tobytes())
    return digest.hexdigest()


def resource_state(model,optimizer,device):
    return dict(parameters=sum(p.numel() for p in model.parameters()),
        model_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
        buffer_tensor_bytes=sum(p.numel()*p.element_size() for p in model.buffers()),
        cuda_allocated_bytes=torch.cuda.memory_allocated(device),cuda_reserved_bytes=torch.cuda.memory_reserved(device),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        architecture=model.resource_counts(optimizer),
        boundary="logical tensor/actual allocator counters, not total physical energy or all CUDA context memory")


def owned_checkpoint(model,optimizer,args,rec,step,cap,source,selection,inputs,suffix):
    path=rec.out/f"transformer-step{step}-cap{cap}-{suffix}.pt"
    if path.exists():raise FileExistsError("new study artifacts must not overwrite existing checkpoints")
    payload=dict(schema="transformer-anchored-competence-v1",args=vars(args),model=model.state_dict(),
        optimizer=optimizer.state_dict(),step=step,cap=cap,source=source,selection=selection,inputs=inputs,
        torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state_all(),architecture_unchanged=True,
        no_teacher_or_intermediate_targets=True)
    began=time.perf_counter();torch.save(payload,path)
    if path.stat().st_size>MAX_FILE:raise RuntimeError("bounded100MiB checkpoint exceeded")
    receipt=dict(file=path.name,sha256=sha256(path),bytes=path.stat().st_size,seconds=time.perf_counter()-began)
    rec.emit("actual_checkpoint",step=step,cap=cap,suffix=suffix,**receipt)
    return path,receipt


def create_model(args,device):
    model=GrowingTaskTransformer(args.scale,vocab=6,n_tasks=6,max_len=18,d_model=args.width).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003 if args.lr is None else args.lr)
    source=dict(kind="fresh_same_Transformer_architecture",prefix_gradient_updates=0,
                prefix_training_episodes=0,prefix_wall_seconds=None)
    if args.checkpoint:
        if args.checkpoint.stat().st_size>MAX_FILE:raise ValueError("owned checkpoint exceeds100MiB")
        payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        old=payload.get("args",{})
        if (payload.get("schema")!="parameter-growth-neural-checkpoint-v1" or payload.get("stage")!=0
                or payload.get("step") not in (100,300,600)
                or old.get("arm") not in ("transformer64","transformer96","replay64","replay96")
                or old.get("scale")!=args.scale or (96 if "96" in old["arm"] else 64)!=args.width):
            raise ValueError("OWN actual task0 Transformer64/96 checkpoint at100/300/600 required")
        model.load_state_dict(payload["model"],strict=True)
        layout=restore_parameter_layouts(model,payload["model"])
        if state_hash(payload["model"])!=state_hash(model.state_dict()):raise RuntimeError("source parameter bits changed on restoration")
        optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        optimizer.load_state_dict(payload["optimizer"])
        if args.lr is not None:
            for group in optimizer.param_groups:group["lr"]=args.lr
        torch.set_rng_state(payload["torch_rng"]);np.random.set_state(payload["numpy_rng"])
        random.setstate(payload["python_rng"]);torch.cuda.set_rng_state_all(payload["cuda_rng"])
        source=dict(kind="actual_existing_prefix_preserved",file=str(args.checkpoint),sha256=sha256(args.checkpoint),
            old_args=old,prefix_gradient_updates=payload["step"],prefix_training_episodes=payload["step"]*64,
            prefix_wall_seconds=None,prefix_wall_boundary="not reconstructed from absent per-update timing; original run receipt remains authoritative",
            parameter_layouts=layout,restored_state_sha256=state_hash(model.state_dict()),
            original_optimizer_state_restored=True,lr_override=args.lr)
    model.set_stage(0)
    return model,optimizer,source


def reload_candidate(path,args,device):
    payload=torch.load(path,map_location="cpu",weights_only=False)
    if payload.get("schema")!="transformer-anchored-competence-v1":raise ValueError("own newly frozen candidate required")
    model=GrowingTaskTransformer(args.scale,vocab=6,n_tasks=6,max_len=18,d_model=args.width).to(device)
    model.load_state_dict(payload["model"],strict=True)
    restore_parameter_layouts(model,payload["model"])
    if state_hash(model.state_dict())!=state_hash(payload["model"]):raise RuntimeError("frozen candidate changed model bits on actual reload")
    return model,payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--scale",type=int,choices=(8,16,32),default=8)
    p.add_argument("--width",type=int,choices=(64,96),default=64)
    p.add_argument("--checkpoint",type=Path)
    p.add_argument("--seed",type=int,default=0);p.add_argument("--lr",type=float)
    p.add_argument("--max-updates",type=int,default=6000)
    p.add_argument("--per-length-budget",type=int,default=1200)
    p.add_argument("--gate-every",type=int,default=100)
    p.add_argument("--evaluate-tests",action="store_true")
    p.add_argument("--max-seconds",type=float,default=850)
    p.add_argument("--device",choices=("cuda:0",),default="cuda:0")
    p.add_argument("--resource-overlap-note",default="not asserted isolated; coordinator records actual concurrent GPU workloads")
    args=p.parse_args()
    if (args.out.exists() or not 100<=args.max_updates<=12000 or not 100<=args.per_length_budget<=3000
            or not 25<=args.gate_every<=200 or not 0<args.max_seconds<=850
            or args.lr is not None and not 0<args.lr<=.01):
        p.error("new output, finite update/length/time bounds, valid optional learning rate required")
    rec=Recorder(args.out,args.max_seconds);index_stream=None;model=optimizer=None;step=0;cap=1
    inputs=[];source={};best=None;last_safe=False;completed_updates=0
    try:
        if not torch.cuda.is_available():raise RuntimeError("authorized GPU required; no fallback")
        device=torch.device(args.device);torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,device)
        torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
        selection_data={s:read_data(args.datasets,s,args.scale) for s in ("train","valid")}
        inputs=[selection_data[s][2] for s in ("train","valid")]
        data,lengths,_=selection_data["train"]
        pools={k:np.flatnonzero(lengths==k) for k in range(9)}
        if any(len(v)==0 for v in pools.values()):raise ValueError("all original TRAIN length bins0..8 must exist")
        model,optimizer,source=create_model(args,device)
        originalP=sum(x.numel() for x in model.parameters())
        source_code={name:sha256(Path(__file__).resolve().parent/name) for name in
                     (Path(__file__).name,"parameter_transformer_baselines.py","torch_baselines.py")}
        rec.emit("configuration",args=vars(args),MAIN="TOPOLOGY_USER_LOCKED",contribution="TODO5 strong Transformer baseline",
            inputs=inputs,source=source,source_code_sha256=source_code,
            architecture="unchanged GrowingTaskTransformer,3 attention/FFN blocks,4heads,learned absolute positions",
            initial_parameters=originalP,teacher_tables_or_intermediate_targets=False,
            policy="32 frontier samples +32 balanced-by-shorter-length anchor samples; advance1→2→...→8",
            competence=dict(min_each_TRAIN_length_accuracy=.99,max_each_TRAIN_length_CE=.10,
                            min_each_available_VALID_length_accuracy=.95),
            model_selection="prefer full0..8 TRAIN_VALID contract, then highest whole VALID accuracy, then lower VALID CE; exact tie retains earlier checkpoint",
            gradient_budget_different_from_original600=True,max_added_training_episodes=args.max_updates*64,
            raw_label_information_unchanged=True,old_labels_reused_with_cost=True,
            test_files_opened_before_selection=False)
        generator=np.random.default_rng(np.random.SeedSequence([args.seed,args.scale,0,39027]))
        index_path=rec.out/"actual-training-indices.i64"
        index_stream=index_path.open("xb")
        costs=dict(training_wall_seconds=0.,training_cpu_seconds=0.,cuda_stream_elapsed_seconds=0.,
                   measurement_wall_seconds=0.,measurement_cpu_seconds=0.,measurement_examples=0,
                   actual_new_training_episodes=0,validation_screens=0)
        cap_started=0;stop_reason=None;curves=[];graduated=False
        used_training_rows=np.zeros(len(lengths),dtype=np.bool_)
        for step in range(args.max_updates+1):
            if rec.deadline-time.perf_counter()<15.:
                stop_reason="bounded_wall_budget_censored_before_next_action";break
            rec.check()
            if step%args.gate_every==0 or step==args.max_updates:
                measured={}
                for split,(part,lens,_) in selection_data.items():
                    measured[split]=measure(model,part,lens,device,rec)
                    costs["measurement_wall_seconds"]+=measured[split]["wall_seconds"]
                    costs["measurement_cpu_seconds"]+=measured[split]["cpu_process_seconds"]
                    costs["measurement_examples"]+=measured[split]["rows"]
                costs["validation_screens"]+=1
                decision=gate(measured["train"],measured["valid"],cap)
                full_contract=gate(measured["train"],measured["valid"],8)
                row=dict(step=step,total_updates_including_prefix=source["prefix_gradient_updates"]+step,
                    cap=cap,measurements=measured,competence=decision,full_length_competence=full_contract,
                    resources=resource_state(model,optimizer,device),
                    cumulative_cost=dict(costs))
                curves.append(row);rec.emit("TRAIN_VALID_curve",**row)
                score=(int(full_contract["passed"]),measured["valid"]["accuracy"],-measured["valid"]["terminal_CE"])
                if best is None or score>tuple(best["score"]):
                    path,receipt=owned_checkpoint(model,optimizer,args,rec,step,cap,source,row,inputs,"best-valid")
                    best=dict(score=list(score),file=path.name,sha256=receipt["sha256"],step=step,cap=cap,
                              TRAIN_VALID=row)
                if full_contract["passed"]:
                    graduated=True;stop_reason="all_original_TRAIN_and_VALID_length_contracts_passed";break
                if decision["passed"]:
                    rec.emit("length_promoted",step=step,previous=cap,new=cap+1,
                             gate_used_only_TRAIN_VALID=True,all_previous_lengths_remain_anchored=True)
                    cap+=1;cap_started=step
                elif step-cap_started>=args.per_length_budget:
                    stop_reason="specific_frontier_exhausted_prospective_update_budget";break
            if step==args.max_updates:
                stop_reason="total_prospective_update_budget_exhausted";break
            indices=make_indices(pools,cap,generator);b=batch(data,indices,device)
            synchronize(device);began,cpu=time.perf_counter(),time.process_time()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();last_safe=False
            model.train();optimizer.zero_grad(set_to_none=True)
            logits=model(b["tokens"],b["starts"],b["task_ids"]);loss=F.cross_entropy(logits,b["targets"])
            if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite actual anchored training loss")
            loss.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(gradient)):raise FloatingPointError("nonfinite actual anchored gradient")
            optimizer.step();end.record();synchronize(device);last_safe=True
            completed_updates+=1;used_training_rows[indices]=True
            elapsed,cpu_elapsed,stream_elapsed=time.perf_counter()-began,time.process_time()-cpu,start.elapsed_time(end)/1000.
            costs["training_wall_seconds"]+=elapsed;costs["training_cpu_seconds"]+=cpu_elapsed
            costs["cuda_stream_elapsed_seconds"]+=stream_elapsed;costs["actual_new_training_episodes"]+=64
            index_stream.write(indices.astype("<i8").tobytes());index_stream.flush()
            if sum(x.numel() for x in model.parameters())!=originalP:raise RuntimeError("repair changed architecture parameter count")
            rec.emit("actual_update",step=step+1,cap=cap,terminal_CE=float(loss.detach()),
                actual_batch_accuracy=float((logits.argmax(-1)==b["targets"]).float().mean()),
                preclip_gradient=float(gradient),lr=optimizer.param_groups[0]["lr"],
                length_counts={str(k):int((lengths[indices]==k).sum()) for k in np.unique(lengths[indices])},
                batch_index_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                training_wall_seconds=elapsed,training_cpu_seconds=cpu_elapsed,cuda_stream_elapsed_seconds=stream_elapsed)
        index_stream.close();index_stream=None
        if best is None:raise RuntimeError("bounded run ended before any valid candidate was measured")
        # Last current weights and best VALID weights are separate immutable
        # artifacts. No test observation chooses either one.
        owned_checkpoint(model,optimizer,args,rec,completed_updates,cap,source,dict(stop_reason=stop_reason),inputs,"last-current")
        rec.json("selected-candidate.json",dict(best=best,stop_reason=stop_reason,
            all_length_contracts_passed=graduated,selection_frozen_before_test_read=True,
            actual_index_file=index_path.name,actual_index_file_sha256=sha256(index_path),
            completed_added_updates=costs["actual_new_training_episodes"]//64,
            index_layout="little-endian int64, rows=actual_completed_added_updates, columns64",
            costs=costs))
        selected,payload=reload_candidate(rec.out/best["file"],args,device)
        selected.eval()
        tests={};test_cost=0.
        if args.evaluate_tests and rec.deadline-time.perf_counter()>15.:
            rec.emit("test_boundary_opened",selected_file=best["file"],selected_sha256=best["sha256"],
                     no_further_training_or_selection_after_this_event=True)
            for split in ("iid","ood","length16"):
                part,lens,receipt=read_data(args.datasets,split,args.scale)
                rec.emit("post_selection_test_dataset_read",**receipt)
                tests[split]=measure(selected,part,lens,device,rec);test_cost+=tests[split]["wall_seconds"]
        rec.json("summary.json",dict(status="BOUNDED_TRANSFORMER_DEVELOPMENT_CHECKPOINT",stop_reason=stop_reason,
            graduated_TRAIN_VALID=graduated,best=best,tests=tests,costs=costs,test_wall_seconds=test_cost,
            actual_added_updates=costs["actual_new_training_episodes"]//64,
            total_updates_including_existing_prefix=source["prefix_gradient_updates"]+costs["actual_new_training_episodes"]//64,
            actual_unique_TRAIN_labels_available=len(lengths),
            distinct_TRAIN_rows_used_in_added_updates=int(used_training_rows.sum()),
            existing_prefix_distinct_row_coverage_not_reconstructed=True,new_external_labels=0,
            parameters=originalP,architecture_unchanged=True,curves=curves,
            selected_real_checkpoint_reload_bits_verified=True,
            inference="one constructive optimizer/curriculum development; does not isolate anchoring from extra budget or prove family superiority",
            next_scope="if acquisition succeeds, carry the SAME policy into continual training; otherwise inspect the first failed measured frontier"))
    except BaseException as error:
        rec.emit("run_failed",error=type(error).__name__,reason=str(error),loop_step=step,cap=cap,
                 actual_completed_added_updates=completed_updates,last_update_completed=last_safe)
        if last_safe and model is not None and optimizer is not None:
            try:owned_checkpoint(model,optimizer,args,rec,completed_updates,cap,source,dict(error=str(error)),inputs,"interrupted")
            except BaseException as save_error:rec.emit("failed_to_save_interrupted_state",reason=str(save_error))
        rec.json("failure.json",dict(error=type(error).__name__,reason=str(error),loop_step=step,cap=cap,
            best_measured_candidate=best,last_update_transaction_completed=last_safe))
        raise
    finally:
        if index_stream is not None:index_stream.close()
        rec.stream.close()


if __name__=="__main__":main()
