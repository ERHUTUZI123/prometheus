"""Bounded raw-terminal DEN-RNN / linear self-modifying-CMS study.

Run only inside the already authorized external GPU isolation/timeout runner.
This script is not a sandbox and never opens a network connection. It does not
import or execute author-downloaded code. Models are disclosed reconstructions,
NOT official reproductions of DEN or full HOPE. No experiment runs at import.

All600 updates/task count, including DEN phase fitting. No old-data replay is
silently donated. The CMS optimizer/clock/pending buffers persist across tasks.
MAIN=TOPOLOGY human-locked. This work serves TODO5 (strong baselines).
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from integrate_native_repair import Recorder, plain, sha256, tensor_bytes, synchronize
from parameter_den_baseline import DENSequenceBaseline, den_model_from_checkpoint
from parameter_hope_baseline import (SelfModCMSReconstruction, make_hope_reconstruction,
                                     make_optimizer as make_cms_optimizer)


SPLITS = ("train", "valid", "iid", "ood", "length16")
KEYS = ("tokens", "starts", "task_ids", "targets")
MAX_FILE = 100 * 1024 * 1024
LABELS = dict(den="DEN-RNN terminal-sequence adaptation (non-official)",
              selfmod_cms="Self-modifying linear memory + CMS reconstruction (not full/official HOPE)")


def array_hash(data):
    # Exactly the current parameter_efficiency_study's common-array hash.
    h = hashlib.sha256()
    for key in KEYS:
        h.update(key.encode()+np.asarray(data[key], dtype="<i8").tobytes())
    return h.hexdigest()


def parameters(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        bytes=sum(p.numel()*p.element_size() for p in model.parameters()))


def non_tensor_metadata(value):
    if torch.is_tensor(value):
        return dict(tensor_shape=list(value.shape), dtype=str(value.dtype), tensor_bytes=tensor_bytes(value))
    if isinstance(value, np.ndarray):
        return dict(array_shape=list(value.shape), dtype=str(value.dtype), array_bytes=value.nbytes)
    if isinstance(value, dict):
        return {str(k):non_tensor_metadata(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [non_tensor_metadata(v) for v in value]
    return value


def resources(model, optimizer, device):
    extra = model.extra_state_dict()
    detail = model.resource_counts(optimizer)
    metadata = json.dumps(plain(non_tensor_metadata(extra)), sort_keys=True, allow_nan=False).encode()
    result = dict(parameters=parameters(model), model_specific=detail,
        model_extra_tensor_bytes=tensor_bytes(extra),
        model_extra_metadata_skeleton_json_bytes=len(metadata),
        optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
        registered_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
        replay_bytes=0, training_replay_episodes=0,
        cuda_allocated_bytes=torch.cuda.memory_allocated(device),
        cuda_reserved_bytes=torch.cuda.memory_reserved(device),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        byte_boundary="logical tensor bytes; full serialized checkpoint bytes reported separately; no claim of hardware energy",
        storage_fields_overlap="model_specific echoes top-level optimizer/buffer counts; extra tensor bytes price serialization payload including snapshots/masks already itemized; do not add duplicate views of the same cost")
    if hasattr(optimizer,"resource_report"):
        result["nested_optimizer"] = optimizer.resource_report()
    return result


def rng_state():
    return dict(torch_cpu=torch.get_rng_state(), numpy=np.random.get_state(),
                python=random.getstate(), torch_cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    torch.set_rng_state(state["torch_cpu"])
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def exact_state_equal(left, right):
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and left.dtype==right.dtype and left.shape==right.shape and torch.equal(left.detach().cpu(),right.detach().cpu())
    if isinstance(left, dict) or isinstance(right, dict):
        return isinstance(left,dict) and isinstance(right,dict) and left.keys()==right.keys() and all(exact_state_equal(left[k],right[k]) for k in left)
    if isinstance(left, (list,tuple)) or isinstance(right,(list,tuple)):
        return type(left)==type(right) and len(left)==len(right) and all(exact_state_equal(a,b) for a,b in zip(left,right))
    return left==right


def model_and_optimizer(args, device):
    if args.arm=="den":
        model = DENSequenceBaseline(args.scale, vocab=6, n_tasks=6, hidden=args.width,
            max_hidden=args.den_max_width, ex_k=args.den_ex_k, l1=args.den_l1, l2=args.den_l2,
            group_lasso=args.den_group_lasso, drift_regularization=args.den_drift,
            loss_threshold=args.den_loss_threshold, split_threshold=args.den_split_threshold,
            max_splits=args.den_max_splits, parameter_budget=args.den_parameter_budget).to(device)
        # set_stage creates the first real head before constructing Adam.
        return model, None
    model = make_hope_reconstruction(args.scale, width=args.width, n_tasks=6,
        n_layers=args.hope_layers, cms_periods=args.cms_periods).to(device)
    return model, make_cms_optimizer(model, lr=args.lr, cms_lr=args.cms_lr)


def new_den_optimizer(model, args):
    return torch.optim.Adam(model.parameters(), lr=args.lr)


def restore_checkpoint(payload, device):
    arm = payload["arm"]
    config = payload["optimizer_configuration"]
    if arm=="den":
        model = den_model_from_checkpoint(payload,device)
        optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])
    elif arm=="selfmod_cms":
        constructor = copy.deepcopy(payload["model_extra"]["constructor"])
        constructor["n_tasks"] = payload["model"]["tasks.weight"].shape[0]
        model = SelfModCMSReconstruction(**constructor).to(device)
        model.load_state_dict(payload["model"],strict=True)
        optimizer = make_cms_optimizer(model,lr=config["lr"],cms_lr=config["cms_lr"])
    else:
        raise ValueError("unknown owned baseline checkpoint")
    optimizer.load_state_dict(payload["optimizer"])
    for name,p in model.named_parameters():
        if name not in payload["requires_grad"]:
            raise ValueError("saved Parameter flags incomplete")
        p.requires_grad_(payload["requires_grad"][name])
    return model,optimizer


def load_data(args, count):
    datasets,receipt = [],[]
    for task in range(count):
        splits = {}
        for split in SPLITS:
            path = args.datasets/f"task{task}-{split}.npz"
            if path.stat().st_size>20*1024*1024:
                raise ValueError("single raw dataset exceeds20MiB bound")
            with np.load(path,allow_pickle=False) as archive:
                if set(archive.files)!=set(KEYS):
                    raise ValueError("only original four raw arrays accepted; no teacher metadata")
                data = {k:np.asarray(archive[k],dtype=np.int64) for k in KEYS}
            rows = len(data["starts"])
            if not 64<=rows<=32768 or data["tokens"].shape!=(rows,18):
                raise ValueError("bounded original paddedL18 arrays required")
            if any(data[k].shape!=(rows,) for k in KEYS[1:]):
                raise ValueError("raw target/start/task shape mismatch")
            if (np.any((data["starts"]<0)|(data["starts"]>=args.scale))
                    or np.any((data["targets"]<0)|(data["targets"]>=args.scale))
                    or np.any(data["task_ids"]!=task)
                    or np.any((data["tokens"]<0)|(data["tokens"]>6))):
                raise ValueError("raw public values outside benchmark domain")
            if np.any(np.maximum.accumulate(data["tokens"]==0,axis=1)&(data["tokens"]!=0)):
                raise ValueError("PAD must be a right suffix")
            if split!="train" and rows!=512:
                raise ValueError("original512-row evaluation splits required")
            splits[split] = data
            receipt.append(dict(task=task,split=split,rows=rows,file=str(path),
                file_sha256=sha256(path),array_sha256=array_hash(data),array_bytes=sum(a.nbytes for a in data.values())))
        datasets.append(splits)
    return datasets,receipt


def batch(data, indices, device):
    return {k:torch.as_tensor(v[indices],dtype=torch.long,device=device) for k,v in data.items()}


def common_indices(data, args, stage, steps=600):
    # Same seed, draw shape and optional override order as the primary runner.
    generator = np.random.default_rng(np.random.SeedSequence([args.seed,args.scale,stage,9027]))
    indices = generator.integers(0,len(data["starts"]),size=(steps,64),dtype=np.int64)
    if args.curriculum:
        lengths = np.count_nonzero(data["tokens"],axis=1)
        for step in range(min(300,steps)):
            pool = np.flatnonzero(lengths<=(1 if step<150 else 3))
            if not len(pool):
                raise ValueError("prospective short-length training pool empty")
            indices[step] = generator.choice(pool,size=64,replace=True)
    return indices


@torch.no_grad()
def evaluate(model, datasets, seen, device, rec):
    synchronize(device)
    began,cpu = time.perf_counter(),time.process_time()
    was = model.training
    model.eval()
    result = {}
    try:
        for task in range(seen+1):
            result[task] = {}
            for split in SPLITS[1:]:
                data = datasets[task][split]
                correct,loss = 0,0.
                for offset in range(0,len(data["starts"]),64):
                    rec.check()
                    b = batch(data,np.arange(offset,min(offset+64,len(data["starts"]))),device)
                    logits = model(b["tokens"],b["starts"],b["task_ids"])
                    if not bool(torch.isfinite(logits).all()):
                        raise FloatingPointError("nonfinite actual baseline evaluation logits")
                    correct += int((logits.argmax(-1)==b["targets"]).sum())
                    loss += float(F.cross_entropy(logits,b["targets"],reduction="sum"))
                n = len(data["starts"])
                result[task][split] = dict(neural=correct/n,deployed=correct/n,terminal_CE=loss/n,n=n,
                    interface="same neural execution; no separate compiled/native interface")
    finally:
        synchronize(device)
        model.train(was)
    return result,dict(wall_seconds=time.perf_counter()-began,cpu_process_seconds=time.process_time()-cpu,
                       episodes=sum(len(datasets[t][s]["starts"]) for t in range(seen+1) for s in SPLITS[1:]))


def successful(metrics):
    return {int(t):all(m[s]["neural"]>=threshold for s,threshold in
            (("iid",.95),("ood",.90),("length16",.90))) for t,m in metrics.items()}


def capabilities(metrics, stage, admitted, stable, previous_parameters, P):
    if metrics[stage]["valid"]["neural"]>=.95:
        admitted.add(stage)
    old_stable = set(stable)
    passed = successful(metrics)
    stable = {t for t in stable if passed[t]}
    if stage in admitted and passed[stage]:
        stable.add(stage)
    lost = sorted(old_stable-stable)
    delta = len(stable)-len(old_stable)
    return stable,dict(admitted=sorted(admitted),stable=sorted(stable),Qstable=len(stable),
        currently_successful=[t for t,ok in passed.items() if ok],
        lost_previously_stable_capabilities=lost,net_delta_stable_capabilities=delta,
        new_successfully_retained=int(stage in stable and not lost),
        delta_parameters=P-previous_parameters,
        deltaP_per_new_retained_capability=(P-previous_parameters)/delta if delta>0 and not lost else None)


def checkpoint(model, optimizer, args, rec, stage, step, complete, caps, inputs,
               reports, suffix="", costs=None):
    name = f"{args.arm}-stage{stage}-step{step}{suffix}.pt"
    path = rec.out/name
    if path.exists():
        raise FileExistsError("checkpoint artifacts are append-only")
    began,cpu = time.perf_counter(),time.process_time()
    payload = dict(schema="parameter-external-baseline-neural-v1",arm=args.arm,
        label=LABELS[args.arm],args=vars(args),stage=int(stage),step=int(step),
        stage_training_complete=bool(complete),model=model.state_dict(),
        model_extra=model.extra_state_dict(),optimizer=optimizer.state_dict(),
        optimizer_configuration=dict(lr=args.lr,cms_lr=args.cms_lr),
        requires_grad={n:p.requires_grad for n,p in model.named_parameters()},
        capabilities=caps,input_receipt=inputs,reports=reports,rng=rng_state(),
        source_hashes=source_hashes(),costs=costs,
        current_route="TODO5: strong baselines; MAIN=TOPOLOGY_USER_LOCKED",
        research_completion_or_topology_advantage_claim=False)
    pending = path.with_suffix(".pt.partial")
    torch.save(payload,pending)
    if pending.stat().st_size>MAX_FILE:
        raise RuntimeError("bounded100MiB actual checkpoint exceeded; partial retained")
    pending.rename(path)
    timing = dict(wall_seconds=time.perf_counter()-began,cpu_process_seconds=time.process_time()-cpu)
    rec.emit("checkpoint",file=name,sha256=sha256(path),bytes=path.stat().st_size,
             stage=stage,step=step,training_complete=complete,timing=timing)
    return path,timing


def actual_reload(path, model, optimizer, datasets, seen, device, rec):
    # Reloading consumes neutral factory RNG only; preserve the training RNG
    # explicitly so next task's genuine initialization is not changed by audit.
    saved_rng = rng_state()
    was = model.training
    synchronize(device)
    began,cpu = time.perf_counter(),time.process_time()
    restored = restored_optimizer = None
    try:
        payload = torch.load(path,map_location="cpu",weights_only=False)
        if payload.get("schema")!="parameter-external-baseline-neural-v1":
            raise ValueError("only own external-baseline checkpoint is accepted")
        restored,restored_optimizer = restore_checkpoint(payload,device)
        if not exact_state_equal(payload["model"],restored.state_dict()):
            raise RuntimeError("real checkpoint roundtrip changed model state")
        if not exact_state_equal(payload["optimizer"],restored_optimizer.state_dict()):
            raise RuntimeError("real checkpoint roundtrip changed nested/ordinary optimizer state")
        model.eval(); restored.eval()
        max_difference = 0.
        with torch.no_grad():
            for task in range(seen+1):
                rec.check()
                b = batch(datasets[task]["valid"],np.arange(64),device)
                left = model(b["tokens"],b["starts"],b["task_ids"])
                right = restored(b["tokens"],b["starts"],b["task_ids"])
                max_difference = max(max_difference,float((left-right).abs().max()))
                if not torch.equal(left,right):
                    raise RuntimeError("real checkpoint roundtrip changed actual neural logits")
        synchronize(device)
        report = dict(verified=True,all_model_state_equal=True,all_optimizer_state_equal=True,
            checked_valid_rows_per_seen_task=64,max_logit_absolute_difference=max_difference,
            wall_seconds=time.perf_counter()-began,cpu_process_seconds=time.process_time()-cpu,
            restored_resources=resources(restored,restored_optimizer,device),
            original_training_rng_restored_after_neutral_factory_audit=True)
        rec.emit("actual_neural_checkpoint_reload_verified",stage=seen,**report)
        return report
    finally:
        model.train(was)
        del restored,restored_optimizer
        restore_rng(saved_rng)


def source_hashes():
    root = Path(__file__).resolve().parent
    names = (Path(__file__).name,"parameter_den_baseline.py","parameter_hope_baseline.py",
             "integrate_native_repair.py")
    return {name:sha256(root/name) for name in names}


def train_update(model, optimizer, b, args, local_step, device, smoke_audit=False):
    synchronize(device)
    began,cpu = time.perf_counter(),time.process_time()
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    optimizer.zero_grad(set_to_none=True)
    logits = model(b["tokens"],b["starts"],b["task_ids"])
    ce = F.cross_entropy(logits,b["targets"])
    regularizer = model.regularization_loss() if args.arm=="den" else ce.new_zeros(())
    objective = ce+regularizer
    if not bool(torch.isfinite(objective)):
        raise FloatingPointError("nonfinite actual terminal objective")
    objective.backward()
    smoke = {}
    if args.arm=="den":
        model.mask_gradients_()  # MUST precede clipping; inactive Adam values restored below.
        if smoke_audit:
            smoke["inactive_gradient_max"] = max([float(p.grad[~model._masks[n]].abs().max())
                for n,p in model.named_parameters() if p.grad is not None and bool((~model._masks[n]).any())] or [0.])
            if smoke["inactive_gradient_max"]!=0.:
                raise RuntimeError("DEN inactive parameter received an unmasked gradient")
            inactive_before = {n:(p.detach().clone(),model._masks[n].clone())
                for n,p in model.named_parameters() if not bool(model._masks[n].all())}
    elif smoke_audit:
        smoke["initial_fast_memory_gradient_norms"] = [
            layer.initial_memories.grad.detach().flatten(1).norm(dim=1).cpu().tolist()
            if layer.initial_memories.grad is not None else [0.]*5 for layer in model.layers]
    gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not bool(torch.isfinite(gradient)):
        raise FloatingPointError("nonfinite actual terminal gradient")
    optimizer.step()
    end.record(); synchronize(device)
    timing = dict(training_wall_seconds=time.perf_counter()-began,
        training_cpu_process_seconds=time.process_time()-cpu,
        cuda_stream_elapsed_seconds=start.elapsed_time(end)/1000.,
        cuda_event_scope="stream elapsed includes host issue gaps; not a hardware GPU-busy or energy measurement")
    event = None
    maintenance_wall = maintenance_cpu = 0.
    if args.arm=="den":
        began,cpu = time.perf_counter(),time.process_time()
        event = model.after_optimizer_step_(float(ce.detach()),local_step)
        if smoke_audit and "split" not in event and "expansion" not in event:
            current = dict(model.named_parameters())
            for name,(value,mask) in inactive_before.items():
                if name not in current or current[name].shape!=value.shape or not torch.equal(current[name][~mask],value[~mask]):
                    raise RuntimeError("DEN inactive Adam values were not actually restored")
            smoke["inactive_value_restore_verified"] = True
        if event["optimizer_rebuild_required"]:
            before_bytes = tensor_bytes(optimizer.state_dict())
            optimizer = new_den_optimizer(model,args)
            event["discarded_optimizer_tensor_bytes"] = before_bytes
            event["new_optimizer_tensor_bytes"] = tensor_bytes(optimizer.state_dict())
            event["reset_matches_declared_DEN_phase_policy"] = True
        synchronize(device)
        maintenance_wall,maintenance_cpu = time.perf_counter()-began,time.process_time()-cpu
    timing.update(maintenance_wall_seconds=maintenance_wall,maintenance_cpu_process_seconds=maintenance_cpu)
    return optimizer,dict(terminal_CE=float(ce.detach()),regularization=float(regularizer.detach()),
        objective=float(objective.detach()),preclip_gradient=float(gradient),
        batch_accuracy=float((logits.argmax(-1)==b["targets"]).float().mean()),
        den_event=event,timing=timing,smoke=smoke)


def run_smoke(model, optimizer, args, datasets, inputs, device, rec):
    reports = []
    stages = 2 if args.arm=="den" else 1
    steps = 20 if args.arm=="den" else max(16,max(args.cms_periods))
    fast_nonzero = [[False]*5 for _ in range(args.hope_layers)]
    for stage in range(stages):
        if args.arm=="den":
            model.set_stage(stage,total_steps=steps)
            optimizer = new_den_optimizer(model,args)
        else:
            model.set_stage(stage)
        indices = common_indices(datasets[stage]["train"],args,stage,600)
        for step in range(1,steps+1):
            rec.check()
            b = batch(datasets[stage]["train"],indices[step-1,:args.smoke_batch],device)
            optimizer,row = train_update(model,optimizer,b,args,step,device,smoke_audit=True)
            for i,norms in enumerate(row["smoke"].get("initial_fast_memory_gradient_norms",[])):
                fast_nonzero[i] = [old or x>0. for old,x in zip(fast_nonzero[i],norms)]
            rec.emit("smoke_update",stage=stage,step=step,**row)
            if args.arm=="selfmod_cms":
                expected = [step//p for p in args.cms_periods]
                if optimizer.level_updates!=expected or optimizer.pending!=[step%p for p in args.cms_periods]:
                    raise RuntimeError("actual CMS clock/pending buffers do not match update frequencies")
                if step==3:
                    path,_ = checkpoint(model,optimizer,args,rec,stage,step,False,{},inputs,[],suffix="-smoke-pending")
                    actual_reload(path,model,optimizer,datasets,stage,device,rec)
        if args.arm=="den":
            if model.phase!="complete" or model.timestamps[stage]!=model.hidden:
                raise RuntimeError("DEN stage/timestamp did not actually complete")
            if stage==1 and model.counters["neurons_added"]==0:
                raise RuntimeError("DEN smoke did not exercise expansion; record this failure before training")
            if stage==1:
                # A concrete learned-state timestamp check, not teacher labels.
                width = model.timestamps[0]
                b = batch(datasets[0]["train"],indices[0,:args.smoke_batch],device)
                b["task_ids"].zero_()
                with torch.no_grad():
                    h=F.relu(F.embedding(b["starts"],model.start_weight[:width].T)+model.hidden_bias[:width])
                    for j in range(b["tokens"].shape[1]):
                        a=b["tokens"][:,j]
                        h2=F.relu(F.linear(h,model.recurrent_weight[:width,:width],model.hidden_bias[:width])
                                  +F.embedding(a,model.input_weight[:width].T))
                        h=torch.where(a[:,None]!=0,h2,h)
                    manual=F.linear(h,model.heads[0].weight[:,:width],model.heads[0].bias)
                    if not torch.equal(model(b["tokens"],b["starts"],b["task_ids"]),manual):
                        raise RuntimeError("DEN old-task timestamp execution did not crop both recurrent axes")
                rec.emit("smoke_DEN_timestamp_crop_verified",old_width=width,current_width=model.hidden,
                         comparison_used_only_own_learned_weights=True)
        path,_ = checkpoint(model,optimizer,args,rec,stage,steps,True,{},inputs,[],suffix="-smoke")
        reload = actual_reload(path,model,optimizer,datasets,stage,device,rec)
        reports.append(dict(stage=stage,steps=steps,resources=resources(model,optimizer,device),reload=reload))
    if args.arm=="selfmod_cms" and not all(all(x) for x in fast_nonzero):
        raise RuntimeError("some actual self-modifying initial memory had no nonzero gradient in smoke")
    rec.json("smoke-summary.json",dict(status="ACTUAL_SMOKE_PASSED",arm=args.arm,label=LABELS[args.arm],
        reports=reports,all_five_fast_memories_receive_gradients=fast_nonzero if args.arm=="selfmod_cms" else None,
        benchmark_training_or_competitive_result=False))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--arm",choices=tuple(LABELS),required=True)
    p.add_argument("--scale",type=int,choices=(8,16,32),required=True)
    p.add_argument("--stages",type=int,default=6)
    p.add_argument("--seed",type=int,default=0);p.add_argument("--width",type=int,default=64)
    p.add_argument("--lr",type=float);p.add_argument("--cms-lr",type=float)
    p.add_argument("--hope-layers",type=int,choices=(1,2,3),default=2)
    p.add_argument("--cms-periods",default="1,4,16")
    p.add_argument("--den-max-width",type=int,default=256)
    p.add_argument("--den-ex-k",type=int,default=10);p.add_argument("--den-max-splits",type=int,default=10)
    p.add_argument("--den-parameter-budget",type=int,default=250000)
    p.add_argument("--den-l1",type=float,default=1e-5);p.add_argument("--den-l2",type=float,default=1e-4)
    p.add_argument("--den-group-lasso",type=float,default=1e-3)
    p.add_argument("--den-drift",type=float,default=.5)
    p.add_argument("--den-loss-threshold",type=float,default=.01)
    p.add_argument("--den-split-threshold",type=float,default=.05)
    p.add_argument("--curriculum",action="store_true")
    p.add_argument("--resume",type=Path)
    p.add_argument("--smoke",action="store_true");p.add_argument("--smoke-batch",type=int,default=8)
    p.add_argument("--max-seconds",type=float,default=850)
    p.add_argument("--device",choices=("cuda:0",),default="cuda:0")
    p.add_argument("--resource-overlap-note",default="not asserted isolated; coordinator must record actual concurrent GPU workloads")
    return p


def main():
    p=parser();args=p.parse_args()
    try:args.cms_periods=tuple(int(x) for x in args.cms_periods.split(","))
    except ValueError:p.error("CMS periods are comma-separated positive integers")
    if (args.out.exists() or not 1<=args.stages<=6 or not 0<args.max_seconds<=850
            or not 1<=args.smoke_batch<=64 or args.smoke and args.resume):
        p.error("new output,1..6 stages,<=850s, bounded smoke batch, no smoke resume required")
    if args.lr is None:args.lr=.001 if args.arm=="den" else .0003
    if not 0<args.lr<=.1:p.error("finite positive learning rate<=.1 required")
    rec=Recorder(args.out,args.max_seconds);device=torch.device(args.device)
    model=optimizer=None;reports=[];stage=-1;step=0;inputs=[];caps={};last_safe=False
    try:
        if not torch.cuda.is_available():raise RuntimeError("authorized CUDA required; no CPU fallback")
        torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,device)
        torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
        # Same immutable six-task raw stream as the primary runner. Future and
        # old labels are never passed to train_update: only current TRAIN is.
        # Keeping the full input receipt also prevents silent stream changes
        # when resuming a completed stage in another bounded process.
        datasets,inputs=load_data(args,6)
        model,optimizer=model_and_optimizer(args,device)
        admitted,stable,previousP,start_stage=set(),set(),0,0
        if args.resume:
            payload=torch.load(args.resume,map_location="cpu",weights_only=False)
            if (payload.get("schema")!="parameter-external-baseline-neural-v1" or payload["arm"]!=args.arm
                    or payload["args"]["scale"]!=args.scale or not payload["stage_training_complete"]
                    or payload["step"]!=600 or payload["args"].get("smoke",False)):
                raise ValueError("own same-arm/scale completed-stage checkpoint required; no old stage reruns")
            for key in ("seed","width","lr","cms_lr","hope_layers","cms_periods","curriculum",
                        "den_max_width","den_ex_k","den_max_splits","den_parameter_budget",
                        "den_l1","den_l2","den_group_lasso","den_drift","den_loss_threshold","den_split_threshold"):
                if payload["args"][key]!=getattr(args,key):raise ValueError("resume changes actual baseline configuration: "+key)
            for old in payload["input_receipt"]:
                now=next((x for x in inputs if (x["task"],x["split"])==(old["task"],old["split"])),None)
                if now is None or now["file_sha256"]!=old["file_sha256"]:raise ValueError("resume raw input differs")
            del model,optimizer
            model,optimizer=restore_checkpoint(payload,device)
            restore_rng(payload["rng"])
            start_stage=payload["stage"]+1
            admitted=set(payload["capabilities"]["admitted"]);stable=set(payload["capabilities"]["stable"])
            previousP=parameters(model)["total"];reports=payload["reports"]
            rec.emit("completed_stage_checkpoint_resumed",source=str(args.resume),sha256=sha256(args.resume),
                next_stage=start_stage,old_stages_not_retrained=True,nested_optimizer_preserved=args.arm=="selfmod_cms",
                source_hashes_at_checkpoint=payload["source_hashes"],source_hashes_now=source_hashes())
        rec.emit("configuration",args=vars(args),label=LABELS[args.arm],inputs=inputs,
            source_hashes=source_hashes(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
            GPU=torch.cuda.get_device_name(device),GPU_total_bytes=torch.cuda.get_device_properties(device).total_memory,
            MAIN="TOPOLOGY_USER_LOCKED",current_contribution="TODO5 strong baselines",primary_baseline="Transformer",
            feedback="only current raw TRAIN terminal labels; no replay, teacher table, hidden state or correct route",
            resident_common_raw_array_bytes=sum(x["array_bytes"] for x in inputs),
            total_steps_per_task=600,common_batch_seed_component=9027,common_batch_size=64,
            evaluation_thresholds=dict(iid=.95,ood=.90,length16=.90,admission_valid=.95),
            test_used_for_training_selection=False,hyperparameter_selection="CLI fixed before run; no automatic test-selected tuning",
            first_stage_failure_means_baseline_needs_diagnosis_not_topology_victory=True,
            stable_scope="consecutive stage-final evaluations after valid admission; intermediate curves retained",
            initial_allocation_fully_charged_in_first_deltaP=True)
        if args.smoke:
            run_smoke(model,optimizer,args,datasets,inputs,device,rec)
            return
        if start_stage>=args.stages:raise ValueError("resume requests no new stage; no duplicate execution")
        for stage in range(start_stage,args.stages):
            rec.check();phase_started,phase_cpu=time.perf_counter(),time.process_time()
            torch.cuda.reset_peak_memory_stats(device)
            stage_cost=dict(training_wall_seconds=0.,training_cpu_process_seconds=0.,cuda_stream_elapsed_seconds=0.,
                maintenance_wall_seconds=0.,maintenance_cpu_process_seconds=0.,batch_assembly_wall_seconds=0.,
                evaluation_wall_seconds=0.,evaluation_cpu_process_seconds=0.,evaluation_episodes=0,
                actual_current_training_episodes=0,actual_old_training_episodes=0)
            began,cpu=time.perf_counter(),time.process_time()
            if args.arm=="den":
                model.set_stage(stage,total_steps=600)
                optimizer=new_den_optimizer(model,args)
            else:
                model.set_stage(stage)
                if model.last_growth_report["optimizer_rebuild_required"]:
                    raise RuntimeError("six preallocated task codes should not reset the CMS optimizer")
            synchronize(device)
            stage_cost["maintenance_wall_seconds"]+=time.perf_counter()-began
            stage_cost["maintenance_cpu_process_seconds"]+=time.process_time()-cpu
            data=datasets[stage]["train"];indices=common_indices(data,args,stage)
            index_path=rec.out/f"stage{stage}-common-indices.npz"
            np.savez_compressed(index_path,indices=indices)
            rec.emit("stage_begin",stage=stage,parameters=parameters(model),indices_sha256=sha256(index_path),
                index_array_sha256=hashlib.sha256(indices.astype("<i8").tobytes()).hexdigest(),
                fixed_curriculum="0..149 length<=1;150..299 length<=3;300..599 allTRAIN" if args.curriculum else None,
                CMS_clock=optimizer.global_step if args.arm=="selfmod_cms" else None,
                CMS_optimizer_reinitialized=False if args.arm=="selfmod_cms" else None,
                DEN_phase=model.phase if args.arm=="den" else None)
            curve=[];last_safe=True
            for step in range(601):
                rec.check()
                if step in (0,100,300,600):
                    metrics,timing=evaluate(model,datasets,stage,device,rec)
                    stage_cost["evaluation_wall_seconds"]+=timing["wall_seconds"]
                    stage_cost["evaluation_cpu_process_seconds"]+=timing["cpu_process_seconds"]
                    stage_cost["evaluation_episodes"]+=timing["episodes"]
                    row=dict(stage=stage,step=step,metrics=metrics,resources=resources(model,optimizer,device),
                             cumulative_stage_cost=copy.deepcopy(stage_cost))
                    curve.append(row);rec.emit("curve",**row)
                    if step in (100,300):
                        checkpoint(model,optimizer,args,rec,stage,step,False,caps,inputs,reports,costs=stage_cost)
                if step==600:break
                began=time.perf_counter();b=batch(data,indices[step],device)
                stage_cost["batch_assembly_wall_seconds"]+=time.perf_counter()-began
                last_safe=False
                model.train()
                optimizer,row=train_update(model,optimizer,b,args,step+1,device)
                last_safe=True
                for key in ("training_wall_seconds","training_cpu_process_seconds","cuda_stream_elapsed_seconds",
                            "maintenance_wall_seconds","maintenance_cpu_process_seconds"):
                    stage_cost[key]+=row["timing"][key]
                stage_cost["actual_current_training_episodes"]+=len(b["starts"])
                rec.emit("actual_update",stage=stage,step=step+1,**row)
            metrics=curve[-1]["metrics"];P=parameters(model)["total"]
            stable,caps=capabilities(metrics,stage,admitted,stable,previousP,P)
            readiness=(metrics[0]["valid"]["neural"]>=.95 and successful(metrics)[0]) if stage==0 else reports[0]["first_task_acquisition_ready"]
            summary=dict(stage=stage,parameters=P,delta_parameters=P-previousP,capabilities=caps,
                first_task_acquisition_ready=readiness,
                first_task_interpretation="ready under frozen criterion" if readiness else "reconstruction requires diagnosis/tuning; this is not a strong-method defeat",
                resources=resources(model,optimizer,device),final_metrics=metrics,curves=curve,costs=stage_cost,
                phase_wall_seconds=time.perf_counter()-phase_started,phase_cpu_process_seconds=time.process_time()-phase_cpu)
            # Put the current summary into the actual checkpoint. Reload report
            # is recorded afterward, rather than pretending it already existed.
            path,save_timing=checkpoint(model,optimizer,args,rec,stage,600,True,caps,inputs,reports+[copy.deepcopy(summary)],costs=stage_cost)
            reload=actual_reload(path,model,optimizer,datasets,stage,device,rec)
            summary.update(checkpoint=path.name,checkpoint_bytes=path.stat().st_size,
                           checkpoint_save=save_timing,actual_reload=reload)
            summary["phase_wall_seconds"]=time.perf_counter()-phase_started
            summary["phase_cpu_process_seconds"]=time.process_time()-phase_cpu
            reports.append(summary);rec.emit("stage_complete",**summary)
            rec.json(f"completed-through-stage{stage}.json",dict(arm=args.arm,label=LABELS[args.arm],reports=reports,
                     baseline_not_official_replication=True,task_completion_not_campaign_completion=True))
            previousP=P;gc.collect();torch.cuda.empty_cache()
        rec.json("summary.json",dict(status="DECLARED_STAGES_COMPLETED",arm=args.arm,label=LABELS[args.arm],
            scale=args.scale,seed=args.seed,reports=reports,completed_new_stages=list(range(start_stage,args.stages)),
            initial_task_acquisition_ready=reports[0]["first_task_acquisition_ready"],
            claim_scope="actual execution of this disclosed reconstruction only; no full-HOPE/DEN ranking or topology advantage claim"))
    except BaseException as error:
        rec.emit("run_failed",error=type(error).__name__,reason=str(error),stage=stage,loop_step=step,
                 completed_stages=[r["stage"] for r in reports],last_update_transaction_completed=last_safe)
        if last_safe and model is not None and optimizer is not None and stage>=0:
            try:checkpoint(model,optimizer,args,rec,stage,step,False,caps,inputs,reports,suffix="-interrupted")
            except BaseException as save_error:rec.emit("partial_checkpoint_failed",error=type(save_error).__name__,reason=str(save_error))
        rec.json("failure.json",dict(error=type(error).__name__,reason=str(error),stage=stage,loop_step=step,
            completed_reports=reports,partial_checkpoint_is_not_completed_training=True))
        raise
    finally:
        rec.stream.close()


if __name__=="__main__":
    main()
