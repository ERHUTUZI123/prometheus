"""Matched gradient-budget continuation of the actual strong N16 baselines.

Load their real stage1-step800 checkpoints. Add200 task1 updates, then train
task2 on the SAVED original800-step schedule plus200 full-distribution updates.
Never retrain stage0 or the source stage1. No teacher operations, intermediate
targets, geometry, network or installation. Fresh Adam LR=.003 at task1-extra
and task2 entry; task2's optimizer continues across its800/200 boundary.

Root must inspect/freeze and run in the existing bounded CUDA sandbox. This
module is not a sandbox. Trusted owned checkpoints only (weights_only=False).
MAIN=TOPOLOGY USER_LOCKED; Transformer primary; M0 NOT PASSED; M3 ultimate.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from tasks import ARRAY_KEYS, data_sha256
from torch_baselines import parameter_counts
from train import Recorder, make_model, NEW_TOKENS, tensor_bytes, fetch, plain


MODELS = ("transformer", "replay", "adapter")
SPLITS = ("train", "valid", "iid", "ood", "length16")
MAX_BYTES = 100 * 1024 * 1024
BATCH, LR, EXTRA = 64, .003, 200


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync(device):
    torch.cuda.synchronize(device)


def state_sha(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode()+b"\0")
        digest.update(str(value.dtype).encode()+b"\0")
        digest.update(json.dumps(list(value.shape)).encode()+b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def device_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: device_tree(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [device_tree(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(device_tree(v, device) for v in value)
    return copy.deepcopy(value)


def restore(name, path, device, recorder, stage=1, step=800):
    recorder.check()
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("trusted owned checkpoint absent or over100MiB")
    began = time.perf_counter()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (checkpoint.get("schema") != "terminal-architecture-checkpoint-v1"
            or checkpoint.get("model_name") != name
            or checkpoint.get("stage") != stage or checkpoint.get("step") != step):
        raise ValueError("checkpoint must match the declared actual arm/stage/step")
    saved = checkpoint["args"]
    if (int(saved["scale"]) != 16 or int(saved["seed"]) != 0
            or int(saved["steps"]) != 800 or float(saved["lr"]) != LR):
        raise ValueError("need actual N16 seed0 LR.003 source with original800-step metadata")
    saved_args = argparse.Namespace(**saved)
    model = make_model(name, saved_args).to(device)
    masks = checkpoint.get("masks_and_initialized_tasks", {})
    if "_initialized_tasks" in masks:
        model._initialized_tasks = copy.deepcopy(masks["_initialized_tasks"])
    model.set_stage(stage, new_token_ids=NEW_TOKENS[stage])
    model.load_state_dict(checkpoint["model"], strict=True)
    layouts = []
    with torch.no_grad():
        for key, parameter in model.named_parameters():
            source = checkpoint["model"][key]
            changed = tuple(parameter.stride()) != tuple(source.stride())
            if changed:
                value = torch.empty_strided(source.shape, source.stride(),
                                            dtype=source.dtype, device=device)
                value.copy_(source.to(device))
                parameter.data = value  # preserve Parameter identity and row-mask hooks
            layouts.append(dict(name=key, shape=list(source.shape),
                                source_stride=list(source.stride()),
                                restored_stride=list(parameter.stride()),
                                layout_replaced=changed,
                                exact_bits=bool(torch.equal(source, parameter.detach().cpu()))))
    loader = getattr(model, "load_extra_state_dict", None)
    extra = checkpoint.get("model_extra", {})
    if callable(loader):
        loader(extra)
    elif extra:
        for key, value in extra.items():
            setattr(model, key, copy.deepcopy(value))
    for key, value in masks.items():
        setattr(model, key, device_tree(value, device))
    if state_sha(model.state_dict()) != state_sha(checkpoint["model"]):
        raise RuntimeError("restoration changed actual learned parameters/buffers")
    if not all(x["exact_bits"] and x["source_stride"] == x["restored_stride"] for x in layouts):
        raise RuntimeError("source parameter layout/bits were not restored")
    torch.set_rng_state(checkpoint["torch_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])
    if checkpoint.get("cuda_rng"):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    model.train()
    sync(device)
    recorder.emit("actual_checkpoint_restored", model=name, source=str(path),
                  sha256=sha(path), bytes=path.stat().st_size, stage=stage, step=step,
                  parameters=parameter_counts(model), source_parameter_layouts=layouts,
                  masks_tensor_bytes=tensor_bytes(masks),
                  source_saved_optimizer_tensor_bytes=tensor_bytes(checkpoint.get("optimizer", {})),
                  source_optimizer_not_resumed=True, fresh_Adam_authorized=True,
                  original_stage0_and800_stage1_updates_not_retrained=True,
                  restore_seconds=time.perf_counter()-began)
    return model, checkpoint


def load_data(folder, recorder):
    datasets, sources = [], []
    for task in range(3):
        splits = {}
        for split in SPLITS:
            recorder.check()
            path = folder/f"task{task}-{split}.npz"
            if not path.is_file() or path.stat().st_size > MAX_BYTES:
                raise ValueError("saved original dataset absent or oversized")
            with np.load(path, allow_pickle=False) as archive:
                arrays = {k: np.asarray(archive[k], dtype=np.int64) for k in ARRAY_KEYS}
            count = 4096 if split == "train" else 512
            if (arrays["tokens"].shape != (count,18)
                    or any(arrays[k].shape != (count,) for k in ARRAY_KEYS if k != "tokens")
                    or np.any(arrays["task_ids"] != task)
                    or np.any(arrays["tokens"] < 0) or np.any(arrays["tokens"] > (3,5,6)[task])
                    or np.any(arrays["starts"] < 0) or np.any(arrays["starts"] >= 16)
                    or np.any(arrays["targets"] < 0) or np.any(arrays["targets"] >= 16)):
                raise ValueError("original N16 dataset shape/task/alphabet mismatch")
            splits[split] = {k: torch.from_numpy(v).long() for k,v in arrays.items()}
            sources.append(dict(task=task, split=split, path=str(path), file_sha256=sha(path),
                                learner_array_sha256=data_sha256(arrays),
                                array_bytes=sum(v.nbytes for v in arrays.values())))
        datasets.append(splits)
    recorder.emit("original_datasets", sources=sources,
                  full_array_bytes=sum(x["array_bytes"] for x in sources),
                  feedback="original final-output labels only; eval never chooses updates")
    return datasets, sources


def schedules(path, recorder):
    if not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("original task2 minibatch-index file absent or oversized")
    with np.load(path, allow_pickle=False) as archive:
        core = {"current": np.asarray(archive["current"], dtype=np.int64),
                "uniform_old": np.asarray(archive["uniform_old"], dtype=np.int64)}
    if (core["current"].shape != (800,64) or core["uniform_old"].shape != (800,32)
            or np.any(core["current"] < 0) or np.any(core["current"] >= 4096)
            or np.any(core["uniform_old"] < 0) or np.any(core["uniform_old"] >= 8192)):
        raise ValueError("must use exact original800-step task2 current/old index arrays")
    result = {"task2_original800":core}
    for task, seed in ((1,9003),(2,9004)):
        current = np.random.default_rng(seed).integers(0,4096,size=(200,64),dtype=np.int64)
        old = np.random.default_rng(seed+1_000_000).integers(0,4096*task,size=(200,32),dtype=np.int64)
        result[f"task{task}_extra200"] = dict(current=current, uniform_old=old)
    for name, arrays in result.items():
        target=recorder.out/(name+"-indices.npz")
        np.savez_compressed(target, **arrays)
        recorder.emit("fixed_matched_schedule", phase=name, file=target.name,
                      source_path=str(path) if name=="task2_original800" else None,
                      source_file_sha256=sha(path) if name=="task2_original800" else None,
                      current_sha256=hashlib.sha256(arrays["current"].tobytes()).hexdigest(),
                      old_sha256=hashlib.sha256(arrays["uniform_old"].tobytes()).hexdigest(),
                      current_seed=9003 if name=="task1_extra200" else 9004 if name=="task2_extra200" else None,
                      replay_policy="first32 current indices +32 uniform old; same total64")
    return result


@torch.no_grad()
def evaluate(model, datasets, device, recorder):
    was_training=model.training
    model.eval()
    sync(device)
    began=time.perf_counter()
    result={}
    for task, splits in enumerate(datasets):
        result[task]={}
        for split, data in splits.items():
            recorder.check()
            count, correct, loss, predictions=0,0,0.,[]
            for start in range(0,len(data["starts"]),BATCH):
                b=fetch(data,np.arange(start,min(start+BATCH,len(data["starts"]))),device)
                logits=model(b["tokens"],b["starts"],b["task_ids"])
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("nonfinite actual baseline evaluation")
                prediction=logits.argmax(-1)
                correct+=int((prediction==b["targets"]).sum().item())
                loss+=float(F.cross_entropy(logits,b["targets"],reduction="sum").item())
                predictions.append(prediction.cpu().numpy())
                count+=len(b["starts"])
            packed=np.asarray(np.concatenate(predictions),dtype="<i8")
            result[task][split]=dict(n=count,accuracy=correct/count,CE=loss/count,
                prediction_sha256=hashlib.sha256(packed.tobytes()).hexdigest())
    sync(device)
    seconds=time.perf_counter()-began
    model.train(was_training)
    return result,seconds


def save(model,optimizer,original,recorder,name,stage,step,phase):
    sync(next(model.parameters()).device)
    began=time.perf_counter()
    path=recorder.out/f"{name}-stage{stage}-step{step}-{phase}.pt"
    payload=dict(schema="terminal-architecture-checkpoint-v1",args=copy.deepcopy(original["args"]),
        model_name=name,stage=stage,step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
        model_extra={},masks_and_initialized_tasks={k:getattr(model,k) for k in
            ("_row_masks","_row_snapshots","_initialized_tasks") if hasattr(model,k)},
        torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),
        cuda_rng=torch.cuda.get_rng_state_all(),M0="NOT_PASSED",
        matched_continuation=dict(original_stage=1,original_step=800,
            task1_additional_updates=200,task2_total_updates=step if stage==2 else 0,
            original_args_steps800_preserved=True,extra_gradient_costs_explicit=True))
    temporary=path.with_name(path.name+".tmp")
    torch.save(payload,temporary)
    if temporary.stat().st_size>MAX_BYTES:
        raise RuntimeError("checkpoint exceeds100MiB")
    temporary.replace(path)
    seconds=time.perf_counter()-began
    recorder.emit("checkpoint",model=name,stage=stage,step=step,phase=phase,
                  file=path.name,bytes=path.stat().st_size,sha256=sha(path),seconds=seconds)
    return path,seconds


def train_phase(model,name,stage,steps,indices,datasets,device,recorder,optimizer,
                original,curves_at,phase,step_offset=0):
    model.train()
    replay=None
    if name=="replay":
        replay={k:torch.cat([datasets[t]["train"][k] for t in range(stage)],0) for k in ARRAY_KEYS}
    sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    started=time.perf_counter()
    costs=dict(actual_gradient_updates=0,train_current_episodes=0,train_old_episodes=0,
               train_seconds=0.,eval_seconds=0.,batch_fetch_seconds=0.,
               frozen_row_maintenance_seconds=0.,checkpoint_seconds=0.)
    losses=[];curves=[];checkpoints=[]
    recorder.emit("phase_begin",model=name,stage=stage,phase=phase,steps=steps,
                  metadata_step_offset=step_offset,parameters=parameter_counts(model),
                  replay_memory_tensor_bytes=tensor_bytes(replay),
                  source_current_rows=4096,source_old_rows=0 if replay is None else len(replay["starts"]))
    for update in range(steps+1):
        recorder.check()
        if update in curves_at:
            metrics,elapsed=evaluate(model,datasets,device,recorder)
            costs["eval_seconds"]+=elapsed
            row=dict(model=name,stage=stage,phase=phase,step=step_offset+update,
                phase_updates_completed=update,metrics=metrics,
                training_CE_since_last_curve=None if not losses else float(np.mean(losses)),
                costs=copy.deepcopy(costs),parameters=parameter_counts(model),
                parameter_tensor_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
                replay_memory_tensor_bytes=tensor_bytes(replay),
                frozen_masks_and_snapshots_tensor_bytes=tensor_bytes(dict(
                    masks=getattr(model,"_row_masks",{}),snapshots=getattr(model,"_row_snapshots",{}))),
                representation_state_bytes=0,protection_guard_seconds=0.,
                gpu_peak_bytes=torch.cuda.max_memory_allocated(device))
            recorder.emit("curve",**row);curves.append(row);losses.clear()
            if update==steps:
                path,elapsed=save(model,optimizer,original,recorder,name,stage,step_offset+update,phase)
                costs["checkpoint_seconds"]+=elapsed;checkpoints.append(str(path))
        if update==steps:
            break
        began=time.perf_counter()
        current_indices=indices["current"][update]
        if replay is None:
            b=fetch(datasets[stage]["train"],current_indices,device)
            costs["train_current_episodes"]+=BATCH
        else:
            current=fetch(datasets[stage]["train"],current_indices[:32],device)
            old=fetch(replay,indices["uniform_old"][update],device)
            b={k:torch.cat((current[k],old[k]),0) for k in ARRAY_KEYS}
            costs["train_current_episodes"]+=32;costs["train_old_episodes"]+=32
        sync(device);costs["batch_fetch_seconds"]+=time.perf_counter()-began
        began=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits=model(b["tokens"],b["starts"],b["task_ids"])
        loss=F.cross_entropy(logits,b["targets"])
        if not torch.isfinite(loss):raise FloatingPointError("nonfinite terminal training CE")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError("nonfinite gradient norm")
        optimizer.step()
        if getattr(model,"_row_masks",{}):
            sync(device);maintenance_started=time.perf_counter()
            model.enforce_frozen_rows_();sync(device)
            costs["frozen_row_maintenance_seconds"]+=time.perf_counter()-maintenance_started
        sync(device);costs["train_seconds"]+=time.perf_counter()-began
        costs["actual_gradient_updates"]+=1
        losses.append(float(loss.item()))
        recorder.emit("training_update",model=name,stage=stage,phase=phase,
                      step=step_offset+update+1,terminal_CE=float(loss.item()),
                      preclip_gradient_norm=float(norm.item()))
    costs["phase_wall_seconds"]=time.perf_counter()-started
    report=dict(model=name,stage=stage,phase=phase,costs=costs,curves=curves,
                final_metrics=curves[-1]["metrics"],parameters=parameter_counts(model),
                optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
                replay_memory_tensor_bytes=tensor_bytes(replay),
                checkpoints=checkpoints,
                timing_scope="train includes row maintenance; full phase wall includes fetch/eval/checkpoint/report; subsets overlap")
    recorder.emit("phase_end",**report)
    return report


def arguments():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in MODELS:parser.add_argument("--"+name+"-checkpoint",type=Path,required=True)
    parser.add_argument("--datasets",type=Path,required=True)
    parser.add_argument("--stage2-indices",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",choices=("cuda:0",),default="cuda:0")
    parser.add_argument("--seconds",type=float,default=850.)
    args=parser.parse_args()
    if not 0<args.seconds<=850 or args.out.exists():
        parser.error("new output and<=850s internal /<=900s external bound required")
    return args


def main():
    args=arguments();recorder=Recorder(args.out,args.seconds)
    completed=[]
    try:
        if not torch.cuda.is_available():raise RuntimeError("CUDA required; no silentCPU fallback")
        device=torch.device(args.device);torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.3,device)
        recorder.emit("configuration",args=vars(args),MAIN="TOPOLOGY_USER_LOCKED",M0="NOT_PASSED",M3="ultimate",
            primary_baseline="Transformer",terminal_only=True,actual_updates_per_arm=1200,
            actual_task1_extra=200,actual_task2_original_schedule=800,actual_task2_extra=200,
            batch64=True,LR=LR,extra_current_seeds=[9003,9004],extra_old_seed_offset=1000000,
            precision_flags_changed=False,torch_float32_matmul_precision=torch.get_float32_matmul_precision(),
            observed_cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            observed_cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            original_precision_flags_not_serialized="inherited existing host defaults; not guessed",
            source_sha256=sha(Path(__file__)))
        datasets,data_sources=load_data(args.datasets,recorder)
        plans=schedules(args.stage2_indices,recorder)
        for name in MODELS:
            recorder.check();model_started=time.perf_counter()
            model,original=restore(name,getattr(args,name+"_checkpoint"),device,recorder)
            optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=LR)
            first=train_phase(model,name,1,200,plans["task1_extra200"],datasets,device,recorder,
                optimizer,original,{0,100,200},"task1-extra200",step_offset=800)
            model.set_stage(2,new_token_ids=NEW_TOKENS[2])
            optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=LR)
            second=train_phase(model,name,2,800,plans["task2_original800"],datasets,device,recorder,
                optimizer,original,{0,200,400,800},"task2-original800")
            third=train_phase(model,name,2,200,plans["task2_extra200"],datasets,device,recorder,
                optimizer,original,{200},"task2-extra200",step_offset=800)
            final_path=Path(third["checkpoints"][-1])
            final_state_hash=state_sha(model.state_dict())
            del optimizer,model;gc.collect();torch.cuda.empty_cache()
            reloaded,_=restore(name,final_path,device,recorder,stage=2,step=1000)
            reloaded_metrics,reload_seconds=evaluate(reloaded,datasets,device,recorder)
            comparisons={str(t):{split:reloaded_metrics[t][split]["prediction_sha256"]==
                third["final_metrics"][t][split]["prediction_sha256"] for split in SPLITS} for t in range(3)}
            exact_state=state_sha(reloaded.state_dict())==final_state_hash
            if not exact_state or not all(all(x.values()) for x in comparisons.values()):
                raise RuntimeError("actual final checkpoint reload changed learned state or predictions")
            result=dict(model=name,status="MATCHED_CONTINUATION_AND_RELOAD_COMPLETED",
                phases=[first,second,third],actual_additional_updates=1200,
                final_metrics=reloaded_metrics,final_checkpoint=str(final_path),
                actual_reload_predictions_equal=comparisons,actual_reload_state_exact=exact_state,
                reload_eval_seconds=reload_seconds,
                model_wall_seconds=time.perf_counter()-model_started,
                comparison_boundary="same data/total gradient budget; topology formation/search/installation remain additional separately charged resources",
                M0="NOT_PASSED")
            recorder.emit("model_complete",**result);completed.append(result)
            recorder.summary(dict(status="PARTIAL_MODELS_COMPLETE",completed=completed,M0="NOT_PASSED"))
            del reloaded,original;gc.collect();torch.cuda.empty_cache()
        recorder.summary(dict(status="ALL_MATCHED_BASELINE_CONTINUATIONS_COMPLETED",completed=completed,
            source_datasets=data_sources,M0="NOT_PASSED",primary_baseline="Transformer",
            no_stage0_or_source800_stage1_retraining=True,
            equal_total_updates_does_not_equal_equal_total_compute=True))
    except BaseException as error:
        recorder.emit("run_failed",error=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
        recorder.summary(dict(status="FAILED_WITH_PARTIAL_RECORDS",completed=completed,
                             error=type(error).__name__,reason=str(error),M0="NOT_PASSED"))
        raise


if __name__=="__main__":main()
