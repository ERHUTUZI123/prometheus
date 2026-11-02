"""Read-only frozen VALID memory interventions on an OWN phase2 checkpoint.

Loads ONLY exact original b8a51 ContinuousPrimitiveBlock source, never live core
or teacher/generator code. No optimizer, training, installation, network or
subprocess. Root supplies existing outside900/inside850/.3 CUDA isolation.
These interventions measure frozen VALID dependence, not acquisition causality
or test performance. Hook statistics/extra reads cost time and are disclosed.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch
from torch.nn import functional as F

CORE_SHA="b8a51a6ca9026b2ebff877493bf9031972a319d097cb3ae2605cc6ad3ea3083f"
INTERVENTIONS=("full","zero_read","uniform_read","joint_slot_roll1","value_slot_roll1","batch_history_roll1")
KEYS=("initial","tokens","lengths","targets")


def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):digest.update(chunk)
    return digest.hexdigest()


def state_hash(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        x=value.detach().cpu().contiguous()
        h.update(key.encode());h.update(str((tuple(x.shape),x.dtype)).encode());h.update(x.numpy().tobytes())
    return h.hexdigest()


def load_source(path):
    if sha(path)!=CORE_SHA:raise ValueError("only exact owned original b8a51 source may execute")
    name="prometheus_owned_original_continuous_for_read_audit"
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module.ContinuousPrimitiveBlock


def raw_hash(data):
    h=hashlib.sha256()
    for key in KEYS:
        a=np.ascontiguousarray(data[key],dtype="<i8")
        h.update(key.encode()+np.asarray(a.shape,dtype="<i8").tobytes()+a.tobytes())
    return h.hexdigest()


def read_valid(data_dir,metadata,payload,stage):
    entry=metadata["stages"][stage];file=entry.get("files",{}).get("valid",f"stage{stage}-valid.npz")
    if isinstance(file,dict):file=file.get("file",file.get("path"))
    if not isinstance(file,str):raise ValueError("public VALID path required")
    path=(data_dir/file).resolve()
    if not path.is_relative_to(data_dir.resolve()) or path.stat().st_size>20*1024*1024:
        raise ValueError("bounded own VALID input required")
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(KEYS):raise ValueError("only raw terminal observation arrays accepted")
        data={k:np.asarray(z[k]) for k in KEYS}
    if any(a.dtype!=np.int64 for a in data.values()):raise ValueError("original int64 observations required")
    b=len(data["initial"]);words=data["tokens"];lens=data["lengths"]
    n=metadata["n_states"];v=metadata["n_tokens"]
    if (b!=256 or words.ndim!=2 or words.shape[0]!=b or words.shape[1]>metadata["max_length"]
        or any(data[k].shape!=(b,) for k in ("initial","lengths","targets"))
        or np.any((lens<0)|(lens>words.shape[1])) or np.any((data["initial"]<0)|(data["initial"]>=n))
        or np.any((data["targets"]<0)|(data["targets"]>=n))):raise ValueError("fixed256-row VALID shape/range differs")
    active=np.arange(words.shape[1])[None]<lens[:,None]
    if np.any(active&((words<0)|(words>=v))):raise ValueError("active public token outside vocabulary")
    expected=[r for r in payload["input_receipts"] if r.get("stage")==stage and r.get("split")=="valid"]
    receipt=dict(stage=stage,split="valid",file=path.name,file_sha256=sha(path),array_sha256=raw_hash(data),rows=b)
    if len(expected)!=1 or any(expected[0].get(k)!=receipt[k] for k in ("file_sha256","array_sha256","rows")):
        raise ValueError("VALID data differs from actual checkpoint lineage")
    return data,receipt


class ReadIntervention:
    def __init__(self,original,mode,model):
        self.original,self.mode,self.model=original,mode,model
        self.calls=0;self.examples=0;self.original_read_invocations=0;self.extra_read_examples=0
        self.entropy=0.;self.baseline_entropy=0.;self.read_norm=0.;self.baseline_read_norm=0.
        self.value_variance=0.;self.read_difference_max=0.;self.rolled_scalars=0;self.cross_example_eligible=0

    def __call__(self,h,x,keys,values):
        baseline,attention=self.original(h,x,keys,values)
        self.original_read_invocations+=1
        if self.mode=="full":read,effective=baseline,attention
        elif self.mode=="zero_read":read,effective=torch.zeros_like(baseline),attention
        elif self.mode=="uniform_read":
            effective=torch.ones_like(attention)/attention.shape[1];read=values.mean(1)
        else:
            changed_keys=keys;changed_values=values
            if self.mode=="joint_slot_roll1":
                changed_keys=keys.roll(1,1);changed_values=values.roll(1,1)
                self.rolled_scalars+=keys.numel()+values.numel()
            elif self.mode=="value_slot_roll1":
                changed_values=values.roll(1,1);self.rolled_scalars+=values.numel()
            elif self.mode=="batch_history_roll1":
                # Own query/current state; another active example's complete
                # key/value memory at this read, not another example's label.
                changed_keys=keys.roll(1,0);changed_values=values.roll(1,0)
                self.rolled_scalars+=keys.numel()+values.numel()
                self.cross_example_eligible+=len(h) if len(h)>1 else 0
            else:raise ValueError("unknown fixed intervention")
            read,effective=self.original(h,x,changed_keys,changed_values)
            self.original_read_invocations+=1;self.extra_read_examples+=len(h)
        self.calls+=1;self.examples+=len(h)
        scale=math.log(attention.shape[1]) if attention.shape[1]>1 else 1.
        def entropy(a):return -(a.double()*a.double().clamp_min(torch.finfo(torch.float64).tiny).log()).sum(-1)/scale
        self.entropy+=float(entropy(effective).sum());self.baseline_entropy+=float(entropy(attention).sum())
        self.read_norm+=float(read.norm(dim=-1).sum());self.baseline_read_norm+=float(baseline.norm(dim=-1).sum())
        self.value_variance+=float(values.var(dim=1,unbiased=False).mean(-1).sum())
        self.read_difference_max=max(self.read_difference_max,float((read-baseline).abs().max()) if read.numel() else 0.)
        return read,effective

    def report(self):
        n=max(1,self.examples);m=self.model
        one_read=(m.internal_states+m.token_dim)*m.key_dim+m.memory_slots*(m.key_dim+m.internal_states)
        return dict(read_calls=self.calls,read_example_calls=self.examples,
            original_read_invocations=self.original_read_invocations,extra_original_read_example_calls=self.extra_read_examples,
            effective_attention_normalized_entropy=self.entropy/n,baseline_attention_normalized_entropy=self.baseline_entropy/n,
            effective_read_norm=self.read_norm/n,baseline_read_norm=self.baseline_read_norm/n,
            value_within_slot_variance=self.value_variance/n,maximum_local_read_difference=self.read_difference_max,
            rolled_tensor_scalars=self.rolled_scalars,cross_example_eligible_read_examples=self.cross_example_eligible,
            extra_original_read_MAC_proxy=self.extra_read_examples*one_read,
            uniform_value_mean_elements=self.examples*m.memory_slots*m.internal_states if self.mode=="uniform_read" else 0,
            instrumentation_scope="statistics/float64 entropy/norm/variance synchronizeGPU and cost actual time; not normal inference throughput")


def summarize(logp,pred,targets,n):
    confusion=np.zeros((n,n),dtype=np.int64);np.add.at(confusion,(targets,pred),1)
    counts=confusion.sum(1);correct=confusion.diagonal();present=counts>0
    recalls=[None if counts[i]==0 else float(correct[i]/counts[i]) for i in range(n)]
    return dict(accuracy=float((pred==targets).mean()),terminal_NLL=float((-logp[np.arange(len(targets)),targets]).mean()),
        class_counts=counts.tolist(),per_class_recall=recalls,
        balanced_accuracy_present_classes=float((correct[present]/counts[present]).mean()),
        classes_present=int(present.sum()),confusion_true_rows_predicted_columns=confusion.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--checkpoint-sha256",required=True)
    p.add_argument("--source",type=Path,required=True);p.add_argument("--data",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--device",choices=("cuda:0",),default="cuda:0")
    p.add_argument("--max-seconds",type=float,default=850.)
    args=p.parse_args()
    if args.out.exists() or not 0<args.max_seconds<=850 or args.checkpoint.stat().st_size>100*1024*1024:
        p.error("new output, owned <=100MiB checkpoint and bounded850s required")
    args.out.mkdir(parents=True);stream=(args.out/"records.jsonl").open("x");began=time.perf_counter();deadline=began+args.max_seconds
    results=[];model=None;before_hash=None
    def check():
        if time.perf_counter()>deadline:raise TimeoutError("finite memory audit reached its bound")
    def emit(kind,**fields):
        stream.write(json.dumps(dict(kind=kind,wall_seconds=time.perf_counter()-began,**fields),sort_keys=True,allow_nan=False)+"\n");stream.flush()
    try:
        if sha(args.checkpoint)!=args.checkpoint_sha256:raise ValueError("owned checkpoint SHA mismatch")
        if not torch.cuda.is_available():raise RuntimeError("authorized CUDA required; no silent CPU fallback")
        device=torch.device(args.device);torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,device)
        cls=load_source(args.source);payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        if (payload.get("schema")!="contextual-primitives-training-v1" or payload.get("stage")!=2
            or payload.get("step")!=600 or payload.get("stage_complete") is not True
            or payload["source_sha256"].get("continuous_primitive_block.py")!=CORE_SHA):
            raise ValueError("owned completed phase2 and EXACT original continuous source required")
        metadata=json.loads((args.data/"metadata.json").read_text())
        if payload["metadata_public"]!={k:metadata[k] for k in ("n_states","n_tokens","max_length")}:
            raise ValueError("public data/checkpoint interface differs")
        model=cls.from_checkpoint(payload["block_checkpoint"]).to(device)
        model.load_state_dict(payload["model_state"],strict=True)
        before_hash=state_hash(model.state_dict())
        if before_hash!=state_hash(payload["model_state"]):raise RuntimeError("actual restored neural tensors differ")
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        data=[];receipts=[]
        for stage in range(3):
            part,receipt=read_valid(args.data,metadata,payload,stage);data.append(part);receipts.append(receipt)
        emit("configuration",checkpoint_sha256=args.checkpoint_sha256,core_sha256=CORE_SHA,
            audit_source_sha256=sha(Path(__file__)),model_state_hash=before_hash,configuration=model.configuration(),
            parameters=model.parameter_count(),input_receipts=receipts,fixed_batch_size=64,
            interventions=list(INTERVENTIONS),interfaces=["soft","native"],training_updates=0,
            scope="frozen VALID dependence only; no TEST input/selection, no learning-causality claim",
            batch_history_definition="roll both keys and values by1 along currently active batch rows at each read; own query/state unchanged",
            timing_scope="includes intrusive hook statistics and extra baseline reads; not ordinary deployment latency")
        original=model._read;full_cache={}
        for intervention in INTERVENTIONS:
            for interface in ("soft","native"):
                for stage,part in enumerate(data):
                    check();torch.cuda.synchronize(device);tick,cpu=time.perf_counter(),time.process_time()
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
                    hook=ReadIntervention(original,intervention,model)
                    def wrapper(_self,h,x,keys,values):return hook(h,x,keys,values)
                    model._read=types.MethodType(wrapper,model)
                    arrays=[];base_proxy=0
                    try:
                        with torch.no_grad():
                            for offset in range(0,256,64):
                                check();sl=slice(offset,offset+64);lens=part["lengths"][sl];width=max(1,int(lens.max(initial=0)))
                                initial=torch.as_tensor(part["initial"][sl],dtype=torch.long,device=device)
                                tokens=torch.as_tensor(part["tokens"][sl,:width],dtype=torch.long,device=device)
                                lengths=torch.as_tensor(lens,dtype=torch.long,device=device)
                                output=model(initial,tokens,lengths,execution=interface,output="logits")
                                if not bool(torch.isfinite(output).all()):raise FloatingPointError("nonfinite frozen readout")
                                arrays.append(output.cpu().numpy())
                                base_proxy+=model.resource_report(64,width,active_steps=int(lens.sum()),execution=interface)["proxy_forward_macs"]
                    finally:model._read=original
                    end.record();torch.cuda.synchronize(device)
                    logp=np.concatenate(arrays);probabilities=np.exp(logp);predictions=logp.argmax(-1);targets=part["targets"]
                    statistics=hook.report();measurements=summarize(logp,predictions,targets,metadata["n_states"])
                    key=(interface,stage)
                    if intervention=="full":full_cache[key]=(probabilities.copy(),predictions.copy())
                    baseline,base_pred=full_cache[key]
                    output_file=args.out/f"{intervention}-{interface}-stage{stage}.npz"
                    with output_file.open("xb") as dest:np.savez_compressed(dest,log_probabilities=logp,probabilities=probabilities,
                        predictions=predictions,targets=targets,initial=part["initial"],lengths=part["lengths"],
                        terminal_NLL=-logp[np.arange(256),targets])
                    current_hash=state_hash(model.state_dict())
                    row=dict(intervention=intervention,interface=interface,stage=stage,**measurements,
                        **statistics,base_model_forward_MAC_proxy=base_proxy,
                        charged_model_plus_extra_read_MAC_proxy=base_proxy+statistics["extra_original_read_MAC_proxy"],
                        paired_full_probability_max_abs_difference=float(np.max(np.abs(probabilities-baseline))),
                        paired_full_prediction_disagreements=int((predictions!=base_pred).sum()),
                        wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-cpu,
                        CUDA_stream_elapsed_seconds=start.elapsed_time(end)/1000.,
                        actual_output_file=output_file.name,actual_output_sha256=sha(output_file),
                        actual_output_bytes=output_file.stat().st_size,model_state_hash=current_hash,
                        validation_label_evaluations=256)
                    if row["model_state_hash"]!=before_hash:raise RuntimeError("read intervention modified actual neural state")
                    results.append(row);emit("actual_read_intervention",**row)
        del model._read
        after_hash=state_hash(model.state_dict())
        if after_hash!=before_hash or sha(args.checkpoint)!=args.checkpoint_sha256 or sha(args.source)!=CORE_SHA:
            raise RuntimeError("owned state/source/checkpoint changed")
        result=dict(status="FROZEN_VALID_AUDIT_COMPLETED",results=results,training_updates=0,
            state_hash_before=before_hash,state_hash_after=after_hash,neural_weights_unchanged=True,
            checkpoint_bytes_unchanged=True,core_source_bytes_unchanged=True,wall_seconds=time.perf_counter()-began,
            validation_label_evaluations=len(results)*256,test_label_evaluations=0,
            claim="frozen VALID read dependence and paired sham effects only; no learning, retention theorem or TEST result",
            joint_roll_scope="joint key/value slot permutation preserves ideal read; measured FP differences/predictions reported without bit-exact claim")
        with (args.out/"result.json").open("x") as dest:json.dump(result,dest,indent=2,sort_keys=True,allow_nan=False)
    except BaseException as error:
        emit("audit_failed",error=type(error).__name__,reason=str(error),completed=len(results))
        with (args.out/"failure.json").open("x") as dest:json.dump(dict(error=type(error).__name__,reason=str(error),
            completed=len(results),wall_seconds=time.perf_counter()-began),dest,indent=2)
        raise
    finally:stream.close()


if __name__=="__main__":main()
