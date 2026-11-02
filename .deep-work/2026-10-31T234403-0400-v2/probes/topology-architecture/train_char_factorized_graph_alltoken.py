"""NEW cost implementation: exact original20721P fcba AR300 with F/B Graph.

Original3e644 class/data/sampler/objective/train-eval protocol retained.
Only B32L64 training F/B is graphed, with existing stablegradzero_; eager
Adam.003/clip1 outside.0SGDsetup + first/100/200/300 all27gradient gates.
No quality gate or source/permission expansion; no authoring-time execution.

Explicit architecture-only development: no anchor/profile auxiliary loss.
Original driver567e source remains frozen; changes documented in launch protocol.

New feedback protocol, never pooled with the old terminal-prefix experiment.
Continuous Block/GRU44/causal Transformer36 see identical windows and all labels.
No import-time work/network/installation, score-targeted retry or Primary claim.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from continuous_primitive_block import ContinuousPrimitiveBlock
from factorized_alltoken_graph import GraphARTrainer


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):h.update(chunk)
    return h.hexdigest()


def sources():return {name:sha(Path(__file__).with_name(name)) for name in (Path(__file__).name,"continuous_primitive_block.py","factorized_alltoken_graph.py","train_char_factorized_alltoken.py")}


def tensor_bytes(value):
    if torch.is_tensor(value):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(tensor_bytes(v) for v in value)
    return 0


class AllTokenLM(nn.Module):
    def __init__(self,arm,vocabulary):
        super().__init__();self.arm=arm;self.vocabulary=vocabulary
        if arm in ("dynamic","static","detached"):
            self.body=ContinuousPrimitiveBlock(n_states=vocabulary,n_tokens=vocabulary,internal_states=32,
                n_primitives=4,token_dim=16,state_dim=32,memory_dim=32,caller_hidden=32,n_slots=1,
                routing=arm,memory_slots=8,key_dim=16,primitive_rank=8,controller_dim=16,carry_gate_groups=8)
            # Full Vx32 encoder remains allocated/counted despite publicinitial0.
        elif arm=="gru":
            self.embedding=nn.Embedding(vocabulary,44);self.body=nn.GRU(44,44,num_layers=2,batch_first=True);self.readout=nn.Linear(44,vocabulary)
        elif arm=="transformer":
            self.embedding=nn.Embedding(vocabulary,36)
            self.body=nn.ModuleList([nn.TransformerEncoderLayer(36,4,dim_feedforward=72,dropout=0.,batch_first=True,norm_first=True) for _ in range(3)])
            self.norm=nn.LayerNorm(36);self.readout=nn.Linear(36,vocabulary)
            position=torch.arange(256,dtype=torch.float32)[:,None];frequency=torch.exp(torch.arange(0,36,2,dtype=torch.float32)*(-math.log(10000.)/36))
            encoding=torch.zeros(256,36);encoding[:,0::2]=torch.sin(position*frequency);encoding[:,1::2]=torch.cos(position*frequency)
            self.register_buffer("positions",encoding);self.register_buffer("causal_mask",torch.ones(256,256,dtype=torch.bool).triu(1))
        else:raise ValueError("unknown fixed architecture")

    def configuration(self):
        if self.arm in ("dynamic","static","detached"):return dict(arm=self.arm,core=self.body.configuration(),core_checkpoint_version=self.body.checkpoint_version,organization_policy=self.body.organization_policy,readout="all post-token normalizedcontinuous outputs",public_initial=0,auxiliary_loss="none")
        return dict(arm=self.arm,vocabulary=self.vocabulary,width=44 if self.arm=="gru" else 36,layers=2 if self.arm=="gru" else 3,
            heads=None if self.arm=="gru" else 4,feedforward=None if self.arm=="gru" else 72,
            position="none" if self.arm=="gru" else "fixedsinusoidal256",causal=True,initial_state="zero" if self.arm=="gru" else "noCLS")

    def forward(self,initial,tokens,lengths,mode="soft"):
        if tokens.ndim!=2 or tokens.shape[1]>256 or not bool((lengths==tokens.shape[1]).all()):raise ValueError("fixed full windows <=256 required")
        if self.arm in ("dynamic","static","detached"):
            return self.body.forward_all(initial,tokens,lengths,execution=mode,output="logits")
        h=self.embedding(tokens)
        if self.arm=="gru":h,_=self.body(h)
        else:
            h=h+self.positions[:tokens.shape[1]][None]
            for layer in self.body:h=layer(h,src_mask=self.causal_mask[:tokens.shape[1],:tokens.shape[1]],is_causal=True)
            h=self.norm(h)
        return F.log_softmax(self.readout(h),-1)

    def sequence_resource_report(self,batch_size,sequence_length,*,active_steps=None,execution="soft"):
        # Exact fcba core has terminal accounting only. Mirror the original
        # 58feb69 all-token adapter, charging physically computed padded outputs.
        report=self.body.resource_report(batch_size,sequence_length,active_steps=active_steps,execution=execution)
        components=dict(report["proxy_forward_macs_by_component"])
        components["readout"]=batch_size*sequence_length*self.body.internal_states*self.vocabulary
        report.update(proxy_forward_macs_by_component=components,proxy_forward_macs=sum(components.values()),
            output_readout_positions=batch_size*sequence_length,
            output_sequence_payload_bytes=batch_size*sequence_length*self.vocabulary*self.body.encoder.weight.element_size(),
            output_scope="one normalized full-batch readout per supplied token including padding; no trace",
            streaming_boundary="same explicit carry API; no cached-generation latency claim")
        return report

    def proxy(self,count,length,mode="soft"):
        if self.arm in ("dynamic","static","detached"):
            return self.sequence_resource_report(batch_size=count,sequence_length=length,
                active_steps=count*length,execution=mode)["proxy_forward_macs"]
        if self.arm=="gru":return count*length*(2*6*44*44+44*self.vocabulary)
        return count*(3*(4*length*36*36+2*length*length*36+2*length*36*72)+length*36*self.vocabulary)


def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--arm",choices=("dynamic","static","detached","gru","transformer"),required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--seed",type=int,default=0);p.add_argument("--max-seconds",type=float,default=850)
    p.add_argument("--resource-overlap-note",default="coordinator records actual concurrency; no isolatedthroughput claim")
    p.add_argument("--source-manifest",type=Path,required=True)
    args=p.parse_args()
    if args.arm!="dynamic" or args.seed!=0 or args.device!="cuda:0":p.error("one frozen dynamic H32 factorized seed0 CUDA Graph arm only")
    if not 1<=args.max_seconds<=850:p.error("internal bounded run<=850seconds")
    args.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();deadline=started+args.max_seconds
    stream=(args.out/"records.jsonl").open("x",buffering=1);device=torch.device(args.device)
    model=optimizer=graph_train=None;completed=0;curves=[];receipts=[];inputs=[];whole_CPU=time.process_time()
    costs=dict(optimizer_updates=0,training_windows=0,training_input_tokens=0,training_label_observations=0,
        training_forward_MAC_proxy=0,training_wall_seconds=0.,training_CPU_seconds=0.,batch_assembly_wall_seconds=0.,
        evaluation_forward_windows=0,evaluation_prediction_positions=0,evaluation_forward_MAC_proxy=0,evaluation_wall_seconds=0.,
        causality_audit_forward_windows=0,causality_audit_forward_MAC_proxy=0,checkpoint_wall_seconds=0.,reload_wall_seconds=0.,reload_forward_windows=0,reload_forward_MAC_proxy=0)
    def emit(kind,**value):stream.write(json.dumps(dict(kind=kind,wall=time.perf_counter()-started,**value),sort_keys=True,allow_nan=False)+"\n")
    def check():
        if time.perf_counter()>deadline:raise TimeoutError("all-token AR development bound")
    def sync():
        if device.type=="cuda":torch.cuda.synchronize(device)
    def write_json(name,value):
        with (args.out/name).open("x") as dest:json.dump(value,dest,indent=2,sort_keys=True,allow_nan=False);dest.write("\n")
    def load(split):
        entry=metadata["files"][split];path=args.data/entry["file"];position=args.data/entry["endpoint_file"]
        if path.stat().st_size>100<<20 or sha(path)!=entry["sha256"] or sha(position)!=entry["endpoint_sha256"]:raise ValueError("owned window/endpoint hash mismatch")
        with np.load(path,allow_pickle=False) as archive:
            if set(archive.files)!={"initial","tokens","targets","lengths"}:raise ValueError("unexpected all-token keys")
            data={k:archive[k] for k in archive.files}
        n,l=entry["rows"],entry["context_length"]
        if (any(v.dtype!=np.int64 for v in data.values()) or data["tokens"].shape!=(n,l) or data["targets"].shape!=(n,l)
                or data["initial"].shape!=(n,) or data["lengths"].shape!=(n,) or np.any(data["initial"]) or np.any(data["lengths"]!=l)
                or np.any(data["tokens"]<0) or np.any(data["tokens"]>=vocabulary) or np.any(data["targets"]<0) or np.any(data["targets"]>=vocabulary)
                or not np.array_equal(data["tokens"][:,1:],data["targets"][:,:-1])):raise ValueError("fixed AR data alignment/shape violation")
        endpoints=np.load(position,allow_pickle=False)
        if endpoints.dtype!=np.int64 or endpoints.shape!=(n,):raise ValueError("invalid endpoint vector")
        inputs.append(dict(split=split,file_sha256=entry["sha256"],endpoint_sha256=entry["endpoint_sha256"],rows=n,length=l,target_observations=n*l))
        return data,endpoints
    def batch(data,indices):return {k:torch.as_tensor(data[k][indices],device=device) for k in ("initial","tokens","targets","lengths")}
    def forward(b,mode="soft"):return model(b["initial"],b["tokens"],b["lengths"],mode)
    def resources():
        report=dict(parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            parameter_bytes=tensor_bytes(dict(model.named_parameters())),buffer_bytes=tensor_bytes(dict(model.named_buffers())),optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
            replay_bytes=0,cpu_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)),
            core_allocated_initial_encoder_parameters=model.body.encoder.weight.numel() if block else 0,
            core_alltoken_resource_report=model.sequence_resource_report(1,64) if block else None,
            data_cache_array_bytes=metadata["uncompressed_array_bytes"])
        if device.type=="cuda":report.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),cuda_reserved_bytes=torch.cuda.memory_reserved(device),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        return report
    def state_hash(value):
        h=hashlib.sha256()
        for name,tensor in sorted(value.state_dict().items()):
            t=tensor.detach().cpu().contiguous();h.update(json.dumps([name,str(t.dtype),list(t.shape)]).encode());h.update(t.numpy().tobytes())
        return h.hexdigest()
    @torch.no_grad()
    def evaluate(data,endpoints,split,save_vectors=False):
        was=model.training;model.eval();reports={};n=len(data["initial"]);length=data["tokens"].shape[1]
        try:
            for mode in (("soft","native") if block else ("soft",)):
                sync();tick=time.perf_counter();all_nll=[];predictions=[];normalizer=0.
                for offset in range(0,n,32):
                    check();b=batch(data,np.arange(offset,min(offset+32,n)));out=forward(b,mode)
                    if out.shape!=(len(b["initial"]),length,vocabulary) or not bool(torch.isfinite(out).all()):raise FloatingPointError("invalid all-token probability output")
                    normalizer=max(normalizer,float(torch.logsumexp(out,-1).abs().max()))
                    losses=F.nll_loss(out.reshape(-1,vocabulary),b["targets"].reshape(-1),reduction="none").reshape(len(b["initial"]),length)
                    all_nll.append(losses.cpu().numpy().astype(np.float64));predictions.append(out.argmax(-1).cpu().numpy())
                    costs["evaluation_forward_windows"]+=len(b["initial"]);costs["evaluation_prediction_positions"]+=len(b["initial"])*length;costs["evaluation_forward_MAC_proxy"]+=model.proxy(len(b["initial"]),length,mode)
                sync();elapsed=time.perf_counter()-tick;costs["evaluation_wall_seconds"]+=elapsed
                nll=np.concatenate(all_nll);prediction=np.concatenate(predictions);suffix=nll[:,-64:]
                if normalizer>1e-5:raise RuntimeError("logprobability normalization failed")
                mean=float(nll.mean());last=float(suffix.mean())
                row=dict(windows=n,context=length,all_token_observations=int(nll.size),all_token_NLL=mean,all_token_PPL=math.exp(mean) if mean<700 else None,
                    last64_paired_observations=int(suffix.size),last64_paired_NLL=last,last64_paired_PPL=math.exp(last) if last<700 else None,
                    all_token_accuracy=float(np.mean(prediction==data["targets"])),last64_paired_accuracy=float(np.mean(prediction[:,-64:]==data["targets"][:,-64:])),
                    maximum_log_normalizer_error=normalizer,wall_seconds=elapsed,forward_MAC_proxy=model.proxy(n,length,mode))
                if save_vectors:
                    path=args.out/f"{split}-{mode}-alltoken-vectors.npz"
                    with path.open("xb") as dest:np.savez_compressed(dest,nll=nll,predictions=prediction,targets=data["targets"],endpoints=endpoints)
                    row.update(vectors_file=path.name,vectors_sha256=sha(path),vectors_bytes=path.stat().st_size)
                reports[mode]=row
        finally:model.train(was)
        return reports
    def rng():
        ns=np.random.get_state()
        return dict(python=random.getstate(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all() if device.type=="cuda" else [],
            numpy=dict(algorithm=str(ns[0]),keys=ns[1].tolist(),position=int(ns[2]),has_gauss=int(ns[3]),cached_gaussian=float(ns[4])))
    def checkpoint(name):
        path=args.out/name;tick=time.perf_counter()
        payload=dict(schema="char-alltoken-ar-graph-neural-v1",graph_audit=graph_train.report() if graph_train is not None else None,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},metadata=metadata,
            model_configuration=model.configuration(),model_state=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng(),
            parameter_flags={k:v.requires_grad for k,v in model.named_parameters()},completed_updates=completed,input_receipts=copy.deepcopy(inputs),
            batch_indices_sha256=batch_sha,source_sha256=sources(),costs=copy.deepcopy(costs),torch_version=str(torch.__version__),
            training_budget=dict(updates=300,batch_size=32,window=64,labels=614400),exact_training_resume_tested=False,formal_Primary=False)
        with path.open("xb") as dest:torch.save(payload,dest)
        if path.stat().st_size>100<<20:raise RuntimeError("owned neuralcheckpoint cap")
        receipt=dict(file=name,sha256=sha(path),bytes=path.stat().st_size,completed_updates=completed,wall_seconds=time.perf_counter()-tick)
        receipts.append(receipt);costs["checkpoint_wall_seconds"]+=receipt["wall_seconds"];emit("actual_checkpoint",**receipt);return path
    try:
        if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("requested GPU unavailable; no fallback")
        source_manifest=json.loads(args.source_manifest.read_text())
        if source_manifest.get("schema")!="factorized-alltoken-graph-sources-v1" or sources()!=source_manifest["source_sha256"] or sha(args.data/"metadata.json")!="f157d58e9246f59f78dbd883b4ccc80284278b9eb270365cb13556dc8703e792":raise ValueError("exact frozen source/data manifest required")
        if sources()["continuous_primitive_block.py"]!="fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158" or sources()["train_char_factorized_alltoken.py"]!="3e6447a2a102c65d216b82b297798429f80b4c9189d0a3f2cc45c4149f584960":raise ValueError("exact unchanged20721P NN lineage required")
        torch.set_num_threads(1);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        if device.type=="cuda":torch.cuda.manual_seed_all(args.seed)
        metadata=json.loads((args.data/"metadata.json").read_text());vocabulary=metadata["n_tokens"]
        if metadata.get("schema")!="char-alltoken-ar-development-v1" or not 2<=vocabulary<=512 or metadata["n_states"]!=vocabulary:raise ValueError("owned fixed all-token AR metadata required")
        block=args.arm in ("dynamic","static","detached")
        train,_=load("train");valid,valid_endpoints=load("valid")
        if train["tokens"].shape!=(8192,64) or valid["tokens"].shape!=(512,64):raise ValueError("fixed TRAIN/VALID window counts required")
        batches=np.random.default_rng(np.random.SeedSequence([args.seed,vocabulary,300,0x41524254])).integers(0,8192,size=(300,32),dtype=np.int64)
        if hashlib.sha256(batches.tobytes(order="C")).hexdigest()!="da55630fb2576a5c174e4215082965f226d1a37ead1b50f1a616cd6fa1d1ac98":raise ValueError("exact original actual300x32 sampled batches required")
        with (args.out/"common-batch-indices.npz").open("xb") as dest:np.savez_compressed(dest,indices=batches)
        batch_sha=sha(args.out/"common-batch-indices.npz")
        model=AllTokenLM(args.arm,vocabulary).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
        emit("configuration",args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},source_sha256=sources(),metadata_sha256=sha(args.data/"metadata.json"),
            batch_indices_sha256=batch_sha,model_configuration=model.configuration(),resources=resources(),fixed_budget=dict(updates=300,batch=32,all_targets_per_window=64,training_label_observations=614400,lr=.003,optimizer="Adam",clip_norm=1.),
            feedback="ALL nextcharacters undercausal execution, no intermediate states/routes/operations supplied",formal_Primary=False,
            no_test_selection=True,old_terminal_prefix_results_not_pooled=True,primary_length_metric="last64 pairedNLL; fullwindowNLL separate")
        # TRAIN-input-only causality contract: changing future tokens cannot
        # change predictions on the unmodified first32 positions. No labels used.
        model.eval();probe=batch(train,np.arange(2));changed={k:v.clone() for k,v in probe.items()};changed["tokens"][:,32:]=(changed["tokens"][:,32:]+1)%vocabulary
        with torch.no_grad():left=forward(probe);right=forward(changed)
        causal_difference=float((left[:,:32]-right[:,:32]).abs().max())
        if causal_difference>2e-6:raise RuntimeError("future-token leakage detected by actual causality contract")
        costs["causality_audit_forward_windows"]+=4;costs["causality_audit_forward_MAC_proxy"]+=model.proxy(4,64)
        emit("actual_causality_contract",changed_input_suffix=[32,64],checked_output_prefix=[0,32],maximum_difference=causal_difference,TRAIN_inputs_only=True)
        graph_train=GraphARTrainer(model,optimizer,batch(train,batches[0]),args.out,check)
        emit("actual_zero_SGD_graph_setup",graph_audit=graph_train.report(),optimizer_updates=0)
        for step in range(301):
            check()
            if step%50==0:
                report=dict(step=step,VALID=evaluate(valid,valid_endpoints,"valid"));curves.append(report);emit("VALID_curve",**report)
                if step and step<300:checkpoint(f"{args.arm}-step-{step:03d}.pt")
            if step in (100,200,300):emit("actual_current_NN_all27_gradient_check",**graph_train.diagnostic(step))
            if step==300:break
            bt=time.perf_counter();b=batch(train,batches[step]);costs["batch_assembly_wall_seconds"]+=time.perf_counter()-bt
            model.train();sync();tick=time.perf_counter();cpu=time.process_time()
            out,loss=graph_train.run(b)
            if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite actual all-token trainloss")
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();graph_train.stable()
            if any(not bool(torch.isfinite(value).all()) for value in model.parameters()):raise FloatingPointError("nonfinite actual parameter")
            sync();completed=step+1;costs["optimizer_updates"]+=1;costs["training_windows"]+=32;costs["training_input_tokens"]+=2048;costs["training_label_observations"]+=2048
            costs["training_forward_MAC_proxy"]+=model.proxy(32,64);costs["training_wall_seconds"]+=time.perf_counter()-tick;costs["training_CPU_seconds"]+=time.process_time()-cpu
            emit("actual_alltoken_update",step=completed,all_token_train_NLL=float(loss.detach()),preclip_gradient_norm=float(norm),
                batch_next_character_accuracy=float((out.argmax(-1)==b["targets"]).float().mean()),cumulative_costs=costs)
        graph_train.close()
        final_path=checkpoint(f"{args.arm}-frozen-final.pt");before_test=state_hash(model);emit("ALL_TRAINING_FROZEN_BEFORE_TEST",completed_updates=completed,checkpoint=final_path.name,model_state_sha256=before_test)
        final=dict(VALID=curves[-1]["VALID"]);paired_endpoints=paired_targets=None
        for split in ("test64","test128","test256"):
            data,endpoints=load(split)
            if len(endpoints)!=512:raise ValueError("fixed512TESTwindows required")
            if paired_endpoints is None:paired_endpoints=endpoints.copy();paired_targets=data["targets"][:,-64:].copy()
            elif not np.array_equal(endpoints,paired_endpoints) or not np.array_equal(data["targets"][:,-64:],paired_targets):raise ValueError("last64 TEST pairing mismatch")
            final[split]=evaluate(data,endpoints,split,save_vectors=True);emit("FROZEN_TEST_RESULT",split=split,metrics=final[split])
        tick=time.perf_counter();payload=torch.load(final_path,map_location="cpu",weights_only=True);other=AllTokenLM(args.arm,vocabulary).to(device);other.load_state_dict(payload["model_state"],strict=True)
        equal=all(torch.equal(value.detach().cpu(),other.state_dict()[name].detach().cpu()) for name,value in model.state_dict().items())
        model.eval();other.eval();b=batch(valid,np.arange(32));mode_equality={}
        with torch.no_grad():
            for mode in (("soft","native") if block else ("soft",)):
                left=forward(b,mode);right=other(b["initial"],b["tokens"],b["lengths"],mode);same=bool(torch.equal(left,right));mode_equality[mode]=same
                if not same:raise RuntimeError("actual output reload mismatch")
                costs["reload_forward_windows"]+=64;costs["reload_forward_MAC_proxy"]+=model.proxy(64,64,mode)
        sync();costs["reload_wall_seconds"]+=time.perf_counter()-tick
        if not equal or before_test!=state_hash(model):raise RuntimeError("inference/reload changed actual weights")
        result=dict(status="completed300updates_alltoken_AR_Graph_development",graph_audit=graph_train.report(),original_parent_driver_sha256="3e6447a2a102c65d216b82b297798429f80b4c9189d0a3f2cc45c4149f584960",all27_gradient_storage_stable=True,graph_count=1,no_compiler_exception=True,no_accuracy_or_speed_gate=True,arm=args.arm,completed_updates=completed,
            model_configuration=model.configuration(),final=final,curves=curves,resources=resources(),costs=costs,checkpoints=receipts,input_receipts=inputs,
            batch_indices_sha256=batch_sha,metadata_sha256=sha(args.data/"metadata.json"),source_sha256=sources(),
            actual_causality_max_difference=causal_difference,reload_audit=dict(actual_tensor_values_equal=equal,actual_sameinput_alltoken_outputs_equal=mode_equality,exact_training_resume_tested=False),
            unchanged_weights_after_test=True,paired_last64_TEST_labels=True,formal_Primary=False,old_prefix_results_not_pooled=True,
            metric_scope="sampledalltokenAR windows, not fullcorpus/wordPPL; primarylengthcomparison same32768last64targets",
            cost_scope="nestedwalltimes do notsum; forwardMACproxies excludebackward/optimizer; allvalidation/native/testwork counted; no exclusivethroughput/energyclaim",
            resource_overlap_note=args.resource_overlap_note,driver_wall_seconds=time.perf_counter()-started,driver_CPU_seconds=time.process_time()-whole_CPU)
        write_json("summary.json",result);emit("completed",actual_updates=completed,training_label_observations=costs["training_label_observations"])
        print(json.dumps(dict(status=result["status"],out=str(args.out),completed_updates=completed)))
    except BaseException as error:
        emit("failed",error=f"{type(error).__name__}: {error}",completed_updates=completed)
        (args.out/"exception.txt").write_text(traceback.format_exc());write_json("failure.json",dict(status="failed_or_bounded_incomplete",error=f"{type(error).__name__}: {error}",completed_updates=completed,costs=costs,curves=curves,checkpoints=receipts))
        if model is not None:
            with (args.out/"partial-actual-state.pt").open("xb") as dest:torch.save(dict(model_state=model.state_dict(),completed_updates=completed),dest)
        raise
    finally:
        if graph_train is not None:graph_train.close()
        stream.close()


if __name__=="__main__":main()
