"""ROOT-RUN frozen dynamic-soft VALID memory-path diagnostic, no learning.

R[s,u,j]=product_{s<t<u}(1-g[t]*a[t,j]) maps memory immediately
AFTER observed WRITE token s to memory immediately BEFORE query token u.
This is a DIRECT coefficient with observed write features held fixed, not a
total Jacobian: h, keys, queries, gates, candidates and later feedback depend
on history. Read interventions measure present frozen dependence only.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import types

import numpy as np
import torch

PARENT="parameter-memory-live-stage-n8-seed0-dynamic-stage4-v1"
CHECKPOINT_SHA="c82acdb59be35002ef19c069d79cf43d143902be6d0c2eb2aa54eebd1c11387a"
CHECKPOINT_BYTES=613915
CORE_SHA="b8a51a6ca9026b2ebff877493bf9031972a319d097cb3ae2605cc6ad3ea3083f"
WRAPPER_SHA="dd5fc0b869d216f29c6ccb7202a1d965086c3fedb060d3c4c3bc4e5620042ad6"
HOOK_SHA="17bb5fdf6ce77d7b0ffdb9bfc6904988ef2c813b77ebf21b8cba690b80f963f4"
MODES=("full","zero_read","joint_slot_roll1","value_slot_roll1")
KEYS=("initial","tokens","lengths","targets")
MAX_FILE=100*1024*1024


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def quantiles(value):
    value=np.asarray(value,dtype=np.float64).reshape(-1)
    if not len(value):return dict(count=0,minimum=None,p10=None,median=None,p90=None,maximum=None,mean=None)
    return dict(count=len(value),minimum=float(value.min()),p10=float(np.quantile(value,.1)),
                median=float(np.median(value)),p90=float(np.quantile(value,.9)),maximum=float(value.max()),mean=float(value.mean()))


def coefficients(tokens,lengths,gates,addresses,read_attention):
    rows,_,slots=addresses.shape
    positions=np.empty((rows,2),dtype=np.int64);queries=np.empty((rows,2),dtype=np.int64)
    shape=(rows,2,2,slots)
    retention=np.empty(shape,dtype=np.float64);nonwrite=np.empty_like(retention);write=np.empty_like(retention)
    injection=np.empty((rows,2,slots),dtype=np.float64);read_weight=np.empty_like(retention)
    multiplication_count=0
    # Reproduce the actual fp32 elementwise write amount, then perform the
    # diagnostic product in fp64. No teacher state or labels enter this step.
    amounts=(gates*addresses).astype(np.float32)
    if not np.isfinite(amounts).all() or np.any((amounts<0)|(amounts>1)):raise ValueError("actual write amount outside[0,1]")
    for row,length in enumerate(lengths):
        word=tokens[row,:int(length)];source=np.flatnonzero(word==9)
        if len(source)!=2 or int(word[-3])!=10 or int(word[-1])!=15:raise ValueError("observed two-WRITE/terminal query shape differs")
        destination=np.array([int(length)-3,int(length)-1],dtype=np.int64)
        positions[row]=source;queries[row]=destination
        for source_index,s in enumerate(source):
            injection[row,source_index]=amounts[row,s].astype(np.float64)
            for query_index,u in enumerate(destination):
                if s>=u:raise ValueError("source WRITE must precede query")
                factor=1.-amounts[row,s+1:u].astype(np.float64)
                mask=word[s+1:u]==9
                r=factor.prod(0);rw=factor[mask].prod(0);rn=factor[~mask].prod(0)
                retention[row,source_index,query_index]=r;write[row,source_index,query_index]=rw
                nonwrite[row,source_index,query_index]=rn
                if not np.allclose(r,rw*rn,rtol=1e-12,atol=1e-15):raise AssertionError("factor partition differs")
                read_weight[row,source_index,query_index]=injection[row,source_index]*r*read_attention[row,u]
                multiplication_count+=len(factor)*slots
    return dict(write_token_positions=positions,query_token_positions=queries,
                direct_retention=retention,direct_non_WRITE_retention=nonwrite,direct_WRITE_retention=write,
                source_write_amount=injection,fixed_feature_source_candidate_to_read_weight=read_weight),multiplication_count


def coefficient_self_test():
    tokens=np.asarray([[9,14,9,10,7,15]],dtype=np.int64);lengths=np.asarray([6],dtype=np.int64)
    gate=np.ones((1,6,1),dtype=np.float32)
    address=np.asarray([[[.1,.2],[.2,.3],[.4,.5],[.6,.7],[.8,.9],[.2,.3]]],dtype=np.float32)
    data,_=coefficients(tokens,lengths,gate,address,np.ones_like(address))
    amount=(gate*address).astype(np.float64)
    assert np.allclose(data["direct_retention"][0,0,0],(1-amount[0,1])*(1-amount[0,2]))
    assert np.array_equal(data["direct_retention"][0,1,0],np.ones(2))
    assert np.allclose(data["direct_retention"][0,1,1],(1-amount[0,3])*(1-amount[0,4]))
    # Fixed-feature scalar recurrence finite difference checks the product;
    # it deliberately does not model feedback through write features.
    def replay(value):
        for t in (1,2):value=value+amount[0,t,0]*(.37-value)
        return value
    derivative=(replay(.6+1e-5)-replay(.6-1e-5))/(2e-5)
    assert abs(derivative-data["direct_retention"][0,0,0,0])<1e-10
    return dict(source_and_destination_excluded=True,empty_interval_identity=True,
                fixed_feature_scalar_recurrence_checked=True,WRITE_non_WRITE_partition_checked=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("checkpoint","frozen-source","manifest","data","out"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--device",choices=("cuda:0",),default="cuda:0");p.add_argument("--max-seconds",type=float,default=850)
    args=p.parse_args()
    if args.out.exists() or not 0<args.max_seconds<=850:raise ValueError("new output/finite850s bound required")
    args.out.mkdir(parents=True);stream=(args.out/"records.jsonl").open("x")
    began=time.perf_counter();deadline=began+args.max_seconds;device=torch.device(args.device)
    model=None;before=None;payload=None;results=[];active_case=None
    def check():
        if time.perf_counter()>=deadline:raise TimeoutError("bounded VALID diagnostic deadline")
    def emit(kind,**fields):
        stream.write(json.dumps(dict(kind=kind,wall_seconds=time.perf_counter()-began,**fields),sort_keys=True,allow_nan=False)+"\n");stream.flush()
    def memory():
        return dict(cuda_allocated_bytes=torch.cuda.memory_allocated(device),cuda_reserved_bytes=torch.cuda.memory_reserved(device),
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    cpu_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=="darwin" else 1024)))
    try:
        if not torch.cuda.is_available():raise RuntimeError("authorized CUDA required; no fallback")
        torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,device)
        manifest=json.loads(args.manifest.read_text())
        if manifest["parent"]!=PARENT or manifest["checkpoint_sha256"]!=CHECKPOINT_SHA:raise ValueError("fixed OWN dynamic stage4 allowlist differs")
        for leaf,expected in manifest["all_source_sha256"].items():
            path=args.frozen_source/leaf
            if path.is_symlink() or sha(path)!=expected:raise ValueError("actual frozen source differs: "+leaf)
        if manifest["parent_source_sha256"].get("memory_live_stage_train.py")!=WRAPPER_SHA or manifest["parent_source_sha256"].get("continuous_primitive_block.py")!=CORE_SHA:
            raise ValueError("actual parent wrapper/core source differs")
        if args.checkpoint.is_symlink() or args.checkpoint.stat().st_size!=CHECKPOINT_BYTES or sha(args.checkpoint)!=CHECKPOINT_SHA:raise ValueError("exact OWN completed checkpoint required")
        load_tick=time.perf_counter();payload=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
        if (payload.get("schema")!="contextual-primitives-training-v1" or payload.get("stage")!=3
                or payload.get("step")!=600 or payload.get("stage_complete") is not True
                or payload["source_sha256"]!=manifest["parent_source_sha256"]
                or payload["args"]["arm"]!="dynamic"):raise ValueError("actual completed dynamic fourth-phase content differs")
        if sha(args.frozen_source/"owned_memory_read_hooks.py")!=HOOK_SHA:raise ValueError("old read-hook bytes differ")
        hooks=load_module(args.frozen_source/"owned_memory_read_hooks.py","owned_stage4_read_hooks")
        cls=hooks.load_source(args.frozen_source/"continuous_primitive_block.py")
        model=cls.from_checkpoint(payload["block_checkpoint"]).to(device);model.load_state_dict(payload["model_state"],strict=True)
        if model.configuration()["n_tokens"]!=16 or model.n_slots!=1 or model.parameter_count()!=28381:raise ValueError("actual grown dynamic configuration differs")
        before=hooks.state_hash(model.state_dict())
        if before!=hooks.state_hash(payload["model_state"]):raise AssertionError("actual NN restore mismatch")
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        source_sha=sha(args.checkpoint)
        # The whole original Adam is retained as read-only CPU checkpoint data;
        # no live optimizer is constructed or updated.
        adam_tensor_bytes=sum(t.numel()*t.element_size() for state in payload["optimizer"]["state"].values() for t in state.values() if torch.is_tensor(t))
        data_path=args.data/"stage3-valid.npz"
        if sha(data_path)!=manifest["VALID_sha256"]:raise ValueError("actual original VALID bytes differ")
        with np.load(data_path,allow_pickle=False) as archive:
            if set(archive.files)!=set(KEYS):raise ValueError("raw terminal arrays only")
            data={key:archive[key].copy() for key in KEYS}
        if (any(v.dtype!=np.int64 for v in data.values()) or data["tokens"].shape!=(512,128)
                or any(data[k].shape!=(512,) for k in ("initial","lengths","targets"))
                or np.any((data["lengths"]<1)|(data["lengths"]>48))):raise ValueError("fixed512 raw VALID interface differs")
        selftest=coefficient_self_test();original=model._read;baseline=None;all_costs=[];maximum=int(data["lengths"].max())
        emit("configuration",parent=PARENT,checkpoint_sha256=CHECKPOINT_SHA,model_state_hash=before,
             parameters=model.parameter_count(),parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
             registered_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
             checkpoint_bytes=CHECKPOINT_BYTES,loaded_readonly_Adam_tensor_bytes=adam_tensor_bytes,live_optimizer_bytes=0,
             loaded_VALID_array_bytes=sum(v.nbytes for v in data.values()),source_sha256=manifest["all_source_sha256"],
             actual_parent_training_costs=payload["costs"],load_restore_wall_seconds=time.perf_counter()-load_tick,
             interventions=MODES,interface="soft",rows=512,trace_prefix_full_memory_rows=8,selftest=selftest,
             training_updates=0,TRAIN_or_TEST_files_opened=0,formal_Primary=False,
             claim="present frozen VALID dependence and direct fixed-feature memory-path coefficients only; no totalJacobian or acquisition-causality claim")
        for mode in MODES:
            check();active_case=dict(mode=mode,completed_rows=0);torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device);tick,cpu=time.perf_counter(),time.process_time()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record()
            gates=np.zeros((512,maximum,1),dtype=np.float32);addresses=np.zeros((512,maximum,model.memory_slots),dtype=np.float32)
            attentions=np.zeros_like(addresses);raw_attentions=np.zeros_like(addresses)
            reads=np.zeros((512,maximum,model.n_slots,model.internal_states),dtype=np.float32)
            outputs=[];full_memory={};proxy=0;hook=hooks.ReadIntervention(original,mode,model);max_trace_gpu_bytes=0
            for offset in range(0,512,64):
                check();lens=data["lengths"][offset:offset+64];width=int(lens.max());captured=[]
                def read_wrapper(_self,h,x,keys,values):
                    read,attention=hook(h,x,keys,values)
                    captured.append(attention.detach().cpu().numpy())
                    return read,attention
                model._read=types.MethodType(read_wrapper,model)
                try:
                    with torch.no_grad():
                        output,trace=model(torch.as_tensor(data["initial"][offset:offset+64],device=device),
                            torch.as_tensor(data["tokens"][offset:offset+64,:width],device=device),
                            torch.as_tensor(lens,device=device),execution="soft",output="logits",return_traces=True)
                    if not bool(torch.isfinite(output).all()) or len(captured)!=width:raise AssertionError("actual trace/output differs")
                    outputs.append(output.cpu().numpy())
                    gates[offset:offset+64,:width]=trace["write_gates"].cpu().numpy()
                    addresses[offset:offset+64,:width]=trace["write_addresses"].cpu().numpy()
                    reads[offset:offset+64,:width]=trace["reads"].cpu().numpy()
                    for t,a in enumerate(captured):
                        ids=np.flatnonzero(t<lens);raw_attentions[offset+ids,t]=a
                        used=(np.zeros_like(a) if mode=="zero_read" else np.roll(a,-1,axis=1)
                              if mode in ("joint_slot_roll1","value_slot_roll1") else a)
                        attentions[offset+ids,t]=used
                    if offset==0:
                        for key in ("states","memory_keys","memory_values"):
                            full_memory["prefix8_"+key]=trace[key][:8].cpu().numpy()
                    max_trace_gpu_bytes=max(max_trace_gpu_bytes,sum(v.numel()*v.element_size() for v in trace.values() if torch.is_tensor(v)))
                    proxy+=model.resource_report(64,width,active_steps=int(lens.sum()),execution="soft",return_traces=True)["proxy_forward_macs"]
                    active_case["completed_rows"]=offset+64
                    del trace,output,captured
                finally:model._read=original
            end.record();torch.cuda.synchronize(device);forward_wall=time.perf_counter()-tick;forward_cpu=time.process_time()-cpu
            gpu_elapsed=start.elapsed_time(end)/1000.;analysis_tick=time.perf_counter()
            logp=np.concatenate(outputs);pred=logp.argmax(-1);prob=np.exp(logp)
            if baseline is None:baseline=(prob.copy(),pred.copy())
            derived,multiplications=coefficients(data["tokens"],data["lengths"],gates,addresses,attentions)
            active=np.arange(maximum)[None,:]<data["lengths"][:,None]
            write_mask=active&(data["tokens"][:,:maximum]==9);other_mask=active&~write_mask
            summary=dict(overall_direct_retention=quantiles(derived["direct_retention"]),
                non_WRITE_direct_retention=quantiles(derived["direct_non_WRITE_retention"]),
                WRITE_direct_retention=quantiles(derived["direct_WRITE_retention"]),
                source_candidate_to_read_weight=quantiles(derived["fixed_feature_source_candidate_to_read_weight"]),
                observed_WRITE_token_gate=quantiles(gates[:,:,0][write_mask]),
                observed_non_WRITE_token_gate=quantiles(gates[:,:,0][other_mask]),
                source_injection_mass=quantiles(derived["source_write_amount"].sum(-1)))
            intervals=[]
            for source_index in range(2):
                mass=derived["source_write_amount"][:,source_index].sum(-1);positive=mass>0
                for query_index in range(2):
                    retained=(derived["source_write_amount"][:,source_index]*derived["direct_retention"][:,source_index,query_index]).sum(-1)
                    intervals.append(dict(source_WRITE_occurrence=source_index,query_occurrence=query_index,
                        query_observed_token=10 if query_index==0 else 15,
                        direct_retention=quantiles(derived["direct_retention"][:,source_index,query_index]),
                        non_WRITE_direct_retention=quantiles(derived["direct_non_WRITE_retention"][:,source_index,query_index]),
                        source_write_mass_weighted_retention=quantiles(retained[positive]/mass[positive]),
                        zero_source_write_mass_rows=int((~positive).sum()),
                        fixed_feature_source_candidate_to_read_total_weight=quantiles(derived["fixed_feature_source_candidate_to_read_weight"][:,source_index,query_index].sum(-1))))
            summary["WRITE_to_query_intervals"]=intervals
            path=args.out/(mode+"-raw-valid-traces.npz")
            with path.open("xb") as dest:np.savez_compressed(dest,**data,log_probabilities=logp,predictions=pred,
                write_gates=gates,write_addresses=addresses,reads=reads,
                hook_effective_read_attention=raw_attentions,value_slot_aligned_read_attention=attentions,**derived,**full_memory)
            if path.stat().st_size>MAX_FILE:raise RuntimeError("100MiB raw trace file bound")
            statistics=hook.report();current_hash=hooks.state_hash(model.state_dict())
            if current_hash!=before:raise AssertionError("actual NN tensors changed during read intervention")
            row=dict(mode=mode,**hooks.summarize(logp,pred,data["targets"],8),summary=summary,
                paired_full_probability_max_abs_difference=float(np.abs(prob-baseline[0]).max()),
                paired_full_prediction_disagreements=int((pred!=baseline[1]).sum()),
                read_hook_statistics=statistics,model_forward_MAC_proxy=proxy,
                model_plus_extra_read_MAC_proxy=proxy+statistics["extra_original_read_MAC_proxy"],
                instrumented_forward_wall_seconds=forward_wall,instrumented_forward_CPU_seconds=forward_cpu,
                instrumented_CUDA_stream_elapsed_seconds=gpu_elapsed,analysis_and_write_wall_seconds=time.perf_counter()-analysis_tick,
                fixed_feature_product_multiplications=multiplications,maximum_trace_tensor_payload_bytes=max_trace_gpu_bytes,
                actual_forward_examples=512,validation_label_evaluations=512,exact_forward_and_trace_FLOPs=None,
                raw_trace_file=path.name,raw_trace_bytes=path.stat().st_size,raw_trace_sha256=sha(path),memory=memory(),model_state_hash=current_hash)
            results.append(row);all_costs.append(row);emit("actual_frozen_VALID_intervention",**row);active_case=None
        del model._read
        if sha(args.checkpoint)!=source_sha or hooks.state_hash(model.state_dict())!=before:raise AssertionError("actual checkpoint/NN changed")
        for leaf,expected in manifest["all_source_sha256"].items():
            if sha(args.frozen_source/leaf)!=expected:raise AssertionError("frozen source bytes changed")
        result=dict(status="FROZEN_STAGE4_DYNAMIC_SOFT_VALID_MEMORY_AUDIT_COMPLETED",parent=PARENT,results=results,
            parameters=model.parameter_count(),checkpoint_sha256=CHECKPOINT_SHA,model_state_before_sha256=before,
            model_state_after_sha256=hooks.state_hash(model.state_dict()),checkpoint_and_NN_unchanged=True,
            Adam_scope="actual original Adam loaded read-only on CPU; no optimizer constructed; unchanged checkpoint SHA verifies storage",
            read_only_Adam_tensor_bytes=adam_tensor_bytes,training_updates=0,validation_label_evaluations=2048,
            TRAIN_or_TEST_files_opened=0,source_sha256=manifest["all_source_sha256"],total_driver_wall_seconds=time.perf_counter()-began,
            timing_scope="intrusive hooks/trace copies/statistics; not inference throughput",formal_Primary=False,
            direct_coefficient_definition="memory-after source WRITE s to memory-before query u; product s<t<u of(1-g_t*a_tj)",
            coefficient_limits="features/gates/addresses/candidates/attention held fixed; not totalJacobian; all indirect feedback paths excluded",
            read_alignment="joint and value-only roll hooks move persistent value slotj to read slotj+1, so align returned attention byroll(-1); zero_read coefficient0",
            query_positions="observed final READ token10 and terminal MIXREAD token15; raw position analysis never passed to NN",
            evidence_boundary="frozen VALID dependence and direct-path attenuation; no acquisition claim, test-based choice, or proof arbitrary-key content addressing")
        with (args.out/"result.json").open("x") as dest:json.dump(result,dest,indent=2,sort_keys=True,allow_nan=False)
    except BaseException as error:
        emit("audit_failed",error=type(error).__name__,reason=str(error),completed_interventions=len(results),active_case=active_case)
        with (args.out/"failure.json").open("x") as dest:json.dump(dict(error=type(error).__name__,reason=str(error),
            completed=results,active_case=active_case,wall_seconds=time.perf_counter()-began),dest,indent=2)
        raise
    finally:stream.close()


if __name__=="__main__":main()
