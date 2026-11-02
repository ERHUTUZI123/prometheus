"""Owned fourth-phase continuation; no teacher/oracle import, SSH or network.

The launcher freezes an OWN parent's complete original execution source as
owned_parent_driver.py plus its exact dependencies. Only vocabulary storage
and public position extent grow. Actual neural/Adam/RNG/Replay128 state is
restored; 600 new updates contain 48 new TRAIN rows and 16 stored old rows.
This new development test does not repair or reinterpret old measurements.
"""
import argparse
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

import owned_parent_driver as old
from context_replay128 import Replay128

SPLITS=("train","valid","test_iid","test_long","test_unseen")
RAW=("initial","tokens","lengths","targets")
MAX_FILE=100*1024*1024


def sha(path): return old.sha256(path)


def verify_file(path,expected):
    if not path.is_file() or path.is_symlink() or path.stat().st_size>MAX_FILE:
        raise ValueError("regular bounded owned file required: "+str(path))
    if sha(path)!=expected: raise ValueError("actual owned file SHA differs: "+str(path))


def read_inputs(args,manifest,newmeta):
    if {k:newmeta[k] for k in ("n_states","n_tokens","max_length","stage","seed")}!=dict(
            n_states=8,n_tokens=16,max_length=128,stage=3,seed=0):
        raise ValueError("frozen fourth-task public metadata differs")
    if newmeta["source_sha256"]!=manifest["data_source_sha256"]:
        raise ValueError("offline generator source SHA differs from prospective protocol")
    if sha(args.data/"metadata.json")!=manifest["data_metadata_sha256"]:
        raise ValueError("actual new metadata differs from frozen generated receipt")
    oldmeta=json.loads((args.old_data/"metadata.json").read_text())
    if sha(args.old_data/"metadata.json")!=manifest["old_metadata_sha256"]:
        raise ValueError("actual old metadata differs from frozen lineage")
    datasets={};receipts=[]
    # Tests remain unopened until the only new training phase is frozen.
    for stage in range(4):
        datasets[stage]={}
        for split in (("valid",) if stage<3 else ("train","valid")):
            data,receipt=load_input(args,manifest,newmeta,oldmeta,stage,split)
            datasets[stage][split]=data;receipts.append(receipt)
    return oldmeta,datasets,receipts


def load_input(args,manifest,newmeta,oldmeta,stage,split):
    if stage<3:
        entry=manifest["old_files"][f"{stage}:{split}"]
        path=args.old_data/entry["file"];metadata=oldmeta
    else:
        entry=newmeta["files"][split];path=args.data/entry["file"]
        metadata=newmeta
    verify_file(path,entry["sha256"])
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(RAW): raise ValueError("raw terminal arrays only")
        if any(z[k].dtype!=np.int64 for k in RAW): raise ValueError("int64 raw arrays required")
        data={k:z[k].copy() for k in RAW}
    b=len(data["initial"]);tokens=data["tokens"];lengths=data["lengths"]
    if (not 1<=b<=4096 or tokens.ndim!=2 or tokens.shape[0]!=b
        or tokens.shape[1]>metadata["max_length"]
        or any(data[k].shape!=(b,) for k in ("initial","lengths","targets"))
        or np.any((lengths<0)|(lengths>tokens.shape[1]))
        or np.any((data["initial"]<0)|(data["initial"]>=8))
        or np.any((data["targets"]<0)|(data["targets"]>=8))):
        raise ValueError("raw data shape/public range differs")
    active=np.arange(tokens.shape[1])[None,:]<lengths[:,None]
    if np.any(active&((tokens<0)|(tokens>=metadata["n_tokens"]))):
        raise ValueError("active token outside actual embedding")
    if stage==3 and b!=newmeta["counts"][split]: raise ValueError("prospective row count differs")
    return data,dict(stage=stage,split=split,file=path.name,file_sha256=sha(path),
                     array_sha256=old.array_hash(data),rows=b,array_bytes=sum(v.nbytes for v in data.values()))


def source_hashes(manifest):
    folder=Path(__file__).resolve().parent
    for leaf,expected in manifest["execution_sources"].items(): verify_file(folder/leaf,expected)
    return {leaf:sha(folder/leaf) for leaf in manifest["execution_sources"]}


def grow_storage(model,optimizer,args):
    """Append ONE random N(0,1) embedding row; migrate actual Adam by name.

    nn.Embedding's original initialization is N(0,1). No operator or route is
    assigned. Existing moment tensors keep their exact bits, including step;
    the appended row has zero first/second moments and uses the inherited step.
    """
    began=time.perf_counter();named=dict(model.named_parameters())
    name="token_embedding.weight" if old.is_block(args) else "tokens.weight"
    previous=named[name]
    if previous.ndim!=2 or previous.shape[0]!=15: raise ValueError("OWN 15-row embedding required")
    original=previous.detach().clone();previous_state=copy.deepcopy(optimizer.state_dict())
    other_tensors={k:v.detach().clone() for k,v in model.state_dict().items() if k!=name}
    states_by_name={n:copy.deepcopy(optimizer.state.get(p,{})) for n,p in named.items()}
    # Factories were neutralized by restore_rng before this one permitted draw.
    appended=torch.randn((1,previous.shape[1]),device=previous.device,dtype=previous.dtype)
    if not bool(torch.isfinite(appended).all()) or not bool(appended.count_nonzero()):
        raise AssertionError("new row is not a finite random learned embedding")
    embedding=nn.Embedding.from_pretrained(torch.cat((previous.detach(),appended),0),
                                           freeze=not previous.requires_grad)
    if old.is_block(args):
        model.token_embedding=embedding;model.n_tokens=16;model.config=replace(model.config,n_tokens=16)
    else:
        model.tokens=embedding;model.spec=dict(model.spec,n_tokens=16,max_length=128)
    replacement=embedding.weight
    matched=0
    for group in optimizer.param_groups:
        for index,p in enumerate(group["params"]):
            if p is previous: group["params"][index]=replacement;matched+=1
    if matched!=1: raise ValueError("old Adam embedding parameter not uniquely named")
    state=optimizer.state.pop(previous,{})
    for key,value in list(state.items()):
        if torch.is_tensor(value) and value.shape==previous.shape:
            state[key]=torch.cat((value,torch.zeros_like(value[:1])),0)
    optimizer.state[replacement]=state
    buffer_added=0;position_prefix=None
    if args.arm=="transformer":
        prefix=model.positions.detach().clone();d=model.spec["width"]
        position=torch.arange(130,device=prefix.device,dtype=prefix.dtype)[:,None]
        frequency=torch.exp(torch.arange(0,d,2,device=prefix.device,dtype=prefix.dtype)*(-math.log(10000.)/d))
        encoding=torch.zeros((130,d),device=prefix.device,dtype=prefix.dtype)
        encoding[:,0::2]=torch.sin(position*frequency);encoding[:,1::2]=torch.cos(position*frequency)
        encoding[:len(prefix)]=prefix;model.positions=encoding
        position_prefix=bool(torch.equal(model.positions[:len(prefix)],prefix))
        buffer_added=(encoding.numel()-prefix.numel())*prefix.element_size()
    if not torch.equal(replacement[:15],original): raise AssertionError("old embedding rows changed")
    for n,before in other_tensors.items():
        after=model.state_dict()[n]
        if n=="positions":after=after[:len(before)]
        if not torch.equal(before,after): raise AssertionError("old actual neural tensor changed: "+n)
    new_named=dict(model.named_parameters())
    if new_named.keys()!=named.keys(): raise AssertionError("actual named parameter set changed")
    for n,old_state in states_by_name.items():
        new_state=optimizer.state.get(new_named[n],{})
        for key,value in old_state.items():
            actual=new_state[key]
            if n==name and torch.is_tensor(value) and value.shape==previous.shape:
                if not torch.equal(actual[:15],value) or bool(actual[15:].count_nonzero()):
                    raise AssertionError("old Adam moments/new zero moments differ")
            elif not old.nested_state_equal(value,actual): raise AssertionError("Adam state changed: "+n+":"+key)
    return dict(embedding_parameter_name=name,parameters_added=previous.shape[1],
        embedding_parameter_bytes_added=appended.numel()*appended.element_size(),
        optimizer_tensor_bytes_before=old.tensor_bytes(previous_state),
        optimizer_tensor_bytes_after=old.tensor_bytes(optimizer.state_dict()),
        optimizer_tensor_bytes_added=old.tensor_bytes(optimizer.state_dict())-old.tensor_bytes(previous_state),
        old_NN_tensors_and_embedding_rows_exact=True,old_Adam_moments_and_steps_exact=True,
        new_row_moments_zero=True,random_row_initialization="ONE independent N(0,1) row; no supplied semantics",
        public_n_tokens=16,public_max_length=128,Transformer_position_prefix_exact=position_prefix,
        registered_position_buffer_bytes_added=buffer_added,additional_position_model_parameters=0,
        growth_wall_seconds=time.perf_counter()-began,exact_growth_FLOPs=None)


def restore(args,manifest,device,rec,oldmeta):
    began=time.perf_counter();verify_file(args.resume,manifest["checkpoint_sha256"])
    payload=torch.load(args.resume,map_location="cpu",weights_only=False)
    if (payload.get("schema")!="contextual-primitives-training-v1" or payload.get("stage")!=2
            or payload.get("step")!=600 or payload.get("stage_complete") is not True):
        raise ValueError("OWN completed phase2/600 checkpoint required")
    if payload["source_sha256"]!=manifest["parent_actual_source_sha256"]:
        raise ValueError("actual checkpoint source lineage differs")
    if payload["metadata_public"]!={k:oldmeta[k] for k in ("n_states","n_tokens","max_length")}:
        raise ValueError("actual checkpoint old public metadata differs")
    for receipt in payload["input_receipts"]:
        key=f'{receipt["stage"]}:{receipt["split"]}'
        if manifest["old_files"][key]["sha256"]!=receipt["file_sha256"]:
            raise ValueError("checkpoint old TRAIN/VALID hashes differ from actual files")
    model,optimizer,saved=old.restore_model(payload,device)
    if old.plain(vars(saved))!=manifest["parent_actual_configuration"]: raise ValueError("actual saved arguments differ")
    if not old.tensor_state_equal(payload["model_state"],model.state_dict()): raise AssertionError("actual NN restore mismatch")
    if not old.nested_state_equal(payload["optimizer"],optimizer.state_dict()): raise AssertionError("actual Adam restore mismatch")
    if device.type=="cuda":
        old.restore_rng(payload["rng"])
        if not old.nested_state_equal(payload["rng"],old.rng_state()): raise AssertionError("actual RNG restore mismatch")
    else:
        # CPU isolation deliberately hides CUDA. CPU checks restore/check all
        # available streams; actual GPU continuation must check CUDA too.
        torch.set_rng_state(payload["rng"]["cpu"]);np.random.set_state(payload["rng"]["numpy"])
        random.setstate(payload["rng"]["python"])
        actual=old.rng_state()
        if any(not old.nested_state_equal(payload["rng"][k],actual[k]) for k in ("cpu","numpy","python")):
            raise AssertionError("available CPU RNG restoration mismatch")
    replay=Replay128.from_state_dict(payload["replay_state"])
    if replay.size!=128 or replay.capacity!=128 or replay.width!=48 or replay.seen!=96000:
        raise ValueError("actual completed three-task Replay128 required")
    if replay.digest()!=manifest["replay_payload_sha256"]: raise ValueError("actual parent replay hash differs")
    if not old.nested_state_equal(replay.state_dict(),payload["replay_state"]):
        raise AssertionError("actual replay arrays/RNG/counters changed")
    growth=grow_storage(model,optimizer,saved)
    rec.emit("actual_parent_restore_and_storage_growth",parent_run=manifest["parent_run"],
             checkpoint_sha256=sha(args.resume),actual_NN_Adam_replay_restored=True,
             available_RNG_streams_exact=True,CUDA_RNG_checked=device.type=="cuda",
             actual_replay_before_growth=replay.statistics(),old_training_repeated=False,
             restore_and_growth_wall_seconds=time.perf_counter()-began,growth=growth,
             historical_costs=manifest["historical_training_reports"],historical_cost_scope="owned earlier runs retained separately")
    return model,optimizer,saved,replay,growth


def checkpoint(model,optimizer,saved,replay,args,manifest,rec,step,complete,costs,growth):
    began=time.perf_counter();path=rec.out/f"{saved.arm}-stage3-step{step}.pt"
    actual_args=dict(vars(saved),data=str(args.data),out=str(args.out),stages=4)
    payload=dict(schema="contextual-primitives-training-v1",args=actual_args,
        metadata_public=dict(n_states=8,n_tokens=16,max_length=128),stage=3,step=step,stage_complete=complete,
        model_state=model.state_dict(),block_checkpoint=model.checkpoint() if old.is_block(saved) else None,
        baseline_spec=None if old.is_block(saved) else model.spec,optimizer=optimizer.state_dict(),
        optimizer_parameter_names=[[n for p in g["params"] for n,q in model.named_parameters() if p is q] for g in optimizer.param_groups],
        parameter_flags={n:p.requires_grad for n,p in model.named_parameters()},rng=old.rng_state(),
        replay_state=replay.state_dict(),source_sha256=source_hashes(manifest),costs=copy.deepcopy(costs),
        stage3_resume_lineage=manifest,growth=growth,formal_Primary=False,
        claim_scope="new fourth-task diagnostic only; old overwritten-first-READ evidence unchanged")
    if path.exists(): raise FileExistsError("checkpoint evidence is append-only")
    pending=path.with_suffix(".pt.partial");torch.save(payload,pending)
    if pending.stat().st_size>MAX_FILE: raise RuntimeError("100MiB checkpoint bound")
    pending.rename(path)
    report=dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,wall_seconds=time.perf_counter()-began)
    rec.emit("actual_fourth_stage_checkpoint",step=step,complete=complete,**report)
    return path,report


def reload_audit(path,model,optimizer,saved,replay,data,device,rec):
    began=time.perf_counter();rng=old.rng_state();was=model.training
    try:
        payload=torch.load(path,map_location="cpu",weights_only=False)
        other,opt,_=old.restore_model(payload,device)
        assert old.tensor_state_equal(model.state_dict(),other.state_dict())
        assert old.nested_state_equal(optimizer.state_dict(),opt.state_dict())
        assert old.nested_state_equal(replay.state_dict(),Replay128.from_state_dict(payload["replay_state"]).state_dict())
        model.eval();other.eval();b,l,active=old.batch(data,np.arange(min(8,len(data["initial"]))),device)
        modes=("soft","native") if old.is_block(saved) else ("soft",)
        with torch.no_grad():
            for mode in modes: assert torch.equal(old.forward(model,b,saved,mode),old.forward(other,b,saved,mode))
        old.synchronize(device)
        result=dict(actual_NN_Adam_replay_exact=True,actual_forward_exact=True,
                    rows=len(b["initial"]),wall_seconds=time.perf_counter()-began,
                    forward_MAC_proxy=sum(2*old.work_proxy(model,saved,len(b["initial"]),l,active,m)["proxy_forward_macs"] for m in modes))
        rec.emit("expanded_checkpoint_reload_audit",**result);return result
    finally: model.train(was);old.restore_rng(rng)


def frequencies(train,data):
    """Ordinary TRAIN label counts and last-four-token lookup; no parsing."""
    began=time.perf_counter();counts=np.bincount(train["targets"],minlength=8);majority=int(counts.argmax());lookup={}
    for word,length,y in zip(train["tokens"],train["lengths"],train["targets"]):
        key=tuple(word[max(0,int(length)-4):int(length)])
        lookup.setdefault(key,np.zeros(8,dtype=np.int64))[y]+=1
    predicted=[]
    for word,length in zip(data["tokens"],data["lengths"]):
        c=lookup.get(tuple(word[max(0,int(length)-4):int(length)]))
        predicted.append(majority if c is None else int(c.argmax()))
    return dict(majority_accuracy=float((data["targets"]==majority).mean()),
                suffix4_lookup_accuracy=float((data["targets"]==np.asarray(predicted)).mean()),
                expected_uniform_chance=.125,train_label_counts=counts.tolist(),lookup_keys=len(lookup),
                TRAIN_fit_label_reads=2*len(train["targets"]),evaluation_label_reads=2*len(data["targets"]),
                wall_seconds=time.perf_counter()-began,exact_FLOPs=None,
                information="TRAIN frequency labels and raw last4 tokens only; no oracle/semantics/parser")


@torch.no_grad()
def predictions(model,data,saved,device,rec):
    result={};model.eval();began=time.perf_counter();macs=0
    for mode in (("soft","native") if old.is_block(saved) else ("soft",)):
        pieces=[]
        for start in range(0,len(data["initial"]),64):
            rec.check();b,l,active=old.batch(data,np.arange(start,min(start+64,len(data["initial"]))),device)
            pieces.append(old.forward(model,b,saved,mode).argmax(-1).cpu().numpy())
            macs+=old.work_proxy(model,saved,len(b["initial"]),l,active,mode)["proxy_forward_macs"]
        result[mode]=np.concatenate(pieces)
    old.synchronize(device)
    return result,dict(wall_seconds=time.perf_counter()-began,forward_MAC_proxy=macs,
                       actual_forward_examples=len(data["initial"])*len(result))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("data","old-data","resume","manifest","out"):p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--self-test",action="store_true");p.add_argument("--device",choices=("cpu","cuda:0"),default="cuda:0")
    p.add_argument("--max-seconds",type=float,default=850)
    args=p.parse_args()
    if args.out.exists() or not 0<args.max_seconds<=(120 if args.self_test else 850):raise ValueError("new output/finite bound required")
    if not args.self_test and args.device!="cuda:0":raise ValueError("actual training requires existing authorized GPU envelope")
    rec=old.Recorder(args.out,args.max_seconds);device=torch.device(args.device)
    model=optimizer=None;completed=0;safe=False;costs={};saved=None;replay=None;growth=None
    try:
        torch.set_num_threads(1)
        if device.type=="cuda":
            if not torch.cuda.is_available():raise RuntimeError("authorized CUDA required")
            torch.cuda.set_per_process_memory_fraction(.3,device)
        manifest=json.loads(args.manifest.read_text());source_hashes(manifest)
        newmeta=json.loads((args.data/"metadata.json").read_text())
        oldmeta,datasets,receipts=read_inputs(args,manifest,newmeta)
        model,optimizer,saved,replay,growth=restore(args,manifest,device,rec,oldmeta)
        if saved.arm!=manifest["arm"] or saved.seed!=0 or saved.states!=8 or saved.lr!=.003:
            raise ValueError("frozen actual training policy differs")
        if old.is_block(saved) and (saved.block_state!="continuous" or saved.block_training!="soft_only"):
            raise ValueError("frozen continuous soft-only parent required")
        # Audit source provenance against actual committed old training streams.
        used={};train_sources={};audit_begin=time.perf_counter()
        for stage in range(3):
            proof=manifest["history_files"][str(stage)];path=args.resume.parent/proof["file"]
            verify_file(path,proof["sha256"])
            with np.load(path,allow_pickle=False) as z:
                if z.files!=["indices"]:raise ValueError("actual used-index proof only")
                matrix=z["indices"].copy()
            if matrix.shape!=(600,64) or matrix.dtype!=np.int64:raise ValueError("actual old common index shape")
            expected=np.random.default_rng(np.random.SeedSequence([0,8,stage,9027])).integers(0,4096,size=(600,64),dtype=np.int64)
            if not np.array_equal(matrix,expected):raise ValueError("actual old common index stream differs")
            used[stage]=matrix if stage==0 else matrix[:,:48]
            train_sources[stage],receipt=load_input(args,manifest,newmeta,oldmeta,stage,"train");receipts.append(receipt)
        provenance=replay.validate_sources(train_sources,used)
        rec.emit("actual_replay_provenance_audit",**provenance,wall_seconds=time.perf_counter()-audit_begin,
                 old_label_reaccesses=128,old_data_source_array_bytes=sum(v.nbytes for d in train_sources.values() for v in d.values()))
        if args.self_test:
            # Actual inherited optimizer/state, not a synthetic bookkeeping fixture.
            path,receipt=checkpoint(model,optimizer,saved,replay,args,manifest,rec,0,False,{},growth)
            audit=reload_audit(path,model,optimizer,saved,replay,datasets[3]["valid"],device,rec)
            # Compare the same old-token function before and after storage growth.
            payload=torch.load(args.resume,map_location="cpu",weights_only=False)
            before,prior_opt,_=old.restore_model(payload,device)
            before.eval();model.eval();b,_,_=old.batch(datasets[0]["valid"],np.arange(8),device)
            with torch.no_grad():
                for mode in (("soft","native") if old.is_block(saved) else ("soft",)):
                    assert torch.equal(old.forward(before,b,saved,mode),old.forward(model,b,saved,mode))
                # Public long extent is exercised with synthetic raw tokens;
                # no held-out file/target is opened during this code check.
                synthetic=dict(initial=b["initial"],tokens=torch.full((8,112),3,dtype=torch.long,device=device),
                               lengths=torch.full((8,),112,dtype=torch.long,device=device))
                for mode in (("soft","native") if old.is_block(saved) else ("soft",)):
                    long_output=old.forward(model,synthetic,saved,mode)
                    assert long_output.shape==(8,8) and bool(torch.isfinite(long_output).all())
            assert growth["parameters_added"]=={"dynamic":16,"static":16,"gru":44,"transformer":32}[saved.arm]
            assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in before.parameters())+growth["parameters_added"]
            control=frequencies(datasets[3]["train"],datasets[3]["valid"])
            assert control["majority_accuracy"]==.125 and control["suffix4_lookup_accuracy"]==.125
            rec.json("self-test.json",dict(status="ACTUAL_OWNED_GROWTH_RELOAD_CPU_CHECK_PASSED",arm=saved.arm,
                growth=growth,reload=audit,old_token_forward_exact_after_growth=True,
                synthetic_length112_forward_finite=True,exact_parameter_growth_checked=True,
                actual_128_row_replay_provenance_checked=True,controls=control,prospective_data_audits=newmeta["audits"],
                no_training_updates_executed=True,formal_Primary=False));return
        rec.emit("configuration",arm=saved.arm,phase=3,public_metadata=dict(n_states=8,n_tokens=16,max_length=128),
            source_sha256=source_hashes(manifest),input_receipts=receipts,growth=growth,
            resources=old.resources(model,optimizer,device,replay),data_creation_receipt=newmeta,
            fixed_updates=600,current_rows_per_update=48,old_rows_per_update=16,
            old_training_updates_repeated=0,formal_Primary=False,
            claim_boundary="new two-live-address development; prior overwritten READ metrics preserved",
            exact_training_FLOPs=None,exact_evaluation_FLOPs=None,
            FLOP_scope="forward MAC proxies and actual time/storage; full forward/backward FLOPs unknown")
        train=datasets[3]["train"]
        indices=np.random.default_rng(np.random.SeedSequence([0,8,3,9027])).integers(0,4096,size=(600,64),dtype=np.int64)
        draws=replay.sample_indices(3,600,16)
        with (rec.out/"stage3-common-indices.npz").open("xb") as stream:np.savez_compressed(stream,indices=indices)
        with (rec.out/"stage3-replay-draws.npz").open("xb") as stream:np.savez_compressed(stream,buffer_indices=draws)
        costs=dict(training_wall_seconds=0.,training_CPU_seconds=0.,CUDA_stream_elapsed_seconds=0.,
            batch_assembly_wall_seconds=0.,training_forward_MAC_proxy=0,SGD_updates=0,
            training_examples=0,current_examples_used_for_training=0,old_examples_used_for_training=0,
            evaluation_wall_seconds=0.,evaluation_CPU_seconds=0.,evaluation_forward_MAC_proxy=0,
            evaluation_forward_examples=0,exact_training_FLOPs=None,exact_evaluation_FLOPs=None,
            compilation_internal_seconds=None,formation_search_FLOPs=None,
            formation_search_scope="no extra search/formation process or configuration selection; inherited model",
            restore_growth=growth,replay_provenance_label_reads=128,replay_maintenance_wall_seconds=0.,
            checkpoint_write_wall_seconds=0.,checkpoint_bytes=0,
            loaded_training_evaluation_array_bytes=sum(v.nbytes for d in datasets.values() for part in d.values() for v in part.values()),
            old_source_audit_array_bytes=sum(v.nbytes for d in train_sources.values() for v in d.values()),
            current_training_label_reads=0,old_replay_training_label_reads=0)
        curves=[]
        rec.emit("stage3_common_schedule",current_indices_sha256=sha(rec.out/"stage3-common-indices.npz"),
            replay_draws_sha256=sha(rec.out/"stage3-replay-draws.npz"),actual_parent_buffer_sha256=replay.digest(),
            fixed_buffer_during_stage=True,no_extra_updates=True)
        for step in range(601):
            rec.check();safe=True
            if step%100==0:
                metrics={}
                for stage in range(4):
                    metrics[stage]={}
                    for split in (("valid",) if stage<3 else ("train","valid")):
                        measured,timing=old.evaluate(model,datasets[stage][split],saved,device,rec)
                        metrics[stage][split]=measured
                        costs["evaluation_wall_seconds"]+=timing["wall_seconds"]
                        costs["evaluation_CPU_seconds"]+=timing["cpu_seconds"]
                        costs["evaluation_forward_MAC_proxy"]+=timing["forward_MAC_proxy"]
                        costs["evaluation_forward_examples"]+=timing["actual_forward_examples"]
                curves.append(dict(step=step,metrics=metrics));rec.emit("TRAIN_VALID_curve",**curves[-1],costs=costs,
                    resources=old.resources(model,optimizer,device,replay))
            if step==600:break
            began=time.perf_counter();mixed=replay.mix(train,indices[step,:48],draws[step]);b,l,active=old.batch(mixed,np.arange(64),device)
            if np.any(replay.arrays["source_stage"][draws[step]]>=3):raise RuntimeError("new/future row entered old buffer")
            costs["batch_assembly_wall_seconds"]+=time.perf_counter()-began
            model.train();old.synchronize(device);began,cpu=time.perf_counter(),time.process_time()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();safe=False;optimizer.zero_grad(set_to_none=True)
            output=old.forward(model,b,saved,"soft");loss=old.loss_value(output,b["targets"],saved)
            if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite terminal training loss")
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not bool(torch.isfinite(norm)):raise FloatingPointError("nonfinite gradient")
            optimizer.step();end.record();old.synchronize(device);safe=True;completed=step+1
            elapsed=time.perf_counter()-began;cpu_elapsed=time.process_time()-cpu;gpu_elapsed=start.elapsed_time(end)/1000.
            proxy=old.work_proxy(model,saved,64,l,active,"soft")["proxy_forward_macs"]
            for key,value in (("training_wall_seconds",elapsed),("training_CPU_seconds",cpu_elapsed),
                ("CUDA_stream_elapsed_seconds",gpu_elapsed),("training_forward_MAC_proxy",proxy),
                ("SGD_updates",1),("training_examples",64),("current_examples_used_for_training",48),
                ("old_examples_used_for_training",16),("current_training_label_reads",48),("old_replay_training_label_reads",16)):
                costs[key]+=value
            rec.emit("actual_update",stage=3,step=completed,execution="soft",terminal_loss=float(loss.detach()),
                batch_accuracy=float((output.argmax(-1)==b["targets"]).float().mean()),preclip_gradient=float(norm),
                forward_MAC_proxy=proxy,active_token_steps=active,padded_length=l,training_wall_seconds=elapsed,
                training_CPU_seconds=cpu_elapsed,CUDA_stream_elapsed_seconds=gpu_elapsed,
                current_training_examples=48,replayed_training_examples=16)
        # Preserve the exact existing128x48 storage budget. TRAIN padded tails
        # are all inactive zeros; trim only these zeros for final reservoir writes.
        began=time.perf_counter();compact=dict(train,tokens=train["tokens"][:,:48])
        if np.any(train["tokens"][:,48:]) or np.any(train["lengths"]>48):raise AssertionError("new TRAIN exceeds old buffer width")
        used[3]=indices[:,:48];train_sources[3]=compact
        consumed=replay.consume(compact,used[3],3,rec.emit);source_audit=replay.validate_sources(train_sources,used)
        costs["replay_maintenance_wall_seconds"]+=time.perf_counter()-began
        costs["replay_maintenance_label_reads"]=consumed["actual_buffer_writes"]+source_audit["checked_buffer_rows"]
        rec.emit("stage3_replay_boundary",consume=consumed,source_audit=source_audit)
        path,cp=checkpoint(model,optimizer,saved,replay,args,manifest,rec,600,True,costs,growth)
        costs["checkpoint_write_wall_seconds"]+=cp["wall_seconds"];costs["checkpoint_bytes"]+=cp["bytes"]
        audit=reload_audit(path,model,optimizer,saved,replay,datasets[3]["valid"],device,rec)
        rec.emit("ALL_TRAINING_FROZEN_BEFORE_TEST",checkpoint=cp,no_later_updates_or_selection=True)
        for param in model.parameters():param.requires_grad_(False)
        tests={};test_receipts=[];prediction_arrays={};controls={};test_costs=[]
        for stage in range(4):
            tests[stage]={}
            for split in SPLITS[2:]:
                rec.check();data,receipt=load_input(args,manifest,newmeta,oldmeta,stage,split);test_receipts.append(receipt)
                tests[stage][split],timing=old.evaluate(model,data,saved,device,rec);test_costs.append(timing)
                if stage==3:
                    controls[split]=frequencies(train,data)
                    pred,pred_cost=predictions(model,data,saved,device,rec);test_costs.append(pred_cost)
                    for mode,values in pred.items():prediction_arrays[split+"_"+mode]=values
        with (rec.out/"stage3-frozen-predictions.npz").open("xb") as stream:np.savez_compressed(stream,**prediction_arrays)
        infer_begin=time.perf_counter();inference=old.inference_report(model,datasets[3]["valid"],saved,device,rec)
        inference_wall=time.perf_counter()-infer_begin
        rec.json("result.json",dict(status="DECLARED_FOURTH_STAGE_DEVELOPMENT_COMPLETED",arm=saved.arm,stage=3,seed=0,
            parent_run=manifest["parent_run"],new_training_updates=completed,old_training_updates_repeated=0,
            final_TRAIN_VALID=curves[-1],curves=curves,frozen_test_metrics=tests,checkpoint=cp,reload=audit,
            resources=old.resources(model,optimizer,device,replay),growth=growth,costs=costs,
            frozen_test_measurement_costs=test_costs,test_input_receipts=test_receipts,
            inference=inference,inference_total_wall_seconds=inference_wall,
            inference_forward_calls_per_mode=23,inference_FLOPs=None,
            ordinary_TRAIN_frequency_controls=controls,offline_generator_costs_and_labels=newmeta,
            predictions_file_sha256=sha(rec.out/"stage3-frozen-predictions.npz"),
            counterfactual_prediction_audit="ROOT joins frozen predictions to excluded generator truth after training; no truth supplied to learner",
            final_replay_state=replay.statistics(),source_sha256=source_hashes(manifest),
            formal_Primary=False,no_test_based_selection=True,
            evidence_boundary="old overwritten-first-READ task/result unchanged; new independent fourth-task diagnostic"))
    except BaseException as error:
        rec.emit("run_failed",error=type(error).__name__,reason=str(error),actual_new_updates_completed=completed,last_update_complete=safe)
        if safe and model is not None and optimizer is not None and saved is not None:
            try:checkpoint(model,optimizer,saved,replay,args,manifest,rec,completed,False,costs,growth)
            except BaseException as saving:rec.emit("interrupted_checkpoint_failed",reason=str(saving))
        rec.json("failure.json",dict(error=type(error).__name__,reason=str(error),actual_new_updates_completed=completed))
        raise
    finally:rec.stream.close()


if __name__=="__main__":main()
