"""Bounded CPU code check of streaming/all-token APIs; no neural training.

Compares actual executed values and gradients, never task success. Root runs
inside its existing CPU envelope. The old reference must be our frozen f67e448
source; no arbitrary downloaded module is executed. No optimizer, network,
subprocess, installation or GPU is used. No import-time model construction.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time

import torch
from torch.nn import functional as F
from continuous_primitive_block import ContinuousPrimitiveBlock

REFERENCE_SHA="f67e448dd393a8f7c7d0e988bab3812cbae21e0ff7b4ffe7691f87213f36b13f"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def same(name,left,right,rows):
    if left.shape!=right.shape:raise AssertionError(name+": shape mismatch")
    maximum=float((left-right).abs().max()) if left.numel() else 0.
    bitwise=torch.equal(left,right)
    if not torch.allclose(left,right,atol=1e-6,rtol=1e-5):raise AssertionError(name+": values differ "+str(maximum))
    rows.append(dict(test=name,bitwise_equal=bitwise,maximum_absolute_error=maximum,
                     checked_elements=left.numel(),atol=1e-6,rtol=1e-5))


def source_check(reference):
    if sha(reference)!=REFERENCE_SHA:raise ValueError("reference must be the exact owned pre-streaming f67 source")
    def methods(path):
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ContinuousPrimitiveBlock")
        return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}
    current=Path(__file__).with_name("continuous_primitive_block.py")
    before,after=methods(reference),methods(current)
    for name,body in before.items():
        if after.get(name)!=body:raise AssertionError("old method was changed: "+name)
    name="prometheus_owned_continuous_pre_streaming"
    spec=importlib.util.spec_from_file_location(name,reference);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module.ContinuousPrimitiveBlock,dict(reference_sha256=REFERENCE_SHA,
        current_sha256=sha(current),unchanged_old_method_count=len(before),all_old_methods_AST_equal=True)


def check_configuration(old_class,routing,slots,out):
    rows=[];spec=dict(n_states=8,n_tokens=9,internal_states=64,n_primitives=4,token_dim=16,
        state_dim=64,memory_dim=64,caller_hidden=32,n_slots=slots,routing=routing,
        memory_slots=8,key_dim=16,primitive_rank=8)
    torch.manual_seed(7103);model=ContinuousPrimitiveBlock(**spec);new_rng=torch.get_rng_state()
    torch.manual_seed(7103);old=old_class(**spec);old_rng=torch.get_rng_state()
    assert torch.equal(new_rng,old_rng),"constructor draw sequence changed"
    assert model.state_dict().keys()==old.state_dict().keys()
    assert all(torch.equal(v,old.state_dict()[n]) for n,v in model.state_dict().items())
    assert model.parameter_count()==old.parameter_count()
    initial=torch.tensor([1,5,2],dtype=torch.long)
    tokens=torch.tensor([[1,2,3,4,5],[2,6,1,0,0],[0,0,0,0,0]],dtype=torch.long)
    lengths=torch.tensor([5,3,0],dtype=torch.long)
    padded=tokens.clone();padded[1,3:]=-999;padded[2,:]=-999
    modes={}
    for mode in ("soft","native"):
        with torch.no_grad():
            legacy=old(initial,padded,lengths,execution=mode,output="logits")
            final=model(initial,padded,lengths,execution=mode,output="logits")
            all_outputs,carry=model.forward_all(initial,padded,lengths,execution=mode,output="logits",return_carry=True)
            same(mode+":old_final_preserved",legacy,final,rows)
            final_from_carry=F.log_softmax(model.readout(carry["h"]),-1)
            same(mode+":all_final_carry_vs_old",legacy,final_from_carry,rows)
            last=all_outputs[torch.arange(3),(lengths-1).clamp_min(0)]
            same(mode+":last_valid_output_vs_old",legacy,last,rows)
            same(mode+":normalized_all_positions",torch.logsumexp(all_outputs,-1),torch.zeros(3,5),rows)
            _,trace=model(initial,padded,lengths,execution=mode,return_traces=True,output="logits")
            same(mode+":final_keys",carry["keys"],trace["memory_keys"][:,-1],rows)
            same(mode+":final_values",carry["values"],trace["memory_values"][:,-1],rows)
            clean=model.forward_all(initial,tokens,lengths,execution=mode,output="logits")
            same(mode+":inactive_padding_ignored",all_outputs,clean,rows)
            for row in range(3):
                if lengths[row]<5:
                    expected=legacy[row].expand(5-int(lengths[row]),-1)
                    same(mode+":inactive_carry_row"+str(row),all_outputs[row,int(lengths[row]):],expected,rows)
            # Explicit carried execution has no access to future input tokens.
            full_lengths=torch.full((3,),5,dtype=torch.long)
            full=model.forward_all(initial,tokens,full_lengths,execution=mode,output="logits")
            sequential=model.initial_carry(initial);parts=[]
            for t in range(5):
                output,sequential=model.step_embedded(sequential,model.token_embedding(tokens[:,t]),execution=mode,output="logits")
                parts.append(output)
            same(mode+":step_carry_vs_full",full,torch.stack(parts,1),rows)
            for name in ("h","keys","values"):
                _,expected_carry=model.forward_all(initial,tokens,full_lengths,execution=mode,return_carry=True)
                same(mode+":step_final_"+name,sequential[name],expected_carry[name],rows)
            future=tokens.clone();future[:,2:]=(future[:,2:]+4)%9
            changed=model.forward_all(initial,future,full_lengths,execution=mode,output="logits")
            same(mode+":future_does_not_change_past",full[:,:2],changed[:,:2],rows)
            prefix=model(initial,tokens[:,:2],torch.full((3,),2,dtype=torch.long),execution=mode,output="logits")
            same(mode+":prefix_terminal_matches_position",full[:,1],prefix,rows)
            empty,empty_carry=model.forward_all(initial,tokens[:,:0],torch.zeros(3,dtype=torch.long),
                execution=mode,return_carry=True)
            assert empty.shape==(3,0,8)
            same(mode+":empty_state",empty_carry["h"],model.encoder(initial),rows)
            modes[mode]=dict(output_shape=list(full.shape),valid_lengths=lengths.tolist(),
                normalized_padding=True,no_future_access=True)
    model.zero_grad(set_to_none=True)
    all_outputs=model.forward_all(initial,tokens,torch.full((3,),5,dtype=torch.long),output="logits")
    targets=(torch.arange(15).reshape(3,5)+2)%8
    loss=F.nll_loss(all_outputs.reshape(-1,8),targets.reshape(-1));loss.backward()
    norms={n:None if p.grad is None else float(p.grad.norm()) for n,p in model.named_parameters()}
    if any(v is not None and not torch.isfinite(torch.tensor(v)) for v in norms.values()):raise AssertionError("nonfinite gradients")
    groups=dict(encoder=["encoder.weight"],caller=[n for n in norms if n.startswith("caller.")],
        primitives=["primitive_down","primitive_up","primitive_bias"],
        memory=[n for n in norms if n.startswith(("read_query.","write_"))]+["initial_memory_keys"])
    totals={g:sum(norms[n] or 0. for n in names) for g,names in groups.items()}
    if any(value<=0 for value in totals.values()):raise AssertionError("missing terminal gradient path "+str(totals))
    assert all(torch.equal(v,old.state_dict()[n]) for n,v in model.state_dict().items()),"gradient check mutated weights"
    payload=model.checkpoint();checkpoint=out/f"{routing}-slots{slots}.pt";torch.save(payload,checkpoint)
    loaded=torch.load(checkpoint,map_location="cpu",weights_only=False)
    restored=ContinuousPrimitiveBlock.from_checkpoint(loaded)
    old_restored=old_class.from_checkpoint(loaded)
    with torch.no_grad():
        restored_all=restored.forward_all(initial,tokens,torch.full((3,),5,dtype=torch.long),output="logits")
        same("actual_checkpoint_all_outputs",all_outputs.detach(),restored_all,rows)
        same("old_source_checkpoint_compatibility",old(initial,tokens,lengths,output="logits"),
             old_restored(initial,tokens,lengths,output="logits"),rows)
    # All-token accounting must charge every readout; constructor/state shapes do not grow.
    terminal=model.resource_report(3,5,active_steps=8)
    sequence=model.sequence_resource_report(3,5,active_steps=8)
    assert sequence["proxy_forward_macs_by_component"]["readout"]==3*5*64*8
    assert sequence["total_parameters"]==terminal["total_parameters"]
    return dict(routing=routing,n_slots=slots,tests=rows,modes=modes,gradient_norms=norms,
        gradient_group_norm_sums=totals,dummy_terminal_loss=float(loss.detach()),
        old_parameters=old.parameter_count(),new_parameters=model.parameter_count(),added_parameters=0,
        initialization_draws_and_tensors_equal=True,state_dict_schema_unchanged=True,
        checkpoint_file=checkpoint.name,checkpoint_sha256=sha(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_compatible_with_old_source=True,sequence_resources=sequence,SGD_updates=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True);parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--max-seconds",type=int,default=30)
    args=parser.parse_args()
    if args.out.exists() or not 1<=args.max_seconds<=60:parser.error("new output and finite CPU time required")
    args.out.mkdir(parents=True);started=time.perf_counter();results=[]
    def alarm(*_):raise TimeoutError("bounded CPU streaming check deadline")
    signal.signal(signal.SIGALRM,alarm);signal.alarm(args.max_seconds);torch.set_num_threads(1)
    try:
        old_class,source=source_check(args.reference)
        for routing in ("dynamic","static","detached"):
            for slots in (1,2):
                row=check_configuration(old_class,routing,slots,args.out);results.append(row)
                with (args.out/f"{routing}-slots{slots}.json").open("x") as stream:json.dump(row,stream,indent=2,allow_nan=False)
        result=dict(status="CPU_CODE_CHECK_PASSED",source=source,results=results,
            wall_seconds=time.perf_counter()-started,training_updates=0,
            scope="finite numerical/code checks; no training success, retention or speed advantage established")
        with (args.out/"result.json").open("x") as stream:json.dump(result,stream,indent=2,allow_nan=False)
        print(json.dumps(dict(status=result["status"],configurations=len(results),wall_seconds=result["wall_seconds"])))
    except BaseException as error:
        with (args.out/"failure.json").open("x") as stream:
            json.dump(dict(error=type(error).__name__,reason=str(error),completed_configurations=len(results),
                wall_seconds=time.perf_counter()-started),stream,indent=2)
        raise
    finally:signal.alarm(0)


if __name__=="__main__":main()
