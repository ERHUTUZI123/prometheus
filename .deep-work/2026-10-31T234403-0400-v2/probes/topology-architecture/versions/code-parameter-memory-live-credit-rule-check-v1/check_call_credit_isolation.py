"""ROOT-RUN bounded CPU checks of OWN b8 CALL-input gradient isolation.

Synthetic tensors only; no optimizer, research datasets, SGD, SSH or network.
Checks exact forward/P/write/proxy equality, isolated and full-terminal
gradients, explicit legacy migration and independent new checkpoint policy.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ORIGINAL_SHA="b8a51a6ca9026b2ebff877493bf9031972a319d097cb3ae2605cc6ad3ea3083f"
CREDIT_SHA="f41cb0dced3d14e11f77f50587573c1c49b6f6aacc5d009ad01d6f47d6fbb499"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path,name,expected):
    if path.is_symlink() or sha(path)!=expected:raise ValueError("exact OWN source required")
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module.ContinuousPrimitiveBlock


def method_nodes(path):
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ContinuousPrimitiveBlock")
    return {n.name:ast.dump(n,include_attributes=False) for n in cls.body if isinstance(n,ast.FunctionDef)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-source",type=Path,required=True);p.add_argument("--credit-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--max-seconds",type=float,default=60)
    args=p.parse_args()
    if args.out.exists() or not 0<args.max_seconds<=60:raise ValueError("new output and finite60s check required")
    import torch
    from torch.nn import functional as F
    args.out.mkdir(parents=True);began=time.perf_counter();deadline=began+args.max_seconds;results=[]
    def check():
        if time.perf_counter()>deadline:raise TimeoutError("finite CPU CALL credit check deadline")
    def equal_state(left,right):return left.keys()==right.keys() and all(torch.equal(v,right[k]) for k,v in left.items())
    try:
        torch.set_num_threads(1);torch.manual_seed(12801)
        Old=load(args.original_source,"owned_b8_call_credit_reference",ORIGINAL_SHA)
        New=load(args.credit_source,"owned_b8_call_credit_control",CREDIT_SHA)
        a,b=method_nodes(args.original_source),method_nodes(args.credit_source)
        changed=[n for n in a if a[n]!=b[n]];assert changed==["_call","checkpoint","from_checkpoint"]
        dims=dict(n_states=8,n_tokens=15,internal_states=16,n_primitives=3,token_dim=4,
                  state_dim=16,memory_dim=16,caller_hidden=8,n_slots=1,memory_slots=3,key_dim=4,primitive_rank=3)
        old=Old(**dims);new=New(**dims);new.load_state_dict(old.state_dict(),strict=True)
        before={n:v.clone() for n,v in new.state_dict().items()}
        assert old.configuration()==new.configuration() and equal_state(old.state_dict(),new.state_dict())
        assert old.parameter_count()==new.parameter_count()
        initial=torch.tensor([0,1],dtype=torch.long)
        tokens=torch.tensor([[1,3,2,9,10,4],[2,4,3,7,0,0]],dtype=torch.long)
        lengths=torch.tensor([6,4],dtype=torch.long)
        forward=[]
        for routing in ("dynamic","static"):
            old.set_routing(routing);new.set_routing(routing)
            for mode in ("soft","straight_through","native"):
                left=old(initial,tokens,lengths,execution=mode,output="logits")
                right=new(initial,tokens,lengths,execution=mode,output="logits")
                assert torch.equal(left,right)
                forward.append(dict(routing=routing,execution=mode,bitwise_equal=True))
        old.set_routing("dynamic");new.set_routing("dynamic");check()
        base_h,base_x,base_r=torch.randn(2,16),torch.randn(2,4),torch.randn(2,16)
        coefficients=torch.tensor([[.2,-.5,.7],[-.4,.3,.6]])
        gradients={}
        for name,model in (("ordinary",old),("credit",new)):
            h=base_h.clone().requires_grad_();x=base_x.clone().requires_grad_();read=base_r.clone().requires_grad_()
            logits=model._call(h,x,read,"soft")[3]
            gradients[name]=torch.autograd.grad((logits*coefficients).sum(),(h,read,x,model.caller[0].weight),allow_unused=True)
        ordinary,credit=gradients["ordinary"],gradients["credit"]
        assert all(g is not None and bool(torch.isfinite(g).all()) and g.norm()>0 for g in ordinary)
        assert credit[0] is None and credit[1] is None
        assert torch.equal(ordinary[2],credit[2]) and torch.equal(ordinary[3],credit[3])
        check();new.zero_grad(set_to_none=True)
        output=new(initial,tokens,lengths,execution="soft",output="logits")
        objective=F.nll_loss(output,torch.tensor([3,5],dtype=torch.long))
        names=("token_embedding.weight","primitive_down","primitive_up","read_query.weight","write_value.weight","caller.0.weight")
        named=dict(new.named_parameters());terminal=torch.autograd.grad(objective,[named[n] for n in names],allow_unused=True)
        assert all(g is not None and bool(torch.isfinite(g).all()) and g.norm()>0 for g in terminal)
        # Other _call feature/residual paths to h/read remain present.
        h=base_h.clone().requires_grad_();x=base_x.clone().requires_grad_();read=base_r.clone().requires_grad_()
        state=new._call(h,x,read,"soft")[0]
        direct=torch.autograd.grad((state*torch.randn_like(state)).sum(),(h,read))
        assert all(bool(torch.isfinite(g).all()) and g.norm()>0 for g in direct)
        # Exact original write function, values AND autograd, under same tensors.
        write_base=(base_h,base_x,base_r,torch.randn(2,3,4),torch.randn(2,3,16));write_data={}
        for name,model in (("ordinary",old),("credit",new)):
            variables=[t.clone().requires_grad_() for t in write_base]
            value=model._write(*variables)
            grads=torch.autograd.grad(value[0].square().sum()+value[1].square().sum(),variables)
            write_data[name]=(value,grads)
        assert all(torch.equal(x,y) for x,y in zip(write_data["ordinary"][0],write_data["credit"][0]))
        assert all(torch.equal(x,y) for x,y in zip(write_data["ordinary"][1],write_data["credit"][1]))
        assert old.resource_report(2,6,active_steps=10)==new.resource_report(2,6,active_steps=10)
        assert equal_state(before,new.state_dict());check()
        migrated=New.from_checkpoint(old.checkpoint());payload=new.checkpoint();restored=New.from_checkpoint(payload)
        assert equal_state(migrated.state_dict(),old.state_dict()) and equal_state(restored.state_dict(),new.state_dict())
        assert payload["version"]=="continuous-primitive-block-call-credit-v1" and payload["call_credit_policy"]=="dynamic-CALL-h-read-stop-gradient-v1"
        for cls,bad in ((Old,payload),(New,dict(payload,call_credit_policy="wrong")),
                        (New,dict(payload,write_update_policy="slot-delta-residual-write-v1")),
                        (New,dict(old.checkpoint(),call_credit_policy="ambiguous"))):
            try:cls.from_checkpoint(bad)
            except ValueError:pass
            else:raise AssertionError("incompatible policy/checkpoint silently accepted")
        for mode in ("soft","native"):
            assert torch.equal(new(initial,tokens,lengths,execution=mode,output="logits"),restored(initial,tokens,lengths,execution=mode,output="logits"))
        path=args.out/"synthetic-credit-checkpoint.pt";torch.save(payload,path)
        loaded=New.from_checkpoint(torch.load(path,map_location="cpu",weights_only=True));assert equal_state(new.state_dict(),loaded.state_dict())
        report=dict(status="ACTUAL_CPU_CALL_CREDIT_CONTRACT_PASSED",source_sha256=dict(original=ORIGINAL_SHA,credit=CREDIT_SHA,checker=sha(Path(__file__))),
            changed_methods=changed,exact_forward_checks=forward,P_unchanged=True,original_write_value_and_gradients_exact=True,
            original_resource_report_exact=True,extra_forward_MAC_proxy=0,extra_parameters=0,
            isolated_CALL_gradient_norms={name:[None if g is None else float(g.norm()) for g in values] for name,values in gradients.items()},
            gradient_coordinate_order=["h","read","x","caller_weight"],
            full_terminal_gradient_norms={n:float(g.norm()) for n,g in zip(names,terminal)},
            full_other_h_read_gradient_norms=[float(g.norm()) for g in direct],
            explicit_legacy_storage_migration=True,old_loader_rejects_new=True,independent_new_policy_rejected_if_wrong=True,
            actual_disk_checkpoint_reload_equal=True,NN_tensors_unchanged=True,optimizer_updates=0,research_labels_read=0,
            synthetic_checkpoint_bytes=path.stat().st_size,wall_seconds=time.perf_counter()-began,
            scope="finite synthetic forward/gradient/storage checks; not evidence of acquisition improvement or independent novelty")
        (args.out/"result.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report))
    except BaseException as error:
        (args.out/"failure.json").write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),wall_seconds=time.perf_counter()-began),indent=2)+"\n")
        raise


if __name__=="__main__":main()
