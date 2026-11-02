"""ROOT-RUN synthetic CPU contracts for fixedG8 state carry, no research fit.

Actual group core and actual wrapper migration function are exercised. Adam
fixtures are explicit synthetic inherited step/moments, not paid training.
Real inherited NN/Adam/RNG/replay are checked separately by the growth selftest.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ORIGINAL_SHA="b8a51a6ca9026b2ebff877493bf9031972a319d097cb3ae2605cc6ad3ea3083f"
GROUP_SHA="d45eed8304095752bbb8a04f69fb5b98b52167b685c81bf5522ef2ed31e3181d"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path,name,expected):
    if path.is_symlink() or sha(path)!=expected:raise ValueError("exact OWN inspected source required")
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module.ContinuousPrimitiveBlock


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-source",type=Path,required=True);p.add_argument("--group-source",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True);p.add_argument("--max-seconds",type=float,default=60)
    args=p.parse_args()
    if args.out.exists() or not 0<args.max_seconds<=60:raise ValueError("new output and finite60s check required")
    import torch
    from torch.nn import functional as F
    import memory_live_group_carry_train as stage
    args.out.mkdir(parents=True);began=time.perf_counter();deadline=began+args.max_seconds
    def check():
        if time.perf_counter()>deadline:raise TimeoutError("finite grouped carry CPU check")
    def equal_state(left,right):return left.keys()==right.keys() and all(torch.equal(value,right[name]) for name,value in left.items())
    def close(left,right):
        if not torch.allclose(left,right,atol=2e-6,rtol=2e-6):raise AssertionError("fixed gate limit differs")
        return float((left-right).abs().max())
    try:
        torch.set_num_threads(1)
        Old=load(args.original_source,"own_b8_group_carry_reference",ORIGINAL_SHA)
        Group=load(args.group_source,"own_fixed8_group_carry",GROUP_SHA)
        dims=dict(n_states=8,n_tokens=15,internal_states=64,n_primitives=4,token_dim=16,
                  state_dim=64,memory_dim=64,caller_hidden=32,n_slots=1,memory_slots=8,key_dim=16,primitive_rank=8)
        torch.manual_seed(1149);original=Old(**dims);original_rng=torch.get_rng_state()
        torch.manual_seed(1149);disabled=Group(**dims,carry_gate_groups=0);disabled_rng=torch.get_rng_state()
        assert not disabled.carry_gate_enabled and disabled.carry_gate_groups==0
        assert equal_state(original.state_dict(),disabled.state_dict()) and torch.equal(original_rng,disabled_rng)
        migrated=Group.from_checkpoint(original.checkpoint())
        assert not migrated.carry_gate_enabled and migrated.caller[2].weight.shape==(4,32)
        assert equal_state(original.state_dict(),migrated.state_dict())
        initial=torch.tensor([0,1],dtype=torch.long);tokens=torch.tensor([[1,3,2,9,10,4],[2,4,3,7,0,0]],dtype=torch.long);lengths=torch.tensor([6,4],dtype=torch.long)
        disabled_forward=[]
        for mode in ("soft","straight_through","native"):
            left=original(initial,tokens,lengths,execution=mode,output="logits")
            right=disabled(initial,tokens,lengths,execution=mode,output="logits")
            assert torch.equal(left,right)
            disabled_forward.append(dict(mode=mode,bitwise_legacy_exact=True))
        # Synthetic old Adam state exercises real migration code without SGD.
        optimizer=torch.optim.Adam(migrated.parameters(),lr=.003)
        for name,param in migrated.named_parameters():
            if name in ("caller.2.weight","caller.2.bias"):
                optimizer.state[param]=dict(step=torch.tensor(1800.),exp_avg=torch.randn_like(param),exp_avg_sq=torch.rand_like(param)+.1)
        before_optimizer=copy.deepcopy(optimizer.state_dict());rng=stage.old.rng_state()
        migration=stage.enable_group_carry(migrated,optimizer)
        assert migration["carry_gate_parameters_added"]==264 and migration["carry_gate_optimizer_bytes_added"]==2112
        assert migration["constant_gate_append_RNG_exact"] and stage.old.nested_state_equal(rng,stage.old.rng_state())
        assert migrated.caller[2].weight.shape==(12,32) and migrated.parameter_count()==original.parameter_count()+264
        assert torch.equal(migrated.caller[2].weight[:4],original.caller[2].weight)
        assert torch.equal(migrated.caller[2].bias[:4],original.caller[2].bias)
        check();torch.manual_seed(641)
        h=torch.randn(2,64);x=torch.randn(2,16);read=torch.randn(2,64)
        limits=[]
        for mode in ("soft","straight_through","native"):
            expected=original._call(h,x,read,mode)
            with torch.no_grad():migrated.caller[2].bias[4:].fill_(1000.)
            one=migrated._call(h,x,read,mode)
            for index in (1,2,3,4):assert torch.equal(expected[index],one[index])
            one_error=close(expected[0],one[0])
            with torch.no_grad():migrated.caller[2].bias[4:].fill_(-1000.)
            zero=migrated._call(h,x,read,mode)
            assert torch.equal(zero[0],h)
            with torch.no_grad():
                migrated.caller[2].bias[4:].fill_(1000.);migrated.caller[2].bias[4+3]=-1000.
            grouped=migrated._call(h,x,read,mode)[0]
            assert torch.equal(grouped[:,24:32],h[:,24:32])
            outside=torch.ones(64,dtype=torch.bool);outside[24:32]=False
            assert torch.equal(grouped[:,outside],one[0][:,outside])
            limits.append(dict(mode=mode,gate0_identity_exact=True,gate1_old_candidate_max_error=one_error,
                               only_group3_eight_coordinates_changed=True,first4_CALL_outputs_exact=True))
        with torch.no_grad():migrated.caller[2].bias[4:].fill_(4.)
        out=migrated(initial,tokens,lengths,execution="soft",output="logits")
        objective=F.nll_loss(out,torch.tensor([3,5],dtype=torch.long))
        names=("caller.2.weight","caller.2.bias","primitive_down","primitive_up","read_query.weight","write_value.weight","token_embedding.weight")
        named=dict(migrated.named_parameters());gradients=torch.autograd.grad(objective,[named[n] for n in names],allow_unused=True)
        assert all(g is not None and bool(torch.isfinite(g).all()) and g.norm()>0 for g in gradients)
        assert gradients[0][4:].norm()>0 and gradients[1][4:].norm()>0
        # Original write and ordinary CALL h/read credit remain active.
        old_ast=ast.parse(args.original_source.read_text());new_ast=ast.parse(args.group_source.read_text())
        def method(tree,name):
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="ContinuousPrimitiveBlock")
            return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(method(old_ast,"_write"),include_attributes=False)==ast.dump(method(new_ast,"_write"),include_attributes=False)
        ah=h.clone().requires_grad_();ar=read.clone().requires_grad_()
        logits=migrated._call(ah,x,ar,"soft")[3]
        ordinary_credit=torch.autograd.grad((logits*torch.randn_like(logits)).sum(),(ah,ar))
        assert all(g.norm()>0 and bool(torch.isfinite(g).all()) for g in ordinary_credit)
        before=original.resource_report(2,5,active_steps=7);after=migrated.resource_report(2,5,active_steps=7)
        assert after["proxy_forward_macs"]-before["proxy_forward_macs"]==7*(256+64)
        assert after["extra_gate_blend_subtractions"]==7*64
        assert before["persistent_example_state_scalars"]==after["persistent_example_state_scalars"]
        assert before["parameter_payload_bytes"]+264*4==after["parameter_payload_bytes"]
        payload=migrated.checkpoint();loaded=Group.from_checkpoint(payload)
        assert payload["version"]=="continuous-primitive-block-group-carry-v1" and payload["configuration"]["carry_gate_enabled"] is True
        assert equal_state(migrated.state_dict(),loaded.state_dict())
        assert torch.equal(out,loaded(initial,tokens,lengths,execution="soft",output="logits"))
        for cls,bad in ((Old,payload),(Group,dict(payload,state_carry_policy="wrong")),
                        (Group,dict(payload,call_credit_policy="mixed-credit")),
                        (Group,dict(payload,write_update_policy="mixed-write"))):
            try:cls.from_checkpoint(bad)
            except ValueError:pass
            else:raise AssertionError("incompatible checkpoint/policy accepted")
        path=args.out/"synthetic-group-carry-checkpoint.pt";torch.save(dict(model=payload,optimizer=optimizer.state_dict(),rng=stage.old.rng_state()),path)
        disk=torch.load(path,map_location="cpu",weights_only=False);disk_model=Group.from_checkpoint(disk["model"])
        disk_optimizer=torch.optim.Adam(disk_model.parameters(),lr=.003);disk_optimizer.load_state_dict(disk["optimizer"])
        assert equal_state(migrated.state_dict(),disk_model.state_dict()) and stage.old.nested_state_equal(optimizer.state_dict(),disk_optimizer.state_dict())
        report=dict(status="ACTUAL_CPU_GROUP8_STATE_CARRY_CONTRACT_PASSED",source_sha256=dict(original=ORIGINAL_SHA,group=GROUP_SHA,checker=sha(Path(__file__)),migration_wrapper=sha(Path(stage.__file__))),
            groups0_disabled_forward=disabled_forward,group_limits=limits,migration=migration,
            first4_CALL_rows_and_logits_exact=True,original_convex_write_AST_exact=True,original_dynamic_CALL_credit_nonzero=True,
            gradient_norms={name:float(g.norm()) for name,g in zip(names,gradients)},new_gate_weight_gradient_norm=float(gradients[0][4:].norm()),
            new_gate_bias_gradient_norm=float(gradients[1][4:].norm()),extra_parameters=264,extra_forward_MAC_proxy_per_call=320,
            extra_blend_subtractions_per_call=64,no_extra_persistent_carry=True,
            numeric_order="h+g_vec*(candidate-h); gate1 limit checked within2e-6 due difference-form rounding",
            gate_activation_preserves_original_function=False,old_loader_rejects_new_policy=True,actual_disk_NN_Adam_reload_equal=True,
            optimizer_updates=0,research_labels_read=0,synthetic_Adam_fixture="only caller tensors step1800 with explicit random moments; no optimizer.step called",
            synthetic_checkpoint_bytes=path.stat().st_size,wall_seconds=time.perf_counter()-began,
            scope="finite synthetic gate/gradient/storage/cost contracts, not acquisition or retention evidence")
        (args.out/"result.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps(report))
    except BaseException as error:
        (args.out/"failure.json").write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),wall_seconds=time.perf_counter()-began),indent=2)+"\n")
        raise


if __name__=="__main__":main()
