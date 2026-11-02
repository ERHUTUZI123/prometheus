"""ROOT-RUN CPU autograd/compatibility checks; no optimizer or task training.

Compare owned prechange8deca source with current detached-routing extension.
This checks specified gradients and forward compatibility, not the scientific
hypothesis about the observed audio optimization gap. No import-time execution.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import sys
import time
import traceback

import torch


ORIGINAL_SHA="8deca750f622c1e238d87219ef9e00c7d32d186214ac64896d428ba1ae6ea12d"


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def main():
    p=argparse.ArgumentParser();p.add_argument('--original-source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--max-seconds',type=float,default=30)
    args=p.parse_args()
    if not 1<=args.max_seconds<=60:p.error('bounded CPUcheck <=60seconds')
    args.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();report={};torch.set_num_threads(1)
    def timed_out(_s,_f):raise TimeoutError('detached CALL autograd check bound')
    signal.signal(signal.SIGALRM,timed_out);signal.setitimer(signal.ITIMER_REAL,args.max_seconds)
    try:
        if digest(args.original_source)!=ORIGINAL_SHA:raise ValueError('exact owned prechange8deca source required')
        current_path=Path(__file__).with_name('continuous_primitive_block.py')
        old=load(args.original_source,'_owned_pre_detached_core');new=load(current_path,'_owned_detached_core')
        kwargs=dict(n_states=8,n_tokens=5,internal_states=16,n_primitives=3,token_dim=4,
            state_dim=16,memory_dim=16,caller_hidden=8,n_slots=1,memory_slots=2,key_dim=4,primitive_rank=3)
        torch.manual_seed(12801);reference=old.ContinuousPrimitiveBlock(**kwargs)
        models={name:new.ContinuousPrimitiveBlock(**kwargs,routing=name) for name in ('dynamic','static','detached')}
        for model in models.values():model.load_state_dict(reference.state_dict(),strict=True);model.eval()
        parameter_names={n:tuple(v.shape) for n,v in reference.named_parameters()}
        identical_parameters=all({n:tuple(v.shape) for n,v in m.named_parameters()}==parameter_names for m in models.values())
        assert identical_parameters
        initial=torch.tensor([0,1],dtype=torch.long);tokens=torch.tensor([[1,3,2],[2,4,0]],dtype=torch.long);lengths=torch.tensor([3,2],dtype=torch.long)
        embedded=torch.randn(2,3,4);compatibility=[]
        for routing in ('dynamic','static'):
            reference.set_routing(routing)
            for mode in ('soft','straight_through','native'):
                for interface in ('tokens','embedded'):
                    if interface=='tokens':
                        left=reference(initial,tokens,lengths,execution=mode,output='logits');right=models[routing](initial,tokens,lengths,execution=mode,output='logits')
                    else:
                        left=reference.forward_embedded(initial,embedded,lengths,execution=mode,output='logits');right=models[routing].forward_embedded(initial,embedded,lengths,execution=mode,output='logits')
                    equal=bool(torch.equal(left,right));assert equal
                    compatibility.append(dict(routing=routing,execution=mode,interface=interface,bitwise_equal=equal))
        detached_forward=[]
        for mode in ('soft','straight_through','native'):
            for interface in ('tokens','embedded'):
                if interface=='tokens':
                    left=models['dynamic'](initial,tokens,lengths,execution=mode,output='logits');right=models['detached'](initial,tokens,lengths,execution=mode,output='logits')
                else:
                    left=models['dynamic'].forward_embedded(initial,embedded,lengths,execution=mode,output='logits');right=models['detached'].forward_embedded(initial,embedded,lengths,execution=mode,output='logits')
                equal=bool(torch.equal(left,right));assert equal
                detached_forward.append(dict(execution=mode,interface=interface,bitwise_dynamic_equal=equal))
        # Isolate CALL logits from direct primitive/residual paths. Detached must
        # remove exactly h/read input gradients, leaving x and caller gradients.
        base_h=torch.randn(2,16);base_x=torch.randn(2,4);base_r=torch.randn(2,16)
        coefficients=torch.tensor([[.2,-.5,.7],[-.4,.3,.6]])
        caller_gradients={};raw_gradients={}
        for routing,model in models.items():
            h=base_h.clone().requires_grad_();x=base_x.clone().requires_grad_();r=base_r.clone().requires_grad_()
            logits=model._call(h,x,r,'soft')[3]
            values=torch.autograd.grad((logits*coefficients).sum(),(h,r,x,model.caller[0].weight),allow_unused=True)
            raw_gradients[routing]=values
            caller_gradients[routing]={name:None if g is None else float(g.norm()) for name,g in zip(('h','read','x','caller_weight'),values)}
        dg,sg,cg=raw_gradients['dynamic'],raw_gradients['static'],raw_gradients['detached']
        assert dg[0] is not None and dg[0].norm()>0 and dg[1] is not None and dg[1].norm()>0
        assert cg[0] is None and cg[1] is None and sg[0] is None and sg[1] is None
        assert cg[2] is not None and cg[2].norm()>0 and cg[3] is not None and cg[3].norm()>0
        assert torch.equal(dg[2],cg[2]) and torch.equal(dg[3],cg[3])
        # Full embedded-terminal objective still trains the external input and
        # actual primitive/read/write/caller parameters. No optimizer is run.
        full_gradients={};all_gradients={}
        for routing in ('dynamic','detached'):
            model=models[routing];value=embedded.clone().requires_grad_()
            out=model.forward_embedded(initial,value,lengths,execution='soft',output='logits')
            objective=-(out[0,1]+out[1,4])/2
            requested=(value,model.primitive_down,model.primitive_up,model.read_query.weight,model.write_value.weight,model.caller[0].weight)
            gradients=torch.autograd.grad(objective,requested,allow_unused=True)
            assert all(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in gradients)
            full_gradients[routing]={n:float(g.norm()) for n,g in zip(('embedded_input','primitive_down','primitive_up','read_query','write_value','caller_weight'),gradients)}
            all_gradients[routing]=gradients
        input_gradient_difference=float((all_gradients['dynamic'][0]-all_gradients['detached'][0]).abs().max())
        assert input_gradient_difference>0
        payload=models['detached'].checkpoint();restored=new.ContinuousPrimitiveBlock.from_checkpoint(payload)
        assert restored.routing=='detached'
        assert all(torch.equal(v,restored.state_dict()[n]) for n,v in models['detached'].state_dict().items())
        assert torch.equal(models['detached'].forward_embedded(initial,embedded,lengths,execution='native',output='logits'),restored.forward_embedded(initial,embedded,lengths,execution='native',output='logits'))
        report=dict(status='passed_actual_CPU_autograd_and_forward_checks',old_source_sha256=ORIGINAL_SHA,current_source_sha256=digest(current_path),
            checker_sha256=digest(Path(__file__)),parameters=reference.parameter_count(),same_parameter_names_shapes=identical_parameters,
            unchanged_dynamic_static=compatibility,detached_same_forward=detached_forward,
            isolated_caller_gradient_norms=caller_gradients,full_embedded_terminal_gradient_norms=full_gradients,
            actual_input_gradient_difference=input_gradient_difference,actual_detached_checkpoint_restore=True,
            optimizer_updates=0,training_examples=0,scope='CPU unit/autograd evidence only, not audio optimization explanation or performance experiment',wall_seconds=time.perf_counter()-started)
        code=0
    except BaseException as error:
        report=dict(status='failed_actual_check',error=f'{type(error).__name__}: {error}',wall_seconds=time.perf_counter()-started)
        (args.out/'exception.txt').write_text(traceback.format_exc());code=1
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    with (args.out/'summary.json').open('x') as stream:json.dump(report,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(report));raise SystemExit(code)


if __name__=='__main__':main()
