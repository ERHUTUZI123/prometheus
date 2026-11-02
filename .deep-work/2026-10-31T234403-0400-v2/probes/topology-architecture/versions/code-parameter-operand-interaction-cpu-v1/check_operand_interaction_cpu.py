"""CPU admission ONLY: OWN stage2 migration, synthetic flow, real reload.

One synthetic Adam update PER mode; second backward checks A/C once B moved.
No task data or score used, no research updates, no CUDA result or training run.
"""
import argparse,copy,hashlib,json,os,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import continuous_primitive_block as base
import operand_interaction_block as interaction
import owned_parent_driver as old
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage
import train_operand_interaction_stage3 as training

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def maximum(t):return float(t.detach().abs().max())if t is not None else None

def CPU_resources(model,opt,replay):
    return dict(parameters=model.parameter_count(),NN_bytes=sum(p.numel()*p.element_size()for p in model.parameters()),
        Adam_bytes=old.tensor_bytes(opt.state_dict()),replay_bytes=sum(v.nbytes for v in replay.arrays.values()),
        cpu_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)),
        CUDA_allocator_bytes=None,full_training_FLOPs=None,memory_scope='CPU diagnostic only; no CUDA calls/counters')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=False);device=torch.device('cpu');torch.set_num_threads(1);started=time.perf_counter();cpu=time.process_time();costs=[]
    try:
        manifest=json.loads((a.inputs/'manifest.json').read_text())
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen source changed')
        checkpoint=a.inputs/'resume.pt'
        if sha(checkpoint)!=training.PARENT_SHA:raise ValueError('OWN exact checkpoint required')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False);training.parent_contract(payload,manifest)
        parent_available={k:payload['rng'][k]for k in('cpu','numpy','python')};initial={};grown={}
        for mode in('product','additive'):
            interaction.restore_rng(payload['rng'],device);before=interaction.rng(device);tick=time.perf_counter()
            model,opt,parent,parent_opt=interaction.migrate_parent(payload,mode,device);migration=interaction.verify_old_migration(model,opt,parent,parent_opt,old)
            if not old.nested_state_equal(before,interaction.rng(device)):raise ValueError('factory changed CPU RNG')
            args=argparse.Namespace(**payload['args']);replay=Replay128.from_state_dict(payload['replay_state'])
            if replay.digest()!=training.REPLAY_SHA or not old.nested_state_equal(replay.state_dict(),payload['replay_state']):raise ValueError('actual replay migration differs')
            row=torch.arange(64,dtype=torch.long);tokens=(row[:,None]*3+torch.arange(8)[None]*5)%15
            b=dict(initial=row%8,tokens=tokens,lengths=torch.full((64,),8,dtype=torch.long),targets=(row+1)%8)
            with torch.no_grad():
                for execution in('soft','native'):
                    oldout=parent(b['initial'],b['tokens'],b['lengths'],execution=execution,output='logits')
                    newout=model(b['initial'],b['tokens'],b['lengths'],execution=execution,output='logits')
                    if not torch.equal(oldout,newout):raise ValueError('Bzero changed original forward')
            initial[mode]={name:getattr(model,name).detach().clone()for name in interaction.NEW_NAMES}
            growth=grow_storage(model,opt,args)
            if model.parameter_count()!=13973 or bool(model.interaction_B.count_nonzero()):raise ValueError('exact13973P after original row growth')
            grown[mode]=model.token_embedding.weight[15].detach().clone()
            initial_steps={name:float(opt.state[p]['step'])for name,p in model.named_parameters()}
            if any(initial_steps[name]!=0 for name in interaction.NEW_NAMES):raise ValueError('new states must begin step0')
            model.train();opt.zero_grad(set_to_none=True);out,_=model.forward_value(b['initial'],b['tokens'],b['lengths']);loss=F.nll_loss(out,b['targets']);loss.backward()
            first_grad={name:maximum(getattr(model,name).grad)for name in interaction.NEW_NAMES}
            if first_grad['interaction_B']is None or first_grad['interaction_B']<=0 or not all(v is not None and np.isfinite(v)for v in first_grad.values()):raise ValueError('initial B gradient must be finite/nonzero')
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if not bool(model.interaction_B.count_nonzero()):raise ValueError('one synthetic step did not update B')
            if any(float(opt.state[p]['step'])!=initial_steps[name]+1 for name,p in model.named_parameters()):raise ValueError('per-name Adam steps differ after one synthetic update')
            opt.zero_grad(set_to_none=True);out,_=model.forward_value(b['initial'],b['tokens'],b['lengths']);second_loss=F.nll_loss(out,b['targets']);second_loss.backward()
            second_grad={name:maximum(getattr(model,name).grad)for name in interaction.NEW_NAMES}
            if any(second_grad[name]is None or second_grad[name]<=0 or not np.isfinite(second_grad[name])for name in('interaction_A','interaction_C')):raise ValueError('A/C gradients after ONE synthetic Adam update must be finite/nonzero')
            p=a.out/(mode+'-synthetic-step1.pt');state=dict(schema='operand-interaction-cpu-synthetic-v1',mode=mode,synthetic_optimizer_steps=1,research_updates=0,block_checkpoint=model.checkpoint(),model_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},optimizer=opt.state_dict(),optimizer_parameter_names=interaction.optimizer_names(model,opt),rng=interaction.rng(device),replay_state=replay.state_dict())
            save_tick=time.perf_counter()
            with p.open('xb')as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
            saved=interaction.rng(device);loaded=torch.load(p,map_location='cpu',weights_only=False);other=interaction.OperandInteractionBlock.from_checkpoint(loaded['block_checkpoint']);other_opt=interaction.restore_optimizer(other,loaded['optimizer'],loaded['optimizer_parameter_names']);other_replay=Replay128.from_state_dict(loaded['replay_state'])
            if not old.tensor_state_equal(model.state_dict(),other.state_dict())or not old.nested_state_equal(opt.state_dict(),other_opt.state_dict())or not old.nested_state_equal(replay.state_dict(),other_replay.state_dict()):raise ValueError('actual synthetic NN/Adam/replay reload differs')
            interaction.restore_rng(loaded['rng'],device)
            if not old.nested_state_equal(interaction.rng(device),loaded['rng']):raise ValueError('CPU RNG reload differs')
            with torch.no_grad():
                for execution in('soft','native'):
                    if not torch.equal(model(b['initial'],b['tokens'],b['lengths'],execution=execution,output='logits'),other(b['initial'],b['tokens'],b['lengths'],execution=execution,output='logits')):raise ValueError('actual reloaded synthetic forward differs')
            for corrupt in (False,True):
                cp=copy.deepcopy(loaded['block_checkpoint'])
                if corrupt:cp['organization_policy']='wrong'
                if corrupt:
                    try:interaction.OperandInteractionBlock.from_checkpoint(cp)
                    except ValueError:pass
                    else:raise ValueError('wrong policy was not rejected')
                else:
                    try:base.ContinuousPrimitiveBlock.from_checkpoint(cp)
                    except ValueError:pass
                    else:raise ValueError('old loader silently reinterpreted branch')
            interaction.restore_rng(saved,device)
            costs.append(dict(mode=mode,parameters_before_growth=13957,parameters_after=13973,extra_parameters=576,migration=migration,growth=growth,initial_branch_sha256={name:training.tensor_sha(value)for name,value in initial[mode].items()},CPU_grown_token_row_sha256=training.tensor_sha(grown[mode]),first_backward_gradient_max_abs=first_grad,after_one_synthetic_update_gradient_max_abs=second_grad,synthetic_optimizer_steps=1,second_backward_optimizer_steps=0,initial_new_steps_all_zero=True,all_Adam_steps_increment_once=True,actual_NN_Adam_CPU_RNG_replay_reload=True,reload_forward_equal_soft_native=True,old_loader_rejects=True,wrong_policy_rejects=True,checkpoint=dict(file=p.name,bytes=p.stat().st_size,sha256=sha(p),save_reload_wall_seconds=time.perf_counter()-save_tick),whole_mode_wall_seconds=time.perf_counter()-tick,resources=CPU_resources(model,opt,replay),full_training_FLOPs=None))
            del model,opt,parent,parent_opt,other,other_opt,other_replay
        if any(not torch.equal(initial['product'][name],initial['additive'][name])for name in interaction.NEW_NAMES)or not torch.equal(grown['product'],grown['additive']):raise ValueError('modes need identical initial A/C/B/token row')
        result=dict(status='CPU_OPERAND_INTERACTION_MIGRATION_PASS',research_updates=0,synthetic_optimizer_updates=2,two_fixed_modes=True,source_sha256=manifest['source_sha256'],parent_checkpoint_sha256=training.PARENT_SHA,shared_A_C_B_initial_exact=True,shared_CPU_token_growth_exact=True,CPU_gate_not_CUDA_admission=True,CUDA_RNG_checked=False,new_states_steps0=True,results=costs,wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,provider_effective_compute=None,full_training_FLOPs=None,scope='synthetic CPU migration/grad-flow/reload only; no task labels/scores/training, not reused as GPU initial NN')
        with(a.out/'result.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
        print(json.dumps(dict(status=result['status'],parameters=13973)))
    except BaseException as error:
        with(a.out/'failure.json').open('x')as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc(),partial=costs),f,indent=2)
        raise
if __name__=='__main__':main()
