"""CPU-only migration/scalar-flow/28-gradient admission, no research SGD."""
import argparse,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import continuous_primitive_block as core
import owned_fcba_core as base
import factorized_masked_graph_fb as fb
import owned_parent_driver as old
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage

PARENT_SHA='9bb47e660e4ea67d0bf1a06e93e418f0c3fa50760fa76ff067d3a0c954a6c40f'
REPLAY_SHA='440344f6cd778e8443081cdeafa95847bf0d23aa825f97590c55436b02a7f470'
COMMON=('continuous_primitive_block.py','owned_fcba_core.py','factorized_masked_graph_fb.py','owned_parent_driver.py','context_replay128.py','memory_live_stage_train.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def snapshot(model,outputs):
    return dict(output=outputs[0].detach().clone(),loss=outputs[1].detach().clone(),grads={n:None if p.grad is None else p.grad.detach().clone()for n,p in model.named_parameters()})
def compare(left,right):
    if not torch.allclose(left['output'],right['output'],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL)or not torch.allclose(left['loss'],right['loss'],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL):raise ValueError('output/loss admission failed')
    result=dict(output_max_abs=float((left['output']-right['output']).abs().max()),loss_abs=float((left['loss']-right['loss']).abs()),gradients={})
    if len(left['grads'])!=28 or left['grads'].keys()!=right['grads'].keys():raise ValueError('exact28 gradient names required')
    for name,l in left['grads'].items():
        r=right['grads'][name]
        if l is None or r is None:raise ValueError('None-vs-zero/unused parameter not admitted: '+name)
        close=bool(torch.allclose(l,r,atol=fb.GRAD_ATOL,rtol=fb.GRAD_RTOL))
        if not close or not bool(torch.isfinite(l).all())or not bool(torch.isfinite(r).all()):raise ValueError('gradient admission failed: '+name)
        result['gradients'][name]=dict(max_abs=float((l-r).abs().max()),allclose=close,reference_norm=float(l.norm()),candidate_norm=float(r.norm()))
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=False);device=torch.device('cpu');torch.set_num_threads(1);started=time.perf_counter();cpu=time.process_time()
    try:
        manifest=json.loads((a.inputs/'manifest.json').read_text())
        if set(manifest['source_sha256'])!=set(COMMON):raise ValueError('exact common6 source map required')
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen source mismatch')
        cp=a.inputs/'resume.pt';cases=a.inputs/'fixed-cases.npz'
        if sha(cp)!=PARENT_SHA or sha(cases)!=manifest['cases_sha256']:raise ValueError('owned checkpoint/cases mismatch')
        payload=torch.load(cp,map_location='cpu',weights_only=False)
        if payload.get('schema')!='factorized-value-development-training-v1'or payload.get('stage')!=2 or payload.get('step')!=600 or payload.get('stage_complete')is not True or payload['args']['anchor']!=0 or payload['source_sha256']!=manifest['parent_source_sha256']:raise ValueError('exact same anchor0stage2 source/NN required')
        args=argparse.Namespace(**payload['args']);core.restore_rng(payload['rng'],device);rng=core.rng(device)
        model,opt,parent,parent_opt=core.migrate_parent(payload,device);migration=core.verify_old_migration(model,opt,parent,parent_opt,old)
        if not old.nested_state_equal(rng,core.rng(device)):raise ValueError('factory changed RNG')
        replay=Replay128.from_state_dict(payload['replay_state'])
        if replay.digest()!=REPLAY_SHA or not old.nested_state_equal(replay.state_dict(),payload['replay_state']):raise ValueError('real replay differs')
        rows=torch.arange(64);tokens=(rows[:,None]*3+torch.arange(8)[None]*5)%15
        b=dict(initial=rows%8,tokens=tokens,lengths=torch.full((64,),8,dtype=torch.long),targets=(rows+1)%8)
        with torch.no_grad():
            for mode in('soft','native'):
                if not torch.equal(model(b['initial'],b['tokens'],b['lengths'],execution=mode,output='logits'),parent(b['initial'],b['tokens'],b['lengths'],execution=mode,output='logits')):raise ValueError('a0 changed old soft/native forward')
            # Pure bridge preserves zero and never receives or resets context.
            if not torch.equal(model.soft_value_bridge(torch.zeros(4,32)),torch.zeros(4,32)):raise ValueError('zero value must stay zero')
        growth=grow_storage(model,opt,args)
        if model.parameter_count()!=13398 or len(list(model.named_parameters()))!=28 or bool(model.value_bridge_alpha.count_nonzero()):raise ValueError('exact one scalar/originalrow growth required')
        before_alpha=float(model.value_bridge_alpha);opt.zero_grad(set_to_none=True);out,_=model.forward_value(b['initial'],b['tokens'],b['lengths']);loss=F.nll_loss(out,b['targets']);loss.backward();gradient=float(model.value_bridge_alpha.grad)
        if not np.isfinite(gradient)or gradient==0:raise ValueError('initial scalar gradient must be finite/nonzero')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        after_alpha=float(model.value_bridge_alpha.detach())
        if not np.isfinite(after_alpha)or after_alpha==before_alpha or float(opt.state[model.value_bridge_alpha]['step'])!=1:raise ValueError('one synthetic update did not move scalar/step')
        synthetic_alpha=after_alpha
        # At the learned nonzero scalar, compare the same synthetic F/B once.
        outputs=fb.original_step(model,b['initial'],b['tokens'],b['lengths'],b['targets'],0);left=snapshot(model,outputs)
        fb.allocate_gradient_buffers(model);outputs=fb.masked_step(model,b['initial'],b['tokens'],b['lengths'],b['targets'],0);right=snapshot(model,outputs);synthetic_compare=compare(left,right)
        state=dict(schema='soft-value-bridge-cpu-synthetic-v1',synthetic_optimizer_steps=1,research_updates=0,block_checkpoint=model.checkpoint(),model_state={k:v.detach().clone()for k,v in model.state_dict().items()},optimizer=opt.state_dict(),optimizer_parameter_names=core.optimizer_names(model,opt),rng=core.rng(device),replay_state=replay.state_dict())
        saved=a.out/'synthetic-step1-checkpoint.pt'
        with saved.open('xb')as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
        old_rng=core.rng(device);loaded=torch.load(saved,map_location='cpu',weights_only=False);other=core.ContinuousPrimitiveBlock.from_checkpoint(loaded['block_checkpoint']);other_opt=core.restore_optimizer(other,loaded['optimizer'],loaded['optimizer_parameter_names']);other_replay=Replay128.from_state_dict(loaded['replay_state'])
        if not old.tensor_state_equal(model.state_dict(),other.state_dict())or not old.nested_state_equal(opt.state_dict(),other_opt.state_dict())or not old.nested_state_equal(replay.state_dict(),other_replay.state_dict()):raise ValueError('real NN/Adam/replay reload differs')
        core.restore_rng(loaded['rng'],device)
        if not old.nested_state_equal(loaded['rng'],core.rng(device)):raise ValueError('actual CPU RNG reload differs')
        with torch.no_grad():
            for mode in('soft','native'):
                if not torch.equal(model(b['initial'],b['tokens'],b['lengths'],execution=mode,output='logits'),other(b['initial'],b['tokens'],b['lengths'],execution=mode,output='logits')):raise ValueError('real reloaded soft/native forward differs')
        try:base.ContinuousPrimitiveBlock.from_checkpoint(loaded['block_checkpoint'])
        except ValueError:pass
        else:raise ValueError('old loader must reject newpolicy')
        core.restore_rng(old_rng,device)
        del model,opt,parent,parent_opt,other,other_opt,other_replay,left,right,loaded,state
        # Discard the synthetic NN. Fixed TRAIN F/B admission restarts from OWN
        # exact stage2 and the original one-row growth, with zero researchSGD.
        core.restore_rng(payload['rng'],device);model,opt,parent,parent_opt=core.migrate_parent(payload,device);growth_admission=grow_storage(model,opt,args);before={k:v.detach().clone()for k,v in model.state_dict().items()};admissions=[]
        with np.load(cases,allow_pickle=False)as z:
            metadata=json.loads(z['metadata_json'].tobytes().decode())
            if metadata.get('TRAIN_only')is not True or metadata.get('schema')!='factorized-fixed-graph-FB-cases-v1':raise ValueError('fixed TRAIN-only cases required')
            for bucket in fb.BUCKETS:
                data={key:torch.as_tensor(z[f'b{bucket}_{key}'])for key in('initial','tokens','lengths','targets')}
                outputs=fb.original_step(model,**data,anchor=0);left=snapshot(model,outputs)
                fb.allocate_gradient_buffers(model);outputs=fb.masked_step(model,**data,anchor=0);right=snapshot(model,outputs);metrics=compare(left,right)
                pointers={n:p.grad.data_ptr()for n,p in model.named_parameters()};outputs=fb.masked_step(model,**data,anchor=0);again=snapshot(model,outputs);repeat=compare(left,again)
                if pointers!={n:p.grad.data_ptr()for n,p in model.named_parameters()}:raise ValueError('CPU stable grad buffers changed')
                raw=dict(reference_output=left['output'].numpy(),candidate_output=right['output'].numpy(),reference_loss=left['loss'].numpy(),candidate_loss=right['loss'].numpy(),gradient_names_json=np.frombuffer(json.dumps(list(left['grads'])).encode(),dtype=np.uint8))
                for index,name in enumerate(left['grads']):raw[f'reference_grad_{index:02d}']=left['grads'][name].numpy();raw[f'candidate_grad_{index:02d}']=right['grads'][name].numpy()
                rawpath=a.out/(f'B64-L{bucket}-28grad-admission.npz')
                with rawpath.open('xb')as f:np.savez_compressed(f,**raw)
                admissions.append(dict(bucket=bucket,metrics=metrics,repeat_zero_accumulation=repeat,raw_file=rawpath.name,raw_sha256=sha(rawpath),gradient_buffers_stable=True,all28_present_finite=True))
        if not old.tensor_state_equal(before,model.state_dict()):raise ValueError('F/B admission changed NN')
        result=dict(status='CPU_SOFT_VALUE_BRIDGE_MIGRATION_GRAPH_ADMISSION_PASS',P=model.parameter_count(),parameters=model.parameter_count(),research_updates=0,SGD_updates=0,synthetic_optimizer_updates=1,source_sha256=manifest['source_sha256'],parent_checkpoint_sha256=PARENT_SHA,cases_sha256=manifest['cases_sha256'],migration=migration,old_soft_native_exact_at_a0=True,factory_RNG_neutral=True,new_scalar_initial_zero=True,new_scalar_Adam_states_steps0=True,initial_scalar_gradient=gradient,synthetic_learned_alpha=synthetic_alpha,synthetic_nonzero_scalar_output_28gradient_admission=synthetic_compare,actual_NN_Adam_CPU_RNG_replay_reload=True,old_loader_rejects_newpolicy=True,all28_gradients_checked=True,admissions=admissions,source_fixed_tolerances=dict(output_atol=fb.OUTPUT_ATOL,output_rtol=fb.OUTPUT_RTOL,gradient_atol=fb.GRAD_ATOL,gradient_rtol=fb.GRAD_RTOL),TRAIN_FB_exposures=64*3*3,synthetic_FB_exposures=64*3,NN_unchanged_by_TRAIN_FB=True,CUDA_RNG_checked=False,CPU_gate_not_CUDA_result=True,growth=growth_admission,NN_bytes=sum(p.numel()*p.element_size()for p in model.parameters()),Adam_bytes=old.tensor_bytes(opt.state_dict()),replay_bytes=sum(x.nbytes for x in replay.arrays.values()),synthetic_checkpoint=dict(file=saved.name,bytes=saved.stat().st_size,sha256=sha(saved)),wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,full_training_FLOPs=None,provider_effective_compute=None,scope='CPU migration/scalarflow/syntheticreload/fixedTRAIN0SGD F/B only; Graph/CUDA numerical/cost result requires actual GPU; discarded syntheticNN never reused')
        with(a.out/'result.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
        print(json.dumps(dict(status=result['status'],parameters=result['P'])))
    except BaseException as error:
        with(a.out/'failure.json').open('x')as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc()),f,indent=2)
        raise
if __name__=='__main__':main()
