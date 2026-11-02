"""Fresh paired posterior/raw CPU rule/API/27gradient/checkpoint admission.

No task data, teacher or historical NN. One discarded synthetic Adam update
PER mode, no researchSGD; GPUfreshNNs must be initialized independently seed0.
"""
import argparse,copy,hashlib,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import continuous_primitive_block as core
import owned_fcba_core as original
import posterior_value_masked_graph_fb as fb
import owned_parent_driver as old
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage

COMMON=('continuous_primitive_block.py','owned_fcba_core.py','posterior_value_masked_graph_fb.py','owned_parent_driver.py','context_replay128.py','memory_live_stage_train.py')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rng():return dict(cpu=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
def restore(s):torch.set_rng_state(s['cpu']);np.random.set_state(s['numpy']);random.setstate(s['python'])
def state_sha(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().contiguous().numpy().tobytes())
    return h.hexdigest()
def snapshot(model,output,loss):return dict(output=output.detach().clone(),loss=loss.detach().clone(),grads={n:None if p.grad is None else p.grad.detach().clone()for n,p in model.named_parameters()})
def original_step(model,b):
    model.zero_grad(set_to_none=True);loss,out=fb.objective(model,fb.original_forward_value,**b);loss.backward();return out.detach(),loss.detach()
def compare(left,right):
    if not torch.allclose(left['output'],right['output'],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL)or not torch.allclose(left['loss'],right['loss'],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL):raise ValueError('output/NLL numeric admission failed')
    if len(left['grads'])!=27 or left['grads'].keys()!=right['grads'].keys():raise ValueError('exact27 original NN names required')
    result=dict(output_max_abs=float((left['output']-right['output']).abs().max()),loss_abs=float((left['loss']-right['loss']).abs()),gradients={})
    for name,l in left['grads'].items():
        r=right['grads'][name]
        if l is None or r is None or not bool(torch.isfinite(l).all())or not bool(torch.isfinite(r).all()):raise ValueError('missing/nonfinite27gradient: '+name)
        close=bool(torch.allclose(l,r,atol=fb.GRAD_ATOL,rtol=fb.GRAD_RTOL))
        if not close:raise ValueError('27gradient numeric admission failed: '+name)
        result['gradients'][name]=dict(max_abs=float((l-r).abs().max()),allclose=True,norm=float(l.norm()))
    return result

def api_checks(model):
    initial=torch.arange(8);tokens=(torch.arange(8)[:,None]*3+torch.arange(6)[None]*5)%model.n_tokens;lengths=torch.tensor([0,1,2,3,4,5,6,6]);basis=model.encode_state(initial)
    with torch.no_grad():
        empty=model(initial,tokens[:,:0],torch.zeros(8,dtype=torch.long),output='logits');expected_empty=F.log_softmax(model.readout(basis),-1)
        if not torch.equal(empty,expected_empty)or model.forward_all(initial,tokens[:,:0],torch.zeros(8,dtype=torch.long)).shape!=(8,0,8):raise ValueError('explicit empty ordinary-readout cache semantics failed')
        init=model.initial_carry(initial)
        if sum(v.numel()for v in init.values())!=8*248 or init['values'].shape!=(8,8,8)or bool(init['values'].count_nonzero()):raise ValueError('exact248 carry/zero posterior memory required')
        for mode in('soft','native','straight_through'):
            out,carry=model.forward_value(initial,tokens,lengths,execution=mode)
            if not torch.equal(out,carry['logp'])or not torch.equal(model(initial,tokens,lengths,execution=mode,output='logits'),out)or not torch.equal(model(initial,tokens,lengths,execution=mode,output='probabilities'),out.exp()):raise ValueError('public/cache/logits/probability API failed')
            embedded=model.forward_embedded(basis,model.token_embedding(tokens),lengths,execution=mode,output='logits')
            if not torch.equal(out,embedded):raise ValueError('embedded API must emit cache')
            allout,allcarry=model.forward_all(initial,tokens,lengths,execution=mode,return_carry=True)
            if not torch.equal(out,allcarry['logp']):raise ValueError('alltoken finalcarry differs')
            state=model.initial_carry(initial);history=[]
            for step in range(tokens.shape[1]):
                active=step<lengths;ids=active.nonzero(as_tuple=False).flatten()
                if len(ids):
                    selected={k:v.index_select(0,ids)for k,v in state.items()};raw=model.token_embedding(tokens[:,step].index_select(0,ids));value,new,trace=model.step_embedded(selected,raw,execution=mode,output='logits')
                    expected=F.log_softmax(model.readout(trace['candidate_h']),-1)
                    if not torch.equal(value,new['logp'])or not torch.equal(value,expected):raise ValueError('candidate decoded once/cache not emitted')
                    posterior=value.exp();mass=new['values'].sum(-1)
                    if float(posterior.min())<0 or float((posterior.sum(-1)-1).abs().max())>2e-6 or float(new['values'].min())< -2e-6 or float(mass.max())>1+2e-6:raise ValueError('posterior/subprobability memory constraint failed')
                    read_p=torch.einsum('bm,bms->bs',trace['read_attention'],selected['values']);read=read_p@basis
                    if not torch.allclose(read,trace['read'],atol=1e-6,rtol=1e-6):raise ValueError('READ must map posterior memory through learnedE')
                    amount=(trace['write_gate']*trace['write_addresses'])[:,:,None];expected_V=selected['values']+amount*(posterior[:,None]-selected['values'])
                    if not torch.equal(new['values'],expected_V):raise ValueError('WRITE must store p, no hardcoded state/context reset')
                    if model.value_projection=='posterior':
                        if not torch.equal(new['h'],trace['projected_h']):raise ValueError('posterior h must be pE')
                        distance=torch.linalg.vector_norm(new['h'][:,None]-basis[None],dim=-1);bound=2*32**.5*(1-posterior)
                        if float((distance-bound).max())>2e-5 or float(torch.linalg.vector_norm(new['h'],dim=-1).max())>32**.5+2e-5:raise ValueError('posterior-only convex value bound failed')
                    elif not torch.equal(new['h'],trace['candidate_h'])or trace['projected_h']is not None:raise ValueError('raw ablation must skip projection and retain candidateh')
                    state={k:v.index_copy(0,ids,new[k])for k,v in state.items()}
                history.append(state['logp'])
            if not torch.equal(torch.stack(history,1),allout)or not torch.equal(state['logp'],out):raise ValueError('step/ragged/alltoken cache API mismatch')
        changed=tokens.clone();changed[:,3:]=(changed[:,3:]+1)%model.n_tokens
        base_all=model.forward_all(initial,tokens,torch.full((8,),6));other_all=model.forward_all(initial,changed,torch.full((8,),6))
        if not torch.equal(base_all[:,:3],other_all[:,:3]):raise ValueError('alltoken causality failed')
    return dict(cache_output_api_checks=True,ordinary_empty_cache_checked=True,soft_native_straight_through_ragged_embedded_alltoken_checked=True,causality_checked=True,logical_carry_scalars=248,Vp_subprobability_verified=True,posterior_h_convex_bound_verified=model.value_projection=='posterior',raw_h_bound_not_assumed=model.value_projection=='raw',no_second_decode=True)

def numeric_case(model,b,outdir,leaf):
    out,loss=original_step(model,b);left=snapshot(model,out,loss)
    fb.allocate_gradient_buffers(model);out,loss=fb.masked_step(model,**b);right=snapshot(model,out,loss);metrics=compare(left,right);pointers={n:p.grad.data_ptr()for n,p in model.named_parameters()}
    out,loss=fb.masked_step(model,**b);repeat=compare(left,snapshot(model,out,loss))
    if pointers!={n:p.grad.data_ptr()for n,p in model.named_parameters()}:raise ValueError('stable CPU gradient buffers changed')
    arrays=dict(reference_output=left['output'].numpy(),candidate_output=right['output'].numpy(),reference_loss=left['loss'].numpy(),candidate_loss=right['loss'].numpy(),gradient_names_json=np.frombuffer(json.dumps(list(left['grads'])).encode(),dtype=np.uint8))
    for i,name in enumerate(left['grads']):arrays[f'reference_grad_{i:02d}']=left['grads'][name].numpy();arrays[f'candidate_grad_{i:02d}']=right['grads'][name].numpy()
    path=outdir/leaf
    with path.open('xb')as f:np.savez_compressed(f,**arrays)
    return dict(metrics=metrics,repeat_zero_accumulation=repeat,raw_file=leaf,raw_sha256=sha(path),all27_present_finite=True),left

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter();cpu=time.process_time();results={};initial_weights={}
    try:
        manifest=json.loads((a.inputs/'manifest.json').read_text())
        if set(manifest['source_sha256'])!=set(COMMON):raise ValueError('exact common6 source map required')
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen source changed')
        for projection in('posterior','raw'):
            torch.manual_seed(0);np.random.seed(0);random.seed(0);initial_rng=rng();model=core.ContinuousPrimitiveBlock(8,15,value_projection=projection);after_init=rng();initial_weights[projection]=state_sha(model)
            restore(initial_rng);old_model=original.ContinuousPrimitiveBlock(8,15)
            if not old.tensor_state_equal(model.state_dict(),old_model.state_dict())or not old.nested_state_equal(after_init,rng()):raise ValueError('same original27 NN init/names/shapes/RNG draws required')
            del old_model;restore(after_init)
            if model.parameter_count()!=13381 or len(list(model.named_parameters()))!=27:raise ValueError('exact P15=13381 and27 tensors')
            model_initial={n:v.detach().clone()for n,v in model.state_dict().items()};api15=api_checks(model)
            opt=torch.optim.Adam(model.parameters(),lr=.003);row=torch.arange(64);tokens=(row[:,None]*3+torch.arange(16)[None]*5)%15;lengths=row%17;lengths[-1]=16
            b=dict(initial=row%8,tokens=tokens,lengths=lengths,targets=(row+3)%8)
            numeric16,golden=numeric_case(model,b,a.out,projection+'-B64-L16-27grad.npz')
            selected=('encoder.weight','initial_memory_keys','controller.weight_hh','primitive_down','primitive_up','write_query.weight','write_key.weight','write_gate.weight','readout.weight')
            if any(golden['grads'][n]is None or float(golden['grads'][n].norm())==0 for n in selected):raise ValueError('value/controller/memory/decoder gradient flow must be nonzero')
            if not old.tensor_state_equal(model_initial,model.state_dict()):raise ValueError('0SGD numeric/API admission changed NN')
            # Exactly one diagnostic NLL/Adam step, on synthetic data only.
            opt.zero_grad(set_to_none=True);loss,out=fb.objective(model,fb.original_forward_value,**b)
            if loss.ndim!=0 or not bool(torch.isfinite(loss)):raise ValueError('actual NLL scalar must be finite')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if not all(float(opt.state[p]['step'])==1 for p in model.parameters()):raise ValueError('actual27 Adam states/steps required')
            args=argparse.Namespace(arm='dynamic',block_state='continuous');probe=dict(initial=torch.tensor([0,3]),tokens=torch.tensor([[1,3],[2,3]]),lengths=torch.tensor([2,2]))
            with torch.no_grad():before=model(**probe,output='logits').clone()
            growth=grow_storage(model,opt,args)
            with torch.no_grad():after=model(**probe,output='logits')
            if model.parameter_count()!=13397 or not torch.equal(before,after)or model.value_projection!=projection:raise ValueError('originalrowgrowth preserves mode/oldtoken execution/P16')
            api16=api_checks(model);tokens48=(row[:,None]*7+torch.arange(48)[None]*3)%16;lengths48=row%49;lengths48[-1]=48
            b48=dict(initial=row%8,tokens=tokens48,lengths=lengths48,targets=(row+5)%8);numeric48,_=numeric_case(model,b48,a.out,projection+'-B64-L48-27grad.npz')
            replay=Replay128(128,48,0,8);raw_data={k:v.numpy()for k,v in b48.items()};used=np.arange(64,dtype=np.int64)[None]
            paths=[]
            for complete in(False,True):
                if complete:replay.consume(raw_data,used,0);replay.validate_sources({0:raw_data},{0:used})
                path=a.out/(projection+('-before-maintenance.pt'if not complete else'-synthetic-step1.pt'));saved_rng=rng();payload=dict(schema='posterior-value-CPU-synthetic-v1',synthetic_optimizer_steps=1,research_updates=0,maintenance_complete=complete,value_projection=projection,block_checkpoint=model.checkpoint(),model_state=model.state_dict(),optimizer=opt.state_dict(),rng=saved_rng,replay=replay.state_dict())
                with path.open('xb')as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
                loaded=torch.load(path,map_location='cpu',weights_only=False);other=core.ContinuousPrimitiveBlock.from_checkpoint(loaded['block_checkpoint'],expected_value_projection=projection);other_opt=torch.optim.Adam(other.parameters(),lr=.003);other_opt.load_state_dict(loaded['optimizer']);other_replay=Replay128.from_state_dict(loaded['replay'])
                if not old.tensor_state_equal(model.state_dict(),other.state_dict())or not old.nested_state_equal(opt.state_dict(),other_opt.state_dict())or not old.nested_state_equal(replay.state_dict(),other_replay.state_dict()):raise ValueError('real NN/Adam/replay reload failed')
                restore(loaded['rng'])
                if not old.nested_state_equal(rng(),loaded['rng']):raise ValueError('actual CPU RNG reload failed')
                with torch.no_grad():
                    for execution in('soft','native'):
                        if not torch.equal(model(**probe,execution=execution,output='logits'),other(**probe,execution=execution,output='logits')):raise ValueError('actual reloaded cache execution differs')
                for target in('old','opposite','wrongpolicy'):
                    cp=copy.deepcopy(loaded['block_checkpoint'])
                    try:
                        if target=='old':original.ContinuousPrimitiveBlock.from_checkpoint(cp)
                        elif target=='opposite':core.ContinuousPrimitiveBlock.from_checkpoint(cp,expected_value_projection='raw'if projection=='posterior'else'posterior')
                        else:cp['organization_policy']='wrong';core.ContinuousPrimitiveBlock.from_checkpoint(cp)
                    except ValueError:pass
                    else:raise ValueError('old/crossmode/wrongpolicy checkpoint was accepted')
                paths.append(dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)));restore(saved_rng)
                del other,other_opt,other_replay,loaded,payload
            results[projection]=dict(P15=13381,P16=model.parameter_count(),initial27_NN_sha256=initial_weights[projection],only27_original_NN_parameters=True,fresh_seed0=True,old_initializer_RNG_exact=True,logical_carry_scalars=248,API15=api15,API16=api16,numeric16=numeric16,numeric48=numeric48,all27_gradients_checked=True,selected_state_memory_parameter_gradient_norms={n:float(golden['grads'][n].norm())for n in selected},synthetic_optimizer_steps=1,synthetic_loss=float(loss.detach()),synthetic_preclip_gradient=float(norm),actual_NN_Adam_CPU_RNG_replay_reload=True,old_crossmode_wrongpolicy_rejected=True,growth=growth,checkpoint_receipts=paths,resource_report=model.resource_report(64,48,active_steps=64*48),NN_bytes=sum(p.numel()*p.element_size()for p in model.parameters()),Adam_bytes=old.tensor_bytes(opt.state_dict()),replay_bytes=sum(v.nbytes for v in replay.arrays.values()))
            del model,opt,replay
        if initial_weights['posterior']!=initial_weights['raw']:raise ValueError('prospective modes must share exactsame27 initial tensors')
        result=dict(status='CPU_POSTERIOR_VALUE_CACHE_GRAPH_ADMISSION_PASS',source_sha256=manifest['source_sha256'],P15=13381,P16=13397,logical_carry_scalars=248,only27_original_NN_parameters=True,same_initial27_tensors_across_modes=True,cache_output_api_checks=True,all27_gradients_checked=True,wrong_mode_and_old_policy_rejected=True,SGD_updates=0,research_updates=0,synthetic_optimizer_updates=2,CPU_only_not_CUDA_result=True,task_data_or_teacher_labels_used=False,fresh_GPU_NN_must_not_load_synthetic_checkpoints=True,results=results,wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,full_training_FLOPs=None,provider_effective_compute=None,scope='freshpairedrule/API/gradient/growth/checkpoint/replay CPUadmissiononly; no capability/retention/fullcarryclosure result')
        with(a.out/'result.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
        print(json.dumps(dict(status=result['status'],P15=13381,P16=13397,modes=['posterior','raw'])))
    except BaseException as error:
        with(a.out/'failure.json').open('x')as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc(),partial_results=results),f,indent=2)
        raise
if __name__=='__main__':main()
