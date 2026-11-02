"""Fresh persistent input-address CPU rule/API/25gradient admission.

No task data, teacher or historical NN. One discarded synthetic Adam update
for this fixed architecture, no researchSGD; GPUfreshNNs must be initialized independently seed0.
"""
import argparse,copy,hashlib,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import continuous_primitive_block as core
import owned_fcba_core as original
import persistent_address_masked_graph_fb as fb
import owned_parent_driver as old
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage

COMMON=('continuous_primitive_block.py','owned_fcba_core.py','persistent_address_masked_graph_fb.py','owned_parent_driver.py','context_replay128.py','memory_live_stage_train.py')
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
    if len(left['grads'])!=25 or left['grads'].keys()!=right['grads'].keys():raise ValueError('exact25 original NN names required')
    result=dict(output_max_abs=float((left['output']-right['output']).abs().max()),loss_abs=float((left['loss']-right['loss']).abs()),gradients={})
    for name,l in left['grads'].items():
        r=right['grads'][name]
        if l is None or r is None or not bool(torch.isfinite(l).all())or not bool(torch.isfinite(r).all()):raise ValueError('missing/nonfinite25gradient: '+name)
        close=bool(torch.allclose(l,r,atol=fb.GRAD_ATOL,rtol=fb.GRAD_RTOL))
        if not close:raise ValueError('25gradient numeric admission failed: '+name)
        result['gradients'][name]=dict(max_abs=float((l-r).abs().max()),allclose=True,norm=float(l.norm()))
    return result

def api_checks(model):
    initial=torch.arange(8);tokens=(torch.arange(8)[:,None]*3+torch.arange(6)[None]*5)%model.n_tokens;lengths=torch.tensor([0,1,2,3,4,5,6,6]);basis=model.encode_state(initial)
    before_rng=rng();before_state={n:v.detach().clone()for n,v in model.state_dict().items()}
    with torch.no_grad():
        empty=model(initial,tokens[:,:0],torch.zeros(8,dtype=torch.long),output='logits')
        if not torch.equal(empty,F.log_softmax(model.readout(basis),-1))or model.forward_all(initial,tokens[:,:0],torch.zeros(8,dtype=torch.long)).shape!=(8,0,8):raise ValueError('ordinary empty-cache API failed')
        first=model.initial_carry(initial)
        if set(first)!=set(('h','controller','keys','values','logp','address'))or sum(v.numel()for v in first.values())!=8*264 or first['values'].shape!=(8,8,8)or bool(first['values'].count_nonzero())or first['address'].shape!=(8,16)or bool(first['address'].count_nonzero()):raise ValueError('exact264carry/zeroVp/zeroaddress required')
        resources=model.resource_report(8,6,active_steps=int(lengths.sum()),return_traces=True)
        if resources['persistent_example_state_scalars']!=8*264 or resources['addressing']['parameter_tensors']!=25:raise ValueError('resource report must include new register and all25 NN tensors')
        if not model.initial_memory_keys.requires_grad:raise ValueError('fixed episode keys must remain trainable across optimizer steps')
        # Same raw input and z, independently vary h and controller. Only h may
        # influence the caller; neither h nor controller may reach the query.
        test={k:v.clone()for k,v in first.items()};test['controller']=torch.arange(128,dtype=basis.dtype).reshape(8,16)/127;test['address']=torch.linspace(-.8,.8,128,dtype=basis.dtype).reshape(8,16)
        h_other={k:v.clone()for k,v in test.items()};h_other['h']=test['h']*3+torch.arange(32,dtype=basis.dtype)[None]/17
        c_other={k:v.clone()for k,v in h_other.items()};c_other['controller']=1-test['controller']*2
        raw=model.token_embedding(torch.arange(8)%model.n_tokens);left,lt=model._transition(test,raw,'soft');right,rt=model._transition(h_other,raw,'soft');third,tt=model._transition(c_other,raw,'soft')
        if not torch.equal(left['controller'],right['controller'])or not torch.equal(lt['write_gate'],rt['write_gate']):raise ValueError('h contaminated input-only controller/write gate')
        expected=test['address']+torch.sigmoid(model.address_gate(raw))*(torch.tanh(model.address_proposal(raw))-test['address'])
        for updated,trace in((left,lt),(right,rt),(third,tt)):
            if not torch.equal(updated['address'],expected)or not torch.equal(trace['address_state'],expected)or not torch.equal(trace['address_query'],expected):raise ValueError('persistent address recurrence/query must be exact and h/c independent')
            if not torch.equal(trace['address_update_gate'],torch.sigmoid(model.address_gate(raw)))or not torch.equal(trace['address_proposal'],torch.tanh(model.address_proposal(raw))):raise ValueError('raw-only address proposal/gate trace mismatch')
            if not torch.equal(trace['read_attention'],lt['read_attention'])or not torch.equal(trace['read_attention'],trace['write_addresses']):raise ValueError('h/controller reached attention or read/write attention differs')
        candidate_change=float((lt['candidate_h']-rt['candidate_h']).abs().max())
        # Checker-only intervention on a discarded deepcopy: finite FP32 -1000
        # bias makes sigmoid exactly zero. Same nonzero held z must give the
        # same query/attention despite different raw tokens, h, and c. This is
        # an API/causal-path check, never a modification of a research NN.
        held=copy.deepcopy(model);held.address_gate.weight.zero_();held.address_gate.bias.fill_(-1000.)
        raw_other=model.token_embedding((torch.arange(8)+1)%model.n_tokens)
        if torch.equal(raw,raw_other)or bool(torch.sigmoid(held.address_gate(raw)).count_nonzero())or bool(torch.sigmoid(held.address_gate(raw_other)).count_nonzero()):raise ValueError('held-address synthetic intervention setup failed')
        hl,ht=held._transition(test,raw,'soft');hr,hu=held._transition(c_other,raw_other,'soft')
        if not torch.equal(hl['address'],test['address'])or not torch.equal(hr['address'],test['address'])or not torch.equal(ht['address_query'],test['address'])or not torch.equal(hu['address_query'],test['address'])or not torch.equal(ht['read_attention'],hu['read_attention']):raise ValueError('direct token/h/controller bypass of held address detected')
        del held
        for mode in('soft','native','straight_through'):
            out,carry=model.forward_value(initial,tokens,lengths,execution=mode)
            if not torch.equal(out,carry['logp'])or not torch.equal(model(initial,tokens,lengths,execution=mode,output='logits'),out)or not torch.equal(model(initial,tokens,lengths,execution=mode,output='probabilities'),out.exp()):raise ValueError('cache/public emission API failed')
            if not torch.equal(model.forward_embedded(basis,model.token_embedding(tokens),lengths,execution=mode,output='logits'),out):raise ValueError('embedded cache output failed')
            allout,allcarry=model.forward_all(initial,tokens,lengths,execution=mode,return_carry=True);state=model.initial_carry(initial);history=[];address_history=[state['address']]
            for step in range(tokens.shape[1]):
                previous={k:v.clone()for k,v in state.items()};ids=(step<lengths).nonzero(as_tuple=False).flatten()
                if len(ids):
                    selected={k:v.index_select(0,ids)for k,v in state.items()};raw=model.token_embedding(tokens[:,step].index_select(0,ids));value,new,trace=model.step_embedded(selected,raw,execution=mode,output='logits')
                    if not torch.equal(new['keys'],selected['keys'])or new['keys'].data_ptr()!=selected['keys'].data_ptr():raise ValueError('episodekeys changed in transition')
                    if not torch.equal(trace['read_attention'],trace['write_addresses']):raise ValueError('second/different attention usedforWRITE')
                    z=selected['address']+torch.sigmoid(model.address_gate(raw))*(torch.tanh(model.address_proposal(raw))-selected['address'])
                    if not torch.equal(new['address'],z)or not torch.equal(trace['address_query'],z)or not torch.equal(trace['address_state'],z)or float(z.abs().max())>1+2e-6:raise ValueError('exact register recurrence/bounded legal episode failed')
                    if not torch.equal(value,new['logp'])or not torch.equal(value,F.log_softmax(model.readout(trace['candidate_h']),-1))or not torch.equal(new['h'],trace['candidate_h']):raise ValueError('RAWcandidate cacheddecoder API failed')
                    posterior=value.exp();mass=new['values'].sum(-1)
                    if float(posterior.min())<0 or float((posterior.sum(-1)-1).abs().max())>2e-6 or float(new['values'].min())< -2e-6 or float(mass.max())>1+2e-6:raise ValueError('posterior/subprobability memory failed')
                    read_p=torch.einsum('bm,bms->bs',trace['read_attention'],selected['values'])
                    if not torch.allclose(read_p@basis,trace['read'],atol=1e-6,rtol=1e-6):raise ValueError('READVpE failed')
                    amount=(trace['write_gate']*trace['write_addresses'])[:,:,None]
                    if not torch.equal(new['values'],selected['values']+amount*(posterior[:,None]-selected['values'])):raise ValueError('WRITEp rule failed')
                    state={k:v.index_copy(0,ids,new[k])for k,v in state.items()}
                inactive=step>=lengths
                if any(not torch.equal(state[k][inactive],previous[k][inactive])for k in state):raise ValueError('inactive row changed a carry field')
                history.append(state['logp']);address_history.append(state['address'])
            traced,tr=model(initial,tokens,lengths,execution=mode,output='logits',return_traces=True)
            if not torch.equal(torch.stack(history,1),allout)or not torch.equal(state['logp'],out)or not torch.equal(traced,out)or not torch.equal(tr['address_states'],torch.stack(address_history,1))or tr['address_states'].shape!=(8,7,16):raise ValueError('ragged/alltoken/address-trace API failed')
            if any(not torch.equal(state[k],carry[k])or not torch.equal(state[k],allcarry[k])or not torch.equal(state[k],tr['final_carry'][k])for k in state):raise ValueError('true final fullcarry differs between public APIs')
            if not torch.equal(allcarry['keys'],first['keys']):raise ValueError('episode keys drifted')
        # All-row masked science helper must carry the new field and preserve
        # every inactive field, including address and cached terminal output.
        masked_out,masked_carry=fb.masked_forward_value(model,initial,tokens,lengths)
        reference_out,reference_carry=model.forward_value(initial,tokens,lengths)
        if not torch.allclose(masked_out,reference_out,atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL)or any(not torch.allclose(masked_carry[k],reference_carry[k],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL)for k in first):raise ValueError('masked fullcarry/address differs from original gather')
        zero_out,zero_carry=fb.masked_forward_value(model,initial,tokens,torch.zeros_like(lengths))
        if not torch.equal(zero_out,first['logp'])or any(not torch.equal(zero_carry[k],first[k])for k in first):raise ValueError('all-row masked helper changed inactive address/carry')
        changed=tokens.clone();changed[:,3:]=(changed[:,3:]+1)%model.n_tokens
        if not torch.equal(model.forward_all(initial,tokens,torch.full((8,),6))[:,:3],model.forward_all(initial,changed,torch.full((8,),6))[:,:3]):raise ValueError('alltoken causality failed')
    if not old.tensor_state_equal(before_state,model.state_dict())or not old.nested_state_equal(before_rng,rng()):raise ValueError('API/held-address checks changed researchNN or RNG')
    return dict(cache_output_api_checks=True,ordinary_empty_cache_checked=True,soft_native_straight_through_ragged_embedded_alltoken_checked=True,causality_checked=True,logical_carry_scalars=264,Vp_subprobability_verified=True,controller_independent_of_h=True,episode_keys_fixed_trainable=True,shared_read_write_attention=True,persistent_address_recurrence_verified=True,address_independent_of_h_and_controller=True,address_query_no_bypass=True,address_masked_on_inactive_rows=True,address_trace_and_true_final_carry_checked=True,held_address_zero_gate_discarded_model_only=True,API_checks_NN_and_RNG_unchanged=True,h_change_candidate_difference_max=candidate_change,no_second_decode=True,raw_h_bound_not_assumed=True)

def numeric_case(model,b,outdir,leaf):
    out,loss=original_step(model,b);left=snapshot(model,out,loss)
    fb.allocate_gradient_buffers(model);out,loss=fb.masked_step(model,**b);right=snapshot(model,out,loss);metrics=compare(left,right);pointers={n:p.grad.data_ptr()for n,p in model.named_parameters()}
    out,loss=fb.masked_step(model,**b);repeat=compare(left,snapshot(model,out,loss))
    if pointers!={n:p.grad.data_ptr()for n,p in model.named_parameters()}:raise ValueError('stable CPU gradient buffers changed')
    arrays=dict(reference_output=left['output'].numpy(),candidate_output=right['output'].numpy(),reference_loss=left['loss'].numpy(),candidate_loss=right['loss'].numpy(),gradient_names_json=np.frombuffer(json.dumps(list(left['grads'])).encode(),dtype=np.uint8))
    for i,name in enumerate(left['grads']):arrays[f'reference_grad_{i:02d}']=left['grads'][name].numpy();arrays[f'candidate_grad_{i:02d}']=right['grads'][name].numpy()
    path=outdir/leaf
    with path.open('xb')as f:np.savez_compressed(f,**arrays)
    return dict(metrics=metrics,repeat_zero_accumulation=repeat,raw_file=leaf,raw_sha256=sha(path),all25_present_finite=True),left

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter();cpu=time.process_time();results={};initial_weights={}
    try:
        manifest=json.loads((a.inputs/'manifest.json').read_text())
        if set(manifest['source_sha256'])!=set(COMMON):raise ValueError('exact common6 source map required')
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen source changed')
        if core.CHECKPOINT_VERSION!='continuous-primitive-block-persistent-input-address-v1'or core.POLICY!='input-gated/persistent-address/static-episode-keys/probvalue-cache/group8/v1':raise ValueError('exact prospective architecture/version/policy required')
        for projection in('persistent_address',):
            torch.manual_seed(0);np.random.seed(0);random.seed(0);initial_rng=rng();model=core.ContinuousPrimitiveBlock(8,15,address_policy=core.POLICY);after_init=rng();initial_weights[projection]=state_sha(model)
            restore(initial_rng);old_model=original.ContinuousPrimitiveBlock(8,15)
            expected={n:v for n,v in old_model.state_dict().items()if not n.startswith(('write_query.','write_key.','read_query.'))}
            expected['controller.weight_ih']=expected['controller.weight_ih'][:,:16].clone()
            if len(expected)!=21:raise ValueError('exact21 retained old parameter tensors required')
            # Reconstruct ordinary fresh modules AFTER the complete old base
            # initialization; deleting old heads/shrinking a tensor draws no RNG.
            factory=dict(device=old_model.encoder.weight.device,dtype=old_model.encoder.weight.dtype)
            proposal=torch.nn.Linear(16,16,**factory);gate=torch.nn.Linear(16,1,**factory)
            with torch.no_grad():gate.bias.fill_(-2.)
            expected.update({'address_proposal.'+n:v for n,v in proposal.state_dict().items()});expected.update({'address_gate.'+n:v for n,v in gate.state_dict().items()})
            if not old.tensor_state_equal(model.state_dict(),expected)or not old.nested_state_equal(after_init,rng())or model.controller.input_size!=16:raise ValueError('exact old subset plus ordinary fresh address initialization/RNG draws required')
            if not torch.equal(model.address_gate.bias,torch.full_like(model.address_gate.bias,-2.))or not all(p.requires_grad for p in model.address_gate.parameters())or not all(p.requires_grad for p in model.address_proposal.parameters()):raise ValueError('initial gatebias -2 and fully learned new address modules required')
            del proposal,gate
            del old_model;restore(after_init)
            if model.parameter_count()!=10550 or len(list(model.named_parameters()))!=25 or not model.initial_memory_keys.requires_grad:raise ValueError('exact P15=10550 and25 tensors')
            model_initial={n:v.detach().clone()for n,v in model.state_dict().items()};api15=api_checks(model)
            opt=torch.optim.Adam(model.parameters(),lr=.003);row=torch.arange(64);tokens=(row[:,None]*3+torch.arange(16)[None]*5)%15;lengths=row%17;lengths[-1]=16
            b=dict(initial=row%8,tokens=tokens,lengths=lengths,targets=(row+3)%8)
            numeric16,golden=numeric_case(model,b,a.out,projection+'-B64-L16-25grad.npz')
            selected=('encoder.weight','initial_memory_keys','controller.weight_ih','controller.weight_hh','primitive_down','primitive_up','address_proposal.weight','address_proposal.bias','address_gate.weight','address_gate.bias','write_gate.weight','readout.weight')
            if any(golden['grads'][n]is None or float(golden['grads'][n].norm())==0 for n in selected):raise ValueError('value/controller/memory/decoder gradient flow must be nonzero')
            if not old.tensor_state_equal(model_initial,model.state_dict()):raise ValueError('0SGD numeric/API admission changed NN')
            # Exactly one diagnostic NLL/Adam step, on synthetic data only.
            opt.zero_grad(set_to_none=True);loss,out=fb.objective(model,fb.original_forward_value,**b)
            if loss.ndim!=0 or not bool(torch.isfinite(loss)):raise ValueError('actual NLL scalar must be finite')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if not all(float(opt.state[p]['step'])==1 for p in model.parameters()):raise ValueError('actual25 Adam states/steps required')
            args=argparse.Namespace(arm='dynamic',block_state='continuous');probe=dict(initial=torch.tensor([0,3]),tokens=torch.tensor([[1,3],[2,3]]),lengths=torch.tensor([2,2]))
            with torch.no_grad():before=model(**probe,output='logits').clone()
            growth=grow_storage(model,opt,args)
            with torch.no_grad():after=model(**probe,output='logits')
            if model.parameter_count()!=10566 or not torch.equal(before,after)or model.configuration()['address_policy']!=core.POLICY:raise ValueError('originalrowgrowth preserves mode/oldtoken execution/P16')
            api16=api_checks(model);tokens48=(row[:,None]*7+torch.arange(48)[None]*3)%16;lengths48=row%49;lengths48[-1]=48
            b48=dict(initial=row%8,tokens=tokens48,lengths=lengths48,targets=(row+5)%8);numeric48,_=numeric_case(model,b48,a.out,projection+'-B64-L48-25grad.npz')
            replay=Replay128(128,48,0,8);raw_data={k:v.numpy()for k,v in b48.items()};used=np.arange(64,dtype=np.int64)[None]
            paths=[]
            for complete in(False,True):
                if complete:replay.consume(raw_data,used,0);replay.validate_sources({0:raw_data},{0:used})
                path=a.out/(projection+('-before-maintenance.pt'if not complete else'-synthetic-step1.pt'));saved_rng=rng();payload=dict(schema='persistent-input-address-CPU-synthetic-v1',synthetic_optimizer_steps=1,research_updates=0,maintenance_complete=complete,address_policy=core.POLICY,block_checkpoint=model.checkpoint(),model_state=model.state_dict(),optimizer=opt.state_dict(),rng=saved_rng,replay=replay.state_dict())
                with path.open('xb')as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
                loaded=torch.load(path,map_location='cpu',weights_only=False);other=core.ContinuousPrimitiveBlock.from_checkpoint(loaded['block_checkpoint'],expected_address_policy=core.POLICY);other_opt=torch.optim.Adam(other.parameters(),lr=.003);other_opt.load_state_dict(loaded['optimizer']);other_replay=Replay128.from_state_dict(loaded['replay'])
                if not old.tensor_state_equal(model.state_dict(),other.state_dict())or not old.nested_state_equal(opt.state_dict(),other_opt.state_dict())or not old.nested_state_equal(replay.state_dict(),other_replay.state_dict()):raise ValueError('real NN/Adam/replay reload failed')
                restore(loaded['rng'])
                if not old.nested_state_equal(rng(),loaded['rng']):raise ValueError('actual CPU RNG reload failed')
                with torch.no_grad():
                    for execution in('soft','native'):
                        if not torch.equal(model(**probe,execution=execution,output='logits'),other(**probe,execution=execution,output='logits')):raise ValueError('actual reloaded cache execution differs')
                for target in('old','wrongconfiguration','wrongpolicy','wrongversion','wrongexpectedpolicy'):
                    cp=copy.deepcopy(loaded['block_checkpoint'])
                    try:
                        if target=='old':original.ContinuousPrimitiveBlock.from_checkpoint(cp)
                        elif target=='wrongconfiguration':cp['configuration']['address_policy']='wrong';core.ContinuousPrimitiveBlock.from_checkpoint(cp)
                        elif target=='wrongpolicy':cp['organization_policy']='wrong';core.ContinuousPrimitiveBlock.from_checkpoint(cp)
                        elif target=='wrongversion':cp['version']='continuous-primitive-block-input-history-shared-address-v1';core.ContinuousPrimitiveBlock.from_checkpoint(cp)
                        else:core.ContinuousPrimitiveBlock.from_checkpoint(cp,expected_address_policy='wrong')
                    except ValueError:pass
                    else:raise ValueError('old/crossmode/wrongpolicy checkpoint was accepted')
                paths.append(dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)));restore(saved_rng)
                del other,other_opt,other_replay,loaded,payload
            results[projection]=dict(P15=10550,P16=model.parameter_count(),initial25_NN_sha256=initial_weights[projection],only25_live_NN_parameters=True,new_address_modules_fresh=True,address_gate_bias_minus2=True,fresh_seed0=True,old_initializer_then_fresh_modules_RNG_exact=True,logical_carry_scalars=264,API15=api15,API16=api16,numeric16=numeric16,numeric48=numeric48,all25_gradients_checked=True,selected_state_memory_parameter_gradient_norms={n:float(golden['grads'][n].norm())for n in selected},synthetic_optimizer_steps=1,synthetic_loss=float(loss.detach()),synthetic_preclip_gradient=float(norm),actual_NN_Adam_CPU_RNG_replay_reload=True,old_schema_and_wrong_policy_configuration_rejected=True,growth=growth,checkpoint_receipts=paths,resource_report=model.resource_report(64,48,active_steps=64*48),NN_bytes=sum(p.numel()*p.element_size()for p in model.parameters()),Adam_bytes=old.tensor_bytes(opt.state_dict()),replay_bytes=sum(v.nbytes for v in replay.arrays.values()))
            del model,opt,replay
        if len(initial_weights)!=1:raise ValueError('one fixed architecture only')
        result=dict(only25_live_NN_parameters=True,new_address_modules_fresh=True,address_gate_bias_minus2=True,persistent_address_recurrence_verified=True,address_independent_of_h_and_controller=True,address_query_no_bypass=True,address_masked_on_inactive_rows=True,initial_remaining_tensors_exact_old_subset=True,wrong_policy_configuration_and_old_schema_rejected=True,h_independent_controller_and_address=True,keys_unchanged_within_episode=True,shared_read_write_attention_exact=True,status='CPU_PERSISTENT_INPUT_ADDRESS_GRAPH_ADMISSION_PASS',source_sha256=manifest['source_sha256'],P15=10550,P16=10566,logical_carry_scalars=264,retained_initialization_exact_subset=True,cache_output_api_checks=True,all25_gradients_checked=True,old_policy_and_wrong_configuration_rejected=True,SGD_updates=0,research_updates=0,synthetic_optimizer_updates=1,controller_independent_of_h=True,episode_keys_fixed_trainable=True,shared_read_write_attention=True,CPU_only_not_CUDA_result=True,task_data_or_teacher_labels_used=False,fresh_GPU_NN_must_not_load_synthetic_checkpoints=True,results=results,wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,full_training_FLOPs=None,provider_effective_compute=None,scope='fresh persistent-input-address rule/API/gradient/growth/checkpoint/replay CPU admission only; no capability/retention/fullcarryclosure result')
        with(a.out/'result.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
        print(json.dumps(dict(status=result['status'],P15=10550,P16=10566,architecture='persistent-input-address')))
    except BaseException as error:
        with(a.out/'failure.json').open('x')as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc(),partial_results=results),f,indent=2)
        raise
if __name__=='__main__':main()
