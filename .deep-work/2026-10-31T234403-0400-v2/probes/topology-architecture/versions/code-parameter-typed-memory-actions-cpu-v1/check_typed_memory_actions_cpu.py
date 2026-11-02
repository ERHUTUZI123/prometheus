"""Fresh typed memory-action CPU rule/API/20gradient admission.

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
import typed_memory_actions_masked_graph_fb as fb
import owned_parent_driver as old
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage

COMMON=('continuous_primitive_block.py','owned_fcba_core.py','typed_memory_actions_masked_graph_fb.py','owned_parent_driver.py','context_replay128.py','memory_live_stage_train.py')
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
    if len(left['grads'])!=20 or left['grads'].keys()!=right['grads'].keys():raise ValueError('exact20 original NN names required')
    result=dict(output_max_abs=float((left['output']-right['output']).abs().max()),loss_abs=float((left['loss']-right['loss']).abs()),gradients={})
    for name,l in left['grads'].items():
        r=right['grads'][name]
        if l is None or r is None or not bool(torch.isfinite(l).all())or not bool(torch.isfinite(r).all()):raise ValueError('missing/nonfinite20gradient: '+name)
        close=bool(torch.allclose(l,r,atol=fb.GRAD_ATOL,rtol=fb.GRAD_RTOL))
        if not close:raise ValueError('20gradient numeric admission failed: '+name)
        result['gradients'][name]=dict(max_abs=float((l-r).abs().max()),allclose=True,norm=float(l.norm()))
    return result

def reference_rule(model,carry,raw_x,execution):
    """Independent equations, never calls model._transition or a task teacher."""
    h,old_c,values,address=(carry[k]for k in('h','controller','values','address'))
    controller=model.controller(raw_x,old_c);x=raw_x+model.slot_embedding.weight[0]
    attention=F.softmax(address,-1);read=torch.einsum('bm,bmh->bh',attention,values)
    features=torch.cat((h,controller,x,read),-1)
    call_features=features if model.routing=='dynamic'else torch.cat((torch.zeros_like(h),torch.zeros_like(controller),x,torch.zeros_like(read)),-1)
    logits=model.caller(call_features);alpha=F.softmax(logits[:,:7],-1);q=F.softmax(logits[:,:4],-1)
    gates=torch.sigmoid(logits[:,7:]).repeat_interleave(4,-1)
    low=F.gelu(torch.einsum('bf,krf->bkr',features,model.primitive_down)+model.primitive_bias)
    experts=torch.einsum('bkr,khr->bkh',low,model.primitive_up)
    delta=torch.einsum('bk,bkh->bh',q,experts)
    chi=h+gates*(model.residual_norm(h+delta)-h)
    proposal=model.address_proposal(features)
    soft_h=h+alpha[:,:4].sum(-1,keepdim=True)*(chi-h)+alpha[:,4:5]*(read-h)
    soft_values=values+alpha[:,5:6,None]*attention[:,:,None]*(h[:,None]-values)
    soft_address=address+alpha[:,6:7]*(proposal-address)
    soft_logp=F.log_softmax(model.readout(soft_h),-1)
    if execution=='soft':new_h,new_values,new_address,new_logp=soft_h,soft_values,soft_address,soft_logp
    else:
        action=logits[:,:7].argmax(-1);compute=action<4;indices=action.clamp_max(3)
        # bmm independently reconstructs the one selected compute expert.
        low_one=F.gelu(torch.bmm(model.primitive_down[indices],features[:,:,None]).squeeze(-1)+model.primitive_bias[indices])
        delta_one=torch.bmm(model.primitive_up[indices],low_one[:,:,None]).squeeze(-1)
        chi_one=h+gates*(model.residual_norm(h+delta_one)-h)
        hard_h=torch.where(compute[:,None],chi_one,torch.where((action==4)[:,None],read,h))
        hard_values=values+(action==5)[:,None,None].to(h)*attention[:,:,None]*(h[:,None]-values)
        hard_address=address+(action==6)[:,None].to(h)*(proposal-address)
        hard_logp=F.log_softmax(model.readout(hard_h),-1)
        if execution=='native':new_h,new_values,new_address,new_logp=hard_h,hard_values,hard_address,hard_logp
        elif execution=='straight_through':
            new_h=hard_h.detach()+(soft_h-soft_h.detach());new_values=hard_values.detach()+(soft_values-soft_values.detach());new_address=hard_address.detach()+(soft_address-soft_address.detach())
            new_logp=hard_logp.detach()+(soft_logp-soft_logp.detach())
        else:raise ValueError('reference execution mode')
    return dict(h=new_h,controller=controller,values=new_values,address=new_address,logp=new_logp),dict(alpha=alpha,conditional_compute=q,chi=chi,attention=attention,read=read,address_proposal=proposal,features=features,logits=logits)

def close_carry(actual,expected,label):
    if actual.keys()!=expected.keys():raise ValueError(label+': carry fields differ')
    for key in actual:
        if not torch.allclose(actual[key],expected[key],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL):raise ValueError(label+': '+key+' differs')

def api_checks(model):
    initial=torch.arange(8);tokens=(torch.arange(8)[:,None]*3+torch.arange(6)[None]*5)%model.n_tokens;lengths=torch.tensor([0,1,2,3,4,5,6,6]);basis=model.encode_state(initial)
    before_rng=rng();before_state={n:v.detach().clone()for n,v in model.state_dict().items()}
    with torch.no_grad():
        first=model.initial_carry(initial)
        if set(first)!=set(('h','controller','values','address','logp'))or sum(v.numel()for v in first.values())!=8*320 or first['values'].shape!=(8,8,32)or first['address'].shape!=(8,8)or bool(first['values'].count_nonzero())or bool(first['address'].count_nonzero()):raise ValueError('exact320carry, zero raw-value memory and zero address required')
        names=dict(model.named_parameters())
        if len(names)!=20 or any(n=='initial_memory_keys'or n.startswith(('read_query.','write_query.','write_key.','write_gate.','address_gate.'))for n in names):raise ValueError('removed keys/separate projections/gates remain in live NN')
        report=model.resource_report(8,6,active_steps=int(lengths.sum()),return_traces=True)
        if report['persistent_example_state_scalars']!=8*320:raise ValueError('resource report omits raw memory/address/cache carry')
        empty=model(initial,tokens[:,:0],torch.zeros_like(lengths),output='logits')
        if not torch.equal(empty,F.log_softmax(model.readout(basis),-1))or model.forward_all(initial,tokens[:,:0],torch.zeros_like(lengths)).shape!=(8,0,8):raise ValueError('ordinary empty/cache/alltoken API failed')
        # Nontrivial synthetic state distinguishes OLDh from compute candidate,
        # old-address READ from next-address READ, and raw values from posteriors.
        test={k:v.clone()for k,v in first.items()};test['h']=(torch.arange(256,dtype=basis.dtype).reshape(8,32)-100)/97
        test['controller']=torch.linspace(-.7,.7,128,dtype=basis.dtype).reshape(8,16)
        test['values']=torch.arange(8,dtype=basis.dtype)[None,:,None].expand(8,-1,32)/3+torch.arange(32,dtype=basis.dtype)[None,None]/100
        test['address']=torch.linspace(-1.,1.,64,dtype=basis.dtype).reshape(8,8)
        test['logp']=F.log_softmax(model.readout(test['h']),-1)
        raw=model.token_embedding(torch.arange(8)%model.n_tokens)
        h_other={k:v.clone()for k,v in test.items()};h_other['h']=test['h']*3+.2
        left,lt=model._transition(test,raw,'soft');right,rt=model._transition(h_other,raw,'soft')
        if not torch.equal(left['controller'],right['controller']):raise ValueError('h contaminated raw-input-only controller')
        # Current attention is solely OLDl even when full caller features differ.
        if not torch.equal(lt['read_attention'],F.softmax(test['address'],-1))or not torch.equal(rt['read_attention'],lt['read_attention']):raise ValueError('current READ is not solely old address logits')
        for mode in('soft','native','straight_through'):
            actual,trace=model._transition(test,raw,mode);expected,rule=reference_rule(model,test,raw,mode);close_carry(actual,expected,'independent '+mode+' transition equations')
            if not torch.equal(trace['read_attention'],rule['attention'])or not torch.equal(trace['write_addresses'],rule['attention'])or not torch.allclose(trace['read'],rule['read'],atol=1e-6,rtol=1e-6):raise ValueError('READ/STORE must share OLDl attention on raw oldV')
            if not torch.equal(actual['logp'],F.log_softmax(model.readout(actual['h']),-1)):raise ValueError('output cache is not actual newh decoder')
            if not torch.equal(trace['action_probabilities'],rule['alpha'])or not torch.equal(trace['call_probabilities'],rule['conditional_compute'])or not torch.equal(trace['action_logits'],rule['logits'][:,:7])or not torch.equal(trace['state_gate'],torch.sigmoid(rule['logits'][:,7:]).repeat_interleave(4,-1)):raise ValueError('router7/conditionalcompute4/G8 slices are wrong')
            if not torch.equal(trace['write_value'],test['h'])or not torch.equal(trace['address_query'],test['address'])or not torch.equal(trace['address_state'],actual['address']):raise ValueError('trace must distinguish OLDh/OLDaddress/newaddress')
            chosen=rule['logits'][:,:7].argmax(-1)
            if not torch.equal(trace['action_indices'],chosen)or not torch.equal(trace['call_indices'],torch.where(chosen<4,chosen,torch.full_like(chosen,-1))):raise ValueError('native action/compute index scope is ambiguous')
            expected_weights=rule['alpha']if mode=='soft'else F.one_hot(chosen,7).to(test['h'])
            if not torch.equal(trace['action_weights'],expected_weights)or not torch.equal(trace['call_weights'],expected_weights[:,:4])or not torch.equal(trace['compute_total'],expected_weights[:,:4].sum(-1,keepdim=True))or not torch.equal(trace['write_gate'],expected_weights[:,5:6])or not torch.equal(trace['address_update_gate'],expected_weights[:,6:7]):raise ValueError('typed action weights/gates do not share declared action budget')
            if mode=='soft':
                alpha=rule['alpha'];remaining=1-alpha[:,:4].sum(-1)-alpha[:,4]
                if float(alpha.min())<0 or float((alpha.sum(-1)-1).abs().max())>2e-6 or not torch.allclose(remaining,alpha[:,5]+alpha[:,6],atol=2e-7,rtol=2e-6):raise ValueError('one7way action budget/mass partition failed')
                old_h_store=test['values']+alpha[:,5:6,None]*rule['attention'][:,:,None]*(test['h'][:,None]-test['values'])
                if not torch.equal(actual['values'],old_h_store):raise ValueError('STORE must write OLDh, never chi/newh/posterior')
        # All seven hard actions are exercised on discarded checker copies.
        # Fixed router logits are synthetic interventions, no teacher action data.
        forced=copy.deepcopy(model)
        for action in range(7):
            forced.caller[2].weight[:7].zero_();forced.caller[2].bias[:7].fill_(-1000.);forced.caller[2].bias[action]=0.
            actual,trace=forced._transition(test,raw,'native');expected,rule=reference_rule(forced,test,raw,'native');close_carry(actual,expected,'forced native action '+str(action))
            if action in(5,6)and not torch.equal(actual['h'],test['h']):raise ValueError('STORE/ADDRESS unexpectedly changed h')
            if action!=5 and not torch.equal(actual['values'],test['values']):raise ValueError('nonSTORE action changed memory')
            if action!=6 and not torch.equal(actual['address'],test['address']):raise ValueError('nonADDRESS action changed address')
            if action==4 and not torch.allclose(actual['h'],rule['read'],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL):raise ValueError('LOAD must directly copy raw memory READ')
        # Force a clearly different proposal. ADDRESS must keep this token's
        # h/V and read unchanged; only a subsequent LOAD may use the new logits.
        forced.address_proposal.weight.zero_();forced.address_proposal.bias.copy_(torch.linspace(-4.,4.,8,dtype=basis.dtype))
        forced.caller[2].bias[:7].fill_(-1000.);forced.caller[2].bias[6]=0.
        addressed,at=forced._transition(test,raw,'native')
        if not torch.allclose(addressed['address'],forced.address_proposal.bias[None].expand(8,-1),atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL)or not torch.equal(at['read_attention'],F.softmax(test['address'],-1))or not torch.equal(addressed['h'],test['h'])or not torch.equal(addressed['values'],test['values']):raise ValueError('ADDRESS leaked into current token or failed proposal update')
        forced.caller[2].bias[:7].fill_(-1000.);forced.caller[2].bias[4]=0.
        loaded,after_trace=forced._transition(addressed,raw,'native');next_read=torch.einsum('bm,bmh->bh',F.softmax(addressed['address'],-1),test['values'])
        if not torch.allclose(loaded['h'],next_read,atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL)or not torch.equal(after_trace['read_attention'],F.softmax(addressed['address'],-1))or torch.allclose(next_read,at['read'],atol=1e-3,rtol=1e-3):raise ValueError('next-token address/LOAD synthetic distinction failed')
        del forced
        for mode in('soft','native','straight_through'):
            out,carry=model.forward_value(initial,tokens,lengths,execution=mode)
            if not torch.equal(out,carry['logp'])or not torch.equal(model(initial,tokens,lengths,execution=mode,output='logits'),out)or not torch.equal(model(initial,tokens,lengths,execution=mode,output='probabilities'),out.exp()):raise ValueError('cached public emission API failed')
            if not torch.equal(model.forward_embedded(basis,model.token_embedding(tokens),lengths,execution=mode,output='logits'),out):raise ValueError('embedded cached output differs')
            allout,allcarry=model.forward_all(initial,tokens,lengths,execution=mode,return_carry=True);state=model.initial_carry(initial);history=[];addresses=[state['address']]
            for step in range(tokens.shape[1]):
                previous={k:v.clone()for k,v in state.items()};ids=(step<lengths).nonzero(as_tuple=False).flatten()
                if len(ids):
                    selected={k:v.index_select(0,ids)for k,v in state.items()};value,updated,trace=model.step_embedded(selected,model.token_embedding(tokens[:,step].index_select(0,ids)),execution=mode,output='logits')
                    if not torch.equal(value,updated['logp'])or not torch.equal(value,F.log_softmax(model.readout(updated['h']),-1)):raise ValueError('step cache/current h mismatch')
                    state={k:v.index_copy(0,ids,updated[k])for k,v in state.items()}
                inactive=step>=lengths
                if any(not torch.equal(state[k][inactive],previous[k][inactive])for k in state):raise ValueError('inactive row changed fullcarry')
                history.append(state['logp']);addresses.append(state['address'])
            traced,tr=model(initial,tokens,lengths,execution=mode,output='logits',return_traces=True)
            if not torch.equal(torch.stack(history,1),allout)or not torch.equal(state['logp'],out)or not torch.equal(traced,out)or not torch.equal(tr['address_states'],torch.stack(addresses,1)):raise ValueError('ragged/alltoken/address traces mismatch')
            if any(not torch.equal(state[k],carry[k])or not torch.equal(state[k],allcarry[k])or not torch.equal(state[k],tr['final_carry'][k])for k in state):raise ValueError('true final fullcarry not returned consistently')
        masked_out,masked_carry=fb.masked_forward_value(model,initial,tokens,lengths);reference_out,reference_carry=model.forward_value(initial,tokens,lengths)
        close_carry(masked_carry,reference_carry,'masked ragged finalcarry')
        if not torch.allclose(masked_out,reference_out,atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL):raise ValueError('masked output differs')
        zero_out,zero_carry=fb.masked_forward_value(model,initial,tokens,torch.zeros_like(lengths))
        if not torch.equal(zero_out,first['logp'])or any(not torch.equal(zero_carry[k],first[k])for k in first):raise ValueError('all-row helper changed inactive carry/address/cache')
        changed=tokens.clone();changed[:,3:]=(changed[:,3:]+1)%model.n_tokens
        if not torch.equal(model.forward_all(initial,tokens,torch.full((8,),6))[:,:3],model.forward_all(initial,changed,torch.full((8,),6))[:,:3]):raise ValueError('alltoken prefix causality failed')
    # ST covers one transition including the separately decoded cached logp.
    # This is the soft Jacobian at the same incoming carry, NOT the derivative
    # of a different entire all-soft recurrent trajectory.
    derivative_inputs=('h','controller','values','address');derivative_outputs=derivative_inputs+('logp',);derivatives={}
    for mode in('soft','straight_through'):
        differentiable={k:v.detach().clone().requires_grad_(k in derivative_inputs)for k,v in test.items()}
        differentiable_raw=raw.detach().clone().requires_grad_();updated,_=model._transition(differentiable,differentiable_raw,mode)
        loss=sum((updated[k]*torch.linspace(.01,.91,updated[k].numel(),dtype=updated[k].dtype).reshape_as(updated[k])).sum()for k in derivative_outputs)
        variables=tuple(model.parameters())+tuple(differentiable[k]for k in derivative_inputs)+(differentiable_raw,)
        derivatives[mode]=torch.autograd.grad(loss,variables,allow_unused=True)
    none_names=[];jacobian_max=0.
    variable_names=list(dict(model.named_parameters()))+['carry.'+k for k in derivative_inputs]+['raw_x']
    for name,l,r in zip(variable_names,derivatives['soft'],derivatives['straight_through']):
        if(l is None)!=(r is None):raise ValueError('ST/soft fixed VJP None pattern differs: '+name)
        if l is None:none_names.append(name);continue
        if not bool(torch.isfinite(l).all())or not bool(torch.isfinite(r).all())or not torch.allclose(l,r,atol=fb.GRAD_ATOL,rtol=fb.GRAD_RTOL):raise ValueError('ST whole-transition/cache fixed VJP differs: '+name)
        jacobian_max=max(jacobian_max,float((l-r).abs().max()))
    if not old.tensor_state_equal(before_state,model.state_dict())or not old.nested_state_equal(before_rng,rng()):raise ValueError('API/synthetic interventions changed research NN or RNG')
    return dict(cache_output_api_checks=True,ordinary_empty_cache_checked=True,soft_native_straight_through_ragged_embedded_alltoken_checked=True,causality_checked=True,logical_carry_scalars=320,controller_independent_of_h=True,no_key_or_separate_gate_parameters=True,typed_transition_rules_verified=True,one7way_action_mass_partition=True,old_G8_conditional_compute_rule_verified=True,all7_hard_actions_discarded_model_checked=True,store_uses_old_h=True,address_applies_next_token=True,raw_value_memory_not_probability_mass=True,all_carry_masked_on_inactive_rows=True,address_trace_and_true_final_carry_checked=True,trace_active_mask_bytes=tr['active_mask'].numel()*tr['active_mask'].element_size(),ST_whole_transition_and_cache_fixed_VJP_checked=True,ST_fixed_VJP_scope='one deterministic linspace cotangent at one synthetic carry/raw input; parameters+carry+raw_x VJP, not full Jacobian/all states',ST_fixed_VJP_max_abs=jacobian_max,ST_fixed_VJP_None_names=none_names,API_checks_NN_and_RNG_unchanged=True)

def numeric_case(model,b,outdir,leaf):
    out,loss=original_step(model,b);left=snapshot(model,out,loss)
    fb.allocate_gradient_buffers(model);out,loss=fb.masked_step(model,**b);right=snapshot(model,out,loss);metrics=compare(left,right);pointers={n:p.grad.data_ptr()for n,p in model.named_parameters()}
    out,loss=fb.masked_step(model,**b);repeat=compare(left,snapshot(model,out,loss))
    if pointers!={n:p.grad.data_ptr()for n,p in model.named_parameters()}:raise ValueError('stable CPU gradient buffers changed')
    arrays=dict(reference_output=left['output'].numpy(),candidate_output=right['output'].numpy(),reference_loss=left['loss'].numpy(),candidate_loss=right['loss'].numpy(),gradient_names_json=np.frombuffer(json.dumps(list(left['grads'])).encode(),dtype=np.uint8))
    for i,name in enumerate(left['grads']):arrays[f'reference_grad_{i:02d}']=left['grads'][name].numpy();arrays[f'candidate_grad_{i:02d}']=right['grads'][name].numpy()
    path=outdir/leaf
    with path.open('xb')as f:np.savez_compressed(f,**arrays)
    return dict(metrics=metrics,repeat_zero_accumulation=repeat,raw_file=leaf,raw_sha256=sha(path),all20_present_finite=True),left

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args();a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);started=time.perf_counter();cpu=time.process_time();results={};initial_weights={}
    try:
        manifest=json.loads((a.inputs/'manifest.json').read_text())
        if set(manifest['source_sha256'])!=set(COMMON):raise ValueError('exact common6 source map required')
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen source changed')
        if core.CHECKPOINT_VERSION!='continuous-primitive-block-typed-memory-actions-v1'or core.POLICY!='typed-memory-actions/compute-load-store-address/raw-value/group8/v1':raise ValueError('exact prospective architecture/version/policy required')
        for projection in('typed_memory_actions',):
            torch.manual_seed(0);np.random.seed(0);random.seed(0);initial_rng=rng();model=core.ContinuousPrimitiveBlock(8,15,action_policy=core.POLICY);after_init=rng();initial_weights[projection]=state_sha(model)
            restore(initial_rng);old_model=original.ContinuousPrimitiveBlock(8,15)
            expected={n:v for n,v in old_model.state_dict().items()if n!='initial_memory_keys'and not n.startswith(('write_query.','write_key.','read_query.','write_gate.'))}
            expected['controller.weight_ih']=expected['controller.weight_ih'][:,:16].clone()
            if len(expected)!=18:raise ValueError('exact18 retained old parameter tensors required')
            # Complete old initialization is retained before pruning. Enlarging
            # the router by concatenating fixed rows consumes no random draws.
            old_weight=expected['caller.2.weight'];old_bias=expected['caller.2.bias']
            expected['caller.2.weight']=torch.cat((old_weight[:4],old_weight.new_zeros(3,32),old_weight[4:]),0)
            expected['caller.2.bias']=torch.cat((old_bias[:4],old_bias.new_full((3,),-2.),old_bias[4:]),0)
            factory=dict(device=old_model.encoder.weight.device,dtype=old_model.encoder.weight.dtype)
            proposal=torch.nn.Linear(96,8,**factory)
            expected.update({'address_proposal.'+n:v for n,v in proposal.state_dict().items()})
            if not old.tensor_state_equal(model.state_dict(),expected)or not old.nested_state_equal(after_init,rng())or model.controller.input_size!=16:raise ValueError('exact old subset/router-row mapping plus ordinary fresh proposal/RNG draws required')
            if not torch.equal(model.caller[2].weight[4:7],torch.zeros_like(model.caller[2].weight[4:7]))or not torch.equal(model.caller[2].bias[4:7],torch.full_like(model.caller[2].bias[4:7],-2.)):raise ValueError('initial new3 router rows must have W0,bias-2')
            del proposal,old_model;restore(after_init)
            if model.parameter_count()!=10975 or len(list(model.named_parameters()))!=20 or not all(p.requires_grad for p in model.parameters()):raise ValueError('exact P15=10975 and20 live tensors')
            model_initial={n:v.detach().clone()for n,v in model.state_dict().items()};api15=api_checks(model)
            opt=torch.optim.Adam(model.parameters(),lr=.003);row=torch.arange(64);tokens=(row[:,None]*3+torch.arange(16)[None]*5)%15;lengths=row%17;lengths[-1]=16
            b=dict(initial=row%8,tokens=tokens,lengths=lengths,targets=(row+3)%8)
            numeric16,golden=numeric_case(model,b,a.out,projection+'-B64-L16-20grad.npz')
            selected=('encoder.weight','controller.weight_ih','controller.weight_hh','primitive_down','primitive_up','address_proposal.weight','address_proposal.bias','caller.2.weight','caller.2.bias','readout.weight')
            if any(golden['grads'][n]is None or float(golden['grads'][n].norm())==0 for n in selected):raise ValueError('value/controller/memory/decoder gradient flow must be nonzero')
            if not old.tensor_state_equal(model_initial,model.state_dict()):raise ValueError('0SGD numeric/API admission changed NN')
            # Exactly one diagnostic NLL/Adam step, on synthetic data only.
            opt.zero_grad(set_to_none=True);loss,out=fb.objective(model,fb.original_forward_value,**b)
            if loss.ndim!=0 or not bool(torch.isfinite(loss)):raise ValueError('actual NLL scalar must be finite')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if not all(float(opt.state[p]['step'])==1 for p in model.parameters()):raise ValueError('actual20 Adam states/steps required')
            args=argparse.Namespace(arm='dynamic',block_state='continuous');probe=dict(initial=torch.tensor([0,3]),tokens=torch.tensor([[1,3],[2,3]]),lengths=torch.tensor([2,2]))
            with torch.no_grad():before=model(**probe,output='logits').clone()
            growth=grow_storage(model,opt,args)
            with torch.no_grad():after=model(**probe,output='logits')
            if model.parameter_count()!=10991 or not torch.equal(before,after)or model.configuration()['action_policy']!=core.POLICY:raise ValueError('originalrowgrowth preserves mode/oldtoken execution/P16')
            api16=api_checks(model);tokens48=(row[:,None]*7+torch.arange(48)[None]*3)%16;lengths48=row%49;lengths48[-1]=48
            b48=dict(initial=row%8,tokens=tokens48,lengths=lengths48,targets=(row+5)%8);numeric48,_=numeric_case(model,b48,a.out,projection+'-B64-L48-20grad.npz')
            replay=Replay128(128,48,0,8);raw_data={k:v.numpy()for k,v in b48.items()};used=np.arange(64,dtype=np.int64)[None]
            paths=[]
            for complete in(False,True):
                if complete:replay.consume(raw_data,used,0);replay.validate_sources({0:raw_data},{0:used})
                path=a.out/(projection+('-before-maintenance.pt'if not complete else'-synthetic-step1.pt'));saved_rng=rng();payload=dict(schema='typed-memory-actions-CPU-synthetic-v1',synthetic_optimizer_steps=1,research_updates=0,maintenance_complete=complete,action_policy=core.POLICY,block_checkpoint=model.checkpoint(),model_state=model.state_dict(),optimizer=opt.state_dict(),rng=saved_rng,replay=replay.state_dict())
                with path.open('xb')as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
                loaded=torch.load(path,map_location='cpu',weights_only=False);other=core.ContinuousPrimitiveBlock.from_checkpoint(loaded['block_checkpoint'],expected_action_policy=core.POLICY);other_opt=torch.optim.Adam(other.parameters(),lr=.003);other_opt.load_state_dict(loaded['optimizer']);other_replay=Replay128.from_state_dict(loaded['replay'])
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
                        elif target=='wrongconfiguration':cp['configuration']['action_policy']='wrong';core.ContinuousPrimitiveBlock.from_checkpoint(cp)
                        elif target=='wrongpolicy':cp['organization_policy']='wrong';core.ContinuousPrimitiveBlock.from_checkpoint(cp)
                        elif target=='wrongversion':cp['version']='continuous-primitive-block-persistent-input-address-v1';core.ContinuousPrimitiveBlock.from_checkpoint(cp)
                        else:core.ContinuousPrimitiveBlock.from_checkpoint(cp,expected_action_policy='wrong')
                    except ValueError:pass
                    else:raise ValueError('old/crossmode/wrongpolicy checkpoint was accepted')
                paths.append(dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path)));restore(saved_rng)
                del other,other_opt,other_replay,loaded,payload
            results[projection]=dict(P15=10975,P16=model.parameter_count(),initial20_NN_sha256=initial_weights[projection],only20_live_NN_parameters=True,new_address_proposal_fresh=True,new_action_rows_zero_weight_minus2_bias=True,fresh_seed0=True,old_initializer_then_fresh_modules_RNG_exact=True,logical_carry_scalars=320,API15=api15,API16=api16,numeric16=numeric16,numeric48=numeric48,all20_gradients_checked=True,selected_state_memory_parameter_gradient_norms={n:float(golden['grads'][n].norm())for n in selected},synthetic_optimizer_steps=1,synthetic_loss=float(loss.detach()),synthetic_preclip_gradient=float(norm),actual_NN_Adam_CPU_RNG_replay_reload=True,old_schema_and_wrong_policy_configuration_rejected=True,growth=growth,checkpoint_receipts=paths,resource_report=model.resource_report(64,48,active_steps=64*48),NN_bytes=sum(p.numel()*p.element_size()for p in model.parameters()),Adam_bytes=old.tensor_bytes(opt.state_dict()),replay_bytes=sum(v.nbytes for v in replay.arrays.values()))
            del model,opt,replay
        if len(initial_weights)!=1:raise ValueError('one fixed architecture only')
        result=dict(status='CPU_TYPED_MEMORY_ACTIONS_GRAPH_ADMISSION_PASS',source_sha256=manifest['source_sha256'],P15=10975,P16=10991,logical_carry_scalars=320,only20_live_NN_parameters=True,retained_initialization_exact_subset=True,new_address_proposal_fresh=True,new_action_rows_zero_weight_minus2_bias=True,cache_output_api_checks=True,all20_gradients_checked=True,old_policy_and_wrong_configuration_rejected=True,SGD_updates=0,research_updates=0,synthetic_optimizer_updates=1,controller_independent_of_h=True,no_key_or_separate_gate_parameters=True,typed_transition_rules_verified=True,store_uses_old_h=True,address_applies_next_token=True,all_carry_masked_on_inactive_rows=True,soft_native_straight_through_checked=True,CPU_only_not_CUDA_result=True,task_data_or_teacher_labels_used=False,fresh_GPU_NN_must_not_load_synthetic_checkpoints=True,results=results,wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,full_training_FLOPs=None,provider_effective_compute=None,scope='fresh typed-memory-action rule/API/gradient/growth/checkpoint/replay CPU admission only; no capability/retention/fullcarryclosure result')
        with(a.out/'result.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
        print(json.dumps(dict(status=result['status'],P15=10975,P16=10991,architecture='typed-memory-actions')))
    except BaseException as error:
        with(a.out/'failure.json').open('x')as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc(),partial_results=results),f,indent=2)
        raise
if __name__=='__main__':main()
