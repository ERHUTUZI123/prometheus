"""One fixed key-composition dual-memory rule; CPU admission only, no training.

AB theta are preserved. C is new zero-initialized episode state, never inferred
from an old V/z checkpoint. Ordinary ridge associative retrieval, no novelty
or learned binding/capability claim. Full solve and state costs are explicit.
"""
import torch
from torch.nn import functional as F
import owned_ab_core as previous
CHECKPOINT_VERSION='continuous-primitive-block-typed-dual-memory-v1'
POLICY='typed-memory-actions/input-address+key-composition-dual/raw-value/group8/v1'
LOAD,STORE,ADDRESS=previous.LOAD,previous.STORE,previous.ADDRESS
surrogate=previous.surrogate
class ContinuousPrimitiveBlock(previous.ContinuousPrimitiveBlock):
    checkpoint_version=CHECKPOINT_VERSION
    organization_policy=POLICY
    def __init__(self,*args,action_policy=POLICY,input_only_address=True,dual_epsilon=1e-4,dual_ridge=1e-4,**kwargs):
        if action_policy!=POLICY or input_only_address is not True or dual_epsilon!=1e-4 or dual_ridge!=1e-4:raise ValueError('one fixed dual-memory rule, no variants')
        super().__init__(*args,action_policy=previous.POLICY,input_only_address=True,coverage_normalization=True,coverage_epsilon=1e-4,**kwargs)
        if self.n_tokens!=16:raise ValueError('OWN frozen AB16 weights only')
        self.dual_epsilon=dual_epsilon;self.dual_ridge=dual_ridge
        self.register_buffer('dual_identity',torch.eye(self.memory_slots,device=self.encoder.weight.device,dtype=self.encoder.weight.dtype),persistent=False)
    def configuration(self):
        return dict(previous.base.ContinuousPrimitiveBlock.configuration(self),action_policy=POLICY,input_only_address=True,dual_epsilon=self.dual_epsilon,dual_ridge=self.dual_ridge)
    @classmethod
    def from_checkpoint(cls,payload,*,expected_action_policy=POLICY):
        cfg=dict(payload['configuration'])
        if expected_action_policy!=POLICY or payload.get('version')!=CHECKPOINT_VERSION or payload.get('organization_policy')!=POLICY or cfg.get('action_policy')!=POLICY or cfg.get('input_only_address')is not True or cfg.get('dual_epsilon')!=1e-4 or cfg.get('dual_ridge')!=1e-4 or 'coverage_normalization'in cfg or 'coverage_epsilon'in cfg:raise ValueError('strict new C-state/checkpoint policy required')
        t=payload['state_dict']['encoder.weight'];model=cls(**cfg,device=t.device,dtype=t.dtype);model.load_state_dict(payload['state_dict'],strict=True);return model
    @classmethod
    def from_owned_ab_weights(cls,payload,*,expected_source_sha256):
        old=payload['block_checkpoint'];cfg=dict(old['configuration'])
        if payload.get('schema')!='typed-normalized-memory-fourstage-graph-training-v1'or payload.get('architecture_arm')!='AB'or payload.get('architecture_flags')!={'input_only_address':True,'coverage_normalization':True}or payload.get('stage')!=3 or payload.get('step')!=600 or not payload.get('stage_complete')or not payload.get('maintenance_complete')or payload.get('source_sha256')!=expected_source_sha256:raise ValueError('exact completed OWN AB weights required')
        if old.get('version')!=previous.CHECKPOINT_VERSION or old.get('organization_policy')!=previous.POLICY or cfg.get('action_policy')!=previous.POLICY or cfg.get('input_only_address')is not True or cfg.get('coverage_normalization')is not True or cfg.get('coverage_epsilon')!=1e-4:raise ValueError('AB source schema differs')
        if payload['model_state'].keys()!=old['state_dict'].keys()or any(not torch.equal(value,old['state_dict'][name])for name,value in payload['model_state'].items()):raise ValueError('true model theta/checkpoint disagree')
        for key in('action_policy','coverage_normalization','coverage_epsilon'):del cfg[key]
        t=payload['model_state']['encoder.weight'];model=cls(**cfg,device=t.device,dtype=t.dtype);model.load_state_dict(payload['model_state'],strict=True)
        if model.parameter_count()!=10351 or len(list(model.parameters()))!=20:raise ValueError('exact20/P10351 theta required')
        return model
    def initial_carry(self,initial_state):
        h=self.encode_state(initial_state)if initial_state.dtype==torch.long else initial_state
        if h.ndim!=2 or h.shape[1]!=self.internal_states:raise ValueError('initial value shape')
        return dict(h=h,controller=h.new_zeros(len(h),self.controller_dim),values=h.new_zeros(len(h),self.memory_slots,self.internal_states),address=h.new_zeros(len(h),self.memory_slots),logp=F.log_softmax(self.readout(h),-1),key_composition=h.new_zeros(len(h),self.memory_slots,self.memory_slots))
    def dual_read(self,values,key_composition,query):
        z=key_composition.sum(-1);denominator=z.clamp_min(self.dual_epsilon)
        Cbar=key_composition/denominator[:,:,None];Vbar=values/denominator[:,:,None]
        gram=torch.bmm(Cbar,Cbar.transpose(1,2))+self.dual_ridge*self.dual_identity
        rhs=torch.bmm(Cbar,query[:,:,None]);b=torch.linalg.solve(gram,rhs).squeeze(-1)
        read=torch.bmm(b[:,None],Vbar).squeeze(1)
        return read,b,z,Cbar,Vbar
    def _transition(self,carry,raw_x,execution):
        if execution not in('soft','native','straight_through'):raise ValueError('execution mode')
        old_h,c,values,old_address=(carry[k]for k in('h','controller','values','address'))
        c=self.controller(raw_x,c)
        x=raw_x+self.slot_embedding.weight[0]
        attention=F.softmax(old_address,-1)
        read,dual_weights,derived_coverage,normalized_keys,normalized_values=self.dual_read(values,carry['key_composition'],attention)
        features=torch.cat((old_h,c,x,read),-1)
        all_logits=self.caller(features);action_logits=all_logits[:,:self.n_primitives+3]
        probabilities=F.softmax(action_logits,-1);qcomp=F.softmax(all_logits[:,:self.n_primitives],-1)
        chosen=action_logits.argmax(-1);chosen_compute=chosen.clamp_max(self.n_primitives-1)
        hard_weights=F.one_hot(chosen,self.n_primitives+3).to(old_h)
        gates=torch.sigmoid(all_logits[:,self.n_primitives+3:self.n_primitives+3+8]).repeat_interleave(self.internal_states//8,-1)
        proposal=self.address_proposal(raw_x if self.input_only_address else features)
        # Every path uses the OLD value for STORE and OLD logits for this read.
        def apply(weights,chi):
            compute_total=weights[:,:self.n_primitives].sum(-1,keepdim=True)
            h=old_h+compute_total*(chi-old_h)+weights[:,LOAD:LOAD+1]*(read-old_h)
            amount=weights[:,STORE:STORE+1,None]*attention[:,:,None]
            v=values+amount*(old_h[:,None]-values)
            address=old_address+weights[:,ADDRESS:ADDRESS+1]*(proposal-old_address)
            composition=carry['key_composition']+amount*(attention[:,None]-carry['key_composition'])
            return h,v,address,composition
        if execution=='native':
            # One candidate is physically computed per row even for a memory
            # action, then multiplied by zero; this work is charged explicitly.
            low=F.gelu(torch.bmm(self.primitive_down[chosen_compute],features[:,:,None]).squeeze(-1)+self.primitive_bias[chosen_compute])
            delta=torch.bmm(self.primitive_up[chosen_compute],low[:,:,None]).squeeze(-1)
            chi=old_h+gates*(self.residual_norm(old_h+delta)-old_h)
            h,v,address,composition=apply(hard_weights,chi);weights=hard_weights
            logp=F.log_softmax(self.readout(h),-1)
        else:
            low=F.gelu(torch.einsum('bf,krf->bkr',features,self.primitive_down)+self.primitive_bias)
            candidates=torch.einsum('bkr,khr->bkh',low,self.primitive_up)
            delta=torch.einsum('bk,bkh->bh',qcomp,candidates)
            soft_chi=old_h+gates*(self.residual_norm(old_h+delta)-old_h)
            soft_h,soft_v,soft_address,soft_composition=apply(probabilities,soft_chi)
            soft_logp=F.log_softmax(self.readout(soft_h),-1)
            if execution=='soft':
                h,v,address,logp,chi,weights=soft_h,soft_v,soft_address,soft_logp,soft_chi,probabilities;composition=soft_composition
            else:
                hard_delta=candidates[torch.arange(len(old_h),device=old_h.device),chosen_compute]
                hard_chi=old_h+gates*(self.residual_norm(old_h+hard_delta)-old_h)
                hard_h,hard_v,hard_address,hard_composition=apply(hard_weights,hard_chi)
                hard_logp=F.log_softmax(self.readout(hard_h),-1)
                h,v,address,logp,chi,weights=(surrogate(a,b)for a,b in((hard_h,soft_h),(hard_v,soft_v),(hard_address,soft_address),(hard_logp,soft_logp),(hard_chi,soft_chi),(hard_weights,probabilities)))
                composition=surrogate(hard_composition,soft_composition)
        updated=dict(h=h,controller=c,values=v,address=address,logp=logp)
        updated['key_composition']=composition
        trace=dict(
            call_probabilities=qcomp,call_weights=weights[:,:self.n_primitives],call_logits=all_logits[:,:self.n_primitives],call_indices=torch.where(chosen<self.n_primitives,chosen,torch.full_like(chosen,-1)),
            action_probabilities=probabilities,action_weights=weights,action_logits=action_logits,action_indices=chosen,
            compute_total=weights[:,:self.n_primitives].sum(-1,keepdim=True),read=read,read_attention=attention,
            write_addresses=attention,write_gate=weights[:,STORE:STORE+1],write_value=old_h,
            address_update_gate=weights[:,ADDRESS:ADDRESS+1],address_proposal=proposal,address_state=address,address_query=old_address,
            state_gate=gates,candidate_h=chi)
        trace.update(key_composition=composition,read_key_composition=carry['key_composition'],derived_coverage=derived_coverage,dual_weights=dual_weights)
        return updated,trace

    def _sequence(self,initial,embedded,lengths,execution,output,all_outputs,trace):
        b,l,_=embedded.shape;carry=self.initial_carry(initial);history=[];states={k:[v]for k,v in carry.items()if k!='logp'}if trace else None;traces=[]
        shapes=dict(call_probabilities=(4,),call_weights=(4,),call_logits=(4,),call_indices=(),action_probabilities=(7,),action_weights=(7,),action_logits=(7,),action_indices=(),compute_total=(1,),read=(32,),write_addresses=(8,),write_gate=(1,),address_update_gate=(1,),address_proposal=(8,),state_gate=(32,),candidate_h=(32,),key_composition=(8,8),read_key_composition=(8,8),derived_coverage=(8,),dual_weights=(8,))
        for step in range(l):
            ids=(step<lengths).nonzero(as_tuple=False).flatten()
            if ids.numel():
                selected={k:v.index_select(0,ids)for k,v in carry.items()};updated,t=self._transition(selected,embedded[:,step].index_select(0,ids),execution);carry={k:v.index_copy(0,ids,updated[k])for k,v in carry.items()}
            else:t=None
            if all_outputs:history.append(carry['logp'])
            if trace:
                for k,v in carry.items():
                    if k in states:states[k].append(v)
                full={}
                for k,shape in shapes.items():
                    zero=lengths.new_full((b,)+shape,-1)if k.endswith('indices')else carry['h'].new_zeros((b,)+shape);full[k]=zero.index_copy(0,ids,t[k])if t is not None else zero
                traces.append(full)
        logp=torch.stack(history,1)if all_outputs and l else(carry['h'].new_empty(b,0,8)if all_outputs else carry['logp']);answer=logp if output=='logits'else logp.exp()
        if not trace:return answer,carry
        record=dict(states=torch.stack(states['h'],1),controllers=torch.stack(states['controller'],1),memory_values=torch.stack(states['values'],1),address_states=torch.stack(states['address'],1),key_composition_states=torch.stack(states['key_composition'],1),final_state=carry['h'],final_carry=carry)
        for k,shape in shapes.items():
            record[k]=torch.stack([r[k]for r in traces],1)if l else(lengths.new_empty((b,0)+shape)if k.endswith('indices')else carry['h'].new_empty((b,0)+shape))
            if k.startswith('call_')or k=='read':record[k]=record[k].unsqueeze(2)
        record.update(reads=record.pop('read'),write_gates=record.pop('write_gate'),active_mask=torch.arange(l,device=lengths.device)[None]<lengths[:,None],execution=execution,routing=self.routing,action_order=('compute0','compute1','compute2','compute3','LOAD','STORE','ADDRESS'),native_scope='hard7action/expert, soft8query; dual signedretrieval')
        return answer,carry,record
    def resource_report(self,batch_size=1,sequence_length=0,*,active_steps=None,execution='soft',return_traces=False):
        result=super().resource_report(batch_size,sequence_length,active_steps=active_steps,execution=execution,return_traces=return_traces);n=batch_size*sequence_length if active_steps is None else active_steps;m,h=self.memory_slots,self.internal_states;scalar=self.encoder.weight.element_size();parts=result['proxy_forward_macs_by_component']
        parts.pop('coverage_update',None);parts.pop('ST_extra_coverage_update',None);parts['dual_value_mix']=parts.pop('read_values');parts.update(key_composition_update=n*m*m,dual_gram=n*m*m*m,dual_rhs=n*m*m)
        if execution=='straight_through':parts['ST_extra_key_composition_update']=n*m*m
        result.update(proxy_forward_macs=sum(parts.values()),persistent_example_state_scalars=batch_size*384,persistent_example_state_payload_bytes=batch_size*384*scalar,registered_buffer_bytes=self.dual_identity.numel()*self.dual_identity.element_size())
        if return_traces:result['trace_logical_payload_bytes']+=batch_size*((sequence_length+1)*(m*m-m)+sequence_length*(2*m*m))*scalar
        result['typed_actions']['carry_fields']=dict(h=32,controller=16,values=256,address=8,logp=8,key_composition=64)
        result.pop('coverage_normalization',None)
        result['dual_memory']=dict(state_shape=[8,8],coverage_derived='C.sum(-1), no persistent z',epsilon=1e-4,ridge=1e-4,read_solves=n,solve_shape=[8,8],solver_forward_MAC_unknown=None,solver_workspace_bytes=None,solver_backward_cost_unknown=None,normalization_divisions=n*(m*h+m*m),coverage_sum_additions=n*m*(m-1),shared_nonpersistent_identity_bytes=256,rule='Cj+=w_j(k-Cj); VwriteunchangedOLDh; signeddualread',failure_scope='erasedidenticalkeys/rankcollapse cannot recover binding; ridgebias/noise/EMAoverwrite remain; no capabilityclaim')
        result['proxy_scope']='explicitknownforwardlinear/dot/blend products EXCLUDES dense solver internals, nonlinearities/indexing/backward; not fullFLOPs; solve costs measured only by actualCPUtimers/workspaceunknown'
        return result
