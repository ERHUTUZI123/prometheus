"""One typed compute/LOAD/STORE/ADDRESS Block, fresh training only.

All seven action choices are learned from terminal NLL. No correct action,
address, token class or intermediate state is supplied. Memory stores OLD h;
direct address logits have no key matrix. Native hardens only the seven-way
action/expert choice; memory attention remains soft. ST has hard forward values
with each one-step soft-transition Jacobian at the same incoming carry.
"""
import torch
from torch.nn import functional as F
import owned_fcba_core as base

BASE_SHA256='fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158'
CHECKPOINT_VERSION='continuous-primitive-block-typed-memory-actions-v1'
POLICY='typed-memory-actions/compute-load-store-address/raw-value/group8/v1'
LOAD,STORE,ADDRESS=4,5,6

class TypedCallerHead(torch.nn.Module):
    """Retain old four compute/eight G8 rows; add three fixed-prior rows.

    Constructing from tensors consumes no random draws. The new address
    proposal is the only fresh factory after the complete original factory.
    State-dict names stay caller.2.weight/bias, with the new strict schema.
    """
    def __init__(self,old_head,k):
        super().__init__();self.in_features=old_head.weight.shape[1];self.out_features=k+3+8
        self.weight=torch.nn.Parameter(torch.cat((old_head.weight[:k].detach(),old_head.weight.new_zeros(3,self.in_features),old_head.weight[k:k+8].detach()),0).clone())
        self.bias=torch.nn.Parameter(torch.cat((old_head.bias[:k].detach(),old_head.bias.new_full((3,),-2.),old_head.bias[k:k+8].detach()),0).clone())
    def forward(self,x):return F.linear(x,self.weight,self.bias)

def surrogate(hard,soft):return hard.detach()+(soft-soft.detach())

class ContinuousPrimitiveBlock(base.ContinuousPrimitiveBlock):
    checkpoint_version=CHECKPOINT_VERSION
    organization_policy=POLICY
    def __init__(self,*args,action_policy=POLICY,**kwargs):
        if action_policy!=POLICY:raise ValueError('one fixed typed-memory-action policy')
        # Retained tensors use the full original factory/draw sequence.
        super().__init__(*args,**kwargs)
        if (self.n_states,self.internal_states,self.memory_dim,self.n_primitives,self.primitive_rank,self.controller_dim,self.carry_gate_groups,self.token_dim,self.memory_slots,self.key_dim,self.routing)!=(8,32,32,4,8,16,8,16,8,16,'dynamic'):raise ValueError('one fixed N8/H32/K4/M8 typed-action prototype')
        self.controller.weight_ih=torch.nn.Parameter(self.controller.weight_ih[:,:self.token_dim].detach().clone())
        self.controller.input_size=self.token_dim
        self.caller[2]=TypedCallerHead(self.caller[2],self.n_primitives)
        del self.initial_memory_keys,self.read_query,self.write_query,self.write_key,self.write_gate
        factory=dict(device=self.encoder.weight.device,dtype=self.encoder.weight.dtype)
        self.address_proposal=torch.nn.Linear(2*self.internal_states+self.controller_dim+self.token_dim,self.memory_slots,**factory)

    def configuration(self):return dict(super().configuration(),action_policy=POLICY)
    @classmethod
    def from_checkpoint(cls,payload,*,expected_action_policy=POLICY):
        configuration=dict(payload['configuration'])
        if expected_action_policy!=POLICY or payload.get('version')!=CHECKPOINT_VERSION or payload.get('organization_policy')!=POLICY or configuration.get('action_policy')!=POLICY:raise ValueError('exact new typed-action checkpoint required; no legacy reinterpretation')
        t=payload['state_dict']['encoder.weight'];model=cls(**configuration,device=t.device,dtype=t.dtype)
        model.load_state_dict(payload['state_dict'],strict=True);return model

    def initial_carry(self,initial_state):
        h=self.encode_state(initial_state)if initial_state.dtype==torch.long else initial_state
        if h.ndim!=2 or h.shape[1]!=self.internal_states:raise ValueError('initial value shape')
        return dict(h=h,controller=h.new_zeros(len(h),self.controller_dim),
            values=h.new_zeros(len(h),self.memory_slots,self.internal_states),
            address=h.new_zeros(len(h),self.memory_slots),logp=F.log_softmax(self.readout(h),-1))

    def step_embedded(self,carry,x,execution='soft',*,output='logits'):
        if execution not in('soft','native','straight_through')or output not in('logits','probabilities'):raise ValueError('execution/output mode')
        updated,trace=self._transition(carry,x,execution)
        return(updated['logp']if output=='logits'else updated['logp'].exp()),updated,trace

    def _transition(self,carry,raw_x,execution):
        if execution not in('soft','native','straight_through'):raise ValueError('execution mode')
        old_h,c,values,old_address=(carry[k]for k in('h','controller','values','address'))
        c=self.controller(raw_x,c)
        x=raw_x+self.slot_embedding.weight[0]
        attention=F.softmax(old_address,-1)
        read=torch.einsum('bm,bmh->bh',attention,values)
        features=torch.cat((old_h,c,x,read),-1)
        all_logits=self.caller(features);action_logits=all_logits[:,:self.n_primitives+3]
        probabilities=F.softmax(action_logits,-1);qcomp=F.softmax(all_logits[:,:self.n_primitives],-1)
        chosen=action_logits.argmax(-1);chosen_compute=chosen.clamp_max(self.n_primitives-1)
        hard_weights=F.one_hot(chosen,self.n_primitives+3).to(old_h)
        gates=torch.sigmoid(all_logits[:,self.n_primitives+3:self.n_primitives+3+8]).repeat_interleave(self.internal_states//8,-1)
        proposal=self.address_proposal(features)
        # Every path uses the OLD value for STORE and OLD logits for this read.
        def apply(weights,chi):
            compute_total=weights[:,:self.n_primitives].sum(-1,keepdim=True)
            h=old_h+compute_total*(chi-old_h)+weights[:,LOAD:LOAD+1]*(read-old_h)
            amount=weights[:,STORE:STORE+1,None]*attention[:,:,None]
            v=values+amount*(old_h[:,None]-values)
            address=old_address+weights[:,ADDRESS:ADDRESS+1]*(proposal-old_address)
            return h,v,address
        if execution=='native':
            # One candidate is physically computed per row even for a memory
            # action, then multiplied by zero; this work is charged explicitly.
            low=F.gelu(torch.bmm(self.primitive_down[chosen_compute],features[:,:,None]).squeeze(-1)+self.primitive_bias[chosen_compute])
            delta=torch.bmm(self.primitive_up[chosen_compute],low[:,:,None]).squeeze(-1)
            chi=old_h+gates*(self.residual_norm(old_h+delta)-old_h)
            h,v,address=apply(hard_weights,chi);weights=hard_weights
            logp=F.log_softmax(self.readout(h),-1)
        else:
            low=F.gelu(torch.einsum('bf,krf->bkr',features,self.primitive_down)+self.primitive_bias)
            candidates=torch.einsum('bkr,khr->bkh',low,self.primitive_up)
            delta=torch.einsum('bk,bkh->bh',qcomp,candidates)
            soft_chi=old_h+gates*(self.residual_norm(old_h+delta)-old_h)
            soft_h,soft_v,soft_address=apply(probabilities,soft_chi)
            soft_logp=F.log_softmax(self.readout(soft_h),-1)
            if execution=='soft':
                h,v,address,logp,chi,weights=soft_h,soft_v,soft_address,soft_logp,soft_chi,probabilities
            else:
                hard_delta=candidates[torch.arange(len(old_h),device=old_h.device),chosen_compute]
                hard_chi=old_h+gates*(self.residual_norm(old_h+hard_delta)-old_h)
                hard_h,hard_v,hard_address=apply(hard_weights,hard_chi)
                hard_logp=F.log_softmax(self.readout(hard_h),-1)
                h,v,address,logp,chi,weights=(surrogate(a,b)for a,b in((hard_h,soft_h),(hard_v,soft_v),(hard_address,soft_address),(hard_logp,soft_logp),(hard_chi,soft_chi),(hard_weights,probabilities)))
        return dict(h=h,controller=c,values=v,address=address,logp=logp),dict(
            call_probabilities=qcomp,call_weights=weights[:,:self.n_primitives],call_logits=all_logits[:,:self.n_primitives],call_indices=torch.where(chosen<self.n_primitives,chosen,torch.full_like(chosen,-1)),
            action_probabilities=probabilities,action_weights=weights,action_logits=action_logits,action_indices=chosen,
            compute_total=weights[:,:self.n_primitives].sum(-1,keepdim=True),read=read,read_attention=attention,
            write_addresses=attention,write_gate=weights[:,STORE:STORE+1],write_value=old_h,
            address_update_gate=weights[:,ADDRESS:ADDRESS+1],address_proposal=proposal,address_state=address,address_query=old_address,
            state_gate=gates,candidate_h=chi)

    def _sequence(self,initial,embedded,lengths,execution,output,all_outputs,trace):
        b,l,_=embedded.shape;carry=self.initial_carry(initial);history=[]
        states=[carry['h']];controllers=[carry['controller']];value_history=[carry['values']];address_history=[carry['address']];traces=[]
        shapes=dict(call_probabilities=(self.n_primitives,),call_weights=(self.n_primitives,),call_logits=(self.n_primitives,),call_indices=(),action_probabilities=(self.n_primitives+3,),action_weights=(self.n_primitives+3,),action_logits=(self.n_primitives+3,),action_indices=(),compute_total=(1,),read=(self.internal_states,),write_addresses=(self.memory_slots,),write_gate=(1,),address_update_gate=(1,),address_proposal=(self.memory_slots,),state_gate=(self.internal_states,),candidate_h=(self.internal_states,))
        for step in range(l):
            ids=(step<lengths).nonzero(as_tuple=False).flatten()
            if ids.numel():
                selected={key:value.index_select(0,ids)for key,value in carry.items()};updated,t=self._transition(selected,embedded[:,step].index_select(0,ids),execution)
                carry={key:value.index_copy(0,ids,updated[key])for key,value in carry.items()}
            else:t=None
            if all_outputs:history.append(carry['logp'])
            if trace:
                states.append(carry['h']);controllers.append(carry['controller']);value_history.append(carry['values']);address_history.append(carry['address']);full={}
                for key,shape in shapes.items():
                    z=lengths.new_full((b,)+shape,-1)if key.endswith('indices')else carry['h'].new_zeros((b,)+shape)
                    full[key]=z.index_copy(0,ids,t[key])if t is not None else z
                traces.append(full)
        logp=torch.stack(history,1)if all_outputs and l else(carry['h'].new_empty(b,0,self.n_states)if all_outputs else carry['logp'])
        answer=logp if output=='logits'else logp.exp()
        if not trace:return answer,carry
        record=dict(states=torch.stack(states,1),controllers=torch.stack(controllers,1),memory_values=torch.stack(value_history,1),address_states=torch.stack(address_history,1),final_state=carry['h'],final_carry=carry)
        for key,shape in shapes.items():
            if l:record[key]=torch.stack([r[key]for r in traces],1)
            else:record[key]=lengths.new_empty((b,0)+shape)if key.endswith('indices')else carry['h'].new_empty((b,0)+shape)
            if key.startswith('call_')or key=='read':record[key]=record[key].unsqueeze(2)
        record.update(reads=record.pop('read'),write_gates=record.pop('write_gate'),active_mask=torch.arange(l,device=lengths.device)[None]<lengths[:,None],execution=execution,routing=self.routing,action_order=('compute0','compute1','compute2','compute3','LOAD','STORE','ADDRESS'),native_scope='hard7action/expert, soft8memoryattention')
        return answer,carry,record

    def resource_report(self,batch_size=1,sequence_length=0,*,active_steps=None,execution='soft',return_traces=False):
        h,c,t,m,k,r,s=self.internal_states,self.controller_dim,self.token_dim,self.memory_slots,self.n_primitives,self.primitive_rank,self.n_states
        n=batch_size*sequence_length if active_steps is None else active_steps;f=2*h+c+t;executed=1 if execution=='native'else k
        parts=dict(controller=n*3*c*(t+c),read_values=n*m*h,caller=n*(f*self.caller_hidden+self.caller_hidden*(k+3+8)),experts=n*executed*r*(f+h),expert_mixing=0 if execution=='native'else n*k*h,compute_G8_blend=n*h,compute_h_blend=n*h,LOAD_h_blend=n*h,STORE_amount=n*m,STORE_value_blend=n*m*h,address_proposal=n*f*m,ADDRESS_logit_blend=n*m,readout=(n+batch_size)*h*s)
        if execution=='straight_through':parts.update(ST_extra_G8_blend=n*h,ST_extra_h_blend=n*2*h,ST_extra_STORE_amount=n*m,ST_extra_STORE_blend=n*m*h,ST_extra_ADDRESS_blend=n*m,ST_extra_readout=n*h*s)
        scalar=self.encoder.weight.element_size();carry=batch_size*(h+c+m*h+m+s)
        shapes_scalars=3*k+1+3*(k+3)+1+1+h+m+1+1+m+h+h
        return dict(configuration=self.configuration(),total_parameters=self.parameter_count(),learnable_parameters=self.parameter_count(True),parameter_counts={name:p.numel()for name,p in self.named_parameters()},parameter_payload_bytes=sum(p.numel()*p.element_size()for p in self.parameters()),persistent_example_state_scalars=carry,persistent_example_state_payload_bytes=carry*scalar,
            proxy_forward_macs=sum(parts.values()),proxy_forward_macs_by_component=parts,active_steps=n,trace_logical_payload_bytes=batch_size*(sequence_length+1)*(h+c+m*h+m)*scalar+batch_size*sequence_length*((shapes_scalars-2)*scalar+2*8+1) if return_traces else 0,
            typed_actions=dict(action_order=['compute0','compute1','compute2','compute3','LOAD','STORE','ADDRESS'],parameter_tensors=20,carry_fields={'h':h,'controller':c,'values':m*h,'address':m,'logp':s},memory='raw OLDh stored intoMbyH; zeroinitialvalues; no keymatrix/probability/E reconstruction',address='Mdirectlogits initial0; softmax OLDlogits; ADDRESS updatesnexttoken only',controller='originalGRUtoken-columns/inputtokenonly',compute='conditionalsoftmax4 EXACToldexperts/G8; jointalpha7computemass scaleschi',native='hard7action/expert withsoftmemoryattention; onecandidatephysicallycomputedperrow evenmemoryaction, charged',straight_through='hardforward; one-step softwholecarry/cache Jacobian atsameincomingcarry; notgradientofentireallsofttrajectory',legacy_initializer_key_dim=self.key_dim,legacy_key_dim_scope='originalfactorydrawcompatibility ONLY, no K parameters/state',output='cachedheadlogp ofupdatedh; no transition exp(logp)/posteriorwrite',protection='no guaranteed binding/closure/retention; all actionroutes learned from terminalNLL'),
            proxy_scope='explicitforward linear/dot/blend product counts fromsource; biases/nonlinear/LN/softmax/indexing/backward/Adam/workspace excluded; notfullFLOPs',full_forward_FLOPs=None,full_training_FLOPs=None,execution=execution)
