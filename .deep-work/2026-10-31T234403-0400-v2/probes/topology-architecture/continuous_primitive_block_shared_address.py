"""Input-history shared-address Block, one fresh N8 development architecture.

RAW candidate h, old learned CALL/operators/G8, cached candidate output, and
probability values remain. Controller receives tokens ONLY. Read and write
share one query/attention against trainable keys fixed within each episode.
No correct address/role/token class or teacher state is supplied.
"""
import torch
from torch.nn import functional as F
import owned_fcba_core as base
BASE_SHA256='fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158'
CHECKPOINT_VERSION='continuous-primitive-block-input-history-shared-address-v1'
POLICY='input-history/shared-address/static-episode-keys/probvalue-cache/group8/v1'

class ContinuousPrimitiveBlock(base.ContinuousPrimitiveBlock):
    checkpoint_version=CHECKPOINT_VERSION
    organization_policy=POLICY
    def __init__(self,*args,address_policy=POLICY,**kwargs):
        if address_policy!=POLICY:raise ValueError('one fixed input-history shared-address policy')
        # Execute exact old initialization/draw sequence before pruning.
        super().__init__(*args,**kwargs)
        if (self.n_states,self.internal_states,self.memory_dim,self.n_primitives,self.primitive_rank,self.controller_dim,self.carry_gate_groups,self.token_dim,self.memory_slots,self.key_dim)!=(8,32,32,4,8,16,8,16,8,16):raise ValueError('one fixed N8/H32 architecture only')
        # This is algebraically the old GRU(cat(raw_x,zeros_like(h)),c), with
        # its original token-column initialization subset and no dead h columns.
        self.controller.weight_ih=torch.nn.Parameter(self.controller.weight_ih[:,:self.token_dim].detach().clone())
        self.controller.input_size=self.token_dim
        del self.write_query,self.write_key
    def configuration(self):return dict(super().configuration(),address_policy=POLICY)
    @classmethod
    def from_checkpoint(cls,payload,*,expected_address_policy=POLICY):
        configuration=dict(payload['configuration'])
        if expected_address_policy!=POLICY or payload.get('version')!=CHECKPOINT_VERSION or payload.get('organization_policy')!=POLICY or configuration.get('address_policy')!=POLICY:raise ValueError('exact new shared-address checkpoint required; no legacy reinterpretation')
        t=payload['state_dict']['encoder.weight'];model=cls(**configuration,device=t.device,dtype=t.dtype)
        model.load_state_dict(payload['state_dict'],strict=True);return model

    def initial_carry(self,initial_state):
        h=self.encode_state(initial_state)if initial_state.dtype==torch.long else initial_state
        if h.ndim!=2 or h.shape[1]!=self.internal_states:raise ValueError('initial value shape')
        # Empty sequences retain ordinary learned readout(E(initial)); no
        # identity prior or hard one-hot output is supplied. h is the RAW value;
        # only the candidate classifier output is cached after transitions.
        return dict(h=h,controller=h.new_zeros(len(h),self.controller_dim),
            keys=self.initial_memory_keys[None].expand(len(h),-1,-1),
            values=h.new_zeros(len(h),self.memory_slots,self.n_states),
            logp=F.log_softmax(self.readout(h),-1))

    def step_embedded(self,carry,x,execution='soft',*,output='logits'):
        if execution not in('soft','native','straight_through')or output not in('logits','probabilities'):raise ValueError('execution/output mode')
        updated,trace=self._transition(carry,x,execution)
        return (updated['logp']if output=='logits'else updated['logp'].exp()),updated,trace

    def _transition(self,carry,raw_x,execution):
        h,c,keys,values=(carry[k] for k in ("h","controller","keys","values"))
        c=self.controller(raw_x,c)
        x=raw_x+self.slot_embedding.weight[0];control=torch.cat((c,x),-1)
        address_query=self.read_query(control)
        attention=F.softmax(torch.einsum("bq,bmq->bm",address_query,keys)/self.key_dim**.5,-1)
        basis=F.normalize(self.encoder.weight,dim=-1,eps=1e-6)*self.internal_states**.5
        read_p=torch.einsum("bm,bms->bs",attention,values)
        read=read_p@basis
        features=torch.cat((h,c,x,read),-1)
        call_features=features if self.routing=="dynamic" else torch.cat((torch.zeros_like(h),torch.zeros_like(c),x,torch.zeros_like(read)),-1)
        all_logits=self.caller(call_features);logits=all_logits[:,:self.n_primitives]
        probabilities=F.softmax(logits,-1);chosen=logits.argmax(-1)
        if execution=="native":
            low=F.gelu(torch.bmm(self.primitive_down[chosen],features[:,:,None]).squeeze(-1)+self.primitive_bias[chosen])
            delta=torch.bmm(self.primitive_up[chosen],low[:,:,None]).squeeze(-1)
            weights=F.one_hot(chosen,self.n_primitives).to(h)
        else:
            low=F.gelu(torch.einsum("bf,krf->bkr",features,self.primitive_down)+self.primitive_bias)
            candidates=torch.einsum("bkr,khr->bkh",low,self.primitive_up)
            delta=torch.einsum("bk,bkh->bh",probabilities,candidates);weights=probabilities
            if execution=="straight_through":
                hard=candidates[torch.arange(len(h),device=h.device),chosen]
                delta=hard.detach()+(delta-delta.detach())
                onehot=F.one_hot(chosen,self.n_primitives).to(h)
                weights=onehot+(probabilities-probabilities.detach())
        gates=torch.sigmoid(all_logits[:,self.n_primitives:]).repeat_interleave(self.internal_states//8,-1)
        candidate_h=h+gates*(self.residual_norm(h+delta)-h)
        logp=F.log_softmax(self.readout(candidate_h),-1)
        posterior=logp.exp()
        h=candidate_h
        write_features=torch.cat((c,raw_x),-1)
        address=attention
        write_gate=torch.sigmoid(self.write_gate(write_features));amount=(write_gate*address)[:,:,None]
        values=values+amount*(posterior[:,None]-values)
        return dict(h=h,controller=c,keys=keys,values=values,logp=logp),dict(call_probabilities=probabilities,call_weights=weights,
            call_logits=logits,call_indices=chosen,read=read,read_attention=attention,write_addresses=address,
            write_gate=write_gate,state_gate=gates,candidate_h=candidate_h,posterior=posterior,read_p=read_p,address_query=address_query)

    def _sequence(self,initial,embedded,lengths,execution,output,all_outputs,trace):
        b,l,_=embedded.shape;carry=self.initial_carry(initial);history=[];states=[carry["h"]];controllers=[carry["controller"]]
        key_history=[carry["keys"]];value_history=[carry["values"]];traces=[]
        for step in range(l):
            ids=(step<lengths).nonzero(as_tuple=False).flatten()
            if ids.numel():
                selected={key:value.index_select(0,ids) for key,value in carry.items()}
                updated,t=self._transition(selected,embedded[:,step].index_select(0,ids),execution)
                carry={key:value.index_copy(0,ids,updated[key]) for key,value in carry.items()}
            else:t=None
            if all_outputs:history.append(carry["logp"])
            if trace:
                states.append(carry["h"]);controllers.append(carry["controller"]);key_history.append(carry["keys"]);value_history.append(carry["values"])
                full={}
                shapes=dict(call_probabilities=(self.n_primitives,),call_weights=(self.n_primitives,),call_logits=(self.n_primitives,),call_indices=(),read=(self.internal_states,),write_addresses=(self.memory_slots,),write_gate=(1,))
                for key,shape in shapes.items():
                    z=carry["h"].new_zeros((b,)+shape) if key!="call_indices" else lengths.new_full((b,),-1)
                    full[key]=z.index_copy(0,ids,t[key]) if t is not None else z
                traces.append(full)
        logp=torch.stack(history,1) if all_outputs and l else (carry["h"].new_empty(b,0,self.n_states) if all_outputs else carry["logp"])
        answer=logp if output=="logits" else logp.exp()
        if not trace:return answer,carry
        record=dict(states=torch.stack(states,1),controllers=torch.stack(controllers,1),memory_keys=torch.stack(key_history,1),memory_values=torch.stack(value_history,1),final_state=carry["h"],final_carry=carry)
        for key in ("call_probabilities","call_weights","call_logits","call_indices","read","write_addresses","write_gate"):
            if l:
                record[key]=torch.stack([x[key] for x in traces],1)
                if key.startswith("call_") or key=="read":record[key]=record[key].unsqueeze(2)
        record.update(reads=record.pop("read",None),write_gates=record.pop("write_gate",None),active_mask=torch.arange(l,device=lengths.device)[None]<lengths[:,None],execution=execution,routing=self.routing)
        return answer,carry,record

    def resource_report(self,batch_size=1,sequence_length=0,*,active_steps=None,execution='soft',return_traces=False):
        report=super().resource_report(batch_size,sequence_length,active_steps=active_steps,execution=execution,return_traces=return_traces)
        n=batch_size*sequence_length if active_steps is None else active_steps;h,s,m,q,c,t=self.internal_states,self.n_states,self.memory_slots,self.key_dim,self.controller_dim,self.token_dim
        parts=dict(report['proxy_forward_macs_by_component'])
        parts.update(controller=n*3*c*(t+c),read_values=n*(m*s+s*h),write_projections=n*(c+t),write_scores=0,write_key_value_blend=n*2*m*s,readout=(n+batch_size)*h*s)
        scalar=self.encoder.weight.element_size();carry=batch_size*(h+c+m*(q+s)+s)
        report.update(proxy_forward_macs_by_component=parts,proxy_forward_macs=sum(parts.values()),
            persistent_example_state_scalars=carry,persistent_example_state_payload_bytes=carry*scalar,
            trace_logical_payload_bytes=(batch_size*(sequence_length+1)*(h+c+m*(q+s))+n*(3*self.n_primitives+h+m+1))*scalar if return_traces else 0,
            value_write='cached candidate posterior probability, convex blend; h remains RAW candidate',
            addressing=dict(controller='GRU(rawtoken,c), originaltokenweightcolumns; no h dependency',controller_input_width=t,removed_zero_input_parameters=3*c*h,removed_write_query_key_parameters=2*((c+t)*q+q),parameter_tensors=23,
                read_write='one read_query(cat(c,raw_x+slot)) and ONE attention shared read/write',keys='learned initial_memory_keys fixed throughout episode, trainable across SGD',
                input_information='raw input history ONLY; no logical address/role/token labels, no truth or binding loss',
                carry_fields={'h':h,'controller':c,'keys':m*q,'values':m*s,'logp':s},
                logical_carry_scope='248 floats/example includes128 statickeys; expandedinitialkeys mayshare parameterstorage but gather/masked copies and graphallocator charged',
                cached_output='candidate_h head decoded once; no second h decoder',value_projection='raw'),
            full_training_FLOPs=None,full_forward_FLOPs=None,
            proxy_scope='forward linear/dot/blend including one sharedscore/cache; nonlinear/exp/norm/backward/Adam/index/workspace excluded; notfullFLOPs')
        return report
