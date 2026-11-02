"""N8 posterior-value Block prototype, freshtrainingonly, no teacher states.

Same old27 NN names/shapes/controller/CALL/G8/keys. Value memory stores
subprobability8-vectors. Each candidate is decoded ONCE to cached logp;
posterior maps through learned normalized input prototypes to the next h.
Continuous soft posterior and full context remain; no protection/closure theorem.
This source is embedded as continuous_primitive_block.py with exact old source
owned_fcba_core.py. Old checkpoints are rejected, never silently converted.
"""
import torch
from torch.nn import functional as F
import owned_fcba_core as base

BASE_SHA256='fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158'
CHECKPOINT_VERSION='continuous-primitive-block-posterior-value-v1'
POLICY='learned-control-history/posterior-value-cache/group8/v1'

class ContinuousPrimitiveBlock(base.ContinuousPrimitiveBlock):
    checkpoint_version=CHECKPOINT_VERSION
    organization_policy=POLICY
    def __init__(self,*args,value_projection='posterior',**kwargs):
        if value_projection not in('posterior','raw'):raise ValueError('two fixed prospective value projections only')
        # Exactly old27 factories/draws: no extra learned parameter or scalar.
        super().__init__(*args,**kwargs)
        self.value_projection=value_projection
        self.organization_policy=POLICY+'/'+value_projection
        if (self.n_states,self.internal_states,self.memory_dim,self.n_primitives,self.primitive_rank,self.controller_dim,self.carry_gate_groups)!=(8,32,32,4,8,16,8):raise ValueError('one fixed N8/H32 posterior-value prototype only')

    def configuration(self):return dict(super().configuration(),value_projection=self.value_projection)

    @classmethod
    def from_checkpoint(cls,payload,*,expected_value_projection=None):
        configuration=dict(payload['configuration']);mode=configuration.get('value_projection')
        if payload.get('version')!=CHECKPOINT_VERSION or mode not in('posterior','raw')or payload.get('organization_policy')!=POLICY+'/'+mode:raise ValueError('exact posterior-value checkpoint policy/configuration required')
        if expected_value_projection is not None and mode!=expected_value_projection:raise ValueError('cross-mode checkpoint forbidden')
        t=payload['state_dict']['encoder.weight'];model=cls(**configuration,device=t.device,dtype=t.dtype)
        model.load_state_dict(payload['state_dict'],strict=True);return model

    def initial_carry(self,initial_state):
        h=self.encode_state(initial_state)if initial_state.dtype==torch.long else initial_state
        if h.ndim!=2 or h.shape[1]!=self.internal_states:raise ValueError('initial value shape')
        # Empty sequences retain ordinary learned readout(E(initial)); no
        # identity prior or hard one-hot output is supplied. h/posterior-value
        # equality is asserted AFTER transitions, not at the empty edge.
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
        c=self.controller(torch.cat((raw_x,h),-1),c)
        x=raw_x+self.slot_embedding.weight[0];control=torch.cat((c,x),-1)
        attention=F.softmax(torch.einsum("bq,bmq->bm",self.read_query(control),keys)/self.key_dim**.5,-1)
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
        projected_h=posterior@basis if self.value_projection=="posterior" else None
        h=projected_h if self.value_projection=="posterior" else candidate_h
        write_features=torch.cat((c,raw_x),-1)
        address=F.softmax(torch.einsum("bq,bmq->bm",self.write_query(write_features),keys)/self.key_dim**.5,-1)
        write_gate=torch.sigmoid(self.write_gate(write_features));amount=(write_gate*address)[:,:,None]
        keys=keys+amount*(torch.tanh(self.write_key(write_features))[:,None]-keys)
        values=values+amount*(posterior[:,None]-values)
        return dict(h=h,controller=c,keys=keys,values=values,logp=logp),dict(call_probabilities=probabilities,call_weights=weights,
            call_logits=logits,call_indices=chosen,read=read,read_attention=attention,write_addresses=address,
            write_gate=write_gate,state_gate=gates,candidate_h=candidate_h,posterior=posterior,read_p=read_p,projected_h=projected_h)

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
        n=batch_size*sequence_length if active_steps is None else active_steps;h,s,m,q,c=self.internal_states,self.n_states,self.memory_slots,self.key_dim,self.controller_dim
        parts=dict(report['proxy_forward_macs_by_component'])
        parts['read_values']=n*(m*s+s*h)
        parts['posterior_to_value']=n*s*h if self.value_projection=='posterior'else 0
        parts['readout']=(n+batch_size)*h*s
        parts['write_key_value_blend']=n*2*m*(q+s)
        scalar=self.encoder.weight.element_size();carry=batch_size*(h+c+m*(q+s)+s)
        report.update(proxy_forward_macs_by_component=parts,proxy_forward_macs=sum(parts.values()),
            persistent_example_state_scalars=carry,persistent_example_state_payload_bytes=carry*scalar,
            trace_logical_payload_bytes=(batch_size*(sequence_length+1)*(h+c+m*(q+s))+n*(3*self.n_primitives+h+m+1))*scalar if return_traces else 0,
            value_memory='M by S subprobability rows, zero initialized; convex posterior write; read_p attention mixture then learnedE',
            posterior_value=dict(states=s,value_width=h,cache_width=s,memory_value_width=s,value_projection=self.value_projection,
                carry_fields={'h':h,'controller':c,'keys':m*q,'values':m*s,'logp':s},
                decoded_candidate_positions=n,initial_cache_decodes=batch_size,posterior_exponentials=n*s,
                posterior_projection_MAC=n*s*h if self.value_projection=='posterior'else 0,read_p_to_value_MAC=n*s*h,projection_forward_computed=self.value_projection=='posterior',
                encoder_basis_normalization_elements_per_transition_call=s*h,
                initial_empty_output='ordinary readout(encode(initial)), no identity guarantee and no posteriorh constraint beforefirsttransition',
                output_rule='cachelogp from candidateh; publicstep/sequence/embedded/alltoken API never redecodes projectedh',
                bound='POSTERIORmodeonly: aftertransition ||h-E_y|| <=2sqrtH*(1-p_y) conditionalnormalizedlearnedprototypes; RAWmode has no suchhbound; neitherfullcarryclosure/retention/protection',
                transient_scope='candidateh32/logp8/posterior8/read_p8 andbasisSxH are computedworkspace, notaddedNNparameters; allocator/autograd/capture peaks measured separately'),
            full_training_FLOPs=None,full_forward_FLOPs=None,
            proxy_scope='forwardlinear/dot/blend MAC including initialcache andpertransitioncandidate decoder/projection; logsoftmax/exp/norm/nonlinear/backward/index/workspace excluded, notfullFLOPs')
        return report
