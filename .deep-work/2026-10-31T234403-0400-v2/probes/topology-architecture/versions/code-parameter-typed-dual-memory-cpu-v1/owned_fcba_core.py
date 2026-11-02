"""Development: separate control history, executable value and content memory.

All token meanings, routes, values and addresses are learned from terminal
feedback. No legacy checkpoint conversion and no topological certificate.
Known GRU/content-memory/low-rank expert ingredients, not a novelty claim.
"""
from dataclasses import dataclass, asdict
from typing import Mapping
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ContinuousPrimitiveConfig:
    n_states:int
    n_tokens:int
    internal_states:int=32
    n_primitives:int=4
    token_dim:int=16
    state_dim:int=32
    memory_dim:int=32
    caller_hidden:int=32
    n_slots:int=1
    routing:str="dynamic"
    initialization_std:float=.02
    memory_slots:int=8
    key_dim:int=16
    primitive_rank:int=8
    controller_dim:int=16
    carry_gate_groups:int=8


class ContinuousPrimitiveBlock(nn.Module):
    checkpoint_version="continuous-primitive-block-control-value-v1"
    organization_policy="learned-control-history/value-identity-write/group8-carry-v1"

    def __init__(self,n_states,n_tokens,internal_states=32,n_primitives=4,
                 token_dim=16,state_dim=None,memory_dim=None,caller_hidden=32,
                 n_slots=1,routing="dynamic",initialization_std=.02,
                 memory_slots=8,key_dim=16,primitive_rank=8,controller_dim=16,
                 carry_gate_groups=8,*,device=None,dtype=None):
        super().__init__()
        state_dim=internal_states if state_dim is None else state_dim
        memory_dim=internal_states if memory_dim is None else memory_dim
        values=locals().copy()
        fields=ContinuousPrimitiveConfig.__dataclass_fields__
        cfg={name:values[name] for name in fields}
        for name,value in cfg.items():
            if name not in ("routing","initialization_std") and (not isinstance(value,int) or isinstance(value,bool) or value<1):
                raise ValueError("positive integer dimensions required")
        if state_dim!=internal_states or memory_dim!=internal_states or n_slots!=1:
            raise ValueError("one call per token; value/read widths equal internal_states")
        if routing not in ("dynamic","static") or not 0<initialization_std<1:
            raise ValueError("dynamic/static and finite positive initialization required")
        if carry_gate_groups!=8 or internal_states%8:raise ValueError("eight arbitrary equal carry groups")
        self.config=ContinuousPrimitiveConfig(**cfg)
        for name,value in cfg.items():setattr(self,name,value)
        h,c,t,q,m,k,r=internal_states,controller_dim,token_dim,key_dim,memory_slots,n_primitives,primitive_rank
        f=2*h+c+t;control=c+t;factory=dict(device=device,dtype=dtype)
        self.encoder=nn.Embedding(n_states,h,**factory)
        self.token_embedding=nn.Embedding(n_tokens,t,**factory)
        self.slot_embedding=nn.Embedding(1,t,**factory)
        self.controller=nn.GRUCell(t+h,c,**factory)
        self.initial_memory_keys=nn.Parameter(torch.empty(m,q,**factory))
        self.read_query=nn.Linear(control,q,**factory)
        self.caller=nn.Sequential(nn.Linear(f,caller_hidden,**factory),nn.Tanh(),nn.Linear(caller_hidden,k+8,**factory))
        self.primitive_down=nn.Parameter(torch.empty(k,r,f,**factory))
        self.primitive_bias=nn.Parameter(torch.zeros(k,r,**factory))
        self.primitive_up=nn.Parameter(torch.empty(k,h,r,**factory))
        self.residual_norm=nn.LayerNorm(h,**factory)
        self.write_query=nn.Linear(control,q,**factory)
        self.write_key=nn.Linear(control,q,**factory)
        self.write_gate=nn.Linear(control,1,**factory)
        self.readout=nn.Linear(h,n_states,**factory)
        nn.init.normal_(self.encoder.weight,std=initialization_std)
        nn.init.normal_(self.initial_memory_keys,std=initialization_std)
        for down,up in zip(self.primitive_down,self.primitive_up):
            nn.init.kaiming_uniform_(down,a=5**.5);nn.init.kaiming_uniform_(up,a=5**.5)
        with torch.no_grad():
            self.caller[2].weight[k:].zero_();self.caller[2].bias[k:].fill_(4.)

    def configuration(self):return asdict(self.config)
    def parameter_count(self,trainable_only=False):
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

    def encode_state(self,indices):
        # Same learned value interface at supplied initial states and observed
        # terminal states. No intermediate state or operation is provided.
        return F.normalize(self.encoder(indices),dim=-1,eps=1e-6)*self.internal_states**.5

    def initial_carry(self,initial_state):
        h=self.encode_state(initial_state) if initial_state.dtype==torch.long else initial_state
        if h.ndim!=2 or h.shape[1]!=self.internal_states:raise ValueError("initial value shape")
        return dict(h=h,controller=h.new_zeros(len(h),self.controller_dim),
            keys=self.initial_memory_keys[None].expand(len(h),-1,-1),
            values=h.new_zeros(len(h),self.memory_slots,self.internal_states))

    def _transition(self,carry,raw_x,execution):
        h,c,keys,values=(carry[k] for k in ("h","controller","keys","values"))
        c=self.controller(torch.cat((raw_x,h),-1),c)
        x=raw_x+self.slot_embedding.weight[0];control=torch.cat((c,x),-1)
        attention=F.softmax(torch.einsum("bq,bmq->bm",self.read_query(control),keys)/self.key_dim**.5,-1)
        read=torch.einsum("bm,bmh->bh",attention,values)
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
        h=h+gates*(self.residual_norm(h+delta)-h)
        write_features=torch.cat((c,raw_x),-1)
        address=F.softmax(torch.einsum("bq,bmq->bm",self.write_query(write_features),keys)/self.key_dim**.5,-1)
        write_gate=torch.sigmoid(self.write_gate(write_features));amount=(write_gate*address)[:,:,None]
        keys=keys+amount*(torch.tanh(self.write_key(write_features))[:,None]-keys)
        values=values+amount*(h[:,None]-values)
        return dict(h=h,controller=c,keys=keys,values=values),dict(call_probabilities=probabilities,call_weights=weights,
            call_logits=logits,call_indices=chosen,read=read,read_attention=attention,write_addresses=address,
            write_gate=write_gate,state_gate=gates)

    def step_embedded(self,carry,x,execution="soft",*,output="logits"):
        if execution not in ("soft","native","straight_through"):raise ValueError("execution mode")
        if output not in ("logits","probabilities"):raise ValueError("output mode")
        updated,trace=self._transition(carry,x,execution)
        logp=F.log_softmax(self.readout(updated["h"]),-1)
        return (logp if output=="logits" else logp.exp()),updated,trace

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
            if all_outputs:history.append(F.log_softmax(self.readout(carry["h"]),-1))
            if trace:
                states.append(carry["h"]);controllers.append(carry["controller"]);key_history.append(carry["keys"]);value_history.append(carry["values"])
                full={}
                shapes=dict(call_probabilities=(self.n_primitives,),call_weights=(self.n_primitives,),call_logits=(self.n_primitives,),call_indices=(),read=(self.internal_states,),write_addresses=(self.memory_slots,),write_gate=(1,))
                for key,shape in shapes.items():
                    z=carry["h"].new_zeros((b,)+shape) if key!="call_indices" else lengths.new_full((b,),-1)
                    full[key]=z.index_copy(0,ids,t[key]) if t is not None else z
                traces.append(full)
        logp=torch.stack(history,1) if all_outputs and l else (carry["h"].new_empty(b,0,self.n_states) if all_outputs else F.log_softmax(self.readout(carry["h"]),-1))
        answer=logp if output=="logits" else logp.exp()
        if not trace:return answer,carry
        record=dict(states=torch.stack(states,1),controllers=torch.stack(controllers,1),memory_keys=torch.stack(key_history,1),memory_values=torch.stack(value_history,1),final_state=carry["h"],final_carry=carry)
        for key in ("call_probabilities","call_weights","call_logits","call_indices","read","write_addresses","write_gate"):
            if l:
                record[key]=torch.stack([x[key] for x in traces],1)
                if key.startswith("call_") or key=="read":record[key]=record[key].unsqueeze(2)
        record.update(reads=record.pop("read",None),write_gates=record.pop("write_gate",None),active_mask=torch.arange(l,device=lengths.device)[None]<lengths[:,None],execution=execution,routing=self.routing)
        return answer,carry,record

    def _validate(self,initial,tokens,lengths):
        if initial.dtype!=torch.long or tokens.dtype!=torch.long or lengths.dtype!=torch.long:raise TypeError("long tensors required")
        if tokens.ndim!=2 or initial.shape!=(len(tokens),) or lengths.shape!=initial.shape:raise ValueError("batch shapes")
        if bool(((initial<0)|(initial>=self.n_states)).any()) or bool(((lengths<0)|(lengths>tokens.shape[1])).any()):raise ValueError("state/length range")
        mask=torch.arange(tokens.shape[1],device=tokens.device)[None]<lengths[:,None]
        if bool((mask&((tokens<0)|(tokens>=self.n_tokens))).any()):raise ValueError("token range")

    def forward(self,initial,tokens,lengths,execution="soft",*,return_traces=False,output="probabilities"):
        if execution not in ("soft","native","straight_through") or output not in ("logits","probabilities"):raise ValueError("execution/output")
        self._validate(initial,tokens,lengths)
        result=self._sequence(initial,self.token_embedding(tokens),lengths,execution,output,False,return_traces)
        return (result[0],result[2]) if return_traces else result[0]

    def forward_all(self,initial,tokens,lengths,execution="soft",*,output="logits",return_carry=False):
        self._validate(initial,tokens,lengths)
        result=self._sequence(initial,self.token_embedding(tokens),lengths,execution,output,True,False)
        return result if return_carry else result[0]

    def forward_value(self,initial,tokens,lengths,execution="soft"):
        self._validate(initial,tokens,lengths)
        return self._sequence(initial,self.token_embedding(tokens),lengths,execution,"logits",False,False)

    def forward_embedded(self,initial_state,embedded_inputs,lengths,execution="soft",*,return_traces=False,output="logits"):
        if embedded_inputs.shape[2]!=self.token_dim:raise ValueError("input interface width")
        result=self._sequence(initial_state,embedded_inputs,lengths,execution,output,False,return_traces)
        return (result[0],result[2]) if return_traces else result[0]

    def resource_report(self,batch_size=1,sequence_length=0,*,active_steps=None,execution="soft",return_traces=False):
        h,c,t,q,m,k,r=self.internal_states,self.controller_dim,self.token_dim,self.key_dim,self.memory_slots,self.n_primitives,self.primitive_rank
        n=batch_size*sequence_length if active_steps is None else active_steps;f=2*h+c+t;a=c+t;executed_k=1 if execution=="native" else k
        parts=dict(controller=n*3*c*(t+h+c),read_query=n*a*q,read_score=n*m*q,read_values=n*m*h,
            caller=n*(f*self.caller_hidden+self.caller_hidden*(k+8)),experts=n*executed_k*r*(f+h),
            expert_mixing=0 if execution=="native" else n*k*h,value_carry_blend=n*h,
            write_projections=n*a*(2*q+1),write_scores=n*m*q,write_key_value_blend=n*2*m*(q+h),
            readout=batch_size*h*self.n_states)
        scalar=self.encoder.weight.element_size();carry=batch_size*(h+c+m*(q+h))
        return dict(configuration=self.configuration(),total_parameters=self.parameter_count(),learnable_parameters=self.parameter_count(True),
            parameter_counts={n:p.numel()for n,p in self.named_parameters()},parameter_payload_bytes=sum(p.numel()*p.element_size()for p in self.parameters()),
            persistent_example_state_scalars=carry,persistent_example_state_payload_bytes=carry*scalar,
            proxy_forward_macs=sum(parts.values()),proxy_forward_macs_by_component=parts,active_steps=n,
            trace_logical_payload_bytes=(batch_size*(sequence_length+1)*(h+c+m*(q+h))+n*(3*k+h+m+1))*scalar if return_traces else 0,
            proxy_scope="forward linear/dot/blend products only; GRU gates/nonlinear/LN/softmax/backward/indexing/workspace not complete FLOPs",
            value_write="identity of newly computed h; removes learned write-value projection, not zero write cost",
            protection="ordinary parameter sharing/replay; no guaranteed topological protection",execution=execution)

    def checkpoint(self):
        return dict(version=self.checkpoint_version,organization_policy=self.organization_policy,configuration=self.configuration(),
            state_dict={n:p.detach().cpu().clone()for n,p in self.state_dict().items()},resources=self.resource_report())

    @classmethod
    def from_checkpoint(cls,payload:Mapping):
        if payload.get("version")!=cls.checkpoint_version or payload.get("organization_policy")!=cls.organization_policy:raise ValueError("only exact new organization; no legacy reinterpretation")
        t=payload["state_dict"]["encoder.weight"];model=cls(**payload["configuration"],device=t.device,dtype=t.dtype)
        model.load_state_dict(payload["state_dict"],strict=True);return model
