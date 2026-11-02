"""Continuous residual primitives with contextual calls and content memory.

An explicit development architecture variant, NOT frozen Primary, not a claim
of a novel rule or topological protection. Categorical-source failures remain
unchanged. Continuous state avoids repeated stochastic-matrix contraction;
LayerNorm and learned nonlinear operations are not guaranteed injective.

Each call reads memory, learns a mixture of low-rank nonlinear residual edits,
and normalizes the residual state. There is no explicit identity/no-op route.
Memory keys/values and state reset per example; only initial keys are learned.
Terminal feedback can train E, operations, caller, memory writes and R together.
No task semantics, operator, route or intermediate target is supplied.

Native hardens CALL ONLY; h, E/R, memory and primitive functions remain
continuous. ST has the same hard-call forward but uses a dense soft-mixture
surrogate Jacobian. Soft/native equivalence is NOT a mathematical assumption.
No import-time run, network, subprocess or installation.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass
from typing import Any,Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ContinuousPrimitiveConfig:
    n_states:int
    n_tokens:int
    internal_states:int=64
    n_primitives:int=4
    token_dim:int=16
    state_dim:int=64
    memory_dim:int=64
    caller_hidden:int=32
    n_slots:int=1
    routing:str="dynamic"
    initialization_std:float=.02
    memory_slots:int=8
    key_dim:int=16
    primitive_rank:int=8


class ContinuousPrimitiveBlock(nn.Module):
    """State h[B,H], memory keys[B,M,Q], values[B,M,H], K residual experts.

    Same forward/checkpoint/resource API as ContextualPrimitiveBlock. Here
    internal_states is continuous H, not a categorical state count. state_dim
    and memory_dim are compatibility names for H and must equal it if given.
    Static zeros ONLY h/read inputs to CALL; experts and memory retain their
    full context, and both arms execute identical shapes and own the same P.
    """
    checkpoint_version="continuous-primitive-block-v1"

    def __init__(self,n_states:int,n_tokens:int,internal_states:int=64,
                 n_primitives:int=4,token_dim:int=16,state_dim:int|None=None,
                 memory_dim:int|None=None,caller_hidden:int=32,n_slots:int=1,
                 routing:str="dynamic",initialization_std:float=.02,
                 memory_slots:int=8,key_dim:int=16,primitive_rank:int=8,
                 *,device:Any=None,dtype:torch.dtype|None=None):
        super().__init__()
        state_dim=internal_states if state_dim is None else state_dim
        memory_dim=internal_states if memory_dim is None else memory_dim
        dims=dict(n_states=n_states,n_tokens=n_tokens,internal_states=internal_states,
            n_primitives=n_primitives,token_dim=token_dim,state_dim=state_dim,memory_dim=memory_dim,
            caller_hidden=caller_hidden,n_slots=n_slots,memory_slots=memory_slots,key_dim=key_dim,
            primitive_rank=primitive_rank)
        if any(not isinstance(v,int) or isinstance(v,bool) or v<1 for v in dims.values()):
            raise ValueError("positive integer dimensions required")
        if state_dim!=internal_states or memory_dim!=internal_states:
            raise ValueError("continuous state/read dimensions must equal internal_states")
        if routing not in ("dynamic","static") or not 0<initialization_std<1:
            raise ValueError("dynamic/static and finite positive initialization_std below1 required")
        self.config=ContinuousPrimitiveConfig(**dims,routing=routing,initialization_std=initialization_std)
        for name,value in dims.items():setattr(self,name,value)
        self.routing=routing
        factory=dict(device=device,dtype=dtype)
        h,q,m,k,r=internal_states,key_dim,memory_slots,n_primitives,primitive_rank
        features=2*h+token_dim
        self.encoder=nn.Embedding(n_states,h,**factory)
        self.token_embedding=nn.Embedding(n_tokens,token_dim,**factory)
        self.slot_embedding=nn.Embedding(n_slots,token_dim,**factory)
        self.initial_memory_keys=nn.Parameter(torch.empty(m,q,**factory))
        self.read_query=nn.Linear(h+token_dim,q,**factory)
        self.caller=nn.Sequential(nn.Linear(features,caller_hidden,**factory),nn.Tanh(),
                                  nn.Linear(caller_hidden,k,**factory))
        self.primitive_down=nn.Parameter(torch.empty(k,r,features,**factory))
        self.primitive_bias=nn.Parameter(torch.zeros(k,r,**factory))
        self.primitive_up=nn.Parameter(torch.empty(k,h,r,**factory))
        self.residual_norm=nn.LayerNorm(h,**factory)
        self.write_query=nn.Linear(features,q,**factory)
        self.write_key=nn.Linear(features,q,**factory)
        self.write_value=nn.Linear(features,h,**factory)
        self.write_gate=nn.Linear(features,1,**factory)
        self.readout=nn.Linear(h,n_states,**factory)
        self.reset_parameters()

    def reset_parameters(self):
        """Data-independent random learned interfaces; no identity or task map."""
        nn.init.normal_(self.encoder.weight,std=self.config.initialization_std)
        self.token_embedding.reset_parameters();self.slot_embedding.reset_parameters()
        nn.init.normal_(self.initial_memory_keys,std=self.config.initialization_std)
        for layer in (self.read_query,self.write_query,self.write_key,self.write_value,self.write_gate,self.readout):
            layer.reset_parameters()
        for layer in self.caller:
            if hasattr(layer,"reset_parameters"):layer.reset_parameters()
        for down,up in zip(self.primitive_down,self.primitive_up):
            nn.init.kaiming_uniform_(down,a=5**.5);nn.init.kaiming_uniform_(up,a=5**.5)
        nn.init.zeros_(self.primitive_bias);self.residual_norm.reset_parameters()

    def configuration(self):
        result=asdict(self.config);result["routing"]=self.routing;return result

    def set_routing(self,routing):
        if routing not in ("dynamic","static"):raise ValueError("dynamic/static required")
        self.routing=routing

    def _validate(self,initial,tokens,lengths):
        values=(initial,tokens,lengths)
        if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.long for v in values):
            raise TypeError("initial,tokens,lengths must be Long tensors")
        if initial.ndim!=1 or tokens.ndim!=2 or lengths.shape!=initial.shape or len(tokens)!=len(initial):
            raise ValueError("initial[B],tokens[B,L],lengths[B] required")
        if any(v.device!=self.encoder.weight.device for v in values):raise ValueError("model/input device differs")
        if bool(((initial<0)|(initial>=self.n_states)).any()):raise ValueError("initial outside public states")
        if bool(((lengths<0)|(lengths>tokens.shape[1])).any()):raise ValueError("length outside supplied extent")
        active=torch.arange(tokens.shape[1],device=tokens.device)[None]<lengths[:,None]
        if bool((active&((tokens<0)|(tokens>=self.n_tokens))).any()):raise ValueError("active token outside public vocabulary")

    def _read(self,h,x,keys,values):
        query=self.read_query(torch.cat((h,x),-1))
        attention=F.softmax(torch.einsum("bq,bmq->bm",query,keys)/self.key_dim**.5,-1)
        return torch.einsum("bm,bmh->bh",attention,values),attention

    def _call(self,h,x,read,execution):
        call_h=h if self.routing=="dynamic" else torch.zeros_like(h)
        call_r=read if self.routing=="dynamic" else torch.zeros_like(read)
        logits=self.caller(torch.cat((call_h,x,call_r),-1))
        probabilities=F.softmax(logits,-1);chosen=logits.argmax(-1)
        features=torch.cat((h,x,read),-1)
        if execution=="native":
            # Actual hard-call evaluation computes only the selected expert.
            down=self.primitive_down[chosen];up=self.primitive_up[chosen]
            bottleneck=torch.bmm(down,features[:,:,None]).squeeze(-1)+self.primitive_bias[chosen]
            delta=torch.bmm(up,F.gelu(bottleneck)[:,:,None]).squeeze(-1)
            weights=F.one_hot(chosen,self.n_primitives).to(h)
        else:
            low=F.gelu(torch.einsum("bf,krf->bkr",features,self.primitive_down)+self.primitive_bias)
            candidates=torch.einsum("bkr,khr->bkh",low,self.primitive_up)
            soft_delta=torch.einsum("bk,bkh->bh",probabilities,candidates)
            if execution=="straight_through":
                hard_delta=candidates[torch.arange(len(h),device=h.device),chosen]
                delta=hard_delta.detach()+(soft_delta-soft_delta.detach())
                hard=F.one_hot(chosen,self.n_primitives).to(h)
                weights=hard+(probabilities-probabilities.detach())
            else:delta=soft_delta;weights=probabilities
        return self.residual_norm(h+delta),probabilities,weights,logits,chosen

    def _write(self,h,x,read,keys,values):
        features=torch.cat((h,x,read),-1)
        query=self.write_query(features)
        address=F.softmax(torch.einsum("bq,bmq->bm",query,keys)/self.key_dim**.5,-1)
        gate=torch.sigmoid(self.write_gate(features))
        amount=(gate*address)[:,:,None]
        new_key=torch.tanh(self.write_key(features))[:,None,:]
        new_value=self.write_value(features)[:,None,:]
        return keys+amount*(new_key-keys),values+amount*(new_value-values),address,gate

    def forward(self,initial,tokens,lengths,execution="soft",*,return_traces=False,output="probabilities"):
        if execution not in ("soft","straight_through","native"):raise ValueError("unknown execution mode")
        if output not in ("probabilities","logits"):raise ValueError("unknown output format")
        self._validate(initial,tokens,lengths)
        if execution=="native":
            with torch.no_grad():return self._execute(initial,tokens,lengths,execution,return_traces,output)
        return self._execute(initial,tokens,lengths,execution,return_traces,output)

    def forward_embedded(self,initial_state,embedded_inputs,lengths,execution="soft",*,
                         return_traces=False,output="probabilities"):
        """The SAME core on supplied raw token features[B,L,token_dim].

        No frontend, normalization, new parameter or extra semantic input is
        introduced. A caller-owned trainable audio/vision projection can feed
        this API, and soft/ST gradients propagate into that projection. Its
        parameters and compute must be accounted separately by the wrapper.
        Native retains the old no_grad/hard-CALL-only contract.
        """
        if execution not in ("soft","straight_through","native"):raise ValueError("unknown execution mode")
        if output not in ("probabilities","logits"):raise ValueError("unknown output format")
        if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.long for v in (initial_state,lengths)):
            raise TypeError("initial_state and lengths must be Long tensors")
        if not isinstance(embedded_inputs,torch.Tensor) or not embedded_inputs.is_floating_point():
            raise TypeError("embedded_inputs must be a floating tensor")
        if (initial_state.ndim!=1 or lengths.shape!=initial_state.shape or embedded_inputs.ndim!=3
                or embedded_inputs.shape[0]!=len(initial_state) or embedded_inputs.shape[2]!=self.token_dim):
            raise ValueError("initial[B],embedded_inputs[B,L,token_dim],lengths[B] required")
        if any(v.device!=self.encoder.weight.device for v in (initial_state,embedded_inputs,lengths)):
            raise ValueError("model/input device differs")
        if embedded_inputs.dtype!=self.encoder.weight.dtype:raise TypeError("model/embedded feature dtype differs")
        if not bool(torch.isfinite(embedded_inputs).all()):raise ValueError("embedded features must be finite")
        if bool(((initial_state<0)|(initial_state>=self.n_states)).any()):raise ValueError("initial outside public states")
        if bool(((lengths<0)|(lengths>embedded_inputs.shape[1])).any()):raise ValueError("length outside supplied extent")
        if execution=="native":
            with torch.no_grad():
                return self._execute(initial_state,None,lengths,execution,return_traces,output,embedded_inputs=embedded_inputs)
        return self._execute(initial_state,None,lengths,execution,return_traces,output,embedded_inputs=embedded_inputs)

    def _execute(self,initial,tokens,lengths,execution,trace,output,*,embedded_inputs=None):
        batch,horizon=tokens.shape if embedded_inputs is None else embedded_inputs.shape[:2]
        h=self.encoder(initial)
        keys=self.initial_memory_keys[None].expand(batch,-1,-1)
        values=h.new_zeros(batch,self.memory_slots,self.internal_states)
        states=[h] if trace else [];key_history=[keys] if trace else [];value_history=[values] if trace else []
        call_p=[];call_w=[];call_l=[];call_i=[];slot_history=[];read_history=[];writes=[];gates=[];masks=[]
        active_steps=0
        for step in range(horizon):
            active=step<lengths;ids=active.nonzero(as_tuple=False).flatten();active_steps+=ids.numel()
            if trace:
                ps=[];ws=[];ls=[];cs=[];rs=[];slot_states=[h]
                write_trace=h.new_zeros(batch,self.memory_slots);gate_trace=h.new_zeros(batch,1)
            if ids.numel():
                ah=h.index_select(0,ids);ak=keys.index_select(0,ids);av=values.index_select(0,ids)
                raw_x=(self.token_embedding(tokens[:,step].index_select(0,ids)) if embedded_inputs is None
                       else embedded_inputs[:,step].index_select(0,ids))
            for slot in range(self.n_slots):
                if trace:
                    p0=h.new_zeros(batch,self.n_primitives);w0=torch.zeros_like(p0);l0=torch.zeros_like(p0)
                    c0=torch.full((batch,),-1,device=h.device,dtype=torch.long);r0=h.new_zeros(batch,self.internal_states)
                if ids.numel():
                    x=raw_x+self.slot_embedding.weight[slot]
                    read,_=self._read(ah,x,ak,av)
                    ah,prob,weight,logit,chosen=self._call(ah,x,read,execution)
                    h=h.index_copy(0,ids,ah)
                    if trace:
                        p0=p0.index_copy(0,ids,prob);w0=w0.index_copy(0,ids,weight);l0=l0.index_copy(0,ids,logit)
                        c0=c0.index_copy(0,ids,chosen);r0=r0.index_copy(0,ids,read)
                if trace:ps.append(p0);ws.append(w0);ls.append(l0);cs.append(c0);rs.append(r0);slot_states.append(h)
            if ids.numel():
                # One write after all calls; last pre-call read and raw token
                # enter the learned write rule, with FINAL token-state h.
                next_keys,next_values,address,gate=self._write(ah,raw_x,read,ak,av)
                keys=keys.index_copy(0,ids,next_keys);values=values.index_copy(0,ids,next_values)
                if trace:write_trace=write_trace.index_copy(0,ids,address);gate_trace=gate_trace.index_copy(0,ids,gate)
            if trace:
                states.append(h);key_history.append(keys);value_history.append(values)
                call_p.append(torch.stack(ps,1));call_w.append(torch.stack(ws,1));call_l.append(torch.stack(ls,1))
                call_i.append(torch.stack(cs,1));slot_history.append(torch.stack(slot_states,1));read_history.append(torch.stack(rs,1))
                writes.append(write_trace);gates.append(gate_trace);masks.append(active)
        log_probabilities=F.log_softmax(self.readout(h),-1)
        answer=log_probabilities if output=="logits" else log_probabilities.exp()
        if not trace:return answer
        def stacked(items,shape,dtype=None):
            return torch.stack(items,1) if horizon else torch.empty(shape,device=h.device,dtype=h.dtype if dtype is None else dtype)
        kh,vh=torch.stack(key_history,1),torch.stack(value_history,1)
        memory=torch.cat((keys.flatten(1),values.flatten(1)),-1)
        return answer,dict(states=torch.stack(states,1),memory_keys=kh,memory_values=vh,
            memories=torch.cat((kh.flatten(2),vh.flatten(2)),-1),
            call_probabilities=stacked(call_p,(batch,0,self.n_slots,self.n_primitives)),
            call_weights=stacked(call_w,(batch,0,self.n_slots,self.n_primitives)),
            call_logits=stacked(call_l,(batch,0,self.n_slots,self.n_primitives)),
            call_indices=stacked(call_i,(batch,0,self.n_slots),torch.long),
            slot_states=stacked(slot_history,(batch,0,self.n_slots+1,self.internal_states)),
            reads=stacked(read_history,(batch,0,self.n_slots,self.internal_states)),
            write_addresses=stacked(writes,(batch,0,self.memory_slots)),write_gates=stacked(gates,(batch,0,1)),
            active_mask=stacked(masks,(batch,0),torch.bool),final_state=h,final_memory=memory,
            lengths=lengths,execution=execution,routing=self.routing,active_steps=active_steps,
            active_calls=active_steps*self.n_slots,
            memory_update_timing="one learned soft write at token end from final h,raw x,last read; key/value memory fixed across calls",
            native_contract="hard CALL only; continuous E/h/keys/values/primitives/R; no equivalence assumed",
            protection="no topological/retention certificate supplied by this development variant",
            resources=self.resource_report(batch_size=batch,sequence_length=horizon,active_steps=active_steps,
                execution=execution,return_traces=True))

    def parameter_count(self,trainable_only=False):
        return sum(p.numel() for p in self.parameters() if not trainable_only or p.requires_grad)

    def resource_report(self,batch_size=1,sequence_length=0,*,active_steps=None,execution="soft",return_traces=False):
        if execution not in ("soft","straight_through","native"):raise ValueError("unknown execution mode")
        if batch_size<0 or sequence_length<0:raise ValueError("nonnegative batch/horizon required")
        steps=batch_size*sequence_length if active_steps is None else active_steps
        if not 0<=steps<=batch_size*sequence_length:raise ValueError("active steps outside extent")
        h,t,q,m,k,r,c,s=self.internal_states,self.token_dim,self.key_dim,self.memory_slots,self.n_primitives,self.primitive_rank,self.caller_hidden,self.n_slots
        f=2*h+t;calls=steps*s;executed_k=1 if execution=="native" else k;scalar_bytes=self.encoder.weight.element_size()
        parts=dict(read_queries=calls*(h+t)*q,read_attention_scores=calls*m*q,read_attention_values=calls*m*h,
            contextual_caller=calls*(f*c+c*k),primitive_down_up=calls*executed_k*r*(f+h),
            primitive_mixing=0 if execution=="native" else calls*k*h,
            write_projections=steps*f*(2*q+h+1),write_attention_scores=steps*m*q,
            write_key_value_blending=steps*2*m*(q+h),readout=batch_size*h*self.n_states)
        carry=batch_size*(h+m*(q+h))
        # Both flattened `memories` and explicit key/value traces are returned,
        # and therefore their duplicate logical payloads are counted.
        floats=(batch_size*(sequence_length+1)*(h+2*m*(q+h))
            +batch_size*sequence_length*(3*s*k+(s+1)*h+s*h+m+1)) if return_traces else 0
        return dict(configuration=self.configuration(),execution=execution,total_parameters=self.parameter_count(),
            learnable_parameters=self.parameter_count(True),frozen_parameters=sum(p.numel() for p in self.parameters() if not p.requires_grad),
            parameter_payload_bytes=sum(p.numel()*p.element_size() for p in self.parameters()),
            parameter_counts={n:p.numel() for n,p in self.named_parameters()},registered_buffer_scalars=sum(b.numel() for b in self.buffers()),
            buffer_payload_bytes=sum(b.numel()*b.element_size() for b in self.buffers()),batch_size=batch_size,
            sequence_length=sequence_length,active_steps=steps,slots_per_token=s,active_calls=calls,
            persistent_example_state_scalars=carry,persistent_example_state_payload_bytes=carry*scalar_bytes,
            memory_key_carry_scalars=batch_size*m*q,memory_value_carry_scalars=batch_size*m*h,
            fresh_memory_initialization="learned initial keys,zero values each forward; no cross-example carry",
            memory_read_attention_elements=calls*m,memory_write_attention_elements=steps*m,
            primitive_candidate_workspace_scalars=batch_size*executed_k*(r+h),
            native_selected_weight_gather_scalars=batch_size*r*(f+h) if execution=="native" else 0,
            trace_logical_payload_bytes=floats*scalar_bytes+(8*s+1)*batch_size*sequence_length if return_traces else 0,
            proxy_forward_macs=sum(parts.values()),proxy_forward_macs_by_component=parts,
            proxy_scope="explicit forward linear/dot/weighted-sum products; blend term is a named scalar-product proxy; excludes GELU/softmax/LayerNorm/activations/indexing/backward/backend workspace",
            native_semantics="hard calls only; continuous state/memory/operators/readout",
            backend_autograd_allocator_memory="not inferred; actual peak RAM/CUDA required",
            architecture_scope="ordinary low-rank nonlinear residual/attention ingredients; development not Primary or novelty evidence")

    def checkpoint(self):
        return dict(version=self.checkpoint_version,configuration=self.configuration(),
            state_dict={name:value.detach().cpu().clone() for name,value in self.state_dict().items()},
            memory_lifecycle="per-example learned initial keys/zero values; transient memory not checkpointed",
            resources=self.resource_report())

    @classmethod
    def from_checkpoint(cls,payload:Mapping[str,Any]):
        if payload.get("version")!=cls.checkpoint_version:raise ValueError("unsupported continuous primitive checkpoint")
        state=payload["state_dict"];tensor=state["encoder.weight"]
        model=cls(**dict(payload["configuration"]),device=tensor.device,dtype=tensor.dtype)
        model.load_state_dict(state,strict=True);return model
