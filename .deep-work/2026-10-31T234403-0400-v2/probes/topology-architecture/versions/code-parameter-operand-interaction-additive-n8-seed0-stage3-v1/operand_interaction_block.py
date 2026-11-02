"""OWN576P operand-interaction development branch; ordinary bilinear NN precedent.

Copied exact fcba transition with ONE shared rank8 preactivation term. Old
controller/caller/gates/read/write/carry math is unchanged. Product versus
nonlinear-additive is the only scientific contrast, never an output replacement.
Unique checkpoint policy prevents the old loader from reinterpreting this NN.
"""
import copy,math,random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import continuous_primitive_block as base

BASE_SHA256='fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158'
NEW_NAMES=('interaction_A','interaction_C','interaction_B')
CHECKPOINT_VERSION='factorized-operand-interaction-block-v1'
POLICY='shared-operand8-before-oldexpert-gelu/v1'

def rng(device):
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()if device.type=='cuda'else[],numpy=np.random.get_state(),python=random.getstate())

def restore_rng(state,device):
    torch.set_rng_state(state['cpu'])
    if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])

class OperandInteractionBlock(base.ContinuousPrimitiveBlock):
    checkpoint_version=CHECKPOINT_VERSION
    def __init__(self,*args,interaction_mode,device=None,dtype=None,**kwargs):
        if interaction_mode not in ('product','additive'):raise ValueError('exact frozen interaction mode required')
        # Factories are CPU-only and globally RNG-neutral. A/C use ONE fixed
        # local generator, shared by both modes; B is exactly zero. .to draws no RNG.
        with torch.random.fork_rng(devices=[]):
            super().__init__(*args,device='cpu',dtype=dtype,**kwargs)
            if (self.internal_states,self.memory_dim,self.primitive_rank,self.n_primitives)!=(32,32,8,4):raise ValueError('fixed fcba dimensions only')
            generator=torch.Generator(device='cpu');generator.manual_seed(0)
            bound=1/math.sqrt(32)
            self.interaction_A=nn.Parameter(torch.empty(8,32,dtype=self.encoder.weight.dtype).uniform_(-bound,bound,generator=generator))
            self.interaction_C=nn.Parameter(torch.empty(8,32,dtype=self.encoder.weight.dtype).uniform_(-bound,bound,generator=generator))
            self.interaction_B=nn.Parameter(torch.zeros(8,8,dtype=self.encoder.weight.dtype))
        self.interaction_mode=interaction_mode
        self.organization_policy=POLICY+'/'+interaction_mode
        if device is not None:self.to(device=device)

    def configuration(self):
        return dict(super().configuration(),interaction_mode=self.interaction_mode)

    def resource_report(self,batch_size=1,sequence_length=0,*,active_steps=None,execution='soft',return_traces=False):
        report=super().resource_report(batch_size,sequence_length,active_steps=active_steps,execution=execution,return_traces=return_traces)
        n=batch_size*sequence_length if active_steps is None else active_steps
        executed_k=1 if execution=='native'else self.n_primitives
        parts=dict(report['proxy_forward_macs_by_component']);parts['shared_operand_linear']=n*576
        report.update(proxy_forward_macs_by_component=parts,proxy_forward_macs=sum(parts.values()),
            added_parameters=576,interaction_mode=self.interaction_mode,
            interaction_operations=dict(active_tokens=n,linear_MAC_per_token=576,
                product_multiplications=n*8 if self.interaction_mode=='product'else 0,
                additive_additions=n*8 if self.interaction_mode=='additive'else 0,
                additive_GELU_elements=n*8 if self.interaction_mode=='additive'else 0,
                expert_preactivation_additions=n*executed_k*8),
            full_forward_FLOPs=None,full_training_FLOPs=None,
            interaction_scope='forwardlinearMAC plus separately counted product/add/GELU/preactivationadds; backward/nonlinear/index/workspace not fullFLOPs')
        return report

    @classmethod
    def from_checkpoint(cls,payload):
        if payload.get('version')!=CHECKPOINT_VERSION:raise ValueError('distinct interaction checkpoint required')
        configuration=dict(payload['configuration']);mode=configuration.get('interaction_mode')
        if mode not in ('product','additive')or payload.get('organization_policy')!=POLICY+'/'+mode:raise ValueError('exact interaction policy/mode required')
        t=payload['state_dict']['encoder.weight'];model=cls(**configuration,dtype=t.dtype)
        model.load_state_dict(payload['state_dict'],strict=True);return model

    def _transition(self,carry,raw_x,execution):
        h,c,keys,values=(carry[k] for k in ("h","controller","keys","values"))
        c=self.controller(torch.cat((raw_x,h),-1),c)
        x=raw_x+self.slot_embedding.weight[0];control=torch.cat((c,x),-1)
        attention=F.softmax(torch.einsum("bq,bmq->bm",self.read_query(control),keys)/self.key_dim**.5,-1)
        read=torch.einsum("bm,bmh->bh",attention,values)
        operand_h=F.linear(h,self.interaction_A)
        operand_read=F.linear(read,self.interaction_C)
        operand=operand_h*operand_read if self.interaction_mode=="product" else F.gelu(operand_h+operand_read)
        shared=F.linear(operand,self.interaction_B)
        features=torch.cat((h,c,x,read),-1)
        call_features=features if self.routing=="dynamic" else torch.cat((torch.zeros_like(h),torch.zeros_like(c),x,torch.zeros_like(read)),-1)
        all_logits=self.caller(call_features);logits=all_logits[:,:self.n_primitives]
        probabilities=F.softmax(logits,-1);chosen=logits.argmax(-1)
        if execution=="native":
            low=F.gelu(torch.bmm(self.primitive_down[chosen],features[:,:,None]).squeeze(-1)+self.primitive_bias[chosen]+shared)
            delta=torch.bmm(self.primitive_up[chosen],low[:,:,None]).squeeze(-1)
            weights=F.one_hot(chosen,self.n_primitives).to(h)
        else:
            low=F.gelu(torch.einsum("bf,krf->bkr",features,self.primitive_down)+self.primitive_bias+shared[:,None,:])
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

def migrate_parent(payload,mode,device):
    """Restore all old tensors/Adam BY NAME; new576P starts at moments/step0.

    This operation preserves all available caller RNG streams. The ONLY later
    training RNG draw is unchanged grow_storage's one original token row.
    """
    available=rng(device)
    try:
        parent=base.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint'])
        parent.load_state_dict(payload['model_state'],strict=True)
        parent_opt=torch.optim.Adam(parent.parameters(),lr=.003);parent_opt.load_state_dict(payload['optimizer'])
        configuration=parent.configuration()
        if configuration['n_tokens']!=15 or parent.parameter_count()!=13381:raise ValueError('exact15token13381P stage2 parent required')
        model=OperandInteractionBlock(**configuration,interaction_mode=mode)
        incompatible=model.load_state_dict(payload['model_state'],strict=False)
        if set(incompatible.missing_keys)!=set(NEW_NAMES)or incompatible.unexpected_keys:raise ValueError('old NN name migration differs')
        model.to(device)
        named=dict(model.named_parameters());parent_named=dict(parent.named_parameters());by_id={id(v):k for k,v in parent_named.items()}
        groups=[]
        for group in parent_opt.param_groups:
            groups.append(dict((k,copy.deepcopy(v))for k,v in group.items()if k!='params'))
            groups[-1]['params']=[named[by_id[id(p)]]for p in group['params']]
        groups[0]['params'] += [named[k]for k in NEW_NAMES]
        opt=torch.optim.Adam(groups,lr=.003)
        for name,p in parent_named.items():
            state=copy.deepcopy(parent_opt.state.get(p,{}))
            for key,value in list(state.items()):
                if torch.is_tensor(value):state[key]=value.clone()if key=='step'else value.to(device=device)
            opt.state[named[name]]=state
        for name in NEW_NAMES:
            p=named[name];opt.state[p]=dict(step=torch.tensor(0.,dtype=torch.float32),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
        if model.parameter_count()!=13957:raise ValueError('exact576P addition required')
        return model,opt,parent,parent_opt
    finally:restore_rng(available,device)

def optimizer_names(model,opt):
    names={id(v):k for k,v in model.named_parameters()}
    return [[names[id(p)]for p in group['params']]for group in opt.param_groups]

def restore_optimizer(model,state,names):
    named=dict(model.named_parameters())
    if len(names)!=len(state['param_groups'])or sorted(x for group in names for x in group)!=sorted(named):raise ValueError('exact optimizer-name coverage required')
    groups=[]
    for saved,order in zip(state['param_groups'],names):
        groups.append(dict((k,copy.deepcopy(v))for k,v in saved.items()if k!='params'))
        groups[-1]['params']=[named[name]for name in order]
    opt=torch.optim.Adam(groups,lr=.003);opt.load_state_dict(state);return opt

def verify_old_migration(model,opt,parent,parent_opt,old):
    named=dict(model.named_parameters());previous=dict(parent.named_parameters())
    if len(previous)!=27 or set(named)!=set(previous)|set(NEW_NAMES):raise ValueError('old27 plus new3 tensors only')
    for name,p in previous.items():
        if not torch.equal(named[name].detach().cpu(),p.detach().cpu()):raise ValueError('old actual NN changed: '+name)
        if not old.nested_state_equal(opt.state[named[name]],parent_opt.state[p]):raise ValueError('old Adam changed: '+name)
    for name in NEW_NAMES:
        state=opt.state[named[name]]
        if set(state)!={'step','exp_avg','exp_avg_sq'}or any(bool(v.count_nonzero())for v in state.values()):raise ValueError('new moments/step not zero')
    if bool(model.interaction_B.count_nonzero()):raise ValueError('initial B must be zero')
    return dict(old27_NN_exact=True,old27_Adam_by_name_exact=True,new_parameters=576,new_parameter_states_zero=True,initial_B_exact_zero=True)
