"""OWN one-scalar continuous value bridge, not a proved retraction.

Exact fcba transition plus one tied encoder/readout bridge AFTER G8 and BEFORE
unchanged WRITE. Context/controller/keys/cells are never reset to terminaly.
No argmax, intermediate label, route teacher, cache or model-output replacement.
Canonical frozen import: this source is continuous_primitive_block.py and its
exact unchanged parent source is owned_fcba_core.py in BOTH CPU/GPU bundles.
"""
import copy,random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import owned_fcba_core as base

BASE_SHA256='fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158'
NEW_NAMES=('value_bridge_alpha',)
CHECKPOINT_VERSION='continuous-primitive-block-soft-value-bridge-v1'
POLICY='learned-control-history/value-identity-write/group8-carry+soft-value-bridge/v1'

def rng(device):
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()if device.type=='cuda'else[],numpy=np.random.get_state(),python=random.getstate())

def restore_rng(state,device):
    torch.set_rng_state(state['cpu'])
    if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])

class ContinuousPrimitiveBlock(base.ContinuousPrimitiveBlock):
    checkpoint_version=CHECKPOINT_VERSION
    organization_policy=POLICY
    def __init__(self,*args,device=None,dtype=None,**kwargs):
        with torch.random.fork_rng(devices=[]):
            super().__init__(*args,device='cpu',dtype=dtype,**kwargs)
            if (self.internal_states,self.n_primitives,self.primitive_rank,self.controller_dim,self.carry_gate_groups)!=(32,4,8,16,8):raise ValueError('fixed fcba dimensions only')
            self.value_bridge_alpha=nn.Parameter(torch.zeros((),dtype=self.encoder.weight.dtype))
        if device is not None:self.to(device=device)

    def soft_value_bridge(self,h):
        # This is a learned continuous residual bridge, not hard decoding.
        # Preserve h's magnitude including zero (no invented zero-memory value).
        basis=F.normalize(self.encoder.weight,dim=-1,eps=1e-6)*self.internal_states**.5
        posterior=F.softmax(self.readout(h),dim=-1)
        target=F.normalize(posterior@basis,dim=-1,eps=1e-6)*torch.linalg.vector_norm(h,dim=-1,keepdim=True)
        return h+torch.tanh(self.value_bridge_alpha)*(target-h)

    @classmethod
    def from_parent_checkpoint(cls,payload):
        if payload.get('version')!=base.ContinuousPrimitiveBlock.checkpoint_version or payload.get('organization_policy')!=base.ContinuousPrimitiveBlock.organization_policy:raise ValueError('exact oldfcba parent required')
        t=payload['state_dict']['encoder.weight'];model=cls(**payload['configuration'],dtype=t.dtype)
        missing=model.load_state_dict(payload['state_dict'],strict=False)
        if set(missing.missing_keys)!=set(NEW_NAMES)or missing.unexpected_keys or bool(model.value_bridge_alpha.count_nonzero()):raise ValueError('old27 plus exactzero scalar only')
        return model

    @classmethod
    def from_checkpoint(cls,payload):
        if payload.get('version')!=CHECKPOINT_VERSION or payload.get('organization_policy')!=POLICY:raise ValueError('exact new valuebridge policy required')
        t=payload['state_dict']['encoder.weight'];model=cls(**payload['configuration'],dtype=t.dtype)
        model.load_state_dict(payload['state_dict'],strict=True);return model

    def resource_report(self,batch_size=1,sequence_length=0,*,active_steps=None,execution='soft',return_traces=False):
        report=super().resource_report(batch_size,sequence_length,active_steps=active_steps,execution=execution,return_traces=return_traces)
        n=batch_size*sequence_length if active_steps is None else active_steps;h,s=self.internal_states,self.n_states
        parts=dict(report['proxy_forward_macs_by_component']);parts['value_bridge_linear']=n*2*h*s
        report.update(proxy_forward_macs_by_component=parts,proxy_forward_macs=sum(parts.values()),added_parameters=1,
            value_bridge=dict(strength_parameter='value_bridge_alpha',strength=float(torch.tanh(self.value_bridge_alpha.detach()).cpu()),linear_MAC_per_active_token=2*h*s,
                posterior_softmax_elements=n*s,h_norm_elements=n*h,projected_value_normalization_elements=n*h,residual_additions=n*2*h,residual_multiplications=n*h,
                encoder_basis_normalization_elements_per_transition_call=s*h,scalar_tanh_per_transition_call=1,
                basis_normalization_call_scope='one per actual _transition call, originalgather <= suppliedsequence_length; maskedGraph paysallfixedcalls evenpadding',
                logical_named_intermediate_payload_bytes_per_full_batch_call=self.encoder.weight.element_size()*(s*h+batch_size*(s+h+1+h)+1),
                intermediate_scope='basis/posterior/projectedvalue/norm/residual/scalar logical payload; tensor lifetimes/aliasing/autograd/allocator peak must be measured, not claimed totalworkspace',
                candidate_h_alias='existing postG8 h; bridge adds no candidateNN or cachedmodel; WRITE uses bridgedh',
                semantics='continuous output-induced residual with tanh strength possiblynegative; not a proved retraction/closure/protection'),
            full_forward_FLOPs=None,full_training_FLOPs=None)
        return report

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
        h=self.soft_value_bridge(h)
        write_features=torch.cat((c,raw_x),-1)
        address=F.softmax(torch.einsum("bq,bmq->bm",self.write_query(write_features),keys)/self.key_dim**.5,-1)
        write_gate=torch.sigmoid(self.write_gate(write_features));amount=(write_gate*address)[:,:,None]
        keys=keys+amount*(torch.tanh(self.write_key(write_features))[:,None]-keys)
        values=values+amount*(h[:,None]-values)
        return dict(h=h,controller=c,keys=keys,values=values),dict(call_probabilities=probabilities,call_weights=weights,
            call_logits=logits,call_indices=chosen,read=read,read_attention=attention,write_addresses=address,
            write_gate=write_gate,state_gate=gates)

def optimizer_names(model,opt):
    names={id(v):k for k,v in model.named_parameters()}
    return [[names[id(p)]for p in group['params']]for group in opt.param_groups]

def restore_optimizer(model,state,names):
    named=dict(model.named_parameters())
    if len(names)!=len(state['param_groups'])or sorted(x for group in names for x in group)!=sorted(named):raise ValueError('exact optimizer-name coverage required')
    groups=[]
    for saved,order in zip(state['param_groups'],names):
        groups.append(dict((k,copy.deepcopy(v))for k,v in saved.items()if k!='params'));groups[-1]['params']=[named[name]for name in order]
    opt=torch.optim.Adam(groups,lr=.003);opt.load_state_dict(state);return opt

def migrate_parent(payload,device):
    available=rng(device)
    try:
        parent=base.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint']);parent.load_state_dict(payload['model_state'],strict=True)
        parent_opt=torch.optim.Adam(parent.parameters(),lr=.003);parent_opt.load_state_dict(payload['optimizer'])
        if parent.n_tokens!=15 or parent.parameter_count()!=13381:raise ValueError('exact13381P15token parent required')
        model=ContinuousPrimitiveBlock.from_parent_checkpoint(payload['block_checkpoint']).to(device)
        named=dict(model.named_parameters());oldnamed=dict(parent.named_parameters());byid={id(v):k for k,v in oldnamed.items()};groups=[]
        for group in parent_opt.param_groups:
            groups.append(dict((k,copy.deepcopy(v))for k,v in group.items()if k!='params'));groups[-1]['params']=[named[byid[id(p)]]for p in group['params']]
        groups[0]['params'] += [named['value_bridge_alpha']];opt=torch.optim.Adam(groups,lr=.003)
        for name,p in oldnamed.items():
            state=copy.deepcopy(parent_opt.state[p])
            for key,value in list(state.items()):
                if torch.is_tensor(value):state[key]=value.clone()if key=='step'else value.to(device=device)
            opt.state[named[name]]=state
        p=named['value_bridge_alpha'];opt.state[p]=dict(step=torch.tensor(0.,dtype=torch.float32),exp_avg=torch.zeros_like(p),exp_avg_sq=torch.zeros_like(p))
        if model.parameter_count()!=13382:raise ValueError('exact one scalar addition required')
        return model,opt,parent,parent_opt
    finally:restore_rng(available,device)

def verify_old_migration(model,opt,parent,parent_opt,old):
    named=dict(model.named_parameters());previous=dict(parent.named_parameters())
    if len(previous)!=27 or set(named)!=set(previous)|set(NEW_NAMES):raise ValueError('old27 plus scalar only')
    for name,p in previous.items():
        if not torch.equal(named[name].detach().cpu(),p.detach().cpu())or not old.nested_state_equal(opt.state[named[name]],parent_opt.state[p]):raise ValueError('old NN/Adam changed: '+name)
    if bool(model.value_bridge_alpha.count_nonzero())or any(bool(v.count_nonzero())for v in opt.state[model.value_bridge_alpha].values()):raise ValueError('alpha/moments/step must start0')
    return dict(old27_NN_exact=True,old27_Adam_by_name_exact=True,new_parameters=1,new_scalar_exact_zero=True,new_parameter_states_zero=True)
