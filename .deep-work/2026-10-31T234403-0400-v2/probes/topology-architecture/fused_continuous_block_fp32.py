"""OWN prospective FP32 soft-forward fusion of frozen core 58feb69e.

INFERENCE ONLY, engineering cost hypothesis, no novelty or performance claim.
One CTA/example walks L64/128/256; fixed num_warps=4, no autotuning/dot/TF32.
All 41,593 original model parameters remain allocated and are copied into a
charged extra packed buffer; only encoder row0 is read under the fixed AR
initial0 contract. No learned parameter or task feature is introduced.

Exact real-valued order: slotted read -> dynamic soft CALL/GELU -> residual LN
-> convex memory write using NEW h, RAW token x and the LAST read -> logsoftmax.
The slot embedding is used by read/CALL and is deliberately absent from write.
Floating reduction/library differences are UNVERIFIED. The prospective output
contract is max abs logp<=2e-4, torch.allclose(atol=2e-4,rtol=2e-5), zero argmax
mismatches against original pure tensor function and its CUDA Graph output.
This source is not compiled/imported/executed as part of authoring.

Primary API checks: Triton v3.4.0 language/core.py (gather/load), language/
semantic.py (gather dimensions), nvidia libdevice.py (erf/tanh/exp/log), NVIDIA
compiler.py (num_warps and enable_fp_fusion). See accompanying audit design.
"""
import math
import time

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

CORE_SHA256 = "58feb69e5d628745f64b6a4d546858aaba528169ae0d3cdae0530de61055d497"
DRIVER_SHA256 = "567e372187e4cee3c8a5ca3d49f5f7d60a8664a07619c3d57dc1c799cbbd9f02"
TOTAL_MODEL_PARAMETERS = 41593
ATOL, RTOL = 2e-4, 2e-5
CONTEXTS, BATCHES = (64, 128, 256), (1, 32)
# Explicit original OI row-major packing; no parameter is pruned/tied/omitted.
PARAMETER_LAYOUT = (
    ("encoder.weight", (100,64)),
    ("token_embedding.weight", (100,16)),
    ("slot_embedding.weight", (1,16)),
    ("initial_memory_keys", (8,16)),
    ("read_query.weight", (16,80)), ("read_query.bias", (16,)),
    ("caller.0.weight", (32,144)), ("caller.0.bias", (32,)),
    ("caller.2.weight", (4,32)), ("caller.2.bias", (4,)),
    ("primitive_down", (4,8,144)), ("primitive_bias", (4,8)),
    ("primitive_up", (4,64,8)),
    ("residual_norm.weight", (64,)), ("residual_norm.bias", (64,)),
    ("write_query.weight", (16,144)), ("write_query.bias", (16,)),
    ("write_key.weight", (16,144)), ("write_key.bias", (16,)),
    ("write_value.weight", (64,144)), ("write_value.bias", (64,)),
    ("write_gate.weight", (1,144)), ("write_gate.bias", (1,)),
    ("readout.weight", (100,64)), ("readout.bias", (100,)),
)
OFFSETS, offset = {}, 0
for _name, _shape in PARAMETER_LAYOUT:
    OFFSETS[_name] = offset
    offset += math.prod(_shape)
PACKED_NUMEL = offset
if PACKED_NUMEL != TOTAL_MODEL_PARAMETERS:
    raise RuntimeError("literal packing layout must account for all original P")
OFFSET_TUPLE = tuple(OFFSETS[name] for name, _ in PARAMETER_LAYOUT)
del _name, _shape, offset


@triton.jit
def _features144(h, x, read):
    f = tl.arange(0,256)
    hv = tl.gather(h, f & 63, axis=0)
    xv = tl.gather(x, (f-64) & 15, axis=0)
    rv = tl.gather(read, (f-80) & 63, axis=0)
    joined = tl.where(f<64,hv,tl.where(f<80,xv,rv))
    return tl.where(f<144,joined,0.0)


@triton.jit
def _features80(h, x):
    f = tl.arange(0,128)
    joined = tl.where(f<64,tl.gather(h,f & 63,axis=0),
                      tl.gather(x,(f-64) & 15,axis=0))
    return tl.where(f<80,joined,0.0)


@triton.jit
def _dense(P, features, W:tl.constexpr, B:tl.constexpr,
           INPUT:tl.constexpr, OUTPUT:tl.constexpr,
           PIN:tl.constexpr, POUT:tl.constexpr):
    rows=tl.arange(0,POUT);cols=tl.arange(0,PIN)
    # Volatile fixes a streamed-weight implementation: avoid silently retaining
    # the entire 41k-weight model in registers across the recurrent loop. This
    # bandwidth/register tradeoff is part of the measured candidate, not tuned.
    weights=tl.load(P+W+rows[:,None]*INPUT+cols[None,:],
                    mask=(rows[:,None]<OUTPUT)&(cols[None,:]<INPUT),
                    other=0.0,volatile=True)
    bias=tl.load(P+B+rows,mask=rows<OUTPUT,other=0.0,volatile=True)
    return tl.sum(weights*features[None,:],axis=1)+bias


@triton.jit
def _softmax(values):
    maximum=tl.max(values,axis=0)
    numer=libdevice.exp(values-maximum)
    return numer/tl.sum(numer,axis=0)


@triton.jit
def _fused_soft_fp32(Tokens,P,Output,HFinal,KeysFinal,ValuesFinal,
                      L:tl.constexpr,O:tl.constexpr):
    example=tl.program_id(0)
    ih=tl.arange(0,64);ix=tl.arange(0,16);im=tl.arange(0,8)
    iq=tl.arange(0,16);ik=tl.arange(0,4);ir=tl.arange(0,8)
    iv=tl.arange(0,128)
    # Learned initialization is loaded anew for every sequence/kernel call.
    h=tl.load(P+O[0]+ih)
    slot=tl.load(P+O[2]+ix)
    keys=tl.load(P+O[3]+im[:,None]*16+iq[None,:])
    values=tl.full((8,64),0.0,tl.float32)
    gamma=tl.load(P+O[13]+ih);beta=tl.load(P+O[14]+ih)
    for step in range(0,L):
        token=tl.load(Tokens+example*L+step).to(tl.int32)
        raw_x=tl.load(P+O[1]+token*16+ix)
        slotted_x=raw_x+slot
        # READ uses old h, old keys/values and the slotted token.
        read_features=_features80(h,slotted_x)
        query=_dense(P,read_features,O[4],O[5],80,16,128,16)
        read_scores=tl.sum(keys*query[None,:],axis=1)*0.25
        attention=_softmax(read_scores)
        read=tl.sum(attention[:,None]*values,axis=0)
        # CALL features are old h, slotted token, and exactly that last read.
        features=_features144(h,slotted_x,read)
        hidden=libdevice.tanh(_dense(P,features,O[6],O[7],144,32,256,32))
        call_logits=_dense(P,hidden,O[8],O[9],32,4,32,4)
        probabilities=_softmax(call_logits)
        bottleneck=_dense(P,features,O[10],O[11],144,32,256,32)
        low=0.5*bottleneck*(1.0+libdevice.erf(bottleneck*0.7071067811865476))
        low=tl.reshape(low,(4,8))
        up=tl.load(P+O[12]+ik[:,None,None]*64*8+ih[None,:,None]*8+ir[None,None,:],volatile=True)
        candidates=tl.sum(up*low[:,None,:],axis=2)
        delta=tl.sum(probabilities[:,None]*candidates,axis=0)
        residual=h+delta
        mean=tl.sum(residual,axis=0)*(1.0/64.0)
        centered=residual-mean
        variance=tl.sum(centered*centered,axis=0)*(1.0/64.0)
        h=(centered*tl.rsqrt(variance+1.0e-5))*gamma+beta
        # WRITE uses NEW normalized h, RAW token (no slot embedding), last read.
        write_features=_features144(h,raw_x,read)
        write_query=_dense(P,write_features,O[15],O[16],144,16,256,16)
        write_scores=tl.sum(keys*write_query[None,:],axis=1)*0.25
        address=_softmax(write_scores)
        gate_vector=_dense(P,write_features,O[21],O[22],144,1,256,1)
        gate=1.0/(1.0+libdevice.exp(-tl.sum(gate_vector,axis=0)))
        amount=gate*address
        new_key=libdevice.tanh(_dense(P,write_features,O[17],O[18],144,16,256,16))
        new_value=_dense(P,write_features,O[19],O[20],144,64,256,64)
        # Exact current convex write, not an update from the initial memory.
        keys=keys+amount[:,None]*(new_key[None,:]-keys)
        values=values+amount[:,None]*(new_value[None,:]-values)
        logits=_dense(P,h,O[23],O[24],64,100,64,128)
        logits=tl.where(iv<100,logits,-float('inf'))
        maximum=tl.max(logits,axis=0)
        log_denominator=libdevice.log(tl.sum(libdevice.exp(logits-maximum),axis=0))
        logp=(logits-maximum)-log_denominator
        tl.store(Output+(example*L+step)*100+iv,logp,mask=iv<100)
    # Final carry is always materialized and charged, including timed calls.
    tl.store(HFinal+example*64+ih,h)
    tl.store(KeysFinal+example*8*16+im[:,None]*16+iq[None,:],keys)
    tl.store(ValuesFinal+example*8*64+im[:,None]*64+ih[None,:],values)


def pack_frozen_model(model):
    """Charge this FP32 layout copy separately; keep original 41,593-P model."""
    if model.training or getattr(model,'arm',None)!='dynamic':
        raise ValueError('only frozen/eval dynamic all-token model supported')
    core=model.body
    required=dict(n_states=100,n_tokens=100,internal_states=64,n_primitives=4,
                  token_dim=16,state_dim=64,memory_dim=64,caller_hidden=32,
                  n_slots=1,memory_slots=8,key_dim=16,primitive_rank=8)
    if any(getattr(core,name,None)!=value for name,value in required.items())or core.routing!='dynamic':
        raise ValueError('fixed 58fe dynamic dimensions/routing required')
    if core.residual_norm.eps!=1e-5:
        raise ValueError('original LayerNorm epsilon changed')
    parameters=dict(core.named_parameters())
    if set(parameters)!={name for name,_ in PARAMETER_LAYOUT}:
        raise ValueError('no original parameter may be omitted or added')
    total=sum(v.numel()for v in model.parameters())
    if total!=TOTAL_MODEL_PARAMETERS:
        raise ValueError('retain exact original total model P')
    for name,shape in PARAMETER_LAYOUT:
        value=parameters[name]
        if tuple(value.shape)!=shape or value.dtype!=torch.float32 or value.device.type!='cuda':
            raise ValueError('original FP32 CUDA parameter shape differs: '+name)
    torch.cuda.synchronize();started=time.perf_counter()
    with torch.no_grad():
        packed=torch.cat([parameters[name].detach().contiguous().reshape(-1)
                          for name,_ in PARAMETER_LAYOUT])
    torch.cuda.synchronize()
    report=dict(original_model_parameters=total,original_model_parameter_bytes=total*4,
        extra_packed_weight_elements=packed.numel(),extra_packed_weight_bytes=packed.numel()*4,
        pack_wall_seconds=time.perf_counter()-started,layout=PARAMETER_LAYOUT,
        source_parameter_strides={name:list(parameters[name].stride())for name,_ in PARAMETER_LAYOUT},
        no_parameter_elimination=True,no_retraining=True,
        cost_scope='extra live packed buffer; add to original weights, not a replacement P count')
    return packed,report



@torch.no_grad()
def formula_reference_from_packed(tokens,packed):
    """Audit-only direct PyTorch reconstruction, CPU or GPU; NEVER a speed arm.

    ROOT may compare short bounded CPU cases to original pure tensor execution
    before authorizing CUDA. This function is not invoked by the fused backend
    or the trusted priming script, and it is not a substitute output model.
    """
    if packed.shape!=(PACKED_NUMEL,)or packed.dtype!=torch.float32:
        raise ValueError('fixed full FP32 parameter packing required')
    if tokens.dtype!=torch.long or tokens.device!=packed.device or tokens.ndim!=2:
        raise ValueError('tokens/packing device and original Long type required')
    if not 1<=tokens.shape[1]<=256:
        raise ValueError('bounded formula-reference extent required')
    F=torch.nn.functional
    weights={name:packed[OFFSETS[name]:OFFSETS[name]+math.prod(shape)].view(shape)
             for name,shape in PARAMETER_LAYOUT}
    def linear(value,name):
        return F.linear(value,weights[name+'.weight'],weights[name+'.bias'])
    batch=len(tokens)
    h=F.embedding(torch.zeros(batch,dtype=torch.long,device=tokens.device),weights['encoder.weight'])
    keys=weights['initial_memory_keys'][None].expand(batch,-1,-1)
    values=h.new_zeros(batch,8,64)
    outputs=[]
    for step in range(tokens.shape[1]):
        raw_x=F.embedding(tokens[:,step],weights['token_embedding.weight'])
        x=raw_x+weights['slot_embedding.weight'][0]
        query=linear(torch.cat((h,x),-1),'read_query')
        attention=F.softmax(torch.einsum('bq,bmq->bm',query,keys)/4.,-1)
        read=torch.einsum('bm,bmh->bh',attention,values)
        features=torch.cat((h,x,read),-1)
        hidden=torch.tanh(linear(features,'caller.0'))
        probability=F.softmax(linear(hidden,'caller.2'),-1)
        low=F.gelu(torch.einsum('bf,krf->bkr',features,weights['primitive_down'])+weights['primitive_bias'])
        candidates=torch.einsum('bkr,khr->bkh',low,weights['primitive_up'])
        delta=torch.einsum('bk,bkh->bh',probability,candidates)
        h=F.layer_norm(h+delta,(64,),weights['residual_norm.weight'],weights['residual_norm.bias'],1e-5)
        # New h + raw token + LAST read; do not silently insert slot into write.
        features=torch.cat((h,raw_x,read),-1)
        query=linear(features,'write_query')
        address=F.softmax(torch.einsum('bq,bmq->bm',query,keys)/4.,-1)
        gate=torch.sigmoid(linear(features,'write_gate'))
        amount=(gate*address)[:,:,None]
        new_key=torch.tanh(linear(features,'write_key'))[:,None,:]
        new_value=linear(features,'write_value')[:,None,:]
        keys=keys+amount*(new_key-keys)
        values=values+amount*(new_value-values)
        outputs.append(F.log_softmax(linear(h,'readout'),-1))
    return torch.stack(outputs,1),dict(h=h,keys=keys,values=values)


def validate_all_active_initial0(initial,tokens,lengths):
    """Run and charge OUTSIDE graph/capture: same fixed full-active AR scope."""
    if tokens.ndim!=2 or tokens.shape[0]not in BATCHES or tokens.shape[1]not in CONTEXTS:
        raise ValueError('fixed B1/32 L64/128/256 only')
    if any(v.dtype!=torch.long for v in(initial,tokens,lengths)):
        raise TypeError('original Long input tensors required')
    if initial.shape!=(len(tokens),)or lengths.shape!=initial.shape:
        raise ValueError('initial/length shapes differ')
    if not bool((initial==0).all())or not bool((lengths==tokens.shape[1]).all()):
        raise ValueError('only all-active AR initial0 scope is admitted')
    if not bool(((tokens>=0)&(tokens<100)).all()):
        raise ValueError('fixed vocabulary100 input required')
    if not tokens.is_contiguous()or tokens.device.type!='cuda':
        raise ValueError('contiguous CUDA tokens required; charge any explicit copy')


def allocate_buffers(tokens):
    batch,length=tokens.shape
    if batch not in BATCHES or length not in CONTEXTS:
        raise ValueError('fixed shape allocation only')
    kwargs=dict(device=tokens.device,dtype=torch.float32)
    return dict(output=torch.empty((batch,length,100),**kwargs),
                h=torch.empty((batch,64),**kwargs),keys=torch.empty((batch,8,16),**kwargs),
                values=torch.empty((batch,8,64),**kwargs))


def launch_packed(tokens,packed,buffers):
    """Allocation-free launch; numerical input validation is separately charged.

    The SAME callable is used for fused eager, CUDA Graph capture and replay.
    No fallback implementation or cache-miss compiler permission is supplied.
    """
    if torch.is_grad_enabled():raise RuntimeError('inference-only: torch.no_grad required')
    if tokens.dtype!=torch.long or not tokens.is_contiguous()or tokens.device.type!='cuda':
        raise ValueError('fixed contiguous Long CUDA tokens required')
    batch,length=tokens.shape
    if batch not in BATCHES or length not in CONTEXTS:
        raise ValueError('fixed B1/32 L64/128/256 only')
    if packed.shape!=(PACKED_NUMEL,)or packed.dtype!=torch.float32 or packed.device!=tokens.device or packed.requires_grad:
        raise ValueError('exact detached FP32 packed weights required')
    shapes=dict(output=(batch,length,100),h=(batch,64),keys=(batch,8,16),values=(batch,8,64))
    if set(buffers)!=set(shapes):raise ValueError('all charged output/carry buffers required')
    for key,shape in shapes.items():
        value=buffers[key]
        if tuple(value.shape)!=shape or value.dtype!=torch.float32 or value.device!=tokens.device or not value.is_contiguous():
            raise ValueError('fixed output/carry buffer differs: '+key)
    return _fused_soft_fp32[(batch,)](tokens,packed,buffers['output'],buffers['h'],
        buffers['keys'],buffers['values'],L=length,O=OFFSET_TUPLE,
        num_warps=4,num_stages=1,enable_fp_fusion=False)
