"""Independent selective-SSM reference adaptation; NOT official Mamba/Mamba-2.

Source mechanisms: Gu & Dao, arXiv:2312.00752v2, Algorithm2, Sec3.4;
official state-spaces/mamba mamba_simple.py and selective_scan_ref.
This newly written plain-PyTorch implementation keeps input-dependent delta,
B and C, causal depthwise convolution, gated output, residuals and RMSNorm.
Its sequential autograd scan has NO fused kernels/parallel scan/recomputation;
its measured speed/memory must NOT be attributed to optimized Mamba.

The injection is delta*B*u as in the official real-valued reference scan,
not the full exact-ZOH B formula in paper Eq4. No imports from external code,
network, subprocess, hidden tasks, supplied operators or teacher routes.
No import-time execution. Root owns finite training and isolation.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self,width,eps=1e-5):
        super().__init__();self.weight=nn.Parameter(torch.ones(width));self.eps=eps

    def forward(self,x):
        return x*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+self.eps).to(x.dtype)*self.weight


class SelectiveSSMBlock(nn.Module):
    """Residual causal block; expanded hidden[B,E*d_model,state_size]."""
    def __init__(self,width=64,state_size=16,expand=2,conv_width=4,dt_rank=None):
        super().__init__()
        if min(width,state_size,expand,conv_width)<1:raise ValueError("positive dimensions required")
        self.width,self.state_size,self.inner,self.conv_width=width,state_size,width*expand,conv_width
        self.dt_rank=math.ceil(width/16) if dt_rank is None else int(dt_rank)
        self.norm=RMSNorm(width)
        self.input_projection=nn.Linear(width,2*self.inner,bias=False)
        self.causal_filter=nn.Conv1d(self.inner,self.inner,conv_width,groups=self.inner,
                                   padding=conv_width-1,bias=True)
        self.selection=nn.Linear(self.inner,self.dt_rank+2*state_size,bias=False)
        self.delta_projection=nn.Linear(self.dt_rank,self.inner,bias=True)
        self.log_rates=nn.Parameter(torch.arange(1,state_size+1,dtype=torch.float32).log()[None].repeat(self.inner,1))
        self.skip=nn.Parameter(torch.ones(self.inner))
        self.output_projection=nn.Linear(self.inner,width,bias=False)
        nn.init.uniform_(self.delta_projection.weight,-self.dt_rank**-.5,self.dt_rank**-.5)
        with torch.no_grad():
            dt=torch.exp(torch.empty(self.inner).uniform_(math.log(.001),math.log(.1)))
            self.delta_projection.bias.copy_(dt+torch.log(-torch.expm1(-dt)))

    def forward(self,embedded,lengths):
        b,l,_=embedded.shape
        active=torch.arange(l,device=embedded.device)[None]<lengths[:,None]
        x,z=self.input_projection(self.norm(embedded)).chunk(2,-1)
        x=F.silu(self.causal_filter(x.transpose(1,2))[:,:,:l].transpose(1,2))
        dt_raw,write,read=self.selection(x).split((self.dt_rank,self.state_size,self.state_size),-1)
        dt=F.softplus(self.delta_projection(dt_raw))
        rates=-self.log_rates.float().exp()
        hidden=torch.zeros(b,self.inner,self.state_size,device=x.device,dtype=torch.float32)
        outputs=[]
        # This explicit loop exposes actual sequential work and autograd memory.
        # No production/fused-Mamba efficiency claim is supported by this code.
        for t in range(l):
            decay=torch.exp(dt[:,t,:,None].float()*rates[None])
            injection=dt[:,t,:,None].float()*write[:,t,None,:].float()*x[:,t,:,None].float()
            candidate=decay*hidden+injection
            hidden=torch.where(active[:,t,None,None],candidate,hidden)
            observed=(hidden*read[:,t,None,:].float()).sum(-1)+self.skip.float()*x[:,t].float()
            observed=observed.to(x.dtype)*F.silu(z[:,t])
            outputs.append(observed)
        y=self.output_projection(torch.stack(outputs,1))
        return torch.where(active[:,:,None],embedded+y,torch.zeros_like(embedded))


class SelectiveSSMBackbone(nn.Module):
    """Reusable causal embedded-sequence Block, independent of terminal tasks."""
    def __init__(self,width=64,layers=3,state_size=16,expand=2,conv_width=4):
        super().__init__()
        self.spec=dict(width=width,layers=layers,state_size=state_size,expand=expand,conv_width=conv_width)
        self.layers=nn.ModuleList([SelectiveSSMBlock(width,state_size,expand,conv_width) for _ in range(layers)])
        self.norm=RMSNorm(width)

    def forward(self,embedded,lengths):
        if embedded.ndim!=3 or lengths.shape!=(len(embedded),):raise ValueError("embedded[B,L,D],lengths[B] required")
        if embedded.shape[1]<1 or embedded.shape[2]!=self.spec["width"]:raise ValueError("nonempty sequence and matching width required")
        active=torch.arange(embedded.shape[1],device=embedded.device)[None]<lengths[:,None]
        x=torch.where(active[:,:,None],embedded,torch.zeros_like(embedded))
        for layer in self.layers:x=layer(x,lengths)
        return torch.where(active[:,:,None],self.norm(x),torch.zeros_like(x))

    def resource_report(self,batch_size,sequence_length):
        d=self.spec["width"];e=d*self.spec["expand"];n=self.spec["state_size"];r=math.ceil(d/16)
        work=batch_size*sequence_length
        products=dict(input_output_projections=work*3*d*e,
            causal_depthwise_filter=work*e*self.spec["conv_width"],
            selection_and_delta_projections=work*(e*(r+2*n)+r*e),
            recurrent_decay_write_read=work*3*e*n)
        return dict(implementation="plain-PyTorch selective-SSM reference adaptation; not official Mamba/Mamba2",
            parameter_count=sum(p.numel() for p in self.parameters()),
            parameter_bytes=sum(p.numel()*p.element_size() for p in self.parameters()),
            recurrent_carry_scalars=batch_size*self.spec["layers"]*e*n,
            causal_filter_carry_scalars=batch_size*self.spec["layers"]*e*self.spec["conv_width"],
            forward_MAC_proxy=self.spec["layers"]*sum(products.values()),
            per_layer_forward_MAC_proxy=products,
            MAC_boundary="padded reference projections/scan counted; scalar exp/gates/norm/masking/backward excluded; recurrent products are a named approximation",
            autograd_memory_boundary="ordinary sequential autograd retains intermediates; measure actual CUDA peak, not fused Mamba estimate",
            final_carry_reset="zero each forward; no cross-example state or replay")


class TerminalSelectiveSSM(nn.Module):
    """Raw initial/tokens/lengths interface with learned prefix and terminal head.

    initial is a single public learned embedding prefix. The last valid causal
    output predicts the terminal label. There is no taskID, oracle memory mode,
    intermediate label, hand-initialized operator or supplied correct routing.
    """
    def __init__(self,n_states,n_tokens,max_length,width=64,layers=3,state_size=16,expand=2,conv_width=4):
        super().__init__()
        self.spec=dict(n_states=n_states,n_tokens=n_tokens,max_length=max_length,width=width,layers=layers,
                       state_size=state_size,expand=expand,conv_width=conv_width)
        self.initial=nn.Embedding(n_states,width);self.tokens=nn.Embedding(n_tokens,width)
        self.backbone=SelectiveSSMBackbone(width,layers,state_size,expand,conv_width)
        self.readout=nn.Linear(width,n_states)

    def forward(self,initial,tokens,lengths):
        b,l=tokens.shape
        if l>self.spec["max_length"] or lengths.shape!=(b,):raise ValueError("declared sequence bounds differ")
        if bool(((lengths<0)|(lengths>l)).any()):raise ValueError("length outside supplied token extent")
        active=torch.arange(l,device=tokens.device)[None]<lengths[:,None]
        safe=torch.where(active,tokens,torch.zeros_like(tokens))
        embedded=torch.cat((self.initial(initial)[:,None],self.tokens(safe)),1)
        sequence=self.backbone(embedded,lengths+1)
        terminal=sequence[torch.arange(b,device=tokens.device),lengths]
        return self.readout(terminal)

    def proxy(self,b,l,active):
        r=self.backbone.resource_report(b,l+1)
        return dict(proxy_forward_macs=r["forward_MAC_proxy"]+b*self.spec["width"]*self.spec["n_states"],
            active_steps=active,reference_padded_steps=b*(l+1),details=r,
            architecture="source-grounded selective-SSM terminal adaptation; not official Mamba or Mamba2")

    def checkpoint(self):
        return dict(version="terminal-selective-ssm-reference-v1",configuration=dict(self.spec),
            state_dict={k:v.detach().cpu().clone() for k,v in self.state_dict().items()},
            implementation="plainTorch reference adaptation, not official optimized Mamba")

    @classmethod
    def from_checkpoint(cls,payload:Mapping[str,Any]):
        if payload.get("version")!="terminal-selective-ssm-reference-v1":raise ValueError("unsupported schema")
        state=payload["state_dict"];example=next(iter(state.values()))
        model=cls(**payload["configuration"]).to(device=example.device,dtype=example.dtype)
        model.load_state_dict(state,strict=True)
        return model
