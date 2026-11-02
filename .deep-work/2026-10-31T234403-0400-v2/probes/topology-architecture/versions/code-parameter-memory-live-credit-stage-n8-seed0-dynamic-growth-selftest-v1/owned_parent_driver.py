"""Finite contextual-primitive development with actual terminal-only training.

Same Prometheus V2. This is600-update development, NOT frozen Primary evidence.
No import-time run, network, SSH, subprocess or dependency installation. Root
supplies the existing external CUDA isolation and process watchdog.

Dynamic is an ordinary contextual-calling mechanism; this comparison does not
establish independent novelty or Monodromy benefit. Static controls use the
same Block/slot/K/dimensions atD=N or preallocated2N in categorical mode;
continuous controls use the declared width or preallocated2x width. GRU and Transformer receive
the identical raw inputs, with NO extra stage/taskID/teacher operation/route.
All data accesses and all failed outcomes remain recorded. Validity is based
on protocol and implementation, never on an accuracy acceptance gate.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

ARMS=("dynamic","static","static_expand","gru","transformer","ssm")
SPLITS=("train","valid","test_iid","test_long","test_unseen")
KEYS=("initial","tokens","lengths","targets")
MAX_FILE=100*1024*1024

# These owned phase0 drivers were statically checked to share the execution,
# objective, batching and RNG functions below (and baseline init/forward).
# Their full hashes remain lineage; the new resume wrapper necessarily differs.
LEGACY_RESUME_DRIVERS={
    "0800a67418f9d5f0202108b52ede18cd5c76b744e0dacdde83d2a896ed8b2c5b",
    "772e2a0ed2e4e02c9775682ffb064c530e268bd9c3197f7ffe9b74bddb958cd6",
    "b50edbfa4033d18a1e785218dcb1c8b30b8bdc64f59c4a45574063c53c5360b3",
}
LEGACY_EXECUTION_SIGNATURES={
    # Measured from BOTH the exact frozen phase0 source and this driver under
    # the actual RunPod Python3.12 AST. ast.dump is not cross-version stable.
    # This changes only the compatibility guard, never execution or training.
    # Raw paired hashes: context-resume-python-ast-diagnosis-v1.json.
    "block":"b9ac3660a437424375257072e8ca80f2a3f31b35116466e046ef1e740f26acfd",
    "gru":"08e6135e617440fd2b64212b7e6bf6fee68c5ce4c43a84c8d72bb5190ed2e6dd",
    "transformer":"4afecb6d6971fb7cce0a655f4544313038b7ceb1cc44512525bc759f200c5429",
}


def plain(value):
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)):return [plain(v) for v in value]
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if torch.is_tensor(value):return value.detach().cpu().tolist()
    if isinstance(value,Path):return str(value)
    if isinstance(value,float) and not math.isfinite(value):raise ValueError("nonfinite evidence value")
    return value


def sha256(path):
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def synchronize(device):
    if device.type=="cuda":torch.cuda.synchronize(device)


def tensor_bytes(value):
    if torch.is_tensor(value):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(tensor_bytes(v) for v in value)
    return 0


class Recorder:
    """Append-only bounded evidence; no model or legacy architecture imports."""
    def __init__(self,out,seconds):
        out.mkdir(parents=True,exist_ok=False)
        self.out,self.started=out,time.perf_counter();self.deadline=self.started+seconds
        self.stream=(out/"records.jsonl").open("x",encoding="utf8");self.bytes=0

    def check(self):
        if time.perf_counter()>=self.deadline:raise TimeoutError("finite contextual training wall budget reached")

    def emit(self,kind,**fields):
        line=json.dumps(plain(dict(kind=kind,wall=time.perf_counter()-self.started,**fields)),sort_keys=True,allow_nan=False)+"\n"
        self.bytes+=len(line.encode())
        if self.bytes>MAX_FILE:raise RuntimeError("100MiB event-file cap")
        self.stream.write(line);self.stream.flush()

    def json(self,name,payload):
        text=json.dumps(plain(payload),sort_keys=True,indent=2,allow_nan=False)+"\n"
        if len(text.encode())>MAX_FILE:raise RuntimeError("100MiB JSON-file cap")
        with (self.out/name).open("x",encoding="utf8") as stream:stream.write(text)


class TerminalGRU(nn.Module):
    """Ordinary two-layer GRU, default width64, learned input/output interface."""
    def __init__(self,n_states,n_tokens,max_length,width=64,layers=2):
        super().__init__();self.spec=dict(n_states=n_states,n_tokens=n_tokens,max_length=max_length,width=width,layers=layers)
        self.tokens=nn.Embedding(n_tokens,width)
        self.initial=nn.Embedding(n_states,layers*width)
        self.rnn=nn.GRU(width,width,num_layers=layers,batch_first=True,dropout=0.)
        self.norm=nn.LayerNorm(width);self.readout=nn.Linear(width,n_states)

    def forward(self,initial,tokens,lengths):
        b=len(initial);w=self.spec["width"];layers=self.spec["layers"]
        h0=self.initial(initial).reshape(b,layers,w).transpose(0,1).contiguous()
        final=h0[-1].clone();ids=(lengths>0).nonzero(as_tuple=False).flatten()
        if ids.numel():
            words=tokens.index_select(0,ids);lens=lengths.index_select(0,ids)
            active=torch.arange(words.shape[1],device=words.device)[None,:]<lens[:,None]
            words=torch.where(active,words,torch.zeros_like(words))
            embedded=self.tokens(words)
            packed=pack_padded_sequence(embedded,lens.detach().cpu(),batch_first=True,enforce_sorted=False)
            _,hidden=self.rnn(packed,h0.index_select(1,ids).contiguous())
            final=final.index_copy(0,ids,hidden[-1])
        return self.readout(self.norm(final))

    def proxy(self,b,l,active):
        d=self.spec["width"];layers=self.spec["layers"]
        return dict(proxy_forward_macs=active*layers*6*d*d+b*d*self.spec["n_states"],
            proxy_scope="GRU input/recurrent linear products and output head; excludes gates,norm,packing,backward and workspace",
            active_steps=active,architecture=f"ordinary2layer{d}GRU; no taskID; no supplied state transition")


class TerminalTransformer(nn.Module):
    """Default64-wide,3-layer,4-head attention; fixed sinusoidal positions.

    A prefix CLS has the same position for every length. Sinusoidal positions
    avoid an untrained new CLS/absolute-position parameter at long evaluation.
    Position extent is public metadata, not learned from held-out targets.
    """
    def __init__(self,n_states,n_tokens,max_length,width=64,layers=3,heads=4):
        super().__init__();self.spec=dict(n_states=n_states,n_tokens=n_tokens,max_length=max_length,width=width,layers=layers,heads=heads)
        self.tokens=nn.Embedding(n_tokens,width);self.initial=nn.Embedding(n_states,width)
        self.cls=nn.Parameter(torch.randn(1,1,width)*.02)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(width,heads,dim_feedforward=2*width,
            dropout=0.,activation="relu",batch_first=True,norm_first=True) for _ in range(layers)])
        self.norm=nn.LayerNorm(width);self.readout=nn.Linear(width,n_states)
        position=torch.arange(max_length+2,dtype=torch.float32)[:,None]
        frequency=torch.exp(torch.arange(0,width,2,dtype=torch.float32)*(-math.log(10000.)/width))
        encoding=torch.zeros(max_length+2,width)
        encoding[:,0::2]=torch.sin(position*frequency);encoding[:,1::2]=torch.cos(position*frequency)
        self.register_buffer("positions",encoding)

    def forward(self,initial,tokens,lengths):
        b,l=tokens.shape
        if l>self.spec["max_length"]:raise ValueError("program exceeds public positional extent")
        active=torch.arange(l,device=tokens.device)[None,:]<lengths[:,None]
        safe_tokens=torch.where(active,tokens,torch.zeros_like(tokens))
        x=torch.cat((self.cls.expand(b,-1,-1),self.initial(initial)[:,None],self.tokens(safe_tokens)),1)
        x=x+self.positions[:l+2][None]
        padding=torch.cat((torch.zeros((b,2),device=tokens.device,dtype=torch.bool),~active),1)
        for layer in self.layers:x=layer(x,src_key_padding_mask=padding)
        return self.readout(self.norm(x[:,0]))

    def proxy(self,b,l,active):
        d=self.spec["width"];t=l+2;layers=self.spec["layers"]
        return dict(proxy_forward_macs=b*layers*(8*t*d*d+2*t*t*d)+b*d*self.spec["n_states"],
            proxy_scope="attention projections,QK/AV,2dFFN,readout dense MACs; padded attention work included; excludes nonlinearities,norm,backward and workspace",
            active_steps=active,architecture=f"ordinary3layer4head{d}Transformer, sinusoidal positions, prefixCLS, no taskID")


def array_hash(data):
    h=hashlib.sha256()
    for k in KEYS:
        a=np.ascontiguousarray(data[k],dtype="<i8")
        h.update(k.encode()+np.asarray(a.shape,dtype="<i8").tobytes()+a.tobytes())
    return h.hexdigest()


def input_path(data_dir,metadata,stage,split):
    entry=metadata["stages"][stage]
    files=entry.get("files",{}) if isinstance(entry,dict) else {}
    specified=files.get(split,f"stage{stage}-{split}.npz")
    if isinstance(specified,dict):specified=specified.get("file",specified.get("path"))
    if not isinstance(specified,str):raise ValueError("public stage file path must be a string")
    path=(data_dir/specified).resolve()
    if not path.is_relative_to(data_dir.resolve()):raise ValueError("raw stage input must stay within the declared data directory")
    return path


def load_split(args,metadata,stage,split):
    path=input_path(args.data,metadata,stage,split)
    if path.stat().st_size>20*1024*1024:raise ValueError("single raw input exceeds20MiB")
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!=set(KEYS):raise ValueError("only initial/tokens/lengths/targets raw arrays accepted")
        data={k:np.asarray(z[k],dtype=np.int64) for k in KEYS}
    b=len(data["initial"]);tokens=data["tokens"];lengths=data["lengths"]
    if (not 1<=b<=32768 or tokens.ndim!=2 or tokens.shape[0]!=b
            or not 1<=tokens.shape[1]<=metadata["max_length"]
            or any(data[k].shape!=(b,) for k in ("initial","lengths","targets"))
            or np.any((lengths<0)|(lengths>tokens.shape[1]))
            or np.any((data["initial"]<0)|(data["initial"]>=args.states))
            or np.any((data["targets"]<0)|(data["targets"]>=args.states))):
        raise ValueError("raw terminal data shape/range differs from public metadata")
    active=np.arange(tokens.shape[1])[None,:]<lengths[:,None]
    if np.any(active&((tokens<0)|(tokens>=metadata["n_tokens"]))):raise ValueError("active token outside public embedding range")
    return data,dict(stage=stage,split=split,path=str(path),file_sha256=sha256(path),array_sha256=array_hash(data),
        rows=b,array_bytes=sum(a.nbytes for a in data.values()),padded_length=tokens.shape[1],
        observed_length_counts={str(x):int((lengths==x).sum()) for x in np.unique(lengths)})


def batch(data,indices,device):
    lengths=data["lengths"][indices]
    # Same label-independent ragged crop for ALL arms, based only on supplied
    # lengths. PAD0 is not used as an active-token/termination oracle.
    width=max(1,int(lengths.max(initial=0)))
    return dict(initial=torch.as_tensor(data["initial"][indices],device=device,dtype=torch.long),
        tokens=torch.as_tensor(data["tokens"][indices,:width],device=device,dtype=torch.long),
        lengths=torch.as_tensor(lengths,device=device,dtype=torch.long),
        targets=torch.as_tensor(data["targets"][indices],device=device,dtype=torch.long)),width,int(lengths.sum())


def build(args,metadata,device):
    common=dict(n_states=args.states,n_tokens=metadata["n_tokens"],max_length=metadata["max_length"])
    if args.arm=="gru":return TerminalGRU(**common,width=getattr(args,"baseline_width",64)).to(device)
    if args.arm=="transformer":return TerminalTransformer(**common,width=getattr(args,"baseline_width",64)).to(device)
    if args.arm=="ssm":
        from selective_ssm_baseline import TerminalSelectiveSSM
        return TerminalSelectiveSSM(**common).to(device)
    cls=block_class(getattr(args,"block_state","categorical"))
    if getattr(args,"block_state","categorical")=="continuous":
        width=args.block_width*(2 if args.arm=="static_expand" else 1)
        return cls(n_states=args.states,n_tokens=metadata["n_tokens"],internal_states=width,
            n_primitives=args.primitives,token_dim=16,state_dim=width,memory_dim=width,
            caller_hidden=32,n_slots=args.slots,routing="dynamic" if args.arm=="dynamic" else "static",
            memory_slots=args.memory_slots,key_dim=16,primitive_rank=args.primitive_rank).to(device)
    return cls(n_states=args.states,n_tokens=metadata["n_tokens"],
        internal_states=args.states*(2 if args.arm=="static_expand" else 1),n_primitives=args.primitives,
        token_dim=16,state_dim=16,memory_dim=16,caller_hidden=32,n_slots=args.slots,
        routing="dynamic" if args.arm=="dynamic" else "static",initialization=args.block_initialization,
        straight_through_backward=args.straight_through_backward).to(device)


def is_block(args):return args.arm in ("dynamic","static","static_expand")


def block_class(block_state="categorical"):
    """Lazy factory: baseline/categorical runs need no continuous dependency."""
    if block_state=="continuous":
        from continuous_primitive_block import ContinuousPrimitiveBlock
        return ContinuousPrimitiveBlock
    if block_state=="categorical":
        from contextual_primitive_block import ContextualPrimitiveBlock
        return ContextualPrimitiveBlock
    raise ValueError("unknown Block state organization")


def forward(model,b,args,execution):
    if is_block(args):return model(b["initial"],b["tokens"],b["lengths"],execution=execution,output="logits")
    return model(b["initial"],b["tokens"],b["lengths"])


def loss_value(logits,targets,args,reduction="mean"):
    return (F.nll_loss(logits,targets,reduction=reduction) if is_block(args)
            else F.cross_entropy(logits,targets,reduction=reduction))


def work_proxy(model,args,b,l,active,execution):
    return (model.resource_report(batch_size=b,sequence_length=l,active_steps=active,
                execution=execution,return_traces=False) if is_block(args) else model.proxy(b,l,active))


def resources(model,optimizer,device,replay=None):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result=dict(parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        registered_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
        optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),replay_bytes=0,
        cuda_allocated_bytes=torch.cuda.memory_allocated(device),cuda_reserved_bytes=torch.cuda.memory_reserved(device),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        cpu_peak_RSS_bytes=int(rss if sys.platform=="darwin" else rss*1024),
        memory_scope="actual process/allocator counters; peak is cumulative and includes audits; no energy claim")
    if replay is not None:
        result["replay_bytes"]=sum(v.nbytes for v in replay.arrays.values())
        result["replay_metadata_JSON_payload_bytes"]=len(json.dumps(plain(replay.statistics()),sort_keys=True).encode())
        result["replay_storage_scope"]="128 raw/provenance array slots plus separately reported metadata; dataset/evidence arrays are not training replay capacity"
    return result


@torch.no_grad()
def evaluate(model,data,args,device,rec,modes=("soft","native")):
    was=model.training;model.eval();result={};synchronize(device)
    began,cpu=time.perf_counter(),time.process_time();total_proxy=0
    try:
        for mode in modes if is_block(args) else ("soft",):
            correct=0;nll=0.;proxy=0
            for offset in range(0,len(data["initial"]),64):
                rec.check();ids=np.arange(offset,min(offset+64,len(data["initial"])))
                b,l,active=batch(data,ids,device);out=forward(model,b,args,mode)
                if not bool(torch.isfinite(out).all()):raise FloatingPointError("nonfinite actual evaluation output")
                correct+=int((out.argmax(-1)==b["targets"]).sum());nll+=float(loss_value(out,b["targets"],args,"sum"))
                proxy+=work_proxy(model,args,len(ids),l,active,mode)["proxy_forward_macs"]
            result[mode]=dict(accuracy=correct/len(data["initial"]),terminal_NLL=nll/len(data["initial"]),
                rows=len(data["initial"]),forward_MAC_proxy=proxy)
            total_proxy+=proxy
        if not is_block(args):result["native"]=dict(result["soft"],same_neural_interface=True)
    finally:synchronize(device);model.train(was)
    return result,dict(wall_seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu,
        forward_MAC_proxy=total_proxy,actual_forward_examples=len(data["initial"])*(2 if is_block(args) else 1))


def rng_state():
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate())


def restore_rng(s):
    torch.set_rng_state(s["cpu"]);torch.cuda.set_rng_state_all(s["cuda"]);np.random.set_state(s["numpy"]);random.setstate(s["python"])


def tensor_state_equal(a,b):
    return a.keys()==b.keys() and all(torch.equal(x.detach().cpu(),b[k].detach().cpu()) for k,x in a.items())


def nested_state_equal(left,right):
    if torch.is_tensor(left):return torch.is_tensor(right) and torch.equal(left.detach().cpu(),right.detach().cpu())
    if isinstance(left,np.ndarray):return isinstance(right,np.ndarray) and np.array_equal(left,right)
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(nested_state_equal(v,right[k]) for k,v in left.items())
    if isinstance(left,(list,tuple)):return isinstance(right,(list,tuple)) and len(left)==len(right) and all(nested_state_equal(a,b) for a,b in zip(left,right))
    return left==right


def execution_signature(arm):
    """Source AST fingerprint of actual execution, not a whole-driver hash."""
    tree=ast.parse(Path(__file__).read_text());nodes=[]
    for name in ("batch","is_block","forward","loss_value","rng_state","restore_rng"):
        nodes.append(next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name))
    if arm in ("gru","transformer"):
        name="TerminalGRU" if arm=="gru" else "TerminalTransformer"
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==name)
        nodes.extend(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name in ("__init__","forward"))
    for node in nodes:
        if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):
            node.body=node.body[1:]
    return hashlib.sha256("\n".join(ast.dump(node,include_attributes=False) for node in nodes).encode()).hexdigest()


def validate_resume(payload,args,metadata,receipts):
    """Owned stage0 continuation, strict applied policy/input/source checks."""
    if payload.get("schema")!="contextual-primitives-training-v1" or payload.get("stage")!=0 or payload.get("step")!=600 or payload.get("stage_complete") is not True:
        raise ValueError("resume requires an owned complete stage0/step600 checkpoint")
    if args.stages!=3 or args.steps!=600:
        raise ValueError("resume requires current --stages3 --steps600; only stages1..2 will train")
    previous=payload["args"]
    checked={}
    for name,default in (("arm",None),("states",None),("seed",None),("lr",.003),("steps",600)):
        if previous.get(name,default)!=getattr(args,name):raise ValueError(f"resume policy mismatch: {name}")
        checked[name]=getattr(args,name)
    if is_block(args):
        fields=[("block_state","categorical"),("primitives",4),("slots",4),("block_training","scheduled")]
        fields += ([("block_width",64),("primitive_rank",8),("memory_slots",8)] if args.block_state=="continuous"
                   else [("block_initialization","iid"),("straight_through_backward","selected")])
        for name,default in fields:
            if previous.get(name,default)!=getattr(args,name):raise ValueError(f"resume Block policy mismatch: {name}")
            checked[name]=getattr(args,name)
    elif args.arm in ("gru","transformer"):
        if payload["baseline_spec"]["width"]!=args.baseline_width:raise ValueError("resume baseline width differs from saved actual shape")
        checked["baseline_width"]=args.baseline_width
    public={key:metadata[key] for key in ("n_states","n_tokens","max_length")}
    if payload["metadata_public"]!=public:raise ValueError("resume public state/vocabulary/horizon differs")
    old={(r["stage"],r["split"]):r for r in payload["input_receipts"] if r.get("stage")==0 and r.get("split") in ("train","valid")}
    for receipt in receipts:
        if receipt["stage"]!=0:continue
        key=(0,receipt["split"])
        if key not in old or any(old[key].get(k)!=receipt.get(k) for k in ("array_sha256","file_sha256","rows")):
            raise ValueError("resume old TRAIN/VALID data differs from checkpoint lineage")
    current_sources=source_hashes(args.arm,args.block_state)
    dependencies=[name for name in current_sources if name!=Path(__file__).name]
    for name in dependencies:
        if payload["source_sha256"].get(name)!=current_sources[name]:raise ValueError(f"resume execution dependency changed: {name}")
    signature=execution_signature(args.arm)
    if "execution_signature" in payload:
        if payload["execution_signature"]!=signature:raise ValueError("resume execution/objective/batching source behavior changed")
        compatibility="saved execution signature matched"
    else:
        old_driver=payload["source_sha256"].get("train_contextual_primitives.py")
        category=args.arm if args.arm in ("gru","transformer") else "block"
        if old_driver not in LEGACY_RESUME_DRIVERS or signature!=LEGACY_EXECUTION_SIGNATURES[category]:
            raise ValueError("unrecognized legacy resume driver behavior; preserve source and inspect compatibility")
        compatibility="recognized owned legacy driver; execution AST and applied policy matched"
    if not all(name in payload["rng"] for name in ("cpu","cuda","numpy","python")):
        raise ValueError("resume checkpoint lacks actual RNG states")
    return dict(checkpoint=str(args.resume),checkpoint_sha256=sha256(args.resume),checkpoint_bytes=args.resume.stat().st_size,
        saved_stage=0,saved_step=600,saved_declared_stages=previous.get("stages"),current_declared_stages=args.stages,
        first_new_training_stage=1,expected_new_training_updates=1200,old_stage_training_repeated=False,
        replay_examples=getattr(args,"replay_capacity",0),historical_replay_examples=previous.get("replay_capacity",0),
        checked_applied_policy=checked,public_metadata_checked=public,stage0_TRAIN_VALID_hashes_matched=True,
        source_compatibility=compatibility,execution_signature=signature,checked_dependency_hashes={name:current_sources[name] for name in dependencies},
        historical_source_sha256=copy.deepcopy(payload["source_sha256"]),historical_checkpoint_costs=copy.deepcopy(payload["costs"]),
        historical_cost_scope="saved phase0 checkpoint costs; original run retains any later evaluation/test costs",
        new_costs_scope="separate stage1/stage2 counters,600 updates each; resume construction/audits logged separately")


def save_checkpoint(model,optimizer,args,metadata,rec,stage,step,complete,inputs,costs,suffix="",lineage=None,replay=None):
    path=rec.out/f"{args.arm}-stage{stage}-step{step}{suffix}.pt"
    if path.exists():raise FileExistsError("checkpoint evidence is append-only")
    began=time.perf_counter()
    payload=dict(schema="contextual-primitives-training-v1",args=vars(args),metadata_public=dict(
        n_states=metadata["n_states"],n_tokens=metadata["n_tokens"],max_length=metadata["max_length"]),
        stage=stage,step=step,stage_complete=complete,model_state=model.state_dict(),
        block_checkpoint=model.checkpoint() if is_block(args) else None,
        baseline_spec=None if is_block(args) else model.spec,optimizer=optimizer.state_dict(),
        parameter_flags={n:p.requires_grad for n,p in model.named_parameters()},rng=rng_state(),
        input_receipts=inputs,costs=costs,source_sha256=source_hashes(args.arm,getattr(args,"block_state","categorical"),getattr(args,"replay_capacity",0)),
        execution_signature=execution_signature(args.arm),resume_lineage=lineage,
        replay_state=replay.state_dict() if replay is not None else None,
        claim_scope="finite architecture development; not frozen Primary, novelty or Monodromy-benefit evidence")
    pending=path.with_suffix(".pt.partial");torch.save(payload,pending)
    if pending.stat().st_size>MAX_FILE:raise RuntimeError("checkpoint100MiB bound exceeded; partial retained")
    pending.rename(path)
    receipt=dict(file=path.name,sha256=sha256(path),bytes=path.stat().st_size,wall_seconds=time.perf_counter()-began)
    rec.emit("actual_neural_checkpoint",stage=stage,step=step,complete=complete,**receipt)
    return path,receipt


def restore_model(payload,device):
    args=argparse.Namespace(**payload["args"])
    if is_block(args):
        model=block_class(getattr(args,"block_state","categorical")).from_checkpoint(payload["block_checkpoint"]).to(device)
    elif args.arm=="gru":model=TerminalGRU(**payload["baseline_spec"]).to(device)
    elif args.arm=="ssm":
        from selective_ssm_baseline import TerminalSelectiveSSM
        model=TerminalSelectiveSSM(**payload["baseline_spec"]).to(device)
    else:model=TerminalTransformer(**payload["baseline_spec"]).to(device)
    model.load_state_dict(payload["model_state"],strict=True)
    for n,p in model.named_parameters():p.requires_grad_(payload["parameter_flags"][n])
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr);optimizer.load_state_dict(payload["optimizer"])
    return model,optimizer,args


def reload_audit(path,model,optimizer,args,valid,device,rec,replay=None):
    saved=rng_state();was=model.training;synchronize(device);began=time.perf_counter()
    other=other_optimizer=None
    try:
        payload=torch.load(path,map_location="cpu",weights_only=False)
        other,other_optimizer,_=restore_model(payload,device)
        if not tensor_state_equal(payload["model_state"],other.state_dict()):raise RuntimeError("saved actual neural tensors differ after reload")
        replay_equal=None
        if replay is not None:
            from context_replay128 import Replay128
            restored_replay=Replay128.from_state_dict(payload["replay_state"])
            replay_equal=nested_state_equal(restored_replay.state_dict(),replay.state_dict())
            if not replay_equal:raise RuntimeError("saved replay arrays/counters/local RNG differ after reload")
        model.eval();other.eval();b,_,_=batch(valid,np.arange(min(64,len(valid["initial"]))),device)
        diffs={}
        with torch.no_grad():
            for mode in ("soft","native") if is_block(args) else ("soft",):
                left,right=forward(model,b,args,mode),forward(other,b,args,mode)
                if not torch.equal(left,right):raise RuntimeError("actual same-weight reload changed executed outputs")
                diffs[mode]=float((left-right).abs().max())
        synchronize(device)
        report=dict(actual_model_tensors_equal=True,actual_forward_equal=True,checked_rows=len(b["initial"]),
            maximum_difference=diffs,optimizer_tensor_bytes_before=tensor_bytes(optimizer.state_dict()),
            optimizer_tensor_bytes_restored=tensor_bytes(other_optimizer.state_dict()),
            replay_state_exactly_restored=replay_equal,
            wall_seconds=time.perf_counter()-began,training_rng_restored_after_neutral_factory=True)
        rec.emit("actual_neural_reload_audit",**report);return report
    finally:
        model.train(was);del other,other_optimizer;restore_rng(saved)


@torch.no_grad()
def inference_report(model,data,args,device,rec):
    ids=np.arange(min(64,len(data["initial"])));b,l,active=batch(data,ids,device)
    result={};model.eval()
    for mode in ("soft","native") if is_block(args) else ("soft",):
        for _ in range(3):rec.check();forward(model,b,args,mode)
        synchronize(device);wall=[];gpu=[]
        for _ in range(20):
            rec.check();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            began=time.perf_counter();start.record();forward(model,b,args,mode);end.record();synchronize(device)
            wall.append(time.perf_counter()-began);gpu.append(start.elapsed_time(end)/1000.)
        result[mode]=dict(batch_size=len(ids),word_length_max=l,wall_latency_p50=float(np.median(wall)),
            wall_latency_p95=float(np.percentile(wall,95)),CUDA_stream_seconds_p50=float(np.median(gpu)),
            examples_per_wall_second=len(ids)/float(np.median(wall)),warmups=3,timed_forwards=20,
            proxy=work_proxy(model,args,len(ids),l,active,mode),
            timing_scope="actual process while other jobs may overlap; CUDA event elapsed is not GPU-busy time")
    return result


def source_hashes(arm=None,block_state="categorical",replay_capacity=0):
    folder=Path(__file__).resolve().parent
    names=[Path(__file__).name]
    if arm in ("dynamic","static","static_expand"):
        names.append("continuous_primitive_block.py" if block_state=="continuous" else "contextual_primitive_block.py")
    if arm=="ssm":names.append("selective_ssm_baseline.py")
    if replay_capacity:names.append("context_replay128.py")
    return {n:sha256(folder/n) for n in names}


def save_replay_snapshot(replay,rec,stage):
    began=time.perf_counter();path=rec.out/f"stage{stage}-replay128.npz"
    with path.open("xb") as stream:
        np.savez_compressed(stream,**replay.arrays,size=np.asarray(replay.size,dtype=np.int64),
            seen=np.asarray(replay.seen,dtype=np.int64))
    result=dict(file=path.name,sha256=sha256(path),bytes=path.stat().st_size,
        buffer_payload_sha256=replay.digest(),wall_seconds=time.perf_counter()-began,statistics=replay.statistics())
    rec.emit("actual_replay_snapshot",stage=stage,**result);return result


def bootstrap_replay(args,metadata,datasets,rec,lineage):
    from context_replay128 import Replay128,self_test
    began=time.perf_counter();before=rng_state()
    rec.emit("replay_protocol_self_test",**self_test())
    manifest=json.loads((args.resume.parent/"resume-lineage.json").read_text())
    reference=manifest["replay_stage0_history"]
    path=args.resume.parent/"replay-stage0-indices.npz"
    if sha256(path)!=reference["sha256"] or manifest["parent_checkpoint_sha256"]!=lineage["checkpoint_sha256"]:
        raise ValueError("owned replay history/checkpoint lineage differs")
    if reference["actual_completed_stage0_updates"]!=600 or reference["actual_indices_file_verified"] is not True:
        raise ValueError("parent paid stage0 history not verified")
    with np.load(path,allow_pickle=False) as z:
        if z.files!=["indices"]:raise ValueError("only original used-index array accepted")
        indices=np.asarray(z["indices"])
    expected=np.random.default_rng(np.random.SeedSequence([args.seed,args.states,0,9027])).integers(
        0,len(datasets[0]["train"]["initial"]),size=(600,64),dtype=np.int64)
    if indices.dtype!=np.int64 or indices.shape!=(600,64) or not np.array_equal(indices,expected):
        raise ValueError("actual parent used TRAIN indices differ from frozen common stream")
    replay=Replay128(args.replay_capacity,metadata["max_length"],args.seed,args.states)
    consume=replay.consume(datasets[0]["train"],indices,0,rec.emit)
    audit=replay.validate_sources({0:datasets[0]["train"]},{0:indices})
    snapshot=save_replay_snapshot(replay,rec,0)
    if not nested_state_equal(before,rng_state()):raise RuntimeError("replay setup modified saved training RNG")
    receipt=dict(parent_indices_file_sha256=reference["sha256"],parent_actual_exposures=indices.size,
        bootstrap_wall_seconds=time.perf_counter()-began,consume=consume,source_audit=audit,snapshot=snapshot,
        training_RNG_unchanged=True,old_label_reaccesses_for_reservoir_writes=consume["actual_buffer_writes"],
        old_label_reaccesses_for_source_audit=audit["checked_buffer_rows"],
        index_evidence_array_bytes=indices.nbytes,valid_or_test_used_for_replay=False)
    lineage["replay_intervention"]=dict(capacity=args.replay_capacity,per_batch=args.replay_per_batch,
        current_per_batch=64-args.replay_per_batch,first_new_stage=1,bootstrap=receipt,
        reservoir_unit="actual committed current-example exposure including repeats",no_extra_SGD_updates=True)
    rec.emit("replay_bootstrap",**receipt);return replay,{0:indices},receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--resume",type=Path,default=None,
        help="read-only owned completed stage0/step600 checkpoint; train stages1..2 only")
    p.add_argument("--replay-capacity",type=int,choices=(0,128),default=0)
    p.add_argument("--replay-per-batch",type=int,choices=(0,16),default=0)
    p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--states",type=int,default=8)
    p.add_argument("--seed",type=int,default=0);p.add_argument("--stages",type=int,choices=(1,3),default=1)
    p.add_argument("--steps",type=int,default=600);p.add_argument("--primitives",type=int,default=4)
    p.add_argument("--baseline-width",type=int,default=64)
    p.add_argument("--slots",type=int,default=None);p.add_argument("--lr",type=float,default=.003)
    p.add_argument("--block-state",choices=("categorical","continuous"),default="categorical")
    p.add_argument("--block-width",type=int,default=64)
    p.add_argument("--primitive-rank",type=int,default=8)
    p.add_argument("--memory-slots",type=int,default=8)
    p.add_argument("--block-initialization",choices=("iid","carry"),default="iid")
    p.add_argument("--block-training",choices=("scheduled","soft_only"),default="scheduled")
    p.add_argument("--straight-through-backward",choices=("selected","dense"),default="selected")
    p.add_argument("--device",choices=("cuda:0",),default="cuda:0")
    p.add_argument("--max-seconds",type=float,default=850.)
    p.add_argument("--resource-overlap-note",default="coordinator must record actual concurrent jobs; no isolated-throughput claim")
    args=p.parse_args()
    if args.slots is None:args.slots=1 if args.block_state=="continuous" else 4
    if (args.out.exists() or not 1<=args.steps<=600 or not 1<=args.states<=32
            or not 1<=args.primitives<=16 or not 1<=args.slots<=8
            or not 1<=args.block_width<=256 or not 1<=args.primitive_rank<=args.block_width
            or not 1<=args.memory_slots<=32
            or not 1<=args.baseline_width<=256 or (args.arm=="transformer" and args.baseline_width%4!=0)
            or not 0<args.lr<=.05 or not 0<args.max_seconds<=850):
        p.error("new output, finite states/K/slots/600-update/850s bounds required")
    if args.resume is not None and (args.stages!=3 or args.steps!=600 or not args.resume.is_file()
                                   or args.resume.stat().st_size>MAX_FILE):
        p.error("--resume requires owned <=100MiB checkpoint and --stages3 --steps600")
    if (args.replay_capacity,args.replay_per_batch) not in ((0,0),(128,16)):
        p.error("replay requires exactly --replay-capacity128 --replay-per-batch16")
    if args.replay_capacity and args.resume is None:p.error("Replay128 is only an owned stage0 continuation intervention")
    rec=Recorder(args.out,args.max_seconds);device=torch.device(args.device)
    model=optimizer=None;stage=-1;completed=0;last_safe=False;reports=[];receipts=[];stage_paths=[];lineage=None
    replay=None;used_sources={};replay_bootstrap_receipt=None
    start_stage=0
    try:
        if not torch.cuda.is_available():raise RuntimeError("authorized CUDA required; no CPU fallback")
        torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,device)
        torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
        metadata_path=args.data/"metadata.json";metadata=json.loads(metadata_path.read_text())
        if (metadata["n_states"]!=args.states or not 1<=metadata["n_tokens"]<=256
                or not 1<=metadata["max_length"]<=128 or len(metadata["stages"])<args.stages):
            raise ValueError("public metadata differs from declared state/vocab/horizon/stage bounds")
        datasets=[]
        for t in range(args.stages):
            parts={}
            for split in ("train","valid"):
                parts[split],receipt=load_split(args,metadata,t,split);receipts.append(receipt)
            datasets.append(parts)
        if args.resume is None:
            model=build(args,metadata,device);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
        else:
            began=time.perf_counter()
            payload=torch.load(args.resume,map_location="cpu",weights_only=False)
            lineage=validate_resume(payload,args,metadata,receipts)
            # A neutral current-CLI construction verifies actual requestedshape.
            # The saved factory then restores the real NN and optimizer. Neither
            # factory's RNG draws are allowed to change the continuation stream.
            requested=build(args,metadata,device)
            requested_spec=requested.configuration() if is_block(args) else requested.spec
            requested_shapes={name:(tuple(value.shape),value.dtype) for name,value in requested.state_dict().items()}
            del requested
            model,optimizer,_saved_args=restore_model(payload,device)
            restored_spec=model.configuration() if is_block(args) else model.spec
            if restored_spec!=requested_spec or requested_shapes!={name:(tuple(value.shape),value.dtype) for name,value in model.state_dict().items()}:
                raise ValueError("resume actual architecture differs from current CLI factory")
            if set(payload["parameter_flags"])!=set(dict(model.named_parameters())):
                raise ValueError("resume parameter flags do not match actual named parameters")
            if not tensor_state_equal(payload["model_state"],model.state_dict()):
                raise RuntimeError("resume actual NN tensor values changed")
            if not nested_state_equal(payload["optimizer"],optimizer.state_dict()):
                raise RuntimeError("resume actual Adam state changed")
            if len(payload["rng"]["cuda"])!=torch.cuda.device_count():
                raise ValueError("resume CUDA RNG device count differs")
            restore_rng(payload["rng"])
            if not nested_state_equal(payload["rng"],rng_state()):raise RuntimeError("resume RNG restoration mismatch")
            lineage.update(actual_NN_tensors_equal=True,actual_optimizer_equal=True,actual_RNG_equal_after_neutral_factories=True,
                neutral_model_factory_constructions=2,restored_optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
                restore_wall_seconds=time.perf_counter()-began,current_CLI_preserved=True)
            rec.emit("actual_stage0_resume",**lineage)
            start_stage=1
            del payload
        if args.replay_capacity:
            replay,used_sources,replay_bootstrap_receipt=bootstrap_replay(args,metadata,datasets,rec,lineage)
        rec.emit("configuration",args=vars(args),public_metadata=dict(n_states=metadata["n_states"],n_tokens=metadata["n_tokens"],
            max_length=metadata["max_length"],stages_requested=args.stages),metadata_sha256=sha256(metadata_path),
            inputs=receipts,source_sha256=source_hashes(args.arm,args.block_state,args.replay_capacity),torch=torch.__version__,CUDA=torch.version.cuda,
            GPU=torch.cuda.get_device_name(device),parameters=resources(model,optimizer,device,replay),
            resume_lineage=lineage,first_training_stage=start_stage,
            declared_new_training_updates=(args.stages-start_stage)*args.steps,
            policy=("48 current rows(first48 of original64 common indices)+16 uniformly sampled raw old-buffer rows;128 total reservoir; terminal labels only"
                    if replay is not None else "current-stage64-row batches only; same seed[seed,N,stage,9027]; no replay/taskID/teacher routes"),
            replay_bootstrap=replay_bootstrap_receipt,
            loaded_train_valid_array_bytes=sum(a.nbytes for part in datasets for split in ("train","valid") for a in part[split].values()),
            dataset_storage_scope="common full TRAIN/VALID arrays used for prescribed evaluation; no use of unbuffered old rows in gradient batches",
            block_state_organization=args.block_state if is_block(args) else None,
            block_configuration=model.configuration() if is_block(args) else None,
            block_resource_model=model.resource_report() if is_block(args) else None,
            block_schedule=("all updates soft" if args.block_training=="soft_only" else
                ("per-stage steps1..200 soft;201..600 hard-call ST with continuous state/E/R/memory; native hardens CALL only"
                 if args.block_state=="continuous" else
                 "per-stage steps1..200 soft;201..600 hard-ST calls/op rows with SOFT E/R; native evaluation hardens E/R")),
            straight_through_backward=(args.straight_through_backward if args.block_state=="categorical"
                                       else "continuous-module dense joint-soft CALL surrogate"),
            categorical_initialization_flags_applied=is_block(args) and args.block_state=="categorical",
            standard_baselines=f"two-layerGRU{args.baseline_width};3layer4headTransformer{args.baseline_width},sinusoidal positions,no extra stage embedding",
            ssm_scope="source-grounded selective-SSM reference adaptation; NOT official Mamba/Mamba2 or fused-kernel cost reproduction",
            control_scope=("ordinary same-mechanism reference; static differs ONLY CALL h/m inputs; static_expand preallocates2x continuous width, all state/memory/P counted"
                           if args.block_state=="continuous" else
                           "ordinary dynamic routing is a same-mechanism reference; static differs ONLY caller inputs;2N is preallocated and counted"),
            validity="implementation/protocol/reproduction, never final accuracy",formal_Primary=False,
            test_policy="test files opened only after ALL declared training finishes and all stage checkpoints are frozen",
            exact_training_FLOPs=None,FLOP_boundary="forward MAC proxies plus actual synchronized timings; no full-autograd FLOP claim")
        for stage in range(start_stage,args.stages):
            rec.check();stage_begin,stage_cpu=time.perf_counter(),time.process_time();completed=0
            torch.cuda.reset_peak_memory_stats(device)
            data=datasets[stage]["train"]
            indices=np.random.default_rng(np.random.SeedSequence([args.seed,args.states,stage,9027])).integers(
                0,len(data["initial"]),size=(args.steps,64),dtype=np.int64)
            ipath=rec.out/f"stage{stage}-common-indices.npz";np.savez_compressed(ipath,indices=indices)
            replay_draws=None
            if replay is not None:
                before=rng_state();setup=time.perf_counter()
                replay_draws=replay.sample_indices(stage,args.steps,args.replay_per_batch)
                draw_path=rec.out/f"stage{stage}-replay-draws.npz"
                with draw_path.open("xb") as stream:np.savez_compressed(stream,buffer_indices=replay_draws)
                if not nested_state_equal(before,rng_state()):raise RuntimeError("local replay draws changed training RNG")
                rec.emit("replay_stage_draw_schedule",stage=stage,file=draw_path.name,sha256=sha256(draw_path),
                    draw_shape=list(replay_draws.shape),buffer_sha256=replay.digest(),
                    local_rng=True,buffer_fixed_during_stage=True,setup_wall_seconds=time.perf_counter()-setup)
            costs=dict(training_wall_seconds=0.,training_CPU_seconds=0.,CUDA_stream_elapsed_seconds=0.,
                batch_assembly_wall_seconds=0.,evaluation_wall_seconds=0.,evaluation_CPU_seconds=0.,
                training_forward_MAC_proxy=0,evaluation_forward_MAC_proxy=0,training_examples=0,
                evaluation_forward_examples=0,SGD_updates=0,old_examples_used_for_training=0,
                current_examples_used_for_training=0,old_training_exposures_by_source_stage={},
                replay_maintenance_wall_seconds=0.,replay_setup_wall_seconds=(time.perf_counter()-setup if replay is not None else 0.))
            curves=[]
            rec.emit("stage_begin",stage=stage,indices_sha256=sha256(ipath),optimizer_retained_across_stages=True,
                parameters=resources(model,optimizer,device,replay),phase_schedule_fixed_before_test=True,
                replay_statistics=replay.statistics() if replay is not None else None)
            for step in range(args.steps+1):
                rec.check();last_safe=True
                if step%100==0 or step==args.steps:
                    allmetrics={}
                    for old in range(stage+1):
                        allmetrics[old]={}
                        for split in ("train","valid"):
                            measured,timing=evaluate(model,datasets[old][split],args,device,rec)
                            allmetrics[old][split]=measured
                            costs["evaluation_wall_seconds"]+=timing["wall_seconds"]
                            costs["evaluation_CPU_seconds"]+=timing["cpu_seconds"]
                            costs["evaluation_forward_MAC_proxy"]+=timing["forward_MAC_proxy"]
                            costs["evaluation_forward_examples"]+=timing["actual_forward_examples"]
                    row=dict(stage=stage,step=step,metrics=allmetrics,resources=resources(model,optimizer,device,replay),costs=copy.deepcopy(costs))
                    curves.append(row);rec.emit("TRAIN_VALID_curve",**row)
                    if step and step!=args.steps:
                        save_checkpoint(model,optimizer,args,metadata,rec,stage,step,False,receipts,costs,lineage=lineage,replay=replay)
                if step==args.steps:break
                began=time.perf_counter()
                if replay is None:
                    b,l,active=batch(data,indices[step],device);current_count=64;old_counts={}
                else:
                    current_count=64-args.replay_per_batch
                    mixed=replay.mix(data,indices[step,:current_count],replay_draws[step])
                    b,l,active=batch(mixed,np.arange(64),device)
                    origins=replay.arrays["source_stage"][replay_draws[step]]
                    if np.any(origins>=stage):raise RuntimeError("current/future task entered old replay during stage")
                    old_counts={str(int(s)):int((origins==s).sum()) for s in np.unique(origins)}
                costs["batch_assembly_wall_seconds"]+=time.perf_counter()-began
                execution="soft" if args.block_training=="soft_only" or step<200 else "straight_through"
                model.train();synchronize(device);began,cpu=time.perf_counter(),time.process_time()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();last_safe=False;optimizer.zero_grad(set_to_none=True)
                output=forward(model,b,args,execution);loss=loss_value(output,b["targets"],args)
                if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite actual terminal training loss")
                loss.backward()
                gradients=None
                if step in (0,99,199,200,299,args.steps-1):
                    gradients={n:None if q.grad is None else float(q.grad.detach().norm()) for n,q in model.named_parameters()}
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not bool(torch.isfinite(norm)):raise FloatingPointError("nonfinite actual parameter gradients")
                optimizer.step();end.record();synchronize(device);last_safe=True;completed=step+1
                elapsed,cpu_elapsed,gpu_elapsed=time.perf_counter()-began,time.process_time()-cpu,start.elapsed_time(end)/1000.
                proxy=work_proxy(model,args,len(b["initial"]),l,active,execution)
                costs["training_wall_seconds"]+=elapsed;costs["training_CPU_seconds"]+=cpu_elapsed
                costs["CUDA_stream_elapsed_seconds"]+=gpu_elapsed;costs["training_forward_MAC_proxy"]+=proxy["proxy_forward_macs"]
                costs["training_examples"]+=len(b["initial"]);costs["SGD_updates"]+=1
                costs["current_examples_used_for_training"]+=current_count
                costs["old_examples_used_for_training"]+=64-current_count
                for origin,count in old_counts.items():
                    costs["old_training_exposures_by_source_stage"][origin]=costs["old_training_exposures_by_source_stage"].get(origin,0)+count
                rec.emit("actual_update",stage=stage,step=step+1,execution=execution if is_block(args) else "standard",
                    terminal_loss=float(loss.detach()),batch_accuracy=float((output.argmax(-1)==b["targets"]).float().mean()),
                    preclip_gradient=float(norm),selected_gradient_norms=gradients,
                    training_wall_seconds=elapsed,training_CPU_seconds=cpu_elapsed,CUDA_stream_elapsed_seconds=gpu_elapsed,
                    forward_MAC_proxy=proxy["proxy_forward_macs"],active_token_steps=active,padded_length_passed=l,
                    current_training_examples=current_count,replayed_training_examples=64-current_count,
                    replayed_source_stages=old_counts)
            replay_boundary=None
            if replay is not None:
                before=rng_state();maintenance=time.perf_counter()
                if completed!=args.steps or costs["SGD_updates"]!=args.steps:
                    raise RuntimeError("cannot update reservoir from unexecuted current proposals")
                used_sources[stage]=indices[:completed,:64-args.replay_per_batch]
                consume=replay.consume(data,used_sources[stage],stage,rec.emit)
                audit=replay.validate_sources({t:datasets[t]["train"] for t in used_sources},used_sources)
                snapshot=save_replay_snapshot(replay,rec,stage)
                if not nested_state_equal(before,rng_state()):raise RuntimeError("replay boundary changed training RNG")
                costs["replay_maintenance_wall_seconds"]+=time.perf_counter()-maintenance
                replay_boundary=dict(consume=consume,source_audit=audit,snapshot=snapshot,
                    current_label_reaccesses_for_reservoir_writes=consume["actual_buffer_writes"],
                    old_and_current_label_reaccesses_for_source_audit=audit["checked_buffer_rows"],
                    maintenance_wall_seconds=costs["replay_maintenance_wall_seconds"],all_sources_actual_committed_TRAIN=True,
                    used_source_index_evidence_array_bytes=sum(a.nbytes for a in used_sources.values()),
                    evidence_storage_scope="used-index proof arrays persist for audit and are charged separately from raw128-row training buffer")
                rec.emit("replay_stage_boundary",stage=stage,**replay_boundary)
            path,cp=save_checkpoint(model,optimizer,args,metadata,rec,stage,args.steps,True,receipts,costs,lineage=lineage,replay=replay)
            audit=reload_audit(path,model,optimizer,args,datasets[stage]["valid"],device,rec,replay)
            summary=dict(stage=stage,final_TRAIN_VALID=curves[-1]["metrics"],curves=curves,costs=costs,
                resources=resources(model,optimizer,device,replay),checkpoint=cp,reload=audit,replay_boundary=replay_boundary,
                stage_wall_seconds=time.perf_counter()-stage_begin,stage_CPU_seconds=time.process_time()-stage_cpu)
            reports.append(summary);stage_paths.append((stage,path));rec.emit("stage_training_complete",**summary)
            rec.json(f"completed-training-through-stage{stage}.json",dict(reports=reports,resume_lineage=lineage,test_data_read=False))
        # No more training, candidate selection, configuration changes or retries.
        # Retrospective tests of each already frozen stage preserve a genuine
        # continual test curve without exposing any result to later training.
        for p0 in model.parameters():p0.requires_grad_(False)
        rec.emit("ALL_TRAINING_FROZEN_BEFORE_TEST",stage_checkpoints=[p0.name for t,p0 in stage_paths],
            learned_stage_indices=[t for t,p0 in stage_paths])
        test_reports=[]
        for learned_stage,path in stage_paths:
            rec.check();saved=rng_state();began=time.perf_counter()
            frozen_payload=torch.load(path,map_location="cpu",weights_only=False)
            frozen,loaded_optimizer,_=restore_model(frozen_payload,device)
            checkpoint_replay=None
            if replay is not None:
                from context_replay128 import Replay128
                checkpoint_replay=Replay128.from_state_dict(frozen_payload["replay_state"])
            for p0 in frozen.parameters():p0.requires_grad_(False)
            results={};inputtest=[]
            for task in range(learned_stage+1):
                results[task]={}
                for split in SPLITS[2:]:
                    part,receipt=load_split(args,metadata,task,split);inputtest.append(receipt)
                    results[task][split],timing=evaluate(frozen,part,args,device,rec)
                    results[task][split]["measurement_cost"]=timing
            inference=inference_report(frozen,datasets[learned_stage]["valid"],args,device,rec)
            trace_receipt=None
            if is_block(args):
                b,_,_=batch(datasets[learned_stage]["valid"],np.arange(min(8,len(datasets[learned_stage]["valid"]["initial"]))),device)
                with torch.no_grad():
                    _,traces=frozen(b["initial"],b["tokens"],b["lengths"],execution="native",return_traces=True)
                trace_path=rec.out/f"stage{learned_stage}-frozen-native-traces.pt"
                torch.save(dict(traces=traces,external_information="only raw observed input; internal traces generated by model"),trace_path)
                trace_receipt=dict(file=trace_path.name,bytes=trace_path.stat().st_size,sha256=sha256(trace_path))
            row=dict(learned_through_stage=learned_stage,all_seen_test_metrics=results,input_receipts=inputtest,
                frozen_checkpoint=path.name,inference=inference,trace_receipt=trace_receipt,
                resources=resources(frozen,loaded_optimizer,device,checkpoint_replay),post_training_measurement_wall_seconds=time.perf_counter()-began,
                no_test_based_selection=True)
            test_reports.append(row);rec.emit("FROZEN_TEST_RESULT",**row)
            del frozen,loaded_optimizer,frozen_payload,checkpoint_replay;restore_rng(saved)
        rec.json("result.json",dict(status="DECLARED_DEVELOPMENT_COMPLETED",arm=args.arm,seed=args.seed,states=args.states,
            training_reports=reports,frozen_test_reports=test_reports,source_sha256=source_hashes(args.arm,args.block_state,args.replay_capacity),
            resume_lineage=lineage,new_training_updates=sum(report["costs"]["SGD_updates"] for report in reports),
            replay_final_state=replay.statistics() if replay is not None else None,
            formal_Primary=False,accuracy_not_an_implementation_validity_gate=True,
            no_claim_of_independent_novelty_or_Monodromy_benefit=True))
    except BaseException as error:
        rec.emit("run_failed",error=type(error).__name__,reason=str(error),stage=stage,actual_completed_stage_updates=completed,
            last_update_completed=last_safe,completed_training_stages=[r["stage"] for r in reports])
        if last_safe and model is not None and optimizer is not None and stage>=0:
            try:save_checkpoint(model,optimizer,args,metadata,rec,stage,completed,False,receipts,{},suffix="-interrupted",lineage=lineage,replay=replay)
            except BaseException as saving:rec.emit("partial_checkpoint_failed",reason=str(saving))
        rec.json("failure.json",dict(error=type(error).__name__,reason=str(error),stage=stage,
            completed_updates=completed,completed_training_reports=reports,resume_lineage=lineage,not_a_campaign_completion=True))
        raise
    finally:rec.stream.close()


if __name__=="__main__":main()
