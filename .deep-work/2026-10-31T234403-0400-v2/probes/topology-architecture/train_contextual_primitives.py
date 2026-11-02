"""Finite contextual-primitive development with actual terminal-only training.

Same Prometheus V2. This is600-update development, NOT frozen Primary evidence.
No import-time run, network, SSH, subprocess or dependency installation. Root
supplies the existing external CUDA isolation and process watchdog.

Dynamic is an ordinary contextual-calling mechanism; this comparison does not
establish independent novelty or Monodromy benefit. Static controls use the
same Block/slot/K/dimensions atD=N or preallocated2N. GRU and Transformer receive
the identical raw inputs, with NO extra stage/taskID/teacher operation/route.
All data accesses and all failed outcomes remain recorded. Validity is based
on protocol and implementation, never on an accuracy acceptance gate.
"""
from __future__ import annotations

import argparse
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

from contextual_primitive_block import ContextualPrimitiveBlock


ARMS=("dynamic","static","static_expand","gru","transformer","ssm")
SPLITS=("train","valid","test_iid","test_long","test_unseen")
KEYS=("initial","tokens","lengths","targets")
MAX_FILE=100*1024*1024


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
    """Ordinary two-layer64-wide GRU, random learned input/output interface."""
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
            active_steps=active,architecture="ordinary2layer64GRU; no taskID; no supplied state transition")


class TerminalTransformer(nn.Module):
    """Standard64-wide,3-layer,4-head attention; fixed sinusoidal positions.

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
            active_steps=active,architecture="ordinary3layer4head64Transformer, sinusoidal positions, prefixCLS, no taskID")


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
    if args.arm=="gru":return TerminalGRU(**common).to(device)
    if args.arm=="transformer":return TerminalTransformer(**common).to(device)
    if args.arm=="ssm":
        from selective_ssm_baseline import TerminalSelectiveSSM
        return TerminalSelectiveSSM(**common).to(device)
    return ContextualPrimitiveBlock(n_states=args.states,n_tokens=metadata["n_tokens"],
        internal_states=args.states*(2 if args.arm=="static_expand" else 1),n_primitives=args.primitives,
        token_dim=16,state_dim=16,memory_dim=16,caller_hidden=32,n_slots=args.slots,
        routing="dynamic" if args.arm=="dynamic" else "static",initialization=args.block_initialization,
        straight_through_backward=args.straight_through_backward).to(device)


def is_block(args):return args.arm in ("dynamic","static","static_expand")


def forward(model,b,args,execution):
    if is_block(args):return model(b["initial"],b["tokens"],b["lengths"],execution=execution,output="logits")
    return model(b["initial"],b["tokens"],b["lengths"])


def loss_value(logits,targets,args,reduction="mean"):
    return (F.nll_loss(logits,targets,reduction=reduction) if is_block(args)
            else F.cross_entropy(logits,targets,reduction=reduction))


def work_proxy(model,args,b,l,active,execution):
    return (model.resource_report(batch_size=b,sequence_length=l,active_steps=active,
                execution=execution,return_traces=False) if is_block(args) else model.proxy(b,l,active))


def resources(model,optimizer,device):
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
        parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        registered_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
        optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),replay_bytes=0,
        cuda_allocated_bytes=torch.cuda.memory_allocated(device),cuda_reserved_bytes=torch.cuda.memory_reserved(device),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        cpu_peak_RSS_bytes=int(rss if sys.platform=="darwin" else rss*1024),
        memory_scope="actual process/allocator counters; peak is cumulative and includes audits; no energy claim")


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


def save_checkpoint(model,optimizer,args,metadata,rec,stage,step,complete,inputs,costs,suffix=""):
    path=rec.out/f"{args.arm}-stage{stage}-step{step}{suffix}.pt"
    if path.exists():raise FileExistsError("checkpoint evidence is append-only")
    began=time.perf_counter()
    payload=dict(schema="contextual-primitives-training-v1",args=vars(args),metadata_public=dict(
        n_states=metadata["n_states"],n_tokens=metadata["n_tokens"],max_length=metadata["max_length"]),
        stage=stage,step=step,stage_complete=complete,model_state=model.state_dict(),
        block_checkpoint=model.checkpoint() if is_block(args) else None,
        baseline_spec=None if is_block(args) else model.spec,optimizer=optimizer.state_dict(),
        parameter_flags={n:p.requires_grad for n,p in model.named_parameters()},rng=rng_state(),
        input_receipts=inputs,costs=costs,source_sha256=source_hashes(args.arm),
        claim_scope="finite architecture development; not frozen Primary, novelty or Monodromy-benefit evidence")
    pending=path.with_suffix(".pt.partial");torch.save(payload,pending)
    if pending.stat().st_size>MAX_FILE:raise RuntimeError("checkpoint100MiB bound exceeded; partial retained")
    pending.rename(path)
    receipt=dict(file=path.name,sha256=sha256(path),bytes=path.stat().st_size,wall_seconds=time.perf_counter()-began)
    rec.emit("actual_neural_checkpoint",stage=stage,step=step,complete=complete,**receipt)
    return path,receipt


def restore_model(payload,device):
    args=argparse.Namespace(**payload["args"])
    if is_block(args):model=ContextualPrimitiveBlock.from_checkpoint(payload["block_checkpoint"]).to(device)
    elif args.arm=="gru":model=TerminalGRU(**payload["baseline_spec"]).to(device)
    elif args.arm=="ssm":
        from selective_ssm_baseline import TerminalSelectiveSSM
        model=TerminalSelectiveSSM(**payload["baseline_spec"]).to(device)
    else:model=TerminalTransformer(**payload["baseline_spec"]).to(device)
    model.load_state_dict(payload["model_state"],strict=True)
    for n,p in model.named_parameters():p.requires_grad_(payload["parameter_flags"][n])
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr);optimizer.load_state_dict(payload["optimizer"])
    return model,optimizer,args


def reload_audit(path,model,optimizer,args,valid,device,rec):
    saved=rng_state();was=model.training;synchronize(device);began=time.perf_counter()
    other=other_optimizer=None
    try:
        payload=torch.load(path,map_location="cpu",weights_only=False)
        other,other_optimizer,_=restore_model(payload,device)
        if not tensor_state_equal(payload["model_state"],other.state_dict()):raise RuntimeError("saved actual neural tensors differ after reload")
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


def source_hashes(arm=None):
    folder=Path(__file__).resolve().parent
    names=[Path(__file__).name,"contextual_primitive_block.py"]
    if arm=="ssm":names.append("selective_ssm_baseline.py")
    return {n:sha256(folder/n) for n in names}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    p.add_argument("--arm",choices=ARMS,required=True);p.add_argument("--states",type=int,default=8)
    p.add_argument("--seed",type=int,default=0);p.add_argument("--stages",type=int,choices=(1,3),default=1)
    p.add_argument("--steps",type=int,default=600);p.add_argument("--primitives",type=int,default=4)
    p.add_argument("--slots",type=int,default=4);p.add_argument("--lr",type=float,default=.003)
    p.add_argument("--block-initialization",choices=("iid","carry"),default="iid")
    p.add_argument("--block-training",choices=("scheduled","soft_only"),default="scheduled")
    p.add_argument("--straight-through-backward",choices=("selected","dense"),default="selected")
    p.add_argument("--device",choices=("cuda:0",),default="cuda:0")
    p.add_argument("--max-seconds",type=float,default=850.)
    p.add_argument("--resource-overlap-note",default="coordinator must record actual concurrent jobs; no isolated-throughput claim")
    args=p.parse_args()
    if (args.out.exists() or not 1<=args.steps<=600 or not 1<=args.states<=32
            or not 1<=args.primitives<=16 or not 1<=args.slots<=8
            or not 0<args.lr<=.05 or not 0<args.max_seconds<=850):
        p.error("new output, finite states/K/slots/600-update/850s bounds required")
    rec=Recorder(args.out,args.max_seconds);device=torch.device(args.device)
    model=optimizer=None;stage=-1;completed=0;last_safe=False;reports=[];receipts=[];stage_paths=[]
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
        model=build(args,metadata,device);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
        rec.emit("configuration",args=vars(args),public_metadata=dict(n_states=metadata["n_states"],n_tokens=metadata["n_tokens"],
            max_length=metadata["max_length"],stages_requested=args.stages),metadata_sha256=sha256(metadata_path),
            inputs=receipts,source_sha256=source_hashes(args.arm),torch=torch.__version__,CUDA=torch.version.cuda,
            GPU=torch.cuda.get_device_name(device),parameters=resources(model,optimizer,device),
            policy="current-stage64-row batches only; same seed[seed,N,stage,9027]; no replay/taskID/teacher routes",
            block_schedule=("all updates soft" if args.block_training=="soft_only" else
                "per-stage steps1..200 soft;201..600 hard-ST calls/op rows with SOFT E/R; native evaluation hardens E/R"),
            straight_through_backward=args.straight_through_backward,
            standard_baselines="two-layerGRU64;3layer4headTransformer64,sinusoidal positions,no extra stage embedding",
            ssm_scope="source-grounded selective-SSM reference adaptation; NOT official Mamba/Mamba2 or fused-kernel cost reproduction",
            control_scope="ordinary dynamic routing is a same-mechanism reference; static differs ONLY caller inputs;2N is preallocated and counted",
            validity="implementation/protocol/reproduction, never final accuracy",formal_Primary=False,
            test_policy="test files opened only after ALL declared training finishes and all stage checkpoints are frozen",
            exact_training_FLOPs=None,FLOP_boundary="forward MAC proxies plus actual synchronized timings; no full-autograd FLOP claim")
        for stage in range(args.stages):
            rec.check();stage_begin,stage_cpu=time.perf_counter(),time.process_time();completed=0
            torch.cuda.reset_peak_memory_stats(device)
            data=datasets[stage]["train"]
            indices=np.random.default_rng(np.random.SeedSequence([args.seed,args.states,stage,9027])).integers(
                0,len(data["initial"]),size=(args.steps,64),dtype=np.int64)
            ipath=rec.out/f"stage{stage}-common-indices.npz";np.savez_compressed(ipath,indices=indices)
            costs=dict(training_wall_seconds=0.,training_CPU_seconds=0.,CUDA_stream_elapsed_seconds=0.,
                batch_assembly_wall_seconds=0.,evaluation_wall_seconds=0.,evaluation_CPU_seconds=0.,
                training_forward_MAC_proxy=0,evaluation_forward_MAC_proxy=0,training_examples=0,
                evaluation_forward_examples=0,SGD_updates=0,old_examples_used_for_training=0)
            curves=[]
            rec.emit("stage_begin",stage=stage,indices_sha256=sha256(ipath),optimizer_retained_across_stages=True,
                parameters=resources(model,optimizer,device),phase_schedule_fixed_before_test=True)
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
                    row=dict(stage=stage,step=step,metrics=allmetrics,resources=resources(model,optimizer,device),costs=copy.deepcopy(costs))
                    curves.append(row);rec.emit("TRAIN_VALID_curve",**row)
                    if step and step!=args.steps:
                        save_checkpoint(model,optimizer,args,metadata,rec,stage,step,False,receipts,costs)
                if step==args.steps:break
                began=time.perf_counter();b,l,active=batch(data,indices[step],device)
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
                rec.emit("actual_update",stage=stage,step=step+1,execution=execution if is_block(args) else "standard",
                    terminal_loss=float(loss.detach()),batch_accuracy=float((output.argmax(-1)==b["targets"]).float().mean()),
                    preclip_gradient=float(norm),selected_gradient_norms=gradients,
                    training_wall_seconds=elapsed,training_CPU_seconds=cpu_elapsed,CUDA_stream_elapsed_seconds=gpu_elapsed,
                    forward_MAC_proxy=proxy["proxy_forward_macs"],active_token_steps=active,padded_length_passed=l)
            path,cp=save_checkpoint(model,optimizer,args,metadata,rec,stage,args.steps,True,receipts,costs)
            audit=reload_audit(path,model,optimizer,args,datasets[stage]["valid"],device,rec)
            summary=dict(stage=stage,final_TRAIN_VALID=curves[-1]["metrics"],curves=curves,costs=costs,
                resources=resources(model,optimizer,device),checkpoint=cp,reload=audit,
                stage_wall_seconds=time.perf_counter()-stage_begin,stage_CPU_seconds=time.process_time()-stage_cpu)
            reports.append(summary);stage_paths.append(path);rec.emit("stage_training_complete",**summary)
            rec.json(f"completed-training-through-stage{stage}.json",dict(reports=reports,test_data_read=False))
        # No more training, candidate selection, configuration changes or retries.
        # Retrospective tests of each already frozen stage preserve a genuine
        # continual test curve without exposing any result to later training.
        for p0 in model.parameters():p0.requires_grad_(False)
        rec.emit("ALL_TRAINING_FROZEN_BEFORE_TEST",stage_checkpoints=[p0.name for p0 in stage_paths])
        test_reports=[]
        for learned_stage,path in enumerate(stage_paths):
            rec.check();saved=rng_state();began=time.perf_counter()
            frozen,loaded_optimizer,_=restore_model(torch.load(path,map_location="cpu",weights_only=False),device)
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
                resources=resources(frozen,loaded_optimizer,device),post_training_measurement_wall_seconds=time.perf_counter()-began,
                no_test_based_selection=True)
            test_reports.append(row);rec.emit("FROZEN_TEST_RESULT",**row)
            del frozen,loaded_optimizer;restore_rng(saved)
        rec.json("result.json",dict(status="DECLARED_DEVELOPMENT_COMPLETED",arm=args.arm,seed=args.seed,states=args.states,
            training_reports=reports,frozen_test_reports=test_reports,source_sha256=source_hashes(args.arm),
            formal_Primary=False,accuracy_not_an_implementation_validity_gate=True,
            no_claim_of_independent_novelty_or_Monodromy_benefit=True))
    except BaseException as error:
        rec.emit("run_failed",error=type(error).__name__,reason=str(error),stage=stage,actual_completed_stage_updates=completed,
            last_update_completed=last_safe,completed_training_stages=[r["stage"] for r in reports])
        if last_safe and model is not None and optimizer is not None and stage>=0:
            try:save_checkpoint(model,optimizer,args,metadata,rec,stage,completed,False,receipts,{},suffix="-interrupted")
            except BaseException as saving:rec.emit("partial_checkpoint_failed",reason=str(saving))
        rec.json("failure.json",dict(error=type(error).__name__,reason=str(error),stage=stage,
            completed_updates=completed,completed_training_reports=reports,not_a_campaign_completion=True))
        raise
    finally:rec.stream.close()


if __name__=="__main__":main()
