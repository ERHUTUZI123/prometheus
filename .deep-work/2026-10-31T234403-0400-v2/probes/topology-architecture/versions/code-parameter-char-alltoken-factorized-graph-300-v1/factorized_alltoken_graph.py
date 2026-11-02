"""Fixed B32L64 fcba AR training F/B graph; no optimizer or architecture change.

All32 rows and64 positions active; only the training path removes original validation
and gather/index_copy. Direct exact _transition and every post-token readout.
Original wrapper remains the eval/reload/reference path. No binding/teacher loss.
"""
import copy,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
B,L,V=32,64,100
OUTPUT_ATOL,OUTPUT_RTOL=2e-5,2e-5
GRAD_ATOL,GRAD_RTOL=2e-5,2e-4

def pure_forward(model,initial,tokens):
 carry=model.body.initial_carry(initial);embedded=model.body.token_embedding(tokens);history=[]
 for step in range(64):
  carry,_=model.body._transition(carry,embedded[:,step],'soft')
  history.append(F.log_softmax(model.body.readout(carry['h']),-1))
 return torch.stack(history,1)

def same(a,b):
 if torch.is_tensor(a)or torch.is_tensor(b):return torch.is_tensor(a)and torch.is_tensor(b)and torch.equal(a.detach().cpu(),b.detach().cpu())
 if isinstance(a,np.ndarray)or isinstance(b,np.ndarray):return isinstance(a,np.ndarray)and isinstance(b,np.ndarray)and np.array_equal(a,b)
 if isinstance(a,dict)or isinstance(b,dict):return isinstance(a,dict)and isinstance(b,dict)and a.keys()==b.keys()and all(same(a[k],b[k])for k in a)
 if isinstance(a,(tuple,list))or isinstance(b,(tuple,list)):return type(a)==type(b)and len(a)==len(b)and all(same(x,y)for x,y in zip(a,b))
 return a==b

def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class GraphARTrainer:
 def __init__(self,model,optimizer,first,out,check):
  started=time.perf_counter();cpu=time.process_time()
  self.model,self.optimizer,self.out,self.check=model,optimizer,Path(out),check;self.params=list(model.named_parameters());self.names=[n for n,_ in self.params];self.graph=None;self.pool=None;self.side=None;self.outputs=None
  self.costs=dict(setup_wall_seconds=0.,setup_CPU_seconds=0.,actual_training_FB_calls=0,warmup_FB_calls=0,reference_diagnostic_FB_calls=0,graph_diagnostic_FB_calls=0,capture_described_FB_calls=0,diagnostic_wall_seconds=0.,diagnostic_CPU_seconds=0.,copy_wall_seconds=0.,copy_payload_bytes=0,training_CUDA_FB_seconds=0.,diagnostics=[])
  if len(self.params)!=27 or sum(p.numel()for _,p in self.params)!=20721 or any(p.dtype!=torch.float32 for _,p in self.params):raise ValueError('exact20721P/all27 FP32 model required')
  self.validate(first);self.first={k:v.clone()for k,v in first.items()};self.static={k:v.clone()for k,v in first.items()}
  for _,p in self.params:p.grad=torch.zeros_like(p,memory_format=torch.preserve_format)
  self.gradptr={n:p.grad.data_ptr()for n,p in self.params};self.paramptr={n:p.data_ptr()for n,p in self.params};saved_mode=model.training;model.train();before_rng=rng();before_NN=self.NN();before_Adam=copy.deepcopy(optimizer.state_dict())
  try:
   value=self.reference(self.first);gold=self.snapshot(value);del value;self.costs['reference_diagnostic_FB_calls']+=1
   self.side=torch.cuda.Stream();self.side.wait_stream(torch.cuda.current_stream());tick=time.perf_counter();c=time.process_time()
   with torch.cuda.stream(self.side):
    for _ in range(3):check();value=self.masked_step();del value;self.costs['warmup_FB_calls']+=1
   torch.cuda.current_stream().wait_stream(self.side);self.sync();self.costs['warmup']=dict(wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-c,SGD_updates=0,actual_FB_calls=3)
   self.costs['memory_before_peak_reset']=self.memory();torch.cuda.reset_peak_memory_stats();before=self.memory();self.pool=torch.cuda.graph_pool_handle();self.graph=torch.cuda.CUDAGraph(keep_graph=True);tick=time.perf_counter();c=time.process_time()
   with torch.cuda.graph(self.graph,pool=self.pool,stream=self.side,capture_error_mode='global'):self.outputs=self.masked_step()
   self.sync();self.costs['capture']=dict(wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-c,memory_before=before,memory_after=self.memory(),private_pool=True,pool_exact_bytes=None,actual_GPU_FB_during_capture=0);self.costs['capture_described_FB_calls']=1
   tick=time.perf_counter();c=time.process_time();self.graph.instantiate();self.sync();self.costs['capture'].update(instantiate_wall_seconds=time.perf_counter()-tick,instantiate_CPU_seconds=time.process_time()-c)
   self.stable();self.graph.replay();self.sync();candidate=self.snapshot(self.outputs);self.costs['graph_diagnostic_FB_calls']+=1;self.costs['admission']=self.compare(gold,candidate,'graph-admission-all27.npz')
   if not same(before_rng,rng())or not same(before_NN,self.NN())or not same(before_Adam,optimizer.state_dict()):raise RuntimeError('0SGD graphsetup changed NN/Adam/RNG')
   self.costs.update(setup_wall_seconds=time.perf_counter()-started,setup_CPU_seconds=time.process_time()-cpu,setup_NN_Adam_RNG_unchanged=True,NN_bytes=sum(p.numel()*p.element_size()for _,p in self.params),gradient_bytes=sum(p.grad.numel()*p.grad.element_size()for _,p in self.params),static_input_bytes=sum(v.numel()*v.element_size()for v in self.static.values()),fixed_diagnostic_input_bytes=sum(v.numel()*v.element_size()for v in self.first.values()),graph_zero_grad_inside=True,optimizer_inside_graph=False,all27_original_grads_present=True)
  except BaseException:self.close();raise
  finally:model.train(saved_mode)
 def sync(self):torch.cuda.synchronize()
 def memory(self):return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
 def NN(self):return{n:v.detach().cpu().clone()for n,v in self.model.state_dict().items()}
 def validate(self,b):
  if set(b)!={'initial','tokens','lengths','targets'}or b['tokens'].shape!=(B,L)or b['targets'].shape!=(B,L)or b['initial'].shape!=(B,)or b['lengths'].shape!=(B,)or any(v.dtype!=torch.long or v.device.type!='cuda'for v in b.values())or bool(b['initial'].any())or not bool((b['lengths']==L).all()):raise ValueError('fixed full32x64/publicinitial0 CUDA long batch required')
 def stable(self):
  if self.gradptr!={n:p.grad.data_ptr()if p.grad is not None else None for n,p in self.params}or self.paramptr!={n:p.data_ptr()for n,p in self.params}:raise RuntimeError('captured parameter/gradient storage changed')
 def zero(self):
  for _,p in self.params:p.grad.zero_()
 def masked_step(self):
  self.zero();out=pure_forward(self.model,self.static['initial'],self.static['tokens']);loss=F.nll_loss(out.reshape(-1,V),self.static['targets'].reshape(-1));loss.backward();return out.detach(),loss.detach()
 def reference(self,b):
  self.zero();out=self.model(b['initial'],b['tokens'],b['lengths'],'soft');loss=F.nll_loss(out.reshape(-1,V),b['targets'].reshape(-1));grads=torch.autograd.grad(loss,tuple(p for _,p in self.params),allow_unused=True)
  if any(g is None for g in grads):raise RuntimeError('actual original TRAIN batch has unused parameter; cannot hide None-vs-zero Adam semantics')
  with torch.no_grad():
   for (_,p),g in zip(self.params,grads):p.grad.copy_(g)
  self.stable();return out.detach(),loss.detach()
 def snapshot(self,value):
  self.sync();return dict(output=value[0].cpu().clone(),loss=value[1].cpu().clone(),grads={n:p.grad.detach().cpu().clone()for n,p in self.params})
 def compare(self,left,right,leaf):
  values=dict(reference_output=left['output'].numpy(),candidate_output=right['output'].numpy(),reference_loss=left['loss'].numpy(),candidate_loss=right['loss'].numpy(),gradient_names_json=np.frombuffer(json.dumps(self.names).encode(),dtype=np.uint8));row=dict(raw_file=leaf,output_max_abs=float((left['output']-right['output']).abs().max()),gradients={})
  for i,n in enumerate(self.names):values[f'reference_grad_{i:02d}']=left['grads'][n].numpy();values[f'candidate_grad_{i:02d}']=right['grads'][n].numpy()
  path=self.out/leaf
  with path.open('xb')as f:np.savez_compressed(f,**values)
  row.update(raw_sha256=sha(path),raw_bytes=path.stat().st_size)
  if not torch.allclose(left['output'],right['output'],atol=OUTPUT_ATOL,rtol=OUTPUT_RTOL)or not torch.allclose(left['loss'],right['loss'],atol=OUTPUT_ATOL,rtol=OUTPUT_RTOL):raise RuntimeError('prospective AR output/loss numeric contract failed')
  for n in self.names:
   l,r=left['grads'][n],right['grads'][n];close=bool(torch.allclose(l,r,atol=GRAD_ATOL,rtol=GRAD_RTOL));row['gradients'][n]=dict(max_abs=float((l-r).abs().max()),allclose=close)
   if not close or not bool(torch.isfinite(r).all()):raise RuntimeError('prospective AR all27-gradient numeric contract failed: '+n)
  return row
 def copy(self,b):
  tick=time.perf_counter();self.validate(b)
  for k,v in b.items():self.static[k].copy_(v)
  self.sync();self.costs['copy_wall_seconds']+=time.perf_counter()-tick;self.costs['copy_payload_bytes']+=sum(v.numel()*v.element_size()for v in b.values());self.stable()
 def run(self,b):
  self.copy(b);self.check();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();self.graph.replay();end.record();self.sync();self.stable();self.costs['training_CUDA_FB_seconds']+=start.elapsed_time(end)/1000.;self.costs['actual_training_FB_calls']+=1;return self.outputs
 def diagnostic(self,step):
  self.check();tick=time.perf_counter();cpu=time.process_time();before_rng=rng();NN=self.NN();Adam=copy.deepcopy(self.optimizer.state_dict());saved=self.model.training;self.model.train();self.copy(self.first);value=self.reference(self.first);left=self.snapshot(value);del value;self.graph.replay();self.sync();right=self.snapshot(self.outputs);row=self.compare(left,right,f'graph-step{step:03d}-all27.npz');self.stable()
  if not same(before_rng,rng())or not same(NN,self.NN())or not same(Adam,self.optimizer.state_dict()):raise RuntimeError('0SGD currentNN diagnostic changed NN/Adam/RNG')
  self.model.train(saved);row.update(step=step,wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-cpu,SGD_updates=0);self.costs['reference_diagnostic_FB_calls']+=1;self.costs['graph_diagnostic_FB_calls']+=1;self.costs['diagnostic_wall_seconds']+=row['wall_seconds'];self.costs['diagnostic_CPU_seconds']+=row['CPU_seconds'];self.costs['diagnostics'].append(row);return row
 def report(self):
  p=self.model.proxy(B,L);result=copy.deepcopy(self.costs);result.update(training_forward_MAC_proxy=result['actual_training_FB_calls']*p,warmup_forward_MAC_proxy=result['warmup_FB_calls']*p,reference_diagnostic_forward_MAC_proxy=result['reference_diagnostic_FB_calls']*p,graph_diagnostic_forward_MAC_proxy=result['graph_diagnostic_FB_calls']*p,extra_diagnostic_and_warmup_label_positions=(result['warmup_FB_calls']+result['reference_diagnostic_FB_calls']+result['graph_diagnostic_FB_calls'])*B*L,graph_count=1,padded_steps_equal_original=True,zeroing_scalar_writes_per_FB=20721,full_training_FLOPs=None,provider_effective_compute=None,scope='same alltoken math/currentNN gradient gates; different FP32 trajectories allowed; puretraining graph, eagerclip+Adam; allwarmup/capture/copy/privatepool/diagnostic costs paid; no quality or speed gate')
  return result
 def close(self):
  if self.graph is not None:self.sync();self.graph.reset();self.graph=None
  self.outputs=self.pool=self.side=None
