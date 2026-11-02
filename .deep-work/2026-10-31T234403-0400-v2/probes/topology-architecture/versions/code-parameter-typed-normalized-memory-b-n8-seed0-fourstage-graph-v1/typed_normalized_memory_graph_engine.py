"""One fixedbucket cached-emission F/B graph; eageroptimizer stays in trainer.

The L16 instance is reused across oldstages0..2. Close it BEFORE token storage
change, then instantiate ONE L48 instance. No constructor/admission/diagnostic
optimizer step, no p.grad replacement aftercapture, no second output decoder.
"""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import typed_normalized_memory_masked_graph_fb as fb
import owned_parent_driver as old

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class GraphEngine:
 def __init__(self,model,optimizer,bucket,first,out,check,stage):
  started=time.perf_counter();cpu=time.process_time();self.model,self.optimizer,self.bucket,self.out,self.check=model,optimizer,bucket,Path(out),check;self.params=list(model.named_parameters());self.names=[n for n,_ in self.params];self.graph=self.side=self.pool=self.outputs=None
  if bucket not in(16,48)or len(self.params)!=20:raise ValueError('fixed typedaction20parameters andL16/48 only')
  self.costs=dict(bucket=bucket,stage_created=stage,capture_count=1,training_FB_calls=0,warmup_FB_calls=0,reference_diagnostic_FB_calls=0,graph_diagnostic_FB_calls=0,capture_description_FB=0,copy_seconds=0.,copy_payload_bytes=0,diagnostic_seconds=0.,diagnostic_CPU_seconds=0.,training_FB_CUDA_seconds=0.,diagnostics=[])
  self.validate(first);self.static={k:torch.empty((64,bucket)if k=='tokens'else(64,),device=first[k].device,dtype=torch.long)for k in('initial','tokens','lengths','targets')};self.first=first;fb.allocate_gradient_buffers(model);self.gradptr={n:p.grad.data_ptr()for n,p in self.params};self.paramptr={n:p.data_ptr()for n,p in self.params};self.copy(first)
  before_rng=old.rng_state();before_NN=self.NN();before_Adam=copy.deepcopy(optimizer.state_dict());was=model.training;model.train()
  try:
   value=self.reference(first);left=self.snapshot(value);del value;self.costs['reference_diagnostic_FB_calls']+=1
   self.side=torch.cuda.Stream();self.side.wait_stream(torch.cuda.current_stream());tick=time.perf_counter();c=time.process_time()
   with torch.cuda.stream(self.side):
    for _ in range(3):check();value=self.masked();del value;self.costs['warmup_FB_calls']+=1
   torch.cuda.current_stream().wait_stream(self.side);self.sync();self.costs['warmup']=dict(wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-c,actual_FB=3,SGD=0)
   self.costs['peak_before_reset']=self.memory();torch.cuda.reset_peak_memory_stats();before=self.memory();self.pool=torch.cuda.graph_pool_handle();self.graph=torch.cuda.CUDAGraph(keep_graph=True);tick=time.perf_counter();c=time.process_time()
   with torch.cuda.graph(self.graph,pool=self.pool,stream=self.side,capture_error_mode='global'):self.outputs=self.masked()
   self.sync();self.costs['capture']=dict(wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-c,memory_before=before,memory_after=self.memory(),private_pool=True,exact_pool_bytes=None,actual_GPU_FB_during_capture=0);self.costs['capture_description_FB']=1
   tick=time.perf_counter();c=time.process_time();self.graph.instantiate();self.sync();self.costs['capture'].update(instantiate_seconds=time.perf_counter()-tick,instantiate_CPU_seconds=time.process_time()-c)
   self.stable();self.graph.replay();self.sync();right=self.snapshot(self.outputs);self.costs['graph_diagnostic_FB_calls']+=1;self.costs['admission']=self.compare(left,right,f'stage{stage}-L{bucket}-admission.npz')
   if not old.nested_state_equal(before_rng,old.rng_state())or not old.tensor_state_equal(self.model.state_dict(),before_NN)or not old.nested_state_equal(self.optimizer.state_dict(),before_Adam):raise RuntimeError('0SGDcapture/admission changedNN/Adam/RNG')
   self.costs.update(setup_seconds=time.perf_counter()-started,setup_CPU_seconds=time.process_time()-cpu,NN_Adam_RNG_neutral=True,grad_payload_bytes=sum(p.grad.numel()*p.grad.element_size()for _,p in self.params),static_input_bytes=sum(v.numel()*v.element_size()for v in self.static.values()),all20_reference_grads_present=True)
  except BaseException:self.close();raise
  finally:model.train(was)
 def sync(self):torch.cuda.synchronize()
 def NN(self):return{n:v.detach().cpu().clone()for n,v in self.model.state_dict().items()}
 def memory(self):return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
 def stable(self):
  if self.gradptr!={n:p.grad.data_ptr()if p.grad is not None else None for n,p in self.params}or self.paramptr!={n:p.data_ptr()for n,p in self.params}:raise RuntimeError('graphNN/gradstorage changed')
 def validate(self,b):
  if set(b)!={'initial','tokens','lengths','targets'}or b['tokens'].shape[0]!=64 or b['tokens'].shape[1]>self.bucket or any(v.dtype!=torch.long or v.device.type!='cuda'for v in b.values())or bool((b['lengths']>self.bucket).any()):raise ValueError('exact64row activebatch withinbucket required')
 def copy(self,b):
  tick=time.perf_counter();self.validate(b);self.static['tokens'].zero_();self.static['tokens'][:,:b['tokens'].shape[1]].copy_(b['tokens'])
  for k in('initial','lengths','targets'):self.static[k].copy_(b[k])
  self.sync();self.stable();self.costs['copy_seconds']+=time.perf_counter()-tick;self.costs['copy_payload_bytes']+=sum(v.numel()*v.element_size()for v in b.values())
 def masked(self):return fb.masked_step(self.model,self.static['initial'],self.static['tokens'],self.static['lengths'],self.static['targets'])
 def reference(self,b):
  for _,p in self.params:p.grad.zero_()
  loss,out=fb.objective(self.model,fb.original_forward_value,b['initial'],b['tokens'],b['lengths'],b['targets']);grads=torch.autograd.grad(loss,tuple(p for _,p in self.params),allow_unused=True)
  if any(g is None for g in grads):raise RuntimeError('original reference has unusedparameter; do not hideNone semantics')
  with torch.no_grad():
   for (_,p),g in zip(self.params,grads):p.grad.copy_(g)
  self.stable();return out.detach(),loss.detach()
 def snapshot(self,value):
  self.sync();return dict(output=value[0].cpu().clone(),loss=value[1].cpu().clone(),grads={n:p.grad.detach().cpu().clone()for n,p in self.params})
 def compare(self,left,right,leaf):
  raw=dict(reference_output=left['output'].numpy(),candidate_output=right['output'].numpy(),reference_loss=left['loss'].numpy(),candidate_loss=right['loss'].numpy(),gradient_names_json=np.frombuffer(json.dumps(self.names).encode(),dtype=np.uint8));row=dict(output_max_abs=float((left['output']-right['output']).abs().max()),gradients={})
  for i,n in enumerate(self.names):raw[f'reference_grad_{i:02d}']=left['grads'][n].numpy();raw[f'candidate_grad_{i:02d}']=right['grads'][n].numpy()
  path=self.out/leaf
  with path.open('xb')as f:np.savez_compressed(f,**raw)
  row.update(raw_file=leaf,raw_sha256=sha(path),raw_bytes=path.stat().st_size)
  if not torch.allclose(left['output'],right['output'],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL)or not torch.allclose(left['loss'],right['loss'],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL):raise RuntimeError('cachedemission output/loss contract failed')
  for n in self.names:
   l,r=left['grads'][n],right['grads'][n];close=bool(torch.allclose(l,r,atol=fb.GRAD_ATOL,rtol=fb.GRAD_RTOL));row['gradients'][n]=dict(max_abs=float((l-r).abs().max()),allclose=close)
   if not close or not bool(torch.isfinite(r).all()):raise RuntimeError('cachedemission all20gradient contract failed: '+n)
  return row
 def diagnostic(self,stage,step,first=None):
  if first is not None:self.first=first
  self.check();tick=time.perf_counter();cpu=time.process_time();rng=old.rng_state();NN=self.NN();Adam=copy.deepcopy(self.optimizer.state_dict());was=self.model.training;self.model.train();self.copy(self.first);v=self.reference(self.first);left=self.snapshot(v);del v;self.graph.replay();self.sync();right=self.snapshot(self.outputs);self.stable();row=self.compare(left,right,f'stage{stage}-step{step:03d}-L{self.bucket}-all20.npz')
  if not old.nested_state_equal(rng,old.rng_state())or not old.tensor_state_equal(NN,self.model.state_dict())or not old.nested_state_equal(Adam,self.optimizer.state_dict()):raise RuntimeError('0SGDdiagnostic changedNN/Adam/RNG')
  self.model.train(was);row.update(stage=stage,step=step,wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-cpu,SGD=0);self.costs['reference_diagnostic_FB_calls']+=1;self.costs['graph_diagnostic_FB_calls']+=1;self.costs['diagnostic_seconds']+=row['wall_seconds'];self.costs['diagnostic_CPU_seconds']+=row['CPU_seconds'];self.costs['diagnostics'].append(row);return row
 def run(self,b):
  self.copy(b);self.check();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();self.graph.replay();end.record();self.sync();self.stable();self.costs['training_FB_calls']+=1;self.costs['training_FB_CUDA_seconds']+=start.elapsed_time(end)/1000.;return self.outputs
 def report(self):
  r=copy.deepcopy(self.costs);r.update(graph_count=1,emission='cachedlogpONLY; nosecondreadout',full_training_FLOPs=None,provider_effective_compute=None,scope='copy nestedintrains; all20fixedfirstTRAINcurrentNNgates; captureprivatepool/workspace onlyallocator snapshots, notexactpoolbytes')
  return r
 def close(self):
  if self.graph is not None:self.sync();self.graph.reset();self.graph=None
  self.outputs=self.pool=self.side=None
