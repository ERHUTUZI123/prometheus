"""NO-RUN fixed-weight CPU/CUDA output+all-gradient/cost contract.

No optimizer is constructed or stepped. Raw gradients are BEFORE clipping.
Original gather uses zero_grad(None), masked uses zero_ with stable buffers;
Adam and clipping belong outside a future replay, not this diagnostic. CPU mode
checks values only; CUDA mode times three sequential backends to avoid changing
captured grad pointers while running the original eager zero_grad(None) policy.
"""
import argparse,hashlib,importlib.util,json,time,traceback,gc,resource,sys
from pathlib import Path

CORE_SHA='fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158'
CHECKPOINT_SHA='4a9e3c88134d7028197459bfb76305ba282cf5328d3b519d0794cfb15e8fd35a'
CHECKPOINT_BYTES=332403
WARMUPS,REPEATS=3,12

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise RuntimeError('owned source unavailable')
 module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in('core','checkpoint','cases','out'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--cases-sha256',required=True);p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu');p.add_argument('--max-seconds',type=float,default=90)
 a=p.parse_args();limit=90 if a.device=='cpu'else 850
 if not 1<=a.max_seconds<=limit or a.out.exists():raise ValueError('new output and bounded CPU90/GPU850 contract required')
 a.out.mkdir();started=time.perf_counter();cpu_started=time.process_time();deadline=started+a.max_seconds
 result=dict(status='RUNNING_FIXED_FACTORIZED_FB_CONTRACT',device=a.device,SGD_updates=0,optimizer_steps=0,clipping_calls=0,cases=[],actual_FB_calls={})
 phase='import'
 try:
  import torch,numpy as np
  import factorized_masked_graph_fb as fb
  torch.set_num_threads(1);device=torch.device(a.device)
  if a.device!='cpu'and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback')
  if a.core.is_symlink()or sha(a.core)!=CORE_SHA:raise ValueError('exact current fcba source required')
  if a.checkpoint.is_symlink()or a.checkpoint.stat().st_size!=CHECKPOINT_BYTES or sha(a.checkpoint)!=CHECKPOINT_SHA:raise ValueError('exact current anchor0 frozen weights required')
  if sha(a.cases)!=a.cases_sha256:raise ValueError('prospectively frozen TRAIN batch artifact changed')
  core=load_module(a.core,'_owned_fcba_graph_core');payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
  if payload.get('stage')!=3 or payload.get('step')!=600 or payload.get('stage_complete')is not True:raise ValueError('completed current stage3 required')
  anchor=payload['args']['anchor']
  if anchor!=0:raise ValueError('fixed anchor0 lineage only; no score selection')
  model=core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint']).to(device);model.train()
  for name,value in model.state_dict().items():
   if not torch.equal(value.detach().cpu(),payload['model_state'][name]):raise RuntimeError('checkpoint NN values differ')
  if model.parameter_count()!=13397 or any(p.dtype!=torch.float32 for p in model.parameters()):raise ValueError('exact13397P FP32 model required')
  parameters=list(model.named_parameters());before={name:value.detach().cpu().clone()for name,value in model.state_dict().items()}
  def tensor_bytes(v):
   if torch.is_tensor(v):return v.numel()*v.element_size()
   if isinstance(v,dict):return sum(tensor_bytes(x)for x in v.values())
   if isinstance(v,(tuple,list)):return sum(tensor_bytes(x)for x in v)
   return 0
  result.update(source_sha256=dict(core=CORE_SHA,checkpoint=CHECKPOINT_SHA,cases=a.cases_sha256,helper=sha(Path(fb.__file__))),P=13397,NN_bytes=53588,gradient_buffer_bytes=53588,
   saved_Adam_tensor_bytes=tensor_bytes(payload['optimizer']),live_Adam_tensor_bytes=0,anchor=anchor,torch_version=str(torch.__version__),
   CPU_contract_is_not_CUDA_result=True,provider_effective_compute=None,full_training_FLOPs=None,
   limits='F/B only; no clipping/Adam step cost or measured full-training speedup; all three backends/capture work paid')
  del payload
  with np.load(a.cases,allow_pickle=False)as stored:
   meta=json.loads(stored['metadata_json'].tobytes().decode());batches={bucket:{key:stored[f'b{bucket}_{key}']for key in('initial','tokens','lengths','targets')}for bucket in fb.BUCKETS}
  if meta.get('schema')!='factorized-fixed-graph-FB-cases-v1'or meta.get('TRAIN_only')is not True:raise ValueError('fixed TRAIN-only batches required')
  result['batch_provenance']=meta
  def check():
   if time.perf_counter()>deadline:raise TimeoutError('fixed F/B diagnostic bound')
  def sync():
   if device.type=='cuda':torch.cuda.synchronize(device)
  def charge(label):result['actual_FB_calls'][label]=result['actual_FB_calls'].get(label,0)+1
  def clear():
   model.zero_grad(set_to_none=True);gc.collect()
   if device.type=='cuda':torch.cuda.empty_cache();torch.cuda.synchronize(device)
  def snapshot(outputs):
   sync();return dict(output=outputs[0].detach().cpu().clone(),loss=outputs[1].detach().cpu().clone(),nll=outputs[2].detach().cpu().clone(),binding=outputs[3].detach().cpu().clone(),grads={name:None if value.grad is None else value.grad.detach().cpu().clone()for name,value in parameters})
  def save_pair(name,left,right):
   values=dict(reference_output=left['output'].numpy(),candidate_output=right['output'].numpy(),reference_loss=left['loss'].numpy(),candidate_loss=right['loss'].numpy(),gradient_names_json=np.frombuffer(json.dumps([n for n,_ in parameters]).encode(),dtype=np.uint8))
   for index,(key,_)in enumerate(parameters):
    if left['grads'][key]is not None:values[f'reference_grad_{index:02d}']=left['grads'][key].numpy()
    if right['grads'][key]is not None:values[f'candidate_grad_{index:02d}']=right['grads'][key].numpy()
   path=a.out/name
   with path.open('xb')as stream:np.savez_compressed(stream,**values)
   return dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size)
  def compare(left,right):
   row=dict(output_max_abs=float((left['output']-right['output']).abs().max()),loss_abs=float((left['loss']-right['loss']).abs()),gradients={})
   if not bool(torch.allclose(left['output'],right['output'],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL))or not bool(torch.allclose(left['loss'],right['loss'],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL)):raise RuntimeError('fixed output/loss FP32 contract failed')
   for name,_ in parameters:
    l,r=left['grads'][name],right['grads'][name]
    if(l is None)!=(r is None):raise RuntimeError('None-vs-zero gradient/Adam semantics differ: '+name)
    if l is None:raise RuntimeError('primary TRAIN case has unused parameter; do not hide Adam skip semantics: '+name)
    maximum=float((l-r).abs().max());close=bool(torch.allclose(l,r,atol=fb.GRAD_ATOL,rtol=fb.GRAD_RTOL))
    row['gradients'][name]=dict(max_abs=maximum,allclose=close,reference_norm=float(l.norm()),candidate_norm=float(r.norm()))
    if not close or not bool(torch.isfinite(r).all()):raise RuntimeError('fixed all-parameter gradient contract failed: '+name)
   return row
  def memory():
   return dict(allocated=torch.cuda.memory_allocated(device),reserved=torch.cuda.memory_reserved(device),peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device))if device.type=='cuda'else{}
  def measure(label,function,warmups=WARMUPS):
   for _ in range(warmups):check();value=function();sync();del value;charge(label+'_warmup')
   torch.cuda.reset_peak_memory_stats(device);baseline=memory();draws=[];last=None
   start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
   for repeat in range(REPEATS):
    check();sync();tick=time.perf_counter();start.record();last=function();end.record();sync()
    draws.append(dict(repeat=repeat,wall_seconds=time.perf_counter()-tick,CUDA_stream_seconds=start.elapsed_time(end)/1000.));charge(label+'_timed')
   snap=snapshot(last);del last
   def percentile(values,q):
    values=sorted(values);x=(len(values)-1)*q;i=int(x);j=min(i+1,len(values)-1);return values[i]+(values[j]-values[i])*(x-i)
   return dict(draws=draws,p50_wall_seconds=percentile([r['wall_seconds']for r in draws],.5),p95_wall_seconds=percentile([r['wall_seconds']for r in draws],.95),p50_CUDA_stream_seconds=percentile([r['CUDA_stream_seconds']for r in draws],.5),p95_CUDA_stream_seconds=percentile([r['CUDA_stream_seconds']for r in draws],.95),memory_before=baseline,memory_after=memory()),snap
  for bucket in fb.BUCKETS:
   check();phase=f'bucket{bucket}_setup';data=batches[bucket]
   if data['tokens'].shape!=(64,bucket)or any(v.dtype!=np.int64 for v in data.values())or np.any(data['lengths']>bucket)or np.any(data['lengths']<0):raise ValueError('fixed64-row bucket schema')
   original_extent=max(1,int(data['lengths'].max()));batch={key:torch.as_tensor(value,device=device)for key,value in data.items()}
   original_tokens=batch['tokens'][:,:original_extent].contiguous();sync()
   row=dict(bucket=bucket,batch=64,original_extent=original_extent,active_transition_steps=int(data['lengths'].sum()),padded_transition_steps=64*bucket,
     padded_to_active_ratio=(64*bucket)/max(1,int(data['lengths'].sum())),original_forward_MAC_proxy=model.resource_report(batch_size=64,sequence_length=original_extent,active_steps=int(data['lengths'].sum()))['proxy_forward_macs'],masked_forward_MAC_proxy=model.resource_report(batch_size=64,sequence_length=bucket,active_steps=64*bucket)['proxy_forward_macs'],
     extra_mask_select_scalars=64*bucket*432,input_tensor_bytes=sum(v.numel()*v.element_size()for v in batch.values())+original_tokens.numel()*original_tokens.element_size(),measurements={})
   result['cases'].append(row)
   def original():return fb.original_step(model,batch['initial'],original_tokens,batch['lengths'],batch['targets'],anchor)
   def masked():return fb.masked_step(model,batch['initial'],batch['tokens'],batch['lengths'],batch['targets'],anchor)
   phase=f'bucket{bucket}_original_reference';outputs=original();charge('original_numeric');golden=snapshot(outputs);del outputs
   if any(v is None for v in golden['grads'].values()):raise RuntimeError('primary TRAIN batch must exercise all parameters before capture')
   clear();fb.allocate_gradient_buffers(model);outputs=masked();charge('masked_numeric');candidate=snapshot(outputs);del outputs
   row['numeric_raw']=save_pair(f'B64-L{bucket}-masked-numeric.npz',golden,candidate);row['numeric']=compare(golden,candidate)
   outputs=masked();charge('masked_zero_accumulation_numeric');again=snapshot(outputs);del outputs
   row['zero_accumulation']=compare(golden,again);del candidate,again
   if device.type=='cuda':
    phase=f'bucket{bucket}_original_timing';clear();row['measurements']['original_gather_eager'],snap=measure('original_gather_eager',original)
    row['original_posttiming_raw']=save_pair(f'B64-L{bucket}-original-posttiming.npz',golden,snap);row['original_posttiming']=compare(golden,snap);del snap
    phase=f'bucket{bucket}_masked_eager_timing';clear();fb.allocate_gradient_buffers(model);row['measurements']['masked_eager'],snap=measure('masked_eager',masked)
    row['masked_posttiming_raw']=save_pair(f'B64-L{bucket}-masked-posttiming.npz',golden,snap);row['masked_posttiming']=compare(golden,snap);del snap
    phase=f'bucket{bucket}_capture';clear();fb.allocate_gradient_buffers(model);side=torch.cuda.Stream(device=device);side.wait_stream(torch.cuda.current_stream(device));tick=time.perf_counter()
    with torch.cuda.stream(side):
     for _ in range(WARMUPS):check();outputs=masked();del outputs;charge('graph_side_warmup')
    torch.cuda.current_stream(device).wait_stream(side);sync();side_warmup=time.perf_counter()-tick
    pointers={name:value.grad.data_ptr()for name,value in parameters};pool=torch.cuda.graph_pool_handle();graph=torch.cuda.CUDAGraph(keep_graph=True);sync();torch.cuda.reset_peak_memory_stats(device);memory_before_capture=memory();tick=time.perf_counter()
    with torch.cuda.graph(graph,pool=pool,stream=side,capture_error_mode='global'):graph_outputs=masked()
    sync();capture_seconds=time.perf_counter()-tick;tick=time.perf_counter();graph.instantiate();sync();instantiate_seconds=time.perf_counter()-tick
    if pointers!={name:value.grad.data_ptr()for name,value in parameters}:raise RuntimeError('grad buffer reallocated during capture')
    row['capture']=dict(side_warmup_seconds=side_warmup,capture_seconds=capture_seconds,instantiate_seconds=instantiate_seconds,described_FB_calls=1,executed_FB_calls_during_capture=0,memory_before=memory_before_capture,memory_after=memory(),zero_grad_inside_capture=True,Adam_inside_capture=False)
    def replay():graph.replay();return graph_outputs
    outputs=replay();sync();charge('graph_numeric');snap=snapshot(outputs);del outputs
    row['graph_numeric_raw']=save_pair(f'B64-L{bucket}-graph-before-timing.npz',golden,snap);row['graph_numeric']=compare(golden,snap);del snap
    phase=f'bucket{bucket}_graph_timing';row['measurements']['masked_CUDAgraph'],snap=measure('masked_CUDAgraph',replay,warmups=0)
    row['graph_posttiming_raw']=save_pair(f'B64-L{bucket}-graph-after-timing.npz',golden,snap);row['graph_posttiming']=compare(golden,snap);del snap
    if pointers!={name:value.grad.data_ptr()for name,value in parameters}:raise RuntimeError('captured grad pointers changed after replay')
    graph.reset();del replay,graph_outputs,graph,pool,side,pointers
   for name,value in model.state_dict().items():
    if not torch.equal(value.detach().cpu(),before[name]):raise RuntimeError('fixed-weight audit changed NN state')
   clear();del golden,batch,original_tokens,original,masked
  result.update(status='COMPLETED_FIXED_WEIGHT_FACTORIZED_FB_CONTRACT',completed=True,all_parameters_outputs_gradients_checked=True,NN_unchanged=True,
    Adam_updated=False,training_speedup_claim=False,
    scope='CPU/GPU numerical and F/B-only cost control; padded work/capture/zeroing/storage paid; no actual training or continuation repair',
    gradient_presence_scope='all primary batches require same all-present grads; None-vs-zero mismatch stops to protect future eager Adam semantics',
    backend_order='sequential original/masked/graph: stable graph-grad pointers; temporal drift remains possible; not an interleaved training benchmark')
 except BaseException as error:
  result.update(status='FAILED_OR_BOUNDED_FACTORIZED_FB_CONTRACT',completed=False,error=f'{type(error).__name__}: {error}',phase=phase)
  (a.out/'exception.txt').write_text(traceback.format_exc());raise
 finally:
  result.update(existing_TRAIN_label_example_exposures=64*sum(result['actual_FB_calls'].values()),anchor_target_encoder_lookups=64*sum(result['actual_FB_calls'].values()),anchor_coordinate_loss_elements=64*32*sum(result['actual_FB_calls'].values()),capture_label_executions_counted=False,wall_seconds=time.perf_counter()-started,cpu_seconds=time.process_time()-cpu_started,cpu_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024))
  with(a.out/('summary.json'if result.get('completed')else'failure.json')).open('x')as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(status=result['status'],out=str(a.out),cases=len(result['cases']),SGD_updates=0)),flush=True)

if __name__=='__main__':main()
