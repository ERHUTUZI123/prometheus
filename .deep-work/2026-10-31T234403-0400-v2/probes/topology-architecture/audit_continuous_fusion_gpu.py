"""Prospective fixed-weight GPU numerical/cost attempt; no training or labels."""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import sys
import time
import traceback

CHECKPOINT_SHA='1e0fc8ec0e063aa1c86f4c7b0d09d91d65c32338273323a4c6d10c2752993f80'
REFERENCE_SHA='3e82598a157a5c41f47d7c3e0b7ddf417899526a4027146d1f2d1dac7dae29b1'
KERNEL_SHA='5cf6929bb0da78d8a7dc65cbdfe7d9dcaf0ba9d182845564f3f36c06e6f88d29'
ATOL,RTOL=2e-4,2e-5
WARMUPS,REPEATS=3,12

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def owned(path,name):
 spec=importlib.util.spec_from_file_location(name,path)
 if spec is None or spec.loader is None:raise RuntimeError('fixed reference import unavailable')
 value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value);return value

def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
 p.add_argument('--max-seconds',type=float,default=730);a=p.parse_args()
 if not 1<=a.max_seconds<=850:raise ValueError('scientific bound<=850s')
 a.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cpu_started=time.process_time();deadline=started+a.max_seconds
 stream=(a.out/'records.jsonl').open('x',buffering=1)
 result=dict(status='running_fused_GPU_attempt',no_training=True,optimizer_updates=0,labels_loaded=0,cases=[],forward_counts={},capture_forward_descriptions=0)
 phase='import';active_case=None
 def emit(kind,**kw):stream.write(json.dumps(dict(kind=kind,phase=phase,wall=time.perf_counter()-started,**kw),sort_keys=True,allow_nan=False)+'\n')
 def check():
  if time.perf_counter()>deadline:raise TimeoutError('fixed fusion GPU scientific bound')
 def charge(label,batch,length):
  row=result['forward_counts'].setdefault(label,dict(calls=0,examples=0,prediction_positions=0))
  row['calls']+=1;row['examples']+=batch;row['prediction_positions']+=batch*length
 try:
  import torch
  import numpy as np
  F=torch.nn.functional;folder=Path(__file__).resolve().parent;device=torch.device('cuda:0')
  if not torch.cuda.is_available():raise RuntimeError('CUDA required; no fallback')
  torch.set_num_threads(1)
  if sha(folder/'fused_continuous_block_fp32.py')!=KERNEL_SHA or sha(folder/'audit_cuda_graph_inference.py')!=REFERENCE_SHA:raise ValueError('fixed source changed')
  # The prologue-created module/JIT object must survive unchanged. Reimporting
  # or compiling another kernel after strict transition is never a fallback.
  fusion=sys.modules.get('fused_continuous_block_fp32')
  priming=json.loads((a.out.parent/'compiler-stage-total.json').read_text())
  if fusion is None or id(fusion)!=priming.get('module_python_id') or id(fusion._fused_soft_fp32)!=priming.get('jit_function_python_id'):
   raise RuntimeError('canonical prologue module/JIT object missing or changed')
  expected_compiled={}
  for entry in priming.get('kernel_cases',[]):
   length=entry['context'];identity=entry['compiled_python_id']
   if length in expected_compiled and expected_compiled[length]!=identity:raise RuntimeError('B1/32 produced different compiled-kernel object; stop before science')
   expected_compiled[length]=identity
  if set(expected_compiled)!={64,128,256}:raise RuntimeError('three fixed L kernels were not primed')
  reference=owned(folder/'audit_cuda_graph_inference.py','_owned_fusion_GPU_reference')
  reference.torch=torch;reference.np=np;reference.F=F
  checkpoint=a.inputs/'checkpoint.pt';metadata_path=a.inputs/'metadata.json';train_path=a.inputs/'train.npz'
  if checkpoint.is_symlink()or checkpoint.stat().st_size!=549107 or sha(checkpoint)!=CHECKPOINT_SHA:raise ValueError('exact completed300 dynamic checkpoint required')
  phase='restore';tick=time.perf_counter();payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
  metadata=json.loads(metadata_path.read_text())
  if metadata!=payload['metadata']or sha(metadata_path)!='f157d58e9246f59f78dbd883b4ccc80284278b9eb270365cb13556dc8703e792':raise ValueError('original metadata changed')
  if sha(train_path)!=metadata['files']['train']['sha256']:raise ValueError('original TRAIN file changed')
  with np.load(train_path,allow_pickle=False)as stored:
   if set(stored.files)!={'initial','tokens','lengths','targets'}:raise ValueError('original TRAIN archive required')
   # Do not decompress targets. No VALID/TEST file is copied to these inputs.
   train={name:stored[name]for name in('initial','tokens','lengths')}
  if train['tokens'].shape!=(8192,64)or np.any(train['initial'])or np.any(train['lengths']!=64):raise ValueError('fixed initial0/full TRAIN windows required')
  model,layouts=reference.restore(argparse.Namespace(arm='dynamic',frozen_source=folder),payload,device)
  before_hash=reference.state_hash(model.state_dict())
  historical=dict(checkpoint_bytes=checkpoint.stat().st_size,saved_optimizer_tensor_bytes=reference.tensor_bytes(payload['optimizer']),source_training_costs=payload['costs'],scope='historical training retained, not added again to frozen inference latency')
  restore_seconds=time.perf_counter()-tick;del payload
  phase='pack';packed,pack_report=fusion.pack_frozen_model(model)
  weights=dict(model.body.named_parameters())
  for name,shape in fusion.PARAMETER_LAYOUT:
   value=packed[fusion.OFFSETS[name]:fusion.OFFSETS[name]+weights[name].numel()].view(shape)
   if not torch.equal(value,weights[name]):raise RuntimeError('actual GPU packed value mismatch: '+name)
  packed_before=hashlib.sha256(packed.detach().cpu().numpy().tobytes()).hexdigest()
  result.update(original_P=sum(v.numel()for v in model.parameters()),original_model_parameter_bytes=reference.tensor_bytes(dict(model.named_parameters())),pack=pack_report,historical=historical,
   original_buffer_bytes=reference.tensor_bytes(dict(model.named_buffers())),live_inference_optimizer_bytes=0,
   restored_layouts=layouts,restore_seconds=restore_seconds,model_state_sha256=before_hash,packed_sha256=packed_before,
   priming_module_identity_verified=True,primed_compiled_objects=expected_compiled,
   precision=dict(matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,float32_matmul_precision=torch.get_float32_matmul_precision()),
   torch_version=str(torch.__version__),triton_version=fusion.triton.__version__,GPU=torch.cuda.get_device_name(device),
   source_sha256=dict(kernel=KERNEL_SHA,reference=REFERENCE_SHA,checkpoint=CHECKPOINT_SHA))
  def sync():torch.cuda.synchronize(device)
  def memory():return dict(allocated=torch.cuda.memory_allocated(device),reserved=torch.cuda.memory_reserved(device),peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device))
  def compare(left,right):
   if left.shape!=right.shape or not bool(torch.isfinite(right).all()):raise RuntimeError('invalid fused full output')
   delta=(left-right).abs();maximum=float(delta.max());mismatches=int((left.argmax(-1)!=right.argmax(-1)).sum());norm=float(torch.logsumexp(right,-1).abs().max())
   row=dict(maximum_absolute_logprob_difference=maximum,allclose=bool(torch.allclose(left,right,atol=ATOL,rtol=RTOL)),prediction_disagreements=mismatches,maximum_log_normalizer_error=norm,atol=ATOL,rtol=RTOL,checked_elements=left.numel())
   if maximum>ATOL or not row['allclose']or mismatches or norm>1e-5:raise RuntimeError('fixed fused numerical contract FAILED: '+json.dumps(row))
   return row
  def capture(label,function,batch,length):
   check();sync();start=time.perf_counter();side=torch.cuda.Stream(device=device);side.wait_stream(torch.cuda.current_stream(device))
   with torch.cuda.stream(side):
    for _ in range(WARMUPS):check();value=function();del value;charge(label+'_side_warmup',batch,length)
   torch.cuda.current_stream(device).wait_stream(side);sync();side_seconds=time.perf_counter()-start
   tick=time.perf_counter();pool=torch.cuda.graph_pool_handle();graph=torch.cuda.CUDAGraph(keep_graph=True);construction=time.perf_counter()-tick
   sync();tick=time.perf_counter()
   with torch.cuda.graph(graph,pool=pool,stream=side,capture_error_mode='global'):output=function()
   sync();capture_seconds=time.perf_counter()-tick;result['capture_forward_descriptions']+=1
   tick=time.perf_counter();graph.instantiate();sync();instantiate=time.perf_counter()-tick
   def replay():graph.replay();return output
   report=dict(side_warmup_seconds=side_seconds,construction_seconds=construction,capture_seconds=capture_seconds,instantiate_seconds=instantiate,memory=memory(),scope='capture describes work; not counted as an executed forward')
   return replay,graph,pool,output,report
  with torch.inference_mode():
   for length in(64,128,256):
    for batch in(1,32):
     check();phase='input_setup';case_started=time.perf_counter();sync();torch.cuda.reset_peak_memory_stats(device)
     tokens=torch.as_tensor(np.tile(train['tokens'][:batch],(1,length//64)),device=device)
     initial=torch.as_tensor(train['initial'][:batch],device=device);lengths=torch.full((batch,),length,dtype=torch.long,device=device)
     fusion.validate_all_active_initial0(initial,tokens,lengths)
     buffers=fusion.allocate_buffers(tokens);sync()
     active_case=dict(batch=batch,context=length,input_sha256=hashlib.sha256(tokens.cpu().numpy().tobytes()).hexdigest(),input_bytes=tokens.numel()*tokens.element_size(),extra_output_carry_bytes=sum(v.numel()*v.element_size()for v in buffers.values()),input_setup_seconds=time.perf_counter()-case_started)
     result['cases'].append(active_case)
     pure=lambda:reference.tensor_forward(model,'dynamic',initial,tokens,'soft')
     def fused():
      compiled=fusion.launch_packed(tokens,packed,buffers)
      if id(compiled)!=expected_compiled[length]:raise RuntimeError('post-strict compiled object/cache key changed; no fallback')
      return buffers['output']
     phase='correctness';public=model(initial,tokens,lengths,'soft');sync();charge('public_correctness',batch,length)
     baseline=public.detach().cpu().contiguous();del public
     pure_value=pure();sync();charge('tensor_correctness',batch,length)
     active_case['public_vs_tensor']=compare(baseline,pure_value.detach().cpu())
     carry_output,reference_carry=model.body.forward_all(initial,tokens,lengths,execution='soft',output='logits',return_carry=True)
     sync();charge('public_carry_correctness',batch,length);active_case['public_carry_output_contract']=compare(baseline,carry_output.detach().cpu());del carry_output
     value=fused();sync();charge('fused_correctness',batch,length)
     raw_path=a.out/f'B{batch}-L{length}-outputs.npz'
     with raw_path.open('xb')as stream_out:
      np.savez_compressed(stream_out,reference=baseline.numpy(),pure=pure_value.detach().cpu().numpy(),fused=value.detach().cpu().numpy(),h=buffers['h'].cpu().numpy(),keys=buffers['keys'].cpu().numpy(),values=buffers['values'].cpu().numpy(),reference_h=reference_carry['h'].cpu().numpy(),reference_keys=reference_carry['keys'].cpu().numpy(),reference_values=reference_carry['values'].cpu().numpy())
     active_case.update(raw_outputs_file=raw_path.name,raw_outputs_sha256=sha(raw_path),raw_outputs_bytes=raw_path.stat().st_size)
     active_case['tensor_vs_fused']=compare(baseline,value.detach().cpu())
     active_case['final_carry_diagnostics']={key:dict(maximum_absolute_difference=float((reference_carry[key]-buffers[key]).abs().max()),allclose=bool(torch.allclose(reference_carry[key],buffers[key],atol=ATOL,rtol=RTOL)),scope='additional state diagnostics, fixed output gate unchanged')for key in('h','keys','values')}
     if any(not bool(torch.isfinite(buffers[key]).all())for key in('h','keys','values')):raise RuntimeError('nonfinite final carry')
     del pure_value,value,reference_carry
     phase='warmup'
     for label,function in(('tensor_eager',pure),('fused_eager',fused)):
      for _ in range(WARMUPS):check();value=function();sync();del value;charge(label+'_warmup',batch,length)
     phase='capture_tensor';pure_replay,pure_graph,pure_pool,pure_output,pure_setup=capture('tensor_graph',pure,batch,length)
     phase='capture_fused';fused_replay,fused_graph,fused_pool,fused_output,fused_setup=capture('fused_graph',fused,batch,length)
     phase='graph_correctness'
     value=pure_replay();sync();charge('tensor_graph_correctness',batch,length);active_case['public_vs_tensor_graph']=compare(baseline,value.detach().cpu())
     value=fused_replay();sync();charge('fused_graph_correctness',batch,length);active_case['public_vs_fused_graph']=compare(baseline,value.detach().cpu())
     active_case.update(tensor_graph_setup=pure_setup,fused_graph_setup=fused_setup,draws=[])
     backends=(('tensor_eager',pure),('tensor_graph',pure_replay),('fused_eager',fused),('fused_graph',fused_replay))
     phase='paired_timing'
     start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
     for repeat in range(REPEATS):
      order=backends[repeat%4:]+backends[:repeat%4]
      for label,function in order:
       check();sync();tick=time.perf_counter();start_event.record();value=function();end_event.record();sync()
       wall=time.perf_counter()-tick;cuda_seconds=start_event.elapsed_time(end_event)/1000.;del value
       charge(label+'_timed',batch,length);row=dict(backend=label,repeat=repeat,wall_seconds=wall,CUDA_stream_seconds=cuda_seconds);active_case['draws'].append(row);emit('latency_draw',batch=batch,context=length,**row)
     timings={}
     for label,_ in backends:
      draws=[d for d in active_case['draws']if d['backend']==label]
      timings[label]=dict(p50_wall_seconds=reference.percentile([d['wall_seconds']for d in draws],.5),p95_wall_seconds=reference.percentile([d['wall_seconds']for d in draws],.95),p50_CUDA_stream_seconds=reference.percentile([d['CUDA_stream_seconds']for d in draws],.5),p95_CUDA_stream_seconds=reference.percentile([d['CUDA_stream_seconds']for d in draws],.95))
     saving=timings['tensor_graph']['p50_CUDA_stream_seconds']-timings['fused_graph']['p50_CUDA_stream_seconds']
     active_case.update(timings=timings,fused_graph_vs_tensor_graph_ratio=timings['fused_graph']['p50_CUDA_stream_seconds']/timings['tensor_graph']['p50_CUDA_stream_seconds'],per_call_CUDA_stream_saving_seconds=saving,memory_after=memory(),case_wall_seconds=time.perf_counter()-case_started)
     if before_hash!=reference.state_hash(model.state_dict())or packed_before!=hashlib.sha256(packed.detach().cpu().numpy().tobytes()).hexdigest():raise RuntimeError('inference changed original or packed weights')
     emit('case_completed',measurement=active_case)
     phase='cleanup';pure_graph.reset();fused_graph.reset();del pure_graph,fused_graph,pure_pool,fused_pool,pure_output,fused_output,pure_replay,fused_replay,backends,order,function,_,pure,fused,buffers,tokens,initial,lengths,baseline
     gc.collect();torch.cuda.empty_cache();sync();active_case=None
  result.update(status='completed_fixed_fusion_GPU_attempt',completed=True,all_six_numerical_contracts_passed=True,unchanged_original_and_packed_weights=True,
   actual_speed_result='report every six-case ratio; no speed/accuracy acceptance gate',
   break_even_scope='whole priming/packing/capture costs retained separately; amortization must use measured positive savings and avoid nested cost double count',
   cost_scope='original41593P retained plus166372B packed duplicate; all output/finalcarry writes, allocations/warmups/captures/instantiation/cleanup charged; no training/energy/exclusive busy-time claim',
   limitation='only frozen-weight dynamic-soft all-active initial0 inference at fixed6shapes; no training/novel learning mechanism claim')
 except BaseException as error:
  result.update(status='failed_or_bounded_fusion_GPU_attempt',completed=False,error=f'{type(error).__name__}: {error}',failed_phase=phase)
  (a.out/'exception.txt').write_text(traceback.format_exc());emit('failed',error=result['error']);raise
 finally:
  result.update(wall_seconds=time.perf_counter()-started,cpu_self_seconds=time.process_time()-cpu_started,cpu_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  with(a.out/('summary.json'if result.get('completed')else'failure.json')).open('x')as output:json.dump(result,output,indent=2,sort_keys=True,allow_nan=False);output.write('\n')
  stream.close()
 print(json.dumps(dict(status=result['status'],completed=result['completed'],cases=len(result['cases']),out=str(a.out))),flush=True)

if __name__=='__main__':main()
