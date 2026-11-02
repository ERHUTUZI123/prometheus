"""One zero-optimizer GPU execution/cost admission; current solve unchanged.

CPU reference is the already executed, SHA-bound B64L16 masked fixture.
All eager evidence precedes one risky CUDA Graph capture. No retry/fallback.
"""
import argparse,copy,hashlib,json,resource,statistics,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import continuous_primitive_block as core
import typed_dual_memory_masked_fb as fb
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def memory():return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
def summarize(rows):return dict(iterations=len(rows),wall_mean=statistics.mean(r['wall_seconds']for r in rows),wall_median=statistics.median(r['wall_seconds']for r in rows),CUDA_stream_elapsed_mean=statistics.mean(r['CUDA_stream_elapsed_seconds']for r in rows),rows=rows,scope='CUDAevent streamelapsed includeshostgaps, notexclusiveGPUbusytime')
def pair(out,left,right,names):
 raw={};checks={}
 for key,a in left.items():
  b=right[key];a=a.detach().cpu();b=b.detach().cpu();raw['CPU_'+key]=a.numpy();raw['GPU_'+key]=b.numpy();atol,rtol=(fb.GRAD_ATOL,fb.GRAD_RTOL)if key.startswith('grad_')else(fb.OUTPUT_ATOL,fb.OUTPUT_RTOL);checks[key]=dict(max_abs=float((a-b).abs().max()),finite=bool(torch.isfinite(a).all()and torch.isfinite(b).all()),allclose=bool(torch.allclose(a,b,atol=atol,rtol=rtol)),atol=atol,rtol=rtol)
 raw['gradient_names_json']=np.frombuffer(json.dumps(names).encode(),dtype=np.uint8)
 with out.open('xb')as f:np.savez_compressed(f,**raw)
 return dict(file=out.name,sha256=sha(out),bytes=out.stat().st_size,all_finite=all(c['finite']for c in checks.values()),fixed_tolerances_pass=all(c['allclose']for c in checks.values()),fields=checks)
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
 parser.add_argument('--max-seconds',type=float,default=120);a=parser.parse_args()
 if not 1<=a.max_seconds<=120:raise ValueError('120science/150outer')
 a.out.mkdir(parents=True,exist_ok=False);began=time.perf_counter();cpu=time.process_time();phase='input';evidence={};capture=None
 try:
  torch.set_num_threads(1)
  if not torch.cuda.is_available():raise RuntimeError('authorizedsameGPU required')
  torch.cuda.set_per_process_memory_fraction(.3,0);manifest=json.loads((a.inputs/'manifest.json').read_text())
  if manifest['schema']!='typed-dual-memory-GPU-execution-v1':raise ValueError('onefixedexecutionprotocol')
  for leaf,h in manifest['source_sha256'].items():
   if sha(Path(__file__).with_name(leaf))!=h:raise ValueError('frozen source changed')
  for leaf,h in manifest['input_sha256'].items():
   if sha(a.inputs/leaf)!=h:raise ValueError('exactOWNinput changed')
  proof=json.loads((a.inputs/'CPU-proof.json').read_text())
  if proof['status']!='CPU_TYPED_DUAL_MEMORY_ADMISSION_PASS'or proof['SGD_updates']!=0 or not proof['all20_eager_masked_gradients_allclose']:raise ValueError('actualCPUproof required')
  rawCPU=torch.load(a.inputs/'CPU-reference.pt',map_location='cpu',weights_only=False);data=rawCPU['synthetic_batch'];row=torch.arange(64)
  expected=dict(initial=row%8,tokens=(row[:,None]*3+torch.arange(16)[None]*5)%16,lengths=row%17,targets=(row+3)%8)
  if data.keys()!=expected.keys()or any(not torch.equal(data[k],expected[k])for k in expected):raise ValueError('exactadmittedB64L16syntheticcase required')
  payload=torch.load(a.inputs/'ab.pt',map_location='cpu',weights_only=False);model=core.ContinuousPrimitiveBlock.from_owned_ab_weights(payload,expected_source_sha256=manifest['parent_source_sha256']);theta={k:v.detach().clone()for k,v in model.state_dict().items()};del payload;model=model.to('cuda:0');model.eval();batch={k:v.to('cuda:0')for k,v in data.items()};params=list(model.named_parameters());names=[k for k,_ in params]
  if model.parameter_count()!=10351 or len(params)!=20:raise ValueError('exactsame20/Ptheta')
  fb.allocate_gradient_buffers(model);pointers={k:(p.data_ptr(),p.grad.data_ptr())for k,p in params}
  def check():
   if time.perf_counter()-began>a.max_seconds:raise TimeoutError('boundedGPUadmission')
  def stable():
   if pointers!={k:(p.data_ptr(),p.grad.data_ptr()if p.grad is not None else None)for k,p in params}:raise ValueError('NN/gradstoragereplaced')
  def snapshot(output,loss):return dict(output=output.detach().cpu(),loss=loss.detach().cpu(),**{f'grad_{i:02d}':p.grad.detach().cpu().clone()for i,(_,p)in enumerate(params)})
  def unchanged():return all(torch.equal(v.detach().cpu(),theta[k])for k,v in model.state_dict().items())
  phase='GPU_reference';torch.cuda.synchronize();tick=time.perf_counter();output,carry=fb.masked_forward_value(model,batch['initial'],batch['tokens'],batch['lengths']);loss=F.nll_loss(output,batch['targets']);grad=torch.autograd.grad(loss,tuple(p for _,p in params),allow_unused=True)
  if any(g is None for g in grad):raise ValueError('GPUall20connectedgradientrequired')
  with torch.no_grad():
   for(_,p),g in zip(params,grad):p.grad.copy_(g)
  torch.cuda.synchronize();left=dict(output=rawCPU['masked_output'],loss=rawCPU['masked_loss'],**{f'grad_{i:02d}':rawCPU['masked_gradients'][name]for i,name in enumerate(names)});right=snapshot(output,loss);evidence['CPU_GPU_eager']=pair(a.out/'CPU-GPU-eager-all20.npz',left,right,names);evidence['GPU_reference_wall_seconds']=time.perf_counter()-tick
  if not evidence['CPU_GPU_eager']['all_finite']:raise ValueError('nonfiniteGPUreference; stopbeforetiming/capture')
  formed={k:v.detach().clone()for k,v in carry.items()};del output,loss,grad,carry
  side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream());phase='warmup';tick=time.perf_counter()
  with torch.cuda.stream(side):
   for _ in range(3):check();fb.masked_step(model,**batch)
  torch.cuda.current_stream().wait_stream(side);torch.cuda.synchronize();stable();evidence['NN_warmup']=dict(actual_FB=3,SGD=0,wall_seconds=time.perf_counter()-tick)
  def time10(function):
   rows=[]
   for _ in range(10):
    check();torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);tick=time.perf_counter();start.record();value=function();end.record();torch.cuda.synchronize();rows.append(dict(wall_seconds=time.perf_counter()-tick,CUDA_stream_elapsed_seconds=start.elapsed_time(end)/1000));del value
   return summarize(rows)
  phase='eager_timing';evidence['NN_eager_timing']=time10(lambda:fb.masked_step(model,**batch));stable();evidence['theta_unchanged_before_capture']=unchanged();evidence['memory_after_NN_eager']=memory()
  # Isolate the unchanged 8x8 solver on a detached, actually formed final
  # memory state. This is algebra cost, not another NN/training case.
  phase='solver_cost';z=formed['key_composition'].sum(-1);A=formed['key_composition']/z.clamp_min(1e-4)[:,:,None];q=F.softmax(formed['address'],-1);G=(torch.bmm(A,A.transpose(1,2))+1e-4*model.dual_identity).detach().requires_grad_();rhs=torch.bmm(A,q[:,:,None]).detach().requires_grad_()
  def solve_FB():
   b=torch.linalg.solve(G,rhs);objective=b.square().mean();g=torch.autograd.grad(objective,(G,rhs));return b,g
  tick=time.perf_counter();b,sg=solve_FB();solverRaw=dict(formed_C=formed['key_composition'].cpu().numpy(),formed_V=formed['values'].cpu().numpy(),formed_query=q.cpu().numpy(),Gram=G.detach().cpu().numpy(),rhs=rhs.detach().cpu().numpy(),solution=b.detach().cpu().numpy(),gradient_Gram=sg[0].detach().cpu().numpy(),gradient_rhs=sg[1].detach().cpu().numpy())
  with(a.out/'formed-solver-reference.npz').open('xb')as f:np.savez_compressed(f,**solverRaw)
  if any(not np.isfinite(v).all()for v in solverRaw.values()):raise ValueError('nonfinitesolvercostpoint')
  for _ in range(3):check();solve_FB()
  torch.cuda.synchronize();evidence['solver_only']=dict(formed_state=True,reference_FB=1,warmup_FB=3,timing=time10(solve_FB),reference_raw_sha256=sha(a.out/'formed-solver-reference.npz'),scope='64detachedformed8x8solves/FB; objectiveb², noNNparameter/SGD; CPU_GPUfullNNreferenceabove, solverworkspaceunknown');del b,sg,G,rhs,A,q,formed;stable()
  props=torch.cuda.get_device_properties(0);evidence['backend']=dict(torch_version=str(torch.__version__),CUDA_version=torch.version.cuda,device=props.name,total_memory=props.total_memory,matmul_precision=torch.get_float32_matmul_precision(),allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),policies_changed=False)
  evidence['memory_before_capture']=memory();put(a.out/'EAGER_EVIDENCE_BEFORE_CAPTURE.json',evidence)
  phase='capture';graph=torch.cuda.CUDAGraph(keep_graph=True);tick=time.perf_counter()
  try:
   side.wait_stream(torch.cuda.current_stream())
   with torch.cuda.graph(graph,stream=side,capture_error_mode='global'):captured=fb.masked_step(model,**batch)
   graph.instantiate();capture=dict(attempts=1,supported=True,wall_seconds=time.perf_counter()-tick)
  except BaseException as error:
   chain=[];current=error
   while current is not None:chain.append(dict(type=type(current).__name__,error=str(current)));current=current.__context__
   capture=dict(attempts=1,supported=False,phase=phase,wall_seconds=time.perf_counter()-tick,exception_chain=chain,traceback=traceback.format_exc(),completed_GPU_FB_during_capture=None,post_failure_NN_calls=0)
   put(a.out/'CAPTURE_FAILURE_CURRENT_SOLVE.json',capture)
   try:graph.reset();capture['cleanup']='graphresetreturned'
   except BaseException as cleanup:capture['cleanup_error']=repr(cleanup)
   # No further NN, GPU synchronization, timing or postfailure θ readback.
  if capture['supported']:
   phase='graph_validation';graph.replay();torch.cuda.synchronize();stable();evidence['CPU_GPU_graph']=pair(a.out/'CPU-GPU-graph-all20.npz',left,snapshot(*captured),names);evidence['NN_graph_timing']=time10(graph.replay);stable();evidence['theta_unchanged_after_graph']=unchanged();evidence['memory_after_graph']=memory();graph.reset()
  result=dict(status='GPU_DUAL_MEMORY_EXECUTION_DIAGNOSTIC_COMPLETE',source_sha256=manifest['source_sha256'],CPU_proof_sha256=manifest['input_sha256']['CPU-proof.json'],CPU_reference_sha256=manifest['input_sha256']['CPU-reference.pt'],checkpoint_sha256=manifest['input_sha256']['ab.pt'],P=10351,parameter_tensors=20,carry_scalars=384,parameter_bytes=41404,gradient_bytes=41404,shared_buffer_bytes=256,static_input_bytes=sum(v.numel()*v.element_size()for v in batch.values()),optimizer_instantiated=False,optimizer_updates=0,SGD_updates=0,new_training_budget=0,CPU_reference_reused_no_CPU_NN_FB=True,evidence=evidence,capture=capture,numerical_GPU_admission=evidence['CPU_GPU_eager']['fixed_tolerances_pass']and(capture['supported']and evidence.get('CPU_GPU_graph',{}).get('fixed_tolerances_pass',False)),actual_NN_eager_FB=14,actual_graph_replay_FB=11 if capture['supported']else 0,solver_only_FB=14,physical_forward_solves_per_NN_FB=1024,active_semantic_steps_per_FB=int(data['lengths'].sum()),solver_backward_FLOPs=None,private_graph_pool_bytes=None,driver_context_bytes=None,full_forward_FLOPs=None,full_training_FLOPs=None,provider_effective_compute=None,wall_seconds=time.perf_counter()-began,CPU_seconds=time.process_time()-cpu,CPU_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)),scope='oneboundedexecution/costtest; capturefailurepreservedwithoutfallback, sameλ/θ/core/tols; noVALID/TEST/optimizer/learning, GPUtrainingnotadmittedbytimingalone')
  put(a.out/'result.json',result);print(json.dumps(dict(status=result['status'],capture_supported=capture['supported'],CPU_GPU_eager_numeric=evidence['CPU_GPU_eager']['fixed_tolerances_pass'],SGD=0),sort_keys=True))
 except BaseException as error:put(a.out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),evidence=evidence,capture=capture,SGD_updates=0));raise
if __name__=='__main__':main()
