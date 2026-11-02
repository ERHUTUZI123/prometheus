"""ROOT-only one-phase600 exact fcba masked-Graph continuation; no import work.

Restores the SAME original anchor0 stage2 NN/Adam/RNG/Replay128, grows once
using the unchanged helper, then captures B64L48 F/B only. Stable existing
grad buffers are zeroed IN the graph; clipping/Adam remain eager. Fixed raw
TRAIN batches and terminal NLL+0*binding objective match no-profile v3.
No SGD in warmup/admission/capture/diagnostics, no TEST before final freeze.
"""
import argparse,copy,hashlib,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import continuous_primitive_block as core
import owned_parent_driver as old
import factorized_masked_graph_fb as fb
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage

RAW=('initial','tokens','lengths','targets')
B,L,STEPS=64,48,600
CORE_SHA='fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158'
RESUME_SHA='9bb47e660e4ea67d0bf1a06e93e418f0c3fa50760fa76ff067d3a0c954a6c40f'
INDICES_PAYLOAD_SHA='a840c72965717b198837fb893c288d30c6d20809ecba22439cbb8d850c787007'
REPLAY_PAYLOAD_SHA='c00780e9f7d57f4c26e3e472da8563b2aaab65d00926aaa55b0a4581fbfa9ab1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arraysha(v):return hashlib.sha256(v.tobytes(order='C')).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in('inputs','out'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--max-seconds',type=float,default=850);a=p.parse_args()
 if not 1<=a.max_seconds<=850:raise ValueError('fixed850s inside/900s outside bound')
 rec=old.Recorder(a.out,a.max_seconds);device=torch.device('cuda:0');model=opt=graph=pool=side=None;completed=0;phase='restore';reports=[]
 whole_cpu=time.process_time();restore_started=time.perf_counter();restore_CPU=time.process_time();costs=dict(SGD_updates=0,training_wall_seconds=0.,training_CPU_seconds=0.,CUDA_stream_elapsed_seconds=0.,graph_FB_CUDA_seconds=0.,padded_forward_MAC_proxy=0,original_active_forward_MAC_proxy=0,active_token_steps=0,padded_token_steps=0,batch_assembly_wall_seconds=0.,static_copy_wall_seconds=0.,batch_H2D_payload_bytes=0,static_D2D_payload_bytes=0,current_training_label_reads=0,old_replay_training_label_reads=0,anchor_target_encoder_lookups=0,anchor_coordinate_loss_elements=0,evaluation_wall_seconds=0.,evaluation_forward_MAC_proxy=0,evaluation_forward_rows=0,diagnostic_wall_seconds=0.,diagnostic_CPU_seconds=0.,diagnostic_original_forward_MAC_proxy=0,diagnostic_graph_forward_MAC_proxy=0,warmup_FB_calls=0,diagnostic_gather_FB_calls=0,diagnostic_graph_FB_calls=0,actual_training_graph_FB_calls=0,capture_described_FB_calls=0,full_training_FLOPs=None)
 try:
  if not torch.cuda.is_available():raise RuntimeError('authorized CUDA required; no fallback')
  torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,device)
  torch.manual_seed(0);np.random.seed(0);random.seed(0)
  manifest=json.loads((a.inputs/'manifest.json').read_text())
  if manifest.get('schema')!='factorized-graph-stage3-inputs-v1':raise ValueError('fixed input schema required')
  for leaf,digest in manifest['source_sha256'].items():
   if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen source changed: '+leaf)
  if sha(Path(core.__file__))!=CORE_SHA:raise ValueError('exact fcba core required')
  args=argparse.Namespace(arm='dynamic',states=8,seed=0,lr=.003,block_state='continuous',block_width=32,primitive_rank=8,memory_slots=8,slots=1,block_training='soft_only',replay_capacity=128,replay_per_batch=16,stages=4,steps=600,anchor=0,organization=core.ContinuousPrimitiveBlock.organization_policy)
  def load(stage,split):
   spec=manifest['files'][f'{stage}:{split}'];path=a.inputs/spec['leaf']
   if sha(path)!=spec['sha256']:raise ValueError('exact original input changed')
   with np.load(path,allow_pickle=False)as z:
    if set(z.files)!=set(RAW):raise ValueError('raw terminal fields only')
    data={k:z[k]for k in RAW}
   n=len(data['initial']);width=data['tokens'].shape[1];mask=np.arange(width)[None]<data['lengths'][:,None]
   if any(v.dtype!=np.int64 for v in data.values())or any(data[k].shape!=(n,)for k in('initial','lengths','targets'))or np.any((data['lengths']<0)|(data['lengths']>width))or np.any((data['initial']<0)|(data['initial']>=8))or np.any((data['targets']<0)|(data['targets']>=8))or np.any(mask&((data['tokens']<0)|(data['tokens']>=(16 if stage==3 else 15)))):raise ValueError('original raw schema/ranges')
   rec.emit('actual_input_opened',stage=stage,split=split,sha256=spec['sha256'],rows=n,array_bytes=sum(v.nbytes for v in data.values()))
   return data
  datasets={s:{k:load(s,k)for k in('train','valid')}for s in range(4)};train_sources={s:datasets[s]['train']for s in range(4)};data=datasets[3]['train']
  if data['tokens'].shape!=(4096,128)or np.any(data['lengths']>48)or np.any(data['tokens'][:,48:]):raise ValueError('raw128 TRAIN cannot be reduced to graph48 without active truncation')
  model=core.ContinuousPrimitiveBlock(8,15).to(device);opt=torch.optim.Adam(model.parameters(),lr=.003)
  resume=a.inputs/'resume.pt'
  if sha(resume)!=RESUME_SHA or RESUME_SHA!=manifest['resume']['checkpoint_sha256']:raise ValueError('same OWN stage2 checkpoint required')
  payload=torch.load(resume,map_location='cpu',weights_only=False)
  if payload.get('schema')!='factorized-value-development-training-v1'or payload.get('stage')!=2 or payload.get('step')!=600 or payload.get('stage_complete')is not True or payload['args']!=vars(args)or payload['source_sha256']!=manifest['resume']['source_sha256']:raise ValueError('exact same anchor0 completed stage2 lineage required')
  model.load_state_dict(payload['model_state'],strict=True);opt.load_state_dict(payload['optimizer']);replay=Replay128.from_state_dict(payload['replay_state'])
  if not old.tensor_state_equal(model.state_dict(),payload['model_state'])or not old.nested_state_equal(opt.state_dict(),payload['optimizer'])or not old.nested_state_equal(replay.state_dict(),payload['replay_state']):raise ValueError('real NN/Adam/replay restore differs')
  if model.configuration()!=payload['block_checkpoint']['configuration']or replay.capacity!=128 or replay.width!=48 or replay.size!=128 or replay.seen!=96000 or replay.draws!=19200 or replay.replacements!=844 or replay.digest()!='440344f6cd778e8443081cdeafa95847bf0d23aa825f97590c55436b02a7f470':raise ValueError('stage2 architecture/replay boundary differs')
  old.restore_rng(payload['rng'])
  if not old.nested_state_equal(old.rng_state(),payload['rng']):raise ValueError('actual RNG restore differs')
  used={}
  for s in range(3):
   entry=manifest['files'][f'report{s}'];path=a.inputs/entry['leaf']
   if sha(path)!=entry['sha256']:raise ValueError('original completed preTEST report differs')
   prior=json.loads(path.read_text())
   if prior.get('test_data_read')is not False or prior['report']['stage']!=s:raise ValueError('original training-only report required')
   inherited=prior['report'];inherited['checkpoint']=dict(inherited['checkpoint'],historical_origin_run=manifest['resume']['parent_run'],not_recreated_here=True);reports.append(inherited)
   ids=np.random.default_rng(np.random.SeedSequence([0,8,s,9027])).integers(0,len(train_sources[s]['initial']),size=(600,64),dtype=np.int64);used[s]=ids if s==0 else ids[:,:48]
  source_audit=replay.validate_sources(train_sources,used);del payload
  rec.emit('actual_restore',checkpoint_sha256=RESUME_SHA,actual_NN_Adam_RNG_replay_restored=True,repeated_old_stages=0,replay=source_audit)
  before=model(torch.tensor([0],device=device),torch.tensor([[1,3]],device=device),torch.tensor([2],device=device),output='logits').detach()
  growth=grow_storage(model,opt,args)
  after=model(torch.tensor([0],device=device),torch.tensor([[1,3]],device=device),torch.tensor([2],device=device),output='logits').detach()
  if not torch.equal(before,after)or model.parameter_count()!=13397:raise RuntimeError('exact original storage growth changed old execution/P')
  restore_and_growth=dict(wall_seconds=time.perf_counter()-restore_started,CPU_seconds=time.process_time()-restore_CPU,includes_input_verification_and_dataset_load=True)
  indices=np.random.default_rng(np.random.SeedSequence([0,8,3,9027])).integers(0,len(data['initial']),size=(600,64),dtype=np.int64);draws=replay.sample_indices(3,600,16)
  if arraysha(indices)!=INDICES_PAYLOAD_SHA or arraysha(draws)!=REPLAY_PAYLOAD_SHA:raise ValueError('actual no-profile v3 batch arrays differ')
  for leaf,key,v in(('stage3-common-indices.npz','indices',indices),('stage3-replay-draws.npz','buffer_indices',draws)):
   with(rec.out/leaf).open('xb')as f:np.savez_compressed(f,**{key:v})
  parameters=list(model.named_parameters());names=[n for n,_ in parameters]
  if len(parameters)!=27:raise ValueError('all27 original parameters required')
  fb.allocate_gradient_buffers(model)
  grad_pointers={n:p.grad.data_ptr()for n,p in parameters};param_pointers={n:p.data_ptr()for n,p in parameters}
  def stable():
   if grad_pointers!={n:p.grad.data_ptr()if p.grad is not None else None for n,p in parameters}or param_pointers!={n:p.data_ptr()for n,p in parameters}:raise RuntimeError('captured NN/gradient storage replaced')
  def sync():torch.cuda.synchronize(device)
  def memory():return dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved(),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved())
  restore_peak=memory();static={k:torch.empty((64,48)if k=='tokens'else(64,),device=device,dtype=torch.long)for k in RAW}
  def assemble(step):
   tick=time.perf_counter();mixed=replay.mix(data,indices[step,:48],draws[step]);b,extent,active=old.batch(mixed,np.arange(64),device)
   if extent>48 or bool((b['lengths']>48).any()):raise ValueError('active train data cannot fit fixed48')
   sync();costs['batch_assembly_wall_seconds']+=time.perf_counter()-tick;costs['batch_H2D_payload_bytes']+=sum(v.numel()*v.element_size()for v in b.values())
   return b,extent,active
  def copy_static(b):
   tick=time.perf_counter();static['tokens'].zero_();static['tokens'][:,:b['tokens'].shape[1]].copy_(b['tokens'])
   for k in('initial','lengths','targets'):static[k].copy_(b[k])
   sync();costs['static_copy_wall_seconds']+=time.perf_counter()-tick;costs['static_D2D_payload_bytes']+=sum(v.numel()*v.element_size()for v in b.values());stable()
  first,first_extent,first_active=assemble(0);copy_static(first)
  def masked():return fb.masked_step(model,static['initial'],static['tokens'],static['lengths'],static['targets'],0)
  def gather_fixed(b):
   # autograd.grad exposes unused-parameter None without reassigning p.grad.
   for _,p in parameters:p.grad.zero_()
   loss,output,nll,binding=fb.objective(model,fb.original_forward_value,b['initial'],b['tokens'],b['lengths'],b['targets'],0)
   gradients=torch.autograd.grad(loss,tuple(p for _,p in parameters),allow_unused=True)
   if any(g is None for g in gradients):raise RuntimeError('original actual batch has unused parameter; None-vs-zero semantics cannot be hidden')
   with torch.no_grad():
    for (_,p),g in zip(parameters,gradients):p.grad.copy_(g)
   stable();return output.detach(),loss.detach(),nll.detach(),binding.detach()
  def snapshot(value):
   sync();return dict(output=value[0].cpu().clone(),loss=value[1].cpu().clone(),grads={n:p.grad.detach().cpu().clone()for n,p in parameters})
  def compare(left,right,leaf):
   raw=dict(reference_output=left['output'].numpy(),candidate_output=right['output'].numpy(),reference_loss=left['loss'].numpy(),candidate_loss=right['loss'].numpy(),gradient_names_json=np.frombuffer(json.dumps(names).encode(),dtype=np.uint8));record={}
   for i,n in enumerate(names):raw[f'reference_grad_{i:02d}']=left['grads'][n].numpy();raw[f'candidate_grad_{i:02d}']=right['grads'][n].numpy()
   path=rec.out/leaf
   with path.open('xb')as f:np.savez_compressed(f,**raw)
   record.update(raw_file=leaf,raw_sha256=sha(path),output_max_abs=float((left['output']-right['output']).abs().max()),gradients={})
   if not torch.allclose(left['output'],right['output'],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL)or not torch.allclose(left['loss'],right['loss'],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL):raise RuntimeError('fixed actual output/loss tolerance failed')
   for n in names:
    l,r=left['grads'][n],right['grads'][n];close=bool(torch.allclose(l,r,atol=fb.GRAD_ATOL,rtol=fb.GRAD_RTOL));record['gradients'][n]=dict(max_abs=float((l-r).abs().max()),allclose=close)
    if not close or not bool(torch.isfinite(r).all()):raise RuntimeError('fixed all27-gradient contract failed: '+n)
   return record
  phase='warmup_capture';tick=time.perf_counter();cpu=time.process_time();setup_rng=old.rng_state();setup_model={n:v.detach().cpu().clone()for n,v in model.state_dict().items()};setup_opt=copy.deepcopy(opt.state_dict());setup_replay=copy.deepcopy(replay.state_dict())
  model.train();value=gather_fixed(first);gold=snapshot(value);del value;costs['diagnostic_gather_FB_calls']+=1;costs['diagnostic_original_forward_MAC_proxy']+=model.resource_report(64,first_extent,active_steps=first_active)['proxy_forward_macs']
  side=torch.cuda.Stream();side.wait_stream(torch.cuda.current_stream());t=time.perf_counter();c=time.process_time()
  with torch.cuda.stream(side):
   for _ in range(3):rec.check();value=masked();del value;costs['warmup_FB_calls']+=1
  torch.cuda.current_stream().wait_stream(side);sync();warmup=dict(wall_seconds=time.perf_counter()-t,CPU_seconds=time.process_time()-c,actual_FB_calls=3)
  pool=torch.cuda.graph_pool_handle();graph=torch.cuda.CUDAGraph(keep_graph=True);torch.cuda.reset_peak_memory_stats();memory_before=memory();t=time.perf_counter();c=time.process_time()
  with torch.cuda.graph(graph,pool=pool,stream=side,capture_error_mode='global'):graph_outputs=masked()
  sync();capture=dict(wall_seconds=time.perf_counter()-t,CPU_seconds=time.process_time()-c,memory_before=memory_before,memory_after=memory(),private_pool=True,pool_bytes_directly_observed=None,described_FB_calls=1,actual_GPU_FB_during_capture=0,grad_zero_inside=True,Adam_or_clip_inside=False);costs['capture_described_FB_calls']=1
  t=time.perf_counter();c=time.process_time();graph.instantiate();sync();capture['instantiate_wall_seconds']=time.perf_counter()-t;capture['instantiate_CPU_seconds']=time.process_time()-c
  stable();graph.replay();sync();candidate=snapshot(graph_outputs);costs['diagnostic_graph_FB_calls']+=1;costs['diagnostic_graph_forward_MAC_proxy']+=model.resource_report(64,48,active_steps=64*48)['proxy_forward_macs'];admission=compare(gold,candidate,'admission-all27-gradients.npz');del gold,candidate
  if not old.tensor_state_equal(model.state_dict(),setup_model)or not old.nested_state_equal(opt.state_dict(),setup_opt)or not old.nested_state_equal(old.rng_state(),setup_rng)or not old.nested_state_equal(replay.state_dict(),setup_replay):raise RuntimeError('0SGD setup changed NN/Adam/RNG/replay')
  setup=dict(wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-cpu,warmup=warmup,capture=capture,admission=admission,NN_Adam_RNG_replay_unchanged=True,restore_peak=restore_peak,grad_buffers_bytes=sum(p.grad.numel()*p.grad.element_size()for _,p in parameters),static_input_bytes=sum(v.numel()*v.element_size()for v in static.values()))
  del setup_model,setup_opt,setup_replay
  rec.emit('actual_graph_setup',**setup,indices_array_sha256=arraysha(indices),replay_array_sha256=arraysha(draws),growth=growth)
  def diagnostic(step):
   tick=time.perf_counter();cpu=time.process_time();saved_rng=old.rng_state();nn={n:v.detach().cpu().clone()for n,v in model.state_dict().items()};adam=copy.deepcopy(opt.state_dict());replay_state=copy.deepcopy(replay.state_dict());copy_static(first);model.train()
   value=gather_fixed(first);left=snapshot(value);del value;graph.replay();sync();right=snapshot(graph_outputs)
   row=compare(left,right,f'diagnostic-step{step:03d}-all27-gradients.npz');stable()
   if not old.tensor_state_equal(model.state_dict(),nn)or not old.nested_state_equal(opt.state_dict(),adam)or not old.nested_state_equal(old.rng_state(),saved_rng)or not old.nested_state_equal(replay.state_dict(),replay_state):raise RuntimeError('0SGD diagnostic changed NN/Adam/RNG/replay')
   costs['diagnostic_gather_FB_calls']+=1;costs['diagnostic_graph_FB_calls']+=1;costs['diagnostic_original_forward_MAC_proxy']+=model.resource_report(64,first_extent,active_steps=first_active)['proxy_forward_macs'];costs['diagnostic_graph_forward_MAC_proxy']+=model.resource_report(64,48,active_steps=64*48)['proxy_forward_macs'];elapsed=time.perf_counter()-tick;costs['diagnostic_wall_seconds']+=elapsed;costs['diagnostic_CPU_seconds']+=time.process_time()-cpu;rec.emit('actual_fixed_TRAIN_gradient_diagnostic',step=step,wall_seconds=elapsed,SGD_updates=0,**row)
  curves=[];loop_started=time.perf_counter();loop_cpu=time.process_time();events=[torch.cuda.Event(enable_timing=True)for _ in range(4)]
  for step in range(601):
   rec.check();phase=f'train_step{step}'
   if step%100==0:
    metrics={}
    for s in range(4):
     metrics[s],timing=old.evaluate(model,datasets[s]['valid'],args,device,rec,modes=('soft',));costs['evaluation_wall_seconds']+=timing['wall_seconds'];costs['evaluation_forward_MAC_proxy']+=timing['forward_MAC_proxy'];costs['evaluation_forward_rows']+=len(datasets[s]['valid']['initial'])
    stable();row=dict(stage=3,step=step,valid=metrics);curves.append(row);rec.emit('actual_VALID_curve',**row)
    if step:diagnostic(step)
   if step==600:break
   b,extent,active=(first,first_extent,first_active)if step==0 else assemble(step);copy_static(b);model.train();stable()
   start,fbend,clipend,end=events;sync();tick=time.perf_counter();cpu=time.process_time();start.record();graph.replay();fbend.record()
   if not bool(torch.isfinite(graph_outputs[1])):raise FloatingPointError('nonfinite actual graph loss')
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);clipend.record();opt.step();end.record();sync()
   if not bool(torch.isfinite(torch.cat([p.detach().reshape(-1)for _,p in parameters])).all()):raise FloatingPointError('nonfinite actual NN after eagerAdam')
   stable();elapsed=time.perf_counter()-tick;completed=step+1;padded_proxy=model.resource_report(64,48,active_steps=3072)['proxy_forward_macs'];original_proxy=model.resource_report(64,extent,active_steps=active)['proxy_forward_macs']
   for k,v in(('SGD_updates',1),('training_wall_seconds',elapsed),('training_CPU_seconds',time.process_time()-cpu),('CUDA_stream_elapsed_seconds',start.elapsed_time(end)/1000.),('graph_FB_CUDA_seconds',start.elapsed_time(fbend)/1000.),('padded_forward_MAC_proxy',padded_proxy),('original_active_forward_MAC_proxy',original_proxy),('active_token_steps',active),('padded_token_steps',3072),('current_training_label_reads',48),('old_replay_training_label_reads',16),('anchor_target_encoder_lookups',64),('anchor_coordinate_loss_elements',2048),('actual_training_graph_FB_calls',1)):costs[k]+=v
   rec.emit('actual_update',stage=3,step=completed,terminal_NLL=float(graph_outputs[2]),value_interface_loss=float(graph_outputs[3]),total_loss=float(graph_outputs[1]),batch_accuracy=float((graph_outputs[0].argmax(-1)==static['targets']).float().mean()),preclip_gradient=float(norm),training_wall_seconds=elapsed,padded_forward_MAC_proxy=padded_proxy,original_active_forward_MAC_proxy=original_proxy,active_token_steps=active,padded_token_steps=3072,graph_FB_CUDA_seconds=start.elapsed_time(fbend)/1000.,clip_CUDA_seconds=fbend.elapsed_time(clipend)/1000.,Adam_CUDA_seconds=clipend.elapsed_time(end)/1000.)
   if step:del b
  costs['loop_including_VALID_diagnostics_batch_wall_seconds']=time.perf_counter()-loop_started;costs['loop_including_VALID_diagnostics_batch_CPU_seconds']=time.process_time()-loop_cpu
  if completed!=600 or not old.nested_state_equal(old.rng_state(),setup_rng):raise RuntimeError('exact600/RNG-neutral trajectory required')
  stable();phase='checkpoint_before_maintenance';used[3]=indices[:,:48]
  def save(leaf,maintenance_complete):
   tick=time.perf_counter();path=rec.out/leaf;payload=dict(schema='factorized-graph-stage3-training-v1',stage=3,step=completed,stage_complete=completed==600,maintenance_complete=maintenance_complete,formal_Primary=False,args=vars(args),block_checkpoint=model.checkpoint(),model_state={n:v.detach().cpu().clone()for n,v in model.state_dict().items()},optimizer=opt.state_dict(),rng=old.rng_state(),replay_state=replay.state_dict(),source_sha256=manifest['source_sha256'],costs=costs,execution_policy='exact_fcba_B64L48_masked_graph_FB_eager_clip1_Adam003',used_current_indices=used[3].copy(),indices_payload_sha256=arraysha(indices),replay_draws_payload_sha256=arraysha(draws),growth=growth)
   with path.open('xb')as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
   return dict(file=leaf,sha256=sha(path),bytes=path.stat().st_size,write_wall_seconds=time.perf_counter()-tick)
  safety=save('trained-before-maintenance.pt',False)
  phase='maintenance';maintenance_data=dict(data,tokens=data['tokens'][:,:48]);train_sources[3]=maintenance_data
  if np.any(data['lengths']>48)or np.any(data['tokens'][:,48:]):raise ValueError('maintenance padding view cannot drop active information')
  rec.emit('replay_storage_padding_view',stage=3,original_columns=128,stored_columns=48,all_removed_entries_zero=True,no_training_change=True,extra_copy=False)
  maintenance=replay.consume(maintenance_data,used[3],3,emit=rec.emit);source_audit=replay.validate_sources(train_sources,used)
  if replay.seen!=124800 or replay.draws!=28800 or replay.digest()!='6405bee32c7b0daa655bac0a8e54e91c44c6516b8442d66560b5e1a2a3323641':raise RuntimeError('actual committed replay counters differ')
  checkpoint=save('factorized-graph-stage3-step600.pt',True)
  # Retire the sole graph before constructing an independent reload model.
  sync();graph.reset();graph=None;del graph_outputs,pool,side;pool=side=None;stable()
  phase='reload';saved_rng=old.rng_state();tick=time.perf_counter();loaded=torch.load(rec.out/checkpoint['file'],map_location='cpu',weights_only=False);other=core.ContinuousPrimitiveBlock.from_checkpoint(loaded['block_checkpoint']).to(device);other_opt=torch.optim.Adam(other.parameters(),lr=.003);other_opt.load_state_dict(loaded['optimizer']);other_replay=Replay128.from_state_dict(loaded['replay_state'])
  if not old.tensor_state_equal(other.state_dict(),model.state_dict())or not old.nested_state_equal(other_opt.state_dict(),opt.state_dict())or not old.nested_state_equal(other_replay.state_dict(),replay.state_dict())or not old.nested_state_equal(loaded['rng'],saved_rng):raise RuntimeError('real final NN/Adam/RNG/replay reload differs')
  check_batch,_,_=old.batch(datasets[3]['valid'],np.arange(16),device);model.eval();other.eval()
  with torch.no_grad():
   for mode in('soft','native'):
    if not torch.equal(old.forward(model,check_batch,args,mode),old.forward(other,check_batch,args,mode)):raise RuntimeError('actual reloaded execution differs')
  del other,other_opt,other_replay,loaded;old.restore_rng(saved_rng);reload=dict(exact_NN_Adam_RNG_replay_forward=True,RNG_factory_neutral=True,wall_seconds=time.perf_counter()-tick)
  resources=old.resources(model,opt,device,replay);report=dict(stage=3,costs=costs,checkpoint=checkpoint,reload=reload,resources=resources,curves=curves,growth=growth,replay_maintenance=maintenance);reports.append(report)
  rec.json('completed-stage3-before-TEST.json',dict(report=report,test_data_read=False));rec.json('ALL_TRAINING_FROZEN_BEFORE_TEST.json',dict(reports=reports,no_more_updates=True,TEST_read=False,checkpoint=checkpoint,actual_new_updates=600))
  phase='frozen_TEST';frozen={n:v.detach().cpu().clone()for n,v in model.state_dict().items()};tests={};test_costs=[]
  for s in range(4):
   tests[s]={}
   for split in('test_iid','test_long','test_unseen'):
    testdata=load(s,split);metrics,timing=old.evaluate(model,testdata,args,device,rec,modes=('soft','native'));tests[s][split]=metrics;test_costs.append(timing)
  tick=time.perf_counter();inference=old.inference_report(model,datasets[3]['valid'],args,device,rec);inference_seconds=time.perf_counter()-tick
  if not old.tensor_state_equal(model.state_dict(),frozen):raise RuntimeError('frozen evaluation changed NN')
  stable();result=dict(status='ACTUAL_FACTORIZED_GRAPH_STAGE3_600_COMPLETE',formal_Primary=False,anchor=0,actual_new_updates_this_process=600,historical_updates_shared=1800,actual_training_updates_retained_lineage=2400,source_sha256=manifest['source_sha256'],parent_stage2_checkpoint_sha256=RESUME_SHA,reports=reports,tests=tests,resources=old.resources(model,opt,device,replay),test_costs=test_costs,inference=inference,inference_total_seconds=inference_seconds,setup=setup,costs=costs,checkpoint=checkpoint,premaintenance_safety_checkpoint=safety,reload=reload,maintenance=maintenance,replay_source_audit=source_audit,all27_grad_storage_stable=True,restore_and_growth=restore_and_growth,training_mask_select_scalars=600*64*48*432,training_gradient_zero_scalar_writes=600*13397,NN_unchanged_by_evaluation=True,teacher_or_profile_loss=False,accuracy_or_speed_gate=False,actual_graph_count=1,no_compiler_exception=True,driver_wall_seconds=time.perf_counter()-rec.started,driver_CPU_seconds=time.process_time()-whole_cpu,full_training_FLOPs=None,provider_effective_compute=None,cost_scope='allsetup/capture/pool/warmup/diagnostics/padding/eagerclip+Adam/copy/eval/restore/storage/maintenance charged; nested times not additive; old1800updates paid once, other historical noaux/profile experiments remain separate; oldsoft-only actual_forward_examples hardcodes2 so evaluation_forward_rows is derived from actual input rows')
  rec.json('result.json',result);rec.emit('completed',actual_updates=600,all27_grad_storage_stable=True);print(json.dumps(dict(status=result['status'],P=model.parameter_count(),new_updates=600)))
 except BaseException as e:
  rec.json('failure.json',dict(status='FAILED_OR_BOUNDED_FACTORIZED_GRAPH_STAGE3',phase=phase,completed_updates=completed,costs=costs,error=repr(e),traceback=traceback.format_exc(),formal_Primary=False))
  if model is not None:
   with(rec.out/'partial-actual-NN.pt').open('xb')as f:torch.save(dict(model_state=model.state_dict(),completed_updates=completed,source_sha256=manifest.get('source_sha256')if 'manifest'in locals()else None),f);f.flush();os.fsync(f.fileno())
  raise
 finally:
  if graph is not None:graph.reset()
  rec.stream.close()
if __name__=='__main__':main()
