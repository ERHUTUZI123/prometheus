"""OWN typed stage2 continuation: paired vs identical-pool shuffled TRAIN.

One --sampling flag selects the immutable input index array ONLY. Restore the
same NN/Adam/all RNG/Replay128, grow token15 once, run exactly600 soft NLL
updates. No old1800 rerun, teacher, extra loss or score-dependent extension.
"""
import argparse,copy,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
import continuous_primitive_block as core
import owned_parent_driver as old
import owned_full_driver as full
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage
from typed_memory_actions_graph_engine import GraphEngine

RAW=('initial','tokens','lengths','targets')
SAMPLING_KEYS={'paired':'indices','same_pool_shuffle':'indices_same_pool_shuffled'}
REPLAY_PAYLOAD_SHA='c00780e9f7d57f4c26e3e472da8563b2aaab65d00926aaa55b0a4581fbfa9ab1'
STAGE2_REPLAY_SHA='440344f6cd778e8443081cdeafa95847bf0d23aa825f97590c55436b02a7f470'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arraysha(v):return hashlib.sha256(v.tobytes(order='C')).hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in('inputs','out'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--sampling',choices=tuple(SAMPLING_KEYS),required=True);p.add_argument('--max-seconds',type=float,default=850);a=p.parse_args()
 if not 1<=a.max_seconds<=850:raise ValueError('same850science/900outer')
 rec=old.Recorder(a.out,a.max_seconds);device=torch.device('cuda:0');whole_cpu=time.process_time();model=opt=engine=None;completed=0;phase='restore';restore_started=time.perf_counter();restore_cpu=time.process_time();costs=dict(SGD_updates=0,training_wall_seconds=0.,training_CPU_seconds=0.,CUDA_stream_elapsed_seconds=0.,padded_forward_MAC_proxy=0,active_forward_MAC_proxy=0,active_token_steps=0,padded_token_steps=0,current_training_label_reads=0,old_replay_training_label_reads=0,batch_assembly_wall_seconds=0.,batch_H2D_bytes=0,evaluation_wall_seconds=0.,evaluation_forward_MAC_proxy=0,evaluation_rows=0,diagnostic_wall_seconds=0.,checkpoint_wall_seconds=0.,full_training_FLOPs=None)
 try:
  if not torch.cuda.is_available():raise RuntimeError('sameauthorizedCUDA only')
  torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,0)
  manifest=json.loads((a.inputs/'manifest.json').read_text())
  if manifest.get('schema')!='typed-paired-tail-stage3-inputs-v1'or manifest['sampling']!=a.sampling:raise ValueError('fixedsampler/source manifest required')
  for leaf,digest in manifest['source_sha256'].items():
   if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen source changed: '+leaf)
  args=argparse.Namespace(arm='dynamic',states=8,seed=0,lr=.003,block_state='continuous',block_width=32,primitive_rank=8,memory_slots=8,slots=1,block_training='soft_only',replay_capacity=128,replay_per_batch=16,stages=4,steps=600,organization=core.POLICY)
  def load(s,split):
   spec=manifest['files'][f'{s}:{split}'];path=a.inputs/spec['leaf']
   if sha(path)!=spec['sha256']:raise ValueError('same original raw input required')
   with np.load(path,allow_pickle=False)as z:
    if set(z.files)!=set(RAW):raise ValueError('rawfields only')
    data={k:z[k]for k in RAW}
   n=len(data['initial']);width=data['tokens'].shape[1];active=np.arange(width)[None]<data['lengths'][:,None]
   if any(v.dtype!=np.int64 for v in data.values())or any(data[k].shape!=(n,)for k in('initial','lengths','targets'))or np.any((data['lengths']<0)|(data['lengths']>width))or np.any((data['initial']<0)|(data['initial']>=8))or np.any((data['targets']<0)|(data['targets']>=8))or np.any(active&((data['tokens']<0)|(data['tokens']>=(16 if s==3 else 15)))):raise ValueError('rawschema/range')
   rec.emit('actual_input_opened',stage=s,split=split,sha256=spec['sha256'],rows=n,array_bytes=sum(v.nbytes for v in data.values()));return data
  datasets={s:{k:load(s,k)for k in('train','valid')}for s in range(4)};train_sources={s:datasets[s]['train']for s in range(4)};data=train_sources[3]
  pair=manifest['pairs'];pair_path=a.inputs/pair['leaf'];pair_result_path=a.inputs/pair['result_leaf']
  if sha(pair_path)!=pair['sha256']or sha(pair_result_path)!=pair['result_sha256']:raise ValueError('actual common pair formation changed')
  pair_result=json.loads(pair_result_path.read_text())
  if pair_result['status']!='CPU_PAIRED_TRAIN_INDICES_GENERATED_PASS'or not pair_result['same_pool_row_exposures_exact']or pair_result['target_array_reads']!=0 or pair_result['VALID_or_TEST_read']or pair_result['eligible_rows']!=4096 or pair_result['eligible_groups']!=2048:raise ValueError('publicTRAIN-only pair proof required')
  with np.load(pair_path,allow_pickle=False)as z:paired=z['indices'].copy();shuffled=z['indices_same_pool_shuffled'].copy();indices=z[SAMPLING_KEYS[a.sampling]].copy()
  if any(x.shape!=(600,64)or x.dtype!=np.int64 or np.any((x<0)|(x>=4096))for x in(paired,shuffled,indices))or not np.array_equal(np.bincount(paired[:,:48].reshape(-1),minlength=4096),np.bincount(shuffled[:,:48].reshape(-1),minlength=4096)):raise ValueError('same pool/shape/range mismatch')
  old_indices=np.random.default_rng(np.random.SeedSequence([0,8,3,9027])).integers(0,4096,size=(600,64),dtype=np.int64)
  if not np.array_equal(paired[:,48:],old_indices[:,48:])or not np.array_equal(shuffled[:,48:],old_indices[:,48:]):raise ValueError('unused tail16 changed')
  if arraysha(indices)!=pair_result['schedule']['arrays'][SAMPLING_KEYS[a.sampling]]['sha256_C_payload']:raise ValueError('actual selected index payload differs')
  resume=manifest['resume'];resume_path=a.inputs/resume['leaf']
  if sha(resume_path)!=resume['sha256']or resume_path.stat().st_size!=resume['bytes']:raise ValueError('exact OWN stage2 checkpoint required')
  payload=torch.load(resume_path,map_location='cpu',weights_only=False)
  if payload.get('schema')!='typed-memory-actions-fourstage-graph-training-v1'or payload.get('stage')!=2 or payload.get('step')!=600 or not payload.get('stage_complete')or not payload.get('maintenance_complete')or payload['source_sha256']!=resume['source_sha256']:raise ValueError('completedOWNstage2 source/policy required')
  for key,value in vars(args).items():
   if payload['args'].get(key)!=value:raise ValueError('parent optimizer/model/data configuration changed: '+key)
  model=core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint'],expected_action_policy=core.POLICY).to(device);opt=torch.optim.Adam(model.parameters(),lr=.003);opt.load_state_dict(payload['optimizer']);replay=Replay128.from_state_dict(payload['replay_state'])
  if model.parameter_count()!=10975 or len(list(model.parameters()))!=20 or not old.tensor_state_equal(model.state_dict(),payload['model_state'])or not old.nested_state_equal(opt.state_dict(),payload['optimizer'])or not old.nested_state_equal(replay.state_dict(),payload['replay_state']):raise ValueError('realNN/Adam/replay restore mismatch')
  if replay.size!=128 or replay.width!=48 or replay.seen!=96000 or replay.draws!=19200 or replay.digest()!=STAGE2_REPLAY_SHA or any(float(v['step'])!=1800 for v in opt.state.values()):raise ValueError('same1800step/replay boundary required')
  used={}
  for s in range(3):
   original=np.random.default_rng(np.random.SeedSequence([0,8,s,9027])).integers(0,len(train_sources[s]['initial']),size=(600,64),dtype=np.int64);used[s]=original if s==0 else original[:,:48]
  if not np.array_equal(payload['used_current_indices'],used[2]):raise ValueError('actual historicalstage2 exposure source differs')
  prior_replay_source_audit=replay.validate_sources(train_sources,used);old.restore_rng(payload['rng'])
  if not old.nested_state_equal(old.rng_state(),payload['rng']):raise ValueError('actual allCPU/CUDA/NumPy/Python RNG restore mismatch')
  parent_restored_theta=full.NNhash(model);parent_rng=copy.deepcopy(payload['rng']);parent_replay_sha=replay.digest();del payload
  probe={k:torch.tensor(v,device=device)for k,v in dict(initial=[0,3],tokens=[[1,3],[2,3]],lengths=[2,2]).items()}
  with torch.no_grad():before=model(**probe,output='logits').clone()
  growth=grow_storage(model,opt,args)
  with torch.no_grad():after=model(**probe,output='logits')
  if model.parameter_count()!=10991 or not torch.equal(before,after):raise ValueError('original one token-row growth changed old execution/P')
  post_growth_theta=full.NNhash(model);new_row_sha=arraysha(model.token_embedding.weight[-1].detach().cpu().numpy());growth_rng=old.rng_state();draws=replay.sample_indices(3,600,16)
  if arraysha(draws)!=REPLAY_PAYLOAD_SHA:raise ValueError('original Replay600x16 draws changed')
  local_index_payload=dict(paired_bytes=paired.nbytes,same_pool_shuffled_bytes=shuffled.nbytes,selected_copy_bytes=indices.nbytes,original_unused_tail_source_array_bytes=old_indices.nbytes,replay_draws_bytes=draws.nbytes,prior_used_index_backing_bytes=sum((v.base.nbytes if isinstance(v.base,np.ndarray)else v.nbytes)for v in used.values()))
  restore=dict(local_index_array_payload=local_index_payload,local_index_array_payload_total_bytes=sum(local_index_payload.values()),local_index_scope='retained localpaired/shuffled/selected/original/priorused backing arrays+draws, additional tosharedgenerator arrays; CPUpeak alsoincludespayload/readtemporaries',wall_seconds=time.perf_counter()-restore_started,CPU_seconds=time.process_time()-restore_cpu,parent_checkpoint_sha256=resume['sha256'],parent_restored_theta_sha256=parent_restored_theta,post_growth_initial_theta_sha256=post_growth_theta,new_token_row_sha256=new_row_sha,all_RNG_restored=True,parent_replay_sha256=parent_replay_sha,replay_source_audit=prior_replay_source_audit,historical_updates_shared=1800,old_updates_rerun=0)
  rec.emit('actual_restore_and_growth',**restore,growth=growth,sampling=a.sampling,pair_schedule_sha256=pair['sha256'],selected_indices_payload_sha256=arraysha(indices),Replay_draws_payload_sha256=arraysha(draws))
  for leaf,key,value in(('stage3-common-indices.npz','indices',indices),('stage3-replay-draws.npz','buffer_indices',draws)):
   with(rec.out/leaf).open('xb')as f:np.savez_compressed(f,**{key:value})
  if data['tokens'].shape!=(4096,128)or np.any(data['lengths']>48)or np.any(data['tokens'][:,48:]):raise ValueError('no active TRAIN may be truncated')
  def assemble(step):
   tick=time.perf_counter();mixed=replay.mix(data,indices[step,:48],draws[step]);b,extent,active=old.batch(mixed,np.arange(64),device);torch.cuda.synchronize();costs['batch_assembly_wall_seconds']+=time.perf_counter()-tick;costs['batch_H2D_bytes']+=sum(v.numel()*v.element_size()for v in b.values())
   if extent>48:raise ValueError('sameB64L48 bucket exceeded')
   return b,extent,active
  first,first_extent,first_active=assemble(0);before_admission_replay=copy.deepcopy(replay.state_dict());engine=GraphEngine(model,opt,48,first,rec.out,rec.check,3)
  if not old.nested_state_equal(before_admission_replay,replay.state_dict()):raise ValueError('0SGDadmission changed replay')
  rec.emit('actual_graph_capture',stage=3,graph=engine.report());curves=[];loop=time.perf_counter();loop_cpu=time.process_time()
  for step in range(601):
   phase=f'train_step{step}';rec.check()
   if step%100==0:
    metrics={};predictions={}
    for s in range(4):
     metrics[s],timing=(full.evaluate_saved(model,datasets[s]['valid'],args,device,rec,f'stage3-valid-task{s}-predictions.npz',modes=('soft',))if step==600 else old.evaluate(model,datasets[s]['valid'],args,device,rec,modes=('soft',)));costs['evaluation_wall_seconds']+=timing['wall_seconds'];costs['evaluation_forward_MAC_proxy']+=timing['forward_MAC_proxy'];costs['evaluation_rows']+=len(datasets[s]['valid']['initial'])
     if 'predictions'in timing:predictions[s]=timing['predictions']
    row=dict(stage=3,step=step,valid=metrics,saved_VALID_predictions=predictions);curves.append(row);rec.emit('actual_VALID_curve',**row);engine.stable()
    if step:
     check=engine.diagnostic(3,step);costs['diagnostic_wall_seconds']+=check['wall_seconds'];rec.emit('actual_all20_current_NN_gradient_check',**check)
   if step==600:break
   b,extent,active=(first,first_extent,first_active)if step==0 else assemble(step);model.train();torch.cuda.synchronize();tick=time.perf_counter();cpu=time.process_time();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();out,loss=engine.run(b)
   if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite terminalNLL')
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();end.record();torch.cuda.synchronize()
   if not bool(torch.isfinite(torch.cat([v.detach().reshape(-1)for v in model.parameters()])).all()):raise FloatingPointError('nonfiniteNNafterAdam')
   engine.stable();elapsed=time.perf_counter()-tick;completed=step+1;padded=model.resource_report(64,48,active_steps=3072)['proxy_forward_macs'];activeMAC=model.resource_report(64,extent,active_steps=active)['proxy_forward_macs']
   for k,v in(('SGD_updates',1),('training_wall_seconds',elapsed),('training_CPU_seconds',time.process_time()-cpu),('CUDA_stream_elapsed_seconds',start.elapsed_time(end)/1000.),('padded_forward_MAC_proxy',padded),('active_forward_MAC_proxy',activeMAC),('active_token_steps',active),('padded_token_steps',3072),('current_training_label_reads',48),('old_replay_training_label_reads',16)):costs[k]+=v
   rec.emit('actual_update',stage=3,step=completed,terminal_NLL=float(loss),preclip_gradient=float(norm),batch_accuracy=float((out.argmax(-1)==b['targets']).float().mean()),training_wall_seconds=elapsed,active_token_steps=active,padded_token_steps=3072,padded_forward_MAC_proxy=padded,active_forward_MAC_proxy=activeMAC)
   if step:del b
  costs.update(loop_including_VALID_diagnostics_batch_wall_seconds=time.perf_counter()-loop,loop_including_VALID_diagnostics_batch_CPU_seconds=time.process_time()-loop_cpu)
  if completed!=600 or replay.draws!=28800 or not old.nested_state_equal(old.rng_state(),growth_rng):raise ValueError('exact600/drawcount/RNG-neutral training required')
  graph_report=engine.report();engine.close();engine=None;used[3]=indices[:,:48]
  def save(leaf,maintenance_complete):
   tick=time.perf_counter();path=rec.out/leaf;payload=dict(schema='typed-paired-tail-stage3-training-v1',sampling=a.sampling,stage=3,step=600,stage_complete=True,maintenance_complete=maintenance_complete,formal_Primary=False,args=vars(args),parent_checkpoint_sha256=resume['sha256'],post_growth_initial_theta_sha256=post_growth_theta,new_token_row_sha256=new_row_sha,block_checkpoint=model.checkpoint(),model_state={n:v.detach().cpu().clone()for n,v in model.state_dict().items()},optimizer=opt.state_dict(),rng=old.rng_state(),replay_state=replay.state_dict(),used_current_indices=used[3].copy(),source_sha256=manifest['source_sha256'],costs=costs,pair_schedule_sha256=pair['sha256'],indices_payload_sha256=arraysha(indices),replay_draws_payload_sha256=arraysha(draws))
   with path.open('xb')as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
   row=dict(file=leaf,sha256=sha(path),bytes=path.stat().st_size,write_wall_seconds=time.perf_counter()-tick);costs['checkpoint_wall_seconds']+=row['write_wall_seconds'];return row
  phase='safety_before_maintenance';safety=save('typed-paired-tail-trained-before-maintenance.pt',False);maintenance_data=dict(data,tokens=data['tokens'][:,:48]);train_sources[3]=maintenance_data;rec.emit('replay_storage_padding_view',stage=3,original_columns=128,stored_columns=48,removed_entries_zero=True,training_raw_unchanged=True)
  maintenance=replay.consume(maintenance_data,used[3],3,emit=rec.emit);replay_source_audit=replay.validate_sources(train_sources,used)
  if replay.seen!=124800 or replay.draws!=28800:raise ValueError('same Replay exposure boundary')
  final=save('typed-paired-tail-stage3-step600.pt',True);phase='reload';saved_rng=old.rng_state();tick=time.perf_counter();loaded=torch.load(rec.out/final['file'],map_location='cpu',weights_only=False);other=core.ContinuousPrimitiveBlock.from_checkpoint(loaded['block_checkpoint'],expected_action_policy=core.POLICY).to(device);other_opt=torch.optim.Adam(other.parameters(),lr=.003);other_opt.load_state_dict(loaded['optimizer']);other_replay=Replay128.from_state_dict(loaded['replay_state'])
  if not old.tensor_state_equal(model.state_dict(),other.state_dict())or not old.nested_state_equal(opt.state_dict(),other_opt.state_dict())or not old.nested_state_equal(replay.state_dict(),other_replay.state_dict())or not old.nested_state_equal(saved_rng,loaded['rng']):raise ValueError('actualNN/Adam/RNG/Replay reload mismatch')
  check_batch,_,_=old.batch(datasets[3]['valid'],np.arange(16),device);model.eval();other.eval()
  with torch.no_grad():
   for mode in('soft','native'):
    if not torch.equal(old.forward(model,check_batch,args,mode),old.forward(other,check_batch,args,mode)):raise ValueError('reload forward mismatch')
  del other,other_opt,other_replay,loaded;old.restore_rng(saved_rng);reload=dict(exact_NN_Adam_RNG_replay_forward=True,RNG_factory_neutral=True,wall_seconds=time.perf_counter()-tick)
  rec.json('ALL_TRAINING_FROZEN_BEFORE_TEST.json',dict(no_more_updates=True,TEST_read=False,checkpoint=final,actual_new_updates=600,historical_parent_updates=1800,sampling=a.sampling));phase='frozen_TEST';frozen=full.NNhash(model);tests={};test_costs=[]
  for s in range(4):
   tests[s]={}
   for split in('test_iid','test_long','test_unseen'):
    td=load(s,split);tests[s][split],timing=full.evaluate_saved(model,td,args,device,rec,f'final-task{s}-{split}-predictions.npz');test_costs.append(timing)
  tick=time.perf_counter();inference=old.inference_report(model,datasets[3]['valid'],args,device,rec);inference_seconds=time.perf_counter()-tick
  if full.NNhash(model)!=frozen:raise ValueError('frozenTEST/inference changedNN')
  result=dict(status='ACTUAL_TYPED_PAIRED_TAIL_STAGE3_600_COMPLETE',formal_Primary=False,sampling=a.sampling,actual_new_updates_this_process=600,historical_updates_shared=1800,old_updates_rerun=0,parent_stage2_checkpoint_sha256=resume['sha256'],parent_run=resume['parent_run'],restore_and_growth=restore,post_growth_initial_theta_sha256=post_growth_theta,new_token_row_sha256=new_row_sha,pair_schedule_sha256=pair['sha256'],pair_formation_actual=pair_result,selected_indices_payload_sha256=arraysha(indices),Replay_draws_payload_sha256=arraysha(draws),source_sha256=manifest['source_sha256'],parent_source_sha256=resume['source_sha256'],setup=graph_report,costs=costs,curves=curves,growth=growth,checkpoint=final,premaintenance_safety_checkpoint=safety,reload=reload,replay_maintenance=maintenance,replay_source_audit=replay_source_audit,resources=old.resources(model,opt,device,replay),tests=tests,test_costs=test_costs,inference=inference,inference_total_seconds=inference_seconds,all20_grad_storage_stable=True,actual_graph_capture_count=1,NN_unchanged_after_TEST=True,accuracy_or_speed_gate=False,full_training_FLOPs=None,provider_effective_compute=None,driver_wall_seconds=time.perf_counter()-rec.started,driver_CPU_seconds=time.process_time()-whole_cpu,cost_scope='oneOWNstage2restore/1800historypaidnotrerun+new600; sharedpairformationchargedonceacross6tails; bothsamplerssamecurrentmultiset/differentbatchgroupingANDtemporalorder; setup/3warmups/7reference+7graphdiagnostics/onecapture/padding/clipAdam/copy/NNAdamReplay/carry+pool/VALID+TEST+predictions+inference/safety+final/reload/maintenance/source+raw/CPU+coldwall charged; no TESTselection/extraobjective; nestedtimesnotadditive')
  rec.json('result.json',result);rec.emit('completed',actual_new_updates=600,sampling=a.sampling);print(json.dumps(dict(status=result['status'],sampling=a.sampling,P=model.parameter_count())))
 except BaseException as error:
  rec.json('failure.json',dict(status='FAILED_OR_BOUNDED_TYPED_PAIRED_TAIL',sampling=a.sampling,phase=phase,completed_updates=completed,costs=costs,error=repr(error),traceback=traceback.format_exc()))
  if model is not None:
   with(rec.out/'partial-actual-NN.pt').open('xb')as f:torch.save(dict(model_state=model.state_dict(),optimizer=opt.state_dict()if opt else None,completed_updates=completed,sampling=a.sampling,source_sha256=manifest.get('source_sha256')if'manifest'in locals()else None),f);f.flush();os.fsync(f.fileno())
  raise
 finally:
  if engine is not None:engine.close()
  rec.stream.close()
if __name__=='__main__':main()
