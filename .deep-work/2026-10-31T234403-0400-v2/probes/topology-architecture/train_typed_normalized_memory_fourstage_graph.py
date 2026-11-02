"""Fresh one fixed A/B typed normalized-memory fourstage Graph arm.

Same original data/indices/Replay128/600x64/Adam003/clip1 terminal NLL only.
Two fixed captures, all20 current-NN gradient gates, no parent/checkpoint restore.
Frozen TEST outputs saved from the same one-shot evaluation forward passes.
"""
import argparse,copy,hashlib,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import continuous_primitive_block as core
import owned_parent_driver as old
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage
from typed_normalized_memory_graph_engine import GraphEngine
RAW=('initial','tokens','lengths','targets')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arraysha(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def NNhash(m):
 h=hashlib.sha256()
 for n,p in m.state_dict().items():
  v=p.detach().cpu().contiguous();h.update(json.dumps([n,list(v.shape),str(v.dtype)]).encode());h.update(v.numpy().tobytes())
 return h.hexdigest()

@torch.no_grad()
def evaluate_saved(model,data,args,device,rec,leaf,modes=('soft','native')):
 # Same old evaluation forward/loss/reduction, plus D2H/save of those outputs.
 # No second NN pass and no feedback/selection from saved predictions.
 was=model.training;model.eval();result={};torch.cuda.synchronize();began,cpu=time.perf_counter(),time.process_time();total_proxy=0;raw={}
 try:
  for mode in modes:
   correct=0;nll=0.;proxy=0;outputs=[]
   for offset in range(0,len(data['initial']),64):
    rec.check();ids=np.arange(offset,min(offset+64,len(data['initial'])));b,l,active=old.batch(data,ids,device);out=old.forward(model,b,args,mode)
    if not bool(torch.isfinite(out).all()):raise FloatingPointError('nonfinite actual evaluation output')
    correct+=int((out.argmax(-1)==b['targets']).sum());nll+=float(old.loss_value(out,b['targets'],args,'sum'));proxy+=old.work_proxy(model,args,len(ids),l,active,mode)['proxy_forward_macs'];outputs.append(out.detach().cpu().numpy().copy())
   result[mode]=dict(accuracy=correct/len(data['initial']),terminal_NLL=nll/len(data['initial']),rows=len(data['initial']),forward_MAC_proxy=proxy);total_proxy+=proxy;raw[mode+'_logp']=np.concatenate(outputs,axis=0)
  raw.update(row_index=np.arange(len(data['initial']),dtype=np.int64),targets=data['targets'],initial=data['initial'],lengths=data['lengths']);tick=time.perf_counter();path=rec.out/leaf
  with path.open('xb')as f:np.savez_compressed(f,**raw)
  receipt=dict(file=leaf,sha256=sha(path),bytes=path.stat().st_size,array_payload_bytes=sum(v.nbytes for v in raw.values()),write_wall_seconds=time.perf_counter()-tick,extra_NN_forward_passes=0,scope='samefrozenevaluationpasses; perrowall8logp/targets forindependentrecompute')
 finally:torch.cuda.synchronize();model.train(was)
 return result,dict(wall_seconds=time.perf_counter()-began,cpu_seconds=time.process_time()-cpu,forward_MAC_proxy=total_proxy,actual_forward_examples=len(data['initial'])*len(modes),predictions=receipt)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in('inputs','out'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--arm',choices=('A','B','AB'),required=True)
 p.add_argument('--max-seconds',type=float,default=850);a=p.parse_args();flags=dict(input_only_address=a.arm in('A','AB'),coverage_normalization=a.arm in('B','AB'));P15=10335 if flags['input_only_address']else 10975;P16=P15+16
 if not 1<=a.max_seconds<=850:raise ValueError('fixedinside850/outside900')
 rec=old.Recorder(a.out,a.max_seconds);device=torch.device("cuda:0");whole_cpu=time.process_time();model=opt=engine=None;stage=-1;completed=0;reports=[];graph_reports=[];used={};phase='source_data';actual_updates=0
 try:
  if not torch.cuda.is_available():raise RuntimeError('sameauthorizedCUDArequired; nofallback')
  torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,0)
  manifest=json.loads((a.inputs/'manifest.json').read_text())
  if manifest.get('schema')!='typed-normalized-memory-fourstage-inputs-v1'or manifest['architecture_policy']!=core.POLICY:raise ValueError('fixedfreshmode/source manifest required')
  if manifest['arm']!=a.arm or manifest['architecture_flags']!=flags:raise ValueError('prospective fixed A/B arm mismatch')
  for leaf,digest in manifest['source_sha256'].items():
   if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen sourcechanged: '+leaf)
  args=argparse.Namespace(arm='dynamic',states=8,seed=0,lr=.003,block_state='continuous',block_width=32,primitive_rank=8,memory_slots=8,slots=1,block_training='soft_only',replay_capacity=128,replay_per_batch=16,stages=4,steps=600,organization=core.POLICY,**flags)
  def load(s,split):
   entry=manifest['files'][f'{s}:{split}'];file=a.inputs/entry['leaf']
   if sha(file)!=entry['sha256']:raise ValueError('exact originaldatachanged')
   with np.load(file,allow_pickle=False)as z:
    if set(z.files)!=set(RAW):raise ValueError('rawterminalfields only')
    data={k:z[k]for k in RAW}
   n=len(data['initial']);width=data['tokens'].shape[1];mask=np.arange(width)[None]<data['lengths'][:,None]
   if any(v.dtype!=np.int64 for v in data.values())or any(data[k].shape!=(n,)for k in('initial','lengths','targets'))or np.any((data['lengths']<0)|(data['lengths']>width))or np.any((data['initial']<0)|(data['initial']>=8))or np.any((data['targets']<0)|(data['targets']>=8))or np.any(mask&((data['tokens']<0)|(data['tokens']>=(16 if s==3 else 15)))):raise ValueError('originalrange/schema')
   rec.emit('actual_input_opened',stage=s,split=split,sha256=entry['sha256'],rows=n,zero_length_rows=int((data['lengths']==0).sum()),array_bytes=sum(v.nbytes for v in data.values()))
   return data
  datasets={s:{k:load(s,k)for k in('train','valid')}for s in range(4)};train_sources={s:datasets[s]['train']for s in range(4)}
  # Retained tensors draw the exact old initialization subset from freshseed0. No restore.
  torch.manual_seed(0);np.random.seed(0);random.seed(0);model=core.ContinuousPrimitiveBlock(8,15,**flags).to(device);opt=torch.optim.Adam(model.parameters(),lr=.003);replay=Replay128(128,48,0,8);initial_theta=NNhash(model)
  if model.parameter_count()!=P15 or len(list(model.parameters()))!=20:raise ValueError('exact live20/prospectiveP freshmodel required')
  rec.emit('configuration',args=vars(args),source_sha256=manifest['source_sha256'],initial_theta_sha256=initial_theta,fresh_initial_NN=True,parent_NN_loaded=False,architecture_arm=a.arm,architecture_flags=flags,initialization_scope='samefulloldtypedfactory/RNG; Aonly copiesoriginalproposalrawxcols48:64 withoutnewdraw; survivingNNtensorsidenticalseed0; Baddsstateonly/noP/noRNG',objective='ONLY terminalmeanNLL; cachedupdatedh-logp; noanchor/teacher/auxloss',same_original_data_and_schedule=True,address_repair='Arawtokenproposalonly/fullcontextalphaA; BcellcoverageV/max(z,1e-4); typedaction/oldhSTORE/oldlquery/CALL/G8/NLL unchanged' ,resources=old.resources(model,opt,device,replay))
  for stage in range(4):
   phase=f'stage{stage}_setup';completed=0;rec.check();bucket=48 if stage==3 else 16;data=datasets[stage]['train'];growth=None;stage_cpu=time.process_time();stage_start=time.perf_counter()
   if np.any(data['lengths']>bucket)or np.any(data['tokens'][:,bucket:]):raise ValueError('cannottruncate activeTRAINtofixedbucket')
   if stage==3:
    if engine is None:raise RuntimeError('oldgraphrequired beforegrowth')
    engine.stable();graph_reports.append(engine.report());engine.close();engine=None
    probe={k:torch.tensor(v,device=device)for k,v in dict(initial=[0],tokens=[[1,3]],lengths=[2]).items()}
    with torch.no_grad():before=model(probe['initial'],probe['tokens'],probe['lengths'],output='logits').clone()
    growth=grow_storage(model,opt,args)
    with torch.no_grad():after=model(probe['initial'],probe['tokens'],probe['lengths'],output='logits')
    if not torch.equal(before,after)or model.parameter_count()!=P16:raise RuntimeError('onceoriginaltokengrowthchanged oldexecution')
   indices=np.random.default_rng(np.random.SeedSequence([0,8,stage,9027])).integers(0,len(data['initial']),size=(600,64),dtype=np.int64);draws=replay.sample_indices(stage,600,16)if stage else None
   with(rec.out/f'stage{stage}-common-indices.npz').open('xb')as f:np.savez_compressed(f,indices=indices)
   if stage:
    with(rec.out/f'stage{stage}-replay-draws.npz').open('xb')as f:np.savez_compressed(f,buffer_indices=draws)
   costs=dict(SGD_updates=0,training_wall_seconds=0.,training_CPU_seconds=0.,CUDA_stream_elapsed_seconds=0.,padded_forward_MAC_proxy=0,active_forward_MAC_proxy=0,active_token_steps=0,padded_token_steps=0,current_training_label_reads=0,old_replay_training_label_reads=0,batch_assembly_wall_seconds=0.,batch_H2D_bytes=0,evaluation_wall_seconds=0.,evaluation_forward_MAC_proxy=0,evaluation_rows=0,diagnostic_wall_seconds=0.,checkpoint_wall_seconds=0.,padded_coverage_divisions=0,padded_coverage_clamps=0,active_coverage_divisions=0,full_training_FLOPs=None)
   def assemble(step):
    tick=time.perf_counter()
    if stage:mixed=replay.mix(data,indices[step,:48],draws[step]);b,extent,active=old.batch(mixed,np.arange(64),device)
    else:b,extent,active=old.batch(data,indices[step],device)
    torch.cuda.synchronize();costs['batch_assembly_wall_seconds']+=time.perf_counter()-tick;costs['batch_H2D_bytes']+=sum(v.numel()*v.element_size()for v in b.values())
    if extent>bucket:raise ValueError('mixedactivebatch exceedsbucket')
    return b,extent,active
   first,first_extent,first_active=assemble(0)
   replay_before_graph_admission=copy.deepcopy(replay.state_dict())
   if stage in(0,3):engine=GraphEngine(model,opt,bucket,first,rec.out,rec.check,stage);rec.emit('actual_graph_capture',stage=stage,graph=engine.report())
   else:
    engine.stable();row=engine.diagnostic(stage,0,first);costs['diagnostic_wall_seconds']+=row['wall_seconds'];rec.emit('actual_all20_current_NN_gradient_check',**row)
   if not old.nested_state_equal(replay_before_graph_admission,replay.state_dict()):raise RuntimeError('0SGD graph admission changedReplay')
   curves=[];loop=time.perf_counter();loopCPU=time.process_time()
   for step in range(601):
    rec.check();phase=f'stage{stage}_step{step}'
    if step%100==0:
     metrics={};prediction_receipts={}
     for s in range(stage+1):
      metrics[s],timing=(evaluate_saved(model,datasets[s]['valid'],args,device,rec,f'stage{stage}-valid-task{s}-predictions.npz',modes=('soft',))if stage in(2,3)and step==600 else old.evaluate(model,datasets[s]['valid'],args,device,rec,modes=('soft',)));costs['evaluation_wall_seconds']+=timing['wall_seconds'];costs['evaluation_forward_MAC_proxy']+=timing['forward_MAC_proxy'];costs['evaluation_rows']+=len(datasets[s]['valid']['initial'])
      if 'predictions'in timing:prediction_receipts[s]=timing['predictions']
     engine.stable();row=dict(stage=stage,step=step,valid=metrics,saved_VALID_predictions=prediction_receipts);curves.append(row);rec.emit('actual_VALID_curve',**row)
     if step:
      row=engine.diagnostic(stage,step);costs['diagnostic_wall_seconds']+=row['wall_seconds'];rec.emit('actual_all20_current_NN_gradient_check',**row)
    if step==600:break
    b,extent,active=(first,first_extent,first_active)if step==0 else assemble(step);model.train();torch.cuda.synchronize();tick=time.perf_counter();cpu=time.process_time();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();out,loss=engine.run(b)
    if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinitecachedterminalNLL')
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();end.record();torch.cuda.synchronize()
    if not bool(torch.isfinite(torch.cat([v.detach().reshape(-1)for v in model.parameters()])).all()):raise FloatingPointError('nonfiniteNNaftereagerAdam')
    engine.stable();elapsed=time.perf_counter()-tick;completed=step+1;actual_updates+=1;padded=model.resource_report(64,bucket,active_steps=64*bucket)['proxy_forward_macs'];activeMAC=model.resource_report(64,extent,active_steps=active)['proxy_forward_macs']
    for k,v in(('SGD_updates',1),('training_wall_seconds',elapsed),('training_CPU_seconds',time.process_time()-cpu),('CUDA_stream_elapsed_seconds',start.elapsed_time(end)/1000.),('padded_forward_MAC_proxy',padded),('active_forward_MAC_proxy',activeMAC),('active_token_steps',active),('padded_token_steps',64*bucket),('current_training_label_reads',48 if stage else 64),('old_replay_training_label_reads',16 if stage else 0)):costs[k]+=v
    if flags['coverage_normalization']:
     costs['padded_coverage_divisions']+=64*bucket*8*32;costs['padded_coverage_clamps']+=64*bucket*8;costs['active_coverage_divisions']+=active*8*32
    rec.emit('actual_update',stage=stage,step=completed,terminal_NLL=float(loss),preclip_gradient=float(norm),batch_accuracy=float((out.argmax(-1)==b['targets']).float().mean()),active_token_steps=active,padded_token_steps=64*bucket,padded_forward_MAC_proxy=padded,active_forward_MAC_proxy=activeMAC,training_wall_seconds=elapsed)
    if step:del b
   costs.update(loop_including_VALID_diagnostics_batch_wall_seconds=time.perf_counter()-loop,loop_including_VALID_diagnostics_batch_CPU_seconds=time.process_time()-loopCPU)
   used[stage]=indices if stage==0 else indices[:,:48]
   def save(leaf,maintenance_complete):
    tick=time.perf_counter();path=rec.out/leaf;payload=dict(schema='typed-normalized-memory-fourstage-graph-training-v1',architecture_policy=core.POLICY,architecture_arm=a.arm,architecture_flags=flags,stage=stage,step=600,stage_complete=True,maintenance_complete=maintenance_complete,formal_Primary=False,fresh_initial_NN=True,initial_theta_sha256=initial_theta,args=vars(args),model_state={n:v.detach().cpu().clone()for n,v in model.state_dict().items()},block_checkpoint=model.checkpoint(),optimizer=opt.state_dict(),rng=old.rng_state(),replay_state=replay.state_dict(),used_current_indices=used[stage].copy(),source_sha256=manifest['source_sha256'],costs=costs,graph_bucket=bucket,indices_payload_sha256=arraysha(indices),replay_draws_payload_sha256=arraysha(draws)if stage else None)
    with path.open('xb')as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    receipt=dict(file=leaf,sha256=sha(path),bytes=path.stat().st_size,write_wall_seconds=time.perf_counter()-tick);costs['checkpoint_wall_seconds']+=receipt['write_wall_seconds'];return receipt
   phase=f'stage{stage}_save_before_maintenance';safety=save(f'typed-normalized-memory-stage{stage}-trained-before-maintenance.pt',False)
   maintenance_data=data
   if data['tokens'].shape[1]>48:
    if np.any(data['lengths']>48)or np.any(data['tokens'][:,48:]):raise ValueError('maintenance48viewcannotdropactiveinformation')
    maintenance_data=dict(data,tokens=data['tokens'][:,:48]);train_sources[stage]=maintenance_data;rec.emit('replay_storage_padding_view',stage=stage,original_columns=data['tokens'].shape[1],stored_columns=48,removed_entries_zero=True,training_raw_unchanged=True)
   maintenance=replay.consume(maintenance_data,used[stage],stage,emit=rec.emit);source_audit=replay.validate_sources(train_sources,used)
   if replay.seen!=38400+stage*28800 or replay.draws!=stage*9600:raise RuntimeError('committedcurrent/replay exposurecounterdiffers')
   final=save(f'typed-normalized-memory-stage{stage}-step600.pt',True);phase=f'stage{stage}_reload';saved_rng=old.rng_state();tick=time.perf_counter();loaded=torch.load(rec.out/final['file'],map_location='cpu',weights_only=False);other=core.ContinuousPrimitiveBlock.from_checkpoint(loaded['block_checkpoint'],expected_input_only_address=flags['input_only_address'],expected_coverage_normalization=flags['coverage_normalization']).to(device);other_opt=torch.optim.Adam(other.parameters(),lr=.003);other_opt.load_state_dict(loaded['optimizer']);other_replay=Replay128.from_state_dict(loaded['replay_state'])
   if not old.tensor_state_equal(model.state_dict(),other.state_dict())or not old.nested_state_equal(opt.state_dict(),other_opt.state_dict())or not old.nested_state_equal(replay.state_dict(),other_replay.state_dict())or not old.nested_state_equal(saved_rng,loaded['rng']):raise RuntimeError('actualperphaseNN/Adam/RNG/replayreload differs')
   check_batch,_,_=old.batch(datasets[stage]['valid'],np.arange(16),device);model.eval();other.eval()
   with torch.no_grad():
    for mode in('soft','native'):
     if not torch.equal(old.forward(model,check_batch,args,mode),old.forward(other,check_batch,args,mode)):raise RuntimeError('actualcachedemissionreload differs')
   del other,other_opt,other_replay,loaded;old.restore_rng(saved_rng);engine.stable();reload=dict(exact_NN_Adam_RNG_replay_forward=True,RNG_factory_neutral=True,wall_seconds=time.perf_counter()-tick)
   report=dict(stage=stage,architecture_policy=core.POLICY,costs=costs,checkpoint=final,premaintenance_safety_checkpoint=safety,reload=reload,resources=old.resources(model,opt,device,replay),curves=curves,growth=growth,replay_maintenance=maintenance,replay_source_audit=source_audit,stage_whole_wall_seconds=time.perf_counter()-stage_start,stage_whole_CPU_seconds=time.process_time()-stage_cpu)
   reports.append(report);rec.json(f'completed-stage{stage}-before-TEST.json',dict(report=report,test_data_read=False));rec.emit('stage_complete',**report)
  graph_reports.append(engine.report());engine.close();engine=None
  if len(graph_reports)!=2 or actual_updates!=2400 or[gr['bucket']for gr in graph_reports]!=[16,48]or[gr['training_FB_calls']for gr in graph_reports]!=[1800,600]:raise RuntimeError('exacttwo-capture2400fresh contract')
  rec.json('ALL_TRAINING_FROZEN_BEFORE_TEST.json',dict(reports=reports,no_more_updates=True,TEST_read=False,actual_new_updates=2400,initial_theta_sha256=initial_theta,architecture_policy=core.POLICY))
  phase='frozen_TEST';frozen=NNhash(model);tests={};test_costs=[]
  for s in range(4):
   tests[s]={}
   for split in('test_iid','test_long','test_unseen'):
    data=load(s,split);tests[s][split],timing=evaluate_saved(model,data,args,device,rec,f'final-task{s}-{split}-predictions.npz');test_costs.append(timing)
  tick=time.perf_counter();inference=old.inference_report(model,datasets[3]['valid'],args,device,rec);inference_seconds=time.perf_counter()-tick
  if NNhash(model)!=frozen:raise RuntimeError('frozenTEST/inference changedNN')
  result=dict(schema='typed-normalized-memory-fourstage-graph-result-v1',status='ACTUAL_FOUR_STAGE_TYPED_NORMALIZED_MEMORY_GRAPH_COMPLETE',formal_Primary=False,architecture_policy=core.POLICY,architecture_arm=a.arm,architecture_flags=flags,fresh_initial_NN=True,parent_NN_loaded=False,initial_theta_sha256=initial_theta,actual_new_updates_this_process=2400,new_architecture_stages_executed=[0,1,2,3],old_experiment_rerun=False,historical_updates_shared=0,reports=reports,graphs=graph_reports,actual_graph_capture_count=2,resources=old.resources(model,opt,device,replay),checkpoint=reports[-1]['checkpoint'],checkpoint_schema='typed-normalized-memory-fourstage-graph-training-v1',source_sha256=manifest['source_sha256'],tests=tests,test_costs=test_costs,inference=inference,inference_total_seconds=inference_seconds,NN_unchanged_after_TEST=True,objective='terminalNLLonly; noaux/teacher/intermediatetruth',all20_grad_storage_stable_within_capture=True,driver_wall_seconds=time.perf_counter()-rec.started,driver_CPU_seconds=time.process_time()-whole_cpu,full_training_FLOPs=None,provider_effective_compute=None,accuracy_or_speed_gate=False,cost_scope='allfresh2400/2captures/6warmups/allfixedTRAINcurrentNNdiagnostics/copy/padding/logpcache/rawV/typedactioncompetition/oldhSTORE/OLDlquery/eagerclip+Adam/VALID/TEST/inference/checkpoint+safety/reload/maintenance/CPU+wholewall; nestedtimesnotadditive; samefulloldtypedfactoryRNG; Arawxproposalprunes640P; Badds8floatcoverage/32Bperexample plusdivisionsclamps; sameoriginalsampler/replay; threearmsfixedprospectively; noquantizedorfullclosureclaim')
  rec.json('result.json',result);rec.emit('completed',actual_new_updates=2400,actual_graph_capture_count=2);print(json.dumps(dict(status=result['status'],P=model.parameter_count(),architecture_policy=core.POLICY)))
 except BaseException as e:
  rec.json('failure.json',dict(status='FAILED_OR_BOUNDED_TYPED_NORMALIZED_MEMORY_FRESH_GRAPH',architecture_policy=core.POLICY,phase=phase,stage=stage,completed_updates_current_stage=completed,actual_updates=actual_updates,completed_stages=reports,error=repr(e),traceback=traceback.format_exc()))
  if model is not None:
   with(rec.out/'partial-actual-NN.pt').open('xb')as f:torch.save(dict(model_state=model.state_dict(),optimizer=opt.state_dict()if opt is not None else None,actual_updates=actual_updates,architecture_policy=core.POLICY,source_sha256=manifest.get('source_sha256')if 'manifest'in locals()else None),f);f.flush();os.fsync(f.fileno())
  raise
 finally:
  if engine is not None:engine.close()
  rec.stream.close()
if __name__=='__main__':main()
