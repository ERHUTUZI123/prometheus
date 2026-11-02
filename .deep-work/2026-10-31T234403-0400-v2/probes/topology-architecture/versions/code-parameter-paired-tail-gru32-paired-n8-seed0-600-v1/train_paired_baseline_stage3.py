"""One OWN ordinary baseline stage2 ->600 paired or same-pool tail updates.

Original NN class/Adam003/clip1/terminal crossentropy/Replay128; no extra loss.
Only sampling flag differs between the two tails of each model. ROOT executes.
"""
import argparse,copy,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import paired_baseline_resume as core
import owned_parent_driver as old

@torch.no_grad()
def evaluate(model,data,args,device,rec,leaf):
 was=model.training;model.eval();started=time.perf_counter();cpu=time.process_time();outs=[];proxy=0
 try:
  for offset in range(0,len(data['initial']),64):
   rec.check();ids=np.arange(offset,min(offset+64,len(data['initial'])));b,l,active=old.batch(data,ids,device);logits=old.forward(model,b,args,'soft');out=F.log_softmax(logits,-1)
   if not bool(torch.isfinite(out).all()):raise FloatingPointError('nonfinite evaluation')
   outs.append(out.cpu().numpy().copy());proxy+=old.work_proxy(model,args,len(ids),l,active,'soft')['proxy_forward_macs']
  logp=np.concatenate(outs);target=data['targets'];score=dict(rows=len(target),accuracy=float((logp.argmax(-1)==target).mean()),terminal_NLL=float(-logp[np.arange(len(target)),target].astype(np.float64).mean()),forward_MAC_proxy=proxy);raw=dict(soft_logp=logp,native_logp=logp,row_index=np.arange(len(target),dtype=np.int64),targets=target,initial=data['initial'],lengths=data['lengths']);path=rec.out/leaf
  with path.open('xb')as f:np.savez_compressed(f,**raw)
  receipt=dict(file=leaf,sha256=core.sha(path),bytes=path.stat().st_size,array_payload_bytes=sum(a.nbytes for a in raw.values()),extra_NN_forward_passes_for_saved_outputs=0,native='sameordinaryNNinterface alias, notanotherforward')
  return dict(soft=score,native=dict(score,same_neural_interface=True)),dict(wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,forward_MAC_proxy=proxy,actual_forward_examples=len(target),predictions=receipt)
 finally:model.train(was)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--arm',choices=tuple(core.ARMS),required=True);p.add_argument('--sampling',choices=('paired','same_pool_shuffle'),required=True);p.add_argument('--max-seconds',type=float,default=850);a=p.parse_args()
 if not 1<=a.max_seconds<=850:raise ValueError('existing850science/900outer only')
 rec=old.Recorder(a.out,a.max_seconds);device=torch.device('cuda:0');whole_cpu=time.process_time();model=opt=replay=None;completed=0;phase='source_restore';costs={}
 try:
  if not torch.cuda.is_available():raise RuntimeError('authorizedsameCUDAonly')
  torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,0);manifest=core.load_manifest(a.inputs);model,opt,args,replay,used,train_sources,parent,restore=core.restore(a.inputs,manifest,a.arm,device);growth,initial=core.grow(model,opt,args,device);del parent
  if sum(p.numel()for p in model.parameters())!=core.ARMS[a.arm][2]:raise ValueError('exact original one-rowgrowthP')
  indices,draws=core.schedules(a.inputs,manifest,replay,a.sampling);train=core.data(a.inputs,manifest,3,'train');valid={s:core.data(a.inputs,manifest,s,'valid')for s in range(4)}
  with(rec.out/'stage3-used-current-indices.npz').open('xb')as f:np.savez_compressed(f,indices=indices)
  with(rec.out/'stage3-replay-draws.npz').open('xb')as f:np.savez_compressed(f,buffer_indices=draws)
  rec.emit('configuration',arm=a.arm,sampling=a.sampling,source_sha256=manifest['source_sha256'],restore=restore,growth=growth,post_growth=initial,objective='ONLY terminalmeanCROSSENTROPYonordinaryNNrawlogits; noaux/teacher',old_effective_updates=1800,old_updates_repeated=0,resources=old.resources(model,opt,device,replay))
  costs=dict(SGD_updates=0,training_wall_seconds=0.,training_CPU_seconds=0.,CUDA_stream_elapsed_seconds=0.,forward_MAC_proxy=0,active_token_steps=0,padded_token_steps=0,batch_assembly_wall_seconds=0.,current_training_label_reads=0,old_replay_training_label_reads=0,evaluation_wall_seconds=0.,evaluation_forward_MAC_proxy=0,checkpoint_wall_seconds=0.,full_training_FLOPs=None);curves=[]
  for step in range(601):
   rec.check();phase='step'+str(step)
   if step%100==0:
    metrics={};receipts={}
    for s in range(4):
     metrics[s],timing=evaluate(model,valid[s],args,device,rec,f'step{step:03d}-valid-task{s}-predictions.npz');costs['evaluation_wall_seconds']+=timing['wall_seconds'];costs['evaluation_forward_MAC_proxy']+=timing['forward_MAC_proxy'];receipts[s]=timing['predictions']
    row=dict(step=step,valid=metrics,predictions=receipts);curves.append(row);rec.emit('actual_VALID_curve',**row)
   if step==600:break
   tick=time.perf_counter();mixed=replay.mix(train,indices[step,:48],draws[step]);b,l,active=old.batch(mixed,np.arange(64),device);torch.cuda.synchronize();costs['batch_assembly_wall_seconds']+=time.perf_counter()-tick
   if l>48:raise ValueError('publiccurrent/replay activeTRAIN<=48 required')
   model.train();opt.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.perf_counter();cpu=time.process_time();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);start.record();logits=old.forward(model,b,args,'soft');loss=old.loss_value(logits,b['targets'],args);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();end.record();torch.cuda.synchronize()
   if not bool(torch.isfinite(loss))or not all(bool(torch.isfinite(v).all())for v in model.parameters()):raise FloatingPointError('nonfiniteactualNN/update')
   wall=time.perf_counter()-tick;mac=old.work_proxy(model,args,64,l,active,'soft')['proxy_forward_macs'];completed=step+1
   for k,v in dict(SGD_updates=1,training_wall_seconds=wall,training_CPU_seconds=time.process_time()-cpu,CUDA_stream_elapsed_seconds=start.elapsed_time(end)/1000.,forward_MAC_proxy=mac,active_token_steps=active,padded_token_steps=64*l,current_training_label_reads=48,old_replay_training_label_reads=16).items():costs[k]+=v
   rec.emit('actual_update',step=completed,stage=3,terminal_NLL=float(loss.detach()),preclip_gradient=float(norm),training_wall_seconds=wall,forward_MAC_proxy=mac,active_token_steps=active,padded_token_steps=64*l)
  if replay.draws!=28800:raise RuntimeError('exact600mix/9600newreplaydraws')
  used[3]=indices[:,:48].copy()
  def save(leaf,complete):
   tick=time.perf_counter();payload=core.saved_payload(model,opt,replay,a.arm,a.sampling,manifest,used,600,costs,complete);path=rec.out/leaf
   with path.open('xb')as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
   row=dict(file=leaf,sha256=core.sha(path),bytes=path.stat().st_size,write_wall_seconds=time.perf_counter()-tick);costs['checkpoint_wall_seconds']+=row['write_wall_seconds'];return row
  phase='save_before_maintenance';safety=save('baseline-stage3-before-maintenance.pt',False);tick=time.perf_counter()
  if np.any(train['lengths']>48)or np.any(train['tokens'][:,48:]):raise ValueError('replaymaintenancecannotdropactiveinput')
  storage=dict(train,tokens=train['tokens'][:,:48]);train_sources[3]=storage;maintenance=replay.consume(storage,used[3],3,emit=rec.emit);audit=replay.validate_sources(train_sources,used)
  if replay.seen!=124800 or replay.draws!=28800:raise RuntimeError('same600current/replayexposures')
  maintenance_wall=time.perf_counter()-tick;checkpoint=save('baseline-stage3-step600.pt',True);phase='reload';saved_rng=core.state_now(device);loaded=torch.load(rec.out/checkpoint['file'],map_location='cpu',weights_only=False);other,otheropt,otherreplay=core.reload(loaded,device);probe,_,_=old.batch(valid[3],np.arange(16),device);model.eval();other.eval()
  with torch.no_grad():
   if not torch.equal(old.forward(model,probe,args,'soft'),old.forward(other,probe,args,'soft')):raise RuntimeError('actualsavedNNforwardreload')
  core.restore_rng(saved_rng,device);del other,otheropt,otherreplay,loaded;frozen=core.NNhash(model);rec.json('ALL_TRAINING_FROZEN_BEFORE_TEST.json',dict(actual_new_updates=600,old_updates_repeated=0,NN_sha256=frozen,checkpoint=checkpoint,safety=safety));phase='frozen_TEST';tests={};test_costs=[]
  for s in range(4):
   tests[s]={}
   for split in('test_iid','test_long','test_unseen'):
    rec.emit('actual_TEST_input_opened',stage=s,split=split);data=core.data(a.inputs,manifest,s,split);tests[s][split],timing=evaluate(model,data,args,device,rec,f'final-task{s}-{split}-predictions.npz');test_costs.append(timing)
  tick=time.perf_counter();inference=old.inference_report(model,valid[3],args,device,rec);inference_seconds=time.perf_counter()-tick
  if core.NNhash(model)!=frozen:raise RuntimeError('frozenTEST/inferenceNNchanged')
  result=dict(status='ACTUAL_PAIRED_BASELINE_STAGE3_600_COMPLETE',schema='paired-baseline-stage3-result-v1',arm=a.arm,sampling=a.sampling,source_sha256=manifest['source_sha256'],parent_checkpoint_sha256=restore['parent_checkpoint_sha256'],restore=restore,growth=growth,post_growth=initial,actual_new_updates=600,old_effective_updates=1800,old_updates_repeated=0,formal_Primary=False,objective='terminalNLL/CROSSENTROPYonly',costs=costs,curves=curves,safety=safety,checkpoint=checkpoint,checkpoint_schema=core.SCHEMA,reload=dict(exact_NN_Adam_RNG_replay_forward=True),replay_maintenance=maintenance,replay_source_audit=audit,maintenance_wall_seconds=maintenance_wall,tests=tests,test_costs=test_costs,inference=inference,inference_seconds=inference_seconds,NN_unchanged_after_TEST=True,resources=old.resources(model,opt,device,replay),driver_wall_seconds=time.perf_counter()-rec.started,driver_CPU_seconds=time.process_time()-whole_cpu,full_training_FLOPs=None,provider_effective_compute=None,no_accuracy_gate=True,cost_scope='OWN1800NN/Adam/RNG/replayrestored, originaltoken/positiongrowth, ONLY600newupdates; ordinaryeagerNN withoriginalbatchcropping, noartificialpadding/graph; allIO/schedule/VALID/TEST/inference/maintenance/checkpoint/reloadcharged; nestedtimesshouldnotsum')
  rec.json('result.json',result);rec.emit('completed',actual_new_updates=600);print(json.dumps(dict(status=result['status'],P=result['resources']['parameters'],arm=a.arm,sampling=a.sampling)))
 except BaseException as e:
  rec.json('failure.json',dict(phase=phase,completed_updates=completed,error=repr(e),traceback=traceback.format_exc(),costs=costs))
  if model is not None:
   with(rec.out/'partial-actual-NN.pt').open('xb')as f:torch.save(dict(model_state=model.state_dict(),optimizer=opt.state_dict()if opt else None,completed_updates=completed),f)
  raise
 finally:rec.stream.close()
if __name__=='__main__':main()
