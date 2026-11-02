"""Two frozen OWN NNs, three fixed V interventions, four public suffix steps.

All six raw condition records precede opening VALID targets. No truth/routing
input, optimizer, training, parameter change, or score-selected donor.
"""
import argparse,hashlib,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import continuous_primitive_block as core
CARRY={'h':(32,),'controller':(16,),'values':(8,32),'address':(8,),'logp':(8,)}
CONDITIONS=('sham','zero_V','donor_V')
VALID_SHA='7d8c81c0e41bcea94785fc63593f781b6042b0d2b87844d841db2d0a1875a31b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
 args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);start=time.perf_counter();cpu=time.process_time();records=[]
 try:
  torch.set_num_threads(1);manifest=json.loads((args.inputs/'manifest.json').read_text())
  if manifest['schema']!='typed-suffix-memory-intervention-CPU-v1'or len(manifest['models'])!=2 or manifest['donor_group_offset']!=127:raise ValueError('fixed2NN/3conditions/offset127')
  for leaf,h in manifest['source_sha256'].items():
   if sha(Path(__file__).with_name(leaf))!=h:raise ValueError('frozen source changed')
  if sha(args.inputs/'valid.npz')!=VALID_SHA:raise ValueError('same512VALID required')
  with np.load(args.inputs/'valid.npz',allow_pickle=False)as z:public={k:z[k]for k in('initial','tokens','lengths')}
  if public['tokens'].shape!=(512,128):raise ValueError('same VALID shape')
  groups={};suffix=np.empty((512,4),dtype=np.int64)
  for row,length in enumerate(public['lengths']):
   word=public['tokens'][row,:length];key=(int(public['initial'][row]),tuple(word[:-4].tolist()));groups.setdefault(key,[]).append(row);suffix[row]=word[-4:]
  members=list(groups.values())
  if len(members)!=256 or any(len(g)!=2 for g in members)or not np.all(np.isin(suffix[:,[0,2]],[7,8]))or not np.all(suffix[:,1]==10)or not np.all(suffix[:,3]==15):raise ValueError('256publicprefix pairs/twoqueryorders')
  group_id=np.empty(512,dtype=np.int64);donor_row=np.empty(512,dtype=np.int64)
  for i,rows in enumerate(members):group_id[rows]=i;donor_row[rows]=members[(i+127)%256][0]
  def check():
   if time.perf_counter()-start>120:raise TimeoutError('120science/150outer CPU bounds')
  for spec in manifest['models']:
   check();checkpoint=args.inputs/spec['leaf'];event_file=args.inputs/spec['event_leaf']
   if sha(checkpoint)!=spec['sha256']or checkpoint.stat().st_size!=spec['bytes']or sha(event_file)!=spec['event_sha256']:raise ValueError('exact frozen checkpoint/raw changed')
   payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
   if payload['schema']!='typed-paired-tail-stage3-training-v1'or payload['stage']!=3 or payload['step']!=600 or not payload['stage_complete']or not payload['maintenance_complete']or payload['sampling']!=spec['sampling']or payload['source_sha256']!=spec['parent_source_sha256']:raise ValueError('OWN completed stage3/source required')
   model=core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint'],expected_action_policy=core.POLICY);del payload;model.eval()
   if model.parameter_count()!=10991 or len(list(model.parameters()))!=20:raise ValueError('typed20/P10991')
   before={k:v.detach().clone()for k,v in model.state_dict().items()}
   for p in model.parameters():p.requires_grad_(False)
   with np.load(event_file,allow_pickle=False)as z:
    entry={k:z['before_'+k][:,4].copy()for k in CARRY};reference=z['after_logp'][:,7].copy()
    if not np.array_equal(z['original_lengths'],public['lengths'])or not np.array_equal(z['event_positions'][:,4],public['lengths']-4):raise ValueError('saved exact suffix-entry event4')
   if any(not np.array_equal(v[rows[0]],v[rows[1]])for v in entry.values()for rows in members):raise ValueError('identical public-prefix pairs must have exact same saved before-carry')
   for condition in CONDITIONS:
    check();tick=time.perf_counter();ct=time.process_time();initial={k:v.copy()for k,v in entry.items()}
    if condition=='zero_V':initial['values'].fill(0)
    elif condition=='donor_V':initial['values']=entry['values'][donor_row].copy()
    logps=np.empty((512,4,8),dtype=np.float32);actions=np.empty((512,4,7),dtype=np.float32);attention=np.empty((512,4,8),dtype=np.float32);final={k:np.empty((512,)+shape,dtype=np.float32)for k,shape in CARRY.items()}
    with torch.no_grad():
     for first in range(0,512,64):
      carry={k:torch.from_numpy(v[first:first+64].copy())for k,v in initial.items()}
      for position in range(4):
       check();carry,trace=model._transition(carry,model.token_embedding(torch.from_numpy(suffix[first:first+64,position].copy())),'soft');logps[first:first+64,position]=carry['logp'].numpy();actions[first:first+64,position]=trace['action_probabilities'].numpy();attention[first:first+64,position]=trace['read_attention'].numpy()
      for k in final:final[k][first:first+64]=carry[k].numpy()
    if any(not np.isfinite(v).all()for v in [logps,actions,attention,*final.values()]):raise ValueError('nonfinite intervention')
    sham=None
    if condition=='sham':
     difference=np.abs(logps[:,-1]-reference);passed=bool(np.all(difference<=2e-5+2e-5*np.abs(reference))and np.array_equal(logps[:,-1].argmax(-1),reference.argmax(-1)))
     sham=dict(max_abs=float(difference.max()),argmax_differences=int(np.sum(logps[:,-1].argmax(-1)!=reference.argmax(-1))),atol=2e-5,rtol=2e-5,passed=passed)
     if not passed:raise ValueError('fixed sham replay admission failed')
    leaf=spec['key']+'-'+condition+'-raw-before-targets.npz';path=args.out/leaf
    with path.open('xb')as stream:np.savez_compressed(stream,logp=logps,action_probabilities=actions,read_attention=attention,**{'final_'+k:v for k,v in final.items()},group_id=group_id,donor_row=donor_row,suffix=suffix)
    records.append(dict(model=spec['key'],condition=condition,P=10991,checkpoint_sha256=spec['sha256'],entry_raw_sha256=spec['event_sha256'],sham_admission=sham,raw_file=leaf,raw_sha256=sha(path),raw_bytes=path.stat().st_size,raw_array_payload_bytes=sum(v.nbytes for v in [logps,actions,attention,*final.values(),group_id,donor_row,suffix]),NN_forward_transitions=512*4,transition_calls=8*4,logical_suffix_passes=1,SGD_updates=0,CPU_seconds=time.process_time()-ct,wall_seconds=time.perf_counter()-tick,VALID_targets_loaded=False))
   if not all(torch.equal(v,before[k])for k,v in model.state_dict().items())or sha(checkpoint)!=spec['sha256']:raise ValueError('frozen NN changed')
   del model,before,entry
  put(args.out/'ALL_SIX_RAW_SAVED_BEFORE_TARGETS.json',dict(all_six_saved=True,VALID_targets_loaded=False,privileged_truth_loaded=False,group_order='first original row encounter',group_count=256,donor_offset=127,donor_representative='first original row in donor prefixgroup',same_donor_for_query_members=True,records=records))
  # No target array was opened above. Teacher states are never loaded at all.
  with np.load(args.inputs/'valid.npz',allow_pickle=False)as z:targets=z['targets']
  scores={}
  for name in [s['key']for s in manifest['models']]:
   with np.load(args.out/(name+'-sham-raw-before-targets.npz'),allow_pickle=False)as z:baseline=z['logp'][:,-1]
   for condition in CONDITIONS:
    with np.load(args.out/(name+'-'+condition+'-raw-before-targets.npz'),allow_pickle=False)as z:logp=z['logp'][:,-1]
    prediction=logp.argmax(-1);prob=np.exp(logp);baseprob=np.exp(baseline);candidate=np.zeros(512,dtype=np.bool_);flips=both=one=changed=0
    for left,right in members:
     candidate[[left,right]]=np.isin(prediction[[left,right]],targets[[left,right]])
     if targets[left]!=targets[right]:
      changed+=1;flips+=prediction[left]!=prediction[right];correct=int(prediction[left]==targets[left])+int(prediction[right]==targets[right]);both+=correct==2;one+=correct==1
    scores[name+'-'+condition]=dict(accuracy=float((prediction==targets).mean()),NLL=float(-logp[np.arange(512),targets].mean()),candidate_coverage=int(candidate.sum()),query_changed_target_pairs=changed,queryflip_disagreements=int(flips),queryflip_both_correct=int(both),queryflip_one_correct=int(one),prediction_changes_vs_sham=int(np.sum(prediction!=baseline.argmax(-1))),probability_TV_vs_sham_mean=float(.5*np.abs(prob-baseprob).sum(-1).mean()),logp_max_abs_vs_sham=float(np.abs(logp-baseline).max()))
  value=dict(status='COMPLETED_TYPED_SUFFIX_MEMORY_INTERVENTION_CPU',source_sha256=manifest['source_sha256'],VALID_sha256=VALID_SHA,NNs=2,conditions=list(CONDITIONS),all_six_raw_saved_before_targets=True,privileged_truth_loaded=False,NN_forward_transitions=sum(r['NN_forward_transitions']for r in records),SGD_updates=0,records=records,scores=scores,wall_seconds=time.perf_counter()-start,CPU_seconds=time.process_time()-cpu,CPU_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)),checkpoint_bytes_loaded=sum(s['bytes']for s in manifest['models']),live_optimizer_bytes=0,full_FLOPs=None,provider_effective_compute=None,scope='fixed-NN suffix sensitivity; zeroV is OOD and donorV is context-mismatched, not a causal training-improvement claim; no labels in donor or NN')
  put(args.out/'result.json',value);print(json.dumps(dict(status=value['status'],NN_forward_transitions=value['NN_forward_transitions'],SGD=0)))
 except BaseException as error:put(args.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),records=records));raise
if __name__=='__main__':main()
