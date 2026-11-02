"""Frozen AB address-lifetime intervention, soft/native, no learning.

Public interval timing is privileged diagnostic information. Each saved
address is the NN's OWN updated state after KEY, with no supplied slot/value.
"""
import argparse,hashlib,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import continuous_primitive_block as core
VALID_SHA='7d8c81c0e41bcea94785fc63593f781b6042b0d2b87844d841db2d0a1875a31b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def positions(public):
 keys=np.empty((512,2),dtype=np.int64);writes=np.empty((512,2),dtype=np.int64);events=np.empty((512,8),dtype=np.int64);L=int(public['lengths'].max());hold=np.zeros((512,L),dtype=np.bool_)
 for row,length in enumerate(public['lengths']):
  word=public['tokens'][row,:length];w=np.flatnonzero(word==9)
  if len(w)!=2 or word[0]not in(7,8)or word[w[0]+1]not in(7,8)or word[-1]!=15 or word[-3]!=10:raise ValueError('same publictwoKEY/WRITE grammar')
  k=[0,int(w[0])+1];keys[row]=k;writes[row]=w;events[row]=[0,w[0],k[1],w[1],length-4,length-3,length-2,length-1]
  for left,right in zip(k,w):hold[row,left+1:right+1]=True
  if bool(hold[row,length-4:].any()):raise ValueError('querysuffix mustremainuntouched')
 return keys,writes,events,hold
def run(model,public,mode,keys,writes,events,hold,out,check):
 began=time.perf_counter();cpu=time.process_time();L=hold.shape[1];carry=model.initial_carry(torch.from_numpy(public['initial']));token=torch.from_numpy(public['tokens']);lengths=torch.from_numpy(public['lengths']);held=carry['address'].clone();snapshots=np.full((512,2,8),np.nan,dtype=np.float32)
 raw=dict(before_address=np.zeros((512,L,8),np.float32),proposed_address=np.zeros((512,L,8),np.float32),after_address=np.zeros((512,L,8),np.float32),action_probabilities=np.zeros((512,L,7),np.float32),action_indices=np.full((512,L),-1,np.int64),w=np.zeros((512,L,8),np.float32),read_attention=np.zeros((512,L,8),np.float32),h=np.zeros((512,L,32),np.float32),read=np.zeros((512,L,32),np.float32),coverage=np.zeros((512,L,8),np.float32),logp=np.zeros((512,L,8),np.float32),active=np.zeros((512,L),np.bool_));calls=steps=overrides=0
 with torch.no_grad():
  for t in range(L):
   check();ids=(t<lengths).nonzero(as_tuple=False).flatten();rows=ids.numpy();incoming={k:v.index_select(0,ids)for k,v in carry.items()};updated,trace=model._transition(incoming,model.token_embedding(token[:,t].index_select(0,ids)),mode)
   raw['before_address'][rows,t]=incoming['address'].numpy();raw['proposed_address'][rows,t]=updated['address'].numpy();raw['h'][rows,t]=incoming['h'].numpy();raw['read'][rows,t]=trace['read'].numpy();raw['read_attention'][rows,t]=trace['read_attention'].numpy();raw['w'][rows,t]=(trace['write_gate']*trace['read_attention']).numpy();raw['action_probabilities'][rows,t]=trace['action_probabilities'].numpy();raw['action_indices'][rows,t]=trace['action_indices'].numpy();raw['active'][rows,t]=True
   for k in range(2):
    local=np.flatnonzero(keys[rows,k]==t)
    if len(local):
     source=torch.from_numpy(local);target=ids.index_select(0,source);value=updated['address'].index_select(0,source);held=held.index_copy(0,target,value);snapshots[rows[local],k]=value.numpy()
   mask=torch.from_numpy(hold[rows,t]);updated['address']=torch.where(mask[:,None],held.index_select(0,ids),updated['address']);overrides+=int(mask.sum());carry={k:v.index_copy(0,ids,updated[k])for k,v in carry.items()}
   raw['after_address'][:,t]=carry['address'].numpy();raw['coverage'][:,t]=carry['coverage'].numpy();raw['logp'][:,t]=carry['logp'].numpy();calls+=1;steps+=len(rows)
 for k,v in carry.items():raw['final_'+k]=v.numpy().copy()
 raw.update(predictions=raw['final_logp'].argmax(-1),key_snapshots=snapshots,key_positions=keys,write_positions=writes,event_positions=events,hold_mask=hold,initial=public['initial'],lengths=public['lengths'])
 if any(not np.isfinite(v).all()for v in raw.values()if v.dtype.kind=='f'):raise ValueError('nonfinite frozenholdexecution')
 maximum=0.
 for row in range(512):
  for k in range(2):
   for t in range(int(keys[row,k])+1,int(writes[row,k])+1):
    maximum=max(maximum,float(np.abs(raw['before_address'][row,t]-snapshots[row,k]).max()),float(np.abs(raw['after_address'][row,t]-snapshots[row,k]).max()))
 if maximum!=0:raise ValueError('heldOWNaddress mustbeexact throughoutuse')
 path=out/('held_'+mode+'-raw-before-targets.npz')
 with path.open('xb')as f:np.savez_compressed(f,**raw)
 return raw,dict(mode=mode,file=path.name,sha256=sha(path),bytes=path.stat().st_size,array_payload_bytes=sum(v.nbytes for v in raw.values()),active_transitions=steps,transition_calls=calls,address_override_rows=overrides,held_before_and_after_max_abs=maximum,public_query_suffix_held=False,wall_seconds=time.perf_counter()-began,CPU_seconds=time.process_time()-cpu,targets_loaded=False,SGD_updates=0)
def metrics(logp,targets,groups):
 p=logp.argmax(-1);pairs=[rows for rows in groups.values()if targets[rows[0]]!=targets[rows[1]]]
 return dict(correct=int((p==targets).sum()),accuracy=float((p==targets).mean()),NLL=float(-logp[np.arange(512),targets].mean()),query_nondegenerate_pairs=len(pairs),query_pred_changes=int(sum(p[x]!=p[y]for x,y in pairs)),query_both_correct=int(sum(p[x]==targets[x]and p[y]==targets[y]for x,y in pairs)),query_relation_consistent=int(sum((p[y]-p[x])%8==(targets[y]-targets[x])%8 for x,y in pairs)))
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
 parser.add_argument('--max-seconds',type=float,default=120);a=parser.parse_args()
 if not 1<=a.max_seconds<=120:raise ValueError('120science/150outerCPU bound')
 a.out.mkdir(parents=True,exist_ok=False);began=time.perf_counter();cpu=time.process_time();records=[]
 try:
  torch.set_num_threads(1);manifest=json.loads((a.inputs/'manifest.json').read_text())
  if manifest['schema']!='AB-own-reference-lifetime-CPU-v1'or sha(a.inputs/'valid.npz')!=VALID_SHA:raise ValueError('fixedownreference/inputsource')
  for leaf,h in manifest['source_sha256'].items():
   if sha(Path(__file__).with_name(leaf))!=h:raise ValueError('frozensourcechanged')
  cp=a.inputs/'ab.pt'
  if sha(cp)!=manifest['checkpoint_sha256']:raise ValueError('exactownABNNrequired')
  payload=torch.load(cp,map_location='cpu',weights_only=False)
  if payload['architecture_arm']!='AB'or payload['stage']!=3 or payload['step']!=600 or not payload['stage_complete']or not payload['maintenance_complete']or payload['source_sha256']!=manifest['parent_source_sha256']:raise ValueError('truecompletedABcheckpoint')
  model=core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint'],expected_input_only_address=True,expected_coverage_normalization=True);del payload;model.eval()
  if model.parameter_count()!=10351 or len(list(model.parameters()))!=20:raise ValueError('fixedAB10351P/20tensors')
  for p in model.parameters():p.requires_grad_(False)
  before={k:v.detach().clone()for k,v in model.state_dict().items()}
  with np.load(a.inputs/'valid.npz',allow_pickle=False)as z:public={k:z[k]for k in('initial','tokens','lengths')}
  keys,writes,events,hold=positions(public)
  def check():
   if time.perf_counter()-began>a.max_seconds:raise TimeoutError('boundedOWNreference-lifetimeaudit')
  outputs={}
  for mode in('soft','native'):
   raw,record=run(model,public,mode,keys,writes,events,hold,a.out,check);outputs[mode]=raw;records.append(record)
  if not all(torch.equal(v,before[k])for k,v in model.state_dict().items())or sha(cp)!=manifest['checkpoint_sha256']:raise ValueError('frozenNNchanged')
  put(a.out/'BOTH_HELD_RAW_SAVED_BEFORE_TARGETS.json',dict(records=records,targets_loaded=False,teacher_loaded=False,source_sha256=manifest['source_sha256']))
  with np.load(a.inputs/'valid.npz',allow_pickle=False)as z:targets=z['targets']
  groups={}
  for row,length in enumerate(public['lengths']):groups.setdefault((int(public['initial'][row]),tuple(public['tokens'][row,:length-4].tolist())),[]).append(row)
  if len(groups)!=256 or any(len(rows)!=2 for rows in groups.values()):raise ValueError('exactpairedVALID')
  baseline={};scores={}
  for mode in('soft','native'):
   spec=manifest['baselines'][mode];path=a.inputs/spec['leaf']
   if sha(path)!=spec['sha256']:raise ValueError('savedoriginalbaselinebyteschanged')
   with np.load(path,allow_pickle=False)as z:old=z[spec['key']][:,7]if spec.get('event7')else z[spec['key']]
   baseline[mode]=metrics(old,targets,groups);logp=outputs[mode]['final_logp'];scores[mode]=dict(metrics(logp,targets,groups),original_baseline=baseline[mode],probability_TV_vs_original_mean=float(.5*np.abs(np.exp(logp)-np.exp(old)).sum(-1).mean()),argmax_changes_vs_original=int((logp.argmax(-1)!=old.argmax(-1)).sum()))
  result=dict(status='COMPLETED_AB_OWN_REFERENCE_LIFETIME_CPU',source_sha256=manifest['source_sha256'],checkpoint_sha256=manifest['checkpoint_sha256'],P=10351,NN_parameter_bytes=41404,carry_scalars=328,records=records,scores=scores,all_raw_before_targets=True,teacher_loaded=False,own_snapshots_only=True,privileged_public_interval_timing=True,query_suffix_untouched=True,original_soft_native_baselines_reused_not_rerun=True,NN_unchanged=True,SGD_updates=0,active_NN_transitions=sum(r['active_transitions']for r in records),NN_transition_calls=sum(r['transition_calls']for r in records),wall_seconds=time.perf_counter()-began,CPU_seconds=time.process_time()-cpu,CPU_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)),checkpoint_bytes=cp.stat().st_size,optimizer_bytes_live=0,full_FLOPs=None,provider_effective_compute=None,scope='diagnostic addresslifetime oracleintervals only, no correctslot/value/teacherstate; allotherupdatesandfuturecallsnaturalclosedloop; notlearnedcapability/baseline/architecturearm; noextraNNtraining')
  put(a.out/'result.json',result);print(json.dumps(dict(status=result['status'],scores=scores,SGD=0),sort_keys=True))
 except BaseException as error:put(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),records=records));raise
if __name__=='__main__':main()
