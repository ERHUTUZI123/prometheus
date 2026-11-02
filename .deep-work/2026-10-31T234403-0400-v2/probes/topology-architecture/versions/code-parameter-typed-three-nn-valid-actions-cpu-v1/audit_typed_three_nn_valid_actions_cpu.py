"""Frozen OWN stage2/paired/control CPU action trace, zero SGD.

NN phase decodes only original initial/tokens/lengths. All three NN raw files
are committed before privileged prefix truth or GPU reference predictions are
opened. Public token locations select trace positions, never NN routes.
"""
import argparse,hashlib,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import continuous_primitive_block as core
EVENTS=('KEY_first_write','WRITE_first','KEY_second_write','WRITE_second','KEY_for_READ','READ','KEY_other','MIX')
CARRY={'h':(32,),'controller':(16,),'values':(8,32),'address':(8,),'logp':(8,)}
TRACE={'action_probabilities':(7,),'action_weights':(7,),'action_logits':(7,),'call_probabilities':(4,),'call_logits':(4,),'compute_total':(1,),'state_gate':(32,),'candidate_h':(32,),'read':(32,),'read_attention':(8,),'write_gate':(1,),'write_value':(32,),'address_update_gate':(1,),'address_proposal':(8,),'address_query':(8,),'address_state':(8,)}
VALID_SHA='7d8c81c0e41bcea94785fc63593f781b6042b0d2b87844d841db2d0a1875a31b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,value):
 with p.open('x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
def tensor_bytes(value):
 if torch.is_tensor(value):return value.numel()*value.element_size()
 if isinstance(value,dict):return sum(tensor_bytes(x)for x in value.values())
 if isinstance(value,(list,tuple)):return sum(tensor_bytes(x)for x in value)
 return 0
def numpy_bytes(value):
 if isinstance(value,np.ndarray):return value.nbytes
 if isinstance(value,dict):return sum(numpy_bytes(x)for x in value.values())
 return 0
def events(tokens,lengths):
 positions=np.empty((512,8),dtype=np.int64)
 for row,length in enumerate(lengths):
  word=tokens[row,:length].tolist();writes=[i for i,t in enumerate(word)if t==9]
  if len(writes)!=2 or word.count(10)!=1 or word.count(15)!=1 or word[-1]!=15:raise ValueError('same original public VALID layout required')
  pos=[0,writes[0],writes[0]+1,writes[1],int(length)-4,int(length)-3,int(length)-2,int(length)-1]
  if any(word[pos[i]]not in(7,8)for i in(0,2,4,6))or word[pos[5]]!=10 or len(set(pos))!=8:raise ValueError('public KEY/WRITE/READ/MIX positions differ')
  positions[row]=pos
 return positions
def run_nn(spec,public,positions,inputs,out,check):
 started=time.perf_counter();cpu=time.process_time();path=inputs/spec['leaf']
 if path.is_symlink()or path.stat().st_size!=spec['bytes']or sha(path)!=spec['sha256']:raise ValueError('exact completed OWN checkpoint required')
 payload=torch.load(path,map_location='cpu',weights_only=False)
 if payload.get('schema')!=spec['checkpoint_schema']or payload.get('stage')!=spec['stage']or payload.get('step')!=600 or not payload.get('stage_complete')or not payload.get('maintenance_complete')or payload['source_sha256']!=spec['parent_source_sha256']:raise ValueError('checkpoint stage/schema/source mismatch')
 if spec['stage']==3 and payload.get('sampling')!=spec['sampling']:raise ValueError('paired/control checkpoint identity mismatch')
 model=core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint'],expected_action_policy=core.POLICY)
 if model.parameter_count()!=spec['P']or len(list(model.parameters()))!=20 or not all(torch.equal(v,payload['model_state'][n])for n,v in model.state_dict().items()):raise ValueError('true typed NN/P restore differs')
 saved_Adam=tensor_bytes(payload['optimizer']);saved_replay=numpy_bytes(payload['replay_state']);before={n:v.detach().clone()for n,v in model.state_dict().items()};del payload;model.eval()
 for p in model.parameters():p.requires_grad_(False)
 tokens=public['tokens'].copy();lengths=public['lengths'].copy()
 if spec['stage']==2:
  rows=np.arange(512);last=lengths-1
  if not np.all(tokens[rows,last]==15):raise ValueError('only final MIX may be trimmed for stage2')
  tokens[rows,last]=0;lengths-=1
 active=np.arange(128)[None]<lengths[:,None]
 if np.any(active&((tokens<0)|(tokens>=model.n_tokens)))or np.any((~active)&(tokens==15)):raise ValueError('invalid inactive new token')
 names={**{'before_'+k:s for k,s in CARRY.items()},**{'after_'+k:s for k,s in CARRY.items()},**TRACE};arrays={k:np.full((512,8)+s,np.nan,dtype=np.float32)for k,s in names.items()};valid=np.zeros((512,8),dtype=np.bool_);action_indices=np.full((512,8),-1,dtype=np.int64);load_wall=time.perf_counter()-started;forward_wall=forward_CPU=0.;transition_calls=active_transitions=0;batch_count=0
 with torch.no_grad():
  for start in range(0,512,64):
   check();rows=np.arange(start,start+64);initial=torch.as_tensor(public['initial'][rows]);word=torch.as_tensor(tokens[rows]);lens=torch.as_tensor(lengths[rows]);carry=model.initial_carry(initial)
   if sum(v.numel()for v in carry.values())!=64*320:raise ValueError('typed carry must be320 floats/example')
   t=time.perf_counter();c=time.process_time()
   for position in range(int(lengths[rows].max())):
    check();ids=(position<lens).nonzero(as_tuple=False).flatten()
    if not ids.numel():continue
    selected={k:v.index_select(0,ids)for k,v in carry.items()};updated,trace=model._transition(selected,model.token_embedding(word[:,position].index_select(0,ids)),'soft');transition_calls+=1;active_transitions+=len(ids)
    carry={k:v.index_copy(0,ids,updated[k])for k,v in carry.items()}
    for event in range(8):
     local=np.flatnonzero(positions[rows[ids.numpy()],event]==position)
     if not len(local):continue
     local=torch.as_tensor(local,dtype=torch.long);target=rows[ids.index_select(0,local).numpy()];valid[target,event]=True
     for k in CARRY:arrays['before_'+k][target,event]=selected[k].index_select(0,local).numpy();arrays['after_'+k][target,event]=updated[k].index_select(0,local).numpy()
     for k in TRACE:arrays[k][target,event]=trace[k].index_select(0,local).numpy()
     action_indices[target,event]=trace['action_indices'].index_select(0,local).numpy()
   forward_wall+=time.perf_counter()-t;forward_CPU+=time.process_time()-c;batch_count+=1
 if not np.all(valid[:,:7])or(spec['stage']==2 and np.any(valid[:,7]))or(spec['stage']==3 and not np.all(valid[:,7]))or any(not np.isfinite(x[valid]).all()for x in arrays.values()):raise ValueError('event coverage/nonfinite output')
 raw=out/(spec['key']+'-NN-events-before-truth.npz');tick=time.perf_counter()
 with raw.open('xb')as f:np.savez_compressed(f,**arrays,action_indices=action_indices,event_valid=valid,event_positions=positions,original_lengths=public['lengths'],runtime_lengths=lengths)
 if not all(torch.equal(v,before[n])for n,v in model.state_dict().items())or sha(path)!=spec['sha256']:raise ValueError('frozen NN/checkpoint changed')
 record=dict(key=spec['key'],stage=spec['stage'],sampling=spec.get('sampling'),checkpoint_sha256=spec['sha256'],parent_source_sha256=spec['parent_source_sha256'],P=spec['P'],NN_bytes=sum(p.numel()*p.element_size()for p in model.parameters()),checkpoint_bytes=path.stat().st_size,saved_historical_Adam_bytes=saved_Adam,saved_replay_bytes=saved_replay,live_optimizer_bytes=0,carry_scalars_per_example=320,carry_bytes_per_example=1280,raw_file=raw.name,raw_sha256=sha(raw),raw_bytes=raw.stat().st_size,raw_array_payload_bytes=sum(v.nbytes for v in arrays.values())+valid.nbytes+action_indices.nbytes,event_counts=valid.sum(0).tolist(),NN_unchanged=True,SGD_updates=0,privileged_truth_loaded=False,logical_dataset_passes=1,batch_initializations=batch_count,transition_calls=transition_calls,active_transitions=active_transitions,derived_runtime_input_sha256=hashlib.sha256(tokens.tobytes()+lengths.tobytes()).hexdigest(),stage2_only_MIX_trimmed_and_PAD0=spec['stage']==2,original_VALID_preserved=True,load_and_setup_wall_seconds=load_wall,event_capture_wall_seconds=forward_wall,event_capture_CPU_seconds=forward_CPU,raw_write_rehash_and_NN_compare_wall_seconds=time.perf_counter()-tick,whole_NN_wall_seconds=time.perf_counter()-started,whole_NN_CPU_seconds=time.process_time()-cpu,forward_MAC_proxy=model.resource_report(512,int(lengths.max()),active_steps=int(lengths.sum()))['proxy_forward_macs'],full_FLOPs=None,scope='ONECPU frozenNN active-row pass includingeventstores; actionsdescriptive/notcorrectroutinglabels; allbeforetruth')
 del model,before;return record
def summarize(path,truth):
 with np.load(path,allow_pickle=False)as z:data={k:z[k]for k in z.files}
 scores={}
 for e,name in enumerate(EVENTS):
  mask=data['event_valid'][:,e];rows=np.flatnonzero(mask)
  if not len(rows):scores[name]=dict(rows=0,scope='MIX not executed by stage2');continue
  pos=data['event_positions'][rows,e];q=truth['true_state_prefix'][rows,pos+1,0];before_q=truth['true_state_prefix'][rows,pos,0];address=truth['true_state_prefix'][rows,pos+1,3];memory=truth['true_state_prefix'][rows,pos,4+address];logp=data['after_logp'][rows,e];alpha=data['action_probabilities'][rows,e];attention=data['read_attention'][rows,e]
  entropy=-np.sum(alpha*np.log(np.maximum(alpha,1e-30)),-1);aentropy=-np.sum(attention*np.log(np.maximum(attention,1e-30)),-1)
  scores[name]=dict(rows=len(rows),cached_q_accuracy=float((logp.argmax(-1)==q).mean()),cached_q_NLL=float(-logp[np.arange(len(rows)),q].mean()),mean_action_probabilities=alpha.mean(0).tolist(),mean_compute_mass=float(alpha[:,:4].sum(-1).mean()),mean_LOAD=float(alpha[:,4].mean()),mean_STORE=float(alpha[:,5].mean()),mean_ADDRESS=float(alpha[:,6].mean()),action_argmax_counts=np.bincount(alpha.argmax(-1),minlength=7).tolist(),action_entropy_mean=float(entropy.mean()),memory_attention_entropy_mean=float(aentropy.mean()),memory_attention_max_mean=float(attention.max(-1).mean()),true_q_before=before_q.tolist(),true_q_after=q.tolist(),true_selected_memory=memory.tolist(),caution='privilegedposthoc semanticq/selectedcell only; rawV32 notprobabilitycoordinates, no actionisdeclareduniversallycorrect')
 return scores
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for key in('inputs','out'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--max-seconds',type=float,default=120);a=p.parse_args()
 if not 1<=a.max_seconds<=120:raise ValueError('120science/150outerCPU bounds')
 a.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cpu=time.process_time();records=[]
 try:
  torch.set_num_threads(1);manifest=json.loads((a.inputs/'manifest.json').read_text())
  if manifest.get('schema')!='typed-three-owned-VALID-action-audit-CPU-v1'or len(manifest['models'])!=3 or sha(a.inputs/'valid.npz')!=VALID_SHA:raise ValueError('fixed3OWNNN/same512VALID required')
  for leaf,h in manifest['source_sha256'].items():
   if sha(Path(__file__).with_name(leaf))!=h:raise ValueError('frozenaudit/core source changed')
  with np.load(a.inputs/'valid.npz',allow_pickle=False)as z:public={k:z[k]for k in('initial','tokens','lengths')}
  if public['tokens'].shape!=(512,128)or any(v.dtype!=np.int64 for v in public.values()):raise ValueError('same publicVALID schema')
  positions=events(public['tokens'],public['lengths'])
  def check():
   if time.perf_counter()-started>a.max_seconds:raise TimeoutError('boundedCPU3NNeventaudit')
  for spec in manifest['models']:records.append(run_nn(spec,public,positions,a.inputs,a.out,check))
  put(a.out/'ALL_THREE_NN_RAW_SAVED_BEFORE_TRUTH.json',dict(all3_NN_raw_saved=True,privileged_truth_loaded=False,NN_input_fields=['initial','tokens','lengths'],NN_transition_arguments='owncarry/learnedembedding(rawtoken)/soft',records=records))
  # All NN computation above has finished before any privileged data load.
  if sha(a.inputs/'prefix-truth.npz')!=manifest['truth_sha256']:raise ValueError('existingprefixtruth differs')
  with np.load(a.inputs/'prefix-truth.npz',allow_pickle=False)as z:truth={k:z[k]for k in z.files}
  if not np.array_equal(truth['original_lengths'],public['lengths']):raise ValueError('truth/public originalrows differ')
  scores={r['key']:summarize(a.out/r['raw_file'],truth)for r in records};comparisons={}
  for spec in manifest['models']:
   if not spec.get('reference_valid_leaf'):continue
   f=a.inputs/spec['reference_valid_leaf']
   if sha(f)!=spec['reference_valid_sha256']:raise ValueError('savedGPUVALID reference differs')
   with np.load(f,allow_pickle=False)as z:reference=z['soft_logp']
   with np.load(a.out/(spec['key']+'-NN-events-before-truth.npz'),allow_pickle=False)as z:current=z['after_logp'][:,7]
   comparisons[spec['key']]=dict(max_abs=float(np.max(np.abs(reference-current))),argmax_differences=int(np.sum(reference.argmax(-1)!=current.argmax(-1))),scope='CPUvsoriginalGPU samefrozenNN; measurednotbitwiseassumed; noaccuracygate')
  result=dict(status='COMPLETED_TYPED_THREE_OWNED_NN_CPU_VALID_ACTION_DIAGNOSIS',source_sha256=manifest['source_sha256'],VALID_sha256=VALID_SHA,privileged_truth_sha256=manifest['truth_sha256'],all3_raw_saved_before_truth=True,teacher_never_NN_feedback_or_routing=True,truth_exclusion_scope='trustedfixedsource ordering; sameUID canreadtruth, notOSblindness',NN_logical_dataset_passes=3,SGD_updates=0,records=records,events=list(EVENTS),action_order=['compute0','compute1','compute2','compute3','LOAD','STORE','ADDRESS'],privileged_posthoc_scores=scores,CPU_vs_saved_GPU_VALID=comparisons,active_transitions=sum(r['active_transitions']for r in records),batch_initializations=sum(r['batch_initializations']for r in records),wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,CPU_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)),full_FLOPs=None,provider_effective_compute=None,scope='oneCPU frozen3checkpoint mechanismdiagnosis; no TEST/SGD/oldNNrerun/parameterchange/newselection; rawVcapacity notdecidedbyoneprefixqaccuracy')
  put(a.out/'result.json',result);print(json.dumps(dict(status=result['status'],NNs=3,SGD=0)))
 except BaseException as error:put(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),records=records));raise
if __name__=='__main__':main()
