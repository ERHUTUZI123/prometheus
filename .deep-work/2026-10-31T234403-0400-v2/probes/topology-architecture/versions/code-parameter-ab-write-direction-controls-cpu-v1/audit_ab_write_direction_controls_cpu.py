"""Frozen AB: reference capture plus three fixed-strength memory replays.

Reference schedules and reference content norms are retained privileged NN
signals. No target, teacher state, future selected content or optimizer enters
the four NN passes. This is a controlled pathway intervention, not a learner.
"""
import argparse,hashlib,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import continuous_primitive_block as core
CONDITIONS=('reference','sham','permute_direction','cached_direction')
VALID_SHA='7d8c81c0e41bcea94785fc63593f781b6042b0d2b87844d841db2d0a1875a31b'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def payload_sha(x):return hashlib.sha256(x.tobytes(order='C')).hexdigest()
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def run(model,public,name,out,check,schedule=None,content=None):
 began=time.perf_counter();cpu=time.process_time();n=512;l=int(public['lengths'].max());carry=model.initial_carry(torch.from_numpy(public['initial']));tokens=torch.from_numpy(public['tokens']);lengths=torch.from_numpy(public['lengths'])
 raw=dict(active=np.zeros((n,l),dtype=np.bool_),w=np.zeros((n,l,8),dtype=np.float32),h=np.zeros((n,l,32),dtype=np.float32),store_selected=np.zeros((n,l),dtype=np.bool_),action_probabilities=np.zeros((n,l,7),dtype=np.float32),action_indices=np.full((n,l),-1,dtype=np.int64),read=np.zeros((n,l,32),dtype=np.float32),coverage=np.zeros((n,l,8),dtype=np.float32));calls=steps=0
 with torch.no_grad():
  for t in range(l):
   check();ids=(t<lengths).nonzero(as_tuple=False).flatten();rows=ids.numpy();incoming={k:v.index_select(0,ids)for k,v in carry.items()};updated,trace=model._transition(incoming,model.token_embedding(tokens[:,t].index_select(0,ids)),'soft');w=trace['write_gate']*trace['read_attention']
   raw['active'][rows,t]=True;raw['h'][rows,t]=incoming['h'].numpy();raw['store_selected'][rows,t]=(trace['action_indices']==core.STORE).numpy();raw['action_probabilities'][rows,t]=trace['action_probabilities'].numpy();raw['action_indices'][rows,t]=trace['action_indices'].numpy();raw['read'][rows,t]=trace['read'].numpy()
   if schedule is not None:
    # Fixed reference weights/content source; all other immediate fields stay
    # endogenous, and their future closed-loop trajectories may change.
    w=torch.from_numpy(schedule['w'][rows,t].copy());u=torch.from_numpy(content[rows,t].copy())
    updated['values']=incoming['values']+w[:,:,None]*(u[:,None]-incoming['values'])
    updated['coverage']=incoming['coverage']+w*(1-incoming['coverage'])
   raw['w'][rows,t]=w.numpy();carry={k:v.index_copy(0,ids,updated[k])for k,v in carry.items()};raw['coverage'][:,t]=carry['coverage'].numpy();calls+=1;steps+=len(rows)
 for key,value in carry.items():raw['final_'+key]=value.numpy().copy()
 raw['soft_logp']=raw['final_logp'];raw['lengths']=public['lengths'];raw['initial']=public['initial'];raw['row_index']=np.arange(512,dtype=np.int64)
 if any(not np.isfinite(v).all()for v in raw.values()if v.dtype.kind=='f'):raise ValueError('nonfinite frozenNN replay')
 path=out/(name+'-raw-before-targets.npz')
 with path.open('xb')as f:np.savez_compressed(f,**raw)
 return raw,dict(condition=name,file=path.name,sha256=sha(path),bytes=path.stat().st_size,array_payload_bytes=sum(v.nbytes for v in raw.values()),transition_calls=calls,active_transitions=steps,wall_seconds=time.perf_counter()-began,CPU_seconds=time.process_time()-cpu,targets_loaded=False,SGD_updates=0)
def replacements(public,ref):
 n,l=ref['active'].shape;groups={}
 for row,length in enumerate(public['lengths']):groups.setdefault((int(public['initial'][row]),tuple(public['tokens'][row,:length-4].tolist())),[]).append(row)
 members=list(groups.values())
 if len(members)!=256 or any(len(v)!=2 for v in members):raise ValueError('exact256 public-prefixquerypairs')
 M=np.zeros((n,l),dtype=np.bool_);donor=np.full((n,l),-1,dtype=np.int64);excluded=np.zeros((n,l),dtype=np.bool_);prefix_twin_checks=0
 for t in range(l):
  eligible=[]
  for g,(left,right)in enumerate(members):
   pair_active=ref['active'][[left,right],t];pair_nonchosen=~ref['store_selected'][[left,right],t]
   if t<int(public['lengths'][left])-4 and bool(pair_active.all()):
    if not np.array_equal(ref['h'][left,t],ref['h'][right,t])or not np.array_equal(ref['w'][left,t],ref['w'][right,t])or ref['store_selected'][left,t]!=ref['store_selected'][right,t]:raise ValueError('querytwins differ beforequery')
    prefix_twin_checks+=1
   if bool((pair_active&pair_nonchosen).all()):eligible.append(g)
  if len(eligible)>1:
   shift=127%len(eligible)or 1
   for index,g in enumerate(eligible):
    rows=members[g];source=members[eligible[(index+shift)%len(eligible)]][0]
    M[rows,t]=True;donor[rows,t]=source
  excluded[:,t]=ref['active'][:,t]&(~ref['store_selected'][:,t])&(~M[:,t])
 norm=np.linalg.norm(ref['h'].astype(np.float64),axis=-1);perm=ref['h'].copy();comp=ref['h'].copy();cache=np.zeros((n,32),dtype=np.float32);seen=np.zeros(n,dtype=np.bool_);pre_first=np.zeros((n,l),dtype=np.bool_);first=np.full(n,-1,dtype=np.int64);perm_zero=cache_zero=source_zero=0;donor_norm=np.zeros((n,l),dtype=np.float64);cache_norm=np.zeros((n,l),dtype=np.float64)
 def scale(direction,source_norm):
  dnorm=float(np.linalg.norm(direction.astype(np.float64)))
  if source_norm==0:return np.zeros(32,dtype=np.float32),dnorm
  if dnorm==0:
   answer=np.zeros(32,dtype=np.float32);answer[0]=source_norm;return answer,dnorm
  return(direction.astype(np.float64)*(source_norm/dnorm)).astype(np.float32),dnorm
 for t in range(l):
  pre_first[:,t]=ref['active'][:,t]&(~seen)
  for row in np.flatnonzero(M[:,t]):
   perm[row,t],dnorm=scale(ref['h'][donor[row,t],t],norm[row,t]);comp[row,t],cnorm=scale(cache[row],norm[row,t]);donor_norm[row,t]=dnorm;cache_norm[row,t]=cnorm;perm_zero+=dnorm==0;cache_zero+=cnorm==0;source_zero+=norm[row,t]==0
  # Cache uses strictly past reference-selected STORE content for nonchosen
  # replacements; selected and treated rows are disjoint at this time.
  chosen=ref['active'][:,t]&ref['store_selected'][:,t];new=chosen&(~seen);first[new]=t;cache[chosen]=ref['h'][chosen,t];seen[chosen]=True
 flags=dict(treatment_mask=M,donor_row=donor,excluded_nonchosen=excluded,pre_first_selected=pre_first,first_selected_time=first,reference_content_norm=norm,donor_unscaled_norm=donor_norm,cached_unscaled_norm=cache_norm,permutation_content=perm,cached_content=comp)
 stats=dict(group_count=256,prefix_twin_checks=prefix_twin_checks,treated_events=int(M.sum()),excluded_nonchosen_events=int(excluded.sum()),reference_nonchosen_events=int((ref['active']&~ref['store_selected']).sum()),treated_pre_first_events=int((M&pre_first).sum()),treated_post_first_events=int((M&~pre_first).sum()),rows_never_reference_selected=int((first<0).sum()),zero_donor_e0_fallback_events=int(perm_zero),zero_cached_or_pre_first_e0_events=int(cache_zero),zero_reference_norm_events=int(source_zero),direction_scope='canonicalfirstrow donor-group direction, sharedfortwins; afterquery NOTfullcontentmultisetpermutation; inputprefix+referenceeligibility only',chosen_write_content='saved referenceOLDh in ALL replays; notliveinterventionh',norm_scope='eachreferencecontentnorm retained, scalarcancontaininformation; matched w,z,decay,contentnorm, NOTVnorm/netupdate/readnorm',unpermutable_scope='incompleteeligiblepairs orfewerthan2eligiblegroups unchangedinBOTH interventions')
 for name,value in [('permutation',perm),('cached',comp)]:
  norms=np.linalg.norm(value.astype(np.float64),axis=-1);delta=norms-norm;denom=np.maximum(norm,1e-30);diff=np.linalg.norm((value-ref['h']).astype(np.float64),axis=-1);dot=(value.astype(np.float64)*ref['h'].astype(np.float64)).sum(-1);valid=M&(norm>0);cos=dot[valid]/(norms[valid]*norm[valid]);stats[name]=dict(max_content_norm_abs_error=float(np.abs(delta[ref['active']]).max()),max_content_norm_relative_error=float((np.abs(delta)/denom)[valid].max())if bool(valid.any())else 0.,treated_content_L2_delta_mean=float(diff[M].mean()),treated_direction_cosine_mean=float(cos.mean()),treated_direction_cosine_min=float(cos.min()),actual_changed_content_events=int((M&(diff>1e-6)).sum()))
  if not np.all(np.abs(delta[ref['active']])<=2e-5+2e-5*norm[ref['active']]):raise ValueError('fixednormmatchedcontent gate failed')
 # Effective retained strength for each event class, using the same numerical
 # decay recurrence as z, not merely the raw sum of write weights.
 mass={}
 for name,mask in [('treated',M),('excluded_nonchosen',excluded),('treated_pre_first',M&pre_first),('treated_post_first',M&~pre_first),('reference_selected',ref['store_selected'])]:
  retained=np.zeros((n,8),dtype=np.float64)
  for t in range(l):retained=(1-ref['w'][:,t])*retained+ref['w'][:,t]*mask[:,t,None]
  flags[name+'_effective_strength']=retained;flags[name+'_raw_weight_sum']=(ref['w']*mask[:,:,None]).sum(1);mass[name]=dict(raw_weight_sum=float(flags[name+'_raw_weight_sum'].sum()),effective_retained_strength=float(retained.sum()))
 stats['mass']=mass;return flags,stats
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
 parser.add_argument('--max-seconds',type=float,default=120)
 a=parser.parse_args()
 if not 1<=a.max_seconds<=120:raise ValueError('120science/150outerCPU bound')
 a.out.mkdir(parents=True,exist_ok=False);began=time.perf_counter();cpu=time.process_time();records=[]
 try:
  torch.set_num_threads(1);manifest=json.loads((a.inputs/'manifest.json').read_text())
  if manifest['schema']!='AB-fixed-strength-direction-controls-CPU-v1'or sha(a.inputs/'valid.npz')!=VALID_SHA:raise ValueError('fixed source/data protocol')
  for leaf,h in manifest['source_sha256'].items():
   if sha(Path(__file__).with_name(leaf))!=h:raise ValueError('frozen source changed')
  cp=a.inputs/'ab.pt'
  if sha(cp)!=manifest['checkpoint_sha256']:raise ValueError('exactownABNN required')
  payload=torch.load(cp,map_location='cpu',weights_only=False)
  if payload['architecture_arm']!='AB'or payload['stage']!=3 or payload['step']!=600 or not payload['stage_complete']or not payload['maintenance_complete']or payload['source_sha256']!=manifest['parent_source_sha256']:raise ValueError('exact completedAB checkpoint')
  model=core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint'],expected_input_only_address=True,expected_coverage_normalization=True);del payload;model.eval()
  if model.parameter_count()!=10351 or len(list(model.parameters()))!=20:raise ValueError('fixedABP20')
  for p in model.parameters():p.requires_grad_(False)
  before={k:v.detach().clone()for k,v in model.state_dict().items()}
  with np.load(a.inputs/'valid.npz',allow_pickle=False)as z:public={k:z[k]for k in('initial','tokens','lengths')}
  def check():
   if time.perf_counter()-began>a.max_seconds:raise TimeoutError('120science/150outerCPU')
  ref,record=run(model,public,'reference',a.out,check);records.append(record);signals,signal_stats=replacements(public,ref);signal_path=a.out/'replacement-signals-before-targets.npz'
  with signal_path.open('xb')as f:np.savez_compressed(f,**signals)
  condition_raw={'reference':ref};strength={}
  for name in CONDITIONS[1:]:
   content=ref['h']if name=='sham'else signals['permutation_content']if name=='permute_direction'else signals['cached_content'];raw,record=run(model,public,name,a.out,check,ref,content);records.append(record);condition_raw[name]=raw
   if not np.array_equal(raw['w'],ref['w'])or not np.array_equal(raw['coverage'],ref['coverage']):raise ValueError('fixedw/z numericalrecurrence notexact')
   strength[name]=dict(w_max_abs=float(np.abs(raw['w']-ref['w']).max()),per_row_slot_sum_max_abs=float(np.abs(raw['w'].sum(1)-ref['w'].sum(1)).max()),z_trajectory_max_abs=float(np.abs(raw['coverage']-ref['coverage']).max()),per_row_weight_sum_difference=(raw['w'].sum((1,2))-ref['w'].sum((1,2))).tolist(),final_V_L2_vs_reference_mean=float(np.linalg.norm((raw['final_values']-ref['final_values']).reshape(512,-1),axis=-1).mean()))
  difference=np.abs(condition_raw['sham']['soft_logp']-ref['soft_logp']);sham=dict(max_logp_abs=float(difference.max()),argmax_differences=int(np.sum(condition_raw['sham']['soft_logp'].argmax(-1)!=ref['soft_logp'].argmax(-1))),passed=bool(np.all(difference<=2e-5+2e-5*np.abs(ref['soft_logp']))))
  if not sham['passed']or sham['argmax_differences']:raise ValueError('fixednumericalsham replayfailed')
  if not all(torch.equal(v,before[k])for k,v in model.state_dict().items())or sha(cp)!=manifest['checkpoint_sha256']:raise ValueError('ownNN changed')
  put(a.out/'ALL_FOUR_RAW_SAVED_BEFORE_TARGETS.json',dict(records=records,targets_loaded=False,teacher_loaded=False,replacement_signals_sha256=sha(signal_path),source_sha256=manifest['source_sha256']))
  # Targets, old diagnostic summaries and saved reference outputs only now open.
  with np.load(a.inputs/'valid.npz',allow_pickle=False)as z:targets=z['targets']
  previous=json.loads((a.inputs/'previous-fractional-result.json').read_text());native_accuracy=previous['final_scores']['native']['accuracy'];groups={}
  for row,length in enumerate(public['lengths']):groups.setdefault((int(public['initial'][row]),tuple(public['tokens'][row,:length-4].tolist())),[]).append(row)
  scores={}
  for name,raw in condition_raw.items():
   logp=raw['soft_logp'];pred=logp.argmax(-1);pairs=[rows for rows in groups.values()if targets[rows[0]]!=targets[rows[1]]];p=pred;accuracy=float((p==targets).mean());scores[name]=dict(correct=int((p==targets).sum()),accuracy=accuracy,NLL=float(-logp[np.arange(512),targets].mean()),native_fixed_baseline_accuracy=native_accuracy,accuracy_gap_to_fixed_original_native=accuracy-native_accuracy,query_nondegenerate_pairs=len(pairs),query_pred_changes=int(sum(p[x]!=p[y]for x,y in pairs)),query_both_correct=int(sum(p[x]==targets[x]and p[y]==targets[y]for x,y in pairs)),query_relation_consistent=int(sum((p[y]-p[x])%8==(targets[y]-targets[x])%8 for x,y in pairs)),TV_vs_reference_mean=float(.5*np.abs(np.exp(logp)-np.exp(ref['soft_logp'])).sum(-1).mean()),argmax_changes_vs_reference=int((p!=ref['soft_logp'].argmax(-1)).sum()))
  result=dict(status='COMPLETED_AB_FIXED_STRENGTH_DIRECTION_CONTROLS_CPU',source_sha256=manifest['source_sha256'],checkpoint_sha256=manifest['checkpoint_sha256'],P=10351,NN_parameter_bytes=41404,carry_scalars=328,conditions=list(CONDITIONS),records=records,all_raw_before_targets=True,teacher_loaded=False,reference_content_norm_retained=True,controlled_pathway_not_endogenous_total_effect=True,signals=dict(file=signal_path.name,sha256=sha(signal_path),bytes=signal_path.stat().st_size,payload_bytes=sum(v.nbytes for v in signals.values()),statistics=signal_stats),strength_checks=strength,sham=sham,scores=scores,existing_native_and_drop_reused_not_rerun=True,SGD_updates=0,active_NN_transitions=sum(x['active_transitions']for x in records),NN_transition_calls=sum(x['transition_calls']for x in records),wall_seconds=time.perf_counter()-began,CPU_seconds=time.process_time()-cpu,CPU_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)),checkpoint_bytes=cp.stat().st_size,optimizer_bytes_live=0,full_FLOPs=None,provider_effective_compute=None,scope='onefrozenAB/reference+3controlledreplays; fixedreferencew/z/decay/contentnorm, canonicaldirectionreplacementnotfullmultisetperm; scalarrefnormandscheduleretainprivilegedinformation; noTRAIN/TEST/teacher/newvariant')
  put(a.out/'result.json',result);print(json.dumps(dict(status=result['status'],scores=scores,SGD=0),sort_keys=True))
 except BaseException as error:put(a.out/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),records=records));raise
if __name__=='__main__':main()
