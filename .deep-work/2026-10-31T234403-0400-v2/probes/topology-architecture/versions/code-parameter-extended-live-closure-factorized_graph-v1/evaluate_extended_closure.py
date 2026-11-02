"""Fixed actual Graph-baseline/softbridge seven-suffix closure, zeroSGD.

Same frozen3584cases, metrics, lengthstrata and oneNNforward each; additionally
export the alreadycomputed finalcarry and CPUgeometry (not a full-state label).
"""
import argparse,hashlib,importlib.util,json,time,resource
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import owned_parent_driver as old
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def logical_carry(model,family):
 if family=='block':
  report=model.resource_report(batch_size=1,sequence_length=0,active_steps=0)
  return dict(scalars_per_example=report.get('persistent_example_state_scalars'),payload_bytes_per_example=report.get('persistent_example_state_payload_bytes'),resource_report=report,scope='logical recurrent payload, excludes transient workspace and allocator peak')
 if family=='gru':
  count=model.spec['layers']*model.spec['width'];scalar=next(model.parameters()).element_size()
  return dict(scalars_per_example=count,payload_bytes_per_example=count*scalar,scope='logical GRU hidden state; ordinary full-window forward, no serving-cache timing claim')
 return dict(scalars_per_example=None,payload_bytes_per_example=None,scope='owned Transformer exposes full-window forward only; reusable KV cache not implemented; transient attention workspace is not zero')

def subset_metrics(ids,data,pred,base_predictions,base_correct,suffix):
 ids=np.asarray(ids,dtype=np.int64);right=pred[ids]==data['targets'][ids]
 source=[int(data['source_train_index'][i])for i in ids]
 ownbase=np.asarray([base_correct[s]for s in source],dtype=bool)
 q=np.asarray([base_predictions[s]for s in source],dtype=np.int64)
 expected=(q,(q+1)%8,(3*q+2)%8,np.zeros_like(q),(q+1)%8,(3*q+2)%8,(3*q+5)%8)[suffix]
 denominator=int(ownbase.sum());joint=int((right&ownbase).sum())
 return dict(rows=len(ids),correct=int(right.sum()),accuracy=float(right.mean())if len(ids)else None,
  joint_base_correct_and_suffix_correct=joint,joint_denominator=len(ids),joint_accuracy=joint/len(ids)if len(ids)else None,
  prediction_consistency_count=int((pred[ids]==expected).sum()),prediction_consistency=float((pred[ids]==expected).mean())if len(ids)else None,
  own_base_correct_count=denominator,conditional_suffix_correct_numerator=joint,conditional_suffix_accuracy_given_own_base_correct=joint/denominator if denominator else None,
  source_train_indices=source,own_base_correct_source_train_indices=[s for s,correct in zip(source,ownbase)if correct],
  conditional_correct_source_train_indices=[s for s,correct in zip(source,right&ownbase)if correct])
def main():
 p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);tick=time.perf_counter();device=torch.device('cuda:0');torch.set_num_threads(1)
 manifest=json.loads((a.inputs/'manifest.json').read_text());cp=a.inputs/'resume.pt'
 if manifest.get('schema')!='extended-live-closure-evaluator-inputs-v1':raise ValueError('exact new manifest schema required')
 if sha(cp)!=manifest['checkpoint_sha256']:raise ValueError('actual OWN NN checksum required')
 for leaf,digest in manifest['source_sha256'].items():
  if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('exact NN/evaluator source differs')
 # NN provenance and restoration precede all new label/metadata reads.
 payload=torch.load(cp,map_location='cpu',weights_only=False)
 if payload.get('schema')!=manifest['checkpoint_schema']or payload['stage']!=3 or payload['step']!=600 or payload['stage_complete']is not True or payload['source_sha256']!=manifest['parent_source_sha256']:raise ValueError('exact OWN schema/completed fourth stage required')
 if payload.get('block_checkpoint')is not None:
  import continuous_primitive_block as core
  for key,value in manifest['block_checkpoint_header'].items():
   if payload['block_checkpoint'].get(key)!=value:raise ValueError('exact checkpoint execution policy required: '+key)
  model=core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint']).to(device);family='block'
 else:
  family=payload['args']['arm'];model=(old.TerminalTransformer if family=='transformer'else old.TerminalGRU)(**payload['baseline_spec']).to(device)
 model.load_state_dict(payload['model_state'],strict=True)
 if not old.tensor_state_equal(model.state_dict(),payload['model_state']):raise ValueError('actual restored NN differs')
 P=sum(v.numel()for v in model.parameters())
 if P!=manifest['P']:raise ValueError('all parameter count differs')
 if family!=manifest['family']:raise ValueError('actual NN family differs')
 saved_optimizer_bytes=old.tensor_bytes(payload['optimizer']);teacher=payload.get('teacher')
 saved_replay_array_bytes=old.tensor_bytes(payload.get('replay_state',{}));del payload
 carry=logical_carry(model,family)
 before={n:v.detach().cpu().clone()for n,v in model.state_dict().items()};model.eval()
 for v in model.parameters():v.requires_grad_(False)
 if sha(a.inputs/'closure.npz')!=manifest['data_sha256']:raise ValueError('frozen data differs')
 if sha(a.inputs/'metadata.json')!=manifest['metadata_sha256']:raise ValueError('frozen metadata differs')
 metadata=json.loads((a.inputs/'metadata.json').read_text())
 if metadata.get('schema')!='extended-live-closure-development-v1'or metadata['file_sha256']!=manifest['data_sha256']or metadata['rows']!=3584 or metadata['maximum_length']>50:raise ValueError('fixed new data metadata required')
 with np.load(a.inputs/'closure.npz',allow_pickle=False)as z:data={k:z[k]for k in z.files}
 n=len(data['initial'])
 if n!=3584 or data['tokens'].shape!=(3584,64):raise ValueError('fixed3584 rows')
 if not np.array_equal(data['within_original48'],(data['lengths']<=48).astype(np.int64)):raise ValueError('length-only stratum changed')
 expected_order=np.asarray([(c,b,s)for c in range(2)for b in range(256)for s in range(7)])
 if not np.array_equal(np.column_stack([data['condition'],data['base_local_index'],data['suffix']]),expected_order):raise ValueError('fixed condition/base/suffix flatten order differs')
 outputs=[];carry_outputs={key:[]for key in('h','controller','keys','values')};carry_D2H_seconds=0.;costs=dict(rows=0,active_token_steps=0,forward_MAC_proxy=0,forward_wall_seconds=0.,CUDA_stream_seconds=0.,H2D_wall_seconds=0.)
 with torch.no_grad():
  for start in range(0,n,64):
   if time.perf_counter()-tick>800:raise TimeoutError('bounded800s')
   ids=np.arange(start,start+64);t=time.perf_counter();b,l,active=old.batch(data,ids,device);old.synchronize(device);costs['H2D_wall_seconds']+=time.perf_counter()-t
   t=time.perf_counter();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
   if family!='block':raise ValueError('only the two fixed actual Block NNs are admitted')
   value,final_carry=model.forward_value(b['initial'],b['tokens'],b['lengths'],execution='soft')
   end.record();old.synchronize(device);costs['forward_wall_seconds']+=time.perf_counter()-t;costs['CUDA_stream_seconds']+=begin.elapsed_time(end)/1000.
   if value.shape!=(64,8)or not bool(torch.isfinite(value).all()):raise ValueError('finite outputs required')
   copy_tick=time.perf_counter();outputs.append(value.cpu().numpy())
   for key in carry_outputs:carry_outputs[key].append(final_carry[key].detach().cpu().numpy())
   carry_D2H_seconds+=time.perf_counter()-copy_tick;costs['rows']+=64;costs['active_token_steps']+=active
   costs['forward_MAC_proxy']+=(model.resource_report(batch_size=64,sequence_length=l,active_steps=active)if family=='block'else model.proxy(64,l,active))['proxy_forward_macs']
 logp=np.concatenate(outputs);pred=logp.argmax(-1);scores={}
 for condition in(0,1):
  mask=data['condition']==condition;base=np.flatnonzero(mask&(data['suffix']==0));by_source={int(data['source_train_index'][i]):int(pred[i])for i in base};by_correct={int(data['source_train_index'][i]):bool(pred[i]==data['targets'][i])for i in base}
  scores[str(condition)]={}
  for suffix,name in enumerate(metadata['suffix_names']):
   ids=np.flatnonzero(mask&(data['suffix']==suffix));right=pred[ids]==data['targets'][ids];inside=data['within_original48'][ids].astype(bool)
   metric=subset_metrics(ids,data,pred,by_source,by_correct,suffix)
   metric.update(within48_count=int(inside.sum()),within48_accuracy=float(right[inside].mean())if inside.any()else None,
    within48=subset_metrics(ids[inside],data,pred,by_source,by_correct,suffix),
    above48=subset_metrics(ids[~inside],data,pred,by_source,by_correct,suffix),
    cache_overlap=manifest['profile_cache_trained']and name in('SET0_STEP','SET1_STEP'),
    continuation_role='base case'if suffix==0 else('TRAINED four-word cache continuation'if manifest['profile_cache_trained']and name in('SET0_STEP','SET1_STEP')else'outside profile four-word cache'),
    caution='empty consistency is tautological'if suffix==0 else('CLEAR consistency equals CLEAR accuracy'if suffix==3 else'functional consistency can hold with an incorrect base prediction'))
   scores[str(condition)][name]=metric
 carry_arrays={key:np.concatenate(values)for key,values in carry_outputs.items()}
 if any(not np.isfinite(value).all()for value in carry_arrays.values()):raise ValueError('finite actual finalcarry required')
 geometry_started=time.perf_counter();carry_metrics={}
 for key,value in carry_arrays.items():
  norms=np.linalg.norm(value.reshape(n,-1).astype(np.float64),axis=1)
  carry_metrics[key]=dict(shape=list(value.shape),array_bytes=value.nbytes,L2_mean=float(norms.mean()),L2_p95=float(np.quantile(norms,.95)),L2_max=float(norms.max()),all_finite=True)
 # Evaluation-only geometry uses the model's own posterior and learned input
 # basis. It neither feeds new labels into execution nor resets any context.
 raw_basis=model.encoder.weight.detach().cpu().numpy().astype(np.float64);basis=raw_basis/np.maximum(np.linalg.norm(raw_basis,axis=-1,keepdims=True),1e-6)*model.internal_states**.5
 probabilities=np.exp(logp.astype(np.float64));projected=probabilities@basis;h=carry_arrays['h'].astype(np.float64)
 R_h=projected/np.maximum(np.linalg.norm(projected,axis=-1,keepdims=True),1e-6)*np.linalg.norm(h,axis=-1,keepdims=True)
 bridge_distance=np.linalg.norm(R_h-h,axis=-1);correct=pred==data['targets']
 bridge_alpha=float(model.value_bridge_alpha.detach().cpu())if hasattr(model,'value_bridge_alpha')else None
 bridge_strength=float(np.tanh(bridge_alpha))if bridge_alpha is not None else None
 carry_geometry=dict(scope='finalcarry values only; terminal label is not full state/context; R(h) uses own finalposterior, not the prebridge candidate; no proved retraction/idempotence',
  own_posterior_R_h_distance_mean=float(bridge_distance.mean()),own_posterior_R_h_distance_p95=float(np.quantile(bridge_distance,.95)),
  own_posterior_R_h_distance_on_correct_mean=float(bridge_distance[correct].mean())if correct.any()else None,
  value_bridge_alpha=bridge_alpha,value_bridge_strength=bridge_strength,geometry_CPU_wall_seconds=time.perf_counter()-geometry_started)
 with(a.out/'raw-outputs.npz').open('xb')as f:np.savez_compressed(f,log_probabilities=logp,predictions=pred,targets=data['targets'],source_train_index=data['source_train_index'],base_local_index=data['base_local_index'],condition=data['condition'],suffix=data['suffix'],within_original48=data['within_original48'],lengths=data['lengths'],own_posterior_R_h_distance=bridge_distance,**{'carry_'+key:value for key,value in carry_arrays.items()})
 if not old.tensor_state_equal(before,model.state_dict())or sha(cp)!=manifest['checkpoint_sha256']:raise ValueError('NN or checkpoint changed')
 for leaf,digest in manifest['source_sha256'].items():
  if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('execution source changed after evaluation')
 for leaf,digest in [('closure.npz',manifest['data_sha256']),('metadata.json',manifest['metadata_sha256'])]:
  if sha(a.inputs/leaf)!=digest:raise ValueError('frozen data changed after evaluation')
 result=dict(status='ACTUAL_EXTENDED_CLOSURE_FROZEN_EVAL_COMPLETE',model=manifest['model'],P=P,parameter_bytes=sum(v.numel()*v.element_size()for v in model.parameters()),checkpoint_bytes=cp.stat().st_size,saved_optimizer_bytes=saved_optimizer_bytes,live_optimizer_bytes=0,model_unchanged=True,SGD_updates=0,scores=scores,costs=costs,driver_wall_seconds=time.perf_counter()-tick,cpu_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),source_sha256=manifest['source_sha256'],parent_checkpoint_sha256=manifest['checkpoint_sha256'],data_sha256=manifest['data_sha256'],raw_sha256=sha(a.out/'raw-outputs.npz'),training_lineage=manifest['training_lineage'],formal_Primary=False,scope='development; cache-overlap closures are trained for newprofilearms; NOP/composed/CLEAR are outside4wordcache; full and <=48 strata kept, not matched-quality architecture victory')
 result.update(actual_final_carry_metrics=carry_metrics,carry_geometry=carry_geometry,raw_carry_array_bytes=sum(v.nbytes for v in carry_arrays.values()),raw_output_and_carry_D2H_wall_seconds=carry_D2H_seconds,value_bridge_alpha=bridge_alpha,value_bridge_strength=bridge_strength,carry_export_scope='same singleforward forward_value returns alreadycomputed carry; no additionalNNforward, finalcarry rawarrays retained; geometricCPUwork andD2H separatelytimed andwhole drivercharged',condition_names={'0':'MIX15','1':'last15 replaced by READ10'},checkpoint_schema=manifest['checkpoint_schema'],block_checkpoint_header=manifest['block_checkpoint_header'],logical_carry=carry,
  registered_buffer_bytes=sum(v.numel()*v.element_size()for v in model.buffers()),source_code_bytes=sum(Path(__file__).with_name(leaf).stat().st_size for leaf in manifest['source_sha256']),
  data_file_bytes=(a.inputs/'closure.npz').stat().st_size,data_array_bytes=sum(v.nbytes for v in data.values()),metadata_bytes=(a.inputs/'metadata.json').stat().st_size,
  input_manifest_bytes=(a.inputs/'manifest.json').stat().st_size,raw_output_bytes=(a.out/'raw-outputs.npz').stat().st_size,
  saved_replay_tensor_bytes=saved_replay_array_bytes,saved_replay_storage_note='tensor-only counter; NumPy replay arrays separately reported in actual parent resources, never infer zero from tensor counter',
  parent_resources=manifest['parent_resources'],parent_teacher_cache=teacher,live_teacher_cache_bytes=0,
  teacher_cache_scope='OWN saved teacher metadata retained for accounting; no teacher/cache target lookup or auxiliary fit during this evaluation',
  generation_paid_lineage=manifest['generation_paid_lineage'],source_and_data_rehashed_after=True,full_training_FLOPs=None,
  metric_scope='each condition256basecases; conditional own-base-correct sets differ by NN and are not an unmatched causal comparison; full/within48/above48 denominators explicit')
 (a.out/'result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(dict(status=result['status'],model=manifest['model'],P=P)))
if __name__=='__main__':main()
