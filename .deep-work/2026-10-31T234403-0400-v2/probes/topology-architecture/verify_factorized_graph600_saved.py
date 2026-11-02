"""Stdlib-only verification of locally synced Graph600 vs retained v3 raw.

No SSH, checkpoint deserialization, NumPy/Torch, model or training execution.
Schedule/source/count/cost checks are strict; trajectory numerical differences
are reported, never required bitwise equal and never used as an accuracy gate.
"""
import argparse,array,ast,hashlib,json,math,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
NEW='parameter-factorized-graph-stage3-anchor0-n8-seed0-600-v1'
OLD='parameter-factorized-value-anchor0-n8-seed0-fourstage-v3'
EXPECTED_INDICES='a840c72965717b198837fb893c288d30c6d20809ecba22439cbb8d850c787007'
EXPECTED_REPLAY='c00780e9f7d57f4c26e3e472da8563b2aaab65d00926aaa55b0a4581fbfa9ab1'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def npz(path):
 result={}
 with zipfile.ZipFile(path)as stored:
  for leaf in stored.namelist():
   raw=stored.read(leaf)
   if raw[:6]!=b'\x93NUMPY':raise ValueError('numeric NPY only')
   major=raw[6]
   if major not in(1,2,3):raise ValueError('unsupported NPY version')
   offset=10 if major==1 else 12;n=int.from_bytes(raw[8:offset],'little');header=ast.literal_eval(raw[offset:offset+n].decode());data=raw[offset+n:]
   if header['fortran_order']:raise ValueError('expected C-order saved arrays')
   dtype=header['descr'];code={'<f4':'f','<f8':'d','<i8':'q','<i4':'i','|u1':'B','|b1':'B'}.get(dtype)
   if code is None:raise ValueError('fixed numeric dtype required: '+dtype)
   values=array.array(code);values.frombytes(data)
   if sys.byteorder!='little'and values.itemsize>1:values.byteswap()
   if len(values)!=math.prod(header['shape']):raise ValueError('array shape/bytes mismatch')
   result[leaf.removesuffix('.npy')]=dict(shape=header['shape'],dtype=dtype,values=values,raw_bytes_sha256=hashlib.sha256(data).hexdigest())
 return result

def percentile(values,q):
 values=sorted(values);x=(len(values)-1)*q;i=int(x);j=min(i+1,len(values)-1);return values[i]+(values[j]-values[i])*(x-i)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--new-run',type=Path,default=ROOT/'runs'/NEW);p.add_argument('--original-run',type=Path,default=ROOT/'runs'/OLD);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists()or a.out.is_symlink():raise FileExistsError('new verification receipt only')
 new=a.new_run/'learn';old=a.original_run/'learn';result=json.loads((new/'result.json').read_text());previous=json.loads((old/'result.json').read_text())
 if result.get('status')!='ACTUAL_FACTORIZED_GRAPH_STAGE3_600_COMPLETE'or result.get('actual_new_updates_this_process')!=600:raise ValueError('actual completed Graph600 result required; setup is not training')
 if result.get('parent_stage2_checkpoint_sha256')!='9bb47e660e4ea67d0bf1a06e93e418f0c3fa50760fa76ff067d3a0c954a6c40f':raise ValueError('same exact stage2 parent required')
 rows=[json.loads(line)for line in(new/'records.jsonl').read_text().splitlines()];oldrows=[json.loads(line)for line in(old/'records.jsonl').read_text().splitlines()];updates=[r for r in rows if r['kind']=='actual_update'];reference=[r for r in oldrows if r['kind']=='actual_update'and r['stage']==3]
 if [r['step']for r in updates]!=list(range(1,601))or[r['step']for r in reference]!=list(range(1,601)):raise ValueError('exact actual600 per-step streams required')
 schedule={}
 for leaf,key,shape,expected in(('stage3-common-indices.npz','indices',(600,64),EXPECTED_INDICES),('stage3-replay-draws.npz','buffer_indices',(600,16),EXPECTED_REPLAY)):
  left=npz(new/leaf)[key];right=npz(old/leaf)[key]
  if left['shape']!=shape or right['shape']!=shape or left['dtype']!='<i8'or right['dtype']!='<i8'or left['raw_bytes_sha256']!=expected or right['raw_bytes_sha256']!=expected or left['values']!=right['values']:raise ValueError('original actual schedule differs: '+key)
  schedule[key]=dict(shape=shape,payload_sha256=expected,elementwise_equal=True,new_file_sha256=sha(new/leaf),original_file_sha256=sha(old/leaf),ZIPfile_SHA_is_not_schedule_identity=True)
  if key=='indices':
   if any(not 0<=v<4096 for v in left['values']):raise ValueError('current indices outside TRAIN')
  elif any(not 0<=v<128 for v in left['values']):raise ValueError('replay index outside128')
 costs=result['costs'];oldcosts=next(r['costs']for r in previous['reports']if r['stage']==3);sum_new_time=math.fsum(r['training_wall_seconds']for r in updates);sum_old_time=math.fsum(r['training_wall_seconds']for r in reference)
 checks=dict(actual600=True,SGD600=costs['SGD_updates']==600,new_source_pins=False,same_schedule=True,same_active_per_step=all(r['active_token_steps']==s['active_token_steps']for r,s in zip(updates,reference)),fixed_padded3072_per_step=all(r['padded_token_steps']==3072 for r in updates),padded_MAC_sum=sum(r['padded_forward_MAC_proxy']for r in updates)==costs['padded_forward_MAC_proxy'],active_MAC_sum=sum(r['original_active_forward_MAC_proxy']for r in updates)==costs['original_active_forward_MAC_proxy'],active_MAC_equals_original_perstep=all(r['original_active_forward_MAC_proxy']==s['forward_MAC_proxy']for r,s in zip(updates,reference)),active_count_sum=sum(r['active_token_steps']for r in updates)==costs['active_token_steps'],padded_count_sum=sum(r['padded_token_steps']for r in updates)==costs['padded_token_steps']==600*3072,current_label_reads=costs['current_training_label_reads']==600*48,old_label_reads=costs['old_replay_training_label_reads']==600*16,training_graph_replays=costs['actual_training_graph_FB_calls']==600,reported_times_match=abs(sum_new_time-costs['training_wall_seconds'])<1e-9 and abs(sum_old_time-oldcosts['training_wall_seconds'])<1e-9,one_graph=result.get('actual_graph_count')==1,NN_storage_stable=result.get('all27_grad_storage_stable')is True,no_profile=result.get('teacher_or_profile_loss')is False,no_score_gate=result.get('accuracy_or_speed_gate')is False)
 version=ROOT/'versions'/('code-'+NEW)
 for leaf,digest in result['source_sha256'].items():
  source=version/leaf
  if sha(source)!=digest:raise ValueError('actual source differs from frozen bytes')
  ast.parse(source.read_bytes())
 checks['new_source_pins']=True
 for receipt in(result['checkpoint'],result['premaintenance_safety_checkpoint']):
  path=new/receipt['file']
  if sha(path)!=receipt['sha256']or path.stat().st_size!=receipt['bytes']:raise ValueError('actual safety/final checkpoint bytes differ')
 if not all(checks.values()):raise ValueError('schedule/count/source/cost contract failure: '+json.dumps(checks,sort_keys=True))
 differences=[]
 for r,s in zip(updates,reference):
  values=[r['terminal_NLL'],s['terminal_NLL'],r['value_interface_loss'],s['value_interface_loss'],r['total_loss'],s['total_loss']]
  if not all(math.isfinite(x)for x in values):raise ValueError('nonfinite actual trajectory record')
  differences.append(dict(step=r['step'],NLL_new=r['terminal_NLL'],NLL_original=s['terminal_NLL'],NLL_difference=r['terminal_NLL']-s['terminal_NLL'],binding_difference=r['value_interface_loss']-s['value_interface_loss'],total_loss_difference=r['total_loss']-s['total_loss']))
 absolute=[abs(r['NLL_difference'])for r in differences]
 diagnostics=[r for r in rows if r['kind']=='actual_fixed_TRAIN_gradient_diagnostic']
 if [r['step']for r in diagnostics]!=[100,200,300,400,500,600]:raise ValueError('all six actual diagnostics required')
 admission=result['setup']['admission'];raw_pairs=[admission]+diagnostics;max_output=0.;max_grad=0.;checked_scalars=0
 for receipt in raw_pairs:
  path=new/receipt['raw_file']
  if sha(path)!=receipt['raw_sha256']:raise ValueError('actual gradient raw SHA changed')
  saved=npz(path);names=json.loads(bytes(saved['gradient_names_json']['values']).decode())
  if len(names)!=27:raise ValueError('all27 gradients required')
  for left,right,atol,rtol in([('reference_output','candidate_output',2e-5,2e-5),('reference_loss','candidate_loss',2e-5,2e-5)]+[(f'reference_grad_{i:02d}',f'candidate_grad_{i:02d}',2e-5,2e-4)for i in range(27)]):
   l,r=saved[left],saved[right]
   if l['shape']!=r['shape']:raise ValueError('gradient pair shape changed')
   values=list(zip(l['values'],r['values']));maximum=max(abs(v-w)for v,w in values)
   if not all(math.isfinite(v)and math.isfinite(w)and abs(v-w)<=atol+rtol*abs(w)for v,w in values):raise ValueError('prospective numeric tolerance failed: '+left)
   if left.startswith('reference_grad'):max_grad=max(max_grad,maximum)
   elif left=='reference_output':max_output=max(max_output,maximum)
   checked_scalars+=len(values)
 valid_rows=[r for r in rows if r['kind']=='actual_VALID_curve'];test_positions=[i for i,r in enumerate(rows)if r['kind']=='actual_input_opened'and r.get('split','').startswith('test')];last_update=max(i for i,r in enumerate(rows)if r['kind']=='actual_update')
 if len(test_positions)!=12 or min(test_positions)<=last_update:raise ValueError('actual TEST reads must follow all600 updates')
 frozen=json.loads((new/'ALL_TRAINING_FROZEN_BEFORE_TEST.json').read_text())
 if frozen.get('TEST_read')is not False or frozen.get('actual_new_updates')!=600 or frozen.get('checkpoint')!=result['checkpoint']:raise ValueError('actual frozen-before-TEST artifact differs')
 quality={str(stage):{split:{mode:dict(new=result['tests'][str(stage)][split][mode],original=previous['tests'][str(stage)][split][mode])for mode in('soft','native')}for split in('test_iid','test_long','test_unseen')}for stage in range(4)}
 summary=dict(status='VERIFIED_SAVED_GRAPH600_SCHEDULE_RAW_AND_COUNTS',no_model_or_SSH_execution=True,new_result_sha256=sha(new/'result.json'),original_result_sha256=sha(old/'result.json'),new_records_sha256=sha(new/'records.jsonl'),original_records_sha256=sha(old/'records.jsonl'),checks=checks,schedule=schedule,P=result['resources']['parameters'],source_sha256=result['source_sha256'],training=dict(new_wall_seconds=sum_new_time,original_wall_seconds=sum_old_time,original_over_new_wall_ratio=sum_old_time/sum_new_time,new_costs=costs,original_retained_stage3_costs=oldcosts,scope='wall sums cover recorded update loops, not complete cold lifecycle; no accuracy gate; extraGraph warmup/gradient checks/VALID/capture/setup must be paid separately; prior shared1800 not rerun'),trajectory=dict(bitwise_trajectory_required=False,NLL_max_abs=max(absolute),NLL_p50_abs=percentile(absolute,.5),NLL_p95_abs=percentile(absolute,.95),NLL_RMSE=math.sqrt(math.fsum(v*v for v in absolute)/600),NLL_final_difference=differences[-1]['NLL_difference'],NLL_exact_equal_steps=sum(v==0 for v in absolute),per_step=differences,scope='same actual samples/parent and real-valued update rule; different FP32 row shapes/reductions may diverge through Adam; no bitwise weight identity claim or numerical tolerance relaxation'),same_current_NN_gradient_checks=dict(diagnostic_steps=[r['step']for r in diagnostics],raw_pairs=len(raw_pairs),scalars_checked=checked_scalars,max_output_abs=max_output,max_gradient_abs=max_grad,atol_output=2e-5,rtol_output=2e-5,atol_gradient=2e-5,rtol_gradient=2e-4),VALID_steps=[r['step']for r in valid_rows],quality_producer_results=quality,quality_scope='actual frozen producer metrics, no per-example vectors retained by these drivers; not independently recomputed accuracy; do not use performance to change fixedbudget',actual_TEST_opened_after_all600=True,setup=result['setup'],checkpoint=result['checkpoint'],safety_checkpoint=result['premaintenance_safety_checkpoint'],maintenance=result['maintenance'],reload=result['reload'],inference_total_seconds=result['inference_total_seconds'],driver_wall_seconds=result['driver_wall_seconds'],driver_CPU_seconds=result['driver_CPU_seconds'],full_training_FLOPs=None,provider_effective_compute=None)
 if summary['VALID_steps']!=[0,100,200,300,400,500,600]:raise ValueError('original VALID schedule changed')
 with a.out.open('x')as f:json.dump(summary,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 print(json.dumps(dict(status=summary['status'],out=str(a.out),sha256=sha(a.out),actual_SGD=600,P=summary['P'],NLL_max_abs=max(absolute),NLL_RMSE=summary['trajectory']['NLL_RMSE'],update_loop_ratio=summary['training']['original_over_new_wall_ratio'],numeric_raw_pairs=len(raw_pairs)),sort_keys=True))
if __name__=='__main__':main()
