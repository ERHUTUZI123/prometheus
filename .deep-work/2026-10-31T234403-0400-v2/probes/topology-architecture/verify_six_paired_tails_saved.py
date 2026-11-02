"""Joint saved-array verification, summaries and curve CSV; zero NN execution.
Reviewer authored the ordinary-baseline wrappers, not an independent source author.
"""
import csv,hashlib,json,math,time
from pathlib import Path
from analyze_posterior_valid_slot_values import load_npz
from verify_paired_baseline_tails_saved import verify as baseline_verify,metric
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def typed(sampling):
 name=f'parameter-typed-paired-tail-{sampling.replace("_","-")}-n8-seed0-600-v1';f=P/'runs'/name/'learn';v=P/'versions'/('code-'+name);r=json.loads((f/'result.json').read_text());c=json.loads((v/'launch-config.json').read_text());assert r['status']=='ACTUAL_TYPED_PAIRED_TAIL_STAGE3_600_COMPLETE'and r['sampling']==sampling and(r['actual_new_updates_this_process'],r['historical_updates_shared'],r['old_updates_rerun'])==(600,1800,0)and r['NN_unchanged_after_TEST']and r['all20_grad_storage_stable'];assert r['source_sha256']==c['manifest']['source_sha256']==c['source_sha256'];assert all(sha(v/leaf)==digest for leaf,digest in c['source_sha256'].items());CPU=P/'runs'/c['CPU_gate_run']/'check/result.json';assert sha(CPU)==c['CPU_proof_sha256']
 for entry in c['input_files'].values():path=P/'runs'/entry['remote_relative'];assert sha(path)==entry['sha256']and path.stat().st_size==entry['bytes']
 data={k:load_npz(P/'runs'/entry['remote_relative'])for k,entry in c['manifest']['files'].items()};assert len(data)==20
 rows=[json.loads(x)for x in(f/'records.jsonl').read_text().splitlines()];u=[x for x in rows if x['kind']=='actual_update'];assert[x['step']for x in u]==list(range(1,601));assert all(math.isfinite(x['terminal_NLL'])and math.isfinite(x['preclip_gradient'])for x in u);sums={key:sum(x[key]for x in u)for key in('training_wall_seconds','active_forward_MAC_proxy','padded_forward_MAC_proxy','active_token_steps','padded_token_steps')};assert all(r['costs'][k]==z for k,z in sums.items());assert(r['costs']['SGD_updates'],r['costs']['current_training_label_reads'],r['costs']['old_replay_training_label_reads'])==(600,28800,9600)
 rawschedule=load_npz(f/'stage3-common-indices.npz')['indices'][1];source=load_npz(P/'runs'/c['input_files']['paired-stage3-indices.npz']['remote_relative']);assert rawschedule==source['indices'if sampling=='paired'else'indices_same_pool_shuffled'][1];draws=load_npz(f/'stage3-replay-draws.npz')['buffer_indices'][1];original=P/'runs/parameter-gru32-n8-seed0-fourstage-v1/learn/stage3-replay-draws.npz';assert draws==load_npz(original)['buffer_indices'][1];hits=[0]*4096
 for step in range(600):
  for row in rawschedule[step*64:step*64+48]:hits[row]+=1
 tests={};finalvalid={};predictions=0
 def check(file,target,reported,modes):
  raw=load_npz(file);assert list(raw['row_index'][1])==list(range(len(target['targets'][1])))
  for key in('targets','initial','lengths'):assert raw[key][1]==target[key][1]
  values={}
  for mode in modes:
   m=metric(raw[mode+'_logp'][1],raw['targets'][1]);assert m['accuracy']==reported[mode]['accuracy']and abs(m['NLL']-reported[mode]['terminal_NLL'])<2e-6 and m['max_logsumexp_abs']<2e-6;values[mode]=m
  return dict(sha256=sha(file),modes=values)
 for s in range(4):
  for split in('test_iid','test_long','test_unseen'):
   key=f'{s}:{split}';tests[key]=check(f/f'final-task{s}-{split}-predictions.npz',data[key],r['tests'][str(s)][split],('soft','native'));predictions+=len(data[key]['targets'][1])*2
  finalvalid[str(s)]=check(f/f'stage3-valid-task{s}-predictions.npz',data[f'{s}:valid'],r['curves'][-1]['valid'][str(s)],('soft',))
 assert predictions==12288
 curve_rows=[x for x in rows if x['kind']=='actual_VALID_curve'];assert[x['step']for x in r['curves']]==list(range(0,601,100))and len(curve_rows)==7
 for curve,logged in zip(r['curves'],curve_rows):assert curve['step']==logged['step']and curve['valid']==logged['valid']
 for key in('checkpoint','premaintenance_safety_checkpoint'):
  entry=r[key];file=f/entry['file'];assert sha(file)==entry['sha256']and file.stat().st_size==entry['bytes']
 freeze=json.loads((f/'ALL_TRAINING_FROZEN_BEFORE_TEST.json').read_text());assert freeze['actual_new_updates']==600;assert min(i for i,x in enumerate(rows)if x['kind']=='actual_input_opened'and x['split'].startswith('test'))>max(i for i,x in enumerate(rows)if x['kind']=='actual_update');assert r['replay_maintenance']['seen']==124800 and r['replay_maintenance']['bookkeeping_counters']['draws']==28800
 setup=r['setup'];assert(r['actual_graph_capture_count'],setup['training_FB_calls'],setup['warmup_FB_calls'],setup['reference_diagnostic_FB_calls'],setup['graph_diagnostic_FB_calls'])==(1,600,3,7,7);receipts=[setup['admission']]+setup['diagnostics'];assert len(receipts)==7;grad=[];pairs=0;maxout=maxgrad=0.
 for entry in receipts:
  file=f/entry['raw_file'];assert sha(file)==entry['raw_sha256']and file.stat().st_size==entry['raw_bytes'];raw=load_npz(file);names=json.loads(bytes(raw['gradient_names_json'][1]).decode());assert len(names)==20;count=0;maximum=0.
  for key,(shape,left)in raw.items():
   if not key.startswith('reference_'):continue
   right=raw['candidate_'+key[10:]];assert shape==right[0];right=right[1];atol,rtol=(2e-5,2e-4)if key.startswith('reference_grad_')else(2e-5,2e-5);diff=0.
   for x,y in zip(left,right):assert math.isfinite(x)and math.isfinite(y)and abs(x-y)<=atol+rtol*abs(y);diff=max(diff,abs(x-y))
   count+=len(left)
   if key=='reference_output':maxout=max(maxout,diff)
   elif key.startswith('reference_grad_'):maxgrad=max(maxgrad,diff)
  assert count==11504;pairs+=count;grad.append(dict(file=file.name,sha256=sha(file),scalar_pairs=count))
 assert pairs==80528;receipt=json.loads((P/(name+'-coordinator-receipt.json')).read_text());assert receipt['returncode']==0;cost=r['costs'];res=r['resources'];peaks=[res]+[];alloc=[res['cuda_peak_allocated_bytes']];reserved=[res['cuda_peak_reserved_bytes']]
 for snap in(setup['peak_before_reset'],setup['capture']['memory_before'],setup['capture']['memory_after']):alloc.append(snap['peak_allocated']);reserved.append(snap['peak_reserved'])
 costs=dict(P=res['parameters'],NN_bytes=res['parameter_bytes'],Adam_bytes=res['optimizer_tensor_bytes'],replay_bytes=res['replay_bytes'],registered_buffer_bytes=res['registered_buffer_bytes'],training_wall_seconds=cost['training_wall_seconds'],science_wall_seconds=r['driver_wall_seconds'],science_CPU_seconds=r['driver_CPU_seconds'],local_cold_seconds=receipt['wall_seconds'],maintenance_seconds=r['replay_maintenance']['maintenance_wall_seconds'],TEST_seconds=sum(x['wall_seconds']for x in r['test_costs']),inference_seconds=r['inference_total_seconds'],CUDA_peak_allocated_bytes=max(alloc),CUDA_peak_reserved_bytes=max(reserved),CPU_peak_RSS_bytes=res['cpu_peak_RSS_bytes'],checkpoint_payload_bytes=r['checkpoint']['bytes']+r['premaintenance_safety_checkpoint']['bytes'],active_steps=cost['active_token_steps'],padded_steps=cost['padded_token_steps'],forward_MAC_proxy=cost['padded_forward_MAC_proxy'],active_forward_MAC_proxy=cost['active_forward_MAC_proxy'],setup_seconds=setup['setup_seconds'],actual_training_and_diagnostic_FB=617,full_training_FLOPs=None,provider_effective_compute=None,logical_carry_scalars=320,logical_carry_bytes=1280,private_pool_exact_bytes=None)
 return dict(run=name,status='PASS_INDEPENDENT_SAVED_TYPED_TAIL_RAW',result_sha256=sha(f/'result.json'),records_sha256=sha(f/'records.jsonl'),source_sha256=r['source_sha256'],CPU_proof_sha256=sha(CPU),parent_checkpoint_sha256=r['parent_stage2_checkpoint_sha256'],post_growth=dict(post_growth_initial_theta_sha256=r['post_growth_initial_theta_sha256'],new_token_row_sha256=r['new_token_row_sha256']),current_exposure_vector=hits,replay_restore_digest=r['restore_and_growth']['parent_replay_sha256'],replay_final_digest=r['replay_maintenance']['after_sha256'],all12_TEST=tests,stored_TEST_predictions_checked=predictions,final4_VALID=finalvalid,curve_scope='all7cadence/28aggregate fields agreeJSONL, onlyfinal4NNperexampleVALIDvectorssaved; earlycurvesproduceraggregates',all7_gradient_raw=grad,gradient_scalar_pairs=pairs,max_output_abs=maxout,max_gradient_abs=maxgrad,old_updates_repeated=0,costs=costs,native_scope='hard7action/expert, softmemoryattention; onecandidatephysicallycomputedperrow evenmemoryaction; maintrainingsoftonly')
def main():
 start=time.perf_counter();cpu=time.process_time();models={};summary=[];curve=[];shared_data=P/'runs/parameter-paired-train-indices-n8-stage3-seed0-v1/generate/result.json';generation=json.loads(shared_data.read_text());reference_hits=None;reference_replay=None
 for model in('gru32','transformer32','typed'):
  pairs={sampler:(typed(sampler)if model=='typed'else baseline_verify(model,sampler))for sampler in('paired','same_pool_shuffle')};left,right=pairs.values();assert left['parent_checkpoint_sha256']==right['parent_checkpoint_sha256']and left['post_growth']['post_growth_initial_theta_sha256']==right['post_growth']['post_growth_initial_theta_sha256']and left['post_growth']['new_token_row_sha256']==right['post_growth']['new_token_row_sha256']and left['current_exposure_vector']==right['current_exposure_vector']and left['replay_restore_digest']==right['replay_restore_digest']and left['costs']['active_steps']==right['costs']['active_steps'];assert left['costs']['active_steps']==1013912
  if reference_hits is None:reference_hits=left['current_exposure_vector'];reference_replay=left['replay_restore_digest']
  assert left['current_exposure_vector']==reference_hits and left['replay_restore_digest']==reference_replay
  models[model]=pairs
  for sampler,z in pairs.items():
   r=json.loads((P/'runs'/z['run']/'learn/result.json').read_text());row=dict(model=model,sampling=sampler,run=z['run'],parent_checkpoint_sha256=z['parent_checkpoint_sha256'],post_growth_initial_theta_sha256=z['post_growth']['post_growth_initial_theta_sha256'],new_token_row_sha256=z['post_growth']['new_token_row_sha256'],new_VALID=r['curves'][-1]['valid']['3']['soft']['accuracy'],**z['costs'])
   for task in range(4):
    for split in('test_iid','test_long','test_unseen'):
     test=z['all12_TEST'][f'{task}:{split}'];soft=test['modes']['soft']if model=='typed'else test;row[f'task{task}_{split}_correct']=soft['correct'];row[f'task{task}_{split}_accuracy']=soft['accuracy'];row[f'task{task}_{split}_NLL']=soft['NLL']
   summary.append(row)
   for c in r['curves']:
    for task in range(4):m=c['valid'][str(task)]['soft'];curve.append(dict(model=model,sampling=sampler,run=z['run'],update=c['step'],task=task,split='VALID',rows=m['rows'],accuracy=m['accuracy'],NLL=m['terminal_NLL'],evidence='perexampleindependentlyrecomputed'if model!='typed'or c['step']==600 else'produceraggregateJSONLexactagreement'))
 totals={key:sum(x[key]for x in summary)for key in('training_wall_seconds','science_wall_seconds','local_cold_seconds','checkpoint_payload_bytes')};totals['new_tail_SGD_updates']=3600;totals['history_unique_effective_parent_updates']=5400
 shared={}
 for key,name in [('pair_formation','parameter-paired-train-indices-n8-stage3-seed0-v1'),('baseline_CPU_resume','parameter-paired-baseline-resume-cpu-v1'),('typed_CPU_resume','parameter-typed-paired-tail-resume-cpu-v1')]:
  receipt=P/(name+'-coordinator-receipt.json');shared[key]=dict(run=name,local_cold_seconds=json.loads(receipt.read_text())['wall_seconds'],scope='chargedONCEacrossrelevanttails, notpermodelrepeat')
 shared['pair_formation'].update(science_seconds=generation['driver_wall_seconds'],array_payload_bytes=generation['serialization']['array_payload_bytes'],schedule_bytes=generation['schedule']['bytes']);totals['all6_plus_once_shared_CPU_local_cold_seconds']=totals['local_cold_seconds']+sum(x['local_cold_seconds']for x in shared.values());inventory=P/'paired-tail-stage2-parent-inventory-v1.json';out=dict(status='PASS_ALL_SIX_PAIRED_SAMEPOOL_SAVED_RAW',source_script_sha256=sha(Path(__file__)),models=models,summary=summary,totals=totals,shared_costs=shared,history_inventory_sha256=sha(inventory),original_random_tail_and_failed_attempt_costs_preserved='inventory/fullfourstage raw retained; nolearningfailureorpriorconfigcost erased',interpretation='singledevelopmentseed, grouping+temporalorderingjointintervention; noaccuracygate/TESTselection; noPrimary/architectureadvantage; effectsonoldlongnotpureforgetting; activetotalssame butbaselineactualpaddingdiffers; nestedtimesshouldnotsum',limitations=['reviewerauthoredbaselinewrappers; independentrawarithmeticisnotsourceauthorshipindependence','baseline nativealiasesoneNNforward; typedsoft/nativeactualseparatepasses','typedearlyVALIDrowvectorsnotstored: onlyduplicateJSONL/source/cadence checked','NN/checkpointsnotdeserializedbyreviewer; fulltrainingFLOPs/GPUbusy/driver-context/privatepool/energy/providerunknown','finalreplayreservoircanlegitimatelydifferunderchangedstreamorder'],new_NN_queries=0,SGD_updates=0,analysis_wall_seconds=time.perf_counter()-start,analysis_CPU_seconds=time.process_time()-cpu)
 for suffix,items in [('summary',summary),('curves',curve)]:
  file=P/f'six-paired-tails-{suffix}-v1.csv';fields=list(dict.fromkeys(k for item in items for k in item))
  with file.open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(items)
  out[suffix+'_CSV']=dict(file=file.name,sha256=sha(file),rows=len(items))
 path=P/'six-paired-tails-independent-raw-checks-v1.json'
 with path.open('x')as f:json.dump(out,f,sort_keys=True,indent=2);f.write('\n')
 print(path.name,sha(path));print('TOTAL',totals,'SHARED',shared)
 for row in summary:print(row['model'],row['sampling'],'VALID',row['new_VALID'],'newTESTcounts',[row[f'task3_{s}_correct']for s in('test_iid','test_long','test_unseen')],'train/science/cold',row['training_wall_seconds'],row['science_wall_seconds'],row['local_cold_seconds'])
if __name__=='__main__':main()
