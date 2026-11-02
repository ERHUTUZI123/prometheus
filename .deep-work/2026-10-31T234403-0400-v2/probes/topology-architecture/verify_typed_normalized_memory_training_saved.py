"""Independent stdlib saved-source/schedule/all20gradient/cost verification.

No NN, Torch/NumPy, checkpoint deserialization, SSH or compilation. The
separate posthoc counterfactual parser owns per-example prediction semantics.
"""
import argparse,ast,hashlib,json,math,time
from pathlib import Path
from verify_factorized_graph600_saved import npz
P=Path(__file__).resolve().parent
CONTROL='parameter-typed-memory-actions-n8-seed0-fourstage-graph-v2'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--arm',choices=('A','B','AB'),required=True);a=parser.parse_args();start=time.perf_counter();cpu=time.process_time()
 name='parameter-typed-normalized-memory-'+a.arm.lower()+'-n8-seed0-fourstage-graph-v1';v=P/'versions'/('code-'+name);folder=P/'runs'/name/'learn';result=json.loads((folder/'result.json').read_text());config=json.loads((v/'launch-config.json').read_text());protocol=json.loads((P/(name+'-protocol.json')).read_text());flags=config['architecture_flags'];P16=10351 if flags['input_only_address']else 10991;P15=P16-16
 if result['status']!='ACTUAL_FOUR_STAGE_TYPED_NORMALIZED_MEMORY_GRAPH_COMPLETE'or result['actual_new_updates_this_process']!=2400 or result['architecture_arm']!=a.arm or result['architecture_flags']!=flags or not result['fresh_initial_NN']or result['parent_NN_loaded']or not result['NN_unchanged_after_TEST']or not result['all20_grad_storage_stable_within_capture']:raise ValueError('actualfixedfresh2400/flags/freeze contract differs')
 if result['source_sha256']!=config['source_sha256']or result['source_sha256']!=protocol['source_sha256']:raise ValueError('actualsource mapping differs')
 for leaf,digest in result['source_sha256'].items():
  if sha(v/leaf)!=digest:raise ValueError('actualfrozen source bytes differ')
  ast.parse((v/leaf).read_bytes())
 proof=P/'runs'/config['CPU_gate_run']/'check/result.json'
 if sha(proof)!=config['CPU_proof_sha256']:raise ValueError('actualCPUproof differs')
 CPU=json.loads(proof.read_text())
 if any(CPU['source_sha256'].get(k)!=h for k,h in config['common_source_sha256'].items()):raise ValueError('actualCPU/GPUsource differs')
 for spec in config['manifest']['files'].values():
  if sha(P/'runs'/spec['remote_relative'])!=spec['sha256']:raise ValueError('originaldata source differs')
 rows=[json.loads(line)for line in(folder/'records.jsonl').read_text().splitlines()];configuration=next(r for r in rows if r['kind']=='configuration');updates=[r for r in rows if r['kind']=='actual_update'];checks=[];samplers={};checkpoints=[]
 if len(updates)!=2400 or configuration['resources']['parameters']!=P15 or configuration['initial_theta_sha256']!=result['initial_theta_sha256']:raise ValueError('actualinit/updatecount/P differs')
 if not all(math.isfinite(r['terminal_NLL'])and math.isfinite(r['preclip_gradient'])for r in updates):raise ValueError('nonfinite actualtrainingrecord')
 for stage in range(4):
  report=result['reports'][stage];actual=[r for r in updates if r['stage']==stage];cost=report['costs']
  if [r['step']for r in actual]!=list(range(1,601))or cost['SGD_updates']!=600 or [r['step']for r in report['curves']]!=list(range(0,601,100)):raise ValueError('fixed600schedule/cadence differs')
  if report['resources']['parameters']!=(P15 if stage<3 else P16)or not report['reload']['exact_NN_Adam_RNG_replay_forward']:raise ValueError('actualP/reload differs')
  totals={key:sum(r[key]for r in actual)for key in('training_wall_seconds','padded_forward_MAC_proxy','active_forward_MAC_proxy','active_token_steps','padded_token_steps')}
  if any(totals[k]!=cost[k]for k in totals):raise ValueError('actualcost sums differ')
  expected_div=cost['padded_token_steps']*256 if flags['coverage_normalization']else 0;expected_clamp=cost['padded_token_steps']*8 if flags['coverage_normalization']else 0
  if cost['padded_coverage_divisions']!=expected_div or cost['padded_coverage_clamps']!=expected_clamp:raise ValueError('trainingcoverage operationcount differs')
  if report['replay_maintenance']['seen']!=38400+stage*28800 or report['replay_maintenance']['bookkeeping_counters']['draws']!=stage*9600:raise ValueError('replayexposures differ')
  for key in('checkpoint','premaintenance_safety_checkpoint'):
   cp=report[key];path=folder/cp['file']
   if sha(path)!=cp['sha256']or path.stat().st_size!=cp['bytes']:raise ValueError('actual NNcheckpoint differs')
   checkpoints.append(dict(stage=stage,kind=key,**cp))
  for leaf in [f'stage{stage}-common-indices.npz']+([f'stage{stage}-replay-draws.npz']if stage else[]):
   left=npz(folder/leaf);right=npz(P/'runs'/CONTROL/'learn'/leaf)
   if left.keys()!=right.keys()or any(left[k]['shape']!=right[k]['shape']or left[k]['raw_bytes_sha256']!=right[k]['raw_bytes_sha256']for k in left):raise ValueError('originalcontrol raw schedule differs')
   samplers[leaf]={k:d['raw_bytes_sha256']for k,d in left.items()}
  checks.append(dict(stage=stage,totals=totals,P=report['resources']['parameters'],coverage_divisions=expected_div,coverage_clamps=expected_clamp))
 files=sorted(folder.glob('*admission.npz'))+sorted(folder.glob('*all20.npz'));gradients=[];scalars=0;maxoutput=maxgrad=0.;names=None
 if len(files)!=28:raise ValueError('all28actualgradientrawgates required')
 for path in files:
  raw=npz(path);current=json.loads(bytes(raw['gradient_names_json']['values']).decode())
  if len(current)!=20 or(names is not None and current!=names):raise ValueError('actualall20gradientnames differ')
  names=current;count=0;maxima={}
  for key,d in raw.items():
   if not key.startswith('reference_'):continue
   other=raw['candidate_'+key[len('reference_'):]];atol,rtol=(2e-5,2e-4)if key.startswith('reference_grad_')else(2e-5,2e-5)
   if d['shape']!=other['shape']:raise ValueError('actualgradientshape differs')
   maximum=0.
   for x,y in zip(d['values'],other['values']):
    if not math.isfinite(x)or not math.isfinite(y)or abs(x-y)>atol+rtol*abs(y):raise ValueError('fixedactualgradientnumericgate fails')
    maximum=max(maximum,abs(x-y));count+=1
   maxima[key]=maximum
   if key.startswith('reference_grad_'):maxgrad=max(maxgrad,maximum)
   elif key=='reference_output':maxoutput=max(maxoutput,maximum)
  scalars+=count;gradients.append(dict(file=path.name,sha256=sha(path),scalar_pairs=count,max_abs=maxima))
 if scalars!=(303856 if flags['input_only_address']else 321776):raise ValueError('actualall20scalar paircount differs')
 frozen=json.loads((folder/'ALL_TRAINING_FROZEN_BEFORE_TEST.json').read_text());last_update=max(i for i,r in enumerate(rows)if r['kind']=='actual_update');testopens=[i for i,r in enumerate(rows)if r['kind']=='actual_input_opened'and r['split'].startswith('test')]
 if len(testopens)!=12 or min(testopens)<=last_update or frozen['actual_new_updates']!=2400 or frozen['TEST_read']or not result['NN_unchanged_after_TEST']:raise ValueError('alltrainingmustfreeze beforeTEST')
 if result['actual_graph_capture_count']!=2 or[g['training_FB_calls']for g in result['graphs']]!=[1800,600]:raise ValueError('exacttwoGraphcounts differ')
 initial=dict(actual_theta_sha256=result['initial_theta_sha256'],scope='savedconfiguration+exactsource/commonCPUproof; no reviewer regenerated NN')
 if a.arm=='B':
  control=json.loads((P/'runs'/CONTROL/'learn/result.json').read_text())
  if result['initial_theta_sha256']!=control['initial_theta_sha256']:raise ValueError('Bmustexactlymatchhistoricalnone initialtheta')
  initial['same_historical_none_theta']=True
 other_name='parameter-typed-normalized-memory-'+('a'if a.arm=='AB'else'ab')+'-n8-seed0-fourstage-graph-v1';other_path=P/'runs'/other_name/'learn/result.json'
 if a.arm in('A','AB')and other_path.exists():
  if json.loads(other_path.read_text())['initial_theta_sha256']!=result['initial_theta_sha256']:raise ValueError('A/AB initialtheta differs')
  initial['same_A_AB_theta']=True
 receipt=json.loads((P/(name+'-coordinator-receipt.json')).read_text());resources=result['resources'];costs=dict(P=resources['parameters'],NN_bytes=resources['parameter_bytes'],Adam_bytes=resources['optimizer_tensor_bytes'],replay_bytes=resources['replay_bytes'],carry_scalars=328 if flags['coverage_normalization']else 320,carry_bytes=1312 if flags['coverage_normalization']else 1280,training_wall_seconds=sum(r['costs']['training_wall_seconds']for r in result['reports']),science_wall_seconds=result['driver_wall_seconds'],science_CPU_seconds=result['driver_CPU_seconds'],local_cold_wall_seconds=receipt['wall_seconds'],padded_forward_MAC=sum(r['costs']['padded_forward_MAC_proxy']for r in result['reports']),active_forward_MAC=sum(r['costs']['active_forward_MAC_proxy']for r in result['reports']),padded_coverage_divisions=sum(r['costs']['padded_coverage_divisions']for r in result['reports']),padded_coverage_clamps=sum(r['costs']['padded_coverage_clamps']for r in result['reports']),setup_seconds=sum(g['setup_seconds']for g in result['graphs']),VALID_seconds=sum(r['costs']['evaluation_wall_seconds']for r in result['reports']),TEST_seconds=sum(r['wall_seconds']for r in result['test_costs']),inference_seconds=result['inference_total_seconds'],safety_and_final_checkpoint_bytes=sum(c['bytes']for c in checkpoints),resources=resources,shared_CPU_attempts=protocol['shared_CPU_admission_attempts'],full_training_FLOPs=None,provider_effective_compute=None,scope='nestedwallscopesnotadditive; MACproxy nottrainingFLOPs; coveragecounts trainingonly, otherpasses chargedinactualtime/resources; CPUadmissionfamilycostsonce')
 snapshots=[result['resources']]+[r['resources']for r in result['reports']]
 allocated=[x['cuda_peak_allocated_bytes']for x in snapshots];reserved=[x['cuda_peak_reserved_bytes']for x in snapshots]
 for graph in result['graphs']:
  for snapshot in(graph['peak_before_reset'],graph['capture']['memory_before'],graph['capture']['memory_after']):allocated.append(snapshot['peak_allocated']);reserved.append(snapshot['peak_reserved'])
 costs.update(CUDA_recorded_peak_allocated_bytes=max(allocated),CUDA_recorded_peak_reserved_bytes=max(reserved),exact_private_pool_bytes=None,driver_context_bytes=None,compilation_or_driver_JIT_component_seconds=None,cache_bytes=None,training_FLOPs=None,search_formation_FLOPs=None,memory_scope='maximumrecordedstageresourceandgraphsnapshots; exactpool/driver/sharedsystemcacheunknown')
 value=dict(status='PASS_INDEPENDENT_SAVED_NORMALIZED_MEMORY_TRAINING',arm=a.arm,run=name,source_script_sha256=sha(Path(__file__)),result_sha256=sha(folder/'result.json'),records_sha256=sha(folder/'records.jsonl'),source_sha256=result['source_sha256'],actual_CPU_proof_sha256=sha(proof),initialization=initial,actual_updates=2400,all20gradient_gates=28,gradient_scalar_pairs=scalars,max_output_abs=maxoutput,max_gradient_abs=maxgrad,gradient_files=gradients,same_original_sampler_payloads=samplers,stages=checks,checkpoints=checkpoints,all_training_frozen_before_TEST=True,perexample_prediction_semantics_scope='separateposthoccounterfactualpipeline; notduplicatedhere',costs=costs,analysis_wall_seconds=time.perf_counter()-start,analysis_CPU_seconds=time.process_time()-cpu,NN_executions=0,SGD_updates=0)
 out=P/('typed-normalized-memory-'+a.arm.lower()+'-independent-training-raw-checks-v1.json')
 with out.open('x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(dict(file=str(out),sha256=sha(out),initialization=initial,costs=costs),sort_keys=True))
if __name__=='__main__':main()
