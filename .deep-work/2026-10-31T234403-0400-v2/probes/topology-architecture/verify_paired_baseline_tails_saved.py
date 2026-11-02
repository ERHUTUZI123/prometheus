"""Stdlib independent saved-output arithmetic. Reviewer authored baseline code.
No NN/Torch/NumPy/checkpoint deserialization/SSH; source-authorship not independent.
"""
import argparse,hashlib,json,math,time
from pathlib import Path
from analyze_posterior_valid_slot_values import load_npz
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def metric(logp,target):
 correct=0;nll=0.;normal=0.
 for row,y in enumerate(target):
  v=logp[row*8:(row+1)*8];assert all(math.isfinite(x)for x in v);correct+=max(range(8),key=v.__getitem__)==y;nll-=v[y];m=max(v);normal=max(normal,abs(m+math.log(sum(math.exp(x-m)for x in v))))
 return dict(rows=len(target),correct=correct,accuracy=correct/len(target),NLL=nll/len(target),max_logsumexp_abs=normal)
def verify(arm,sampling):
 name=f'parameter-paired-tail-{arm}-{sampling.replace("_","-")}-n8-seed0-600-v1';folder=P/'runs'/name/'learn';version=P/'versions'/('code-'+name);r=json.loads((folder/'result.json').read_text());c=json.loads((version/'launch-config.json').read_text());protocol=json.loads((P/(name+'-protocol.json')).read_text());assert r['status']=='ACTUAL_PAIRED_BASELINE_STAGE3_600_COMPLETE'and r['arm']==arm and r['sampling']==sampling and(r['actual_new_updates'],r['old_effective_updates'],r['old_updates_repeated'])==(600,1800,0)and r['NN_unchanged_after_TEST']
 assert r['source_sha256']==c['manifest']['source_sha256']==protocol['common_source_sha256'];assert all(sha(version/k)==v for k,v in c['source_sha256'].items());proof=P/'runs'/c['CPU_name']/'check/result.json';assert sha(proof)==c['CPU_proof_sha256'];inputs={};datahash={}
 for key,spec in c['manifest']['files'].items():
  f=P/'runs'/spec['remote_relative'];assert sha(f)==spec['sha256'];datahash[key]=spec['sha256']
  if key.startswith('data:'):inputs[key]=load_npz(f)
 assert len(inputs)==20;assert r['parent_checkpoint_sha256']==c['manifest']['parents'][arm]['checkpoint_sha256'];rows=[json.loads(x)for x in(folder/'records.jsonl').read_text().splitlines()];u=[x for x in rows if x['kind']=='actual_update'];assert[x['step']for x in u]==list(range(1,601));assert all(math.isfinite(x['terminal_NLL'])and math.isfinite(x['preclip_gradient'])for x in u);cost=r['costs'];sums={k:sum(x[k]for x in u)for k in('training_wall_seconds','forward_MAC_proxy','active_token_steps','padded_token_steps')};assert all(cost[k]==v for k,v in sums.items());assert(cost['SGD_updates'],cost['current_training_label_reads'],cost['old_replay_training_label_reads'])==(600,28800,9600)
 selected=load_npz(folder/'stage3-used-current-indices.npz')['indices'][1];expected=load_npz(P/'runs'/c['manifest']['files']['paired_schedule']['remote_relative']);key='indices'if sampling=='paired'else'indices_same_pool_shuffled';assert selected==expected[key][1];draws=load_npz(folder/'stage3-replay-draws.npz')['buffer_indices'][1];assert draws==load_npz(P/'runs'/c['manifest']['files']['replay_draws']['remote_relative'])['buffer_indices'][1];hits=[0]*4096
 for s in range(600):
  for i in selected[s*64:s*64+48]:hits[i]+=1
 tests={};valid={};predictions=0
 def check(f,target,reported):
  d=load_npz(f);assert list(d['row_index'][1])==list(range(len(target['targets'][1])))
  for field in('targets','initial','lengths'):assert d[field][1]==target[field][1]
  assert d['soft_logp'][1]==d['native_logp'][1];m=metric(d['soft_logp'][1],d['targets'][1]);assert m['accuracy']==reported['soft']['accuracy']and abs(m['NLL']-reported['soft']['terminal_NLL'])<2e-6 and m['max_logsumexp_abs']<2e-6;assert reported['soft']['accuracy']==reported['native']['accuracy'];return dict(sha256=sha(f),**m)
 for s in range(4):
  for split in('test_iid','test_long','test_unseen'):
   k=f'{s}:{split}';tests[k]=check(folder/f'final-task{s}-{split}-predictions.npz',inputs['data:'+k],r['tests'][str(s)][split]);predictions+=tests[k]['rows']*2
 assert predictions==12288 and len(tests)==12;assert[x['step']for x in r['curves']]==list(range(0,601,100))
 for curve in r['curves']:
  for s in range(4):valid[f'{curve["step"]}:{s}']=check(folder/f'step{curve["step"]:03d}-valid-task{s}-predictions.npz',inputs[f'data:{s}:valid'],curve['valid'][str(s)])
 for field in('safety','checkpoint'):
  receipt=r[field];f=folder/receipt['file'];assert sha(f)==receipt['sha256']and f.stat().st_size==receipt['bytes']
 freeze=json.loads((folder/'ALL_TRAINING_FROZEN_BEFORE_TEST.json').read_text());assert freeze['actual_new_updates']==600;assert min(i for i,x in enumerate(rows)if x['kind']=='actual_TEST_input_opened')>max(i for i,x in enumerate(rows)if x['kind']=='actual_update');assert r['replay_maintenance']['seen']==124800 and r['replay_maintenance']['bookkeeping_counters']['draws']==28800
 receipt=json.loads((P/(name+'-coordinator-receipt.json')).read_text());assert receipt['returncode']==0;costs=dict(P=r['resources']['parameters'],NN_bytes=r['resources']['parameter_bytes'],Adam_bytes=r['resources']['optimizer_tensor_bytes'],replay_bytes=r['resources']['replay_bytes'],registered_buffer_bytes=r['resources']['registered_buffer_bytes'],training_wall_seconds=cost['training_wall_seconds'],science_wall_seconds=r['driver_wall_seconds'],science_CPU_seconds=r['driver_CPU_seconds'],local_cold_seconds=receipt['wall_seconds'],maintenance_seconds=r['maintenance_wall_seconds'],TEST_seconds=sum(x['wall_seconds']for x in r['test_costs']),inference_seconds=r['inference_seconds'],CUDA_peak_allocated_bytes=r['resources']['cuda_peak_allocated_bytes'],CUDA_peak_reserved_bytes=r['resources']['cuda_peak_reserved_bytes'],CPU_peak_RSS_bytes=r['resources']['cpu_peak_RSS_bytes'],checkpoint_payload_bytes=r['safety']['bytes']+r['checkpoint']['bytes'],active_steps=cost['active_token_steps'],padded_steps=cost['padded_token_steps'],forward_MAC_proxy=cost['forward_MAC_proxy'],full_training_FLOPs=None,provider_effective_compute=None)
 return dict(run=name,status='PASS_INDEPENDENT_SAVED_BASELINE_TAIL_RAW',result_sha256=sha(folder/'result.json'),records_sha256=sha(folder/'records.jsonl'),source_sha256=r['source_sha256'],CPU_proof_sha256=sha(proof),all_input_hashes=datahash,parent_checkpoint_sha256=r['parent_checkpoint_sha256'],restore=r['restore'],post_growth=r['post_growth'],updates600_finite=True,exact_scheduled_current_and_replay_arrays=True,current_exposure_vector=hits,all12_TEST=tests,soft_native_stored_row_predictions_checked=predictions,all28_VALID=valid,old_updates_repeated=0,replay_restore_digest=c['manifest']['parents'][arm]['replay_digest'],replay_final_digest=r['replay_maintenance']['after_sha256'],replay_scope='sameoldbuffer/drawsfortraining, finalreservoircanlegitimatelydifferduetoorder-dependentmaintenance; finalcontentsnotdeserializedbyreviewer',checkpoint_hashes={k:r[k]for k in('safety','checkpoint')},costs=costs,reload_scope='actualfileNN/Adam/replayselfreload+same16rowforward andfactory-neutralruntimeRNG restore; noindependentcheckpointdeserialization/liveAdamcomparisonbyreviewer',native_scope='sameordinaryNNforwardalias;12288storedoutputrows are6144actualTESTexamples, not12288NNqueries')
def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=('gru32','transformer32','both'),required=True);a=p.parse_args();start=time.perf_counter();cpu=time.process_time();arms=('gru32','transformer32')if a.arm=='both'else(a.arm,);result={}
 for arm in arms:
  pair={s:verify(arm,s)for s in('paired','same_pool_shuffle')};left,right=pair.values();assert left['post_growth']==right['post_growth']and left['parent_checkpoint_sha256']==right['parent_checkpoint_sha256']and left['current_exposure_vector']==right['current_exposure_vector']and left['replay_restore_digest']==right['replay_restore_digest']and left['costs']['active_steps']==right['costs']['active_steps'];result[arm]=pair
 out=dict(status='PASS_BASELINE_PAIRED_VS_SAMEPOOL_SAVED_RAW',source_script_sha256=sha(Path(__file__)),models=result,SGD_updates=0,new_NN_queries=0,reviewer_authored_baseline_sources=True,scope='independentstdlibarithmeticonraw/sourcebytes, notindependentsourceauthorship, noarchitectureadvantage; fixedrootprotocol, noTESTtuning; nestedcostsnotadditive; baselinepaddingactualandmayvary',analysis_wall_seconds=time.perf_counter()-start,analysis_CPU_seconds=time.process_time()-cpu)
 f=P/f'paired-tail-{a.arm}-independent-raw-checks-v1.json'
 with f.open('x')as stream:json.dump(out,stream,sort_keys=True,indent=2);stream.write('\n')
 print(f.name,sha(f))
 for arm,pair in result.items():
  for sampler,r in pair.items():print(arm,sampler,'newTEST',[r['all12_TEST'][f'3:{s}']['correct']for s in('test_iid','test_long','test_unseen')],'start',r['post_growth'],'cost',r['costs'])
if __name__=='__main__':main()
