"""Actual continuation-learning, teacher provenance and complete-cost ledger.

Reads original paid TRAIN only for a POSTTRAIN teacher-quality audit. Never
supplies those old labels to a learner or chooses a new configuration.
"""
import importlib.util,json,hashlib,zipfile,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('owned_npy_stdlib',ROOT/'verify_live_composition_records.py');npy=importlib.util.module_from_spec(sp);sp.loader.exec_module(npy)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def arrays(p):
 with zipfile.ZipFile(p)as z:return{k[:-4]:npy.array(z.read(k))for k in z.namelist()}
def main():
 old=ROOT/'runs/parameter-context-data-n8-seed0-v1/data';meta=json.loads((old/'metadata.json').read_text());train=arrays(old/meta['stages'][0]['files']['train']);a={k:v[0]for k,v in train.items()};width=train['tokens'][1][1];observed={}
 for i,length in enumerate(a['lengths']):
  if 1<=length<=2:
   key=(a['initial'][i],tuple(a['tokens'][i*width:i*width+length]));value=a['targets'][i]
   if key in observed and observed[key]!=value:raise ValueError('old short TRAIN behavior not deterministic')
   observed[key]=value
 totals={r['arm']:r for r in json.loads((ROOT/'memory-live-stage-total-paid-training-cost-evidence-v1.json').read_text())['rows']}
 rows=[]
 for family,revision in [('block',1),('gru',2),('transformer',1)]:
  run=f'parameter-continuation-feedback-{family}-n8-seed0-v{revision}';folder=ROOT/'runs'/run/'learn';p=folder/'result.json';v=json.loads(p.read_text());cache=arrays(folder/'old-continuation-cache.npz')['probabilities'];prob,shape=cache
  if shape!=(4,8,8):raise ValueError('actual4word8state8class cache')
  comparisons=[]
  for wi,w in enumerate(v['teacher']['words']):
   for initial in range(8):
    target=observed[(initial,tuple(w))];row=prob[(wi*8+initial)*8:(wi*8+initial+1)*8];pred=max(range(8),key=lambda x:row[x]);comparisons.append(dict(word=w,initial=initial,already_observed_old_TRAIN_target=target,teacher_prediction=pred,correct=pred==target))
  if family=='block':
   parent=ROOT/'runs/parameter-factorized-value-anchor0-n8-seed0-fourstage-v2/learn';history=[json.loads((parent/f'completed-stage{s}-before-TEST.json').read_text())['report']for s in range(3)];oldseconds=sum(r['costs']['training_wall_seconds']for r in history);oldmac=sum(r['costs']['forward_MAC_proxy']for r in history)
  else:
   previous=json.loads((ROOT/('runs/parameter-memory-live-stage-n8-seed0-'+family+'-stage4-v1/learn/result.json')).read_text());oldseconds=totals[family]['total_training_wall_seconds']-previous['costs']['training_wall_seconds'];oldmac=totals[family]['total_training_forward_MAC_proxy']-previous['costs']['training_forward_MAC_proxy']
  current=v['costs'];base=current['base_forward_MAC_proxy'];tail=current['continuation_forward_MAC_proxy']
  compact=dict(family=family,P=v['resources']['parameters'],new_VALID=v['curves'][-1]['valid']['3']['soft']['accuracy'],new_IID=v['tests']['3']['test_iid']['soft']['accuracy'],old0_IID=v['tests']['0']['test_iid']['soft']['accuracy'],old1_IID=v['tests']['1']['test_iid']['soft']['accuracy'],old2_IID=v['tests']['2']['test_iid']['soft']['accuracy'],current600_training_seconds=current['training_wall_seconds'],fourphase_effective_training_seconds=oldseconds+current['training_wall_seconds'],fourphase_forward_GMAC=(oldmac+base+tail)/1e9,teacher_correct_observed32=sum(x['correct']for x in comparisons),teacher_rows=32,teacher_cache_bytes=v['teacher']['cache_bytes'],NN_bytes=v['resources']['parameter_bytes'],Adam_bytes=v['resources']['optimizer_tensor_bytes'],Replay_bytes=v['resources']['replay_bytes'])
  rows.append(dict(summary=compact,result_sha256=sha(p),run=run,all_costs=current,resources=v['resources'],teacher=v['teacher'],teacher_quality_comparisons=comparisons,teacher_quality_scope='posttraining compare cached outputs with32alreadyobserved phase0 TRAIN pairs, NOT a teacher correction or training signal',growth=v['growth'],checkpoint=v['checkpoint'],inference=v['inference'],inference_seconds=v['inference_total_seconds'],all_test_costs=v['test_costs'],full_parent_lineage=v['parent_paid_lineage']))
 prefix=ROOT/'continuation-learning-parameter-complete-cost-evidence-v1'
 with prefix.with_suffix('.json').open('x')as f:json.dump(dict(scope='three actual same600-feedback/Replay128 own-parent continuation-feedback methods; one seed development, not Primary',rows=rows,past_raw_input_access='current version reopens original phase0 TRAIN and selects first4distinct short raw programs; this is explicit past-input access beyond128replay, common across three algorithms. Only firstsix raw rows used for selection, no old terminal labels used by learner. All loaded arrays/storage, selection and modelqueries paid. A future strict streaming version must retain pool prospectively; do not retroactively claim that occurred.',parent_confound='factorized anchor1 and anchor0 had different firstthree-stage objectives; current profiles share anchor0 parent with no-profile comparison. Do not attribute all anchor1-vs-profile difference solely to current-stage constraint.',cost_overlap='fourphase training sum excludes separately reported teacher setup/eval/reload/checkpoint/inference/failedversions; root original histories reused once, not physical regeneration per branch; MAC is forward only',unknowns={'complete_training_FLOPs':None,'energy':None}),f,indent=2,sort_keys=True)
 with prefix.with_suffix('.csv').open('x',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]['summary']));w.writeheader();w.writerows(r['summary']for r in rows)
 print(json.dumps([r['summary']for r in rows],indent=2))
if __name__=='__main__':main()
