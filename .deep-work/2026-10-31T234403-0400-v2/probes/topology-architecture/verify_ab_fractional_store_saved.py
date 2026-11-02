"""Stdlib-only saved-array check of actual AB three-condition causal audit."""
import collections,hashlib,importlib.util,json,math,time
from pathlib import Path
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('saved_addresses',P/'verify_typed_three_nn_saved_addresses.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
FOLDER=P/'runs/parameter-ab-fractional-store-cpu-v1/audit'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def argmax(v):return max(range(len(v)),key=lambda k:v[k])
def norm(v):return math.sqrt(sum(x*x for x in v))
def main():
 start=time.perf_counter();cpu=time.process_time();r=json.loads((FOLDER/'result.json').read_text());config=json.loads((P/'versions/code-parameter-ab-fractional-store-cpu-v1/launch-config.json').read_text());version=P/'versions/code-parameter-ab-fractional-store-cpu-v1'
 if r['status']!='COMPLETED_AB_FROZEN_FRACTIONAL_STORE_CAUSAL_AUDIT_CPU'or r['SGD_updates']!=0 or r['active_transitions']!=51456 or r['NNs']!=1 or not r['all3_raw_saved_before_truth']:raise ValueError('actualfixedNN/3condition scope differs')
 for leaf,h in r['source_sha256'].items():
  if sha(version/leaf)!=h:raise ValueError('actualfrozen source differs')
 for item in config['input_files'].values():
  if sha(P/'runs'/item['remote_relative'])!=item['sha256']:raise ValueError('exactOWNinputs differ')
 public=m.saved.npz(P/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated/data/stage3-valid.npz');targets=public['targets']['values'];groups=collections.defaultdict(list)
 for row,length in enumerate(public['lengths']['values']):groups[(public['initial']['values'][row],tuple(public['tokens']['values'][row*128:row*128+length-4]))].append(row)
 if len(groups)!=256 or any(len(v)!=2 for v in groups.values()):raise ValueError('exactpublic256querypairs required')
 conditions={};baseline=None
 for record in r['records']:
  key=record['key'];path=FOLDER/record['raw_file']
  if sha(path)!=record['raw_sha256']or record['checkpoint_sha256']!=config['manifest']['models'][0]['sha256']or not record['NN_unchanged']:raise ValueError('exactactualNN/raw differs')
  z=m.saved.npz(path);logps=[m.vec(z,'after_logp',row,7)for row in range(512)];pred=[argmax(v)for v in logps]
  if key=='soft':baseline=logps
  correct=sum(p==t for p,t in zip(pred,targets));nll=-sum(v[t]for v,t in zip(logps,targets))/512;TV=m.mean([m.tv([math.exp(x)for x in v],[math.exp(x)for x in b])for v,b in zip(logps,baseline)]);changes=sum(argmax(v)!=argmax(b)for v,b in zip(logps,baseline));maximum=dict(read=0.,h=0.,V=0.,coverage=0.,address=0.,attention=0.);event_stats={}
  for e,name in enumerate(r['events']):
   per=[]
   for row in range(512):
    h=m.vec(z,'before_h',row,e);V=m.vec(z,'before_values',row,e);cov=m.vec(z,'before_coverage',row,e);address=m.vec(z,'before_address',row,e);attention=m.vec(z,'read_attention',row,e);weight=m.vec(z,'action_weights',row,e);prob=m.vec(z,'action_probabilities',row,e);chosen=z['action_indices']['values'][row*8+e];read=m.vec(z,'read',row,e);chi=m.vec(z,'candidate_h',row,e);proposal=m.vec(z,'address_proposal',row,e)
    if chosen!=argmax(prob):raise ValueError('OWNargmax trace differs')
    if key=='native':
     if any(w!=int(i==chosen)for i,w in enumerate(weight)):raise ValueError('nativehard7 actionweight differs')
    elif weight!=prob:raise ValueError('soft/drop mustretainexactsoftweights')
    expected_h=[x+sum(weight[:4])*(c-x)+weight[4]*(y-x)for x,c,y in zip(h,chi,read)]
    expectedV=[v+weight[5]*attention[k//32]*(h[k%32]-v)for k,v in enumerate(V)];expectedcov=[v+weight[5]*a*(1-v)for v,a in zip(cov,attention)];proposedcov=expectedcov[:]
    if key=='soft_drop_nonchosen_STORE'and chosen!=5:expectedV=V;expectedcov=cov
    expectedaddress=[v+weight[6]*(p-v)for v,p in zip(address,proposal)];expectedread=[sum(attention[s]*V[s*32+k]/max(cov[s],1e-4)for s in range(8))for k in range(32)]
    fields=[('h',expected_h,m.vec(z,'after_h',row,e)),('V',expectedV,m.vec(z,'after_values',row,e)),('coverage',expectedcov,m.vec(z,'after_coverage',row,e)),('address',expectedaddress,m.vec(z,'after_address',row,e)),('read',expectedread,read),('attention',m.softmax(address),attention),('coverage',proposedcov,m.vec(z,'coverage',row,e))]
    for field,left,right in fields:
     for x,y in zip(left,right):
      if not math.isfinite(x)or not math.isfinite(y)or abs(x-y)>2e-5+2e-5*abs(y):raise ValueError('saved exactcondition transition mismatch: '+key+'/'+field)
      maximum[field]=max(maximum[field],abs(x-y))
    per.append(dict(store_argmax=chosen==5,store_weight=weight[5],coverage_mean=m.mean(cov),coverage_all_zero=all(v==0 for v in cov),coverage_above_eps=sum(v>=1e-4 for v in cov),read_norm=norm(read),h_norm=norm(h),V_norm=norm(V)))
   event_stats[name]={field:(sum(v[field]for v in per)if field in('store_argmax','coverage_all_zero')else m.mean([v[field]for v in per]))for field in per[0]}
  expected=r['final_scores'][key]
  if correct!=expected['correct']or abs(nll-expected['NLL'])>2e-6 or abs(TV-expected['probability_TV_vs_soft_mean'])>2e-6 or changes!=expected['argmax_changes_vs_soft']:raise ValueError('rawscores differ fromactualproducer')
  if record['active_transitions']!=17152 or sum(record['whole_program_action_argmax_counts'])!=17152:raise ValueError('wholeprogramcall counts differ')
  pairs=[]
  for left,right in groups.values():
   if targets[left]!=targets[right]:pairs.append((pred[left]!=pred[right],pred[left]==targets[left]and pred[right]==targets[right],(pred[right]-pred[left])%8==(targets[right]-targets[left])%8))
  conditions[key]=dict(raw_sha256=sha(path),correct=correct,accuracy=correct/512,NLL=nll,TV_vs_soft=TV,argmax_changes_vs_soft=changes,nondegenerate_query_pairs=len(pairs),query_pred_changes=sum(x[0]for x in pairs),query_both_correct=sum(x[1]for x in pairs),query_relation_consistent=sum(x[2]for x in pairs),transition_formula_max_abs=maximum,whole_program_action_argmax_counts=record['whole_program_action_argmax_counts'],dropped_nonchosen_STORE_rows=record['dropped_nonchosen_STORE_rows'],public_event_stats=event_stats)
 value=dict(status='PASS_SAVED_AB_FRACTIONAL_STORE_CAUSAL_RAW',result_sha256=sha(FOLDER/'result.json'),source_script_sha256=sha(Path(__file__)),source_sha256=r['source_sha256'],conditions=conditions,actual_NN_transitions=51456,SGD_updates=0,NN_executions_by_verifier=0,science_wall_seconds=r['wall_seconds'],science_CPU_seconds=r['CPU_seconds'],CPU_peak_RSS_bytes=r['CPU_peak_RSS_bytes'],full_FLOPs=None,provider_effective_compute=None,interpretation='droppingnonchosenfractionalwritescauseslargeclosedloopchangeforfrozenAB; doesnotattributeallnativegapnorprovequerybinding. Softcomputationisvalid; baselineGRU/TFnotarbitrarilyhardened. NoextraSGD/trainingorNNbyverifier.',analysis_wall_seconds=time.perf_counter()-start,analysis_CPU_seconds=time.process_time()-cpu)
 out=P/'ab-fractional-store-independent-raw-checks-v1.json'
 with out.open('x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(dict(file=str(out),sha256=sha(out),conditions=conditions),sort_keys=True))
if __name__=='__main__':main()
