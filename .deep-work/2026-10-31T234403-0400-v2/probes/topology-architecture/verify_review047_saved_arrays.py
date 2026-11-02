"""Read saved numeric artifacts only with stdlib; no model/package execution."""
from pathlib import Path
import ast,array,hashlib,json,math,struct,sys,zipfile
P=Path('/Users/xiaoyangliu/Desktop/prometheus/.deep-work/2026-10-31T234403-0400-v2/probes/topology-architecture')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def npz(path):
 out={}
 with zipfile.ZipFile(path)as z:
  for leaf in z.namelist():
   b=z.read(leaf);assert b[:6]==b'\x93NUMPY';v=b[6];n=int.from_bytes(b[8:10]if v==1 else b[8:12],'little');s=10 if v==1 else 12;h=ast.literal_eval(b[s:s+n].decode());assert not h['fortran_order'];d=h['descr'];body=b[s+n:];typ={'<f4':'f','<f8':'d','<i8':'q','<i4':'i','|u1':'B','|b1':'B'}[d];a=array.array(typ);a.frombytes(body)
   if sys.byteorder!='little'and a.itemsize>1:a.byteswap()
   assert len(a)==math.prod(h['shape']);out[leaf[:-4]]=(h['shape'],a,d)
 return out
report=dict(scope='independent raw numeric/source verification only; no Torch/NumPy/model execution, SSH or training',closure={},AR={},profile_training={})
shared=None
for tag in('factorized0','factorized1','gru','transformer','profile_block','profile_gru','profile_transformer'):
 folder=P/f'runs/parameter-extended-live-closure-{tag}-v1/evaluate';summary=json.loads((folder/'result.json').read_text());raw=folder/'raw-outputs.npz';assert sha(raw)==summary['raw_sha256'];z=npz(raw);values={k:v[1]for k,v in z.items()};n=3584;assert z['log_probabilities'][0]==(n,8)
 for i in range(n):
  row=values['log_probabilities'][i*8:i*8+8];assert all(math.isfinite(v)for v in row);assert max(range(8),key=lambda k:row[k])==values['predictions'][i]
 targets=tuple(values['targets']);coordinates=tuple((values['condition'][i],values['source_train_index'][i],values['suffix'][i],values['lengths'][i])for i in range(n))
 if shared is None:shared=(targets,coordinates)
 else:assert shared==(targets,coordinates)
 checked=0;out={}
 for condition in(0,1):
  base={values['source_train_index'][i]:values['predictions'][i]for i in range(n)if values['condition'][i]==condition and values['suffix'][i]==0};basecorrect={values['source_train_index'][i]:values['predictions'][i]==targets[i]for i in range(n)if values['condition'][i]==condition and values['suffix'][i]==0};out[condition]={}
  for suffix,name in enumerate(('empty','SET0_STEP','SET1_STEP','CLEAR','SET0_NOP_STEP','SET1_NOP_STEP','SET0_STEP_SET1_STEP')):
   ids=[i for i in range(n)if values['condition'][i]==condition and values['suffix'][i]==suffix]
   for stratum,rows in(('full',ids),('within48',[i for i in ids if values['within_original48'][i]]),('above48',[i for i in ids if not values['within_original48'][i]])):
    saved=summary['scores'][str(condition)][name];saved=saved if stratum=='full'else saved[stratum];right=sum(values['predictions'][i]==targets[i]for i in rows);own=sum(basecorrect[values['source_train_index'][i]]for i in rows);joint=sum(values['predictions'][i]==targets[i]and basecorrect[values['source_train_index'][i]]for i in rows)
    def expected(q):return(q,(q+1)%8,(3*q+2)%8,0,(q+1)%8,(3*q+2)%8,(3*q+5)%8)[suffix]
    consistency=sum(values['predictions'][i]==expected(base[values['source_train_index'][i]])for i in rows)
    for k,v in dict(rows=len(rows),correct=right,own_base_correct_count=own,joint_base_correct_and_suffix_correct=joint,prediction_consistency_count=consistency).items():assert saved[k]==v,(tag,condition,name,stratum,k)
    for k,v in dict(accuracy=right/len(rows)if rows else None,prediction_consistency=consistency/len(rows)if rows else None,conditional_suffix_accuracy_given_own_base_correct=joint/own if own else None).items():assert saved[k]==v
    checked+=1
   out[condition][name]={k:summary['scores'][str(condition)][name][k]for k in('rows','correct','accuracy','prediction_consistency','conditional_suffix_accuracy_given_own_base_correct','within48_accuracy')}
 report['closure'][tag]=dict(raw_sha256=sha(raw),result_sha256=sha(folder/'result.json'),all_argmax_recomputed=True,subsets_checked=checked,scores=out,P=summary['P'],costs=summary['costs'],storage={k:summary[k]for k in('parameter_bytes','registered_buffer_bytes','saved_optimizer_bytes','checkpoint_bytes','data_file_bytes','data_array_bytes','raw_output_bytes')},retained_updates=summary['training_lineage']['retained_lineage_SGD_updates'],retained_train_seconds=summary['training_lineage']['retained_lineage_training_wall_seconds'],retained_forward_MAC_proxy=summary['training_lineage']['retained_lineage_training_forward_MAC_proxy'],other_paid_costs=summary['training_lineage']['other_paid_costs'])
for family,suffix in(('block','v1'),('gru','v2'),('transformer','v1')):
 folder=P/f'runs/parameter-continuation-feedback-{family}-n8-seed0-{suffix}/learn';x=json.loads((folder/'result.json').read_text());records=[json.loads(s)for s in(folder/'records.jsonl').read_text().splitlines()];updates=[r for r in records if r['kind']=='actual_update'];assert [r['step']for r in updates]==list(range(1,601));total=sum(r['training_wall_seconds']for r in updates);assert total==x['costs']['training_wall_seconds'];assert sum(r['base_forward_MAC_proxy']for r in updates)==x['costs']['base_forward_MAC_proxy'];assert sum(r['continuation_forward_MAC_proxy']for r in updates)==x['costs']['continuation_forward_MAC_proxy'];cp=folder/x['checkpoint']['file'];assert sha(cp)==x['checkpoint']['sha256'];cache=folder/'old-continuation-cache.npz';assert sha(cache)==x['teacher']['teacher_output_file_sha256'];prob=npz(cache)['probabilities'];assert prob[0]==(4,8,8);normalizer=max(abs(math.fsum(prob[1][i:i+8])-1)for i in range(0,256,8))
 report['profile_training'][family]=dict(result_sha256=sha(folder/'result.json'),records_sha256=sha(folder/'records.jsonl'),actual_updates=len(updates),actual_update_wall_seconds=total,update_loss_decomposition_max_abs=max(abs(r['loss']-r['terminal_NLL']-r['continuation_KL'])for r in updates),cache_row_normalizer_max_abs=normalizer,teacher=x['teacher'],costs=x['costs'],resources=x['resources'],new_IID=x['tests']['3']['test_iid']['soft'],old_IID={s:x['tests'][s]['test_iid']['soft']['accuracy']for s in('0','1','2')},checkpoint=x['checkpoint'],driver_wall_seconds=x['driver_wall_seconds'])
common=None
for tag in('factorized','transformer','gru','mamba2'):
 folder=P/f'runs/parameter-char-alltoken-{tag}-300-v1/learn';x=json.loads((folder/'summary.json').read_text());rows=[json.loads(s)for s in(folder/'records.jsonl').read_text().splitlines()];updates=[r for r in rows if r['kind']=='actual_alltoken_update'];assert [r['step']for r in updates]==list(range(1,301));last=updates[-1]['cumulative_costs'];assert last['optimizer_updates']==300 and last['training_label_observations']==614400
 for k in('training_wall_seconds','training_CPU_seconds','training_forward_MAC_proxy'):assert last[k]==x['costs'][k]
 labels=[];out={}
 for length in(64,128,256):
  split=f'test{length}';saved=x['final'][split]['soft'];path=folder/saved['vectors_file'];assert sha(path)==saved['vectors_sha256'];z=npz(path);assert z['nll'][0]==(512,length);nll=z['nll'][1];target=z['targets'][1];pred=z['predictions'][1];paired=[nll[i*length+j]for i in range(512)for j in range(length-64,length)];selected_targets=tuple(target[i*length+j]for i in range(512)for j in range(length-64,length));labels.append((tuple(z['endpoints'][1]),selected_targets));mean=math.fsum(nll)/len(nll);pm=math.fsum(paired)/len(paired)
  assert abs(mean-saved['all_token_NLL'])<2e-7 and abs(pm-saved['last64_paired_NLL'])<2e-7
  correct=sum(p==t for p,t in zip(pred,target));assert correct/len(target)==saved['all_token_accuracy']
  out[length]=dict(raw_sha256=sha(path),alltoken_NLL=mean,alltoken_PPL=math.exp(mean),last64_NLL=pm,last64_PPL=math.exp(pm),target_positions=len(nll),last64_positions=len(paired))
 assert labels[0]==labels[1]==labels[2]
 if common is None:common=(labels,x['batch_indices_sha256'],x['metadata_sha256'],x['input_receipts'])
 else:assert common==(labels,x['batch_indices_sha256'],x['metadata_sha256'],x['input_receipts'])
 sourcefolder=P/'versions'/('code-parameter-char-alltoken-'+tag+'-300-v1');checked={}
 for leaf,digest in x['source_sha256'].items():
  src=sourcefolder/leaf
  if src.exists():assert sha(src)==digest;ast.parse(src.read_bytes());checked[leaf]=digest
 report['AR'][tag]=dict(summary_sha256=sha(folder/'summary.json'),records_sha256=sha(folder/'records.jsonl'),actual_updates=len(updates),source_hashes_rechecked=checked,costs=x['costs'],resources=x['resources'],driver_wall_seconds=x['driver_wall_seconds'],outputs=out,all_weights_frozen_before_TEST=x['unchanged_weights_after_test'],configuration=x['model_configuration'])
report['all_closure_shared_targets_and_coordinates']=True;report['AR_common_data_sampler_endpoints_and_paired_targets']=True
out=P/'review047-learning-cost-raw-checks.json'
with out.open('x')as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out),closure_models=len(report['closure']),closure_rows=7*3584,closure_subsets=7*42,AR_models=4,AR_raw_target_positions=4*512*(64+128+256),new_IID={k:v['new_IID']['accuracy']for k,v in report['profile_training'].items()})))
