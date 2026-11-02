"""Stdlib arithmetic of SAVED VALID predictions: no NN/SSH/SGD/intervention."""
import argparse,json,math
from pathlib import Path
from verify_factorized_graph600_saved import ROOT,npz,sha
AUDIT=ROOT/'runs/parameter-posterior-value-four-nn-valid-diagnosis-v1/audit'
GEN=ROOT/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated'
TRUTH=ROOT/'runs/parameter-posterior-value-valid-prefix-truth-v1/truth/prefix-truth.npz'
def argmax(x):return max(range(len(x)),key=x.__getitem__)
def score(rows,key):return dict(rows=len(rows),correct=sum(r['pred']==r[key]for r in rows),agreement=sum(r['pred']==r[key]for r in rows)/len(rows))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('new receipt only')
 meta=json.loads((GEN/'data/metadata.json').read_text());assert sha(GEN/'counterfactual-truth.npz')==meta['counterfactual_truth_sha256'];assert sha(GEN/'data/stage3-valid.npz')==meta['files']['valid']['sha256']
 old=npz(GEN/'counterfactual-truth.npz')['valid_values_group_order'];assert old['shape']==(512,5)
 public=npz(GEN/'data/stage3-valid.npz');truth=npz(TRUTH);audit=json.loads((AUDIT/'result.json').read_text());assert sha(TRUTH)==audit['privileged_truth_sha256'];out=dict(schema='saved-VALID-binding-symmetry-arithmetic-v1',source_script_sha256=sha(Path(__file__)),audit_result_sha256=sha(AUDIT/'result.json'),original_VALID_sha256=sha(GEN/'data/stage3-valid.npz'),counterfactual_truth_sha256=sha(GEN/'counterfactual-truth.npz'),prefix_truth_sha256=sha(TRUTH),scope='savedfrozenVALIDpredictions+privilegedtruth arithmeticonly; noNN/SSH/SGD ororaclefeedback; prospectivelygeneratedbalancedgrid, notnewexamples',models={})
 base=[];lookup={}
 for r in range(512):
  group,v0,v1,o,w=map(int,old['values'][r*5:r*5+5]);length=public['lengths']['values'][r];word=list(public['tokens']['values'][r*128:r*128+length]);y=int(public['targets']['values'][r]);first,last=(v0,v1)if w==0 else(v1,v0)
  assert y==int(truth['original_targets']['values'][r])==(3*(v0 if o==0 else v1)+(v1 if o==0 else v0))%8
  assert word[-4:]==[7+o,10,8-o,15]
  b=dict(row=r,group=group,v0=v0,v1=v1,query_order=o,write_order=w,target=y,logical0=(3*v0+v1)%8,logical1=(3*v1+v0)%8,temporal_first=(3*first+last)%8,temporal_last=(3*last+first)%8,temporal_query0_first=((3*first+last)if o==0 else(3*last+first))%8,temporal_query0_last=((3*last+first)if o==0 else(3*first+last))%8,query_label_invariant=(2*(v0-v1))%8==0,word=word)
  base.append(b);lookup[(group,v0,v1,o,w)]=r
 assert len(lookup)==512
 formula=('target','logical0','logical1','temporal_first','temporal_last','temporal_query0_first','temporal_query0_last')
 for rec in audit['records']:
  if rec['stage']!=3:continue
  f=AUDIT/rec['raw_file'];assert sha(f)==rec['raw_sha256'];z=npz(f);assert all(z['event_valid']['values'][r*5+4]for r in range(512));rows=[dict(b,pred=argmax(z['after_logp']['values'][(b['row']*5+4)*8:(b['row']*5+5)*8]))for b in base]
  querypairs=[];writepairs=[]
  for r in rows:
   g,v0,v1,o,w=(r[k]for k in('group','v0','v1','query_order','write_order'))
   if o==0:
    s=rows[lookup[(g,v0,v1,1,w)]];assert r['word'][:-4]==s['word'][:-4]and len(r['word'])==len(s['word']);querypairs.append(dict(group=g,v0=v0,v1=v1,write_order=w,row0=r['row'],row1=s['row'],pred0=r['pred'],pred1=s['pred'],target0=r['target'],target1=s['target'],label_changed=r['target']!=s['target'],prediction_changed=r['pred']!=s['pred'],both_correct=r['pred']==r['target']and s['pred']==s['target'],one_correct=(r['pred']==r['target'])!=(s['pred']==s['target'])))
   if w==0:
    s=rows[lookup[(g,v0,v1,o,1)]];assert r['target']==s['target']and r['word'][-4:]==s['word'][-4:];writepairs.append(dict(group=g,v0=v0,v1=v1,query_order=o,row0=r['row'],row1=s['row'],pred0=r['pred'],pred1=s['pred'],target=r['target'],label_changed=False,prediction_changed=r['pred']!=s['pred'],both_correct=r['pred']==r['target']and s['pred']==s['target']))
  def ps(v):return dict(pairs=len(v),label_changes=sum(r['label_changed']for r in v),prediction_changes=sum(r['prediction_changed']for r in v),prediction_change_rate=sum(r['prediction_changed']for r in v)/len(v),both_correct=sum(r['both_correct']for r in v))
  strata={}
  for group in range(2):
   for o in range(2):
    for w in range(2):
     rr=[r for r in rows if(r['group'],r['query_order'],r['write_order'])==(group,o,w)];strata[f'noise{group}:query{o}:write{w}']={k:score(rr,k)for k in formula}
  valuepairs=[]
  for group in range(2):
   for v0 in range(8):
    for v1 in range(8):
     rr=[rows[lookup[(group,v0,v1,o,w)]]for o in range(2)for w in range(2)];valuepairs.append(dict(group=group,v0=v0,v1=v1,query_label_invariant=rr[0]['query_label_invariant'],rows_query_major_write_minor=[r['row']for r in rr],pred_query_major_write_minor=[r['pred']for r in rr],target_query_major_write_minor=[r['target']for r in rr],matches={k:sum(r['pred']==r[k]for r in rr)for k in formula}))
  out['models'][rec['key']]=dict(raw_sha256=rec['raw_sha256'],formula_agreement={k:score(rows,k)for k in formula},strata=strata,query_flip=dict(all=ps(querypairs),different_label_only=ps([r for r in querypairs if r['label_changed']]),same_label_only=ps([r for r in querypairs if not r['label_changed']]),by_noise_group={str(g):ps([r for r in querypairs if r['group']==g])for g in range(2)},pairs=querypairs),write_order_flip=dict(all=ps(writepairs),different_query_targets_value_pairs=ps([r for r in writepairs if(2*(r['v0']-r['v1']))%8!=0]),by_noise_group={str(g):ps([r for r in writepairs if r['group']==g])for g in range(2)},pairs=writepairs),value_pairs=valuepairs,per_row=[{k:v for k,v in r.items()if k!='word'}for r in rows])
 out['analytic_order_blind_optimal_accuracy']=dict(exact_fraction='5/8',decimal=.625,reason='balancedtwoqueryorders; targetsequal iff2(v0-v1)=0 mod8,16of64pairs; .25*1+.75*.5=.625; scorealone notmechanismidentification')
 with a.out.open('x')as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 print(json.dumps(dict(out=str(a.out),sha256=sha(a.out),summary={k:dict(formulas=v['formula_agreement'],queryflip=v['query_flip']['all'],writeflip=v['write_order_flip']['all'])for k,v in out['models'].items()})))
if __name__=='__main__':main()
