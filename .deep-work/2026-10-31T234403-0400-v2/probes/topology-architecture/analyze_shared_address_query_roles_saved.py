"""Saved new2NN query/key role arithmetic. No NN/SSH/SGD/intervention."""
import argparse,json,math
from pathlib import Path
from analyze_shared_address_valid_addresses_saved import AUDIT,TRUTH,EVENTS,vector,matrix,mean,dist,cosine,dot,norm,tv,summary,argmax
from verify_factorized_graph600_saved import ROOT,npz,sha
GEN=ROOT/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated'
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise FileExistsError('new receipt only')
 audit=json.loads((AUDIT/'result.json').read_text());truth=npz(TRUTH);meta=json.loads((GEN/'data/metadata.json').read_text());assert sha(TRUTH)==audit['privileged_truth_sha256']and sha(GEN/'counterfactual-truth.npz')==meta['counterfactual_truth_sha256'];grid=npz(GEN/'counterfactual-truth.npz')['valid_values_group_order'];public=npz(GEN/'data/stage3-valid.npz');lookup={tuple(grid['values'][r*5:r*5+5]):r for r in range(512)};assert len(lookup)==512
 out=dict(schema='saved-new-shared-address-query-role-analysis-v1',source_script_sha256=sha(Path(__file__)),audit_result_sha256=sha(AUDIT/'result.json'),truth_sha256=sha(TRUTH),counterfactual_truth_sha256=sha(GEN/'counterfactual-truth.npz'),scope='posthoc savedarray arithmetic only; physicalslots notlogicalkeys; rolematchedpairs generatedbeforetraining; no NN/checkpointload/SSH/SGD',models={})
 keys_by_stage={}
 def word(r):return list(public['tokens']['values'][r*128:r*128+public['lengths']['values'][r]])
 def compare(z,r,s,e,f=None):
  f=e if f is None else f;q=vector(z,'address_query',r,e);v=vector(z,'address_query',s,f);ra=vector(z,'read_attention',r,e);rb=vector(z,'read_attention',s,f)
  return dict(query_L2=dist(q,v),relative_query_L2=dist(q,v)/(1e-30+.5*(norm(q)+norm(v))),query_cosine=cosine(q,v),attention_TV=tv(ra,rb),top_slot_changed=float(argmax(ra)!=argmax(rb)))
 def sm(rows):return{key:summary([r[key]for r in rows])for key in rows[0]}if rows else dict(rows=0)
 for rec in audit['records']:
  f=AUDIT/rec['raw_file'];assert sha(f)==rec['raw_sha256'];z=npz(f);valid=z['event_valid']['values'];keys=matrix(z,'before_keys',0,0,16);flatkeys=sum(keys,[]);maxreconstruction=0.;events={}
  for e,name in enumerate(EVENTS):
   rr=[r for r in range(512)if valid[r*5+e]]
   if not rr:continue
   for r in rr:
    assert vector(z,'before_keys',r,e)==flatkeys==vector(z,'after_keys',r,e)
    q=vector(z,'address_query',r,e);aa=vector(z,'read_attention',r,e);assert aa==vector(z,'write_addresses',r,e)
    logits=[dot(q,k)/4 for k in keys];mx=max(logits);ex=[math.exp(v-mx)for v in logits];den=math.fsum(ex);reconstruct=[v/den for v in ex];maxreconstruction=max(maxreconstruction,max(abs(v-w)for v,w in zip(aa,reconstruct)))
   events[name]=dict(query_norm=summary([norm(vector(z,'address_query',r,e))for r in rr]),write_gate=summary([vector(z,'write_gate',r,e)[0]for r in rr]))
  assert maxreconstruction<2e-6
  # Query-flip pairs: same complete prefix through both writes/noise, only
  # two suffix KEY tokens differ. READ/MIX current opcode is held fixed.
  qflip={}
  for e in(2,3,4):
   if not valid[e]:continue
   pairs=[]
   for g in range(2):
    for v0 in range(8):
     for v1 in range(8):
      for w in range(2):
       r,s=lookup[(g,v0,v1,0,w)],lookup[(g,v0,v1,1,w)];assert word(r)[:-4]==word(s)[:-4];pairs.append(compare(z,r,s,e))
   qflip[EVENTS[e]]=dict(pairs=len(pairs),metrics=sm(pairs))
  # Same value/order/write-order in both pre-generated noise templates. c
  # begins at zero and takes no h, so writes before noise should be identical.
  noise={}
  for e in range(5):
   if not valid[e]:continue
   pairs=[compare(z,lookup[(0,v0,v1,o,w)],lookup[(1,v0,v1,o,w)],e)for v0 in range(8)for v1 in range(8)for o in range(2)for w in range(2)]
   noise[EVENTS[e]]=dict(pairs=len(pairs),metrics=sm(pairs))
  # At first WRITE hold payload value and every preceding token fixed except
  # KEY0 versus KEY1; future other-value/query/noise cannot affect this event.
  firstkey=[]
  for g in range(2):
   for value in range(8):
    r,s=lookup[(g,value,0,0,0)],lookup[(g,0,value,0,1)];wa,wb=word(r),word(s);pos=z['event_positions']['values'][r*5];pos2=z['event_positions']['values'][s*5];assert pos==pos2 and wa[1:pos+1]==wb[1:pos2+1]and {wa[0],wb[0]}=={7,8};firstkey.append(dict(value=value,group=g,**compare(z,r,s,0)))
  # Same logical address at WRITE and query, compared in query-vector space
  # as well as actual fixed-key attention. No slot/address identification.
  cross={}
  for e in(2,4):
   rr=[r for r in range(512)if valid[r*5+e]]
   if not rr:continue
   target=[];other=[]
   for r in rr:
    off=(r*5+e)*6;address=truth['true_state_after_event']['values'][off+3];which=next(j for j in(0,1)if truth['true_state_after_event']['values'][(r*5+j)*6+3]==address);target.append(compare(z,r,r,e,which));other.append(compare(z,r,r,e,1-which))
   cross[EVENTS[e]]=dict(same_logical_address=sm(target),other_logical_address=sm(other),caution='historicalWRITEkeys staticbutcontent latermutable; sameaddressquerymatch ismeasured notassumed')
  keys_by_stage[rec['stage']]=keys
  out['models'][rec['key']]=dict(raw_sha256=rec['raw_sha256'],all_event_keys_exact_same_across_episode_and_examples=True,shared_read_write_attention_exact=True,query_key_softmax_reconstruction_max_abs=maxreconstruction,event_query_norm_and_gate=events,query_order_flip=qflip,noise_template_flip=noise,first_WRITE_only_KEY_flip=dict(pairs=len(firstkey),metrics=sm([{k:v for k,v in x.items()if k not in('group','value')}for x in firstkey]),by_value=firstkey),cross_token_logical_address=cross,keys=keys)
 out['keys_stage2_to_stage3_Frobenius_distance']=math.sqrt(math.fsum(dist(x,y)**2 for x,y in zip(keys_by_stage[2],keys_by_stage[3])))
 with a.out.open('x')as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
 print(json.dumps(dict(out=str(a.out),sha256=sha(a.out))))
if __name__=='__main__':main()
