"""Stdlib-only posthoc verification of already executed three-NN VALID audit."""
import collections,hashlib,importlib.util,json,math
from pathlib import Path
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('saved_raw',P/'verify_factorized_graph600_saved.py');saved=importlib.util.module_from_spec(spec);spec.loader.exec_module(saved)
AUDIT=P/'runs/parameter-typed-three-nn-valid-actions-cpu-v1/audit'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(v):return math.fsum(v)/len(v)
def softmax(x):
 maximum=max(x);v=[math.exp(t-maximum)for t in x];total=sum(v);return[t/total for t in v]
def tv(x,y):return .5*sum(abs(a-b)for a,b in zip(x,y))
def vec(z,key,row,event):
 a=z[key];n=math.prod(a['shape'][2:]);offset=(row*8+event)*n;return list(a['values'][offset:offset+n])
def summary(values):return dict(mean=mean(values),min=min(values),max=max(values),median=saved.percentile(values,.5))
def main():
 result=json.loads((AUDIT/'result.json').read_text());manifest=json.loads((P/'versions/code-parameter-typed-three-nn-valid-actions-cpu-v1/launch-config.json').read_text())
 for leaf,digest in result['source_sha256'].items():
  if sha(P/'versions/code-parameter-typed-three-nn-valid-actions-cpu-v1'/leaf)!=digest:raise ValueError('actual source changed')
 public=saved.npz(P/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated/data/stage3-valid.npz');words=[list(public['tokens']['values'][i*128:i*128+public['lengths']['values'][i]])for i in range(512)]
 groups=collections.defaultdict(list)
 for row,word in enumerate(words):groups[(public['initial']['values'][row],tuple(word[:-4]))].append(row)
 if len(groups)!=256 or any(len(v)!=2 for v in groups.values()):raise ValueError('exact256 public query-flip pairs required')
 reports={}
 for record in result['records']:
  path=AUDIT/record['raw_file']
  if sha(path)!=record['raw_sha256']:raise ValueError('actual raw SHA differs')
  z=saved.npz(path);stage=record['stage'];events={};maximum=dict(attention=0.,h=0.,values=0.,address=0.,read=0.)
  for event,name in enumerate(result['events']):
   if stage==2 and event==7:continue
   attention=[vec(z,'read_attention',r,event)for r in range(512)];alpha=[vec(z,'action_probabilities',r,event)for r in range(512)]
   top=collections.Counter(max(range(8),key=lambda k:a[k])for a in attention)
   events[name]=dict(mean_action_probabilities=[mean([a[k]for a in alpha])for k in range(7)],attention_entropy=mean([-sum(t*math.log(max(t,1e-30))for t in a)for a in attention]),attention_top_probability=mean([max(a)for a in attention]),topslot_counts=dict(top))
   for r in range(512):
    h=vec(z,'before_h',r,event);V=vec(z,'before_values',r,event);address=vec(z,'before_address',r,event);a=attention[r];ap=alpha[r];chi=vec(z,'candidate_h',r,event);read=vec(z,'read',r,event);proposal=vec(z,'address_proposal',r,event)
    expected_h=[x+sum(ap[:4])*(c-x)+ap[4]*(y-x)for x,c,y in zip(h,chi,read)]
    expected_V=[v+ap[5]*a[k//32]*(h[k%32]-v)for k,v in enumerate(V)]
    expected_address=[v+ap[6]*(p-v)for v,p in zip(address,proposal)]
    expected_read=[sum(a[s]*V[s*32+k]for s in range(8))for k in range(32)]
    for key,left,right in [('attention',a,softmax(address)),('h',expected_h,vec(z,'after_h',r,event)),('values',expected_V,vec(z,'after_values',r,event)),('address',expected_address,vec(z,'after_address',r,event)),('read',expected_read,read)]:maximum[key]=max(maximum[key],max(abs(x-y)for x,y in zip(left,right)))
  retrieval=[];write_separation=[];key_after={};keys_by_context={}
  for event in(0,2,4,6):
   buckets={7:[],8:[]}
   for r,word in enumerate(words):
    position=z['event_positions']['values'][r*8+event];key=word[position];buckets[key].append(softmax(vec(z,'after_address',r,event)))
   centroids={k:[mean([v[j]for v in values])for j in range(8)]for k,values in buckets.items()}
   key_after[result['events'][event]]=dict(mean_attention_after_KEY_by_public_key=centroids,KEY0_vs_KEY1_centroid_TV=tv(centroids[7],centroids[8]),within_key_TV_to_centroid={k:mean([tv(v,centroids[k])for v in values])for k,values in buckets.items()},topslot_after_KEY={k:dict(collections.Counter(max(range(8),key=lambda j:v[j])for v in values))for k,values in buckets.items()})
  for r,word in enumerate(words):
   positions=z['event_positions']['values'][r*8:r*8+8];firstkey=word[positions[0]];querykey=word[positions[4]];same=1 if firstkey==querykey else 3;other=3 if same==1 else 1
   a=vec(z,'read_attention',r,5);s=vec(z,'read_attention',r,same);o=vec(z,'read_attention',r,other)
   retrieval.append((tv(a,s),tv(a,o),sum(x*y for x,y in zip(a,s)),sum(x*y for x,y in zip(a,o)),int(max(range(8),key=lambda j:a[j])==max(range(8),key=lambda j:s[j]))))
   write_separation.append(tv(s,o))
  flip={}
  for event in(4,5,6)+( (7,)if stage==3 else()):
   distances=[];changed=0
   for left,right in groups.values():
    a=vec(z,'read_attention',left,event);b=vec(z,'read_attention',right,event);distances.append(tv(a,b));changed+=max(range(8),key=lambda k:a[k])!=max(range(8),key=lambda k:b[k])
   flip[result['events'][event]]=dict(queryflip_attention_TV=summary(distances),topslot_changed=changed,pairs=256)
  reports[record['key']]=dict(P=record['P'],raw_sha256=record['raw_sha256'],events=events,formula_max_abs_error=maximum,KEY_after_update_geometry=key_after,READ_vs_true_historical_WRITE=dict(same_key_TV=summary([x[0]for x in retrieval]),other_key_TV=summary([x[1]for x in retrieval]),same_key_dot=summary([x[2]for x in retrieval]),other_key_dot=summary([x[3]for x in retrieval]),same_key_topslot_match=sum(x[4]for x in retrieval)/512,fraction_TV_prefers_same_key=mean([int(x[0]<x[1])for x in retrieval]),write_footprint_separation_TV=summary(write_separation),scope='public KEY identity maps READ to historical WRITE; slot index never equals logical key by assumption'),queryflip=flip)
 value=dict(status='SAVED_ARRAYS_CHECKED_NO_NN',audit_result_sha256=sha(AUDIT/'result.json'),source_sha256=result['source_sha256'],records=reports,SGD_updates=0,NN_executions=0,scope='posthoc stdlib raw attention/address and exacttransition arithmetic, no checkpoint deserialization/NN/NumPy/Torch; content decoding not performed; historicalwriteattention is a soft footprint, not an oracle address')
 out=P/'typed-three-nn-saved-address-verification-v1.json'
 with out.open('x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(dict(path=str(out),sha256=sha(out),records={k:dict(formula_max_abs_error=v['formula_max_abs_error'],retrieval=v['READ_vs_true_historical_WRITE'],queryflip=v['queryflip'])for k,v in reports.items()}),sort_keys=True))
if __name__=='__main__':main()
