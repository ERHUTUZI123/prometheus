"""Stdlib-only posthoc key/address analysis of frozen new two-NN shared-address VALID raw.

No model/checkpoint deserialization, training, SSH, or input intervention.
Physical slot indices are tracked but never equated to logical addresses.
"""
import argparse, json, math
from pathlib import Path
from verify_factorized_graph600_saved import ROOT, npz, sha

EVENTS=('write_first','write_second','after_read','before_mix_after_key_other','at_mix')
AUDIT=ROOT/'runs/parameter-input-history-shared-address-two-nn-valid-diagnosis-v1/audit'
TRUTH=ROOT/'runs/parameter-posterior-value-valid-prefix-truth-v1/truth/prefix-truth.npz'
def mean(v): return math.fsum(v)/len(v) if v else None
def dot(a,b): return math.fsum(x*y for x,y in zip(a,b))
def norm(a): return math.sqrt(dot(a,a))
def dist(a,b): return norm([x-y for x,y in zip(a,b)])
def argmax(v): return max(range(len(v)),key=v.__getitem__)
def entropy(v): return -math.fsum(x*math.log(x) for x in v if x>0)/math.log(8)
def cosine(a,b):
    den=norm(a)*norm(b)
    return dot(a,b)/den if den>1e-30 else 0.
def tv(a,b): return .5*math.fsum(abs(x-y)for x,y in zip(a,b))
def part(gram):
    trace=math.fsum(gram[i][i]for i in range(8));den=math.fsum(x*x for row in gram for x in row)
    return trace*trace/den if den>1e-60 else 0.
def vector(z,key,r,e):
    a=z[key];width=math.prod(a['shape'][2:]);start=(r*5+e)*width
    return list(a['values'][start:start+width])
def matrix(z,key,r,e,width):
    a=vector(z,key,r,e);return[a[i:i+width]for i in range(0,len(a),width)]
def geom(keys):
    center=[mean([row[j]for row in keys])for j in range(16)]
    centered=[[x-y for x,y in zip(row,center)]for row in keys]
    gram=[[dot(a,b)for b in keys]for a in keys]
    cgram=[[dot(a,b)for b in centered]for a in centered]
    pairs=[(i,j)for i in range(8)for j in range(i+1,8)]
    return dict(mean_key_norm=mean([norm(x)for x in keys]),mean_pair_cosine=mean([cosine(keys[i],keys[j])for i,j in pairs]),rms_pair_distance=math.sqrt(mean([dist(keys[i],keys[j])**2 for i,j in pairs])),rms_within_slot_center_distance=math.sqrt(mean([dot(x,x)for x in centered])),Gram_participation=part(gram),centered_Gram_participation=part(cgram))
def summary(values):
    values=sorted(values);n=len(values)
    return dict(mean=mean(values),min=values[0],p50=values[n//2],max=values[-1])
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('new receipt only')
    truth=npz(TRUTH);audit=json.loads((AUDIT/'result.json').read_text());marker=json.loads((AUDIT/'ALL_NN_OUTPUTS_SAVED_BEFORE_PRIVILEGED_TRUTH.json').read_text())
    assert sha(TRUTH)==audit['privileged_truth_sha256']and marker['all2_NN_raw_outputs_saved']and marker['privileged_truth_loaded']is False
    out=dict(schema='frozen-shared-address-two-NN-VALID-address-key-analysis-v1',source_script_sha256=sha(Path(__file__)),audit_result_sha256=sha(AUDIT/'result.json'),truth_sha256=sha(TRUTH),scope='stdlib savedraw posthoc; no NN/SSH/SGD/intervention; physicalslots notlogicalkeys; correctvalues availability ownedbyseparateanalysis',models={})
    def state(r,e,before=False):
        x=truth['true_state_before_event'if before else'true_state_after_event']['values'];i=(r*5+e)*6;return x[i:i+6]
    for rec in audit['records']:
        f=AUDIT/rec['raw_file'];assert sha(f)==rec['raw_sha256'];z=npz(f);valid=z['event_valid']['values'];events={}
        for e,name in enumerate(EVENTS):
            rows=[r for r in range(512)if valid[r*5+e]]
            if not rows:events[name]=dict(rows=0);continue
            before=[];after=[];read=[];write=[];gates=[];updates=[];deltas=[];grammean=[[0.]*8 for _ in range(8)];cosmean=[[0.]*8 for _ in range(8)]
            grouped={k:[]for k in range(2)}
            for r in rows:
                kb=matrix(z,'before_keys',r,e,16);ka=matrix(z,'after_keys',r,e,16);gb=geom(kb);ga=geom(ka);before.append(gb);after.append(ga)
                ra=vector(z,'read_attention',r,e);wa=vector(z,'write_addresses',r,e);read.append(ra);write.append(wa);gates.append(vector(z,'write_gate',r,e)[0]);updates.append(math.sqrt(math.fsum(dist(x,y)**2 for x,y in zip(kb,ka))))
                deltas.append(ga['rms_pair_distance']/gb['rms_pair_distance']if gb['rms_pair_distance']>1e-30 else 0.)
                grouped[int(state(r,e)[3])].append(ra)
                for i in range(8):
                    for j in range(8):grammean[i][j]+=dot(kb[i],kb[j])/len(rows);cosmean[i][j]+=cosine(kb[i],kb[j])/len(rows)
            centroids={str(k):[mean([v[j]for v in vs])for j in range(8)]for k,vs in grouped.items()if vs}
            events[name]=dict(rows=len(rows),read_attention_normalized_entropy=summary([entropy(v)for v in read]),write_attention_normalized_entropy=summary([entropy(v)for v in write]),read_max=summary([max(v)for v in read]),write_max=summary([max(v)for v in write]),write_gate=summary(gates),key_update_Frobenius_norm=summary(updates),key_pair_distance_after_over_before=summary(deltas),keys_before={k:summary([g[k]for g in before])for k in before[0]},keys_after={k:summary([g[k]for g in after])for k in after[0]},mean_before_key_Gram=grammean,mean_before_key_cosine_Gram=cosmean,read_top_slot_histogram=[sum(argmax(v)==j for v in read)for j in range(8)],write_top_slot_histogram=[sum(argmax(v)==j for v in write)for j in range(8)],read_attention_centroid_by_logical_address=centroids,read_centroid_logical_address_TV=tv(centroids['0'],centroids['1'])if len(centroids)==2 else None)
        # Same physical slots can be followed through time without granting a
        # fixed task address/slot correspondence. Compare target vs other write.
        queries={}
        for e in(2,4):
            rows=[r for r in range(512)if valid[r*5+e]]
            if not rows:continue
            metrics=[]
            for r in rows:
                address=state(r,e)[3];target=next(j for j in(0,1)if state(r,j)[3]==address);other=1-target
                ra=vector(z,'read_attention',r,e);wt=vector(z,'write_addresses',r,target);wo=vector(z,'write_addresses',r,other)
                kt=matrix(z,'after_keys',r,target,16);kq=matrix(z,'before_keys',r,e,16)
                y=int(truth['true_selected_memory_at_query']['values'][r*5+e]);values=matrix(z,'before_values',r,e,8);masses=[math.fsum(v)for v in values];cond=[v[y]/m if m>1e-6 else 0. for v,m in zip(values,masses)]
                best=max(cond);maxslots=[i for i,x in enumerate(cond)if abs(x-best)<1e-8];selected=argmax(ra);achieved=dot(ra,[v[y]for v in values])/dot(ra,masses)
                metrics.append(dict(read_minus_target_write_TV=tv(ra,wt),read_minus_other_write_TV=tv(ra,wo),target_minus_other_address_alignment=dot(ra,wt)-dot(ra,wo),top_read_matches_target_write=float(selected==argmax(wt)),top_read_matches_other_write=float(selected==argmax(wo)),target_other_write_TV=tv(wt,wo),target_other_write_overlap=1-tv(wt,wo),target_other_write_top_slot_same=float(argmax(wt)==argmax(wo)),target_write_to_query_key_Frobenius=math.sqrt(math.fsum(dist(x,y)**2 for x,y in zip(kt,kq))),best_conditional_target=best,achieved_conditional_target=achieved,attention_on_any_best_target_slot=math.fsum(ra[i]for i in maxslots),top_read_is_any_best_target_slot=float(selected in maxslots),correct_value_slot_exists=float(best>.5),correct_value_exists_read_not_target=float(best>.5 and argmax(vector(z,'read_p',r,e))!=y)))
            queries[EVENTS[e]]=dict(rows=len(rows),metrics={k:summary([v[k]for v in metrics])for k in metrics[0]},caution='matching historicalwritefootprints notsemanticproof; valuebestslots privilegedposthoconly, not NNinput; READ/MIX have differentopcode so pairedattentiondiff notpureaddress intervention')
        pairrows=[r for r in range(512)if valid[r*5+4]]
        paired=dict(rows=len(pairrows),READ_MIX_attention_TV=summary([tv(vector(z,'read_attention',r,2),vector(z,'read_attention',r,4))for r in pairrows]))if pairrows else dict(rows=0)
        out['models'][rec['key']]=dict(raw_sha256=rec['raw_sha256'],checkpoint_sha256=rec['checkpoint_sha256'],events=events,queries=queries,paired_READ_MIX=paired)
    with a.out.open('x')as f:json.dump(out,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(dict(out=str(a.out),sha256=sha(a.out),models=list(out['models']))))
if __name__=='__main__':main()
