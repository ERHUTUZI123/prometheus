"""Stdlib linear decomposition of saved normalized reads, no NN execution."""
import hashlib,importlib.util,json,math,time
from pathlib import Path
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('saved',P/'verify_factorized_graph600_saved.py');saved=importlib.util.module_from_spec(spec);spec.loader.exec_module(saved)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def norm(v):return math.sqrt(sum(x*x for x in v))
def mean(v):return math.fsum(v)/len(v)
def main():
 start=time.perf_counter();cpu=time.process_time();folder=P/'runs/parameter-ab-write-direction-controls-cpu-v1/audit';result=json.loads((folder/'result.json').read_text());path=folder/'reference-raw-before-targets.npz'
 if sha(path)!=result['records'][0]['sha256']:raise ValueError('actualreferencebytes changed')
 raw=saved.npz(path);L=raw['w']['shape'][1];public=saved.npz(P/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated/data/stage3-valid.npz');maximum=0.;events={name:[]for name in('before_first_WRITE','READ','MIX')};final_mass={};rows=[]
 def vec(key,row,t,width):return list(raw[key]['values'][(row*L+t)*width:(row*L+t+1)*width])
 for row,length in enumerate(public['lengths']['values']):
  word=list(public['tokens']['values'][row*128:row*128+length]);writes=[i for i,k in enumerate(word)if k==9];w1,w2=writes
  def region(i):
   if i==w1:return'first_WRITE'
   if i==w2:return'second_WRITE'
   if i<w1:return'first_value_construction'
   if i<w2:return'second_value_construction'
   if i<length-4:return'distractor'
   return'suffix'
  decay=[1.]*8
  for i in range(int(length)-1,-1,-1):
   w=vec('w',row,i,8);lam=[a*b for a,b in zip(w,decay)];name=region(i);final_mass[name]=final_mass.get(name,0.)+sum(lam);decay=[b*(1-a)for a,b in zip(w,decay)]
  for name,t in [('before_first_WRITE',w1),('READ',int(length)-3),('MIX',int(length)-1)]:
   gate=vec('action_probabilities',row,t,7)[5]
   if gate<=0:raise ValueError('savedpositiveSoftSTORE required to inferattention')
   attention=[x/gate for x in vec('w',row,t,8)];coverage=vec('coverage',row,t-1,8)if t else[0.]*8;scale=[a/max(z,1e-4)for a,z in zip(attention,coverage)];decay=[1.]*8;coeff=[0.]*int(t);components={key:[0.]*32 for key in('first_WRITE','second_WRITE','other')}
   for i in range(int(t)-1,-1,-1):
    w=vec('w',row,i,8);lam=[a*b for a,b in zip(w,decay)];b=sum(x*y for x,y in zip(scale,lam));coeff[i]=b;h=vec('h',row,i,32);key='first_WRITE'if i==w1 else'second_WRITE'if i==w2 else'other';components[key]=[x+b*y for x,y in zip(components[key],h)];decay=[b*(1-a)for a,b in zip(w,decay)]
   reconstructed=[sum(v[j]for v in components.values())for j in range(32)];actual=vec('read',row,t,32);error=max(abs(a-b)for a,b in zip(reconstructed,actual));maximum=max(maximum,error);b1=coeff[w1]if w1<t else 0.;b2=coeff[w2]if w2<t else 0.;querykey=word[t-1]if name in('READ','MIX')else word[0];firstkey=word[0];same=b1 if querykey==firstkey else b2;other=b2 if querykey==firstkey else b1
   item=dict(row=row,coefficient_sum=sum(coeff),first_WRITE_coefficient=b1,second_WRITE_coefficient=b2,other_coefficient=sum(coeff)-b1-b2,logical_requested_WRITE_coefficient=same,logical_other_WRITE_coefficient=other,read_norm=norm(actual),component_norms={k:norm(v)for k,v in components.items()},component_cosines_to_read={k:(sum(x*y for x,y in zip(v,actual))/(norm(v)*norm(actual))if norm(v)>0 and norm(actual)>0 else None)for k,v in components.items()},reconstruction_max_abs=error)
   events[name].append(item)
 reports={}
 for name,items in events.items():
  keys=('coefficient_sum','first_WRITE_coefficient','second_WRITE_coefficient','other_coefficient','logical_requested_WRITE_coefficient','logical_other_WRITE_coefficient','read_norm')
  reports[name]={key:mean([v[key]for v in items])for key in keys};reports[name].update(requested_more_than_other=sum(v['logical_requested_WRITE_coefficient']>v['logical_other_WRITE_coefficient']for v in items),component_norm_mean={k:mean([v['component_norms'][k]for v in items])for k in('first_WRITE','second_WRITE','other')},component_cosine_to_read_mean={k:mean([v['component_cosines_to_read'][k]for v in items if v['component_cosines_to_read'][k]is not None])if any(v['component_cosines_to_read'][k]is not None for v in items)else None for k in('first_WRITE','second_WRITE','other')})
 value=dict(status='SAVED_REFERENCE_LINEAR_READ_DECOMPOSITION',source_script_sha256=sha(Path(__file__)),reference_raw_sha256=sha(path),control_result_sha256=sha(folder/'result.json'),formula='read_t = sum_i beta_i h_i; beta_i=sum_j attention_tj*w_ij*product_{i<k<t}(1-w_kj)/max(z_pre_tj,eps)',max_reconstruction_abs_error=maximum,events=reports,final_coverage_effective_mass_by_public_region=final_mass,per_row_events=events,scope='posthoc publicprogram positions labeltemporal/logicalWRITE only; actualNNOLDh direction notassumed semanticclass; linearcontribution/norm/cosine doesnotprovealllatentinformation absent; noNN/Torch/NumPy/checkpointloading',NN_executions=0,SGD_updates=0,analysis_wall_seconds=time.perf_counter()-start,analysis_CPU_seconds=time.process_time()-cpu)
 out=P/'ab-reference-read-history-decomposition-v1.json'
 with out.open('x')as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(dict(file=str(out),sha256=sha(out),events=reports,effective_mass=final_mass,max_reconstruction_abs_error=maximum),sort_keys=True))
if __name__=='__main__':main()
