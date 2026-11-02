"""NPZ-only raw-h memory geometry/formula analysis; no NN or probability decode."""
import hashlib,json,math,resource,sys,time
from pathlib import Path
from verify_factorized_graph600_saved import npz
from analyze_posterior_valid_slot_values import write_npz
P=Path(__file__).resolve().parent;A=P/'runs/parameter-typed-three-nn-valid-actions-cpu-v1/audit';EPS=1e-8
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def vector(d,k,row,event):
 x=d[k];shape=x['shape'];n=math.prod(shape[2:])if len(shape)>2 else 1;off=(row*shape[1]+event)*n;return list(x['values'][off:off+n])
def norm(v):return math.sqrt(sum(x*x for x in v))
def cosine(a,b):
 na,nb=norm(a),norm(b);return sum(x*y for x,y in zip(a,b))/(na*nb)if na>EPS and nb>EPS else None
def stat(values):
 a=sorted(float(x)for x in values if x is not None and math.isfinite(x))
 def quantile(q):
  n=(len(a)-1)*q;i=int(n);j=math.ceil(n);return a[i]+(n-i)*(a[j]-a[i])
 return dict(n=len(a),mean=sum(a)/len(a)if a else None,p05=quantile(.05)if a else None,p50=quantile(.5)if a else None,p95=quantile(.95)if a else None,min=min(a)if a else None,max=max(a)if a else None)
def slots(flat):return[flat[j*32:(j+1)*32]for j in range(8)]
def availability(values,reference,read):
 nr=norm(reference);eligible=[j for j,s in enumerate(values)if norm(s)>EPS and nr>EPS]
 if not eligible:return dict(best_cos=None,winning_slot_norm_ratio=None,best_relative_L2=None,read_cos=cosine(read,reference),read_norm_ratio=norm(read)/nr if nr>EPS else None,qualified_slots=0)
 best=max(eligible,key=lambda j:cosine(values[j],reference));return dict(best_cos=cosine(values[best],reference),winning_slot_norm_ratio=norm(values[best])/nr,best_relative_L2=min(norm([x-y for x,y in zip(s,reference)])/nr for s in values),read_cos=cosine(read,reference),read_norm_ratio=norm(read)/nr,qualified_slots=len(eligible))
def main():
 start=time.perf_counter();cpu=time.process_time();receipt=json.loads((A/'result.json').read_text());assert receipt['status']=='COMPLETED_TYPED_THREE_OWNED_NN_CPU_VALID_ACTION_DIAGNOSIS';events=receipt['events'];models={}
 for record in receipt['records']:
  key=record['key'];raw=A/record['raw_file'];assert sha(raw)==record['raw_sha256'];d=npz(raw);summary={};survival={};fields={};maxerrors={k:0. for k in('read_aV','STORE_oldh','write_value_OLDh','h_compute_LOAD','weights_soft','compute_total')}
  for e,name in enumerate(events):
   rows=[]
   for row in range(512):
    if not d['event_valid']['values'][row*8+e]:continue
    h=vector(d,'before_h',row,e);after=vector(d,'after_h',row,e);before=slots(vector(d,'before_values',row,e));post=slots(vector(d,'after_values',row,e));read=vector(d,'read',row,e);a=vector(d,'read_attention',row,e);weights=vector(d,'action_weights',row,e);prob=vector(d,'action_probabilities',row,e);chi=vector(d,'candidate_h',row,e);stored=vector(d,'write_value',row,e);store=vector(d,'write_gate',row,e)[0];comp=vector(d,'compute_total',row,e)[0];load=weights[4]
    rr=[sum(a[j]*before[j][i]for j in range(8))for i in range(32)];vv=[[before[j][i]+store*a[j]*(h[i]-before[j][i])for i in range(32)]for j in range(8)];hh=[h[i]+comp*(chi[i]-h[i])+load*(read[i]-h[i])for i in range(32)]
    for target,left,right in [('read_aV',read,rr),('STORE_oldh',[x for s in post for x in s],[x for s in vv for x in s]),('write_value_OLDh',stored,h),('h_compute_LOAD',after,hh),('weights_soft',weights,prob)]:maxerrors[target]=max(maxerrors[target],max(abs(x-y)for x,y in zip(left,right)))
    maxerrors['compute_total']=max(maxerrors['compute_total'],abs(comp-sum(weights[:4])));assert abs(store-weights[5])<1e-7 and abs(sum(a)-1)<2e-6
    dc=[comp*(chi[i]-h[i])for i in range(32)];dl=[load*(read[i]-h[i])for i in range(32)];nc,nl=norm(dc),norm(dl);deltaV=norm([post[j][i]-before[j][i]for j in range(8)for i in range(32)]);item=dict(old_h_norm=norm(h),read_norm=norm(read),after_h_norm=norm(after),candidate_h_norm=norm(chi),STORE_probability=store,LOAD_probability=load,compute_probability=comp,STORE_Frobenius_delta=deltaV,STORE_delta_per_oldh_norm=deltaV/norm(h)if norm(h)>EPS else None,read_to_oldh_cos=cosine(read,h),read_to_afterh_cos=cosine(read,after),LOAD_term_norm=nl,compute_term_norm=nc,LOAD_norm_fraction=nl/(nl+nc)if nl+nc>EPS else None,LOAD_vs_compute_cos=cosine(dl,dc),write_value_vs_candidate_relativeL2=norm([x-y for x,y in zip(stored,chi)])/norm(h)if norm(h)>EPS else None)
    rows.append(item)
    for field,value in item.items():
     fields.setdefault(field,[float('nan')]*4096)[row*8+e]=value if value is not None else float('nan')
   summary[name]=dict(rows=len(rows),metrics={k:stat([x[k]for x in rows])for k in rows[0]}if rows else{},LOAD_term_larger_than_compute=sum(x['LOAD_term_norm']>x['compute_term_norm']for x in rows),scope='componentnormsnotcausalexplainedvariance; LOADandcompute contributionsmaycancel')
  # Compare later raw slots/read with the actual OLD h offered at each WRITE.
  # This is own-vector geometry; no assumption it is a canonical taskvalue.
  for operand,write_event in enumerate((1,3)):
   timeline={}
   for e,name in enumerate(events):
    for phase in('before','after'):
     entries=[]
     for row in range(512):
      if not d['event_valid']['values'][row*8+e]:continue
      ref=vector(d,'write_value',row,write_event);entry=availability(slots(vector(d,phase+'_values',row,e)),ref,vector(d,'read',row,e));entries.append(entry)
     if entries:timeline[name+'_'+phase]=dict(rows=len(entries),metrics={k:stat([x[k]for x in entries])for k in entries[0]},tiny_norm_winner=sum(x['winning_slot_norm_ratio']is not None and x['winning_slot_norm_ratio']<1e-3 for x in entries))
   survival['OWN_OLDh_at_WRITE'+str(operand+1)]=timeline
  assert max(maxerrors.values())<2e-6
  path=P/('typed-three-nn-raw-value-derived-'+key+'-v1.npz');write_npz(path,{k:((512,8),v)for k,v in fields.items()});models[key]=dict(raw_sha256=sha(raw),checkpoint_sha256=record['checkpoint_sha256'],P=record['P'],events=summary,own_write_vector_availability=survival,formula_max_abs_errors=maxerrors,derived_perrow=dict(file=path.name,sha256=sha(path)))
 result=dict(status='VERIFIED_TYPED_RAW_VALUE_FORMULAS_AND_SAVED_GEOMETRY',source_sha256=sha(Path(__file__)),audit_result_sha256=sha(A/'result.json'),EPS=EPS,models=models,limits=['rawV/h/read are32D signed NNcoordinates, NOTprobabilities; no mass/argmax-as-category/retraction claim','no prototype/readoutNNdecode isexecuted byreviewer; OWNwritevectorcosine isnotsemanticcorrectness','best-any-slot cosine isoraclegeometry, notlearnedretrieval; winnernorm andrelativeL2 suppliedtoavoidtinyvectorartifact','between-event changes do notlocateuniqueoverwrite token; laterwritevaluescanindirectlyretain/reencodeinformation','smallLOAD doesnotremove read fromCALL/caller features, so notproof ofnomemoryuse','stage2MIXunexecuted; CPUvsGPUfinalVALID differsup2.43e-5 butargmax0differences, peractualaudit'],costs=dict(original_audit_CPU_seconds=receipt['CPU_seconds'],original_audit_wall_seconds=receipt['wall_seconds'],original_active_transitions=receipt['active_transitions'],original_NN_logical_passes=3,original_NN_queries='alreadyexecutedROOTCPU3passes, noneadded',original_raw_NPZ_bytes=sum(x['raw_bytes']for x in receipt['records']),original_raw_array_payload_bytes=sum(x['raw_array_payload_bytes']for x in receipt['records']),original_CPU_peak_RSS_bytes=receipt['CPU_peak_RSS_bytes'],analysis_wall_seconds=time.perf_counter()-start,analysis_CPU_seconds=time.process_time()-cpu,analysis_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)),new_NN_queries=0,SGD_updates=0,full_FLOPs=None,provider_effective_compute=None))
 path=P/'typed-three-nn-raw-value-analysis-v1.json'
 with path.open('x')as f:json.dump(result,f,sort_keys=True,indent=2);f.write('\n')
 print(path.name,sha(path))
 for key,m in models.items():
  print('MODEL',key,'formula',m['formula_max_abs_errors'])
  for event in('WRITE_first','WRITE_second','READ','MIX'):
   z=m['events'][event]
   if not z['rows']:continue
   print(event,{k:round(z['metrics'][k]['mean'],6)for k in('old_h_norm','read_norm','STORE_probability','STORE_Frobenius_delta','LOAD_probability','LOAD_term_norm','compute_term_norm','LOAD_norm_fraction','read_to_afterh_cos')},'LOADlarger',z['LOAD_term_larger_than_compute'])
  for op,timeline in m['own_write_vector_availability'].items():
   print(op,{name:{k:round(z['mean'],5)for k,z in table['metrics'].items()if k!='qualified_slots'}for name,table in timeline.items()if name in('WRITE_first_after','WRITE_second_after','READ_before','MIX_before')})
if __name__=='__main__':main()
