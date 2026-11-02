"""Saved-array-only privileged value/slot analysis. No NumPy/Torch/NN/SSH.

Literal-q availability is not a general information-capacity test. Best-any-slot
is an oracle diagnostic upper bound, not a learned retrieval policy. Effective
conditional slot weights are attention*slotmass/readmass, not attention alone.
"""
import array,ast,hashlib,json,math,resource,struct,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
AUDIT=ROOT/'runs/parameter-input-history-shared-address-two-nn-valid-diagnosis-v1/audit'
TRUTH=ROOT/'runs/parameter-posterior-value-valid-prefix-truth-v1/truth/prefix-truth.npz'
EVENTS=('write_first','write_second','after_read','before_mix_after_key_other','at_mix')
EPS=1e-6

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load_npz(path):
    out={}
    with zipfile.ZipFile(path)as z:
        for name in z.namelist():
            raw=z.read(name);v=raw[6];offset=10 if v==1 else 12;length=int.from_bytes(raw[8:10]if v==1 else raw[8:12],'little');header=ast.literal_eval(raw[offset:offset+length].decode());assert not header['fortran_order'];kind={'<f4':'f','<f8':'d','<i8':'q','|b1':'b','|u1':'B'}[header['descr']];data=array.array(kind);data.frombytes(raw[offset+length:]);out[name[:-4]]=(header['shape'],data)
    return out

def at(d,key,row,event=0):
    shape,values=d[key];width=math.prod(shape[2:])if len(shape)>2 else 1;offset=(row*shape[1]+event)*width
    return list(values[offset:offset+width])
def truthstate(t,key,row,event):return at(t,key,row,event)
def valid(d,row,event):return bool(d['event_valid'][1][row*5+event])
def argmax(v):return max(range(len(v)),key=v.__getitem__)
def mean(x):return sum(x)/len(x)if x else None
def quantile(v,p):
    s=sorted(v);index=(len(s)-1)*p;lo=int(index);hi=math.ceil(index);return s[lo]+(index-lo)*(s[hi]-s[lo])
def stats(values):
    v=[float(x)for x in values if x is not None and math.isfinite(float(x))]
    return dict(n=len(v),mean=mean(v),p05=quantile(v,.05)if v else None,p50=quantile(v,.5)if v else None,p95=quantile(v,.95)if v else None,min=min(v)if v else None,max=max(v)if v else None)
def avail(slots,y):
    masses=[sum(slot)for slot in slots];eligible=[j for j,m in enumerate(masses)if m>EPS]
    conditional=[slots[j][y]/masses[j]for j in eligible];best=eligible[argmax(conditional)]if eligible else None;rawbest=argmax([s[y]for s in slots])
    return dict(best_cond=slots[best][y]/masses[best]if best is not None else None,best_cond_slot=best,best_cond_slot_mass=masses[best]if best is not None else None,best_unconditional=max(s[y]for s in slots),best_unconditional_slot=rawbest,best_unconditional_slot_mass=masses[rawbest],any_slot_argmax_true=any(argmax(slots[j])==y for j in eligible),qualified_slots=len(eligible),masses=masses)
def slots(d,key,row,event):
    flat=at(d,key,row,event);return[flat[j*8:(j+1)*8]for j in range(8)]
def entropy(p):return-sum(x*math.log(x)for x in p if x>0)
def write_npz(path,fields):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED)as z:
        for name,(shape,values)in fields.items():
            raw=array.array('d',values).tobytes();header=repr({'descr':'<f8','fortran_order':False,'shape':shape}).encode();header+=b' '*((16-(10+len(header)+1)%16)%16)+b'\n';z.writestr(name+'.npy',b'\x93NUMPY'+bytes([1,0])+struct.pack('<H',len(header))+header+raw)

def main():
    assert sys.byteorder=='little';started=time.perf_counter();CPU=time.process_time();truth=load_npz(TRUTH);receipt=json.loads((AUDIT/'result.json').read_text());models={};derived_files={}
    for key in('shared-address-stage2','shared-address-stage3'):
        raw=AUDIT/(key+'-NN-events-before-truth.npz');d=load_npz(raw);record=next(x for x in receipt['records']if x['key']==key);assert sha(raw)==record['raw_sha256'];assert sha(TRUTH)==receipt['privileged_truth_sha256'];basis_flat=d['encoder_basis'][1];E=[list(basis_flat[i*32:(i+1)*32])for i in range(8)];Gram=[[sum(a*b for a,b in zip(x,y))for y in E]for x in E];summary={};fields={};all_rows={};max_u_error=0.;max_weight_error=0.;max_cond_upper_violation=0.;max_raw_upper_violation=0.
        metricnames=('mass','conditional_entropy','read_norm','conditional_read_norm','norm_over_mass_sqrtH','true_read_raw_mass','true_read_conditional_mass','best_slot_conditional_mass','best_cond_slot_mass','best_slot_unconditional_mass','best_slot_argmax_true','conditional_best_minus_achieved_gap','unconditional_best_minus_achieved_gap','raw_attention_on_best_cond_slot','effective_attention_on_best_cond_slot','equal_slotmass_attention_conditional_true_mass','actual_minus_equal_slotmass_true_mass')
        per={name:[float('nan')]*(512*5)for name in metricnames}
        for event,name in enumerate(EVENTS):
            rows=[];values={n:[]for n in metricnames};qcorrect=readcorrect=anycorrect=0;nonzero=0;distinct=[];same=[]
            for row in range(512):
                if not valid(d,row,event):continue
                true=truthstate(truth,'true_state_after_event',row,event);y=int(truth['true_selected_memory_at_query'][1][row*5+event]);qtrue=true[0];u=at(d,'read_p',row,event);m=sum(u);before=slots(d,'before_values',row,event);alpha=at(d,'read_attention',row,event);a=avail(before,y);slotm=a['masses'];prediction=argmax(at(d,'after_logp',row,event));qcorrect+=prediction==qtrue;readcorrect+=argmax(u)==y;anycorrect+=a['any_slot_argmax_true'];nonzero+=m>EPS
                reconstructed=[sum(alpha[j]*before[j][s]for j in range(8))for s in range(8)];max_u_error=max(max_u_error,max(abs(x-z)for x,z in zip(u,reconstructed)))
                cond=[x/m for x in u]if m>EPS else None;r=[sum(u[s]*E[s][i]for s in range(8))for i in range(32)];norm=math.sqrt(sum(x*x for x in r));best=a['best_cond_slot'];effective=[alpha[j]*slotm[j]/m for j in range(8)]if m>EPS else None
                eligible=[j for j,v in enumerate(slotm)if v>EPS];alpha_mass=sum(alpha[j]for j in eligible);equalmass=sum(alpha[j]*before[j][y]/slotm[j]for j in eligible)/alpha_mass if alpha_mass>EPS else None
                if m>EPS:
                    effective_true=sum(effective[j]*(before[j][y]/slotm[j]if slotm[j]>EPS else 0.)for j in range(8));max_weight_error=max(max_weight_error,abs(effective_true-cond[y]));max_cond_upper_violation=max(max_cond_upper_violation,cond[y]-a['best_cond'])
                max_raw_upper_violation=max(max_raw_upper_violation,u[y]-a['best_unconditional'])
                item=dict(mass=m,conditional_entropy=entropy(cond)if cond else None,read_norm=norm,conditional_read_norm=norm/m if m>EPS else None,norm_over_mass_sqrtH=norm/(m*32**.5)if m>EPS else None,true_read_raw_mass=u[y],true_read_conditional_mass=cond[y]if cond else None,best_slot_conditional_mass=a['best_cond'],best_cond_slot_mass=a['best_cond_slot_mass'],best_slot_unconditional_mass=a['best_unconditional'],best_slot_argmax_true=float(a['any_slot_argmax_true']),conditional_best_minus_achieved_gap=a['best_cond']-cond[y]if cond and a['best_cond']is not None else None,unconditional_best_minus_achieved_gap=a['best_unconditional']-u[y],raw_attention_on_best_cond_slot=alpha[best]if best is not None else None,effective_attention_on_best_cond_slot=effective[best]if effective and best is not None else None,equal_slotmass_attention_conditional_true_mass=equalmass,actual_minus_equal_slotmass_true_mass=cond[y]-equalmass if cond and equalmass is not None else None)
                for metric,value in item.items():
                    values[metric].append(value);per[metric][row*5+event]=value if value is not None else float('nan')
                w1=truthstate(truth,'true_state_after_event',row,0)[0];w2=truthstate(truth,'true_state_after_event',row,1)[0];(distinct if w1!=w2 else same).append(item);rows.append(dict(row=row,q_correct=prediction==qtrue,read_argmax_correct=argmax(u)==y,availability=a,item=item))
            if not rows:summary[name]=dict(rows=0,scope='stage2MIXunexecuted; KEYotherreadnotMIXread');continue
            summary[name]=dict(rows=len(rows),cached_q_correct=qcorrect,read_argmax_true_memory_correct=readcorrect,best_any_slot_argmax_true_memory_correct=anycorrect,nonzero_read_mass_rows=nonzero,metrics={n:stats(v)for n,v in values.items()},distinct_true_written_values_rows=len(distinct),distinct_value_metrics={n:stats([x[n]for x in distinct])for n in metricnames},equal_true_written_values_rows=len(same),equal_value_metrics={n:stats([x[n]for x in same])for n in metricnames},best_cond_atleast_half=sum(x['availability']['best_cond']>=.5 for x in rows),best_cond_atleast_point9=sum(x['availability']['best_cond']>=.9 for x in rows),best_cond_winner_mass_atmost_eps10=sum(x['availability']['best_cond_slot_mass']<=10*EPS for x in rows))
            all_rows[event]=rows
        # Literal-q content trajectory for each actual written value, independent
        # of which logical key was queried. No physical-slot=logical-key mapping.
        timeline=[(0,'before'),(0,'after'),(1,'before'),(1,'after'),(2,'before'),(2,'after'),(3,'before'),(3,'after'),(4,'before'),(4,'after')];survival={}
        for operand in(0,1):
            operand_result={};per_timeline=[]
            for event,phase in timeline:
                entries=[]
                for row in range(512):
                    if not valid(d,row,event):continue
                    y=truthstate(truth,'true_state_after_event',row,operand)[0];a=avail(slots(d,phase+'_values',row,event),y);w1=truthstate(truth,'true_state_after_event',row,0)[0];w2=truthstate(truth,'true_state_after_event',row,1)[0];entries.append((row,a,w1!=w2))
                if not entries:continue
                label=EVENTS[event]+'_'+phase;operand_result[label]=dict(rows=len(entries),any_slot_argmax_true_value=sum(x[1]['any_slot_argmax_true']for x in entries),best_conditional=stats([x[1]['best_cond']for x in entries]),best_cond_slotmass=stats([x[1]['best_cond_slot_mass']for x in entries]),best_unconditional=stats([x[1]['best_unconditional']for x in entries]),distinct_rows=sum(x[2]for x in entries),distinct_any_argmax=sum(x[1]['any_slot_argmax_true']for x in entries if x[2]),distinct_best_conditional=stats([x[1]['best_cond']for x in entries if x[2]]),distinct_best_unconditional=stats([x[1]['best_unconditional']for x in entries if x[2]]));per_timeline.append((label,{r:a for r,a,_ in entries}))
            deltas={}
            for left,right in[(EVENTS[operand]+'_before',EVENTS[operand]+'_after'),('write_second_before','write_second_after'),('write_second_after','after_read_before'),('after_read_before','after_read_after'),('after_read_after','before_mix_after_key_other_before'),('before_mix_after_key_other_after','at_mix_before')]:
                table=dict(per_timeline)
                if left not in table or right not in table:continue
                common=sorted(set(table[left])&set(table[right]));dc=[table[right][r]['best_cond']-table[left][r]['best_cond']for r in common];du=[table[right][r]['best_unconditional']-table[left][r]['best_unconditional']for r in common]
                deltas[left+'__TO__'+right]=dict(rows=len(common),best_conditional_delta=stats(dc),best_unconditional_delta=stats(du),lost_any_literal_argmax=sum(table[left][r]['any_slot_argmax_true']and not table[right][r]['any_slot_argmax_true']for r in common),gained_any_literal_argmax=sum(not table[left][r]['any_slot_argmax_true']and table[right][r]['any_slot_argmax_true']for r in common))
            survival['true_value_at_write_'+str(operand+1)]=dict(timeline=operand_result,deltas=deltas)
        # End-task correctness can coexist with nonliteral prefix/read codes.
        contingency={}
        if 4 in all_rows:
            final={x['row']:x['q_correct']for x in all_rows[4]}
            for event in(2,3,4):
                if event not in all_rows:continue
                for metric in('q_correct','read_argmax_correct'):
                    table=[[0,0],[0,0]]
                    for x in all_rows[event]:table[int(x[metric])][int(final[x['row']])]+=1
                    contingency[EVENTS[event]+'_'+metric+'_vs_MIX_correct']=dict(rows_predictor_false_true_columns_final_false_true=table,scope='descriptivejointcounts, notcausalnecessity ormatchedconditionalcomparison')
        fields.update({name:((512,5),v)for name,v in per.items()});derived=ROOT/('shared-address-valid-value-derived-'+key+'-v1.npz');write_npz(derived,fields);derived_files[key]=dict(file=derived.name,sha256=sha(derived))
        models[key]=dict(raw_file=str(raw.relative_to(ROOT)),raw_sha256=sha(raw),P=record['resources']['P'],events=summary,true_written_value_survival=survival,contingency=contingency,encoder_basis_Gram=Gram,encoder_basis_norms=[math.sqrt(Gram[i][i])for i in range(8)],reconstruction=dict(max_u_minus_attention_V_abs=max_u_error,max_conditional_effective_weight_identity_abs=max_weight_error,max_best_cond_upperbound_violation=max_cond_upper_violation,max_best_raw_upperbound_violation=max_raw_upper_violation),derived_per_row=derived_files[key])
        assert max_u_error<2e-6 and max_weight_error<2e-6 and max_cond_upper_violation<2e-6 and max_raw_upper_violation<2e-6
    result=dict(status='COMPLETED_SAVED_ARRAY_VALUE_SLOT_DIAGNOSIS',analysis='stdlibZIP/NPY/float64 arithmetic only; noNN/Torch/NumPy/SSH/training',EPS=EPS,audit_result_sha256=sha(AUDIT/'result.json'),truth_sha256=sha(TRUTH),models=models,definitions=dict(read_conditional='qREAD=u/sum(u) for mass>1e-6',slot_conditional='V_j[y]/sum(V_j) for slotmass>1e-6',effective_slot_weight='a_j*slotmass_j/readmass',best_slot_scope='oracle diagnostic upperbound, no slot=logicalkey assumption; report conditionalwinner mass andbestunconditional toavoidtiny-massartifact',read_norm='||u@E|| recomputedfloat64 fromsavedFP32u/E, notadditionalNNexecution; norm/(mass*sqrt32) separatesmass fromdirection/prototype cancellation',survival='literaltruevalueavailability inanyslot; notcausalinformationretention orabsenceofhiddencontroller/h/keycoding',distinct_stratum='two true written values unequal; all512 main andequal64 separate, no rowdeletion'),cost_scope=dict(original_audit_SGD_updates=0,original_audit_wall_seconds=receipt['wall_seconds'],original_audit_CPU_seconds=receipt['CPU_seconds'],original_audit_active_transitions=receipt['active_transitions'],original_NN_logical_passes=2,new_NN_queries=0,new_teacher_feedback=0,analysis_wall_seconds=time.perf_counter()-started,analysis_CPU_seconds=time.process_time()-CPU,analysis_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)),full_FLOPs=None,provider_effective_compute=None),interpretation_limits=['cached/WRITEp have onlyterminalsupervision; prefixq mismatch notinformation-capacity proof','bestconditionalgap alone notqueryfailure: slotmass changes effectiveweights','overallmass-normalization cannot change conditionalq/argmax/entropy, onlyreadscale','stage2stopbeforeMIX hasnoMIXtokenconditionedread;KEYotherisnotitsproxy'])
    path=ROOT/'shared-address-valid-value-slot-analysis-v1.json'
    with path.open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print('saved',path.name,sha(path))
    for key,m in models.items():
        for event in('after_read','at_mix'):
            r=m['events'][event]
            if not r['rows']:continue
            z=r['metrics'];print(key,event,'n',r['rows'],'qcorrect',r['cached_q_correct'],'readcorrect',r['read_argmax_true_memory_correct'],'anyslot',r['best_any_slot_argmax_true_memory_correct'],'mass',z['mass']['mean'],'readcond_y',z['true_read_conditional_mass']['mean'],'bestcond',z['best_slot_conditional_mass']['mean'],'bestslotmass',z['best_cond_slot_mass']['mean'],'best_raw',z['best_slot_unconditional_mass']['mean'],'H',z['conditional_entropy']['mean'],'norm_ratio',z['norm_over_mass_sqrtH']['mean'],'gap',z['conditional_best_minus_achieved_gap']['mean'])
if __name__=='__main__':main()
