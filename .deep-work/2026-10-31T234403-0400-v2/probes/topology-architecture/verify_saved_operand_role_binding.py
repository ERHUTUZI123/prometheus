"""Independent stdlib saved512 VALID role/binding arithmetic, no NN/SSH."""
import hashlib,json,math,time
from fractions import Fraction
from pathlib import Path
from analyze_posterior_valid_slot_values import load_npz
ROOT=Path(__file__).resolve().parent
GEN=ROOT/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated'
AUDIT=ROOT/'runs/parameter-posterior-value-four-nn-valid-diagnosis-v1/audit'
NEW=ROOT/'runs/parameter-input-history-shared-address-n8-seed0-fourstage-graph-v1/learn'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def argmax(v):return max(range(len(v)),key=v.__getitem__)
def pair_stats(pairs):
    return dict(pairs=len(pairs),label_changes=sum(x['label_changed']for x in pairs),prediction_changes=sum(x['pred_changed']for x in pairs),both_correct=sum(x['both_correct']for x in pairs),one_correct=sum(x['one_correct']for x in pairs),neither_correct=sum(not x['both_correct']and not x['one_correct']for x in pairs))
def main():
    started=time.perf_counter();CPU=time.process_time();truthpath=GEN/'counterfactual-truth.npz';publicpath=GEN/'data/stage3-valid.npz';metadata=json.loads((GEN/'data/metadata.json').read_text());assert sha(truthpath)==metadata['counterfactual_truth_sha256']and sha(publicpath)==metadata['files']['valid']['sha256'];truth=load_npz(truthpath)['valid_values_group_order'][1];public=load_npz(publicpath);old=json.loads((ROOT/'actual-posterior-valid-binding-symmetry-v1.json').read_text());base=[];lookup={}
    for row in range(512):
        g,v0,v1,o,w=map(int,truth[row*5:row*5+5]);target=int(public['targets'][1][row]);length=int(public['lengths'][1][row]);word=list(public['tokens'][1][row*128:row*128+length]);first,last=(v0,v1)if w==0 else(v1,v0);A=(3*v0+v1)%8;B=(3*v1+v0)%8
        assert target==(A if o==0 else B)and word[-4:]==[7+o,10,8-o,15]
        b=dict(row=row,group=g,v0=v0,v1=v1,query_order=o,write_order=w,target=target,logical0=A,logical1=B,temporal_first=(3*first+last)%8,temporal_last=(3*last+first)%8,temporal_query0_first=((3*first+last)if o==0 else(3*last+first))%8,temporal_query0_last=((3*last+first)if o==0 else(3*first+last))%8,query_label_invariant=A==B,word=word)
        assert b['query_label_invariant']==((2*(v0-v1))%8==0);base.append(b);lookup[g,v0,v1,o,w]=row
    assert len(lookup)==512 and all((g,a,b,o,w)in lookup for g in range(2)for a in range(8)for b in range(8)for o in range(2)for w in range(2))
    equal=sum((3*a+b)%8==(3*b+a)%8 for a in range(8)for b in range(8));ceiling=Fraction(equal*2+(64-equal),128);assert equal==16 and ceiling==Fraction(5,8)
    files={'posterior-stage3':AUDIT/'posterior-stage3-NN-events-before-truth.npz','raw-stage3':AUDIT/'raw-stage3-NN-events-before-truth.npz','shared-address-stage3':NEW/'stage3-valid-task3-predictions.npz'};models={}
    formula=('target','logical0','logical1','temporal_first','temporal_last','temporal_query0_first','temporal_query0_last')
    for key,path in files.items():
        d=load_npz(path)
        if key=='shared-address-stage3':
            assert list(d['row_index'][1])==list(range(512))
            for field in('targets','initial','lengths'):assert d[field][1]==public[field][1]
            pred=[argmax(d['soft_logp'][1][r*8:r*8+8])for r in range(512)]
        else:
            assert all(d['event_valid'][1][r*5+4]for r in range(512));pred=[argmax(d['after_logp'][1][(r*5+4)*8:(r*5+5)*8])for r in range(512)]
            for row,r in enumerate(old['models'][key]['per_row']):
                assert r['row']==row and r['pred']==pred[row]
                for field in formula:assert r[field]==base[row][field]
        rows=[dict(b,pred=pred[b['row']])for b in base];query=[];write=[]
        for r in rows:
            g,a,b,o,w=[r[x]for x in('group','v0','v1','query_order','write_order')]
            if o==0:
                s=rows[lookup[g,a,b,1,w]];assert r['word'][:-4]==s['word'][:-4]and len(r['word'])==len(s['word']);correct0=r['pred']==r['target'];correct1=s['pred']==s['target'];query.append(dict(row0=r['row'],row1=s['row'],group=g,v0=a,v1=b,write_order=w,label_changed=r['target']!=s['target'],pred_changed=r['pred']!=s['pred'],both_correct=correct0 and correct1,one_correct=correct0!=correct1))
            if w==0:
                s=rows[lookup[g,a,b,o,1]];assert r['target']==s['target']and r['word'][-4:]==s['word'][-4:]and r['word'][8+a+b:-4]==s['word'][8+a+b:-4];correct0=r['pred']==r['target'];correct1=s['pred']==s['target'];write.append(dict(row0=r['row'],row1=s['row'],group=g,v0=a,v1=b,query_order=o,label_changed=False,pred_changed=r['pred']!=s['pred'],both_correct=correct0 and correct1,one_correct=correct0!=correct1,query_nondegenerate=(2*(a-b))%8!=0))
        coverage=sum(r['pred']in(r['logical0'],r['logical1'])for r in rows);agreements={f:dict(correct=sum(r['pred']==r[f]for r in rows),rows=512)for f in formula}
        qdiff=[x for x in query if x['label_changed']];qsame=[x for x in query if not x['label_changed']]
        if key in old['models']:
            for f in formula:assert agreements[f]['correct']==old['models'][key]['formula_agreement'][f]['correct']
            for field,kfield in(('prediction_changes','pred_changed'),('label_changes','label_changed'),('both_correct','both_correct')):assert pair_stats(query)[field]==old['models'][key]['query_flip']['all'][field]
        models[key]=dict(raw_sha256=sha(path),rows=512,formula_agreement=agreements,candidate_set_coverage=coverage,outside_candidate_rows=[r['row']for r in rows if r['pred']not in(r['logical0'],r['logical1'])],query_flip_all=pair_stats(query),query_flip_non_degenerate=pair_stats(qdiff),query_flip_same_target=pair_stats(qsame),write_flip_all=pair_stats(write),write_flip_non_degenerate=pair_stats([x for x in write if x['query_nondegenerate']]),noise_groups={str(g):dict(correct=sum(r['pred']==r['target']for r in rows if r['group']==g),rows=256,coverage=sum(r['pred']in(r['logical0'],r['logical1'])for r in rows if r['group']==g),query_flip=pair_stats([x for x in query if x['group']==g]),write_flip=pair_stats([x for x in write if x['group']==g]))for g in range(2)},per_value_pair=[dict(group=g,v0=a,v1=b,query_label_invariant=(2*(a-b))%8==0,pred_query_major_write_minor=[pred[lookup[g,a,b,o,w]]for o in range(2)for w in range(2)],target_query_major_write_minor=[base[lookup[g,a,b,o,w]]['target']for o in range(2)for w in range(2)])for g in range(2)for a in range(8)for b in range(8)],query_pairs=query,write_pairs=write)
    result=dict(status='VERIFIED_INDEPENDENT_SAVED_BINDING_ARITHMETIC',source_sha256=sha(Path(__file__)),original_VALID_sha256=sha(publicpath),privileged_counterfactual_truth_sha256=sha(truthpath),prior_binding_JSON_sha256=sha(ROOT/'actual-posterior-valid-binding-symmetry-v1.json'),prior_per_row_pred_and_formula_fields_exactly_verified=True,balanced_conditional_grid_complete=True,order_blind_ceiling=dict(equal_target_value_pairs=16,value_pairs=64,zero_difference_pairs=8,difference_four_pairs=8,conditional_fraction=str(ceiling),correct_count=320,rows=512,derivation='y0=y1 iff2(v0-v1)=0mod8; ignoringqueryselector conditionalotherfeatures gets2/2 on16pairs,1/2 on48pairs; 5/8 is conditionalambiguitybound, scorealone notlearnedmechanismproof'),models=models,cost_scope=dict(new_NN_queries=0,SGD_updates=0,no_SSH_or_checkpointdeserialization=True,analysis_wall_seconds=time.perf_counter()-started,analysis_CPU_seconds=time.process_time()-CPU),limits=['singledevseed/finite512VALIDgrid; noPrimary orarchitectureadvantage','two-candidatecoverage notproofindividualoperandsdecoded','queryresponse notzero: RAWnotonefixedlogical/temporalformula','truth privilegedanalysis only, neverlearnerfeedback'])
    output=ROOT/'saved-binding-independent-recheck-plus-shared-address-v1.json'
    with output.open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(output.name,sha(output))
    for key,m in models.items():print(key,'correct',m['formula_agreement']['target']['correct'],'coverage',m['candidate_set_coverage'],'query_all',m['query_flip_all'],'query_non',m['query_flip_non_degenerate'],'write',m['write_flip_all'],'formulas',{k:v['correct']for k,v in m['formula_agreement'].items()})
if __name__=='__main__':main()
