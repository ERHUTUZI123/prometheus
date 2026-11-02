"""Posthoc counterfactual TEST metrics from already saved predictions.

Parse public raw programs, SHA-bind inputs and outputs, score both endpoints.
No prefix truth file, model/checkpoint execution, Torch/NumPy, SSH or new labels.
"""
import argparse,hashlib,itertools,json,math,time
from pathlib import Path
from verify_factorized_graph600_saved import npz
from check_saved_valid_counterfactuals import aggregate
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(spec):
    path=ROOT/spec['path'];assert sha(path)==spec['sha256'],str(path)
    return path
def argmax(v):return max(range(len(v)),key=v.__getitem__)
def parse_grid(public):
    assert public['tokens']['shape']==(512,128)
    groups={};base=[];lookup={}
    for row,length in enumerate(public['lengths']['values']):
        word=list(public['tokens']['values'][row*128:row*128+length]);position=0;writes=[]
        for occurrence in range(2):
            assert word[position]in(7,8)and word[position+1:position+3]==[14,1]
            key=word[position]-7;position+=3;value=0
            while word[position]==3:position+=1;value+=1
            assert word[position]==9 and 0<=value<8;position+=1;writes.append((key,value))
        assert {k for k,v in writes}=={0,1}
        suffix=word[-4:];query=suffix[0]-7
        assert query in(0,1)and suffix==[7+query,10,8-query,15]
        noise=tuple(word[position:-4]);assert all(t in(1,2,3,4,14)for t in noise)
        initial=public['initial']['values'][row]
        group=groups.setdefault((initial,noise),len(groups))
        values=dict(writes);v0,v1=values[0],values[1];write=writes[0][0]
        expected=(3*(v0 if query==0 else v1)+(v1 if query==0 else v0))%8
        assert public['targets']['values'][row]==expected
        key=(group,v0,v1,query,write);assert key not in lookup;lookup[key]=row
        base.append(dict(row=row,group=group,v0=v0,v1=v1,query=query,write=write,target=expected,noise=list(noise),initial=initial,length=length))
    assert len(groups)==2 and len(lookup)==512
    query=[];write=[];value0=[];value1=[]
    for g,v0,v1,q,w in itertools.product(range(2),range(8),range(8),range(2),range(2)):
        a=lookup[(g,v0,v1,q,w)]
        if q==0:query.append((a,lookup[(g,v0,v1,1,w)]))
        if w==0:write.append((a,lookup[(g,v0,v1,q,1)]))
    for g,other,q,w in itertools.product(range(2),range(8),range(2),range(2)):
        for lo,hi in itertools.combinations(range(8),2):
            value0.append((lookup[(g,lo,other,q,w)],lookup[(g,hi,other,q,w)]))
            value1.append((lookup[(g,other,lo,q,w)],lookup[(g,other,hi,q,w)]))
    truth=list(public['targets']['values'])
    pairs=dict(query_all=query,query_nondegenerate=[(a,b)for a,b in query if truth[a]!=truth[b]],
        query_degenerate=[(a,b)for a,b in query if truth[a]==truth[b]],write_order_invariance=write,
        value_key0_all28_unordered=value0,value_key1_all28_unordered=value1)
    assert {k:len(v)for k,v in pairs.items()}==dict(query_all=256,query_nondegenerate=192,query_degenerate=64,write_order_invariance=256,value_key0_all28_unordered=1792,value_key1_all28_unordered=1792)
    return base,pairs,truth,[dict(group=g,initial=initial,noise=list(noise))for(initial,noise),g in groups.items()]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError('new evidence artifact only')
    tick=time.perf_counter();cpu=time.process_time();manifest=json.loads(a.manifest.read_text())
    assert manifest['schema']=='saved-fullfactorial-TEST-counterfactual-inputs-v1'
    datasets={};provenance={}
    for split,spec in manifest['data'].items():
        path=load(spec);public=npz(path);base,pairs,truth,groups=parse_grid(public)
        datasets[split]=(public,base,pairs,truth)
        provenance[split]=dict(path=str(path),sha256=sha(path),groups=groups,
            pair_counts={k:len(v)for k,v in pairs.items()},
            pair_order_sha256={k:hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()for k,v in pairs.items()})
    models=[]
    for spec in manifest['models']:
        resultpath=load(spec['result']);result=json.loads(resultpath.read_text());splits={}
        for split,predspec in spec['predictions'].items():
            path=load(predspec);z=npz(path);public,base,pairs,truth=datasets[split]
            for field in ('initial','lengths','targets'):
                if field in z:assert z[field]['values']==public[field]['values']
            if 'row_index'in z:assert list(z['row_index']['values'])==list(range(512))
            modes={};preds={}
            for mode,columns in predspec['modes'].items():
                array=z[columns['key']];values=array['values'];stats={}
                if columns['format']=='argmax':
                    assert array['shape']==(512,)and array['dtype']in('<i8','<i4')
                    pred=list(values);stats.update(NLL=None,reason_NLL_unavailable='Only frozen argmax predictions were saved.')
                elif columns['format']=='logp':
                    assert array['shape']==(512,8)and all(math.isfinite(v)for v in values)
                    pred=[argmax(values[i*8:(i+1)*8])for i in range(512)]
                    error=max(abs(math.fsum(math.exp(v)for v in values[i*8:(i+1)*8])-1)for i in range(512));assert error<2e-5
                    stats.update(NLL=math.fsum(-values[i*8+truth[i]]for i in range(512))/512,max_probability_normalization_error=error)
                else:raise ValueError('explicit argmax/logp only')
                assert all(0<=v<8 for v in pred);preds[mode]=pred
                stats.update(examples=512,correct=sum(x==y for x,y in zip(pred,truth)),accuracy=sum(x==y for x,y in zip(pred,truth))/512,
                    order_candidate_set_coverage=sum(pred[r['row']]in{(3*r['v0']+r['v1'])%8,(3*r['v1']+r['v0'])%8}for r in base),
                    counterfactuals={k:aggregate(v,pred,truth)for k,v in pairs.items()},
                    per_row_prediction_sha256=hashlib.sha256(bytes(pred)).hexdigest())
                if spec.get('score_layout')=='typed':expected=result['tests']['3'][split][mode]['accuracy']
                elif spec.get('score_layout')=='memory_live':expected=result['frozen_test_metrics']['3'][split][mode]['accuracy']
                elif spec.get('score_layout')=='original_gru32_extract':expected=result['scores'][split]['accuracy']
                else:expected=None
                if expected is not None:assert abs(stats['accuracy']-expected)<1e-12
                modes[mode]=stats
            gap=dict(available='soft'in preds and'native'in preds)
            if gap['available']:gap.update(soft_minus_native_accuracy=modes['soft']['accuracy']-modes['native']['accuracy'],argmax_disagreements=sum(x!=y for x,y in zip(preds['soft'],preds['native'])))
            else:gap['reason']='No saved native perexample predictions for this TEST/model; gap unknown, not zero.'
            splits[split]=dict(path=str(path),sha256=sha(path),modes=modes,soft_native_gap=gap)
        reports=result.get('reports',[])
        costs=result.get('costs')
        train_s=costs.get('training_wall_seconds')if costs else math.fsum(r['costs']['training_wall_seconds']for r in reports)
        coldpath=ROOT/(spec['run']+'-coordinator-receipt.json');cold=json.loads(coldpath.read_text())if coldpath.exists()else{}
        models.append(dict(id=spec['id'],run=spec['run'],lineage=spec['lineage'],P=result['resources']['parameters'],splits=splits,
            result_path=str(resultpath),result_sha256=sha(resultpath),source_sha256=result['source_sha256'],execution_scope=spec.get('execution_scope'),
            cost_scope=dict(training_wall_seconds=train_s,resources=result['resources'],costs=costs,reports=reports,
                scientific_wall_seconds=result.get('driver_wall_seconds',result.get('wall_seconds')),local_cold_wall_seconds=cold.get('wall_seconds'),
                inference=result.get('inference'),test_costs=result.get('test_costs',result.get('frozen_test_measurement_costs')),
                checkpoint=result.get('checkpoint'),history=spec.get('history'),original_training_history_result_sha256=spec.get('original_training_history_result',{}).get('sha256'),original_training_history_result=json.loads(load(spec['original_training_history_result']).read_text())if spec.get('original_training_history_result')else None,full_training_FLOPs=None,provider_effective_compute=None)))
    out=dict(status='FROZEN_TEST_COUNTERFACTUALS_STDLIB_RECOMPUTED_NO_SELECTION',manifest_sha256=sha(a.manifest),
        checker_sha256=sha(Path(__file__)),aggregate_source_sha256=sha(Path(__file__).with_name('check_saved_valid_counterfactuals.py')),
        raw_program_parser='First two KEY,CLEAR,SET0,STEP^v,WRITE blocks decoded structurally; raw middle noise and initial define template group; suffix KEY,READ,otherKEY,MIX gives query. Targets only verify parser arithmetic after grouping.',
        data=provenance,models=models,unavailable_original_models=manifest.get('unavailable_original_models',[]),
        forthcoming_unbound_models=manifest.get('forthcoming_unbound_models',[]),
        boundaries=['All paired successes require BOTH endpoint answers correct AND correct modular difference. Relation alone never counts as endpoint success.',
            'Write-order invariance includes both-correct. Query nondegenerate192 pairs are separate from64 equal-label pairs. Each operand has1792 unordered pairs across8 values.',
            'STEP counts change raw program lengths; operation template, other operand, query/write/noise/initial are held fixed, not the entire token stream.',
            'TEST outputs were already frozen. This is posthoc diagnosis and human-requested acceptance evidence; no TRAIN/config/seed/arm selection or extra NN execution.',
            'Original GRU/TF missing artifacts stay explicitly unavailable. No sampler-conditioned tail is substituted as an original architecture baseline.',
            'Resources, historical training/setup and current arm costs have different scopes. Missing FLOPs/provider-effective compute/native vectors are unknown, not zero.'],
        verification_cost=dict(wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-cpu,NN_forwards=0,SGD_updates=0,privileged_prefix_truth_files_loaded=0))
    with a.out.open('x')as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(out=str(a.out),sha256=sha(a.out),models={m['id']:{s:{mode:dict(correct=v['correct'],query_both=v['counterfactuals']['query_nondegenerate']['both_correct_AND_relation_consistent'],write_both=v['counterfactuals']['write_order_invariance']['both_correct_AND_relation_consistent'],key0_both=v['counterfactuals']['value_key0_all28_unordered']['both_correct_AND_relation_consistent'],key1_both=v['counterfactuals']['value_key1_all28_unordered']['both_correct_AND_relation_consistent'])for mode,v in d['modes'].items()}for s,d in m['splits'].items()}for m in models})))
if __name__=='__main__':main()
