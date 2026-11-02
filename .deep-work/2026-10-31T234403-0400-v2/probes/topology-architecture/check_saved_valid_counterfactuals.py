"""Counterfactual metrics on frozen full-factorial VALID predictions only.

Explicit SHA-pinned manifest; stdlib NPZ decoder; no model execution, checkpoint
deserialization, Torch/NumPy, new examples, parameter change or learner feedback.
Truth is privileged posthoc analysis input, never a transition/routing input.
"""
import argparse,hashlib,itertools,json,math,time
from pathlib import Path
from verify_factorized_graph600_saved import npz
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(spec):
    path=ROOT/spec['path']
    assert sha(path)==spec['sha256'],str(path)
    return path
def argmax(values):return max(range(len(values)),key=values.__getitem__)
def aggregate(pairs,pred,truth):
    both=consistent=joint=one=changed=0
    for a,b in pairs:
        c0=pred[a]==truth[a];c1=pred[b]==truth[b]
        relation=(pred[b]-pred[a])%8==(truth[b]-truth[a])%8
        both+=c0 and c1;consistent+=relation;joint+=c0 and c1 and relation
        one+=c0!=c1;changed+=pred[a]!=pred[b]
    n=len(pairs)
    return dict(pairs=n,both_endpoints_correct=both,both_endpoints_correct_rate=both/n,
        modular_relation_consistent=consistent,modular_relation_consistent_rate=consistent/n,
        both_correct_AND_relation_consistent=joint,joint_rate=joint/n,
        exactly_one_endpoint_correct=one,prediction_changes=changed,
        target_changes=sum(truth[a]!=truth[b]for a,b in pairs))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError('new frozen evidence receipt only')
    tick=time.perf_counter();cpu=time.process_time();manifest=json.loads(args.manifest.read_text())
    assert manifest['schema']=='saved-fullfactorial-VALID-counterfactual-inputs-v1'
    valid_path=load(manifest['original_VALID']);truth_path=load(manifest['posthoc_truth'])
    public=npz(valid_path);truthraw=npz(truth_path)['valid_values_group_order']
    assert truthraw['shape']==(512,5)and public['tokens']['shape']==(512,128)
    truth=list(public['targets']['values']);base=[];lookup={}
    for row in range(512):
        noise,v0,v1,query,write=map(int,truthraw['values'][row*5:(row+1)*5])
        assert noise in (0,1)and query in (0,1)and write in (0,1)and 0<=v0<8 and 0<=v1<8
        assert truth[row]==(3*(v0 if query==0 else v1)+(v1 if query==0 else v0))%8
        length=public['lengths']['values'][row];word=list(public['tokens']['values'][row*128:row*128+length])
        assert word[-4:]==[7+query,10,8-query,15]
        key=(noise,v0,v1,query,write);assert key not in lookup;lookup[key]=row
        base.append(dict(row=row,noise=noise,v0=v0,v1=v1,query=query,write=write,target=truth[row],word=word))
    assert len(lookup)==512
    queries=[];writes=[];values0=[];values1=[]
    for noise,v0,v1,query,write in itertools.product(range(2),range(8),range(8),range(2),range(2)):
        a=lookup[(noise,v0,v1,query,write)]
        if query==0:
            b=lookup[(noise,v0,v1,1,write)]
            assert base[a]['word'][:-4]==base[b]['word'][:-4]
            queries.append((a,b))
        if write==0:
            b=lookup[(noise,v0,v1,query,1)]
            assert truth[a]==truth[b]and base[a]['word'][-4:]==base[b]['word'][-4:]
            writes.append((a,b))
    for noise,other,query,write in itertools.product(range(2),range(8),range(2),range(2)):
        for lo,hi in itertools.combinations(range(8),2):
            a=lookup[(noise,lo,other,query,write)];b=lookup[(noise,hi,other,query,write)]
            assert (truth[b]-truth[a])%8==((3 if query==0 else 1)*(hi-lo))%8
            values0.append((a,b))
            a=lookup[(noise,other,lo,query,write)];b=lookup[(noise,other,hi,query,write)]
            assert (truth[b]-truth[a])%8==((1 if query==0 else 3)*(hi-lo))%8
            values1.append((a,b))
    nondegenerate=[(a,b)for a,b in queries if truth[a]!=truth[b]]
    degenerate=[(a,b)for a,b in queries if truth[a]==truth[b]]
    assert len(queries)==len(writes)==256 and len(nondegenerate)==192 and len(degenerate)==64
    assert len(values0)==len(values1)==1792
    pairsets=dict(query_all=queries,query_nondegenerate=nondegenerate,query_degenerate=degenerate,
                  write_order_invariance=writes,value_key0_all28_unordered=values0,value_key1_all28_unordered=values1)
    pairsha={k:hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()for k,v in pairsets.items()}
    output=[]
    for spec in manifest['models']:
        path=load(spec['predictions']);resultpath=load(spec['result']);result=json.loads(resultpath.read_text());z=npz(path)
        assert z['targets']['shape']==(512,)and list(z['targets']['values'])==truth
        assert list(z['row_index']['values'])==list(range(512))
        assert list(z['initial']['values'])==list(public['initial']['values'])
        assert list(z['lengths']['values'])==list(public['lengths']['values'])
        modes={};preds={}
        for mode,key in [('soft','soft_logp'),('native','native_logp')]:
            if key not in z:
                modes[mode]=dict(available=False,reason='No saved per-example VALID output for this execution mode; no NN rerun authorized.')
                continue
            array=z[key];assert array['shape']==(512,8)
            values=array['values'];assert all(math.isfinite(v)for v in values)
            pred=[argmax(values[i*8:(i+1)*8])for i in range(512)];preds[mode]=pred
            normalized=max(abs(math.fsum(math.exp(v)for v in values[i*8:(i+1)*8])-1)for i in range(512))
            assert normalized<2e-5
            metrics=dict(available=True,examples=512,correct=sum(p==t for p,t in zip(pred,truth)),
                accuracy=sum(p==t for p,t in zip(pred,truth))/512,
                NLL=math.fsum(-values[i*8+truth[i]]for i in range(512))/512,
                max_probability_normalization_error=normalized,
                order_candidate_set_coverage=sum(pred[r['row']]in{(3*r['v0']+r['v1'])%8,(3*r['v1']+r['v0'])%8}for r in base),
                counterfactuals={name:aggregate(pairs,pred,truth)for name,pairs in pairsets.items()},
                strata={f"noise{noise}/query{q}/write{w}":dict(rows=64,
                    correct=sum(pred[r['row']]==r['target']for r in base if(r['noise'],r['query'],r['write'])==(noise,q,w)))
                    for noise,q,w in itertools.product(range(2),repeat=3)})
            assert all(m['both_endpoints_correct']==m['both_correct_AND_relation_consistent']for m in metrics['counterfactuals'].values())
            modes[mode]=metrics
        gap=dict(available='soft'in preds and'native'in preds)
        if gap['available']:
            gap.update(soft_minus_native_accuracy=modes['soft']['accuracy']-modes['native']['accuracy'],
                argmax_disagreements=sum(a!=b for a,b in zip(preds['soft'],preds['native'])),
                note='Execution-mode gap on the identical frozen NN; baseline native may alias its ordinary dense execution, not a typed hard-action policy.')
        else:gap['reason']='Saved VALID native vectors absent; unknown, not zero.'
        coldpath=ROOT/(spec['run']+'-coordinator-receipt.json');cold=json.loads(coldpath.read_text())if coldpath.exists()else{}
        output.append(dict(id=spec['id'],run=spec['run'],lineage=spec['lineage'],prediction_path=str(path),prediction_sha256=sha(path),
            result_sha256=sha(resultpath),source_sha256=result.get('source_sha256'),P=result['resources']['parameters'],modes=modes,soft_native_gap=gap,
            full_cost_scope=dict(costs=result.get('costs'),resources=result.get('resources'),setup=result.get('setup'),
                inference=result.get('inference'),test_costs=result.get('test_costs'),
                scientific_wall_seconds=result.get('driver_wall_seconds'),local_cold_wall_seconds=cold.get('wall_seconds'),
                history_shared_updates=result.get('historical_updates_shared'),new_updates=result.get('actual_new_updates_this_process'),
                checkpoints=result.get('checkpoint',result.get('checkpoints')),full_training_FLOPs=None,provider_effective_compute=None)))
    artifact=dict(status='SAVED_VALID_COUNTERFACTUALS_RECOMPUTED_NO_ARCHITECTURE_ACCEPTANCE_CLAIM',
        manifest_sha256=sha(args.manifest),checker_sha256=sha(Path(__file__)),original_VALID_sha256=sha(valid_path),posthoc_truth_sha256=sha(truth_path),
        paired_grid=dict(rows=512,noise_groups=2,values_per_operand=8,query_orders=2,write_orders=2,
            pair_counts={k:len(v)for k,v in pairsets.items()},row_pair_order_sha256=pairsha),
        models=output,unavailable_historical_models=manifest.get('unavailable_historical_models',[]),
        forthcoming_unbound_models=manifest.get('forthcoming_unbound_models',[]),
        analytical_identity=dict(query_order_blind_balanced_optimal_accuracy='5/8',
            equal_target_pairs=64,all_query_pairs=256,derivation='16/64 value pairs have2(v0-v1)=0 mod8; .25 + .75/2 = .625. Score equality alone does not identify a learned mechanism.'),
        scope=['Each success requires BOTH endpoints correct AND the modulo8 target relation. Relation consistency alone can be vacuous or coexist with a shared wrong offset.',
            'Both-correct mathematically implies the stated relation; reporting both separately makes false flip-only success visible.',
            'Value interventions hold noise/query/write/other value fixed. STEP-count encoding changes raw length; semantic operation template is fixed, not the exact token stream.',
            'Write-order invariance requires both-correct, not merely identical wrong predictions. Nondegenerate query pairs are separated from invariant-label pairs.',
            'Privileged value metadata and targets are posthoc scoring only. No new labels, NN forwards, checkpoint load, parameter changes or teacher routing occurred.',
            'Historical sampler-conditioned tails remain separately attributed; no pooling as additional seeds or as original random-tail baselines.',
            'These are descriptive diagnostics on existing single-seed VALID data, not a new confirmatory architecture-acceptance threshold or strong-baseline victory.'],
        verification_cost=dict(wall_seconds=time.perf_counter()-tick,CPU_seconds=time.process_time()-cpu,NN_forwards=0,SGD_updates=0))
    with args.out.open('x')as f:json.dump(artifact,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(out=str(args.out),sha256=sha(args.out),models={m['id']:dict(P=m['P'],accuracy=m['modes']['soft']['accuracy'],query_joint=m['modes']['soft']['counterfactuals']['query_nondegenerate']['both_correct_AND_relation_consistent'],key0_joint=m['modes']['soft']['counterfactuals']['value_key0_all28_unordered']['both_correct_AND_relation_consistent'],key1_joint=m['modes']['soft']['counterfactuals']['value_key1_all28_unordered']['both_correct_AND_relation_consistent'])for m in output})))
if __name__=='__main__':main()
