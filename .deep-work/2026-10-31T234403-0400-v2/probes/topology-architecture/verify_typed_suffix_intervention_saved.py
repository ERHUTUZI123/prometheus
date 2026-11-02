"""Independent stdlib arithmetic on six already saved suffix interventions.

No checkpoint deserialization, Torch/NumPy, model forward, SSH, or new labels.
Original VALID targets are read only for posthoc scoring of frozen raw arrays.
"""
import argparse, ast, hashlib, json, math, resource, time
from pathlib import Path
from verify_factorized_graph600_saved import npz

ROOT = Path(__file__).resolve().parent
RUN = 'parameter-typed-suffix-memory-intervention-cpu-v1'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def argmax(row):
    return max(range(len(row)), key=row.__getitem__)

def compare(left, right):
    assert len(left) == len(right)
    return dict(max_abs=max(abs(x-y) for x,y in zip(left,right)),
                allclose_2e5=all(abs(x-y)<=2e-5+2e-5*abs(y) for x,y in zip(left,right)),
                argmax_differences=sum(argmax(left[i:i+8])!=argmax(right[i:i+8])
                                       for i in range(0,len(left),8)))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): raise FileExistsError('preserve prior receipts')
    tick=time.perf_counter(); cpu=time.process_time()
    audit=ROOT/'runs'/RUN/'audit'; result_path=audit/'result.json'
    result=json.loads(result_path.read_text())
    assert result['status']=='COMPLETED_TYPED_SUFFIX_MEMORY_INTERVENTION_CPU'
    assert result['NN_forward_transitions']==12288 and result['SGD_updates']==0
    version=ROOT/'versions'/('code-'+RUN)
    for leaf,digest in result['source_sha256'].items():
        assert sha(version/leaf)==digest
        ast.parse((version/leaf).read_bytes())
    valid_path=ROOT/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated/data/stage3-valid.npz'
    assert sha(valid_path)==result['VALID_sha256']
    valid=npz(valid_path)
    assert valid['tokens']['shape']==(512,128)
    targets=valid['targets']['values']; groups={}; suffix=[]
    for row,length in enumerate(valid['lengths']['values']):
        word=list(valid['tokens']['values'][row*128:row*128+length])
        groups.setdefault((valid['initial']['values'][row],tuple(word[:-4])),[]).append(row)
        suffix.extend(word[-4:])
    members=list(groups.values())
    assert len(members)==256 and all(len(g)==2 for g in members)
    group_id=[None]*512; donor=[None]*512
    for i,rows in enumerate(members):
        for row in rows: group_id[row]=i; donor[row]=members[(i+127)%256][0]
        assert suffix[rows[0]*4:rows[0]*4+4]!=suffix[rows[1]*4:rows[1]*4+4]
        for row in rows:
            assert suffix[row*4+1]==10 and suffix[row*4+3]==15
            assert {suffix[row*4],suffix[row*4+2]}=={7,8}
    receipt=json.loads((audit/'ALL_SIX_RAW_SAVED_BEFORE_TARGETS.json').read_text())
    assert receipt['all_six_saved'] and not receipt['VALID_targets_loaded']
    assert receipt['records']==result['records']
    reports=[]; payload=compressed=0
    for model,oldrun in [('paired','paired'),('control','same-pool-shuffle')]:
        event_path=ROOT/'runs/parameter-typed-three-nn-valid-actions-cpu-v1/audit'/(model+'-NN-events-before-truth.npz')
        event=npz(event_path)
        assert list(event['original_lengths']['values'])==list(valid['lengths']['values'])
        assert all(event['event_positions']['values'][i*8+4]==valid['lengths']['values'][i]-4 for i in range(512))
        for field,width in [('h',32),('controller',16),('values',256),('address',8),('logp',8)]:
            values=event['before_'+field]['values']
            assert all(values[(a*8+4)*width:(a*8+5)*width]==values[(b*8+4)*width:(b*8+5)*width] for a,b in members)
        original_path=ROOT/f'runs/parameter-typed-paired-tail-{oldrun}-n8-seed0-600-v1/learn/stage3-valid-task3-predictions.npz'
        original=npz(original_path)
        assert list(original['targets']['values'])==list(targets)
        assert list(original['row_index']['values'])==list(range(512))
        reference=[]
        for i in range(512): reference.extend(event['after_logp']['values'][(i*8+7)*8:(i*8+8)*8])
        sham=npz(audit/(model+'-sham-raw-before-targets.npz'))
        baseline=list(sham['final_logp']['values'])
        sham_cpu=compare(baseline,reference)
        assert sham_cpu['allclose_2e5'] and sham_cpu['argmax_differences']==0
        original_cmp=compare(baseline,original['soft_logp']['values'])
        for condition in ['sham','zero_V','donor_V']:
            record=next(r for r in result['records'] if r['model']==model and r['condition']==condition)
            path=audit/record['raw_file']; z=npz(path)
            assert sha(path)==record['raw_sha256'] and path.stat().st_size==record['raw_bytes']
            assert sha(event_path)==record['entry_raw_sha256']
            assert list(z['group_id']['values'])==group_id and list(z['donor_row']['values'])==donor
            assert list(z['suffix']['values'])==suffix
            assert z['logp']['shape']==(512,4,8) and z['final_logp']['shape']==(512,8)
            logp=list(z['final_logp']['values'])
            assert all(logp[i*8:(i+1)*8]==list(z['logp']['values'][(i*4+3)*8:(i*4+4)*8]) for i in range(512))
            assert all(math.isfinite(x) for a in z.values() if a['dtype'] in ('<f4','<f8') for x in a['values'])
            pred=[argmax(logp[i*8:(i+1)*8]) for i in range(512)]
            basepred=[argmax(baseline[i*8:(i+1)*8]) for i in range(512)]
            correct=sum(a==b for a,b in zip(pred,targets))
            score=dict(correct=correct,examples=512,accuracy=correct/512,
                NLL=math.fsum(-logp[i*8+targets[i]] for i in range(512))/512,
                candidate_coverage=sum(pred[row] in {targets[a],targets[b]} for a,b in members for row in (a,b)),
                query_changed_target_pairs=sum(targets[a]!=targets[b] for a,b in members),
                queryflip_disagreements=sum(pred[a]!=pred[b] for a,b in members if targets[a]!=targets[b]),
                queryflip_both_correct=sum(pred[a]==targets[a] and pred[b]==targets[b] for a,b in members if targets[a]!=targets[b]),
                queryflip_one_correct=sum((pred[a]==targets[a])+(pred[b]==targets[b])==1 for a,b in members if targets[a]!=targets[b]),
                prediction_changes_vs_sham=sum(a!=b for a,b in zip(pred,basepred)),
                probability_TV_vs_sham_mean=math.fsum(abs(math.exp(x)-math.exp(y)) for x,y in zip(logp,baseline))/1024,
                logp_max_abs_vs_sham=max(abs(x-y) for x,y in zip(logp,baseline)),
                maximum_probability_normalization_error=max(abs(math.fsum(math.exp(x) for x in logp[i*8:(i+1)*8])-1) for i in range(512)))
            expected=result['scores'][model+'-'+condition]
            for key,value in expected.items():
                assert abs(score[key]-value)<(3e-7 if key in ('NLL','probability_TV_vs_sham_mean') else 1e-12),(key,score[key],value)
            for field,width in [('action_probabilities',7),('read_attention',8)]:
                values=z[field]['values']
                assert min(values)>=0
                score[field+'_normalization_max_abs']=max(abs(math.fsum(values[i:i+width])-1) for i in range(0,len(values),width))
                assert score[field+'_normalization_max_abs']<1e-6
            size=sum(len(a['values'])*a['values'].itemsize for a in z.values())
            assert size==record['raw_array_payload_bytes']==868352
            payload+=size; compressed+=path.stat().st_size
            reports.append(dict(model=model,condition=condition,raw_sha256=sha(path),raw_bytes=path.stat().st_size,
                scores=score,sham_vs_prior_CPU=sham_cpu,sham_vs_original_GPU_VALID=original_cmp,
                original_GPU_VALID_sha256=sha(original_path),entry_raw_sha256=sha(event_path)))
    coordinator=ROOT/(RUN+'-coordinator-receipt.json')
    rc=json.loads(coordinator.read_text()); assert rc['returncode']==0
    out=dict(status='PASS_SIX_FROZEN_SUFFIX_INTERVENTIONS_STDLIB_RECOMPUTED',
        result_path=str(result_path),result_sha256=sha(result_path),source_sha256=result['source_sha256'],
        original_VALID_sha256=sha(valid_path),raw_record_receipt_sha256=sha(audit/'ALL_SIX_RAW_SAVED_BEFORE_TARGETS.json'),
        checks=dict(six_raw_file_hashes=True,all_saved_final_logp_recomputed=True,input_only_256_prefixpairs=True,
            same_group_plus127_firstrow_donor_for_both_queries=True,identical_entry_carry_within_querypair=True,
            cached_sham_allclose_and_argmax_match_prior_CPU=True,all_producer_score_arithmetic_matches=True),
        rows=reports,cost_scope=dict(original_NN_forward_transitions=12288,original_SGD_updates=0,
            original_science_wall_seconds=result['wall_seconds'],original_CPU_seconds=result['CPU_seconds'],
            original_CPU_peak_RSS_bytes=result['CPU_peak_RSS_bytes'],checkpoint_bytes_loaded=result['checkpoint_bytes_loaded'],
            live_optimizer_bytes=0,raw_compressed_bytes=compressed,raw_array_payload_bytes=payload,
            coordinator_wall_seconds=rc.get('wall_seconds'),coordinator_receipt_sha256=sha(coordinator),
            full_FLOPs=None,provider_effective_compute=None),
        scope=['Only saved numeric arrays and source AST/hash were inspected. No checkpoint load or NN execution.',
            'Source orders all six raw writes before opening targets; metadata/order checks do not prove an OS-enforced truth boundary.',
            'Zero V is out of distribution and donor V is context mismatched. This tests frozen suffix sensitivity, not learning benefit or global memory irrelevance.',
            'Raw signed h/V are not category probabilities. No raw-state decoding or teacher routing was introduced.'],
        verification_wall_seconds=time.perf_counter()-tick,verification_CPU_seconds=time.process_time()-cpu,
        verification_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if __import__('sys').platform=='darwin' else 1024))
    with args.out.open('x') as stream:json.dump(out,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps(dict(status=out['status'],out=str(args.out),sha256=sha(args.out),rows=[(r['model'],r['condition'],r['scores']['correct'])for r in reports],cost_scope=out['cost_scope'])))

if __name__=='__main__': main()
