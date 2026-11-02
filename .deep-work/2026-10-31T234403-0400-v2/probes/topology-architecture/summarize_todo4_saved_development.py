"""Bounded stdlib evidence inventory; no model/checkpoint loading or SSH.

Summaries/logs/checkpoint bytes and saved AR per-position vectors are inspected.
This is a development inventory, not a claim that historical versions transfer
their measurements to the latest typed architecture or to formal benchmarks.
"""
import argparse,ast,hashlib,json,math,time
from pathlib import Path
from verify_factorized_graph600_saved import npz
ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def receipt(run):
    path=ROOT/(run+'-coordinator-receipt.json')
    if not path.exists(): return dict(path=None,local_cold_wall_seconds=None,remote_wall_seconds=None)
    raw=json.loads(path.read_text()); nested=[]
    for line in raw.get('stdout','').splitlines():
        try: value=json.loads(line)
        except json.JSONDecodeError: continue
        if isinstance(value,dict):nested.append(value)
    remote=next((v.get('coordinator_wall_seconds_including_copy',v.get('coordinator_wall_seconds',v.get('wall_seconds'))) for v in nested if any(k in v for k in ['coordinator_wall_seconds_including_copy','coordinator_wall_seconds','wall_seconds'])),None)
    return dict(path=str(path),sha256=sha(path),returncode=raw['returncode'],local_cold_wall_seconds=raw.get('wall_seconds'),remote_wall_seconds=remote,
        timing_scope='local launcher clock if present; older nested wall is remote coordinator/runner clock, not SSH wall; nested times do not sum')
def record(run):
    folder=ROOT/'runs'/run/'learn'; path=folder/'summary.json'; data=json.loads(path.read_text())
    records=folder/'records.jsonl'; lines=[json.loads(x)for x in records.read_text().splitlines()]
    config=next(x for x in lines if x.get('kind')=='configuration')
    updates=[x for x in lines if x.get('kind')in('actual_update','actual_alltoken_update','actual_LM_update')]
    assert len(updates)==data['completed_updates']==data['costs']['optimizer_updates']==300
    assert [x['step']for x in updates]==list(range(1,301))
    cost=data['costs']; timekey=next((k for k in ('training_wall_seconds','wall_seconds')if k in updates[0]),None)
    if timekey: assert abs(math.fsum(x[timekey]for x in updates)-cost['training_wall_seconds'])<1e-6
    final=data['checkpoints'][-1]; checkpoint=folder/final['file']
    assert checkpoint.stat().st_size==final['bytes'] and sha(checkpoint)==final['sha256']
    source=data.get('source_sha256',{}); version=ROOT/'versions'/('code-'+run)
    if not source:
        source={x.name:sha(x)for x in version.glob('*.py')}
    for leaf,digest in source.items():
        f=version/leaf
        if f.exists(): assert sha(f)==digest;ast.parse(f.read_bytes())
    batchfile=folder/'common-batch-indices.npz'; batch=npz(batchfile)['indices'] if batchfile.exists() else None
    return dict(run=run,path=str(path),summary_sha256=sha(path),records_sha256=sha(records),status=data['status'],arm=data['arm'],
        source_sha256=source,configuration=config,model_configuration=data.get('model_configuration'),
        parameters=data['resources']['parameters'],resources=data['resources'],updates=300,costs=cost,
        science_wall_seconds=data['driver_wall_seconds'],cold=receipt(run),final=data['final'],checkpoint=final,
        batch_file_sha256=sha(batchfile)if batchfile.exists()else None,batch_payload_sha256=batch['raw_bytes_sha256']if batch else None,
        cost_scope=data.get('cost_scope'),scope=data.get('scope',data.get('metric_scope')),formal_Primary=data['formal_Primary'],
        reload_audit=data.get('reload_audit'),training_FLOPs=None,provider_effective_compute=None,
        causality_max_difference=data.get('actual_causality_max_difference'),
        graph_audit=data.get('graph_audit'))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists(): raise FileExistsError('new evidence artifact only')
    start=time.perf_counter();real=[];sampled=[];ar=[];rawchecks=[]
    for directory in sorted((ROOT/'runs').iterdir()):
        name=directory.name
        if not(directory/'learn/summary.json').exists():continue
        if name.startswith(('parameter-real-speech-','parameter-real-cifar-')): real.append(record(name))
        elif name.startswith('parameter-char-alltoken-')and '-300-'in name and 'inference'not in name:ar.append(record(name))
        elif name.startswith('parameter-char-')and 'alltoken'not in name and '-300-'in name:sampled.append(record(name))
    assert len(real)==18 and len(ar)==10 and len(sampled)==7
    for dataset in ('speech','cifar'):
        selected=[r for r in real if ('parameter-real-'+dataset+'-')in r['run']]
        assert len({r['batch_payload_sha256']for r in selected})==1
    assert len({r['batch_payload_sha256']for r in ar})==1
    for r in ar:
        folder=ROOT/'runs'/r['run']/'learn';targets64=None;endpoint=None
        for context in (64,128,256):
            for mode in ('soft','native'):
                path=folder/f'test{context}-{mode}-alltoken-vectors.npz'
                if not path.exists():continue
                z=npz(path);nll=z['nll']['values'];pred=z['predictions']['values'];targets=z['targets']['values']
                assert z['nll']['shape']==z['targets']['shape']==(512,context)
                assert len(pred)==len(targets)and all(math.isfinite(x)for x in nll)
                suffix=[i*context+j for i in range(512)for j in range(context-64,context)]
                fullNLL=math.fsum(nll)/len(nll);suffixNLL=math.fsum(nll[i]for i in suffix)/32768
                measured=dict(all_token_NLL=fullNLL,all_token_PPL=math.exp(fullNLL),all_token_accuracy=sum(a==b for a,b in zip(pred,targets))/len(targets),
                    last64_paired_NLL=suffixNLL,last64_paired_PPL=math.exp(suffixNLL),last64_paired_accuracy=sum(pred[i]==targets[i]for i in suffix)/32768)
                for key,value in measured.items():assert abs(value-r['final'][f'test{context}'][mode][key])<1e-10,(r['run'],key)
                current=[targets[i]for i in suffix]
                if context==64 and mode=='soft': targets64=current;endpoint=list(z['endpoints']['values'])
                else: assert current==targets64 and list(z['endpoints']['values'])==endpoint
                rawchecks.append(dict(run=r['run'],context=context,mode=mode,path=str(path),sha256=sha(path),recomputed=measured,rows=512,
                    full_positions=512*context,paired_suffix_positions=32768))
    data=[]
    for run in ('parameter-real-speech-data-v1','parameter-real-cifar-data-v1','parameter-char-data-v1','parameter-char-alltoken-data-v1'):
        path=ROOT/'runs'/run/'data/metadata.json';meta=json.loads(path.read_text());data.append(dict(run=run,path=str(path),sha256=sha(path),metadata=meta))
    mamba=ROOT/'runs/parameter-char-alltoken-mamba2-300-v1';stage=json.loads((mamba/'compiler-stage-total.json').read_text())
    failures=[]
    for run in ('parameter-official-mamba2-smoke-v1','parameter-official-mamba2-staged-smoke-v1','parameter-official-mamba2-staged-smoke-v2','parameter-mamba-compiler-policy-v1','parameter-mamba-compiler-policy-v2'):
        folder=ROOT/'runs'/run
        if folder.exists():failures.append(dict(run=run,cold=receipt(run),artifacts=[dict(path=str(f),sha256=sha(f),bytes=f.stat().st_size)for f in folder.rglob('*.json')if f.name!='launch-config.json']))
    result=dict(status='VERIFIED_SAVED_TODO4_DEVELOPMENT_INVENTORY',requested_report_window_UTC='2026-11-02 04:00 to 12:00',
        time_attribution_limit='Selected actual campaign run family, not a timestamp-filtered census: older artifacts have relative clocks without uniform absolute UTC start/end. Sync mtime is not execution time.',
        conclusion='No broad capability and complete-resource architecture advantage is established. These older development models do not transfer their real-data or AR results to the latest typed mechanism.',
        real_data_development=real,sampled_terminal_LM=sampled,causal_alltoken_AR=ar,AR_perposition_saved_vector_checks=rawchecks,
        data_preparation=data,official_Mamba2_compile=dict(path=str(mamba/'compiler-stage-total.json'),sha256=sha(mamba/'compiler-stage-total.json'),
            report=stage,prologue_sha256=sha(mamba/'compiler-prologue.json')),
        supporting_compiler_attempts_and_smokes_not_learning_baselines=failures,
        evidence_boundaries=['18 audio/CIFAR static first10-class development runs, public35/100 heads; not formal35/100 Class-IL.',
            'Speech logmel40x98 and CIFAR64x48 shared TRAIN-normalized features; token64 changes both interface width and parameter/call dimensions, not a pure rank test.',
            'Seven earlier sampled-terminal char LM runs supervise19200 single next-character labels; ten later AR runs supervise614400 all-token labels. Their PPL values are not pooled.',
            'AR metrics concern sampled possibly overlapping windows, not corpuswide or word PPL. Compare last64 labels at common endpoints for length effects.',
            'AR TEST saved nll/predictions/targets/endpoints recomputed; audio/CIFAR and sampled LM lack saved per-example outputs in these folders, so their scores remain frozen-summary/log evidence.',
            'Checkpoint file hashes checked without deserialization; this inventory does not rerun reload, gradients, causality or training.',
            'Training/science/cold clocks are nested scopes, not additive. Data download/prep, earlier compiler failures/smokes and graph setup remain separate paid history.',
            'No single-seed uncertainty estimate, energy, provider-effective compute or measured full training FLOPs is available; MAC proxies are not full FLOPs.',
            'Old real-data/continuous/factorized cores differ from current typed model. Latest typed model has not inherited these real-data results.'],
        verification_wall_seconds=time.perf_counter()-start,NN_runs_by_verifier=0,checkpoint_deserializations=0)
    with args.out.open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(out=str(args.out),sha256=sha(args.out),real_runs=len(real),sampled_runs=len(sampled),AR_runs=len(ar),AR_raw_vectors=len(rawchecks),seconds=result['verification_wall_seconds'])))
if __name__=='__main__':main()
