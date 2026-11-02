"""Stdlib verification of actual shared-addressNN saved outputs/gradients/costs.
No model, Torch/NumPy, checkpointdeserialization, SSH, compilation or training.
"""
import hashlib,json,math,time,resource,sys
from pathlib import Path
from analyze_posterior_valid_slot_values import load_npz
ROOT=Path(__file__).resolve().parent
NAME='parameter-input-history-shared-address-n8-seed0-fourstage-graph-v1'
OLD='parameter-posterior-value-raw-n8-seed0-fourstage-graph-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(logp,target):
    n=len(target);correct=0;nll=0.;error=0.
    for row,y in enumerate(target):
        v=logp[row*8:(row+1)*8];assert all(math.isfinite(x)for x in v);pred=max(range(8),key=v.__getitem__);correct+=pred==y;nll-=v[y];mx=max(v);error=max(error,abs(mx+math.log(sum(math.exp(x-mx)for x in v))))
    return dict(rows=n,correct=correct,accuracy=correct/n,NLL=nll/n,max_logsumexp_abs=error)
def main():
    start=time.perf_counter();cpu=time.process_time();folder=ROOT/'runs'/NAME/'learn';result=json.loads((folder/'result.json').read_text());config=json.loads((ROOT/'versions'/('code-'+NAME)/'launch-config.json').read_text());protocol=json.loads((ROOT/(NAME+'-protocol.json')).read_text());version=ROOT/'versions'/('code-'+NAME)
    assert result['status']=='ACTUAL_FOUR_STAGE_INPUT_HISTORY_SHARED_ADDRESS_GRAPH_COMPLETE'and result['actual_new_updates_this_process']==2400 and not result['parent_NN_loaded']and result['NN_unchanged_after_TEST']
    assert result['source_sha256']==config['source_sha256']==protocol['source_sha256'];sourcechecks={}
    for leaf,digest in result['source_sha256'].items():assert sha(version/leaf)==digest;sourcechecks[leaf]=digest
    inputs={};inputhash={}
    for key,spec in config['manifest']['files'].items():
        f=ROOT/'runs'/spec['remote_relative'];assert sha(f)==spec['sha256'];inputs[key]=load_npz(f);inputhash[key]=spec['sha256']
    assert len(inputs)==20
    rows=[json.loads(x)for x in(folder/'records.jsonl').read_text().splitlines()];updates=[x for x in rows if x['kind']=='actual_update'];assert len(updates)==2400 and all(math.isfinite(x['terminal_NLL'])and math.isfinite(x['preclip_gradient'])for x in updates)
    stagechecks=[];checkpointchecks=[];sampler={}
    for s in range(4):
        r=result['reports'][s];u=[x for x in updates if x['stage']==s];assert [x['step']for x in u]==list(range(1,601));c=r['costs'];sums={key:sum(x[key]for x in u)for key in('training_wall_seconds','padded_forward_MAC_proxy','active_forward_MAC_proxy','active_token_steps','padded_token_steps')}
        assert all(sums[key]==c[key]for key in sums)and c['SGD_updates']==600
        assert [x['step']for x in r['curves']]==list(range(0,601,100))
        assert r['replay_maintenance']['seen']==38400+s*28800 and r['replay_maintenance']['bookkeeping_counters']['draws']==s*9600
        assert r['resources']['parameters']==(10789 if s<3 else 10805)
        for label in('checkpoint','premaintenance_safety_checkpoint'):
            cp=r[label];f=folder/cp['file'];assert f.stat().st_size==cp['bytes']and sha(f)==cp['sha256'];checkpointchecks.append(dict(stage=s,kind=label,**cp))
        assert abs(c['checkpoint_wall_seconds']-r['checkpoint']['write_wall_seconds']-r['premaintenance_safety_checkpoint']['write_wall_seconds'])<1e-12
        for leaf in([f'stage{s}-common-indices.npz']+([f'stage{s}-replay-draws.npz']if s else[])):
            a=load_npz(folder/leaf);b=load_npz(ROOT/'runs'/OLD/'learn'/leaf);assert a.keys()==b.keys()and all(a[k][0]==b[k][0]and a[k][1]==b[k][1]for k in a);sampler[leaf]=dict(sha256=sha(folder/leaf),same_original_RAW_NPY_header_payload=True)
        stagechecks.append(dict(stage=s,sums=sums,resources=r['resources'],reload=r['reload'],current_label_reads=c['current_training_label_reads'],replay_label_reads=c['old_replay_training_label_reads']))
    # Every saved same-forward TEST row is checked against its original data.
    tests={};normalizer=0.;NN_predictions=0
    for s in range(4):
        for split in('test_iid','test_long','test_unseen'):
            f=folder/f'final-task{s}-{split}-predictions.npz';d=load_npz(f);original=inputs[f'{s}:{split}'];n=len(d['targets'][1]);assert list(d['row_index'][1])==list(range(n))
            for key in('targets','initial','lengths'):assert d[key][1]==original[key][1]
            modes={}
            for mode in('soft','native'):
                m=metrics(d[mode+'_logp'][1],d['targets'][1]);expected=result['tests'][str(s)][split][mode];assert m['accuracy']==expected['accuracy']and abs(m['NLL']-expected['terminal_NLL'])<2e-6 and m['max_logsumexp_abs']<2e-6;modes[mode]=m;normalizer=max(normalizer,m['max_logsumexp_abs']);NN_predictions+=n
            tests[f'{s}:{split}']=dict(raw_sha256=sha(f),rows=n,modes=modes)
    assert len(tests)==12 and NN_predictions==12288
    validchecks={}
    for s in (2,3):
        for task in range(s+1):
            f=folder/f'stage{s}-valid-task{task}-predictions.npz';d=load_npz(f);original=inputs[f'{task}:valid'];assert d['targets'][1]==original['targets'][1]and d['initial'][1]==original['initial'][1]and d['lengths'][1]==original['lengths'][1]
            m=metrics(d['soft_logp'][1],d['targets'][1]);expected=result['reports'][s]['curves'][-1]['valid'][str(task)]['soft'];assert m['accuracy']==expected['accuracy']and abs(m['NLL']-expected['terminal_NLL'])<2e-6;validchecks[f'stage{s}:task{task}']=dict(raw_sha256=sha(f),**m)
    # Reference-vs-CUDAgraph output, loss and ALL23 gradients in all28 raw gates.
    gradients=[];pairs=0;maxoutput=maxgrad=maxloss=0.;names=None
    files=sorted(folder.glob('*all23.npz'))+sorted(folder.glob('*admission.npz'));assert len(files)==28
    for f in files:
        d=load_npz(f);thisnames=json.loads(bytes(d['gradient_names_json'][1]).decode());assert len(thisnames)==23
        if names is None:names=thisnames
        assert thisnames==names;checked=0;report={}
        for key,(shape,left)in d.items():
            if not key.startswith('reference_'):continue
            right=d['candidate_'+key[len('reference_'):]];assert right[0]==shape;right=right[1];atol,rtol=(2e-5,2e-4)if key.startswith('reference_grad_')else(2e-5,2e-5);maximum=0.
            for x,y in zip(left,right):assert math.isfinite(x)and math.isfinite(y)and abs(x-y)<=atol+rtol*abs(y);maximum=max(maximum,abs(x-y));checked+=1
            if key=='reference_output':maxoutput=max(maxoutput,maximum)
            elif key=='reference_loss':maxloss=max(maxloss,maximum)
            else:maxgrad=max(maxgrad,maximum)
            report[key]=dict(shape=shape,pairs=len(left),max_abs=maximum)
        pairs+=checked;gradients.append(dict(file=f.name,sha256=sha(f),scalar_pairs=checked,fields=report))
    assert pairs==316568 and len(names)==23 and not any(n.startswith(('write_query.','write_key.'))for n in names)
    frozen=json.loads((folder/'ALL_TRAINING_FROZEN_BEFORE_TEST.json').read_text());lastupdate=max(i for i,x in enumerate(rows)if x['kind']=='actual_update');testopens=[i for i,x in enumerate(rows)if x['kind']=='actual_input_opened'and x['split'].startswith('test')];assert len(testopens)==12 and min(testopens)>lastupdate and frozen['actual_new_updates']==2400
    assert result['actual_graph_capture_count']==2 and [g['training_FB_calls']for g in result['graphs']]==[1800,600]
    cost=dict(P=result['resources']['parameters'],NN_bytes=result['resources']['parameter_bytes'],Adam_bytes=result['resources']['optimizer_tensor_bytes'],replay_bytes=result['resources']['replay_bytes'],training_wall_seconds=sum(r['costs']['training_wall_seconds']for r in result['reports']),training_CPU_seconds=sum(r['costs']['training_CPU_seconds']for r in result['reports']),science_wall_seconds=result['driver_wall_seconds'],science_CPU_seconds=result['driver_CPU_seconds'],padded_MAC=sum(r['costs']['padded_forward_MAC_proxy']for r in result['reports']),active_MAC=sum(r['costs']['active_forward_MAC_proxy']for r in result['reports']),active_steps=sum(r['costs']['active_token_steps']for r in result['reports']),padded_steps=sum(r['costs']['padded_token_steps']for r in result['reports']),safety_and_final_checkpoint_bytes=sum(x['bytes']for x in checkpointchecks),setup_seconds=sum(g['setup_seconds']for g in result['graphs']),warmup_FB=sum(g['warmup_FB_calls']for g in result['graphs']),reference_gates=sum(g['reference_diagnostic_FB_calls']for g in result['graphs']),graph_gates=sum(g['graph_diagnostic_FB_calls']for g in result['graphs']),validation_seconds=sum(r['costs']['evaluation_wall_seconds']for r in result['reports']),TEST_seconds=sum(x['wall_seconds']for x in result['test_costs']),inference_seconds=result['inference_total_seconds'],full_training_FLOPs=None,provider_effective_compute=None,scope='measured nested scopes retained; notadditive, forwardMAC notfullFLOPs, no architectureadvantage')
    receipt=json.loads((ROOT/(NAME+'-coordinator-receipt.json')).read_text());cost['local_cold_coordinator_seconds']=receipt['wall_seconds'];CPUgate=config['CPU_gate_run'];CPUreceipt=json.loads((ROOT/(CPUgate+'-coordinator-receipt.json')).read_text());cost['separate_CPU_admission_local_seconds']=CPUreceipt['wall_seconds']
    record=dict(status='PASS_INDEPENDENT_SAVED_SHARED_ADDRESS_RAW_VERIFICATION',source_script_sha256=sha(Path(__file__)),run=NAME,result_sha256=sha(folder/'result.json'),records_sha256=sha(folder/'records.jsonl'),source_hashes_rechecked=sourcechecks,all20_input_hashes_rechecked=inputhash,all2400_finite_updates=True,stages=stagechecks,checkpoints=checkpointchecks,same_four_indices_three_replay_arrays=sampler,all12_TEST_sameforward_predictions=tests,TEST_soft_native_predictions_checked=NN_predictions,max_TEST_logsumexp_abs=normalizer,all7_saved_final_phase_VALID_predictions=validchecks,early_phase_VALID_scope="stages0/1 producer summary and curve cadence only; per-example snapshots not saved",gradient_raw_gates=gradients,gradient_names=names,gradient_scalar_pairs=pairs,max_output_abs=maxoutput,max_gradient_abs=maxgrad,max_loss_abs=maxloss,freeze_before_all12TEST=True,costs=cost,method='stdlib ZIP/NPY/math/JSON/SHA only; no Torch/NumPy/model/SSH/NNcheckpointdeserialization',analysis_wall_seconds=time.perf_counter()-start,analysis_CPU_seconds=time.process_time()-cpu)
    out=ROOT/'actual-shared-address-fourstage-independent-raw-checks-v1.json'
    with out.open('x')as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
    print(out.name,sha(out));print('TEST',{k:{m:v['correct']for m,v in x['modes'].items()}for k,x in tests.items()});print('grad',pairs,maxoutput,maxgrad,'cost',cost)
if __name__=='__main__':main()
