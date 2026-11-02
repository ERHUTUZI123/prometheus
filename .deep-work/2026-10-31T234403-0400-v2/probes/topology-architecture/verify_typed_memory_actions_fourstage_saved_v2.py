"""Stdlib verification of actual typed-memory-actionNN saved outputs/gradients/costs.
No model, Torch/NumPy, checkpointdeserialization, SSH, compilation or training.
"""
import hashlib,json,math,time,resource,sys
from pathlib import Path
from analyze_posterior_valid_slot_values import load_npz
from verify_saved_operand_role_binding import pair_stats
from fractions import Fraction
ROOT=Path(__file__).resolve().parent
NAME='parameter-typed-memory-actions-n8-seed0-fourstage-graph-v2'
OLD='parameter-posterior-value-raw-n8-seed0-fourstage-graph-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(logp,target):
    n=len(target);correct=0;nll=0.;error=0.
    for row,y in enumerate(target):
        v=logp[row*8:(row+1)*8];assert all(math.isfinite(x)for x in v);pred=max(range(8),key=v.__getitem__);correct+=pred==y;nll-=v[y];mx=max(v);error=max(error,abs(mx+math.log(sum(math.exp(x-mx)for x in v))))
    return dict(rows=n,correct=correct,accuracy=correct/n,NLL=nll/n,max_logsumexp_abs=error)

def binding_check(path,inputs):
    public=inputs['3:valid'];gen=ROOT/'runs/parameter-memory-live-stage-data-n8-seed0-v1/generated';truthpath=gen/'counterfactual-truth.npz';truth=load_npz(truthpath)['valid_values_group_order'][1];metadata=json.loads((gen/'data/metadata.json').read_text());assert sha(truthpath)==metadata['counterfactual_truth_sha256']
    raw=load_npz(path);assert raw['soft_logp'][0]==(512,8);pred=[max(range(8),key=lambda k:raw['soft_logp'][1][r*8+k])for r in range(512)];base=[];lookup={}
    for row in range(512):
        g,v0,v1,o,w=map(int,truth[row*5:row*5+5]);length=int(public['lengths'][1][row]);word=list(public['tokens'][1][row*128:row*128+length]);first,last=(v0,v1)if w==0 else(v1,v0);A=(3*v0+v1)%8;B=(3*v1+v0)%8;target=int(public['targets'][1][row]);assert target==(A if o==0 else B)and word[-4:]==[7+o,10,8-o,15]
        base.append(dict(row=row,group=g,v0=v0,v1=v1,query_order=o,write_order=w,target=target,pred=pred[row],logical0=A,logical1=B,temporal_first=(3*first+last)%8,temporal_last=(3*last+first)%8,temporal_query0_first=((3*first+last)if o==0 else(3*last+first))%8,temporal_query0_last=((3*last+first)if o==0 else(3*first+last))%8,word=word));lookup[g,v0,v1,o,w]=row
    assert len(lookup)==512 and all((g,a,b,o,w)in lookup for g in range(2)for a in range(8)for b in range(8)for o in range(2)for w in range(2));equal=sum((3*a+b)%8==(3*b+a)%8 for a in range(8)for b in range(8));assert equal==16 and Fraction(equal*2+64-equal,128)==Fraction(5,8)
    query=[];write=[]
    for r in base:
        g,a,b,o,w=[r[x]for x in('group','v0','v1','query_order','write_order')]
        if o==0:
            t=base[lookup[g,a,b,1,w]];assert r['word'][:-4]==t['word'][:-4];c0=r['pred']==r['target'];c1=t['pred']==t['target'];query.append(dict(row0=r['row'],row1=t['row'],group=g,v0=a,v1=b,write_order=w,label_changed=r['target']!=t['target'],pred_changed=r['pred']!=t['pred'],both_correct=c0 and c1,one_correct=c0!=c1))
        if w==0:
            t=base[lookup[g,a,b,o,1]];assert r['target']==t['target']and r['word'][-4:]==t['word'][-4:]and r['word'][8+a+b:-4]==t['word'][8+a+b:-4];c0=r['pred']==r['target'];c1=t['pred']==t['target'];write.append(dict(row0=r['row'],row1=t['row'],group=g,v0=a,v1=b,query_order=o,label_changed=False,pred_changed=r['pred']!=t['pred'],both_correct=c0 and c1,one_correct=c0!=c1,query_nondegenerate=(2*(a-b))%8!=0))
    formulas=('target','logical0','logical1','temporal_first','temporal_last','temporal_query0_first','temporal_query0_last');coverage=sum(r['pred']in(r['logical0'],r['logical1'])for r in base)
    return dict(raw_sha256=sha(path),privileged_counterfactual_truth_sha256=sha(truthpath),rows=512,formula_agreement={f:sum(r['pred']==r[f]for r in base)for f in formulas},candidate_set_coverage=coverage,outside_candidate_rows=[r['row']for r in base if r['pred']not in(r['logical0'],r['logical1'])],query_flip_all=pair_stats(query),query_flip_non_degenerate=pair_stats([x for x in query if x['label_changed']]),query_flip_same_target=pair_stats([x for x in query if not x['label_changed']]),write_flip_all=pair_stats(write),write_flip_non_degenerate=pair_stats([x for x in write if x['query_nondegenerate']]),noise_groups={str(g):dict(rows=256,correct=sum(r['pred']==r['target']for r in base if r['group']==g),coverage=sum(r['pred']in(r['logical0'],r['logical1'])for r in base if r['group']==g),query_flip=pair_stats([x for x in query if x['group']==g]),write_flip=pair_stats([x for x in write if x['group']==g]))for g in range(2)},per_value_pair=[dict(group=g,v0=a,v1=b,query_label_invariant=(2*(a-b))%8==0,pred_query_major_write_minor=[pred[lookup[g,a,b,o,w]]for o in range(2)for w in range(2)],target_query_major_write_minor=[base[lookup[g,a,b,o,w]]['target']for o in range(2)for w in range(2)])for g in range(2)for a in range(8)for b in range(8)],query_pairs=query,write_pairs=write,conditional_query_blind_ceiling='5/8 onexactbalancedpairs, scorealone notmechanismproof',scope='privilegedsavedarraydiagnostic only, noNNfeedback ornewNNquery; nofixedslot/logicalkey assumption')


def failed_v1_check(current_configuration,current_config,current_version):
    name='parameter-typed-memory-actions-n8-seed0-fourstage-graph-v1';folder=ROOT/'runs'/name/'learn';failure=json.loads((folder/'failure.json').read_text());rows=[json.loads(x)for x in(folder/'records.jsonl').read_text().splitlines()];initial=next(x for x in rows if x['kind']=='configuration');receipt=json.loads((ROOT/(name+'-coordinator-receipt.json')).read_text());remote=json.loads(receipt['stdout']);version=ROOT/'versions'/('code-'+name);old_config=json.loads((version/'launch-config.json').read_text())
    assert failure['actual_updates']==0 and failure['completed_updates_current_stage']==0 and failure['completed_stages']==[] and failure['phase']=='stage0_setup' and 'sha206'in failure['error'];assert not(folder/'result.json').exists();assert not any(r['kind']=='actual_update'or(r['kind']=='actual_input_opened'and r['split'].startswith('test'))for r in rows)
    assert initial['initial_theta_sha256']==current_configuration['initial_theta_sha256'];assert old_config['CPU_proof_sha256']==current_config['CPU_proof_sha256'];assert old_config['manifest']['files']==current_config['manifest']['files'];changed=[leaf for leaf,digest in old_config['source_sha256'].items()if current_config['source_sha256'][leaf]!=digest];assert changed==['typed_memory_actions_graph_engine.py']
    engine='typed_memory_actions_graph_engine.py';assert (current_version/engine).read_bytes()==(version/engine).read_bytes().replace(b'sha206',b'sha256');assert all(sha(version/leaf)==digest for leaf,digest in old_config['source_sha256'].items());assert receipt['returncode']==1 and remote['returncode']==1
    files={f.name:dict(bytes=f.stat().st_size,sha256=sha(f))for f in sorted(folder.iterdir())if f.is_file()};assert 'partial-actual-NN.pt'in files and 'stage0-L16-admission.npz'in files
    return dict(status='ACTUAL_V1_SETUP_FAILURE_PRESERVED_NOT_TRAINING_COMPLETE',actual_SGD_updates=0,initial_theta_SHA_record=initial['initial_theta_sha256'],same_v1_v2_initial_theta_records=True,initial_theta_scope='actualconfigurationSHArecords+unchangedcore/helper/driver/sourcepath, noNNdeserialization orregeneratedinitialweights byreviewer',failure_sha256=sha(folder/'failure.json'),failed_records_sha256=sha(folder/'records.jsonl'),failure_phase=failure['phase'],error=failure['error'],learn_files=files,learn_payload_bytes=sum(x['bytes']for x in files.values()),only_scientific_source_change=engine,engine_only_sha206_to_sha256_fix_verified=True,CPU_admission_reused_exact_proof=True,data_manifest_exact=True,local_coordinator_wall_seconds=receipt['wall_seconds'],remote_coordinator_wall_seconds=remote['coordinator_wall_seconds'],reaped_child_CPU_seconds=remote['reaped_child_CPU_seconds'],last_record_wall_lowerbound=max(x.get('wall',0)for x in rows),science_wall_seconds=None,suboperation_times=None,warmup_capture_admission_scope='savedadmissionfile+verifiedenginefailurepath implies3warmup+1reference+1graphGPUFB, onecapture description with0GPUFB; SOURCEPATHinferredcounts, noindividualtimerspreserved',metadata_scope_correction='failedcoordinator inheritedprobmemory word isstale; actualsamecoreusesrawOLDh slots8x32',full_training_FLOPs=None,provider_effective_compute=None)

def main():
    start=time.perf_counter();cpu=time.process_time();folder=ROOT/'runs'/NAME/'learn';result=json.loads((folder/'result.json').read_text());config=json.loads((ROOT/'versions'/('code-'+NAME)/'launch-config.json').read_text());protocol=json.loads((ROOT/(NAME+'-protocol.json')).read_text());version=ROOT/'versions'/('code-'+NAME)
    assert result['status']=='ACTUAL_FOUR_STAGE_TYPED_MEMORY_ACTIONS_GRAPH_COMPLETE'and result['actual_new_updates_this_process']==2400 and not result['parent_NN_loaded']and result['NN_unchanged_after_TEST']
    assert result['source_sha256']==config['source_sha256']==protocol['source_sha256'];sourcechecks={}
    assert result['fresh_initial_NN'] and result['historical_updates_shared']==0 and result['all20_grad_storage_stable_within_capture']
    assert protocol['parameter_tensors']==20 and protocol['logical_carry_scalars']==320
    CPUproofpath=ROOT/'runs'/config['CPU_gate_run']/'check/result.json';assert sha(CPUproofpath)==config['CPU_proof_sha256'];CPUproof=json.loads(CPUproofpath.read_text());assert CPUproof['status']=='CPU_TYPED_MEMORY_ACTIONS_GRAPH_ADMISSION_PASS'
    assert all(CPUproof[field]is True for field in protocol['CPU_required_boolean_fields'])
    assert CPUproof['source_sha256']==config['common_source_sha256'] and all(result['source_sha256'][k]==v for k,v in CPUproof['source_sha256'].items())
    for leaf,digest in result['source_sha256'].items():assert sha(version/leaf)==digest;sourcechecks[leaf]=digest
    inputs={};inputhash={}
    for key,spec in config['manifest']['files'].items():
        f=ROOT/'runs'/spec['remote_relative'];assert sha(f)==spec['sha256'];inputs[key]=load_npz(f);inputhash[key]=spec['sha256']
    assert len(inputs)==20
    rows=[json.loads(x)for x in(folder/'records.jsonl').read_text().splitlines()];configuration=next(x for x in rows if x['kind']=='configuration');assert configuration['initial_theta_sha256']==result['initial_theta_sha256'];assert configuration['resources']['parameters']==10975 and configuration['args']['block_training']=='soft_only';updates=[x for x in rows if x['kind']=='actual_update'];assert len(updates)==2400 and all(math.isfinite(x['terminal_NLL'])and math.isfinite(x['preclip_gradient'])for x in updates)
    stagechecks=[];checkpointchecks=[];sampler={}
    for s in range(4):
        r=result['reports'][s];u=[x for x in updates if x['stage']==s];assert [x['step']for x in u]==list(range(1,601));c=r['costs'];sums={key:sum(x[key]for x in u)for key in('training_wall_seconds','padded_forward_MAC_proxy','active_forward_MAC_proxy','active_token_steps','padded_token_steps')}
        assert all(sums[key]==c[key]for key in sums)and c['SGD_updates']==600
        assert [x['step']for x in r['curves']]==list(range(0,601,100))
        assert r['replay_maintenance']['seen']==38400+s*28800 and r['replay_maintenance']['bookkeeping_counters']['draws']==s*9600
        assert r['resources']['parameters']==(10975 if s<3 else 10991)
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
    # Reference-vs-CUDAgraph output, loss and ALL20 gradients in all28 raw gates.
    gradients=[];pairs=0;maxoutput=maxgrad=maxloss=0.;names=None
    files=sorted(folder.glob('*all20.npz'))+sorted(folder.glob('*admission.npz'));assert len(files)==28
    for f in files:
        d=load_npz(f);thisnames=json.loads(bytes(d['gradient_names_json'][1]).decode());assert len(thisnames)==20
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
    assert set(('address_proposal.weight','address_proposal.bias')).issubset(names) and 'initial_memory_keys'not in names
    assert pairs==321776 and len(names)==20 and not any(n.startswith(('write_query.','write_key.','read_query.','write_gate.'))for n in names)
    frozen=json.loads((folder/'ALL_TRAINING_FROZEN_BEFORE_TEST.json').read_text());lastupdate=max(i for i,x in enumerate(rows)if x['kind']=='actual_update');testopens=[i for i,x in enumerate(rows)if x['kind']=='actual_input_opened'and x['split'].startswith('test')];assert len(testopens)==12 and min(testopens)>lastupdate and frozen['actual_new_updates']==2400
    assert result['actual_graph_capture_count']==2 and [g['training_FB_calls']for g in result['graphs']]==[1800,600]
    cost=dict(P=result['resources']['parameters'],NN_bytes=result['resources']['parameter_bytes'],Adam_bytes=result['resources']['optimizer_tensor_bytes'],replay_bytes=result['resources']['replay_bytes'],training_wall_seconds=sum(r['costs']['training_wall_seconds']for r in result['reports']),training_CPU_seconds=sum(r['costs']['training_CPU_seconds']for r in result['reports']),science_wall_seconds=result['driver_wall_seconds'],science_CPU_seconds=result['driver_CPU_seconds'],padded_MAC=sum(r['costs']['padded_forward_MAC_proxy']for r in result['reports']),active_MAC=sum(r['costs']['active_forward_MAC_proxy']for r in result['reports']),active_steps=sum(r['costs']['active_token_steps']for r in result['reports']),padded_steps=sum(r['costs']['padded_token_steps']for r in result['reports']),safety_and_final_checkpoint_bytes=sum(x['bytes']for x in checkpointchecks),setup_seconds=sum(g['setup_seconds']for g in result['graphs']),warmup_FB=sum(g['warmup_FB_calls']for g in result['graphs']),reference_gates=sum(g['reference_diagnostic_FB_calls']for g in result['graphs']),graph_gates=sum(g['graph_diagnostic_FB_calls']for g in result['graphs']),validation_seconds=sum(r['costs']['evaluation_wall_seconds']for r in result['reports']),TEST_seconds=sum(x['wall_seconds']for x in result['test_costs']),inference_seconds=result['inference_total_seconds'],full_training_FLOPs=None,provider_effective_compute=None,scope='measured nested scopes retained; notadditive, forwardMAC notfullFLOPs, no architectureadvantage')
    binding=binding_check(folder/'stage3-valid-task3-predictions.npz',inputs)
    failed_v1=failed_v1_check(configuration,config,version)
    memory_snapshots=[result['resources']]+[r['resources']for r in result['reports']]
    allocated_peaks=[r['cuda_peak_allocated_bytes']for r in memory_snapshots];reserved_peaks=[r['cuda_peak_reserved_bytes']for r in memory_snapshots]
    for graph in result['graphs']:
        for snapshot in(graph['peak_before_reset'],graph['capture']['memory_before'],graph['capture']['memory_after']):
            allocated_peaks.append(snapshot['peak_allocated']);reserved_peaks.append(snapshot['peak_reserved'])
    cost.update(memory_peak_scope='maximum of final/stage resource counters and recordedgraph pre-reset/capture before-after snapshots; driver/context and exactprivatepool unknown',CUDA_recorded_peak_allocated_bytes=max(allocated_peaks),CUDA_recorded_peak_reserved_bytes=max(reserved_peaks),logical_carry_scalars=320,logical_carry_bytes=1280,cpu_peak_RSS_bytes=result['resources']['cpu_peak_RSS_bytes'],CUDA_peak_allocated_bytes=result['resources']['cuda_peak_allocated_bytes'],CUDA_peak_reserved_bytes=result['resources']['cuda_peak_reserved_bytes'],initialization_scope='NEW20tensor architecture;18retained/common names inclGRUsubset/mappedcallerheadrows+2freshproposal tensors; initialtheta/RNGnotclaimedidenticaltoolderarchitectures; independentlyseededdataarrayscomparedpayloadexact',native_execution_scope='hard7action/expert choice withsoftmemoryattention; ONEcandidatephysicallycomputedperrow evenmemoryaction; maintrainingSOFTonly, STnotused',memory_scope='raw OLDh STORE/8x32values, OLDdirectlogits8readattention, ADDRESSappliesnexttoken; noVp/probabilitymemory/keymatrix')
    receipt=json.loads((ROOT/(NAME+'-coordinator-receipt.json')).read_text());cost['local_cold_coordinator_seconds']=receipt['wall_seconds'];CPUgate=config['CPU_gate_run'];CPUreceipt=json.loads((ROOT/(CPUgate+'-coordinator-receipt.json')).read_text());cost['separate_CPU_admission_local_seconds']=CPUreceipt['wall_seconds'];cost['both_typed_attempts_local_coordinator_seconds']=receipt['wall_seconds']+failed_v1['local_coordinator_wall_seconds'];cost['both_attempts_plus_once_CPU_admission_local_seconds']=cost['both_typed_attempts_local_coordinator_seconds']+CPUreceipt['wall_seconds']
    record=dict(status='PASS_INDEPENDENT_SAVED_TYPED_MEMORY_ACTIONS_RAW_VERIFICATION',source_script_sha256=sha(Path(__file__)),reviewer_dependency_sha256={leaf:sha(ROOT/leaf)for leaf in('analyze_posterior_valid_slot_values.py','verify_saved_operand_role_binding.py')},run=NAME,CPU_actual_proof_sha256=sha(CPUproofpath),initial20_theta_SHA_record=result['initial_theta_sha256'],binding_diagnostic=binding,failed_v1_preserved=failed_v1,result_sha256=sha(folder/'result.json'),records_sha256=sha(folder/'records.jsonl'),source_hashes_rechecked=sourcechecks,all20_input_hashes_rechecked=inputhash,all2400_finite_updates=True,stages=stagechecks,checkpoints=checkpointchecks,same_four_indices_three_replay_arrays=sampler,all12_TEST_sameforward_predictions=tests,TEST_soft_native_predictions_checked=NN_predictions,max_TEST_logsumexp_abs=normalizer,all7_saved_final_phase_VALID_predictions=validchecks,early_phase_VALID_scope="stages0/1 producer summary and curve cadence only; per-example snapshots not saved",gradient_raw_gates=gradients,gradient_names=names,gradient_scalar_pairs=pairs,max_output_abs=maxoutput,max_gradient_abs=maxgrad,max_loss_abs=maxloss,freeze_before_all12TEST=True,costs=cost,method='stdlib ZIP/NPY/math/JSON/SHA only; no Torch/NumPy/model/SSH/NNcheckpointdeserialization',analysis_wall_seconds=time.perf_counter()-start,analysis_CPU_seconds=time.process_time()-cpu)
    out=ROOT/'actual-typed-memory-actions-fourstage-independent-raw-checks-v2.json'
    with out.open('x')as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
    print(out.name,sha(out));print('TEST',{k:{m:v['correct']for m,v in x['modes'].items()}for k,x in tests.items()});print('grad',pairs,maxoutput,maxgrad,'cost',cost);print('binding',{k:binding[k]for k in('candidate_set_coverage','formula_agreement','query_flip_non_degenerate','write_flip_all')})
if __name__=='__main__':main()
