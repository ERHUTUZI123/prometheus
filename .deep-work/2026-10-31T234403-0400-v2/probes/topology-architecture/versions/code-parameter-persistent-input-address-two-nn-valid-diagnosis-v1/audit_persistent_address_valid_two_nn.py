"""PRIVILEGED frozen VALID diagnosis, zeroSGD, two NEW persistent-address OWN checkpoints only.

NN phase: raw initial/tokens/lengths plus public token-row module table ONLY; exact _transition once/activeposition.
Both raw event files are saved BEFORE privilegedtruth is loaded. Teacher
is used solely for posthoc comparisons, never calls/routing/updates. This is a
source-enforced trusted audit ordering, not an OS-blindness claim.
"""
import argparse,hashlib,json,resource,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
import continuous_primitive_block as core

EVENTS=('write_first','write_second','after_read','before_mix_after_key_other','at_mix')
VALID_SHA='7d8c81c0e41bcea94785fc63593f781b6042b0d2b87844d841db2d0a1875a31b'
CARRY_SHAPES={'h':(32,),'controller':(16,),'keys':(8,16),'values':(8,8),'logp':(8,),'address':(16,)}
TRACE_SHAPES={'read_p':(8,),'write_p':(8,),'candidate_h':(32,),'read_attention':(8,),'write_addresses':(8,),'write_gate':(1,),'call_probabilities':(4,),'address_query':(16,),'address_state':(16,),'address_update_gate':(1,),'address_proposal':(16,)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor_bytes(v):
    if torch.is_tensor(v):return v.numel()*v.element_size()
    if isinstance(v,dict):return sum(tensor_bytes(x)for x in v.values())
    if isinstance(v,(list,tuple)):return sum(tensor_bytes(x)for x in v)
    return 0
def numpy_bytes(v):
    if isinstance(v,np.ndarray):return v.nbytes
    if isinstance(v,dict):return sum(numpy_bytes(x)for x in v.values())
    return 0

def public_events(tokens,lengths):
    result=np.empty((512,5),dtype=np.int64)
    for row,length in enumerate(lengths):
        word=tokens[row,:length].tolist();writes=[i for i,t in enumerate(word)if t==9]
        if len(writes)!=2 or word.count(10)!=1 or word.count(15)!=1 or word[-1]!=15:raise ValueError('fixed public VALID eventlayout required')
        result[row]=[writes[0],writes[1],word.index(10),int(length)-2,int(length)-1]
    return result

def save_token_address_table(model,spec,out):
    # All CURRENT public vocabulary rows, no labels/program/truth and no
    # sequence transition. Same frozen NN; this adds two tiny Linear queries.
    torch.cuda.synchronize();started=time.perf_counter();cpu=time.process_time();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
    with torch.no_grad():
        ids=torch.arange(model.n_tokens,device='cuda:0');embedded=model.token_embedding(ids)
        gate_logits=model.address_gate(embedded);proposal_logits=model.address_proposal(embedded)
        gates=torch.sigmoid(gate_logits);proposals=torch.tanh(proposal_logits)
    end.record();torch.cuda.synchronize();stream=begin.elapsed_time(end)/1000.
    table=dict(schema='frozen-NN-public-token-address-module-table-v1',key=spec['key'],stage=spec['stage'],checkpoint_sha256=spec['sha256'],architecture_policy=core.POLICY,token_ids=list(range(model.n_tokens)),gate_logits=gate_logits.detach().cpu().numpy().tolist(),address_gate_sigmoid=gates.detach().cpu().numpy().tolist(),address_proposal_tanh=proposals.detach().cpu().numpy().tolist(),address_gate_bias=model.address_gate.bias.detach().cpu().numpy().tolist(),source_scope='publicvocabulary rowIDs only; no semantic tokenclass/address labels/teacher; noNNparameter update',token_rows=model.n_tokens,module_Linear_calls=2,extra_sequence_forward_passes=0,SGD_updates=0,privileged_truth_loaded=False,forward_MAC_proxy=model.n_tokens*model.token_dim*(model.key_dim+1),nonlinear_elements={'sigmoid':model.n_tokens,'tanh':model.n_tokens*model.key_dim},embedding_elements_read=model.n_tokens*model.token_dim,output_logical_FP32_elements=model.n_tokens*(1+1+model.key_dim)+1,CUDA_stream_elapsed_seconds=stream,full_FLOPs=None)
    file=out/(spec['key']+'-token-address-table-before-truth.json')
    with file.open('x')as f:json.dump(table,f,indent=2,sort_keys=True);f.write('\n')
    return dict(file=file.name,sha256=sha(file),bytes=file.stat().st_size,token_rows=model.n_tokens,forward_MAC_proxy=table['forward_MAC_proxy'],module_Linear_calls=2,extra_sequence_forward_passes=0,SGD_updates=0,wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,CUDA_stream_elapsed_seconds=stream,privileged_truth_loaded=False)

def run_nn(spec,public,event_positions,inputs,out,check):
    tick=time.perf_counter();cpu=time.process_time();path=inputs/spec['leaf']
    if path.is_symlink()or path.stat().st_size!=spec['bytes']or sha(path)!=spec['sha256']:raise ValueError('exact completed OWN checkpoint required')
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('schema')!='persistent-input-address-fourstage-graph-training-v1'or payload['stage']!=spec['stage']or payload['step']!=600 or payload.get('stage_complete')is not True or payload.get('maintenance_complete')is not True or payload['architecture_policy']!=core.POLICY or payload['source_sha256']!=spec['parent_source_sha256']:raise ValueError('OWN checkpoint stage/mode/source differs')
    model=core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint'],expected_address_policy=core.POLICY).to('cuda:0')
    if model.parameter_count()!=spec['P']or not all(torch.equal(value.detach().cpu(),payload['model_state'][name])for name,value in model.state_dict().items()):raise ValueError('actual restored NN/P differs')
    saved_Adam=tensor_bytes(payload['optimizer']);saved_replay=numpy_bytes(payload['replay_state']);before={n:v.detach().cpu().clone()for n,v in model.state_dict().items()};model.eval()
    for p in model.parameters():p.requires_grad_(False)
    del payload
    # This derived public copy removes the only unseen15 for stage2. The original
    # fixed VALID file and its original128columns/lengths remain unchanged.
    tokens=public['tokens'].copy();lengths=public['lengths'].copy()
    if spec['stage']==2:
        rows=np.arange(512);last=lengths-1
        if not np.all(tokens[rows,last]==15):raise ValueError('onlyterminalMIX may be trimmed')
        tokens[rows,last]=0;lengths-=1
    active=np.arange(128)[None]<lengths[:,None]
    if np.any(active&((tokens<0)|(tokens>=model.n_tokens)))or np.any((~active)&(tokens==15)):raise ValueError('inactive15 must also be replaced with PAD0')
    derived_sha=hashlib.sha256(tokens.tobytes()+lengths.tobytes()).hexdigest();load_wall=time.perf_counter()-tick
    token_address_table=save_token_address_table(model,spec,out)
    names={**{'before_'+k:s for k,s in CARRY_SHAPES.items()},**{'after_'+k:s for k,s in CARRY_SHAPES.items()},**TRACE_SHAPES}
    arrays={name:np.full((512,5)+shape,np.nan,dtype=np.float32)for name,shape in names.items()};valid=np.zeros((512,5),dtype=np.bool_);forward_wall=0.;forward_CPU=0.;stream_seconds=0.;D2H_seconds=0.;transition_calls=0;active_transitions=0;batch_count=0
    with torch.no_grad():
        basis=model.encode_state(torch.arange(8,device='cuda:0')).detach().cpu().numpy()
        for start in range(0,512,64):
            check();rows=np.arange(start,start+64);initial=torch.as_tensor(public['initial'][rows],device='cuda:0');word=torch.as_tensor(tokens[rows],device='cuda:0');lens=torch.as_tensor(lengths[rows],device='cuda:0');events=torch.as_tensor(event_positions[rows],device='cuda:0')
            gpu={name:torch.full((64,5)+shape,float('nan'),device='cuda:0')for name,shape in names.items()};gpu_valid=torch.zeros((64,5),device='cuda:0',dtype=torch.bool)
            torch.cuda.synchronize();t=time.perf_counter();c=time.process_time();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record();carry=model.initial_carry(initial)
            if sum(v.numel()for v in carry.values())!=64*264:raise ValueError('actual carry must have264floats/example')
            for position in range(int(lengths[rows].max())):
                check();ids=(position<lens).nonzero(as_tuple=False).flatten()
                if not ids.numel():continue
                selected={k:v.index_select(0,ids)for k,v in carry.items()};x=model.token_embedding(word[:,position].index_select(0,ids))
                updated,trace=model._transition(selected,x,'soft');transition_calls+=1;active_transitions+=len(ids)
                if not torch.equal(updated['keys'],selected['keys'])or not torch.equal(trace['read_attention'],trace['write_addresses']):raise ValueError('fixedkeys/sharedattention source contract differs')
                carry={k:value.index_copy(0,ids,updated[k])for k,value in carry.items()}
                for event in range(5):
                    local=(events.index_select(0,ids)[:,event]==position).nonzero(as_tuple=False).flatten()
                    if not local.numel():continue
                    target=ids.index_select(0,local);gpu_valid[target,event]=True
                    for key in CARRY_SHAPES:gpu['before_'+key][target,event]=selected[key].index_select(0,local);gpu['after_'+key][target,event]=updated[key].index_select(0,local)
                    for key in TRACE_SHAPES:gpu[key][target,event]=(trace['posterior']if key=='write_p'else trace[key]).index_select(0,local)
            end.record();torch.cuda.synchronize();forward_wall+=time.perf_counter()-t;forward_CPU+=time.process_time()-c;stream_seconds+=begin.elapsed_time(end)/1000.;batch_count+=1
            t=time.perf_counter()
            for key,value in gpu.items():arrays[key][rows]=value.cpu().numpy()
            valid[rows]=gpu_valid.cpu().numpy();D2H_seconds+=time.perf_counter()-t
            del gpu,gpu_valid,carry,initial,word,lens,events
        if not np.all(valid[:,:4])or(spec['stage']==2 and np.any(valid[:,4]))or(spec['stage']==3 and not np.all(valid[:,4])):raise ValueError('exact publicevent capture coverage failed')
        if any(not np.isfinite(value[valid]).all()for value in arrays.values()):raise ValueError('nonfinite actualNN event state')
    raw=out/(spec['key']+'-NN-events-before-truth.npz');write=time.perf_counter()
    with raw.open('xb')as f:np.savez_compressed(f,**arrays,event_valid=valid,event_positions=event_positions,original_lengths=public['lengths'],runtime_lengths=lengths,encoder_basis=basis)
    if not all(torch.equal(v.detach().cpu(),before[n])for n,v in model.state_dict().items())or sha(path)!=spec['sha256']:raise ValueError('frozenNN/checkpoint changed')
    resources=dict(P=model.parameter_count(),NN_bytes=sum(p.numel()*p.element_size()for p in model.parameters()),checkpoint_bytes=path.stat().st_size,saved_Adam_bytes=saved_Adam,saved_replay_array_bytes=saved_replay,live_optimizer_bytes=0,runtime_carry_scalars_per_example=264,runtime_carry_bytes_per_example=1056,cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    record=dict(key=spec['key'],architecture_policy=core.POLICY,stage=spec['stage'],token_address_table=token_address_table,checkpoint_sha256=spec['sha256'],parent_source_sha256=spec['parent_source_sha256'],raw_file=raw.name,raw_sha256=sha(raw),raw_bytes=raw.stat().st_size,raw_array_payload_bytes=sum(v.nbytes for v in arrays.values()),event_counts=valid.sum(0).tolist(),NN_unchanged=True,SGD_updates=0,privileged_truth_loaded=False,logical_dataset_passes=1,batch_initializations=batch_count,exact_transition_calls=transition_calls,active_transitions=active_transitions,source_original_VALID_sha256=VALID_SHA,stage2_only_terminal_MIX_trimmed=spec['stage']==2,stage2_inactive15_replaced_PAD0=spec['stage']==2,derived_runtime_public_input_sha256=derived_sha,original_lengths_preserved=True,load_and_verify_wall_seconds=load_wall,NN_event_capture_wall_seconds=forward_wall,NN_event_capture_CPU_seconds=forward_CPU,CUDA_stream_elapsed_seconds=stream_seconds,raw_D2H_wall_seconds=D2H_seconds,raw_write_and_rehash_wall_seconds=time.perf_counter()-write,whole_NN_wall_seconds=time.perf_counter()-tick,whole_NN_CPU_seconds=time.process_time()-cpu,resources=resources,forward_MAC_proxy=model.resource_report(batch_size=512,sequence_length=int(lengths.max()),active_steps=int(lengths.sum()))['proxy_forward_macs'],full_FLOPs=None,scope='exactmanualactive-rowtransition NNpass including264carry/address-state/gate/proposal eventstores pluspublictokenmodulequery; CUDAstream notGPUbusy; prefixstate/read_p alignment onlyposthoc privileged diagnosis, notcapacityproof')
    del model,before;return record

def summarize(raw,truth):
    with np.load(raw,allow_pickle=False)as z:data={k:z[k]for k in z.files}
    if not np.array_equal(data['event_positions'],truth['event_positions']):raise ValueError('publicevents differfrom privilegedaudit indexing')
    scores={}
    for event,name in enumerate(EVENTS):
        mask=data['event_valid'][:,event];n=int(mask.sum())
        if not n:scores[name]=dict(rows=0,scope='MIX not executed by oldstage2 NN');continue
        q=truth['true_state_after_event'][mask,event,0];before_q=truth['true_state_before_event'][mask,event,0];expected_read=truth['true_selected_memory_at_query'][mask,event]
        logp=data['after_logp'][mask,event];pred=logp.argmax(-1);read=data['read_p'][mask,event];mass=read.sum(-1);read_pred=read.argmax(-1);write=data['write_p'][mask,event];conditional=mass>1e-6
        scores[name]=dict(rows=n,cached_q_correct=int((pred==q).sum()),cached_q_accuracy=float((pred==q).mean()),cached_q_NLL=float(-logp[np.arange(n),q].mean()),write_p_argmax_q_correct=int((write.argmax(-1)==q).sum()),write_p_true_q_mass_mean=float(write[np.arange(n),q].mean()),read_p_argmax_selected_memory_correct=int((read_pred==expected_read).sum()),read_p_argmax_selected_memory_accuracy=float((read_pred==expected_read).mean()),read_p_total_mass_mean=float(mass.mean()),read_p_total_mass_min=float(mass.min()),read_p_expected_selected_memory_mass_mean=float(read[np.arange(n),expected_read].mean()),read_nonzero_mass_rows=int(conditional.sum()),read_p_accuracy_given_nonzero_mass=float((read_pred[conditional]==expected_read[conditional]).mean())if conditional.any()else None,write_gate_mean=float(data['write_gate'][mask,event].mean()),true_current_q_after=q.tolist(),true_current_q_before=before_q.tolist(),true_selected_memory_at_query=expected_read.tolist(),cached_predicted_q=pred.tolist(),read_p_predicted_memory_q=read_pred.tolist(),caution='PRIVILEGEDposthocNNprefix/read-coordinate alignment, not suppliedtrainingstate or proof that everyinternaldistribution must encode taskstate')
    return scores

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--max-seconds',type=float,default=850);a=parser.parse_args()
    if not 1<=a.max_seconds<=850:raise ValueError('boundedinside850required')
    a.out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();cpu=time.process_time();records=[]
    try:
        if not torch.cuda.is_available():raise RuntimeError('sameauthorizedCUDArequired,noalternative')
        torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,0)
        manifest=json.loads((a.inputs/'manifest.json').read_text())
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('exactNN/audit source changed')
        if manifest.get('schema')!='two-owned-persistent-address-VALID-audit-v1'or len(manifest['models'])!=2 or sha(a.inputs/'valid.npz')!=VALID_SHA:raise ValueError('fixed2NEWNN/512VALIDjob required')
        # Do NOT extract originalterminaltargets in NNphase. Onlypublicfields.
        with np.load(a.inputs/'valid.npz',allow_pickle=False)as z:public={k:z[k]for k in('initial','tokens','lengths')}
        if public['tokens'].shape!=(512,128)or any(v.dtype!=np.int64 for v in public.values()):raise ValueError('exact public512programs required')
        events=public_events(public['tokens'],public['lengths'])
        def check():
            if time.perf_counter()-started>a.max_seconds:raise TimeoutError('boundedfrozenVALIDdiagnosis')
        if [(v['stage'],v['P'])for v in manifest['models']]!=[(2,10550),(3,10566)]:raise ValueError('only NEW stage2/stage3 persistent-address NN pair')
        for spec in manifest['models']:records.append(run_nn(spec,public,events,a.inputs,a.out,check))
        marker=dict(all2_NN_raw_outputs_saved=True,privileged_truth_loaded=False,NN_input_fields=['initial','tokens','lengths'],NN_transition_arguments='selectedowncarry, learnedembedding(publictoken), soft',module_table_inputs='allcurrentpublictokenrowIDs only',raw_records=records)
        with(a.out/'ALL_NN_OUTPUTS_SAVED_BEFORE_PRIVILEGED_TRUTH.json').open('x')as f:json.dump(marker,f,indent=2,sort_keys=True);f.write('\n')
        # PRIVILEGED truth begins ONLY after both NEW raw NN passes completed.
        teacher_path=a.inputs/'prefix-truth.npz'
        if sha(teacher_path)!=manifest['truth_sha256']or sha(a.inputs/'truth-result.json')!=manifest['truth_result_sha256']:raise ValueError('actualCPUprivilegedtruth changed')
        teacher_meta=json.loads((a.inputs/'truth-result.json').read_text())
        if teacher_meta['status']!='PRIVILEGED_VALID_PREFIX_TRUTH_COMPLETE'or teacher_meta['VALID_sha256']!=VALID_SHA or teacher_meta['truth_sha256']!=manifest['truth_sha256']:raise ValueError('actualcompletedprivilegedCPUtruth required')
        with np.load(teacher_path,allow_pickle=False)as z:truth={k:z[k]for k in z.files}
        scores={record['key']:summarize(a.out/record['raw_file'],truth)for record in records}
        result=dict(status='COMPLETED_TWO_PERSISTENT_ADDRESS_OWNED_NN_PRIVILEGED_VALID_DIAGNOSIS',source_sha256=manifest['source_sha256'],VALID_sha256=VALID_SHA,privileged_truth_sha256=manifest['truth_sha256'],privileged_truth_result_sha256=manifest['truth_result_sha256'],privileged_teacher_source_sha256=teacher_meta['official_oracle_sha256'],all2_NN_raw_saved_before_truthload=True,teacher_never_NN_feedback_or_routing=True,truth_exclusion_scope='trustedfixedaudit sourceordering/arguments; sameUID canreadfile, noOSblindness claim',SGD_updates=0,NN_logical_dataset_passes=2,actual_batch_initializations=sum(r['batch_initializations']for r in records),active_transitions=sum(r['active_transitions']for r in records),records=records,privileged_posthoc_scores=scores,teacher_CPU_costs=teacher_meta,wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu,cpu_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin'else 1024)),full_FLOPs=None,provider_effective_compute=None,scope='frozenVALIDprivilegedmechanismlocalization only; prefixmisalignment mayreflect generalization/interface failure, notcapacityimpossibility; noTESTselection/oldNNrerun/extraNNrepair/tuning')
        with(a.out/'result.json').open('x')as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
        print(json.dumps(dict(status=result['status'],NNs=2,SGD=0)))
    except BaseException as error:
        with(a.out/'failure.json').open('x')as f:json.dump(dict(error=repr(error),traceback=traceback.format_exc(),completed_NN_records=records),f,indent=2)
        raise
if __name__=='__main__':main()
