"""Discarded CPU-only OWN typed-stage2 resume/growth/paired-gradient check.

No optimizer updates, graph capture, task regeneration, route labels or TEST.
Saved CUDA RNG bytes roundtrip as opaque data; CUDA restoration is not executed.
CPU-grown checkpoints must never replace the original GPU research parent.
"""
import time
MODULE_WALL_STARTED=time.perf_counter();MODULE_CPU_STARTED=time.process_time()
import argparse,copy,hashlib,json,os,random,resource,traceback
from pathlib import Path
import numpy as np
import torch
import continuous_primitive_block as core
import typed_memory_actions_masked_graph_fb as fb
import owned_parent_driver as old
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage

SCHEMA='typed-paired-tail-resume-CPU-v1'
STATUS='CPU_TYPED_PAIRED_TAIL_RESUME_PASS'
PARENT_RUN='parameter-typed-memory-actions-n8-seed0-fourstage-graph-v2'
PARENT_SHA='c5a3cbe14ebc1e592a0f0e462f6bd07062870c8551f345a2b0c64bf92a3a908e'
PARENT_BYTES=525491
PAIR_SHA='9a0b33d7a7a10dd03ecdec8dd3c45b45cdf85247bbe185b8161dc2bcf650fecf'
PAIR_RESULT_SHA='ebc7cab85d73afb6048dab116ef22dbfa63ed29dfacbac526b71314755d69b42'
REPLAY_SHA='440344f6cd778e8443081cdeafa95847bf0d23aa825f97590c55436b02a7f470'
REPLAY_DRAW_SHA='c00780e9f7d57f4c26e3e472da8563b2aaab65d00926aaa55b0a4581fbfa9ab1'
COMMON={
    'continuous_primitive_block.py':'efabc8cbd2754a4a1d60f52b907cb5842f11fcf716c52646f8f935db5170d3f2',
    'typed_memory_actions_masked_graph_fb.py':'3216c90f3fddd324ae81072f01c9ab858ec09ba8d32c2594dd4869818864c321',
    'owned_fcba_core.py':'fcba2e82defe2e3fdb9b02efe1eee37456f33defa172c7eb632ea07774b88158',
    'owned_parent_driver.py':'cc1b8b2a2102050b071376bd26329bbb3ead8d5a540cff4e79a7e73cfaf5cccc',
    'context_replay128.py':'cbe370360f0bf52efc82d614db54551f000c9f7e44230dc358dfdb76bcbad140',
    'memory_live_stage_train.py':'dd5fc0b869d216f29c6ccb7202a1d965086c3fedb060d3c4c3bc4e5620042ad6'}
PARENT_SOURCES=dict(COMMON,**{
    'train_typed_memory_actions_fourstage_graph.py':'16379dda778c0a97f194b238685ee80955d73bd44cfedf71dd36a7ed2b62adb3',
    'typed_memory_actions_graph_engine.py':'99acb12d2661f9bf8a346353c62da2f701ea8599aacf88c8bfb868d9225bced6'})
TRAIN_SHA={
    '0:train':'41075db49ec66e1579421852f71e5cf06dc5b3e1259765ef42abfe26103df62f',
    '1:train':'7edab82e7879fab9eb8e31cdabcd3e9dafaf12930d03bb8453ac168f9aba5d3b',
    '2:train':'27bfaa326de195b551fbe7e726e4c6b5e6a38b742560df301196c92f7f328e5e',
    '3:train':'b950377e180c06fc1ae2137c47637f1327a1a9b0d3dfebd54260cec45b037609'}
RAW=('initial','tokens','lengths','targets')
SAMPLERS=('indices','indices_same_pool_shuffled')

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def arraysha(array):return hashlib.sha256(array.tobytes(order='C')).hexdigest()
def put(path,value):
    with path.open('x')as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
def NNhash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        value=value.detach().cpu().contiguous();h.update(json.dumps([name,list(value.shape),str(value.dtype)]).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()
def CPUrng():return dict(cpu=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
def restore_CPUrng(value):torch.set_rng_state(value['cpu']);np.random.set_state(value['numpy']);random.setstate(value['python'])
def CUDApayload(value):
    h=hashlib.sha256();total=0
    for tensor in value:
        raw=tensor.detach().cpu().contiguous();h.update(str(tuple(raw.shape)).encode());h.update(str(raw.dtype).encode());h.update(raw.numpy().tobytes());total+=raw.numel()*raw.element_size()
    return dict(streams=len(value),bytes=total,sha256=h.hexdigest(),executed_restore=False)
def file_input(folder,spec,expected,expected_bytes=None):
    leaf=spec['leaf']
    if not isinstance(leaf,str)or Path(leaf).name!=leaf:raise ValueError('simple frozen input leaf required')
    path=folder/leaf
    if path.is_symlink()or not path.is_file()or path.stat().st_size>100*1024*1024 or spec['sha256']!=expected or sha(path)!=expected:raise ValueError('exact bounded OWN input SHA differs: '+leaf)
    if expected_bytes is not None and(path.stat().st_size!=expected_bytes or spec.get('bytes')!=expected_bytes):raise ValueError('exact OWN input bytes differ: '+leaf)
    return path

def read_training(inputs,files):
    if set(files)!=set(TRAIN_SHA):raise ValueError('only the four frozen TRAIN files may be supplied to CPU resume check')
    data={};receipts=[]
    for stage in range(4):
        key=str(stage)+':train';path=file_input(inputs,files[key],TRAIN_SHA[key])
        with np.load(path,allow_pickle=False)as archive:
            if set(archive.files)!=set(RAW):raise ValueError('raw terminal TRAIN arrays only')
            rows={name:archive[name].copy()for name in RAW}
        width=rows['tokens'].shape[1];mask=np.arange(width)[None]<rows['lengths'][:,None]
        if rows['tokens'].ndim!=2 or rows['tokens'].shape[0]!=4096 or any(rows[n].shape!=(4096,)for n in('initial','lengths','targets'))or any(v.dtype!=np.int64 for v in rows.values())or np.any((rows['lengths']<0)|(rows['lengths']>(48 if stage==3 else 16)))or np.any((rows['initial']<0)|(rows['initial']>=8))or np.any((rows['targets']<0)|(rows['targets']>=8))or np.any(mask&((rows['tokens']<0)|(rows['tokens']>=(16 if stage==3 else 15)))):raise ValueError('frozen TRAIN schema/range differs')
        if width>48 and np.any(rows['tokens'][:,48:]):raise ValueError('inactive zero tail required for original48 bucket')
        data[stage]=rows;receipts.append(dict(stage=stage,file=path.name,sha256=sha(path),rows=4096,array_bytes=sum(v.nbytes for v in rows.values()),targets_array_values_decoded=4096))
    return data,receipts

def read_pairs(inputs,spec):
    path=file_input(inputs,spec,PAIR_SHA);result_path=file_input(inputs,dict(leaf=spec['result_leaf'],sha256=spec['result_sha256']),PAIR_RESULT_SHA)
    result=json.loads(result_path.read_text())
    if result.get('status')!='CPU_PAIRED_TRAIN_INDICES_GENERATED_PASS'or result.get('input_sha256')!=TRAIN_SHA['3:train']or result.get('target_array_reads')!=0 or result.get('same_pool_row_exposures_exact')is not True or result['schedule']['sha256']!=PAIR_SHA:raise ValueError('actual TRAIN-only pair formation receipt differs')
    shapes=dict(groups=(2048,2),pairdraws=(600,24),pairorders=(600,24),indices=(600,64),indices_same_pool_shuffled=(600,64))
    with np.load(path,allow_pickle=False)as archive:
        if set(archive.files)!=set(shapes):raise ValueError('exact five actual pair/control arrays required')
        arrays={k:archive[k].copy()for k in shapes}
    for key,array in arrays.items():
        if array.dtype!=np.int64 or array.shape!=shapes[key]or arraysha(array)!=result['schedule']['arrays'][key]['sha256_C_payload']:raise ValueError('actual pair array payload differs: '+key)
    if not np.array_equal(np.sort(arrays['groups'].reshape(-1)),np.arange(4096)):raise ValueError('all4096 rows must remain eligible')
    rng=np.random.default_rng(np.random.SeedSequence([0,8,3,9027,4]));draws=rng.integers(0,2048,size=(600,24),dtype=np.int64);orders=rng.integers(0,2,size=(600,24),dtype=np.int64)
    if not np.array_equal(draws,arrays['pairdraws'])or not np.array_equal(orders,arrays['pairorders']):raise ValueError('fixed pair RNG draws/orders differ')
    ordered=np.take_along_axis(arrays['groups'][draws],np.stack((orders,1-orders),-1),axis=2).reshape(600,48)
    original=np.random.default_rng(np.random.SeedSequence([0,8,3,9027])).integers(0,4096,size=(600,64),dtype=np.int64)
    pool=arrays['indices'][:,:48].reshape(-1);control=np.random.default_rng(np.random.SeedSequence([0,8,3,9027,4,0])).permutation(pool).reshape(600,48)
    if not np.array_equal(arrays['indices'][:,:48],ordered)or not np.array_equal(arrays['indices_same_pool_shuffled'][:,:48],control)or not np.array_equal(np.bincount(pool,minlength=4096),np.bincount(control.reshape(-1),minlength=4096))or any(not np.array_equal(arrays[key][:,48:],original[:,48:])for key in SAMPLERS):raise ValueError('paired/samepool/original unusedtail contract differs')
    return arrays,result

def current_steps(model,optimizer,expected):
    named=dict(model.named_parameters())
    if len(named)!=20 or len(optimizer.state)!=20 or any('step'not in optimizer.state[p]or float(optimizer.state[p]['step'])!=expected for p in named.values()):raise ValueError('all20 actual inherited Adam step counters required')
    if len(optimizer.param_groups)!=1 or len(optimizer.param_groups[0]['params'])!=20 or optimizer.param_groups[0]['lr']!=.003:raise ValueError('unchanged original Adam003/20parameter group required')
    for name,p in named.items():
        state=optimizer.state[p]
        if state['exp_avg'].shape!=p.shape or state['exp_avg_sq'].shape!=p.shape:raise ValueError('Adam moment shape differs: '+name)

def compare(left,right):
    if left['grads'].keys()!=right['grads'].keys()or len(left['grads'])!=20:raise ValueError('all20 named gradients must be compared')
    if not torch.allclose(left['output'],right['output'],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL)or not torch.allclose(left['loss'],right['loss'],atol=fb.OUTPUT_ATOL,rtol=fb.OUTPUT_RTOL):raise ValueError('fixed current-NN output/NLL tolerance failed')
    metrics=dict(output_max_abs=float((left['output']-right['output']).abs().max()),loss_abs=float((left['loss']-right['loss']).abs()),gradients={})
    for name,l in left['grads'].items():
        r=right['grads'][name]
        if(l is None)!=(r is None):raise ValueError('relative gradient None pattern differs: '+name)
        if l is None:metrics['gradients'][name]=dict(both_None=True);continue
        if not bool(torch.isfinite(l).all())or not bool(torch.isfinite(r).all())or not torch.allclose(l,r,atol=fb.GRAD_ATOL,rtol=fb.GRAD_RTOL):raise ValueError('fixed current-NN gradient tolerance failed: '+name)
        metrics['gradients'][name]=dict(both_None=False,max_abs=float((l-r).abs().max()),reference_norm=float(l.norm()),allclose=True)
    return metrics

def gradient_admission(model,batch,out,leaf):
    params=list(model.named_parameters());pointers={n:p.grad.data_ptr()for n,p in params};parameter_pointers={n:p.data_ptr()for n,p in params};before=CPUrng();started=time.perf_counter();cpu=time.process_time()
    def stable():
        if pointers!={n:p.grad.data_ptr()if p.grad is not None else None for n,p in params}or parameter_pointers!={n:p.data_ptr()for n,p in params}:raise ValueError('fixed gradient/parameter storage changed during CPU admission')
    def explicit(forward):
        for _,p in params:p.grad.zero_()
        loss,output=fb.objective(model,forward,**batch);gradients=torch.autograd.grad(loss,tuple(p for _,p in params),allow_unused=True)
        with torch.no_grad():
            for(_,p),g in zip(params,gradients):
                if g is not None:p.grad.copy_(g)
        stable();return dict(output=output.detach().clone(),loss=loss.detach().clone(),grads={n:None if g is None else g.detach().clone()for(n,_),g in zip(params,gradients)})
    reference=explicit(fb.original_forward_value);candidate=explicit(fb.masked_forward_value);metrics=compare(reference,candidate)
    output,loss=fb.masked_step(model,**batch);stable()
    for n,p in params:
        if candidate['grads'][n]is None and bool(p.grad.count_nonzero()):raise ValueError('unused masked gradient buffer must remain zero: '+n)
    repeat=dict(output=output.detach().clone(),loss=loss.detach().clone(),grads={n:None if candidate['grads'][n]is None else p.grad.detach().clone()for n,p in params});repeat_metrics=compare(reference,repeat)
    if not old.nested_state_equal(before,CPUrng()):raise ValueError('0SGD current-NN F/B changed CPU RNG')
    raw=dict(reference_output=reference['output'].numpy(),candidate_output=candidate['output'].numpy(),reference_loss=reference['loss'].numpy(),candidate_loss=candidate['loss'].numpy(),helper_repeat_output=repeat['output'].numpy(),helper_repeat_loss=repeat['loss'].numpy(),gradient_names_json=np.frombuffer(json.dumps([n for n,_ in params]).encode(),dtype=np.uint8),reference_None_mask=np.asarray([reference['grads'][n]is None for n,_ in params],dtype=np.bool_),candidate_None_mask=np.asarray([candidate['grads'][n]is None for n,_ in params],dtype=np.bool_),helper_repeat_None_mask=np.asarray([repeat['grads'][n]is None for n,_ in params],dtype=np.bool_))
    for i,(n,p)in enumerate(params):
        for prefix,snapshot in(('reference',reference),('candidate',candidate),('helper_repeat',repeat)):
            value=snapshot['grads'][n];raw[f'{prefix}_grad_{i:02d}']=np.empty((0,),dtype=np.float32)if value is None else value.numpy()
    for key,value in batch.items():raw['batch_'+key]=value.detach().numpy()
    path=out/leaf
    with path.open('xb')as stream:np.savez_compressed(stream,**raw)
    return dict(metrics=metrics,actual_helper_backward_repeat=repeat_metrics,raw_file=path.name,raw_sha256=sha(path),raw_bytes=path.stat().st_size,relative_None_pattern_checked=True,minimum_gradient_quality_gate=False,all20_named_gradients_checked=True,gradient_and_parameter_pointers_stable=True,CPU_FB_calls=3,SGD_updates=0,wall_seconds=time.perf_counter()-started,CPU_seconds=time.process_time()-cpu)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1);phase='manifest'
    try:
        manifest=json.loads((args.inputs/'manifest.json').read_text())
        if manifest.get('schema')!=SCHEMA or manifest.get('source_sha256')!=COMMON:raise ValueError('exact new CPU/common6 manifest required')
        for leaf,digest in COMMON.items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen common source differs: '+leaf)
        resume=manifest['resume']
        if resume.get('parent_run')!=PARENT_RUN or resume.get('source_sha256')!=PARENT_SOURCES:raise ValueError('exact OWN typed-v2 parent source8 lineage required')
        parent_path=file_input(args.inputs,resume,PARENT_SHA,PARENT_BYTES);phase='actual_parent_load';payload=torch.load(parent_path,map_location='cpu',weights_only=False)
        if payload.get('schema')!='typed-memory-actions-fourstage-graph-training-v1'or payload.get('stage')!=2 or payload.get('step')!=600 or payload.get('stage_complete')is not True or payload.get('maintenance_complete')is not True or payload.get('architecture_policy')!=core.POLICY or payload.get('source_sha256')!=PARENT_SOURCES:raise ValueError('only exact maintained typed-v2 stage2/step600 checkpoint accepted, never finalstage3')
        saved_args=argparse.Namespace(**payload['args'])
        if saved_args.arm!='dynamic'or saved_args.states!=8 or saved_args.seed!=0 or saved_args.lr!=.003 or saved_args.block_state!='continuous'or saved_args.block_training!='soft_only':raise ValueError('same source/NN/terminalsoft configuration required')
        phase='TRAIN_and_pairs';training,training_receipts=read_training(args.inputs,manifest['files']);pairs,pair_result=read_pairs(args.inputs,manifest['pairs'])
        used={s:np.random.default_rng(np.random.SeedSequence([0,8,s,9027])).integers(0,4096,size=(600,64),dtype=np.int64)for s in range(3)}
        if arraysha(used[2])!=payload['indices_payload_sha256']:raise ValueError('parent original stage2 raw indices SHA differs')
        used={s:matrix if s==0 else matrix[:,:48].copy()for s,matrix in used.items()}
        if not np.array_equal(payload['used_current_indices'],used[2]):raise ValueError('actual committed parent stage2 indices differ')
        phase='restore_CPU_NN_Adam_RNG_replay';model=core.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint'],expected_action_policy=core.POLICY);optimizer=torch.optim.Adam(model.parameters(),lr=.003);optimizer.load_state_dict(payload['optimizer']);replay=Replay128.from_state_dict(payload['replay_state'])
        if model.parameter_count()!=10975 or len(list(model.parameters()))!=20 or model.n_tokens!=15 or not old.tensor_state_equal(model.state_dict(),payload['model_state'])or not old.nested_state_equal(optimizer.state_dict(),payload['optimizer'])or not old.nested_state_equal(replay.state_dict(),payload['replay_state']):raise ValueError('real parent NN/Adam/Replay restore differs')
        current_steps(model,optimizer,1800)
        if(replay.size,replay.capacity,replay.width,replay.seed,replay.states,replay.seen,replay.draws)!=(128,128,48,0,8,96000,19200)or replay.digest()!=REPLAY_SHA:raise ValueError('exact parent Replay128 payload/counters required')
        replay_audit=replay.validate_sources({s:training[s]for s in range(3)},used);cuda_receipt=CUDApayload(payload['rng']['cuda']);restore_CPUrng(payload['rng'])
        if any(not old.nested_state_equal(CPUrng()[k],payload['rng'][k])for k in('cpu','numpy','python')):raise ValueError('actual available CPU RNG restore differs')
        parent_theta=NNhash(model);parent_NN={n:v.detach().clone()for n,v in model.state_dict().items()};parent_Adam_by_name={n:copy.deepcopy(optimizer.state[p])for n,p in model.named_parameters()};before_growth_rng=copy.deepcopy(CPUrng());phase='one_original_storage_growth';growth=grow_storage(model,optimizer,saved_args);after_growth_rng=copy.deepcopy(CPUrng())
        if model.parameter_count()!=10991 or model.n_tokens!=16 or model.configuration()['action_policy']!=core.POLICY:raise ValueError('one original15to16 growth required')
        for name,p in model.named_parameters():
            before=parent_NN[name]
            if not torch.equal(p[:15]if name=='token_embedding.weight'else p,before):raise ValueError('old NN/prefix changed during growth: '+name)
            for key,value in parent_Adam_by_name[name].items():
                actual=optimizer.state[p][key]
                if name=='token_embedding.weight'and torch.is_tensor(value)and value.shape==before.shape:
                    if not torch.equal(actual[:15],value)or bool(actual[15:].count_nonzero()):raise ValueError('old/new Adam embedding moments differ')
                elif not old.nested_state_equal(actual,value):raise ValueError('old Adam step/moment changed: '+name+':'+key)
        current_steps(model,optimizer,1800);grown_theta=NNhash(model);grown_NN={n:v.detach().clone()for n,v in model.state_dict().items()};grown_Adam=copy.deepcopy(optimizer.state_dict());replay_before=copy.deepcopy(replay.state_dict());phase='discarded_replay_sampling'
        sampling=Replay128.from_state_dict(copy.deepcopy(replay_before));draws=sampling.sample_indices(3,600,16)
        if arraysha(draws)!=REPLAY_DRAW_SHA or not old.nested_state_equal(after_growth_rng,CPUrng())or not old.nested_state_equal(replay_before,replay.state_dict()):raise ValueError('original stage3 replay draws/global RNG/research replay changed')
        with(args.out/'original-stage3-replay-draws.npz').open('xb')as stream:np.savez_compressed(stream,buffer_indices=draws)
        fb.allocate_gradient_buffers(model);stable_ptrs={n:p.grad.data_ptr()for n,p in model.named_parameters()};admissions=[];phase='two_fixed_TRAIN_firstbatch_0SGD'
        for sampler in SAMPLERS:
            discarded=Replay128.from_state_dict(copy.deepcopy(sampling.state_dict()));mixed=discarded.mix(training[3],pairs[sampler][0,:48],draws[0]);batch,extent,active=old.batch(mixed,np.arange(64),torch.device('cpu'))
            if extent>48 or len(mixed['targets'])!=64:raise ValueError('exact current48/replay16 within originalL48 bucket required')
            padded=torch.zeros((64,48),dtype=torch.long);padded[:,:extent].copy_(batch['tokens']);batch['tokens']=padded
            audit=gradient_admission(model,batch,args.out,sampler+'-firstbatch-all20.npz');audit.update(sampler=sampler,current_first48_payload_sha256=arraysha(pairs[sampler][0,:48]),replay_first16_payload_sha256=arraysha(draws[0]),active_steps=active,padded_steps=64*48,forward_MAC_proxy_for_CPU_FB_calls=model.resource_report(64,48,active_steps=active)['proxy_forward_macs']+2*model.resource_report(64,48,active_steps=64*48)['proxy_forward_macs'],discarded_replay_draws_after_mix=discarded.draws)
            admissions.append(audit)
            if NNhash(model)!=grown_theta or not old.tensor_state_equal(model.state_dict(),grown_NN)or not old.nested_state_equal(optimizer.state_dict(),grown_Adam)or not old.nested_state_equal(replay.state_dict(),replay_before)or not old.nested_state_equal(CPUrng(),after_growth_rng)or stable_ptrs!={n:p.grad.data_ptr()for n,p in model.named_parameters()}:raise ValueError('0SGD firstbatch admission changed NN/Adam/RNG/researchReplay/storage')
        phase='actual_CPU_checkpoint_roundtrip';roundtrip=args.out/'cpu-grown-admission-roundtrip.pt';saved_rng=dict(after_growth_rng,cuda=copy.deepcopy(payload['rng']['cuda']))
        state=dict(schema='typed-paired-tail-resume-CPU-roundtrip-v1',CPU_admission_only=True,must_not_load_for_research_GPU=True,research_updates=0,optimizer_updates=0,historical_optimizer_steps=1800,parent_checkpoint_sha256=PARENT_SHA,block_checkpoint=model.checkpoint(),model_state=model.state_dict(),optimizer=optimizer.state_dict(),rng=saved_rng,replay_state=replay.state_dict())
        with roundtrip.open('xb')as stream:torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
        loaded=torch.load(roundtrip,map_location='cpu',weights_only=False);other=core.ContinuousPrimitiveBlock.from_checkpoint(loaded['block_checkpoint'],expected_action_policy=core.POLICY);other_optimizer=torch.optim.Adam(other.parameters(),lr=.003);other_optimizer.load_state_dict(loaded['optimizer']);other_replay=Replay128.from_state_dict(loaded['replay_state'])
        if not old.tensor_state_equal(model.state_dict(),other.state_dict())or not old.nested_state_equal(optimizer.state_dict(),other_optimizer.state_dict())or not old.nested_state_equal(replay.state_dict(),other_replay.state_dict())or not old.nested_state_equal(saved_rng,loaded['rng'])or not old.nested_state_equal(payload['rng']['cuda'],loaded['rng']['cuda']):raise ValueError('actual NN/Adam/CPU RNG/opaque CUDA RNG/Replay checkpoint roundtrip differs')
        restore_CPUrng(loaded['rng']);current_steps(other,other_optimizer,1800)
        if not old.nested_state_equal(CPUrng(),after_growth_rng):raise ValueError('roundtrip factories not neutralized for available CPU RNG')
        with torch.no_grad():
            if not torch.equal(model(batch['initial'],batch['tokens'],batch['lengths'],output='logits'),other(batch['initial'],batch['tokens'],batch['lengths'],output='logits')):raise ValueError('actual CPU checkpoint forward reload differs')
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result=dict(schema=SCHEMA,status=STATUS,source_sha256=COMMON,parent_source_sha256=PARENT_SOURCES,parent_run=PARENT_RUN,parent_checkpoint_sha256=PARENT_SHA,parent_checkpoint_bytes=PARENT_BYTES,parent_stage=2,parent_step=600,maintenance_complete=True,P15=10975,P16=10991,parameter_tensors=20,SGD_updates=0,research_updates=0,optimizer_updates=0,historical_optimizer_steps=1800,historical1800_reexecuted=False,CPU_only_not_CUDA_result=True,CUDA_RNG_checked=False,CUDA_RNG_executed_restore=False,CUDA_RNG_opaque_payload_roundtrip=cuda_receipt,available_CPU_RNG_restored=True,parent_NN_Adam_replay_exact=True,actual_NN_Adam_CPU_RNG_replay_reload=True,once_original_token_growth=True,old_rows_and_Adam_moments_exact=True,new_row_Adam_moments_zero=True,CPU_new_row_not_required_equal_GPU=True,parent_trained_NN_sha256=parent_theta,CPU_grown_NN_sha256=grown_theta,CPU_new_token_row_sha256=hashlib.sha256(model.token_embedding.weight[15:].detach().numpy().tobytes()).hexdigest(),CPU_RNG_before_growth_sha256=hashlib.sha256(before_growth_rng['cpu'].numpy().tobytes()).hexdigest(),CPU_RNG_after_growth_sha256=hashlib.sha256(after_growth_rng['cpu'].numpy().tobytes()).hexdigest(),growth=growth,replay_payload_sha256=REPLAY_SHA,replay_counters=dict(seen=replay.seen,draws=replay.draws,size=replay.size,width=replay.width),replay_actual_TRAIN_source_audit=replay_audit,used_stage2_indices_verified=True,replay_stage3_draws_payload_sha256=arraysha(draws),original_replay_draws_regenerated=True,research_replay_unchanged=True,pair_schedule_sha256=PAIR_SHA,pair_generation_result_sha256=PAIR_RESULT_SHA,pair_formation_actual_costs={k:pair_result[k]for k in('driver_wall_seconds','driver_CPU_seconds','CPU_peak_RSS_bytes','serialization')},pair_formation_cost_scope='shared already-completed CPU job; referenced once, not new training or6independent formation runs',both_sampler_same_pool_verified=True,unused_last16_not_Replay=True,TRAIN_files=training_receipts,VALID_or_TEST_opened=False,teacher_or_counterfactual_truth_or_route_labels_used=False,all20_gradients_checked=True,relative_None_patterns_checked=True,minimum_gradient_quality_gate=False,all20_gradient_storage_stable=True,NN_Adam_RNG_Replay_unchanged_by_FB=True,admissions=admissions,CPU_FB_calls=6,CPU_checkpoint_forward_calls=2,CPU_checkpoint_forward_examples=128,CPU_checkpoint_forward_MAC_proxy=2*model.resource_report(64,48,active_steps=int(batch['lengths'].sum()))['proxy_forward_macs'],CPU_FB_label_positions=384,CPU_FB_current_label_positions=288,CPU_FB_replay_label_positions=96,TRAIN_target_array_values_decoded=4*4096,source_fixed_tolerances=dict(output_atol=fb.OUTPUT_ATOL,output_rtol=fb.OUTPUT_RTOL,gradient_atol=fb.GRAD_ATOL,gradient_rtol=fb.GRAD_RTOL),NN_bytes=sum(p.numel()*p.element_size()for p in model.parameters()),Adam_bytes=old.tensor_bytes(optimizer.state_dict()),replay_array_bytes=sum(v.nbytes for v in replay.arrays.values()),gradient_buffer_bytes=sum(p.grad.numel()*p.grad.element_size()for p in model.parameters()),pair_archive_bytes=(args.inputs/manifest['pairs']['leaf']).stat().st_size,pair_array_payload_bytes=sum(v.nbytes for v in pairs.values()),checkpoint=dict(file=roundtrip.name,sha256=sha(roundtrip),bytes=roundtrip.stat().st_size,CPU_admission_only=True),wall_seconds=time.perf_counter()-MODULE_WALL_STARTED,CPU_seconds=time.process_time()-MODULE_CPU_STARTED,CPU_peak_RSS_bytes=int(peak)*1024,full_training_FLOPs=None,provider_effective_compute=None,scope='discarded CPU resume/growth/checkpoint and two fixed TRAIN firstbatch20gradient gates only; no7action/VJP rerun, SGD, capture, GPU restoration or capability inference; GPU tail must restore originalOWNstage2 and actualCUDA RNG independently')
        put(args.out/'result.json',result);print(json.dumps(dict(status=STATUS,P15=10975,P16=10991,research_updates=0,optimizer_updates=0,pair_schedule_sha256=PAIR_SHA),sort_keys=True))
    except BaseException as error:put(args.out/'failure.json',dict(phase=phase,error=repr(error),traceback=traceback.format_exc(),research_updates=0,optimizer_updates=0));raise

if __name__=='__main__':main()
