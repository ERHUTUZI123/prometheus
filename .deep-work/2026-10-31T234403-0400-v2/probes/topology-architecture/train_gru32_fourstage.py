"""One fresh P-matched ordinary GRU32 baseline; four600-step terminal-NLL stages.

Exact unmodified cc1 TerminalGRU, width32/layers2 chosen by P only, never score.
Original data/schedules/Replay128/Adam.003/clip1/seed0. No auxiliary objective,
resume, extra curriculum, teacher or automatic retry. TEST stays unopened until
all2400 updates freeze. Every stage writes NN/Adam/RNG/replay BEFORE maintenance.
Only posttraining replay storage may view verified all-zero128 padding as48.
"""
import argparse,copy,hashlib,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import owned_parent_driver as old
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage

RAW=('initial','tokens','lengths','targets')
SCHEMA='terminal-gru32-fourstage-training-v1'
PARENT_SHA='cc1b8b2a2102050b071376bd26329bbb3ead8d5a540cff4e79a7e73cfaf5cccc'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def parameters(model):return sum(v.numel()for v in model.parameters())
def rng(device):
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all()if device.type=='cuda'else[],numpy=np.random.get_state(),python=random.getstate())
def restore_rng(state,device):
    torch.set_rng_state(state['cpu'])
    if device.type=='cuda':torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy']);random.setstate(state['python'])
def maintenance_view(data,width):
    """Preserve full training matrix/lengths; crop only an all-zero storage tail."""
    if data['tokens'].shape[1]<=width:return data,dict(cropped=False,extra_copy=False)
    if np.any(data['lengths']>width)or np.any(data['tokens'][:,width:]):raise ValueError('cannot discard active or nonzero replay bytes')
    result=dict(data,tokens=data['tokens'][:,:width])
    if not np.shares_memory(result['tokens'],data['tokens']):raise AssertionError('replay padding crop must be a zero-copy view')
    return result,dict(cropped=True,original_columns=data['tokens'].shape[1],stored_columns=width,all_removed_entries_zero=True,lengths_unchanged=True,full_training_matrix_unchanged=True,extra_copy=False)
def save_state(rec,leaf,model,opt,replay,device,stage,complete,used,pending,costs,sources):
    began=time.perf_counter();path=rec.out/leaf
    payload=dict(schema=SCHEMA,stage=stage,step=600,neural_stage_training_complete=True,stage_complete=complete,maintenance_complete=complete,
        model_spec=copy.deepcopy(model.spec),model_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},optimizer=opt.state_dict(),
        rng=rng(device),replay_state=replay.state_dict(),committed_used_indices={s:x.copy()for s,x in used.items()},pending_current_used_indices=None if pending is None else pending.copy(),
        source_sha256=sources,costs=copy.deepcopy(costs),formal_Primary=False,fixed_policy=dict(seed=0,steps_per_stage=600,lr=.003,clip=1.,width=32,layers=2))
    with path.open('xb')as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    receipt=dict(file=leaf,sha256=sha(path),bytes=path.stat().st_size,write_wall_seconds=time.perf_counter()-began,stage_complete=complete,NN600steps_preserved=True)
    rec.emit('actual_checkpoint',stage=stage,**receipt);return path,receipt

def verify_reload(path,model,opt,replay,device,valid,args,train_sources):
    began=time.perf_counter();saved_rng=rng(device);was=model.training
    try:
        loaded=torch.load(path,map_location='cpu',weights_only=False)
        if loaded.get('schema')!=SCHEMA or loaded.get('step')!=600:raise ValueError('owned GRU32 checkpoint required')
        other=old.TerminalGRU(**loaded['model_spec']).to(device);other.load_state_dict(loaded['model_state'],strict=True)
        other_opt=torch.optim.Adam(other.parameters(),lr=.003);other_opt.load_state_dict(loaded['optimizer'])
        replay_copy=Replay128.from_state_dict(loaded['replay_state'])
        if not old.tensor_state_equal(model.state_dict(),other.state_dict())or not old.nested_state_equal(opt.state_dict(),other_opt.state_dict())or not old.nested_state_equal(replay.state_dict(),replay_copy.state_dict()):raise RuntimeError('actual NN/Adam/replay reload mismatch')
        restore_rng(loaded['rng'],device)
        if not old.nested_state_equal(rng(device),loaded['rng']):raise RuntimeError('actual RNG reload mismatch')
        source_audit=replay_copy.validate_sources(train_sources,loaded['committed_used_indices'])
        b,_,_=old.batch(valid,np.arange(min(16,len(valid['initial']))),device);model.eval();other.eval()
        with torch.no_grad():
            if not torch.equal(old.forward(model,b,args,'soft'),old.forward(other,b,args,'soft')):raise RuntimeError('actual GRU32 reload forward mismatch')
        return dict(exact_NN_Adam_RNG_replay_forward=True,RNG_factory_neutral=True,CUDA_RNG_checked=device.type=='cuda',source_audit=source_audit,wall_seconds=time.perf_counter()-began)
    finally:model.train(was);restore_rng(saved_rng,device)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--max-seconds',type=float,default=850)
    a=p.parse_args()
    if not 1<=a.max_seconds<=850:p.error('fixed whole scientific <=850s')
    rec=old.Recorder(a.out,a.max_seconds);device=torch.device('cuda:0');model=opt=None;reports=[];stage=-1;completed=0;safe_checkpoint=None
    try:
        if not torch.cuda.is_available():raise RuntimeError('explicit CUDA required, no fallback')
        torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,device);torch.manual_seed(0);np.random.seed(0);random.seed(0)
        manifest=json.loads((a.inputs/'manifest.json').read_text())
        if manifest['schema']!='terminal-gru32-fourstage-inputs-v1':raise ValueError('fixed GRU32 input schema required')
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen source differs: '+leaf)
        if sha(Path(__file__).with_name('owned_parent_driver.py'))!=PARENT_SHA:raise ValueError('unmodified original TerminalGRU required')
        args=argparse.Namespace(arm='gru',states=8,seed=0,lr=.003,baseline_width=32,replay_capacity=128,replay_per_batch=16,stages=4,steps=600)
        def load(s,split):
            entry=manifest['files'][f'{s}:{split}'];path=a.inputs/entry['leaf']
            if sha(path)!=entry['sha256']:raise ValueError('frozen original data differs')
            with np.load(path,allow_pickle=False)as z:
                if set(z.files)!=set(RAW):raise ValueError('raw terminal fields only')
                data={k:z[k]for k in RAW}
            n=len(data['initial']);tokens=data['tokens'];lengths=data['lengths']
            if any(v.dtype!=np.int64 for v in data.values())or tokens.ndim!=2 or tokens.shape[0]!=n or any(data[k].shape!=(n,)for k in('initial','lengths','targets'))or np.any((lengths<0)|(lengths>tokens.shape[1])):raise ValueError('raw integer shape/lengths differ')
            active=np.arange(tokens.shape[1])[None]<lengths[:,None]
            if np.any((data['initial']<0)|(data['initial']>=8))or np.any((data['targets']<0)|(data['targets']>=8))or np.any(active&((tokens<0)|(tokens>=(16 if s==3 else 15)))):raise ValueError('public ranges differ')
            rec.emit('actual_input_opened',stage=s,split=split,file_sha256=entry['sha256'],array_sha256=old.array_hash(data),rows=n,array_bytes=sum(v.nbytes for v in data.values()),raw_token_columns=tokens.shape[1]);return data
        datasets={s:{split:load(s,split)for split in('train','valid')}for s in range(4)}
        model=old.TerminalGRU(8,15,48,width=32,layers=2).to(device);opt=torch.optim.Adam(model.parameters(),lr=.003)
        if parameters(model)!=13992:raise RuntimeError('fixed GRU32 parameter layout differs')
        replay=Replay128(128,48,0,8);used={};train_sources={s:datasets[s]['train']for s in range(4)}
        rec.emit('configuration',args=vars(args),model_class='unmodified old.TerminalGRU',model_spec=model.spec,actual_parameters=parameters(model),source_sha256=manifest['source_sha256'],manifest_sha256=sha(a.inputs/'manifest.json'),objective='terminal scalar crossentropy/NLL only; no auxiliary/teacher loss',width_selection='P count only, fixed before scores',TEST_read=False,resources=old.resources(model,opt,device,replay))
        for stage in range(4):
            rec.check();completed=0;safe_checkpoint=None;growth=None
            if stage==3:
                probe=dict(initial=torch.tensor([0],device=device),tokens=torch.tensor([[1,3]],device=device),lengths=torch.tensor([2],device=device))
                model.eval()
                with torch.no_grad():before=old.forward(model,probe,args,'soft').clone()
                growth=grow_storage(model,opt,args)
                with torch.no_grad():after=old.forward(model,probe,args,'soft')
                if not torch.equal(before,after)or parameters(model)!=14024:raise RuntimeError('GRU32 vocabulary growth changed old execution/P')
            data=datasets[stage]['train'];indices=np.random.default_rng(np.random.SeedSequence([0,8,stage,9027])).integers(0,len(data['initial']),size=(600,64),dtype=np.int64)
            draws=replay.sample_indices(stage,600,16)if stage else None
            with(rec.out/f'stage{stage}-common-indices.npz').open('xb')as f:np.savez_compressed(f,indices=indices)
            if stage:
                with(rec.out/f'stage{stage}-replay-draws.npz').open('xb')as f:np.savez_compressed(f,buffer_indices=draws)
            rec.emit('stage_begin',stage=stage,parameters=parameters(model),growth=growth,indices_sha256=sha(rec.out/f'stage{stage}-common-indices.npz'),replay=replay.statistics())
            costs=dict(SGD_updates=0,training_wall_seconds=0.,training_CPU_seconds=0.,CUDA_stream_elapsed_seconds=0.,forward_MAC_proxy=0,batch_assembly_wall_seconds=0.,current_training_label_reads=0,old_replay_training_label_reads=0,evaluation_wall_seconds=0.,evaluation_CPU_seconds=0.,evaluation_forward_MAC_proxy=0,evaluation_forward_examples=0,checkpoint_wall_seconds=0.,checkpoint_bytes=0,reload_wall_seconds=0.,maintenance_wall_seconds=0.,exact_training_FLOPs=None,search_executions=0,search_FLOPs=0,search_scope='one fixed P-selected architecture, no runtime search',auxiliary_or_teacher_loss=False)
            curves=[]
            for step in range(601):
                rec.check()
                if step%100==0:
                    metrics={}
                    for s in range(stage+1):
                        metrics[s],timing=old.evaluate(model,datasets[s]['valid'],args,device,rec,modes=('soft',))
                        for key,field in(('evaluation_wall_seconds','wall_seconds'),('evaluation_CPU_seconds','cpu_seconds'),('evaluation_forward_MAC_proxy','forward_MAC_proxy'),('evaluation_forward_examples','actual_forward_examples')):costs[key]+=timing[field]
                    row=dict(stage=stage,step=step,valid=metrics);curves.append(row);rec.emit('actual_VALID_curve',**row)
                if step==600:break
                tick=time.perf_counter()
                if stage:mixed=replay.mix(data,indices[step,:48],draws[step]);b,l,active=old.batch(mixed,np.arange(64),device)
                else:b,l,active=old.batch(data,indices[step],device)
                old.synchronize(device);costs['batch_assembly_wall_seconds']+=time.perf_counter()-tick
                model.train();opt.zero_grad(set_to_none=True);start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);tick,cpu=time.perf_counter(),time.process_time();start.record()
                out=old.forward(model,b,args,'soft');loss=F.cross_entropy(out,b['targets'])
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite terminal NLL')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();end.record();old.synchronize(device)
                if any(not bool(torch.isfinite(v).all())for v in model.parameters()):raise FloatingPointError('nonfinite actual NN parameter')
                elapsed=time.perf_counter()-tick;proxy=model.proxy(64,l,active)['proxy_forward_macs'];completed=step+1
                for key,value in(('SGD_updates',1),('training_wall_seconds',elapsed),('training_CPU_seconds',time.process_time()-cpu),('CUDA_stream_elapsed_seconds',start.elapsed_time(end)/1000.),('forward_MAC_proxy',proxy),('current_training_label_reads',48 if stage else 64),('old_replay_training_label_reads',16 if stage else 0)):costs[key]+=value
                rec.emit('actual_update',stage=stage,step=completed,terminal_NLL=float(loss.detach()),batch_accuracy=float((out.argmax(-1)==b['targets']).float().mean()),preclip_gradient=float(norm),forward_MAC_proxy=proxy,training_wall_seconds=elapsed,active_token_steps=active)
            current_used=indices if stage==0 else indices[:,:48]
            # Persist all paid600 NN steps BEFORE any replay padding/maintenance.
            pre,pre_receipt=save_state(rec,f'gru32-stage{stage}-step600-before-maintenance.pt',model,opt,replay,device,stage,False,used,current_used,costs,manifest['source_sha256']);safe_checkpoint=pre_receipt
            costs['checkpoint_wall_seconds']+=pre_receipt['write_wall_seconds'];costs['checkpoint_bytes']+=pre_receipt['bytes']
            pre_reload=verify_reload(pre,model,opt,replay,device,datasets[stage]['valid'],args,train_sources);costs['reload_wall_seconds']+=pre_reload['wall_seconds']
            maintenance_data,padding=maintenance_view(data,replay.width)
            if padding['cropped']:rec.emit('replay_storage_padding_view',stage=stage,**padding)
            train_sources[stage]=maintenance_data
            maintenance=replay.consume(maintenance_data,current_used,stage,emit=rec.emit);used[stage]=current_used
            source_audit=replay.validate_sources(train_sources,used);costs['maintenance_wall_seconds']+=maintenance['maintenance_wall_seconds']
            rec.emit('replay_boundary',stage=stage,maintenance=maintenance,source_audit=source_audit)
            post,post_receipt=save_state(rec,f'gru32-stage{stage}-step600.pt',model,opt,replay,device,stage,True,used,None,costs,manifest['source_sha256'])
            costs['checkpoint_wall_seconds']+=post_receipt['write_wall_seconds'];costs['checkpoint_bytes']+=post_receipt['bytes']
            reload=verify_reload(post,model,opt,replay,device,datasets[stage]['valid'],args,train_sources);costs['reload_wall_seconds']+=reload['wall_seconds']
            report=dict(stage=stage,costs=costs,checkpoint=post_receipt,premaintenance_checkpoint=pre_receipt,premaintenance_reload=pre_reload,reload=reload,resources=old.resources(model,opt,device,replay),curves=curves,growth=growth,replay_maintenance=maintenance,padding_storage=padding)
            reports.append(report);rec.json(f'completed-stage{stage}-before-TEST.json',dict(report=report,test_data_read=False));rec.emit('stage_complete',**report)
        frozen={k:v.detach().cpu().clone()for k,v in model.state_dict().items()}
        rec.json('ALL_TRAINING_FROZEN_BEFORE_TEST.json',dict(reports=reports,no_more_updates=True,TEST_read=False,actual_training_updates=2400))
        tests={};test_costs=[]
        for stage in range(4):
            tests[stage]={}
            for split in('test_iid','test_long','test_unseen'):
                data=load(stage,split);tests[stage][split],timing=old.evaluate(model,data,args,device,rec,modes=('soft',));test_costs.append(dict(stage=stage,split=split,**timing))
        if not old.tensor_state_equal(frozen,model.state_dict()):raise RuntimeError('TEST evaluation changed frozen NN')
        result=dict(status='ACTUAL_FOUR_STAGE_GRU32_DEVELOPMENT_COMPLETE',schema='terminal-gru32-fourstage-result-v1',formal_Primary=False,actual_training_updates=2400,actual_new_updates_this_process=2400,fresh_initial_NN=True,stages_rerun=0,model_spec=model.spec,parameters=parameters(model),source_sha256=manifest['source_sha256'],reports=reports,tests=tests,test_costs=test_costs,resources=old.resources(model,opt,device,replay),exact_training_FLOPs=None,teacher_or_intermediate_state_supplied=False,objective='terminal NLL only',unchanged_NN_after_TEST=True,cost_boundary='actual training backward+Adam wall/CPU/CUDAstream counted; forwardMAC proxy is not trainingFLOPs; all checkpoint/reload/growth/replay/eval/storage costs reported separately, peak cumulative')
        rec.json('result.json',result);rec.emit('completed',actual_updates=2400,parameters=parameters(model),Primary_frozen=False);print(json.dumps(dict(status=result['status'],parameters=parameters(model))))
    except BaseException as error:
        rec.json('failure.json',dict(stage=stage,completed_updates_in_current_stage=completed,completed_stages=reports,paid600_safe_checkpoint=safe_checkpoint,error=repr(error),traceback=traceback.format_exc()));raise
    finally:rec.stream.close()
if __name__=='__main__':main()
