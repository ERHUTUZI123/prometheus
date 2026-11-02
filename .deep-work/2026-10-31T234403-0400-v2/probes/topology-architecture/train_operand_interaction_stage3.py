"""NO-RUN preparation: fixed600 stage3 Product/Additive operand diagnostic.

Same owned anchor0 stage2 NN/Adam/RNG/Replay128, original data/schedules and
one random token-row growth. Old parameters remain trainable. Scalar terminal
NLL only; no teacher, anchor/profile, tuning, extra phase or accuracy gate.
"""
import argparse,copy,hashlib,json,os,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import continuous_primitive_block as base
import operand_interaction_block as interaction
import owned_parent_driver as old
from context_replay128 import Replay128
from memory_live_stage_train import grow_storage

SCHEMA='operand-interaction-stage3-training-v1'
PARENT_SHA='9bb47e660e4ea67d0bf1a06e93e418f0c3fa50760fa76ff067d3a0c954a6c40f'
REPLAY_SHA='440344f6cd778e8443081cdeafa95847bf0d23aa825f97590c55436b02a7f470'
INDICES_SHA='a840c72965717b198837fb893c288d30c6d20809ecba22439cbb8d850c787007'
DRAWS_SHA='c00780e9f7d57f4c26e3e472da8563b2aaab65d00926aaa55b0a4581fbfa9ab1'
RAW=('initial','tokens','lengths','targets')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def parent_contract(payload,manifest):
    if payload.get('schema')!='factorized-value-development-training-v1'or payload.get('stage')!=2 or payload.get('step')!=600 or payload.get('stage_complete')is not True:raise ValueError('exact completed OWN stage2 required')
    if payload['args']['anchor']!=0 or payload['source_sha256']!=manifest['parent_source_sha256']:raise ValueError('same anchor0 source required')
    if payload['block_checkpoint']['version']!=base.ContinuousPrimitiveBlock.checkpoint_version or payload['block_checkpoint']['organization_policy']!=base.ContinuousPrimitiveBlock.organization_policy:raise ValueError('exact original fcba parent only')

def save(rec,leaf,model,opt,replay,device,complete,used,pending,costs,sources,parent):
    tick=time.perf_counter();p=rec.out/leaf
    value=dict(schema=SCHEMA,stage=3,step=600,stage_complete=complete,maintenance_complete=complete,NN600_complete=True,formal_Primary=False,
        interaction_mode=model.interaction_mode,block_checkpoint=model.checkpoint(),model_state={k:v.detach().cpu().clone()for k,v in model.state_dict().items()},
        optimizer=opt.state_dict(),optimizer_parameter_names=interaction.optimizer_names(model,opt),rng=interaction.rng(device),replay_state=replay.state_dict(),
        committed_used_indices={s:x.copy()for s,x in used.items()},pending_current_used_indices=None if pending is None else pending.copy(),
        source_sha256=sources,parent_checkpoint_sha256=parent,costs=copy.deepcopy(costs),objective='terminal NLL only',fixed_policy=dict(seed=0,steps=600,current=48,replay=16,Adam_lr=.003,clipnorm=1.))
    with p.open('xb')as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    receipt=dict(file=leaf,bytes=p.stat().st_size,sha256=sha(p),write_wall_seconds=time.perf_counter()-tick,stage_complete=complete)
    rec.emit('actual_checkpoint',**receipt);return p,receipt

def verify_reload(path,model,opt,replay,device,args,valid,train_sources):
    tick=time.perf_counter();saved=interaction.rng(device);was=model.training
    try:
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload.get('schema')!=SCHEMA or payload['interaction_mode']!=model.interaction_mode:raise ValueError('unique interaction checkpoint/mode required')
        other=interaction.OperandInteractionBlock.from_checkpoint(payload['block_checkpoint']).to(device)
        other_opt=interaction.restore_optimizer(other,payload['optimizer'],payload['optimizer_parameter_names'])
        replay_copy=Replay128.from_state_dict(payload['replay_state'])
        if not old.tensor_state_equal(model.state_dict(),other.state_dict())or not old.nested_state_equal(opt.state_dict(),other_opt.state_dict()):raise ValueError('real NN/Adam reload differs')
        if interaction.optimizer_names(other,other_opt)!=interaction.optimizer_names(model,opt)or not old.nested_state_equal(replay.state_dict(),replay_copy.state_dict()):raise ValueError('name ordering/replay reload differs')
        provenance=replay_copy.validate_sources(train_sources,payload['committed_used_indices'])
        interaction.restore_rng(payload['rng'],device)
        if not old.nested_state_equal(interaction.rng(device),payload['rng']):raise ValueError('actual RNG reload differs')
        b,_,_=old.batch(valid,np.arange(16),device);model.eval();other.eval()
        with torch.no_grad():
            for mode in('soft','native'):
                if not torch.equal(old.forward(model,b,args,mode),old.forward(other,b,args,mode)):raise ValueError('actual reloaded forward differs')
        try:base.ContinuousPrimitiveBlock.from_checkpoint(payload['block_checkpoint'])
        except ValueError:pass
        else:raise ValueError('old loader must reject interaction checkpoint')
        return dict(actual_NN_Adam_RNG_replay_equal=True,optimizer_by_name=True,forward_equal_soft_native=True,old_loader_rejects=True,replay_provenance=provenance,wall_seconds=time.perf_counter()-tick)
    finally:model.train(was);interaction.restore_rng(saved,device)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in('inputs','out'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--mode',choices=('product','additive'),required=True);parser.add_argument('--max-seconds',type=float,default=850)
    a=parser.parse_args()
    if not 1<=a.max_seconds<=850:parser.error('bounded internal time1..850 required')
    rec=old.Recorder(a.out,a.max_seconds);device=torch.device('cuda:0');completed=0;safe=None;costs={}
    try:
        if not torch.cuda.is_available():raise RuntimeError('same authorized CUDA required, no fallback')
        torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,device)
        manifest=json.loads((a.inputs/'manifest.json').read_text())
        if manifest['interaction_mode']!=a.mode:raise ValueError('frozen mode differs')
        for leaf,digest in manifest['source_sha256'].items():
            if sha(Path(__file__).with_name(leaf))!=digest:raise ValueError('frozen source changed')
        cp=a.inputs/'resume.pt'
        if sha(cp)!=PARENT_SHA or manifest['parent_checkpoint_sha256']!=PARENT_SHA:raise ValueError('exact same OWN stage2 checkpoint required')
        admission_tick,admission_CPU=time.perf_counter(),time.process_time()
        payload=torch.load(cp,map_location='cpu',weights_only=False);parent_contract(payload,manifest);args=argparse.Namespace(**payload['args'])
        interaction.restore_rng(payload['rng'],device);before=interaction.rng(device)
        model,opt,parent,parent_opt=interaction.migrate_parent(payload,a.mode,device)
        migration=interaction.verify_old_migration(model,opt,parent,parent_opt,old)
        if not old.nested_state_equal(before,interaction.rng(device)):raise ValueError('factory changed parent RNG')
        replay=Replay128.from_state_dict(payload['replay_state'])
        if (replay.size,replay.width,replay.seen,replay.draws,replay.replacements)!=(128,48,96000,19200,844)or replay.digest()!=REPLAY_SHA:raise ValueError('exact original stage2 replay required')
        if not old.nested_state_equal(replay.state_dict(),payload['replay_state']):raise ValueError('replay restoration differs')
        def load(stage,split):
            spec=manifest['files'][f'{stage}:{split}'];path=a.inputs/spec['leaf']
            if sha(path)!=spec['sha256']:raise ValueError('raw data source differs')
            with np.load(path,allow_pickle=False)as z:
                if set(z.files)!=set(RAW):raise ValueError('raw terminal fields only')
                data={k:z[k]for k in RAW}
            n=len(data['initial'])
            if any(x.dtype!=np.int64 for x in data.values())or data['tokens'].ndim!=2 or any(data[k].shape!=(n,)for k in('initial','lengths','targets')):raise ValueError('original int64 shapes required')
            if np.any((data['lengths']<0)|(data['lengths']>data['tokens'].shape[1])):raise ValueError('length range')
            active=np.arange(data['tokens'].shape[1])[None]<data['lengths'][:,None]
            if np.any((data['initial']<0)|(data['initial']>=8))or np.any((data['targets']<0)|(data['targets']>=8))or np.any(active&((data['tokens']<0)|(data['tokens']>=(16 if stage==3 else 15)))):raise ValueError('public ranges')
            rec.emit('actual_input_opened',stage=stage,split=split,sha256=spec['sha256'],rows=n,array_bytes=sum(x.nbytes for x in data.values()));return data
        datasets={s:{split:load(s,split)for split in('train','valid')}for s in range(4)}
        sources={s:datasets[s]['train']for s in range(4)};used={}
        for s in range(3):
            ids=np.random.default_rng(np.random.SeedSequence([0,8,s,9027])).integers(0,len(sources[s]['initial']),size=(600,64),dtype=np.int64);used[s]=ids if s==0 else ids[:,:48]
        prior_provenance=replay.validate_sources(sources,used)
        probe,_,_=old.batch(datasets[0]['valid'],np.arange(16),device);parent.to(device);model.eval();parent.eval()
        with torch.no_grad():
            for mode in('soft','native'):
                if not torch.equal(old.forward(model,probe,args,mode),old.forward(parent,probe,args,mode)):raise ValueError('Bzero must exactly preserve old forward')
        initial_branch={name:tensor_sha(getattr(model,name))for name in interaction.NEW_NAMES}
        del parent,parent_opt,payload
        with torch.no_grad():before_growth=old.forward(model,probe,args,'soft').clone()
        growth=grow_storage(model,opt,args)
        with torch.no_grad():after_growth=old.forward(model,probe,args,'soft')
        if not torch.equal(before_growth,after_growth)or model.parameter_count()!=13973:raise ValueError('only576P + original16P row allowed')
        growth['new_token_row_sha256']=tensor_sha(model.token_embedding.weight[15])
        rec.emit('actual_restore_migration_growth',migration=migration,parent_checkpoint_sha256=PARENT_SHA,old_stages_repeated=0,factory_rng_neutral=True,CUDA_RNG_checked=True,parent_replay=prior_provenance,initial_branch_sha256=initial_branch,growth=growth,parameters=model.parameter_count())
        admission_wall=time.perf_counter()-admission_tick;admission_CPU_seconds=time.process_time()-admission_CPU
        train=datasets[3]['train'];indices=np.random.default_rng(np.random.SeedSequence([0,8,3,9027])).integers(0,len(train['initial']),size=(600,64),dtype=np.int64);draws=replay.sample_indices(3,600,16)
        if hashlib.sha256(indices.tobytes()).hexdigest()!=INDICES_SHA or hashlib.sha256(draws.tobytes()).hexdigest()!=DRAWS_SHA:raise ValueError('original actual600 common schedule differs')
        for leaf,field,value in(('stage3-common-indices.npz','indices',indices),('stage3-replay-draws.npz','buffer_indices',draws)):
            with(rec.out/leaf).open('xb')as f:np.savez_compressed(f,**{field:value})
        costs=dict(SGD_updates=0,training_wall_seconds=0.,training_CPU_seconds=0.,CUDA_stream_elapsed_seconds=0.,forward_MAC_proxy=0,shared_operand_linear_MAC=0,interaction_product_multiplications=0,interaction_additions=0,interaction_GELU_elements=0,expert_preactivation_additions=0,batch_assembly_wall_seconds=0.,current_label_reads=0,replay_label_reads=0,evaluation_wall_seconds=0.,evaluation_CPU_seconds=0.,evaluation_forward_MAC_proxy=0,evaluation_actual_soft_rows=0,checkpoint_wall_seconds=0.,checkpoint_bytes=0,reload_wall_seconds=0.,maintenance_wall_seconds=0.,full_training_FLOPs=None,search_FLOPs=0,search_scope='two fixed modes, no score selection/tuning; two actual arms independently paid',teacher_or_auxiliary_loss=False)
        costs.update(restore_migration_admission_wall_seconds=admission_wall,restore_migration_admission_CPU_seconds=admission_CPU_seconds,migration_temporary_parent_NN_bytes=53524,migration_temporary_parent_Adam_bytes=107156)
        curves=[];loop_tick=time.perf_counter();rec.emit('configuration',mode=a.mode,parameters=model.parameter_count(),source_sha256=manifest['source_sha256'],objective='terminal NLL only',fixed600=True,initial_branch_sha256=initial_branch)
        for step in range(601):
            rec.check()
            if step%100==0:
                metrics={}
                for s in range(4):
                    metrics[s],timing=old.evaluate(model,datasets[s]['valid'],args,device,rec,modes=('soft',))
                    costs['evaluation_wall_seconds']+=timing['wall_seconds'];costs['evaluation_CPU_seconds']+=timing['cpu_seconds'];costs['evaluation_forward_MAC_proxy']+=timing['forward_MAC_proxy'];costs['evaluation_actual_soft_rows']+=len(datasets[s]['valid']['initial'])
                row=dict(step=step,valid=metrics);curves.append(row);rec.emit('actual_VALID_curve',**row)
            if step==600:break
            tick=time.perf_counter();mixed=replay.mix(train,indices[step,:48],draws[step]);b,length,active=old.batch(mixed,np.arange(64),device);old.synchronize(device);costs['batch_assembly_wall_seconds']+=time.perf_counter()-tick
            model.train();opt.zero_grad(set_to_none=True);start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);tick,cpu=time.perf_counter(),time.process_time();start.record()
            out,_=model.forward_value(b['initial'],b['tokens'],b['lengths']);loss=F.nll_loss(out,b['targets'])
            if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite actual terminal NLL')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();end.record();old.synchronize(device)
            elapsed=time.perf_counter()-tick;report=model.resource_report(batch_size=64,sequence_length=length,active_steps=active);ops=report['interaction_operations'];completed=step+1
            for key,value in(('SGD_updates',1),('training_wall_seconds',elapsed),('training_CPU_seconds',time.process_time()-cpu),('CUDA_stream_elapsed_seconds',start.elapsed_time(end)/1000.),('forward_MAC_proxy',report['proxy_forward_macs']),('shared_operand_linear_MAC',active*576),('interaction_product_multiplications',ops['product_multiplications']),('interaction_additions',ops['additive_additions']),('interaction_GELU_elements',ops['additive_GELU_elements']),('expert_preactivation_additions',ops['expert_preactivation_additions']),('current_label_reads',48),('replay_label_reads',16)):costs[key]+=value
            rec.emit('actual_update',stage=3,step=completed,terminal_NLL=float(loss.detach()),loss=float(loss.detach()),batch_accuracy=float((out.argmax(-1)==b['targets']).float().mean()),preclip_gradient=float(norm),training_wall_seconds=elapsed,forward_MAC_proxy=report['proxy_forward_macs'],active_token_steps=active,shared_operand_linear_MAC=active*576)
        costs['training_loop_including_VALID_wall_seconds']=time.perf_counter()-loop_tick
        pre,pre_receipt=save(rec,'operand-stage3-step600-before-maintenance.pt',model,opt,replay,device,False,used,indices[:,:48],costs,manifest['source_sha256'],PARENT_SHA);safe=pre_receipt
        costs['checkpoint_wall_seconds']+=pre_receipt['write_wall_seconds'];costs['checkpoint_bytes']+=pre_receipt['bytes']
        pre_reload=verify_reload(pre,model,opt,replay,device,args,datasets[3]['valid'],sources);costs['reload_wall_seconds']+=pre_reload['wall_seconds']
        if np.any(train['lengths']>48)or np.any(train['tokens'][:,48:]):raise ValueError('cannot discard active replay bytes')
        view=dict(train,tokens=train['tokens'][:,:48])
        if not np.shares_memory(view['tokens'],train['tokens']):raise ValueError('zero-copy replay storage view required')
        sources[3]=view;maintenance=replay.consume(view,indices[:,:48],3,emit=rec.emit);used[3]=indices[:,:48];audit=replay.validate_sources(sources,used)
        if replay.seen!=124800 or replay.draws!=28800 or replay.digest()!='6405bee32c7b0daa655bac0a8e54e91c44c6516b8442d66560b5e1a2a3323641':raise ValueError('original replay after600 differs')
        costs['maintenance_wall_seconds']+=maintenance['maintenance_wall_seconds']
        post,checkpoint=save(rec,'operand-stage3-step600.pt',model,opt,replay,device,True,used,None,costs,manifest['source_sha256'],PARENT_SHA)
        costs['checkpoint_wall_seconds']+=checkpoint['write_wall_seconds'];costs['checkpoint_bytes']+=checkpoint['bytes']
        reload=verify_reload(post,model,opt,replay,device,args,datasets[3]['valid'],sources);costs['reload_wall_seconds']+=reload['wall_seconds']
        frozen={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};rec.json('ALL_TRAINING_FROZEN_BEFORE_TEST.json',dict(checkpoint=checkpoint,no_more_updates=True,actual_new_updates=600,parent_updates=1800,TEST_read=False,no_selection=True))
        tests={};test_costs=[]
        for s in range(4):
            tests[s]={}
            for split in('test_iid','test_long','test_unseen'):
                data=load(s,split);tests[s][split],timing=old.evaluate(model,data,args,device,rec,modes=('soft','native'));test_costs.append(dict(stage=s,split=split,**timing))
        if not old.tensor_state_equal(frozen,model.state_dict()):raise ValueError('TEST changed frozen NN')
        result=dict(status='ACTUAL_OPERAND_INTERACTION_STAGE3_600_COMPLETE',schema=SCHEMA,mode=a.mode,formal_Primary=False,parameters=model.parameter_count(),new_parameters=576,actual_new_updates=600,parent_updates=1800,old_stages_repeated=0,source_sha256=manifest['source_sha256'],parent_checkpoint_sha256=PARENT_SHA,parent_paid_lineage=manifest['parent_paid_lineage'],migration=migration,growth=growth,initial_branch_sha256=initial_branch,costs=costs,curves=curves,premaintenance_checkpoint=pre_receipt,premaintenance_reload=pre_reload,checkpoint=checkpoint,reload=reload,replay_maintenance=maintenance,replay_source_audit=audit,tests=tests,test_costs=test_costs,resources=old.resources(model,opt,device,replay),interaction_resources=model.resource_report(),gradient_tensor_bytes=sum(p.grad.numel()*p.grad.element_size()for p in model.parameters()if p.grad is not None),TRAIN_VALID_data_array_bytes=sum(value.nbytes for stage in datasets.values()for data in stage.values()for value in data.values()),input_snapshot_file_bytes=manifest['input_snapshot_file_bytes'],compiler_cache_bytes=None,cache_measurement_scope='no compiler subprocess used; transitiveCUDA/vendor library cache bytes unmeasured',provider_effective_compute=None,driver_wall_seconds=time.perf_counter()-rec.started,objective='terminal NLL only',no_accuracy_gate=True,NN_unchanged_after_TEST=True,cost_boundary='all actual600 F/B+Adam/clip and batch/VALID/growth/migration/checkpoint/reload/replay/TEST/storage costs retained; whole driver includes restoration/admission, forwardMAC is not fullFLOPs; parent1800 shared lineage paid once, two new600 arms paid separately')
        rec.json('result.json',result);rec.emit('completed',actual_new_updates=600,parameters=model.parameter_count());print(json.dumps(dict(status=result['status'],mode=a.mode,parameters=model.parameter_count())))
    except BaseException as error:rec.json('failure.json',dict(completed_updates=completed,paid600_safe_checkpoint=safe,costs=costs,error=repr(error),traceback=traceback.format_exc()));raise
    finally:rec.stream.close()
if __name__=='__main__':main()
