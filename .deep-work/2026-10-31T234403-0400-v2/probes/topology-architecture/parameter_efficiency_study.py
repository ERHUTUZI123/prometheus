"""Six-stage parameter-growth comparison using only raw terminal feedback.

Primary metric: actual persistent P versus newly acquired AND retained tasks,
with all non-parameter storage charged separately. Same compact formation is
available to topology and ordinary constraints. No claim of a topology saving
is built into the growth policy. Existing external CUDA sandbox is required.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from integrate_native_repair import Recorder, plain, sha256, tensor_bytes, synchronize
from parameter_native_models import (CompactSemanticPromotionModel, CompactContinuousNativeModel,
                                     compact_model_from_checkpoint)
from diagnose_projection_numeric import restore_parameter_layouts
from parameter_transformer_baselines import GrowingTaskTransformer
from terminal_profile_formation import observe, candidate
from semantic_promotion_model import SemanticPromotionModel
from workspace_refinement import grow_workspace
from parameter_consistency_refresh import refresh_native_consistency
import continuous_native_model

ALPHABET=tuple(range(1,7))
SPLITS=('train','valid','iid','ood','length16')
ARMS=('topology','constraint_compact','constraint_dense','freeze','modular',
      'transformer64','replay64','transformer96','replay96')
continuous_native_model.ALPHABETS={s:ALPHABET for s in range(6)}


def native_arm(arm):return arm in ARMS[:5]


def array_hash(data):
    h=hashlib.sha256()
    for k in ('tokens','starts','task_ids','targets'):h.update(k.encode()+np.asarray(data[k],dtype='<i8').tobytes())
    return h.hexdigest()


def parameters(model):return dict(total=sum(p.numel() for p in model.parameters()),
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
    bytes=sum(p.numel()*p.element_size() for p in model.parameters()))


def build_native(arm,n,dim,seed,task_count=6,n_ops=1):
    cls=CompactContinuousNativeModel if arm=='topology' else (SemanticPromotionModel if arm=='constraint_dense' else CompactSemanticPromotionModel)
    kwargs=dict(vocab=6,n_tasks=task_count,latent_factor=dim//n,n_ops=n_ops,emb_dim=48,max_bank=48,
                protection=arm not in ('freeze','modular'),seed=seed)
    if arm=='topology':kwargs['homotopy_lift_cache_capacity']=128
    return cls(n,**kwargs)


def restore_actual(payload,args,device):
    if args.arm in ('freeze','modular'):
        model=PrivateModels(args.arm,args.scale,args.seed).to(device)
        for i,spec in enumerate(payload['specs']):
            prefix=f'models.{i}.'
            state={k[len(prefix):]:v for k,v in payload['model'].items() if k.startswith(prefix)}
            m=compact_model_from_checkpoint(dict(model=state,model_extra=payload['model_extra'][i]),device)
            restore_parameter_layouts(m,state)
            if args.arm=='freeze' and i:
                first=model.models[0]
                for a,b in ((m.encoder,first.encoder),(m.decoder,first.decoder),(m.token.weight,first.token.weight)):
                    if not torch.equal(a,b):raise RuntimeError('frozen shared-body checkpoint values disagree')
                m.encoder,m.decoder,m.token=first.encoder,first.decoder,first.token
            model.models.append(m)
        model.freeze_old()
    elif native_arm(args.arm):
        if args.arm=='constraint_dense':
            spec=payload['specs'];model=build_native(args.arm,spec['n'],spec['d'],args.seed,n_ops=1).to(device)
            model.load_state_dict(payload['model']);model.load_extra_state_dict(payload['model_extra'])
        else:model=compact_model_from_checkpoint(payload,device)
        restore_parameter_layouts(model,payload['model'])
    else:
        model=GrowingTaskTransformer(args.scale,vocab=6,n_tasks=6,max_len=18,d_model=96 if '96' in args.arm else 64).to(device)
        model.load_state_dict(payload['model']);restore_parameter_layouts(model,payload['model'])
    for n,p in model.named_parameters():p.requires_grad_(payload['parameter_requires_grad'][n])
    return model


class PrivateModels(nn.Module):
    """Frozen shared E/R/token backbone or separate growing full modules."""
    def __init__(self,arm,n,seed):
        super().__init__();self.arm,self.n,self.seed=arm,n,seed;self.models=nn.ModuleList()

    def add(self,observed_states,device):
        dim=2*self.n if self.arm=='freeze' else (self.n if observed_states<=self.n else 2*self.n)
        m=build_native(self.arm,self.n,dim,self.seed+len(self.models)*997,task_count=1).to(device)
        if self.arm=='freeze' and self.models:
            # Actual Parameter identity sharing, not uncounted copied weights.
            first=self.models[0];m.encoder=first.encoder;m.decoder=first.decoder;m.token=first.token
        self.models.append(m);self.freeze_old();return m

    def freeze_old(self):
        for i,m in enumerate(self.models):
            for p in m.parameters():p.requires_grad_(i==len(self.models)-1)
        if self.arm=='freeze' and len(self.models)>1:
            m=self.models[-1]
            for p in (m.encoder,m.decoder,*m.token.parameters()):p.requires_grad_(False)

    def forward(self,tokens,starts,task_ids):
        out=self.models[-1].encoder.new_empty((len(starts),self.n))
        for t in task_ids.unique().tolist():
            take=task_ids==t;out[take]=self.models[t](tokens[take],starts[take],torch.zeros_like(task_ids[take]))
        return out

    @torch.no_grad()
    def hard_eval(self,tokens,starts,task_ids):
        out=self.models[-1].encoder.new_empty((len(starts),self.n))
        for t in task_ids.unique().tolist():
            take=task_ids==t;out[take]=self.models[t].hard_eval(tokens[take],starts[take],torch.zeros_like(task_ids[take]))
        return out


def minibatch(data,indices,device):
    return {k:torch.as_tensor(v[indices],dtype=torch.long,device=device) for k,v in data.items()}


@torch.no_grad()
def evaluate(model,datasets,seen,device,native,rec):
    began=time.perf_counter();was=model.training;model.eval();result={}
    for task in range(seen+1):
        result[task]={}
        for split in SPLITS[1:]:
            data=datasets[task][split];correct=hard=0;loss=0.
            for start in range(0,len(data['starts']),64):
                rec.check();b=minibatch(data,np.arange(start,min(start+64,len(data['starts']))),device)
                logits=model(b['tokens'],b['starts'],b['task_ids'])
                if not torch.isfinite(logits).all():raise FloatingPointError('nonfinite actual neural evaluation')
                correct+=int((logits.argmax(-1)==b['targets']).sum());loss+=float(F.cross_entropy(logits,b['targets'],reduction='sum'))
                if native:hard+=int((model.hard_eval(b['tokens'],b['starts'],b['task_ids']).argmax(-1)==b['targets']).sum())
            total=len(data['starts']);result[task][split]=dict(neural=correct/total,terminal_CE=loss/total,
                deployed=hard/total if native else correct/total,n=total)
    synchronize(device);model.train(was);return result,time.perf_counter()-began


def resource_state(model,optimizer,arm,replay_bytes=0):
    refs=[];geometry=[]
    members=list(model.models) if isinstance(model,PrivateModels) else [model]
    for m in members:
        if hasattr(m,'reference'):refs.extend(m.reference)
        if hasattr(m,'geometry_resource_report'):geometry.append(m.geometry_resource_report())
    packed=json.dumps(plain(refs),sort_keys=True,separators=(',',':')).encode()
    return dict(parameters=parameters(model),optimizer_tensor_bytes=tensor_bytes(optimizer.state_dict()),
        registered_buffer_bytes=sum(b.numel()*b.element_size() for b in model.buffers()),
        model_extra_json_bytes=len(json.dumps(plain(
            [m.extra_state_dict() for m in members] if hasattr(members[0],'extra_state_dict') else {}),
            sort_keys=True,separators=(',',':')).encode()),
        complete_reference_json_bytes=len(packed),reference_array_bytes=sum(
            a.nbytes for g in refs for a in [g['enc'],g['dec'],*g['trans'].values()]),
        replay_tensor_bytes=replay_bytes,geometry=geometry,
        parameter_claim_boundary='frozen and preallocated parameters counted; operation scores reconstructed from trainable coordinates are activations, not stored parameters')


def scored(metrics):
    """Same deployed+neural criterion, test only scores, never selects a model."""
    return {int(t):all(m[s][mode]>=threshold for s,threshold in (('iid',.95),('ood',.90),('length16',.90))
            for mode in ('deployed','neural')) for t,m in metrics.items()}


def save_model(model,optimizer,args,stage,step,rec,resources,capabilities,complete):
    path=rec.out/f'{args.arm}-stage{stage}-step{step}.pt'
    if isinstance(model,PrivateModels):
        specs=[dict(n=m.n,d=m.d,n_ops=m.n_ops,n_tasks=m.n_tasks) for m in model.models]
    elif native_arm(args.arm):specs=dict(n=model.n,d=model.d,n_ops=model.n_ops,n_tasks=model.n_tasks)
    else:specs=dict(n=args.scale,width=96 if '96' in args.arm else 64)
    extras=([m.extra_state_dict() for m in model.models] if isinstance(model,PrivateModels)
            else model.extra_state_dict() if hasattr(model,'extra_state_dict') else None)
    payload=dict(schema='parameter-growth-neural-checkpoint-v1',args=vars(args),stage=stage,step=step,
        model=model.state_dict(),optimizer=optimizer.state_dict(),specs=specs,model_extra=extras,
        parameter_requires_grad={n:p.requires_grad for n,p in model.named_parameters()},
        capabilities=capabilities,stage_training_complete=complete,resources=resources,
        torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all(),M0='NOT_PASSED')
    torch.save(payload,path)
    if path.stat().st_size>100*1024*1024:raise RuntimeError('bounded checkpoint100MiB limit')
    rec.emit('checkpoint',file=path.name,sha256=sha256(path),bytes=path.stat().st_size,stage=stage,step=step)
    return path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--scale',type=int,choices=(8,16),required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=600)
    p.add_argument('--stop-after-stage',type=int,default=5);p.add_argument('--resume',type=Path)
    p.add_argument('--consistency-refresh',action='store_true')
    p.add_argument('--curriculum',action='store_true')
    p.add_argument('--max-seconds',type=float,default=850);p.add_argument('--device',choices=('cuda:0',),default='cuda:0')
    args=p.parse_args()
    if args.out.exists() or not 1<=args.steps<=600 or not 0<args.max_seconds<=850 or not 0<=args.stop_after_stage<6:p.error('new output, bounded600 updates/phase and850 seconds required')
    rec=Recorder(args.out,args.max_seconds);device=torch.device(args.device)
    if not torch.cuda.is_available():raise RuntimeError('GPU required; no fallback')
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.3,device)
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    datasets=[];input_info=[]
    for t in range(6):
        splits={}
        for split in SPLITS:
            path=args.datasets/f'task{t}-{split}.npz'
            with np.load(path,allow_pickle=False) as z:splits[split]={k:np.asarray(z[k],dtype=np.int64) for k in ('tokens','starts','task_ids','targets')}
            input_info.append(dict(task=t,split=split,array_sha256=array_hash(splits[split]),file_sha256=sha256(path)))
        datasets.append(splits)
    private=args.arm in ('freeze','modular');isnative=native_arm(args.arm)
    if private:model=PrivateModels(args.arm,args.scale,args.seed).to(device)
    elif isnative:model=build_native(args.arm,args.scale,args.scale,args.seed).to(device)
    else:model=GrowingTaskTransformer(args.scale,vocab=6,n_tasks=6,max_len=18,d_model=96 if '96' in args.arm else 64).to(device)
    start_stage=0;initial_allocated_parameters=parameters(model)['total'];previous_parameters=0;stable=set();admitted=set();reports=[]
    if args.resume:
        c=torch.load(args.resume,map_location='cpu',weights_only=False)
        if c['schema']!='parameter-growth-neural-checkpoint-v1' or c['args']['arm']!=args.arm or c['args']['scale']!=args.scale or c['args']['steps']!=args.steps or not c['stage_training_complete']:raise ValueError('actual completed preceding stage checkpoint required')
        model=restore_actual(c,args,device)
        start_stage=c['stage']+1;stable=set(c['capabilities']['stable']);admitted=set(c['capabilities']['admitted']);previous_parameters=parameters(model)['total']
        torch.set_rng_state(c['torch_rng']);np.random.set_state(c['numpy_rng']);random.setstate(c['python_rng']);torch.cuda.set_rng_state_all(c['cuda_rng'])
        rec.emit('actual_completed_stage_resume',source=str(args.resume),sha256=sha256(args.resume),start_stage=start_stage,stage_not_retrained=c['stage'])
    rec.emit('configuration',args=vars(args),inputs=input_info,MAIN='TOPOLOGY_USER_LOCKED',M0='NOT_PASSED',M3='ultimate',
        primary_baseline='Transformer64+96 with/without replay',initial_parameters=initial_allocated_parameters,
        first_capability_delta_includes_all_initial_allocated_parameters=True,
        future_task_embedding_capacity_preallocated_and_counted=True,
        capability_thresholds=dict(iid=.95,ood=.90,length16=.90,both_neural_and_deployed=True),
        stability_scope='primary Qstable at successive stage-final evaluations; intermediate curves separately retained; not every-step neural guarantee',
        growth_uses_train_only=True,admission_uses_valid_only=True,test_never_selects_growth=True,
        coordinate_compression_available_to_nontopological_controls=True)
    try:
        for stage in range(start_stage,args.stop_after_stage+1):
            rec.check();phase_began=time.perf_counter()
            data=datasets[stage]['train'];formation=None;maintenance_seconds=0.;train_seconds=0.;eval_seconds=0.
            if isnative:
                local={k:v.copy() for k,v in data.items()}
                if private:local['task_ids'].fill(0)
                table=observe(local,ALPHABET);rec.json(f'stage{stage}-observations.json',table['audit'])
                if private:
                    current=model.add(table['required_states'],device);current.set_stage(0)
                else:
                    current=model;current.set_stage(stage)
                    if table['required_states']>current.d:
                        grown=(grow_workspace(current,2*args.scale) if args.arm=='constraint_dense'
                               else current.grow_safe(2*args.scale))
                        rec.emit('actual_workspace_growth',stage=stage,report=grown)
                        if not grown['success']:raise RuntimeError('observed state growth failed')
                graph,audit=candidate(current,table,local);rec.json(f'stage{stage}-formation.json',audit)
                formation=current.materialize(current.stage,ALPHABET,None,candidate_graph=graph)
                rec.emit('actual_neural_formation',stage=stage,report=formation)
                if not formation['success']:raise RuntimeError('native formation transaction failed')
                if private:model.freeze_old()
            else:model.set_stage(stage)
            optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.01 if isnative else .003)
            batch_rng=np.random.default_rng(np.random.SeedSequence([args.seed,args.scale,stage,9027]))
            indices=batch_rng.integers(0,len(data['starts']),size=(args.steps,64),dtype=np.int64)
            if args.curriculum:
                lengths=np.count_nonzero(data['tokens'],axis=1)
                for update in range(min(300,args.steps)):
                    pool=np.flatnonzero(lengths<=(1 if update<150 else 3))
                    indices[update]=batch_rng.choice(pool,size=64,replace=True)
                rec.emit('prospective_short_to_long_curriculum',stage=stage,
                    fixed_schedule='0..149 length<=1;150..299 length<=3;300..599 entire originalTRAIN',
                    new_labels=0,gradient_budget_changed=False,test_used_to_choose_schedule=False)
            np.savez_compressed(rec.out/f'stage{stage}-common-indices.npz',indices=indices)
            replay=None
            if args.arm.startswith('replay') and stage:
                replay={k:np.concatenate([datasets[t]['train'][k] for t in range(stage)]) for k in data}
            old_indices=(np.random.default_rng(np.random.SeedSequence([args.seed,args.scale,stage,9028])).integers(0,len(replay['starts']),size=(args.steps,32),dtype=np.int64) if replay else None)
            if old_indices is not None:np.savez_compressed(rec.out/f'stage{stage}-actual-replay-indices.npz',indices=old_indices)
            curve=[];refresh_report=None;refresh_seconds=0.;optimizer_before_refresh_bytes=None
            for step in range(args.steps+1):
                rec.check()
                if step==args.steps and isnative and args.consistency_refresh:
                    # Teacher-free maintenance at a FIXED stage boundary. The
                    # provisional current graph is an OWN target, not an
                    # already validated capability. Restore old refs afterward.
                    original_refs=current.reference
                    if not any(g['task']==current.stage for g in original_refs):
                        current.reference=original_refs+[current.graph(current.stage,ALPHABET)]
                    old_optimizer_state=optimizer.state_dict()
                    optimizer_before_refresh_bytes=tensor_bytes(old_optimizer_state)
                    whole_model_parameter_count_before=parameters(model)['total']
                    began=time.perf_counter()
                    try:
                        refresh_report=refresh_native_consistency(current,max_length=16,
                            preserve_endpoints=args.arm=='freeze')
                    finally:current.reference=original_refs
                    synchronize(device);refresh_seconds=time.perf_counter()-began
                    rec.emit('fixed_stage_boundary_own_native_refresh',stage=stage,report=refresh_report,
                        current_provisional_target_not_an_admitted_capability=True,
                        old_references_restored=True,teacher_feedback_used=False)
                    if not refresh_report['success']:raise RuntimeError('class-consistency maintenance failed: '+refresh_report.get('reason','unknown'))
                    if private:model.freeze_old()
                    if parameters(model)['total']!=whole_model_parameter_count_before:
                        raise RuntimeError('maintenance duplicated a shared Parameter or added global parameters')
                    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.01)
                    optimizer.load_state_dict(old_optimizer_state)
                    rec.emit('optimizer_rebound_after_parameter_replacement',stage=stage,
                        state_bytes_before=optimizer_before_refresh_bytes,state_bytes_after=tensor_bytes(optimizer.state_dict()),
                        state_transported_by_same_parameter_order_and_shape=True,
                        next_task_materialization_will_reset_Adam_as_before=True)
                if step in (0,100,300,args.steps):
                    metrics,elapsed=evaluate(model,datasets,stage,device,isnative,rec);eval_seconds+=elapsed
                    resources=resource_state(model,optimizer,args.arm,0 if replay is None else sum(x.nbytes for x in replay.values()))
                    row=dict(stage=stage,step=step,metrics=metrics,resources=resources)
                    curve.append(row);rec.emit('curve',**row)
                    if step and step!=args.steps:save_model(model,optimizer,args,stage,step,rec,resources,dict(stable=sorted(stable),admitted=sorted(admitted)),False)
                if step==args.steps:break
                b=minibatch(data,indices[step],device)
                if replay is not None:
                    old=minibatch(replay,old_indices[step],device)
                    b={k:torch.cat((v[:32],old[k]),0) for k,v in b.items()}
                synchronize(device);began=time.perf_counter();optimizer.zero_grad(set_to_none=True)
                logits=model(b['tokens'],b['starts'],b['task_ids']);ce=F.cross_entropy(logits,b['targets']);geo=ce.new_zeros(())
                if args.arm=='topology':geo=model.geometry_regularizer()
                loss=ce+.01*geo
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite actual training loss')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(grad):raise FloatingPointError('nonfinite actual parameter gradient')
                before=({n:p.detach().clone() for n,p in model.named_parameters()} if args.arm in ('topology','constraint_compact','constraint_dense') else None)
                optimizer.step();synchronize(device);train_seconds+=time.perf_counter()-began
                guard=None
                if before is not None:
                    began=time.perf_counter();guard=model.project_update(before);synchronize(device);maintenance_seconds+=time.perf_counter()-began
                rec.emit('actual_update',stage=stage,step=step+1,terminal_CE=float(ce),geometry=float(geo),objective=float(loss),preclip_gradient=float(grad),guard=guard)
            metrics=curve[-1]['metrics'];new_valid=all(metrics[stage]['valid'][k]>=.95 for k in ('neural','deployed'))
            if new_valid:
                admitted.add(stage)
                if isnative:current.consolidate(current.stage,ALPHABET)
            previous_stable=set(stable)
            successful=scored(metrics);stable={t for t in stable if successful[t]}
            if stage in admitted and successful[stage]:stable.add(stage)
            P=parameters(model)['total'];deltaP=P-previous_parameters
            # Newly stable acquisition is distinct from count recovered later.
            lost_old=sorted(previous_stable-stable)
            net_deltaC=len(stable)-len(previous_stable)
            deltaC=int(stage in stable and not lost_old)
            resources=resource_state(model,optimizer,args.arm,0 if replay is None else sum(x.nbytes for x in replay.values()))
            caps=dict(admitted=sorted(admitted),stable=sorted(stable),currently_successful=[t for t,s in successful.items() if s],
                new_successfully_retained=deltaC,total_stable=len(stable),delta_parameters=deltaP,
                lost_previously_stable_capabilities=lost_old,net_delta_stable_capabilities=net_deltaC,
                deltaP_per_new_retained_capability=deltaP/net_deltaC if net_deltaC>0 and not lost_old else None)
            path=save_model(model,optimizer,args,stage,args.steps,rec,resources,caps,True)
            # Real checkpoint roundtrip; charge new cold caches and keep RNG
            # unchanged for the next stage's genuinely new parameters.
            rng=(torch.get_rng_state(),np.random.get_state(),random.getstate(),torch.cuda.get_rng_state_all())
            began=time.perf_counter();loaded=torch.load(path,map_location='cpu',weights_only=False)
            restored=restore_actual(loaded,args,device)
            if any(not torch.equal(v.detach().cpu(),restored.state_dict()[k].detach().cpu()) for k,v in model.state_dict().items()):
                raise RuntimeError('actual neural checkpoint changed learned tensors')
            with torch.no_grad():
                for t in range(stage+1):
                    b=minibatch(datasets[t]['valid'],np.arange(64),device)
                    left=model(b['tokens'],b['starts'],b['task_ids']).argmax(-1)
                    right=restored(b['tokens'],b['starts'],b['task_ids']).argmax(-1)
                    if not torch.equal(left,right):raise RuntimeError('actual reload changed neural decisions')
                    if isnative and not torch.equal(model.hard_eval(b['tokens'],b['starts'],b['task_ids']),restored.hard_eval(b['tokens'],b['starts'],b['task_ids'])):
                        raise RuntimeError('actual reload changed native execution')
            synchronize(device);reload_seconds=time.perf_counter()-began
            reload_resources=resource_state(restored,optimizer,args.arm)
            rec.emit('actual_neural_checkpoint_reload_verified',stage=stage,all_parameter_bits_equal=True,
                checked_rows_per_seen_task=64,seconds=reload_seconds,resources=reload_resources)
            del restored
            torch.set_rng_state(rng[0]);np.random.set_state(rng[1]);random.setstate(rng[2]);torch.cuda.set_rng_state_all(rng[3])
            summary=dict(stage=stage,parameters=P,delta_parameters=deltaP,capabilities=caps,resources=resources,
                final_metrics=metrics,formation=formation,train_seconds=train_seconds,guard_seconds=maintenance_seconds,
                eval_seconds=eval_seconds,phase_wall_seconds=time.perf_counter()-phase_began,
                checkpoint=path.name,checkpoint_bytes=path.stat().st_size,curves=curve,
                actual_reload_verified=True,reload_seconds=reload_seconds,reload_resources=reload_resources)
            summary['boundary_consistency_refresh']=refresh_report
            summary['boundary_consistency_seconds']=refresh_seconds
            summary['optimizer_before_refresh_tensor_bytes']=optimizer_before_refresh_bytes
            summary['actual_current_training_episodes']=args.steps*(32 if replay is not None else 64)
            summary['actual_old_training_episodes']=args.steps*32 if replay is not None else 0
            reports.append(summary);rec.emit('stage_complete',**summary)
            rec.json(f'completed-through-stage{stage}.json',dict(arm=args.arm,reports=reports,M0='NOT_PASSED'))
            previous_parameters=P;del optimizer;gc.collect();torch.cuda.empty_cache()
        rec.json('summary.json',dict(status='DECLARED_STAGES_COMPLETED',arm=args.arm,scale=args.scale,seed=args.seed,
            reports=reports,actual_completed_stages=list(range(start_stage,args.stop_after_stage+1)),
            M0='NOT_PASSED',prior_stages_not_retrained=bool(args.resume)))
    except BaseException as error:
        rec.emit('run_failed',error=type(error).__name__,reason=str(error),completed_stages=[r['stage'] for r in reports])
        rec.json('failure.json',dict(error=type(error).__name__,reason=str(error),reports=reports,M0='NOT_PASSED'))
        raise
    finally:rec.stream.close()


if __name__=='__main__':main()
