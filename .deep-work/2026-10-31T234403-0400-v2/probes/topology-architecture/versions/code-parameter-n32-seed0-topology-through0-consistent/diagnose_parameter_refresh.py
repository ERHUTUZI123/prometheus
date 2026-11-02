"""Actual pre/post repair of ONE owned six-stage neural checkpoint, no training.

The refresh helper sees no task labels. Evaluation is separately read only
before/after the fixed intervention; it never selects targets or hyperparameters.
Old failed continual-retention records remain historical failures.
"""
import argparse
import copy
import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch

from integrate_native_repair import Recorder, sha256, synchronize
from parameter_efficiency_study import restore_actual, evaluate, resource_state, parameters, SPLITS
from parameter_consistency_refresh import refresh_native_consistency


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--datasets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=150)
    args=p.parse_args()
    if args.out.exists() or not 0<args.max_seconds<=150:p.error('new output and bounded150 seconds required')
    rec=Recorder(args.out,args.max_seconds)
    source=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    original_args=argparse.Namespace(**source['args'])
    if source.get('schema')!='parameter-growth-neural-checkpoint-v1' or original_args.arm not in ('topology','constraint_compact'):
        raise ValueError('actual compact shared-model checkpoint required')
    device=torch.device('cuda:0');torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('authorized CUDA required')
    torch.cuda.set_per_process_memory_fraction(.3,device)
    model=restore_actual(source,original_args,device)
    datasets=[]
    for task in range(6):
        splits={}
        for split in SPLITS:
            path=args.datasets/f'task{task}-{split}.npz'
            with np.load(path,allow_pickle=False) as z:splits[split]={k:np.asarray(z[k],dtype=np.int64) for k in ('tokens','starts','task_ids','targets')}
        datasets.append(splits)
    rec.emit('configuration',source=str(args.checkpoint),source_sha256=sha256(args.checkpoint),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),MAIN='TOPOLOGY_USER_LOCKED',M0='NOT_PASSED',
        primary_metric='deltaP/delta_successfully_retained_capability',
        labels_to_refresh_helper=False,extra_gradient_updates=0,
        interpretation='repair current neural/native agreement; does not rewrite earlier failed continuous-retention trajectory',
        possible_GPU_overlap='other authorized baseline process may overlap; no independent latency superiority claim')
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    before_resource=resource_state(model,optimizer,original_args.arm)
    before,pre_seconds=evaluate(model,datasets,int(source['stage']),device,True,rec)
    rec.emit('before',metrics=before,resources=before_resource,seconds=pre_seconds)
    report=refresh_native_consistency(model,max_length=16)
    rec.emit('actual_parameter_consistency_refresh',report=report)
    rec.json('refresh-report.json',report)
    if not report.get('success'):
        rec.json('failure.json',dict(status='FIXED_REFRESH_REJECTED_OR_ROLLED_BACK',report=report,
            before=before,M0='NOT_PASSED'))
        raise RuntimeError(report.get('reason','refresh failed'))
    optimizer=torch.optim.Adam(model.parameters(),lr=.01)
    after,post_seconds=evaluate(model,datasets,int(source['stage']),device,True,rec)
    after_resource=resource_state(model,optimizer,original_args.arm)
    if before_resource['parameters']['total']!=after_resource['parameters']['total']:
        raise RuntimeError('repair unexpectedly added parameters')
    payload=copy.copy(source)
    payload.update(model=model.state_dict(),model_extra=model.extra_state_dict(),optimizer=optimizer.state_dict(),
        parameter_requires_grad={n:p.requires_grad for n,p in model.named_parameters()},
        consistency_refresh=dict(source_checkpoint_sha256=sha256(args.checkpoint),report=report,
            actual_extra_gradient_updates=0,old_optimizer_state_reset=True,
            earlier_capability_history_NOT_rewritten=True),resources=after_resource)
    path=rec.out/'topology-stage5-consistency-refreshed-real.pt';torch.save(payload,path)
    if path.stat().st_size>100*1024*1024:raise RuntimeError('checkpoint100MiB cap')
    start=time.perf_counter();loaded=torch.load(path,map_location='cpu',weights_only=False)
    restored=restore_actual(loaded,original_args,device)
    if any(not torch.equal(v.detach().cpu(),restored.state_dict()[k].detach().cpu()) for k,v in model.state_dict().items()):
        raise RuntimeError('refreshed real neural checkpoint changed parameters on reload')
    reloaded,reload_eval_seconds=evaluate(restored,datasets,int(source['stage']),device,True,rec)
    for task in after:
        for split in after[task]:
            for key in ('neural','deployed'):
                if after[task][split][key]!=reloaded[task][split][key]:raise RuntimeError('refreshed checkpoint decisions changed on actual reload')
    synchronize(device)
    result=dict(status='REAL_NEURAL_REFRESH_AND_RELOAD_COMPLETED',before=before,after=after,
        report=report,delta_parameters=parameters(model)['total']-before_resource['parameters']['total'],
        before_resources=before_resource,after_resources=after_resource,
        original_optimizer_tensor_bytes=source['resources']['optimizer_tensor_bytes'],
        new_optimizer_reset=True,new_gradient_updates=0,new_external_information=0,
        historical_Qstable_unchanged=source['capabilities'],
        before_eval_seconds=pre_seconds,after_eval_seconds=post_seconds,
        actual_reload_eval_seconds=reload_eval_seconds,actual_reload_total_seconds=time.perf_counter()-start,
        actual_reload_resources=resource_state(restored,optimizer,original_args.arm),
        checkpoint=path.name,checkpoint_bytes=path.stat().st_size,checkpoint_sha256=sha256(path),
        all_parameter_bits_reload_equal=True,all24_seen_split_interfaces_reload_equal=True,
        M0='NOT_PASSED',wall_seconds=time.perf_counter()-rec.started)
    rec.json('summary.json',result);rec.stream.close()


if __name__=='__main__':main()
