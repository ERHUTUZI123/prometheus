"""Paired maintenance training at THREE actual learned growth checkpoints.

This is a cost diagnostic, NOT nine new capability acquisitions. All arms at
each size start from the same learned neural state and perform200 identical
terminalCE+.01geometry updates, protecting all admitted old/current contracts.
Compare legacy exact-omega cache, homotopy-certified cache and cached direct
compiled-map execution. Direct receives exactly the same learned classes.
Existing external CUDA isolation, <=900s external /850s internal required.
MAIN TOPOLOGY human_only; Transformer primary; M0 not passed; M3 ultimate.
"""
import argparse
from collections import OrderedDict
import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import time
import types

import numpy as np
import torch
from torch.nn import functional as F

from continuous_monodromy import ConfigurationLoop, lift_sequence
from diagnose_projection_numeric import restore_parameter_layouts
from homotopy_lift_cache import HomotopyLiftCache
from integrate_native_repair import (Recorder, source_model, evaluate, sha256,
                                     synchronize, tensor_bytes, MAX_FILE_BYTES)
from observed_operation_formation import read_dataset, ALPHABETS, SPLITS

BACKENDS = ('legacy', 'certified', 'cached_direct')
STEPS, BATCH, CAPACITY = 200, 64, 128


def state_hash(model):
    digest = hashlib.sha256()
    for name, p in model.state_dict().items():
        q = p.detach().cpu().contiguous()
        digest.update(name.encode()+q.numpy().tobytes())
    return digest.hexdigest()


class DirectCache:
    def __init__(self):
        self.entries = OrderedDict()
        self.hits = self.misses = self.evictions = 0
        self.seconds = 0.

    def get(self, permutation):
        began = time.perf_counter()
        key = tuple(int(x) for x in permutation)
        if key in self.entries:
            self.hits += 1
            result = self.entries.pop(key)
            self.entries[key] = result
        else:
            self.misses += 1
            if sorted(key) != list(range(len(key))):
                raise ValueError('ordinary map is not a permutation')
            result = np.asarray(key, dtype=np.int64)
            result.setflags(write=False)
            if len(self.entries) == CAPACITY:
                self.entries.popitem(last=False)
                self.evictions += 1
            self.entries[key] = result
        self.seconds += time.perf_counter()-began
        return result

    def statistics(self):
        return dict(entries=len(self.entries), hits=self.hits, misses=self.misses,
                    evictions=self.evictions, seconds=self.seconds,
                    numerical_array_bytes=sum(v.nbytes for v in self.entries.values()),
                    key_serialized_bytes=sum(len(json.dumps(k).encode()) for k in self.entries),
                    scope='same learned permutations; cached ordinary executable maps; no trajectory lifting')


def direct_lift_map(model, permutation, omega, deadline):
    # Execution shares actual learned encoding, calls and native bank extraction.
    # It eliminates only geometric realization; no target/teacher answer is used.
    key = tuple(int(x) for x in permutation)
    if key in model._lifted_maps:
        model._geometry_costs['hard_cache_class_hits'] += 1
        return model._lifted_maps[key]
    result = model._study_direct_cache.get(permutation)
    model._lifted_maps[key] = result
    count = sum(x.nbytes for x in model._lifted_maps.values())
    model._geometry_costs['map_cache_bytes_current'] = count
    model._geometry_costs['map_cache_bytes_peak'] = max(count, model._geometry_costs['map_cache_bytes_peak'])
    return result


def configure(model, backend, family_cache, direct_cache):
    model.set_homotopy_lift_cache(CAPACITY if backend == 'certified' else 0,
                                  shared_cache=family_cache if backend == 'certified' else None)
    if backend == 'cached_direct':
        model._study_direct_cache = direct_cache
        model._lift_map = types.MethodType(direct_lift_map, model)
    model._invalidate_derived_caches()
    return model


def rng_restore(source):
    torch.set_rng_state(source['torch_rng'])
    np.random.set_state(source['numpy_rng'])
    random.setstate(source['python_rng'])
    if source.get('cuda_rng'):
        torch.cuda.set_rng_state_all(source['cuda_rng'])


def checkpoint(model, optimizer, source, out, backend, stage, indices_sha):
    path = out/f'{backend}-growthstage{stage}-real-maintenance.pt'
    payload = dict(schema='terminal-architecture-checkpoint-v1', args=copy.deepcopy(source['args']),
        model_name='topo', stage=stage, step=int(source['step'])+STEPS,
        model=model.state_dict(), optimizer=optimizer.state_dict(), model_extra=model.extra_state_dict(),
        masks_and_initialized_tasks={}, torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
        python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all(),
        growth_cost_diagnostic=dict(backend=backend, additional_updates=STEPS,
            no_new_capability_acquisition_claim=True, batch_indices_sha256=indices_sha,
            learned_growth_checkpoint_source_step=source['step'], caches_not_persisted=True),
        M0='NOT_PASSED')
    torch.save(payload, path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise RuntimeError('checkpoint exceeds100MiB')
    return path


def run_arm(stage, backend, source, data, indices, device, root, cache, direct_cache):
    root.check()
    out = root.out/f'stage{stage}-{backend}'
    rec = Recorder(out, max(.001, root.deadline-time.perf_counter()))
    began = time.perf_counter()
    model = source_model(source, device)
    layouts = restore_parameter_layouts(model, source['model'])
    model.set_stage(stage)
    model.protection = True
    configure(model, backend, cache, direct_cache)
    model.set_geometry_deadline(time.monotonic()+max(.001, root.deadline-time.perf_counter()))
    rng_restore(source)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    initial_hash = state_hash(model)
    initial_geometry = model.geometry_resource_report()
    initial_direct_cache = copy.deepcopy(direct_cache.statistics())
    initial_omega = model.path_omega.detach().cpu().clone()
    old = copy.deepcopy(model.reference)
    costs = dict(train_seconds=0., guard_seconds=0., evaluation_seconds=0.,
        evaluation_soft_episodes=0, evaluation_native_transition_uses=0,
        train_episodes=0, native_monitor_seconds=0., guard_full=0, guard_backtracked=0,
        guard_rejected=0, final_independent_actual_lift_audit_seconds=0.)
    seen = {t:data[t] for t in range(stage+1)}
    alphabets = {t:ALPHABETS[t] for t in seen}
    curves, hashes = [], []
    rec.emit('configuration', stage=stage, backend=backend, source_step=source['step'],
        parameters=sum(p.numel() for p in model.parameters()), parameter_layouts=layouts,
        source_parameter_hash=initial_hash, exact_source_RNG_restored=True,
        no_new_capability_acquisition_claim=True, scope='maintenance at actual learned growth states',
        topology_family_reuse='ideal operation semantics certified by fixed-endpoint shear homotopy',
        cached_reuse_not_claimed_new_numerical_trajectory_or_free_physical_execution=True)
    for step in range(STEPS+1):
        root.check(); rec.check()
        if step in (0,100,200):
            metrics = evaluate(model, seen, alphabets, device, BATCH, rec, costs)
            row = dict(step=step, metrics=metrics, resources=model.geometry_resource_report(),
                       direct_cache=direct_cache.statistics() if backend=='cached_direct' else None,
                       costs=copy.deepcopy(costs), omega=model.path_omega.detach().cpu().tolist())
            curves.append(row); rec.emit('curve', **row)
        if step == STEPS:
            break
        batch = {k:torch.as_tensor(v[indices[step]], device=device, dtype=torch.long)
                 for k,v in data[stage]['train'].items()}
        synchronize(device); started=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        terminal=F.cross_entropy(model(batch['tokens'],batch['starts'],batch['task_ids']),batch['targets'])
        geometry=model.geometry_regularizer()
        loss=terminal+.01*geometry
        if not torch.isfinite(loss): raise FloatingPointError('nonfinite actual learning objective')
        loss.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        optimizer.step(); synchronize(device)
        costs['train_seconds']+=time.perf_counter()-started
        started=time.perf_counter(); guard=model.project_update(before); synchronize(device)
        costs['guard_seconds']+=time.perf_counter()-started
        alpha=float(guard.get('alpha',1.))
        costs['guard_full' if alpha==1 else 'guard_backtracked' if alpha>0 else 'guard_rejected']+=1
        costs['train_episodes']+=BATCH
        hashes.append(state_hash(model))
        rec.emit('actual_update', step=step+1, terminal_CE=float(terminal.item()),
                 geometry_action=float(geometry.item()), objective=float(loss.item()),
                 shared_omega=model.path_omega, guard=guard, parameter_sha256=hashes[-1])
        if (step+1)%20==0:
            started=time.perf_counter()
            with torch.no_grad():
                for t in seen:
                    b={k:torch.as_tensor(v[:16],device=device,dtype=torch.long) for k,v in data[t]['valid'].items()}
                    actual=model.hard_eval(b['tokens'],b['starts'],b['task_ids'])
                    if not torch.isfinite(actual).all(): raise FloatingPointError('nonfinite native monitor')
            synchronize(device); costs['native_monitor_seconds']+=time.perf_counter()-started
    checks=[]
    for g in old:
        okay,pairs=model.related(g,model.graph(g['task'],g['tokens']))
        checks.append(dict(task=g['task'],related=bool(okay),pairs=pairs))
        if not okay: raise RuntimeError('maintenance violated an admitted ability')
    # One independently executed fresh final trajectory, even for cached/direct
    # arms. This costs real work and is reported separately from online reuse.
    audit_started=time.perf_counter()
    g=model.graph(stage,ALPHABETS[stage]); p=g['permutations'][int(g['routes'][0,0])]
    loop=ConfigurationLoop(p,model.path_omega.detach().cpu().double().numpy())
    audit=lift_sequence([loop],min(time.monotonic()+110, time.monotonic()+max(.001,root.deadline-time.perf_counter())))
    if not np.array_equal(np.asarray(audit['monodromy']),p): raise RuntimeError('fresh final trajectory audit mismatch')
    costs['final_independent_actual_lift_audit_seconds']=time.perf_counter()-audit_started
    rec.emit('fresh_final_actual_trajectory_audit', certificate=loop.certificate(),result=audit,
             not_claimed_full_library_or_all_omega_numerical_verification=True)
    indices_sha=hashlib.sha256(indices.tobytes()).hexdigest()
    path=checkpoint(model,optimizer,source,out,backend,stage,indices_sha)
    final_hash=state_hash(model)
    reloaded=source_model(torch.load(path,map_location='cpu',weights_only=False),device)
    restore_parameter_layouts(reloaded,model.state_dict())
    configure(reloaded,backend,HomotopyLiftCache(CAPACITY),DirectCache())
    reloaded.set_geometry_deadline(time.monotonic()+max(.001,root.deadline-time.perf_counter()))
    if state_hash(reloaded)!=final_hash: raise RuntimeError('real neural checkpoint failed exact reload')
    checkdata=data[stage]['valid']
    b={k:torch.as_tensor(v[:64],device=device,dtype=torch.long) for k,v in checkdata.items()}
    with torch.no_grad():
        if not torch.equal(model.hard_eval(b['tokens'],b['starts'],b['task_ids']),reloaded.hard_eval(b['tokens'],b['starts'],b['task_ids'])):
            raise RuntimeError('fresh-cache actual checkpoint outputs differ')
    result=dict(stage=stage,backend=backend,status='200_ACTUAL_UPDATES_COMPLETE',
        parameters=sum(p.numel() for p in model.parameters()),parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        optimizer_bytes=tensor_bytes(optimizer.state_dict()),reference_bytes=model.representation_state_bytes(),
        initial_source_hash=initial_hash,final_hash=final_hash,post_update_hashes=hashes,
        actual_shared_omega_change=float((model.path_omega.detach().cpu()-initial_omega).abs().max()),
        costs=costs,curves=curves,initial_geometry=initial_geometry,final_geometry=model.geometry_resource_report(),
        direct_cache=direct_cache.statistics() if backend=='cached_direct' else None,
        initial_direct_cache=initial_direct_cache if backend=='cached_direct' else None,
        checkpoint=path.name,checkpoint_bytes=path.stat().st_size,checkpoint_sha256=sha256(path),
        actual_checkpoint_reload_verified=True,old_contract_checks=checks,
        arm_wall_seconds=time.perf_counter()-began,M0='NOT_PASSED')
    rec.json('summary.json',result);rec.stream.close()
    del model,reloaded,optimizer;gc.collect();torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for t in range(3):p.add_argument(f'--checkpoint{t}',type=Path,required=True)
    p.add_argument('--datasets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=850)
    p.add_argument('--device',choices=('cuda:0',),default='cuda:0')
    args=p.parse_args()
    if args.out.exists() or not 0<args.max_seconds<=850:p.error('new output;<=850s internal required')
    root=Recorder(args.out,args.max_seconds); device=torch.device(args.device)
    torch.set_num_threads(1)
    if not torch.cuda.is_available():raise RuntimeError('GPU required; no silent fallback')
    torch.cuda.set_per_process_memory_fraction(.3,device)
    data={};sources=[]
    for t in range(3):
        data[t]={}
        for split in SPLITS:
            data[t][split],source=read_dataset(args.datasets,t,split);sources.append(source)
    checkpoints=[]
    for t in range(3):
        path=getattr(args,f'checkpoint{t}')
        if not path.is_file() or path.stat().st_size>MAX_FILE_BYTES:raise ValueError('trusted bounded checkpoint required')
        c=torch.load(path,map_location='cpu',weights_only=False)
        if c.get('model_name')!='topo' or c.get('stage')!=t or 'path_omega' not in c['model']:raise ValueError('actual continuous growth checkpoint required')
        checkpoints.append(c)
    cache,dcache=HomotopyLiftCache(CAPACITY),DirectCache()
    results=[]
    root.emit('configuration',MAIN='TOPOLOGY_USER_LOCKED',M0='NOT_PASSED',M3='ultimate',
        primary_baseline='Transformer',scope='three real learned growth states; paired maintenance costs, not fresh acquisition',
        original_capability_acquisition_costs_in_separate_success_lineage_ledger=True,
        source_datasets=sources,checkpoints=[dict(stage=t,path=str(getattr(args,f'checkpoint{t}')),sha256=sha256(getattr(args,f'checkpoint{t}'))) for t in range(3)])
    try:
        for t in range(3):
            indices=np.random.default_rng(9020+t).integers(0,4096,size=(STEPS,BATCH),dtype=np.int64)
            np.savez_compressed(root.out/f'stage{t}-common-indices.npz',indices=indices)
            order=BACKENDS[t:]+BACKENDS[:t]
            for backend in order:
                result=run_arm(t,backend,checkpoints[t],data,indices,device,root,cache,dcache)
                results.append(result)
                root.json('partial-summary.json',dict(results=results,M0='NOT_PASSED'))
        comparisons=[]
        for t in range(3):
            arms=[r for r in results if r['stage']==t]
            equality=all(arms[0]['post_update_hashes']==r['post_update_hashes'] for r in arms[1:])
            if not equality:raise RuntimeError('execution-only change altered actual training trajectory')
            comparisons.append(dict(stage=t,all200_update_parameter_hashes_identical=True,
                final_parameter_count_equal=len({a['parameters'] for a in arms})==1))
        root.json('summary.json',dict(status='THREE_GROWTH_STATE_COST_DIAGNOSTIC_COMPLETE',results=results,
            paired_checks=comparisons,M0='NOT_PASSED',wall_seconds=time.perf_counter()-root.started,
            no_fresh_three_stage_acquisition_claim=True,
            no_hardware_energy_or_asymptotic_advantage_claim=True))
    except BaseException as error:
        root.emit('run_failed',error=type(error).__name__,reason=str(error),completed_arms=len(results))
        root.json('failure.json',dict(results=results,error=type(error).__name__,reason=str(error),M0='NOT_PASSED'))
        raise
    finally:root.stream.close()


if __name__=='__main__':main()
