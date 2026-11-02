"""Bounded execution/reload check for the N32 -> D64 path backend; no training.

This tests a prerequisite for the requested scale experiment. It measures the
actual multi-loop lift and quadrature; factorization is not an acquisition rule.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from factored_monodromy_cache import (FactoredMonodromyCache,
                                     factor_permutation, factored_kinetic_moments)
from parameter_native_models import (CompactContinuousNativeModel,
                                     compact_model_from_checkpoint)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--max-seconds',type=float,default=110.)
    args=p.parse_args()
    if args.out.exists() or not 0<args.max_seconds<=110: p.error('new bounded output required')
    args.out.mkdir(parents=True)
    deadline=time.monotonic()+args.max_seconds
    torch.set_num_threads(1);torch.manual_seed(0)
    rng=np.random.default_rng(3817)
    plans=[]
    for dimension in (8,16,32,64):
        for q in [np.arange(dimension),np.arange(dimension)[::-1]]+[
                rng.permutation(dimension) for _ in range(8)]:
            plan=factor_permutation(q)
            plans.append({k:plan[k] for k in ['dimension','factor_count',
                'full_adjacent_exchanges','canonical_chunk_lengths','composition_verified']})
    cache=FactoredMonodromyCache(32)
    requested=np.arange(64,dtype=np.int64)[::-1]
    mapping,cold=cache.get(requested,[0.]*4,deadline)
    if not np.array_equal(mapping,requested):raise RuntimeError('cold lift wrong')
    mapping,warm=cache.get(requested,[.1,-.2,.3,-.4],deadline)
    if not np.array_equal(mapping,requested) or not warm['cache_hit']:
        raise RuntimeError('same-family ideal reuse failed')
    start=time.perf_counter()
    moments=factored_kinetic_moments([requested])
    moments_seconds=time.perf_counter()-start
    model=CompactContinuousNativeModel(32,latent_factor=2,seed=0,factored_monodromy=True,
                                      homotopy_lift_cache_capacity=32)
    model.stage=0
    loss=model.geometry_regularizer()
    loss.backward()
    if model.path_omega.grad is None or not torch.isfinite(model.path_omega.grad).all():
        raise RuntimeError('factored geometry gradient missing/nonfinite')
    payload={'model':model.state_dict(),'model_extra':model.extra_state_dict()}
    path=args.out/'actual-model.pt';torch.save(payload,path)
    actual=torch.load(path,map_location='cpu',weights_only=False)
    restored=compact_model_from_checkpoint(actual)
    for name,value in model.state_dict().items():
        if not torch.equal(value,restored.state_dict()[name]):raise RuntimeError('checkpoint weights changed')
    if not restored.factored_monodromy:raise RuntimeError('checkpoint lost factor policy')
    if model._homotopy_lift_cache.family_fingerprint!=restored._homotopy_lift_cache.family_fingerprint:
        raise RuntimeError('checkpoint namespace changed')
    if time.monotonic()>=deadline:raise TimeoutError('diagnostic wall cap')
    report=dict(scope='N32 backend execution and actual NN reload prerequisite; no task learning',
        factor_plans=plans,cold=cold,warm=warm,actual_cache_statistics=cache.statistics(),
        quadrature={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in moments.items()},
        quadrature_seconds=moments_seconds,actual_saved_checkpoint_bytes=path.stat().st_size,
        actual_nn_bits_reloaded=True,actual_geometry_objective=float(loss.detach()),
        actual_geometry_gradient=model.path_omega.grad.tolist(),
        resources=model.compact_resource_counts(),new_training_updates=0)
    (args.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed',composition_checks=len(plans),
        actual_cold_factor_count=cold['certificate']['factorization']['factor_count'],
        actual_lift_seconds=cold['actual_lift_seconds'],nn_reloaded=True)))


if __name__=='__main__':main()
