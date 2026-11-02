"""Owned bounded TODO3 interface diagnostic. No execution at import.

Frame-frozen development variant, not Primary and not shared-plasticity evidence.
All heldout use follows an actual saved neural-state freeze. CPU runner external
120s, driver internal90s; at most100 update transactions. No teacher lift/call.
"""
from __future__ import annotations
import argparse, hashlib, json, random, time, traceback
from pathlib import Path
import torch
from factorized_reset_block import (ResetInterfaceSession, LearnedClassicalChannel,
    CanonicalFactorizedLiftControl, proposed_limits)
from loop_boundary_learning import CostLedger, BudgetStop, NumericalFailure, ProgramExample


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=90)
    p.add_argument('--max-updates',type=int,default=100)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--gradient-samples',type=int,default=1)
    p.add_argument('--controls-result',type=Path)
    a=p.parse_args()
    if not 5<=a.seconds<=90 or not 1<=a.max_updates<=100 or a.gradient_samples not in (1,4):
        raise ValueError('bounded owned diagnostic required')
    a.out.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();deadline=time.monotonic()+a.seconds
    result={'scope':'TODO3 frame-frozen interface development, not Primary/acceptance/topological advantage',
        'MAIN':'TOPOLOGY_USER_LOCKED','seed':a.seed,'source_checkpoint_sha256':digest(a.checkpoint),
        'max_transactions':a.max_updates,'gradient_samples_per_transaction':a.gradient_samples,
        'source_sha256':{q.name:digest(q) for q in Path(__file__).parent.glob('*.py')},
        'status':'started','arms':{},'heldout_used_for_selection':False}
    streams=[]
    def ledger(name):
        f=(a.out/(name+'.jsonl')).open('x');streams.append(f)
        def emit(row):
            f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');f.flush()
        return CostLedger(proposed_limits(),deadline,writer=emit)
    def save(session,rng,name):
        q=a.out/name
        with q.open('xb') as f:
            torch.save({'interface':session.checkpoint(),'new_audit_rng':rng.getstate(),
                'torch_cpu_rng':torch.get_rng_state(),'sources':result['source_sha256'],
                'exact_training_resume_tested':False},f)
        return {'path':str(q),'sha256':digest(q),'bytes':q.stat().st_size}
    try:
        cost=ledger('topological');session=ResetInterfaceSession.from_core_checkpoint(a.checkpoint,cost)
        train,heldout,manifest=session.diagnostic_specification()
        result['prospective_manifest_from_source']=manifest
        if a.controls_result:
            result['reused_controls']={'path':str(a.controls_result),'sha256':digest(a.controls_result),'new_control_training':False,'scope':'identical fixed conventional controls from completed V1, cost remains historical not newly incurred'}
        else:
            # Strong ordinary control sees exactly the same records and old core.
            ccost=ledger('classical');control_session=ResetInterfaceSession.from_core_checkpoint(a.checkpoint,ccost)
            ctr=LearnedClassicalChannel(control_session);learned=ctr.fit(train)
            result['classical_control_freeze']={'learned_channel':list(learned),'source':'TRAIN singleton terminal observations only'}
            direct={'learned_channel':learned,'train':ctr.evaluate(train),'heldout':ctr.evaluate(heldout),
                'storage':'3 learned int64 entries; inherited core and source metadata separately counted'}
            lift=CanonicalFactorizedLiftControl(ctr);pair=lift.compile()
            compiled={'pair_map':pair,'train':lift.evaluate(train),'heldout':lift.evaluate(heldout),
                'cold_actuation':lift.cold_actuation()}
            result['arms']['ordinary_direct_channel']=direct
            result['arms']['ordinary_learned_factorized_lift']=compiled
            result['common_control_harness_allocation']=control_session.parameter_storage()
            result['classical_combined_costs']=ccost.snapshot()
        rng=random.Random(a.seed);torch.manual_seed(a.seed)
        arm={'curve':[],'admitted':False,'maintenance_used':False,'parameter_storage':session.parameter_storage()}
        result['arms']['topological']=arm
        before=session.evaluate(train);arm['initial']=before
        for step in range(a.max_updates):
            if a.gradient_samples==1:
                update=session.step(train,rng)
            else:
                draws=[session.model.gradient(train,rng) for _ in range(a.gradient_samples)]
                grads=[sum(g['total'][i] for g in draws)/a.gradient_samples for i in range(len(draws[0]['total']))]
                update={'averaged_gradient_samples':a.gradient_samples,
                    'records':[d['record'] for d in draws],
                    'transaction':session.model.transaction(grads,.05)}
            evaluation=session.evaluate(train)
            arm['curve'].append({'step':step+1,'update':update,
                'soft_accuracy':evaluation['soft_accuracy'],'native_accuracy':evaluation['native_accuracy'],
                'logical_channel':evaluation['logical_channel'],'losses':[
                    -__import__('math').log(max(1e-12,row['soft_probability'][row['example']['target']]))
                    for row in evaluation['rows']], 'costs':cost.snapshot()})
            # TRAIN-only trigger, fixed before the first evaluation; one normal
            # transaction remains inside the declared total100-update cap.
            if evaluation['soft_accuracy']==evaluation['native_accuracy']==1.:
                if cost.counts.get('updates',0)<a.max_updates and not arm['maintenance_used']:
                    arm['maintenance']=session.maintain_own_call_margin();arm['maintenance_used']=True
                arm['admitted']=session.admit_if_successful(train)
                if arm['admitted']:break
            if (step+1)%10==0:save(session,rng,f'topological-step-{step+1:03d}.pt')
        arm['train']=session.evaluate(train)
        # No open arena is carried through parameter updates in this first
        # diagnostic. Cold evaluation is performed only after this saved freeze.
        arm['frozen_checkpoint']=save(session,rng,'topological-frozen.pt')
        arm['cold_audit']=session.audit_cold_local_maps()
        arm['heldout']=session.evaluate(heldout)
        old=tuple(ProgramExample(f'own-old-{x}-{i}',x,w,
            __import__('hybrid_loop_block').native_output(session.model.reference,
                ProgramExample('reference',x,w,0),cost))
            for x in range(3) for i,w in enumerate(((),(0,),(1,),(0,1),(1,0))))
        arm['old']=session.evaluate(old)
        # Actual disk load; no exact training-resume claim for this new variant.
        payload=torch.load(a.out/'topological-frozen.pt',map_location='cpu',weights_only=True)
        restored=ResetInterfaceSession.restore(payload['interface'],cost)
        reloaded=restored.evaluate(heldout)
        arm['actual_reload']={'native_equal':[
            r['native_output'] for r in reloaded['rows']]==[r['native_output'] for r in arm['heldout']['rows']],
            'soft_equal':all(r['soft_probability']==s['soft_probability']
                for r,s in zip(reloaded['rows'],arm['heldout']['rows'])),
            'storage':restored.parameter_storage()}
        arm['costs']=cost.snapshot()
        if a.controls_result:
            previous=json.loads(a.controls_result.read_text())
            for name in ('ordinary_direct_channel','ordinary_learned_factorized_lift'):
                result['arms'][name]=previous['arms'][name]
            result['historical_classical_combined_costs']=previous['classical_combined_costs']
        result['status']='completed_measured_diagnostic'
    except (BudgetStop,NumericalFailure) as e:
        result['status']='bounded_diagnostic_censored';result['error']=f'{type(e).__name__}: {e}'
        (a.out/'exception.txt').write_text(traceback.format_exc())
    except Exception as e:
        result['status']='implementation_failed';result['error']=f'{type(e).__name__}: {e}'
        (a.out/'exception.txt').write_text(traceback.format_exc())
    finally:
        if 'session' in locals() and result['status']!='completed_measured_diagnostic':
            q=a.out/'partial-neural-tensors.pt'
            with q.open('xb') as f:torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in session.model.state_dict().items()},'scope':'partial actual tensors, not a claimed complete restorable session'},f)
            result['partial_neural_state']={'sha256':digest(q),'bytes':q.stat().st_size}
            result['partial_costs']=cost.snapshot()
        result['wall_seconds']=time.perf_counter()-started
        for f in streams:f.close()
        (a.out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps({'status':result['status'],'error':result.get('error'),
            'topological_acquired':result['arms'].get('topological',{}).get('admitted')}))
    if result['status']=='implementation_failed':raise SystemExit(1)

if __name__=='__main__':main()
