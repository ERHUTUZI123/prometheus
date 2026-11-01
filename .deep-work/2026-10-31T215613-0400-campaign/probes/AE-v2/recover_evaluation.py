"""Evaluate saved AE-v2 policies only. Never call Policy.update or train."""
from pathlib import Path
import time
HERE=Path(__file__).resolve().parent
text=(HERE/'acquire.py').read_text()
# Reuse inspected pure mechanism/policy definitions, excluding all original execution loops.
prefix=text.split('# Reference model is evaluator-only.')[0]
policy_defs=text.split('def features(',1)[1].split('# Finite difference score check',1)[0]
exec(compile(prefix+'\ndef features('+policy_defs,str(HERE/'acquire.py')+'[definitions]','exec'))
saved=np.load(OUT/'reference.npz')
TASK=saved['task'];WORK=saved['work'];REWARD=saved['reward']
assert np.array_equal(saved['configs'],ALL)
assert np.array_equal(saved['contexts'],np.array(CONTEXTS))
physics=simulate(np.array([[1,1,1,0,0,0,1],[1,1,1,0,0,0,1],
    [1,1,1,1,4,4,2],[1,1,1,1,4,4,2],[1,1,2,2,2,2,2],
    [1,1,2,2,2,2,2],[0,1,1,1,1,1,2],[1,0,1,1,1,1,2]]),
    np.array([10.,0.,10.,10.,4.,4.,1.,1.]))
check_code=text.split('# Fixed nontrivial checks:',1)[1].split('def features(',1)[0]
exec(compile('# Fixed nontrivial checks:'+check_code,str(HERE/'acquire.py')+'[checks]','exec'))
gradient_code=text.split('# Finite difference score check',1)[1].split('results=[]',1)[0]
exec(compile('# Finite difference score check'+gradient_code,str(HERE/'acquire.py')+'[gradient check]','exec'))

results=[]
for linear in [False,True]:
    for seed in [631,632,633]:
        initial=Policy(seed,linear);before=evaluate(initial)
        p=Policy(seed+10000,linear)
        weights=np.load(OUT/f'policy_{"linear" if linear else "neural"}_{seed}.npz')
        p.params=[weights[f'p{i}'].copy() for i in range(len(p.params))]
        after=evaluate(p);load_blind=evaluate(p,blind=True)
        results.append({'kind':'linear' if linear else 'neural','seed':seed,
          'frozen_evaluation_seed':seed+10000,'before':before,'after':after,'load_blind':load_blind,
          'training_installations_code_budget':2000,'training_logical_assays_code_budget':10000,
          'training_curves':'UNAVAILABLE: final JSON serialization failed',
          'training_material':'UNAVAILABLE','weighted_training_work':'UNAVAILABLE',
          'unique_reward_queries':'UNAVAILABLE','parameters':sum(a.size for a in p.params)})

oracle_code=text.split('oracle=[]',1)[1].split('neural=[r for r in results',1)[0]
exec(compile('oracle=[]'+oracle_code,str(HERE/'acquire.py')+'[ordinary allocator replay]','exec'))
neural=[r for r in results if r['kind']=='neural']
improvements=[float(np.mean([v['reward'] for v in r['after'][:9]])-np.mean([v['reward'] for v in r['before'][:9]])) for r in neural]
testscore=float(np.mean([v['task'] for r in neural for v in r['after'][9:]]))
summary={'neural_train_reward_improvements':improvements,'neural_mean_heldout_task':testscore,
  'partial_acquisition_criterion':bool(sanity['passed'] and min(improvements)>=.05 and testscore>=.75)}
res={'version':'AE-v2 evaluation-only recovery','first_execution_status':'all policy files saved; final JSON serialization failed',
  'training_measurement_status':'per-episode curves/costs/unique queries unavailable',
  'frozen_policy_evaluation_status':'new fixed replay, no weight updates',
  'sanity':sanity,'checks':checks,'policies':results,'ordinary_random_allocator_static_replay':ordinary,
  'exhaustive_allocator':oracle,'reference_trajectories_static_count':len(ALL)*len(CONTEXTS)*5,
  'summary':summary,'recovery_runtime_seconds':time.perf_counter()-START}
(OUT/'results_recovered.json').write_text(json.dumps(res,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n')
print(json.dumps({'sanity':sanity,'summary':summary,'recovery_runtime_seconds':res['recovery_runtime_seconds']},indent=2,default=lambda x:x.item()))
