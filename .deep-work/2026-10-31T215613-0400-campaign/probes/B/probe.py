"""Bounded branch-B mechanism check. No network, shell execution, or parameter search.

Frozen checks: N=256 observed examples/module; 64 held-out cases/condition;
one concentration threshold .8; fixed ODE horizon 60; seed 314159.
Includes a newly derived central attack: a four-factor chain plus its entailed
shortcut has a stable incorrect winding state. This supplements, never rewrites,
the frozen v0's explicit distinction between local stiffness and global dynamics.
"""
from __future__ import annotations
import itertools, json, math, hashlib, platform
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent
RNG=np.random.default_rng(314159)
NTRAIN=256
NQUERY=64
THRESH=.8
SIGMA=.05
HORIZON=60.

def angle(x):
    return np.angle(np.exp(1j*np.asarray(x)))

def dictionary(n=4):
    rows=[]
    for row in itertools.product((-1,0,1),repeat=n):
        if sum(row) or not any(row):
            continue
        if next(v for v in row if v) < 0:
            continue
        rows.append(row)
    return np.asarray(rows,dtype=int)

def observations(delta,n):
    # Order is r, x, y, z, with global phase and both input differences random.
    r,a,b=RNG.uniform(-np.pi,np.pi,(3,n))
    z=r+a+b+delta+RNG.normal(0,SIGMA,n)
    return np.stack([r,r+a,r+b,z],axis=1)

def update_trace(samples,rows):
    # Execute the actual online sufficient-statistic update, not only np.mean.
    q=np.zeros(len(rows),dtype=complex)
    for k,theta in enumerate(samples,1):
        q+=(np.exp(1j*(rows@theta))-q)/k
    return q

def learn(delta):
    rows=dictionary()
    train=observations(delta,NTRAIN)
    val=observations(delta,NQUERY)
    q=update_trace(train,rows)
    qv=update_trace(val,rows)
    selected=np.flatnonzero((abs(q)>=THRESH)&(abs(qv)>=THRESH))
    shuffled=train.copy()
    shuffled[:,3]=shuffled[RNG.permutation(NTRAIN),3]
    qs=update_trace(shuffled,rows)
    assert len(selected)==1, (rows[selected],abs(q[selected]))
    assert np.array_equal(rows[selected[0]],[1,-1,-1,1])
    assert np.max(abs(qs))<THRESH
    rec={'dictionary_size':len(rows),'selected':rows[selected].tolist(),
         'rho_train':abs(q[selected]).tolist(),'rho_validation':abs(qv[selected]).tolist(),
         'delta_true':delta,'delta_learned':float(np.angle(q[selected[0]])),
         'max_unselected_rho':float(np.max(np.delete(abs(q),selected))),
         'max_shuffled_rho':float(np.max(abs(qs))),
         'expected_population_rho':float(np.exp(-SIGMA**2/2))}
    return rows[selected[0]],rec['delta_learned'],rec

def field(theta,C,delta,K,free):
    return (-C.T@(K*np.sin(C@theta-delta)))[free]

def integrate(C,delta,K,theta,free,target):
    theta=theta.copy()
    def fun(t,y):
        state=theta.copy()
        state[free]=y
        return field(state,C,delta,K,free)
    sol=solve_ivp(fun,(0,HORIZON),theta[free],rtol=1e-8,atol=1e-10,
                  max_step=.25,t_eval=np.linspace(0,HORIZON,1201))
    assert sol.success
    errors=np.max(abs(angle(sol.y.T-target)),axis=1)
    outside=np.flatnonzero(errors>1e-3)
    settle=float(sol.t[outside[-1]+1]) if len(outside) and outside[-1]<len(sol.t)-1 else (0. if not len(outside) else None)
    states=np.tile(theta,(len(sol.t),1))
    states[:,free]=sol.y.T
    energies=-(np.cos(states@C.T-delta)*K).sum(axis=1)
    return sol.y[:,-1],{'final_learned_solution_error':float(errors[-1]),
                         'settle_time_to_1e-3':settle,
                         'max_energy_increase':float(np.max(np.diff(energies)))}

rowA,da,learnA=learn(.43)
rowB,db,learnB=learn(-.31)
CA=np.zeros(6,dtype=int); CA[[0,1,2,3]]=rowA
CB=np.zeros(6,dtype=int); CB[[0,3,4,5]]=rowB
C=np.stack([CA,CB])
# The growth operation actually constructs the row; it is not entered by hand.
shared=np.flatnonzero((CA*CB)==-1)
assert shared.tolist()==[3]
merged=CA+CB
assert merged[3]==0 and merged.sum()==0
CC=np.vstack([C,merged])
deltas=np.array([da,db])
deltac=np.array([da,db,da+db])

records={key:[] for key in ['forward','inverse','composition','shortcut_more_budget','shortcut_matched_proxy','removed_A']}
for query in range(NQUERY):
    theta=RNG.uniform(-np.pi,np.pi,6)
    r,x,y,_,w,_=theta
    z=x+y-r+da
    v=z+w-r+db
    # Analytic oracle is separate from numerical energy/field calculation.
    target=np.array([z,v])
    for name,rows,ds,ks in [
        ('composition',C,deltas,np.array([1.,1.])),
        ('shortcut_more_budget',CC,deltac,np.array([1.,1.,1.])),
        ('shortcut_matched_proxy',CC,deltac,np.array([.5,.5,.5])),
        ('removed_A',C[1:],deltas[1:],np.array([1.]))]:
        end,rec=integrate(rows,ds,ks,theta,np.array([3,5]),target)
        rec['ground_truth_output_error']=float(abs(angle(end[1]-(x+y+w-2*r+.43-.31))))
        records[name].append(rec)
    end,rec=integrate(C[:1],deltas[:1],np.array([1.]),theta,np.array([3]),np.array([z]))
    rec['ground_truth_output_error']=float(abs(angle(end[0]-(x+y-r+.43))))
    records['forward'].append(rec)
    inv=theta.copy(); inv[3]=z; inv[1]=RNG.uniform(-np.pi,np.pi)
    end,rec=integrate(C[:1],deltas[:1],np.array([1.]),inv,np.array([1]),np.array([x]))
    rec['ground_truth_output_error']=float(abs(angle(end[0]-x)))
    records['inverse'].append(rec)

summary={}
for name,recs in records.items():
    times=[r['settle_time_to_1e-3'] for r in recs if r['settle_time_to_1e-3'] is not None]
    summary[name]={'n':len(recs),
        'max_learned_solution_error':max(r['final_learned_solution_error'] for r in recs),
        'mean_ground_truth_output_error':float(np.mean([r['ground_truth_output_error'] for r in recs])),
        'max_ground_truth_output_error':max(r['ground_truth_output_error'] for r in recs),
        'settled_by_horizon':len(times),
        'median_settle_time':float(np.median(times)) if times else None,
        'max_energy_increase':max(r['max_energy_increase'] for r in recs)}
for name in records:
    if name!='removed_A':
        assert summary[name]['max_learned_solution_error']<1e-3,summary[name]
        assert summary[name]['max_ground_truth_output_error']<.03,summary[name]
        assert summary[name]['max_energy_increase']<1e-7,summary[name]
assert summary['removed_A']['mean_ground_truth_output_error']>.5

def jacobian_check(rows,ds,ks):
    theta=np.array([0.,.7,-.4,.7-.4+da,1.2,.7-.4+da+1.2+db])
    free=np.array([3,5])
    analytic=-rows[:,free].T@np.diag(ks)@rows[:,free]
    eps=1e-6
    numeric=np.zeros((2,2))
    for k,j in enumerate(free):
        plus=theta.copy();plus[j]+=eps
        minus=theta.copy();minus[j]-=eps
        numeric[:,k]=(field(plus,rows,ds,ks,free)-field(minus,rows,ds,ks,free))/(2*eps)
    assert np.max(abs(numeric-analytic))<1e-7
    return {'jacobian':analytic.tolist(),'eigenvalues':np.linalg.eigvalsh(analytic).tolist(),
        'finite_difference_max_discrepancy':float(np.max(abs(numeric-analytic))),
        'full_port_squared_coefficient_budget':float(np.sum(ks*np.sum(rows**2,axis=1)))}
jacs={'parents':jacobian_check(C,deltas,np.array([1.,1.])),
      'shortcut_more_budget':jacobian_check(CC,deltac,np.ones(3)),
      'shortcut_matched_proxy':jacobian_check(CC,deltac,np.full(3,.5))}

# Serious adversarial check: four-edge parent chain plus entailed final shortcut.
# Work in deviations from any compatible world solution: parents e_i=y_i-y_i-1,
# y_0=0. The direct shortcut error is y_4, so all zero is the correct answer.
L=4
D=np.eye(L)
for i in range(1,L):D[i,i-1]=-1
shortcut=np.zeros(L);shortcut[-1]=1
A=np.vstack([D,shortcut])
alpha=2*np.pi/(L+1)
winding=np.arange(1,L+1)*alpha
grad=A.T@np.sin(A@winding)
H=A.T@np.diag(np.cos(A@winding))@A
assert np.max(abs(grad))<1e-12
assert np.min(np.linalg.eigvalsh(H))>.1
near=winding+np.array([.02,-.01,.015,-.005])
attack_sol=solve_ivp(lambda t,u:-A.T@np.sin(A@u),(0,HORIZON),near,
                    rtol=1e-9,atol=1e-11,max_step=.25)
attack_end=attack_sol.y[:,-1]
assert np.max(abs(angle(attack_end-winding)))<1e-4
assert abs(angle(attack_end[-1]))>1.
attack={'parent_count':L,'shortcut_is_sum_of_parent_rows':True,
        'winding_state':winding.tolist(),'parent_residuals':(D@winding).tolist(),
        'shortcut_residual':float(winding[-1]),
        'gradient_max':float(np.max(abs(grad))),'hessian_eigenvalues':np.linalg.eigvalsh(H).tolist(),
        'settled_near_wrong_state_max_error':float(np.max(abs(angle(attack_end-winding)))),
        'wrong_output_phase_error':float(abs(angle(attack_end[-1]))),
        'interpretation':'Entailed positive shortcut can introduce stable incorrect minima; local-stiffness improvement is not global computational improvement.'}

result={'configuration':{'seed':314159,'N_train_per_module':NTRAIN,'N_validation_per_module':NQUERY,
         'N_queries_per_condition':NQUERY,'threshold':THRESH,'observation_sigma':SIGMA,
         'fixed_ODE_horizon':HORIZON,'numpy':np.__version__,'platform':platform.platform()},
        'learning':{'A':learnA,'B':learnB},'constructed_rows':{'A':CA.tolist(),'B':CB.tolist(),'shortcut':merged.tolist()},
        'conditions':summary,'jacobians':jacs,'central_attack':attack,
        'checks':'All explicit assertions passed. This is program-level verification of the stated model, not independent scientific validation.'}
(HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
(HERE/'query-records.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(result,indent=2))
