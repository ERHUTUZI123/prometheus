"""Bounded offline probe. No network, subprocesses, eval, or host interaction.
Execute: infrastructure/venv/bin/python probes/F/probe.py
The analytic transport and RK4 implementations are separate computational paths.
"""
from pathlib import Path
import itertools
import json
import platform
import time
import numpy as np

BASE = Path(__file__).resolve().parent
CFG = json.loads((BASE / 'config.json').read_text())
I = np.eye(3)
LOC = [(1,0), (2,0), (2,1)]
Astar = I.copy(); Astar[1,0] = 1
Bstar = I.copy(); Bstar[2,1] = 1
Cstar = I.copy(); Cstar[2,0] = 1
PATHS = [np.array([[0,0],[0,1],[2,1],[2,0],[0,0]],dtype=float),
         np.array([[0,0],[2,0],[2,1],[4,1],[4,0],[0,0]],dtype=float)]
NAMES = ['A','A^-1','B','B^-1']


def mat(v):
    m = np.zeros((3,3))
    for z, ij in zip(v, LOC): m[ij] = z
    return m


def exact_flow(m, scale=1.0):
    # Exact exp(scale*m), since every 3x3 strict-lower matrix has m^3=0.
    return I + scale*m + (scale*scale/2)*(m @ m)


def actions(theta):
    m0,m1 = mat(theta[:3]), mat(theta[3:])
    return [exact_flow(m0), exact_flow(m0,-1),
            exact_flow(m1), exact_flow(m1,-1)]


def loss(theta):
    aa = actions(theta)
    return float(0.5*(np.sum((aa[0]-Astar)**2)+np.sum((aa[2]-Bstar)**2)))


def train(seed):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,CFG['initialization_sigma'],6)
    history=[]; evals=0
    for epoch in range(CFG['max_epochs']):
        center=loss(theta); evals+=1
        if epoch % 25 == 0: history.append([epoch, center])
        if center < CFG['stop_training_loss']: break
        grad=np.zeros(6)
        for j in range(6):
            d=np.zeros(6); d[j]=CFG['finite_difference_delta']
            grad[j]=(loss(theta+d)-loss(theta-d))/(2*CFG['finite_difference_delta'])
            evals+=2
        theta=np.clip(theta-CFG['learning_rate']*grad,
                      -CFG['parameter_clip'],CFG['parameter_clip'])
    final=loss(theta); evals+=1
    return theta, {'epochs':epoch+1,'final_loss':final,'loss_evaluations':evals,
                   'primitive_loop_basis_executions':6*evals,'history':history,
                   'weights':theta.tolist()}


def word_matrix(word, aa):
    h=I.copy()
    for a in word: h=aa[a] @ h
    return h


def reduced_words(maxlen):
    for n in range(1,maxlen+1):
        for w in itertools.product(range(4), repeat=n):
            if all(w[j] != (w[j-1]^1) for j in range(1,n)):
                yield w


def search(aa):
    trace=[]; first=None
    for w in reduced_words(CFG['max_reduced_word_length']):
        err=float(np.max(np.abs(word_matrix(w,aa)-Cstar)))
        record={'word':[NAMES[a] for a in w], 'length':len(w), 'max_basis_error':err}
        trace.append(record)
        if first is None and err < CFG['acceptance_max_basis_error']:
            first=(w,len(trace),record)
    return first,trace


def hats(x):
    return [max(0.,1-abs(x-1)),max(0.,1-abs(x-3))]


def rk4_path(theta, path, h):
    # Independent numeric implementation of dq=u and dh=u_x*y*phi*M*h.
    m0,m1=mat(theta[:3]),mat(theta[3:]); total_steps=0
    state=np.concatenate([path[0],h])
    for p0,p1 in zip(path[:-1],path[1:]):
        delta=p1-p0; dist=float(np.linalg.norm(delta))
        if dist==0: continue
        steps=max(1,int(np.ceil(dist*CFG['rk4_steps_per_unit_distance'])))
        dt=1./steps
        def rhs(s):
            x,y=s[:2]; f0,f1=hats(x)
            dh=delta[0]*y*((f0*m0+f1*m1) @ s[2:])
            return np.concatenate([delta,dh])
        for _ in range(steps):
            k1=rhs(state); k2=rhs(state+dt*k1/2)
            k3=rhs(state+dt*k2/2); k4=rhs(state+dt*k3)
            state=state+dt*(k1+2*k2+2*k3+k4)/6
        total_steps+=steps
    return state, total_steps


def stability(theta,word,seed):
    rng=np.random.default_rng(12000+seed)
    out=[]
    for sigma in CFG['noise_sigmas']:
        for draw in range(CFG['noise_draws_per_sigma_per_seed']):
            noise=rng.normal(0,sigma,6)
            pert=theta+noise; aa=actions(pert)
            central=noise.copy(); central[[0,2,3,5]]=0
            ca=actions(theta+central)
            out.append({'sigma':sigma,'draw':draw,
              'primitive_max_error':max(float(np.max(np.abs(aa[0]-Astar))),
                                        float(np.max(np.abs(aa[2]-Bstar)))),
              'retained_C_max_error':float(np.max(np.abs(word_matrix(word,aa)-Cstar))),
              'central_only_C_error':float(np.max(np.abs(word_matrix(word,ca)-Cstar))),
              'central_only_primitive_error':max(float(np.max(np.abs(ca[0]-Astar))),
                                               float(np.max(np.abs(ca[2]-Bstar))))})
    return out


def main():
    started=time.perf_counter()
    held=np.random.default_rng(CFG['heldout_seed']).uniform(-2,2,(CFG['heldout_count'],3))
    result={'config':CFG,'environment':{'python':platform.python_version(),
             'numpy':np.__version__,'platform':platform.platform()},'seeds':[]}
    numeric=[]
    for seed in CFG['seeds']:
        theta,learning=train(seed)
        first,trace=search(actions(theta))
        rec={'seed':seed,'learning':learning,'search_trace':trace,'accepted':None}
        if first is not None:
            word,index,accepted=first
            hw=word_matrix(word,actions(theta))
            rec['accepted']={**accepted,'first_acceptance_candidate_index':index,
                'heldout_max_error':float(np.max(np.abs((hw@held.T-Cstar@held.T)))),
                'expanded_path_edges':sum(len(PATHS[a//2])-1 for a in word),
                'primitive_loop_actions':len(word),
                'word_indices':list(word),
                'analytic_matrix':hw.tolist()}
            rec['stability']=stability(theta,word,seed)
            # Direct four-loop integration on one data example, then matrix basis.
            vals=[]
            for h0 in [np.array([2.,5.,7.]),*list(I)]:
                h=h0.copy(); states=[]; steps=0
                for a in word:
                    path=PATHS[a//2]
                    if a%2: path=path[::-1]
                    st,n=rk4_path(theta,path,h); h=st[2:]; steps+=n
                    states.append(st.tolist())
                vals.append({'initial_h':h0.tolist(),'final_h':h.tolist(),
                    'analytic_difference':float(np.max(np.abs(h-hw@h0))),
                    'q_closure_max':max(float(np.max(np.abs(np.array(s[:2])))) for s in states),
                    'steps':steps,'intermediate_full_states':states})
            numeric.append({'seed':seed,'runs':vals})
        result['seeds'].append(rec)
    # Degenerate/commuting negative control: both fields act only through E21.
    theta=np.array([1,0,0,0.7,0,0],dtype=float)
    first,trace=search(actions(theta))
    result['negative_control']={'accepted':first is not None,
        'minimum_target_error':min(t['max_basis_error'] for t in trace),
        'evaluated_words':len(trace)}
    result['independent_rk4']=numeric
    result['elapsed_seconds']=time.perf_counter()-started
    (BASE/'results.json').write_text(json.dumps(result,indent=2))
    summary={'seeds':[], 'negative_control':result['negative_control'],
             'elapsed_seconds':result['elapsed_seconds']}
    for rec in result['seeds']:
        a=rec['accepted']
        summary['seeds'].append({'seed':rec['seed'],'epochs':rec['learning']['epochs'],
            'loss':rec['learning']['final_loss'],'accepted':a,
            'tested_reduced_words':len(rec['search_trace'])})
    summary['noise_summary']=[]
    for sigma in CFG['noise_sigmas']:
        rows=[r for s in result['seeds'] for r in s.get('stability',[]) if r['sigma']==sigma]
        summary['noise_summary'].append({'sigma':sigma,
            **{k+'_median':float(np.median([r[k] for r in rows])) for k in
              ['primitive_max_error','retained_C_max_error','central_only_C_error','central_only_primitive_error']}})
    summary['rk4_max_analytic_difference']=max(r['analytic_difference'] for s in numeric for r in s['runs'])
    summary['rk4_max_q_closure_error']=max(r['q_closure_max'] for s in numeric for r in s['runs'])
    (BASE/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
