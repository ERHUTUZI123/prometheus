"""Bounded Branch E probe. No network, subprocesses, or host execution.
Execution: microscopic reversible channels, finite-state CTMC.
Learning: terminal scalar reward times local count-minus-hazard trace.
Exact matrix evolution is a verifier, never a training signal.
"""
from __future__ import annotations
import itertools, json, math, time
from pathlib import Path
import numpy as np
from scipy.linalg import expm

OUT=Path(__file__).resolve().parent
EPS=8.0; MU=16.0; GAMMA=.03; ALPHA=4.0; MATERIAL=4.0
# Input source 0=A,1=B, internal source 2=m,3=z.
EDGES=[]
for target, sources in [(0,[0,1,3]),(1,[0,1,2])]:
    for order in [1,2]:
        for pair in itertools.combinations(sources,order): EDGES.append((target,pair))
NEDGE=len(EDGES)
NAMES=['+'.join('ABmz'[s] for s in src)+'->'+'mz'[j] for j,src in EDGES]
CASES=[('AB',[(1,0),(0,0),(0,1)],1),('BA',[(0,1),(0,0),(1,0)],0),
       ('A_only',[(1,0),(0,0),(0,0)],0),('B_only',[(0,0),(0,0),(0,1)],0),
       ('empty',[(0,0),(0,0),(0,0)],0)]
DURATION=np.array([1.,1.,1.])
WEIGHTS=np.array([.5,.125,.125,.125,.125])
STATES=np.array([[s&1,(s>>1)&1] for s in range(4)],dtype=int)
# Gate activation by input pair, finite state and edge. Precomputed once.
GATES={}
for a,b in itertools.product([0,1],repeat=2):
    vals=np.column_stack([np.full(4,a),np.full(4,b),STATES])
    GATES[(a,b)]=np.array([np.prod(vals[:,list(src)],axis=1) for j,src in EDGES]).T
TARGET=np.array([j for j,src in EDGES])
DEST=np.array([[s^(1<<j) for j,src in EDGES]+[s^1,s^2] for s in range(4)])

def softmax(theta):
    p=np.exp(theta-np.max(theta)); return p/p.sum()

def make_tables(c,mu=MU,gamma=GAMMA,alpha=ALPHA):
    tables={}
    for inp,gates in GATES.items():
        target=STATES[:,TARGET]
        fuel=alpha*c[None,:]*gates*np.where(target==0,1.,np.exp(EPS-mu))
        reset=gamma*np.where(STATES==1,1.,np.exp(-EPS))
        rates=np.column_stack([fuel,reset])
        work=np.column_stack([mu*np.where(target==0,1.,-1.),np.zeros((4,2))])
        heat=np.column_stack([np.where(target==0,mu-EPS,EPS-mu),np.where(STATES==1,EPS,-EPS)])
        tables[inp]=(rates,work,heat)
    return tables

def exact_case(c,case=CASES[0],mu=MU,gamma=GAMMA,durations=DURATION):
    tables=make_tables(c,mu,gamma)
    v=np.zeros(6);v[0]=1
    for inp,dt in zip(case[1],durations):
        rates,work,heat=tables[inp]
        Q=np.zeros((4,4))
        for s in range(4):
            for e,r in enumerate(rates[s]): Q[s,DEST[s,e]]+=r
            Q[s,s]-=rates[s].sum()
        # Column-vector augmented forward equation, integrated work and heat.
        H=np.zeros((6,6)); H[:4,:4]=Q.T
        H[4,:4]=np.sum(rates*work,axis=1)
        H[5,:4]=np.sum(rates*heat,axis=1)
        v=expm(H*float(dt))@v
    p=float(v[1]+v[3]) # m bit, provided for accounting
    pz=float(v[2]+v[3])
    internal_energy=EPS*float(v[:4]@STATES.sum(axis=1))
    return {'p_output':pz,'p_memory':p,'chemical_work_kT':float(v[4]),'heat_kT':float(v[5]),
            'internal_energy_kT':internal_energy,'first_law_residual':float(v[4]-v[5]-internal_energy),
            'state_probability':v[:4].tolist()}

def evaluate(c,**kw):
    details={case[0]:exact_case(c,case,**kw) for case in CASES}
    scores=np.array([details[case[0]]['p_output'] if case[2] else 1-details[case[0]]['p_output'] for case in CASES])
    return {'balanced_accuracy':float(scores@WEIGHTS),'cases':details}

def trajectory(tables,case,rng):
    state=0; E=np.zeros(NEDGE);nfuel=0; nevents=0
    for inp,dt in zip(case[1],DURATION):
        rates,_,_=tables[inp];remain=float(dt)
        while remain>0:
            rr=rates[state]; total=float(rr.sum())
            tau=rng.exponential(1/total) if total else math.inf
            dwell=min(tau,remain)
            E-=rr[:NEDGE]*dwell
            remain-=dwell
            if tau>=dwell+remain and remain<=1e-14: break
            # More direct event criterion: tau < previous remaining.
            if tau>dwell+1e-14: break
            draw=rng.random()*total
            e=int(np.searchsorted(np.cumsum(rr),draw,side='right'))
            e=min(e,len(rr)-1)
            if e<NEDGE:
                E[e]+=1
                nfuel+=1 if STATES[state,TARGET[e]]==0 else -1
            state=int(DEST[state,e]);nevents+=1
    output=(state>>1)&1
    return float(output==case[2]),E,nfuel,nevents,output


def train(seed):
    rng=np.random.default_rng(seed)
    theta=rng.normal(0,.05,NEDGE)
    initial=evaluate(MATERIAL*softmax(theta))
    batch=64;ntrajectory=19968;nbatch=ntrajectory//batch;eta=1.5
    baseline=.5;history=[];events=0;fuel=0
    started=time.perf_counter()
    for it in range(nbatch):
        p=softmax(theta);c=MATERIAL*p; tables=make_tables(c)
        hs=np.zeros(NEDGE);rrs=[]
        for k in range(batch):
            ci=int(rng.choice(len(CASES),p=WEIGHTS))
            R,E,nf,nev,_=trajectory(tables,CASES[ci],rng)
            hs+=(R-baseline)*E
            rrs.append(R);events+=nev;fuel+=nf
        h=hs/batch
        theta+=eta*(h-p*np.sum(h))
        baseline=.95*baseline+.05*float(np.mean(rrs))
        if (it+1)%52==0 or it==nbatch-1:
            c=MATERIAL*softmax(theta)
            history.append({'trajectories':(it+1)*batch,'exact_balanced_accuracy':evaluate(c)['balanced_accuracy'],
                            'material':dict(zip(NAMES,c.tolist()))})
    finalc=MATERIAL*softmax(theta)
    return {'seed':seed,'n_training_trajectories':ntrajectory,'eta':eta,'batch':batch,
            'initial':initial,'final':evaluate(finalc),'final_material':dict(zip(NAMES,finalc.tolist())),
            'history':history,'training_events':events,'net_training_fuel':fuel,
            'chemical_training_work_kT':MU*fuel,'runtime_seconds':time.perf_counter()-started}


def score_check(n=6000):
    rng=np.random.default_rng(85117);theta=np.linspace(-.2,.2,NEDGE);p=softmax(theta);c=MATERIAL*p
    tables=make_tables(c);vals=[];scores=[];outputs=[]
    for k in range(n):
        ci=int(rng.choice(len(CASES),p=WEIGHTS));R,E,_,_,_=trajectory(tables,CASES[ci],rng)
        h=(R-.5)*E
        vals.append(h-p*np.sum(h));scores.append(R)
    vals=np.array(vals)
    grad=np.zeros(NEDGE);delta=1e-5
    for e in range(NEDGE):
        t=theta.copy();t[e]+=delta;a=evaluate(MATERIAL*softmax(t))['balanced_accuracy']
        t=theta.copy();t[e]-=delta;b=evaluate(MATERIAL*softmax(t))['balanced_accuracy']
        grad[e]=(a-b)/(2*delta)
    mean=vals.mean(axis=0);se=vals.std(axis=0,ddof=1)/np.sqrt(n)
    exact=evaluate(c)['balanced_accuracy'];mc=float(np.mean(scores));mcse=float(np.std(scores,ddof=1)/np.sqrt(n))
    return {'n':n,'exact_gradient':grad.tolist(),'mc_gradient':mean.tolist(),'mc_standard_error':se.tolist(),
            'max_standardized_discrepancy':float(np.max(abs(mean-grad)/se)),
            'gradient_within_4se_plus_0p015':bool(np.all(abs(mean-grad)<4*se+.015)),
            'mc_accuracy':mc,'exact_accuracy':exact,'mc_accuracy_se':mcse,
            'accuracy_within_4se_plus_0p01':bool(abs(mc-exact)<4*mcse+.01)}


def explicit_binding_fanout(fanout,kon=50.,koff=5.,kcat=8.):
    """One scarce active m can bind one of f downstream catalysts at a time.
    State (m,z_1,...,z_f,occupancy), occupancy 0=free,1..f=bound.
    Total downstream material 2 is shared across f contacts.
    Occupancy may be zero if m=0; resetting m releases it.
    Every set channel and reverse is retained. Binding is reversible.
    The constant binding rates and forced release are an explicit kinetic
    loading model, not a complete microscopic free-energy model.
    """
    tuples=[]
    for bits in itertools.product([0,1],repeat=fanout+1):
        for occ in range(fanout+1):
            if occ and not bits[0]:continue
            tuples.append((bits,occ))
    index={s:i for i,s in enumerate(tuples)};n=len(tuples)
    p=np.zeros(n);p[index[((0,)*(fanout+1),0)]]=1
    for a,b in [(1,0),(0,0),(0,1)]:
        Q=np.zeros((n,n))
        def add(s, bits,occ,rate):
            Q[s,index[(tuple(bits),occ)]]+=rate;Q[s,s]-=rate
        for s,(bits,occ) in enumerate(tuples):
            # m's A-set channel, passive reset (if bound, release on reset).
            mm=list(bits);mm[0]=1-bits[0]
            add(s,mm,0 if bits[0] else occ,8*a*(1 if bits[0]==0 else np.exp(EPS-MU)))
            add(s,mm,0 if bits[0] else occ,GAMMA*(1 if bits[0] else np.exp(-EPS)))
            if bits[0] and not occ and b:
                for branch in range(1,fanout+1):add(s,bits,branch,kon*2/fanout)
            if occ:add(s,bits,0,koff)
            for j in range(1,fanout+1):
                zz=list(bits);zz[j]=1-bits[j]
                add(s,zz,occ,GAMMA*(1 if bits[j] else np.exp(-EPS)))
                if occ==j and b:add(s,zz,occ,kcat*(1 if bits[j]==0 else np.exp(EPS-MU)))
        p=p@expm(Q)
    marg=[sum(p[i] for i,(bits,occ) in enumerate(tuples) if bits[j]) for j in range(1,fanout+1)]
    joint=sum(p[i] for i,(bits,occ) in enumerate(tuples) if all(bits[1:]))
    return {'fanout':fanout,'n_states':n,'total_downstream_material':2.,'per_branch_output_probabilities':list(map(float,marg)),
            'all_outputs_probability':float(joint),'probability_sum':float(p.sum())}


def main():
    report={'version':'frozen-v0 probe','parameters':{'epsilon_kT':EPS,'mu_kT':MU,'gamma':GAMMA,'alpha':ALPHA,'material':MATERIAL},
            'contact_names':NAMES,'training_case_weights':dict(zip([c[0] for c in CASES],WEIGHTS.tolist()))}
    hand=np.zeros(NEDGE);hand[NAMES.index('A->m')]=2.;hand[NAMES.index('B+m->z')]=2.
    report['hand_constructed']=evaluate(hand)
    report['fuel_off']=evaluate(hand,mu=0.)
    removal={}
    for name in ['A->m','B+m->z']:
        c=hand.copy();c[NAMES.index(name)]=0;removal[name]=evaluate(c)
    report['contact_removal']=removal
    report['short_memory']=evaluate(hand,gamma=3.)
    # Q*pi=0 with zero chemical affinity, independent of catalyst allocation.
    c=np.linspace(.1,1,NEDGE);tables=make_tables(c,mu=0.)
    pi=np.exp(-EPS*STATES.sum(axis=1));pi/=pi.sum();res=[]
    for rates,_,_ in tables.values():
        Q=np.zeros((4,4))
        for s in range(4):
            for e,r in enumerate(rates[s]):Q[s,DEST[s,e]]+=r
            Q[s,s]-=rates[s].sum()
        res.append(float(abs(pi@Q).max()))
    report['fuel_off_boltzmann_stationarity_max_residual']=max(res)
    report['score_check']=score_check()
    print('Hand accuracy',report['hand_constructed']['balanced_accuracy'],flush=True)
    print('Score check',report['score_check']['max_standardized_discrepancy'],flush=True)
    report['learning_seeds']=[]
    for seed in [1301,1302,1303,1304]:
        rec=train(seed);report['learning_seeds'].append(rec)
        (OUT/'results_partial.json').write_text(json.dumps(report,indent=2))
        print('Seed',seed,'accuracy',rec['final']['balanced_accuracy'],'runtime',rec['runtime_seconds'],flush=True)
    report['explicit_sequestration']=[explicit_binding_fanout(f) for f in [1,2,4]]
    report['runtime_note']='single local Python process, 79,872 training trajectories + 6,000 verifier trajectories; exact evaluation is separate and never feeds updates'
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print('Saved',OUT/'results.json',flush=True)

if __name__=='__main__':main()
