"""Inspected bounded AE-v2 acquisition probe. No network, subprocess or host execution."""
from __future__ import annotations
import itertools, json, math, resource, time
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp

resource.setrlimit(resource.RLIMIT_CPU, (110,115))
START=time.perf_counter(); OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
E=json.loads((ROOT/'probes/E/results.json').read_text()); LEARNED=E['learning_seeds'][0]
assert LEARNED['seed']==1301
C=np.array([LEARNED['final_material'][n] for n in E['contact_names']])
EDGES=[]
for target,sources in [(0,[0,1,3]),(1,[0,1,2])]:
    for order in [1,2]:
        for pair in itertools.combinations(sources,order): EDGES.append((target,pair))
STATES=[(m,z) for z in range(3) for m in range(2)]; INDEX={s:i for i,s in enumerate(STATES)}
CASES=[('AB',[(1,0),(0,0),(0,1)],1),('BA',[(0,1),(0,0),(1,0)],0),
       ('A_only',[(1,0),(0,0),(0,0)],0),('B_only',[(0,0),(0,0),(0,1)],0),
       ('empty',[(0,0),(0,0),(0,0)],0)]
CW=np.array([.5,.125,.125,.125,.125]); EPS=8.; MU=16.; GAMMA=.03
CAP=np.array([1,1,2,2,4,4,2]); COST=np.array([1.,1.,4.,.01,25.,.05])
ALL=np.array(list(itertools.product(range(2),range(2),range(3),range(3),range(5),range(5),range(3))),int)
LOOK={tuple(row):i for i,row in enumerate(ALL)}; MASS=ALL[:,:6]@COST+.1*(ALL[:,6]>0)
NACT=ALL[:,:6].sum(1)+(ALL[:,6]>0)
TRAIN=[(l,b) for l in [0.,1.,10.] for b in [35.,65.,115.]]
TEST=[(l,b) for l in [.5,4.,7.] for b in [50.,90.]]; CONTEXTS=TRAIN+TEST

def source(m_present,z_present,r,inp):
    a,b=inp; Q=np.zeros((6,6)); w=np.zeros(6)
    for si,(m,z) in enumerate(STATES):
        vals=[a,b,m,int(z==1)]
        def jump(dest,rate,work=0.):
            Q[INDEX[dest],si]+=rate;Q[si,si]-=rate;w[si]+=rate*work
        for (target,pair),cc in zip(EDGES,r*C):
            if (target==0 and not m_present) or (target==1 and not z_present):continue
            gate=float(np.prod([vals[k] for k in pair]))
            if target==0:jump((1-m,z),4*cc*gate*(1 if m==0 else math.exp(EPS-MU)),MU*(1-2*m))
            elif z<2:jump((m,1-z),4*cc*gate*(1 if z==0 else math.exp(EPS-MU)),MU*(1-2*z))
        if m_present:jump((1-m,z),GAMMA*(1 if m else math.exp(-EPS)))
        if z_present and z<2:jump((m,1-z),GAMMA*(1 if z else math.exp(-EPS)))
    return Q,w

def binding(active,pool,c0,kon,koff,h):
    # Exact reversible binding subflow, including conditional-distribution decay.
    kd=koff/kon; total=active+pool+kd
    delta=np.sqrt(np.maximum(total*total-4*active*pool,0.))
    lo=2*active*pool/(total+delta); hi=(total+delta)/2
    q0=(c0-lo)/(c0-hi); q=q0*np.exp(-kon*delta*h)
    cn=(lo-q*hi)/(1-q)
    decay=np.exp(-kon*(pool+kd-lo)*h)*(1-q0)/(1-q)
    return cn,decay

def simulate(configs,loads,h=.02):
    """Full model terminal quantities. Exact subflows; Strang for source/binding,
    symmetric carrier/load splitting. Work integrates each driven subflow."""
    n=len(configs); nsteps=round(1/h); assert abs(nsteps*h-1)<1e-10
    groups={key:np.flatnonzero(np.all(configs[:,:3]==key,axis=1))
            for key in itertools.product(range(2),range(2),range(3))}
    mats={}
    for key in groups:
        for inp in [(0,0),(1,0),(0,1)]:
            Q,w=source(*key,inp); aug=np.zeros((7,7));aug[:6,:6]=Q;aug[6,:6]=w
            mats[key,inp]=expm(aug*h)
    port=configs[:,6]; sensor=.01*configs[:,3]; YT=25.*configs[:,4]
    G=12.5*configs[:,5]; kc=1250.*configs[:,5]
    source_pool=np.where(port==1,loads,np.where(port==2,sensor,0.))
    export_load=np.where(port==2,loads,0.)
    kon=np.where(port==1,10.,100.);koff=np.where(port==1,1.,10000.)
    answers=[];works=[]; minstate=0.; maxmass=0.
    for name,sequence,label in CASES:
        p=np.zeros((n,6));p[:,0]=1.; y=np.zeros(n);d=np.zeros(n);work=np.zeros(n)
        def sbind(dt):
            act=p[:,2:4]+p[:,4:6]; A=act.sum(1); old=p[:,4:6].copy(); c0=old.sum(1)
            cn,decay=binding(A,source_pool,c0,kon,koff,dt)
            frac=np.divide(act,A[:,None],out=np.zeros_like(act),where=A[:,None]>0)
            bound=frac*cn[:,None]+(old-frac*c0[:,None])*decay[:,None]
            p[:,4:6]=bound;p[:,2:4]=act-bound
        def carrier(dt):
            nonlocal y,work
            c=p[:,4:6].sum(1); alpha=kc*c+G*math.exp(-16); beta=kc*c*math.exp(-16)+G
            rate=alpha+beta; eq=np.divide((YT-d)*alpha,rate,out=np.zeros(n),where=rate>0)
            decay=np.exp(-rate*dt); integral=eq*dt+np.divide((y-eq)*(1-decay),rate,out=y*dt,where=rate>0)
            work+=32*kc*c*((YT-d)*dt-(1+math.exp(-16))*integral)
            y=eq+(y-eq)*decay
        for inp in sequence:
            for tick in range(nsteps):
                sbind(h/2)
                for key,ix in groups.items():
                    if len(ix):
                        mat=mats[key,inp];work[ix]+=p[ix]@mat[6,:6];p[ix]=p[ix]@mat[:6,:6].T
                sbind(h/2)
                carrier(h/2)
                active=y+d;dn,_=binding(active,export_load,d,10.,1.,h);y=active-dn;d=dn
                carrier(h/2)
            if time.perf_counter()-START>100:raise TimeoutError('internal 100s boundary')
        out=np.where(port==1,p[:,2:4].sum(1),np.where(port==2,y,0.))
        answers.append(out);works.append(work)
        minstate=min(minstate,float(p.min()),float(y.min()),float(d.min()),float((YT-y-d).min()))
        maxmass=max(maxmass,float(np.max(abs(p.sum(1)-1))))
    ports=np.array(answers); task=np.array([np.clip(v,0,1) if case[2] else 1-np.clip(v,0,1) for v,case in zip(ports,CASES)])
    return {'ports':ports,'task':CW@task,'work':CW@np.array(works),'case_work':np.array(works),'minstate':minstate,'maxmass':maxmass}

def bdf(config,load,case):
    m,z,r,s,ny,k,port=config; SS=.01*s;YT=25.*ny;G=12.5*k;kc=1250.*k
    v=np.zeros(10);v[0]=1.
    for inp in case[1]:
        Q,w=source(m,z,r,inp)
        def rhs(t,v):
            p=v[:6];y,d=v[6:8];bound=p[4:6].sum(); dp=Q@p
            pool=load if port==1 else SS if port==2 else 0.
            kon=10. if port==1 else 100.;koff=1. if port==1 else 10000.
            for mm in [0,1]:
                f=2+mm;c=4+mm;flux=kon*(pool-bound)*p[f]-koff*p[c]
                dp[f]-=flux;dp[c]+=flux
            y0=YT-y-d;drive=kc*bound*(y0-y*math.exp(-16));reset=G*(y-y0*math.exp(-16))
            lf=10*((load if port==2 else 0.)-d)*y-d
            return np.r_[dp,drive-reset-lf,lf,w@p,32*drive]
        sol=solve_ivp(rhs,(0,1),v,method='BDF',rtol=1e-9,atol=1e-11)
        assert sol.success;v=sol.y[:,-1]
    return float(v[2:4].sum() if port==1 else v[6] if port==2 else 0.),float(v[8]+v[9])

def reward(task,work,ix):return task-.0002*MASS[ix]-.0005*NACT[ix]-1e-5*work

# Reference model is evaluator-only. Train sees scalar queries, never this table/argmax.
batch=np.tile(ALL,(len(CONTEXTS),1)); loads=np.repeat([c[0] for c in CONTEXTS],len(ALL))
physics=simulate(batch,loads); TASK=physics['task'].reshape(len(CONTEXTS),-1);WORK=physics['work'].reshape(len(CONTEXTS),-1)
REWARD=TASK-.0002*MASS[None,:]-.0005*NACT[None,:]-1e-5*WORK
np.savez_compressed(OUT/'reference.npz',configs=ALL,contexts=CONTEXTS,task=TASK,work=WORK,reward=REWARD)

# Fixed nontrivial checks: missing source, direct, economical driver, full driver,
# positive and order-reversed, high and zero load. No favorable outcome selection.
CHECKS=[([1,1,1,0,0,0,1],10.,0),([1,1,1,0,0,0,1],0.,1),
        ([1,1,1,1,4,4,2],10.,0),([1,1,1,1,4,4,2],10.,1),
        ([1,1,2,2,2,2,2],4.,0),([1,1,2,2,2,2,2],4.,1),
        ([0,1,1,1,1,1,2],1.,0),([1,0,1,1,1,1,2],1.,0)]
checks=[]
for cfg,load,ci in CHECKS:
    a=simulate(np.array([cfg]),np.array([load]),.02);small=simulate(np.array([cfg]),np.array([load]),.01)
    port,ww=bdf(cfg,load,CASES[ci]);ix=LOOK[tuple(cfg)]
    checks.append({'config':cfg,'load':load,'case':CASES[ci][0],
      'split_port':float(a['ports'][ci,0]),'bdf_port':port,
      'port_error':abs(float(a['ports'][ci,0])-port),
      'work_error':abs(float(a['case_work'][ci,0])-ww),
      'case_reward_error_bound':abs(float(a['ports'][ci,0])-port)+1e-5*abs(float(a['case_work'][ci,0])-ww),
      'step_halving_port_error':abs(float(a['ports'][ci,0])-float(small['ports'][ci,0]))})
sanity={'max_terminal_port_error':max(x['port_error'] for x in checks),
        'max_checked_case_reward_error_bound':max(x['case_reward_error_bound'] for x in checks),
        'max_step_halving_port_error':max(x['step_halving_port_error'] for x in checks),
        'minstate':physics['minstate'],'max_source_probability_mass_error':physics['maxmass']}
sanity['passed']=sanity['max_terminal_port_error']<=.005 and sanity['max_checked_case_reward_error_bound']<=.005 and sanity['max_step_halving_port_error']<=.005 and sanity['minstate']>=-1e-8 and sanity['max_source_probability_mass_error']<=1e-8

def features(cfg,load,budget,blind=False):
    mass=float(cfg[:6]@COST+.1*(cfg[6]>0))
    return np.r_[1.,0. if blind else math.log1p(load)/math.log(11),budget/115,(budget-mass)/115,cfg/CAP]

def mask(cfg,budget):
    mass=float(cfg[:6]@COST+.1*(cfg[6]>0)); ok=np.ones(9,bool)
    ok[:6]=(cfg[:6]<CAP[:6])&(mass+COST<=budget+1e-8)
    ok[6:8]=(cfg[6]==0)&(mass+.1<=budget+1e-8)
    return ok

class Policy:
    def __init__(self,seed,linear=False):
        self.rng=np.random.default_rng(seed);self.linear=linear; D=11
        self.params=[self.rng.normal(0,.05,(9,D))] if linear else [self.rng.normal(0,.15,(16,D)),self.rng.normal(0,.05,(9,17))]
        self.mm=[np.zeros_like(p) for p in self.params];self.vv=[p.copy() for p in self.mm];self.t=0
    def act(self,cfg,load,budget,greedy=False):
        x=features(cfg,load,budget)
        if self.linear:h=x;logits=self.params[0]@h
        else:h=np.r_[1.,np.tanh(self.params[0]@x)];logits=self.params[1]@h
        logits=np.where(mask(cfg,budget),logits,-1e9);pr=np.exp(logits-logits.max());pr/=pr.sum()
        action=int(pr.argmax()) if greedy else int(self.rng.choice(9,p=pr))
        dl=-pr;dl[action]+=1
        if self.linear:grad=[np.outer(dl,x)]
        else:grad=[np.outer((self.params[1][:,1:].T@dl)*(1-h[1:]**2),x),np.outer(dl,h)]
        return action,grad
    def episode(self,context,train=False,assumed_load=None):
        load,budget=CONTEXTS[context];load=load if assumed_load is None else assumed_load
        cfg=np.zeros(7,int);grad=[np.zeros_like(p) for p in self.params];acts=[]
        for step in range(16):
            action,g=self.act(cfg,load,budget)
            if action==8:break
            if step==15:break
            for acc,gg in zip(grad,g):acc+=gg
            acts.append(action)
            if action<6:cfg[action]+=1
            else:cfg[6]=action-5
        # STOP's log probability must also receive delayed credit.
        if action==8:
            for acc,gg in zip(grad,g):acc+=gg
        ix=LOOK[tuple(cfg)]
        return ix,grad,acts
    def update(self,grad,adv):
        self.t+=1
        for i,g in enumerate(grad):
            g=np.clip(adv*g,-1,1);self.mm[i]=.9*self.mm[i]+.1*g;self.vv[i]=.999*self.vv[i]+.001*g*g
            self.params[i]+=.008*(self.mm[i]/(1-.9**self.t))/(np.sqrt(self.vv[i]/(1-.999**self.t))+1e-8)

def evaluate(policy,per=64,blind=False):
    ans=[]
    for c,(load,budget) in enumerate(CONTEXTS):
        ix=np.array([policy.episode(c,assumed_load=0. if blind else None)[0] for _ in range(per)])
        assert np.all(MASS[ix]<=budget+1e-8)
        ans.append({'load':load,'budget':budget,'reward':float(REWARD[c,ix].mean()),'task':float(TASK[c,ix].mean()),
             'material':float(MASS[ix].mean()),'work':float(WORK[c,ix].mean()),
             'driver_fraction':float(np.mean(ALL[ix,6]==2)),
             'common_config':ALL[np.bincount(ix,minlength=len(ALL)).argmax()].tolist()})
    return ans

# Finite difference score check through fixed one-step action: no evaluator involved.
gpol=Policy(311); cfg=np.zeros(7,int);saved=gpol.rng.bit_generator.state
aa,gg=gpol.act(cfg,1.,65.)
def logprob():
    x=features(cfg,1.,65.);h=np.r_[1.,np.tanh(gpol.params[0]@x)];l=gpol.params[1]@h;l=np.where(mask(cfg,65.),l,-1e9)
    return float(l[aa]-np.log(np.exp(l-l.max()).sum())-l.max())
old=gpol.params[0][2,1];eps=1e-6;gpol.params[0][2,1]=old+eps;plus=logprob();gpol.params[0][2,1]=old-eps;minus=logprob();gpol.params[0][2,1]=old
sanity['score_gradient_error']=abs((plus-minus)/(2*eps)-gg[0][2,1]);sanity['passed'] &= sanity['score_gradient_error']<1e-7

results=[]
for linear in [False,True]:
    for seed in [631,632,633]:
        p=Policy(seed,linear);before=evaluate(p);baseline=np.full(len(TRAIN),.5);curves=[];training_work=0.;training_material=0.;unique=set()
        for t in range(2000):
            c=int(p.rng.integers(len(TRAIN)));ix,grad,acts=p.episode(c,True);rr=float(REWARD[c,ix])
            adv=rr-baseline[c];baseline[c]=.95*baseline[c]+.05*rr;p.update(grad,adv)
            training_work+=float(WORK[c,ix]);training_material+=float(MASS[ix]);unique.add((c,ix))
            if (t+1)%250==0:curves.append({'episode':t+1,'EMA_reward':float(baseline.mean())})
        after=evaluate(p);load_blind=evaluate(p,blind=True)
        results.append({'kind':'linear' if linear else 'neural','seed':seed,'before':before,'after':after,'load_blind':load_blind,'curves':curves,
          'training_installations':2000,'training_logical_assays':10000,'unique_reward_queries':len(unique),
          'weighted_training_work':training_work,'training_material':training_material,
          'parameters':sum(a.size for a in p.params)})
        np.savez_compressed(OUT/f'policy_{"linear" if linear else "neural"}_{seed}.npz',**{f'p{i}':a for i,a in enumerate(p.params)})
        if time.perf_counter()-START>100:raise TimeoutError('internal 100s boundary')

oracle=[]
for c,(load,budget) in enumerate(CONTEXTS):
    feasible=np.flatnonzero(MASS<=budget+1e-8);ix=int(feasible[np.argmax(REWARD[c,feasible])])
    oracle.append({'load':load,'budget':budget,'config':ALL[ix].tolist(),'reward':float(REWARD[c,ix]),'task':float(TASK[c,ix]),'material':float(MASS[ix]),'work':float(WORK[c,ix])})
ordinary=[]
for seed in [631,632,633]:
    rng=np.random.default_rng(seed);observed={c:[] for c in range(len(TRAIN))}
    for t in range(2000):
        c=int(rng.integers(len(TRAIN)));feasible=np.flatnonzero(MASS<=TRAIN[c][1]+1e-8);ix=int(rng.choice(feasible));observed[c].append(ix)
    ans=[]
    for c,(load,budget) in enumerate(CONTEXTS):
        # Choose closest known training context with at least one feasible observed install.
        order=sorted(range(len(TRAIN)),key=lambda j:abs(math.log1p(TRAIN[j][0])-math.log1p(load))+abs(TRAIN[j][1]-budget)/115)
        for j in order:
            options=[i for i in observed[j] if MASS[i]<=budget+1e-8]
            if options:break
        ix=max(options,key=lambda i:REWARD[j,i])
        ans.append({'load':load,'budget':budget,'reward':float(REWARD[c,ix]),'task':float(TASK[c,ix]),'material':float(MASS[ix]),'work':float(WORK[c,ix]),'config':ALL[ix].tolist()})
    ordinary.append({'seed':seed,'queries':2000,'after':ans})

neural=[r for r in results if r['kind']=='neural']
improvements=[float(np.mean([v['reward'] for v in r['after'][:9]])-np.mean([v['reward'] for v in r['before'][:9]])) for r in neural]
testscore=float(np.mean([v['task'] for r in neural for v in r['after'][9:]]))
summary={'neural_train_reward_improvements':improvements,'neural_mean_heldout_task':testscore,
  'partial_acquisition_criterion':bool(sanity['passed'] and min(improvements)>=.05 and testscore>=.75)}
res={'version':'AE-v2 finite installation acquisition','inherited_E_seed':1301,'constants':{'kcat_per_turnover':1250,'G_per_turnover':12.5,'dt':.02},
     'sanity':sanity,'checks':checks,'policies':results,'ordinary_random_allocator':ordinary,'exhaustive_allocator':oracle,
     'reference_trajectories':len(ALL)*len(CONTEXTS)*5,'summary':summary,'runtime_seconds':time.perf_counter()-START}
(OUT/'results.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({'sanity':sanity,'summary':summary,'runtime_seconds':res['runtime_seconds']},indent=2))
