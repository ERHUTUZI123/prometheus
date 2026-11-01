"""Inspected bounded synthetic mechanism pilot. No external inputs or network."""
import json
import time
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
RAW = (OUT / 'raw.jsonl').open('x')
START = time.monotonic()
P = np.linspace(-1., 1., 9)
AMP = np.array([-1., .5, 2.])
SEEDS = [3101, 3102, 3103]

def emit(kind, **kw):
    row = dict(kind=kind, **kw)
    s = json.dumps(row, allow_nan=False)
    assert json.loads(s)['kind'] == kind
    RAW.write(s+'\n'); RAW.flush()

def phi(x):
    x = np.asarray(x)
    return np.stack((x, x*x, np.sin(np.pi*x), np.ones_like(x)), axis=-1)

def cell(th, x):
    return np.tanh(phi(x) @ th)

class World:
    def __init__(self, seed):
        r = np.random.default_rng(seed)
        self.th = r.normal(0, .35, (10, 4))
        # The following fields are environmental secrets, never Q inputs.
        base = r.normal(0, .65, (3, 4))
        self.group = r.integers(0, 3, 10)
        self.teacher = base[self.group] + r.normal(0, .08, (10, 4))
        self.edges = []
        for j in range(10):
            self.edges.append((j, (j+1) % 10, float(r.choice([-1, 1]))))
        for _ in range(8):
            i,j = r.choice(10, 2, replace=False)
            self.edges.append((int(i), int(j), float(r.choice([-1, 1]))))
        self.mem = [[] for _ in range(10)]
        self.cost = dict(cell_forward=0, backward=0, labels=0,
                         distill_steps=0, score_calls=0, mlp_multiplies=0)
        self.changed = set()

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def f(self, pr, x, theta=None, count=True):
        i,j,s = self.edges[pr]
        th = self.th if theta is None else theta
        a,b = cell(th[i], x), cell(th[j], x)
        if count: self.cost['cell_forward'] += 2*len(x)
        z = a + .7*s*b + .3*a*b
        return np.tanh(z)

    def target(self, pr, x):
        return self.f(pr, x, self.teacher, False)

    def consumers(self,j):
        return [k for k,(a,b,s) in enumerate(self.edges) if j in (a,b)]

    def deriv(self, pr, x, j):
        a,b,s = self.edges[pr]
        va,vb = cell(self.th[a],x),cell(self.th[b],x)
        self.cost['cell_forward'] += 2*len(x)
        z = va + .7*s*vb + .3*va*vb
        y = np.tanh(z)
        c = (1+.3*vb) if j == a else (.7*s+.3*va)
        return y, (1-y*y)*c, -2*y*(1-y*y)*c*c

    def batch(self, j, rng, n=8):
        ps = rng.choice(self.consumers(j), n)
        xs = rng.uniform(-1,1,n)
        ys = np.array([self.target(int(p),np.array([x]))[0] for p,x in zip(ps,xs)])
        self.cost['labels'] += n
        return [(int(p),float(x),float(y)) for p,x,y in zip(ps,xs,ys)]

    def remember(self,j,batch):
        self.mem[j] = (self.mem[j]+batch)[-8:]

    def loss(self, batch, theta=None):
        return float(np.mean([.5*(self.f(p,np.array([x]),theta)[0]-y)**2
                              for p,x,y in batch]))

    def grad(self,j,batch):
        g = np.zeros(4)
        for p,x,y in batch:
            x = np.array([x])
            out,c,_ = self.deriv(p,x,j)
            v = cell(self.th[j],x)
            self.cost['cell_forward'] += 1
            g += (out[0]-y)*c[0]*(1-v[0]**2)*phi(x)[0]/len(batch)
        self.cost['backward'] += len(batch)
        return g

    def history(self,j,mem=None):
        mem = self.mem[j] if mem is None else mem
        if not mem: return np.zeros(9),np.zeros(9)
        xs,m,k = [],[],[]
        for pr,x,y in mem:
            out,c,h = self.deriv(pr,np.array([x]),j)
            e=out[0]-y
            xs.append(x);m.append(e*c[0]);k.append(c[0]**2+e*h[0])
        weights=np.exp(-((P[:,None]-np.array(xs)[None,:])/.45)**2)+1e-6
        weights /= weights.sum(1,keepdims=True)
        return weights @ m, weights @ k

    def transport(self,j,packet,a):
        before = self.th[j].copy()
        start = cell(before,P)
        self.cost['cell_forward'] += 9
        th=before.copy();target=start+a*packet
        for _ in range(4):
            v=cell(th,P)
            g=phi(P).T @ ((v-target)*(1-v*v))/9
            th-=1.5*g
            self.cost['cell_forward']+=9
            self.cost['backward']+=9
            self.cost['distill_steps']+=1
        actual=cell(th,P)-start
        self.cost['cell_forward']+=9
        return th,actual,float(np.mean((actual-a*packet)**2))

    def features(self,j,d,mem=None):
        v=cell(self.th[j],P);self.cost['cell_forward']+=9
        m,k=self.history(j,mem)
        # All variables are observations/known derivatives, not teacher identities.
        feat=np.r_[v,d,4*m,k,4*m*d,4*k*d*d,
                   np.linalg.norm(d),np.mean(d),np.mean(v*v)]
        first=-float(np.mean(m*d))
        second=first-.5*float(np.mean(k*d*d))
        return feat,first,second

    def evaluate(self, xs):
        vals=[];stable=[]
        for p,(i,j,s) in enumerate(self.edges):
            val=float(np.mean(.5*(self.f(p,xs)-self.target(p,xs))**2))
            vals.append(val)
            if i not in self.changed and j not in self.changed:stable.append(val)
        return float(np.mean(vals)),float(np.mean(stable)) if stable else None

    def change(self):
        self.changed=set(np.flatnonzero(self.group==0).tolist())
        self.teacher[list(self.changed)] *= -1

def source_step(w,rng,t):
    pr=int(rng.integers(18));j=int(rng.choice(w.edges[pr][:2]))
    xs=rng.uniform(-1,1,8)
    ys=w.target(pr,xs)
    w.cost['labels']+=8
    batch=[(pr,float(x),float(y)) for x,y in zip(xs,ys)]
    before=cell(w.th[j],P);w.cost['cell_forward']+=9
    rate=[.15,.6,2.][t%3]
    g=w.grad(j,batch);g*=min(1.,1.2/(np.linalg.norm(g)*rate+1e-12))
    w.th[j]-=rate*g
    w.remember(j,batch)
    packet=cell(w.th[j],P)-before;w.cost['cell_forward']+=9
    return j,packet

class Predictor:
    def __init__(self,X,y,seed):
        self.mean=X.mean(0);self.std=X.std(0)+.03
        self.ym=float(y.mean());self.ys=float(y.std()+1e-4)
        r=np.random.default_rng(seed)
        self.par=[r.normal(0,1/np.sqrt(X.shape[1]),(X.shape[1],48)),np.zeros(48),
                  r.normal(0,.1,(48,1)),np.zeros(1)]
        z=(X-self.mean)/self.std;target=((y-self.ym)/self.ys)[:,None]
        ms=[np.zeros_like(p) for p in self.par];vs=[p.copy() for p in ms]
        for t in range(1,351):
            ids=r.integers(0,len(X),128);xx=z[ids];yy=target[ids]
            a=np.tanh(xx@self.par[0]+self.par[1]);pred=a@self.par[2]+self.par[3]
            err=np.clip(pred-yy,-5,5)/len(ids)
            da=(err@self.par[2].T)*(1-a*a)
            gs=[xx.T@da,da.sum(0),a.T@err,err.sum(0)]
            for p,g,m,v in zip(self.par,gs,ms,vs):
                m*=.9;m+=.1*g;v*=.999;v+=.001*g*g
                p-=.003*(m/(1-.9**t))/(np.sqrt(v/(1-.999**t))+1e-8)
        self.bytes=sum(p.nbytes for p in self.par)+self.mean.nbytes+self.std.nbytes

    def predict(self,X):
        return ((np.tanh((X-self.mean)/self.std@self.par[0]+self.par[1])@
                 self.par[2]+self.par[3])[:,0]*self.ys+self.ym)

def checks():
    w=World(800);pr=0;j=0;x=np.array([-.6,.2,.9])
    y,c,h=w.deriv(pr,x,j);eps=1e-4
    a,b,s=w.edges[pr];va=cell(w.th[a],x);vb=cell(w.th[b],x)
    def pert(d):return np.tanh((va+d)+.7*s*vb+.3*(va+d)*vb)
    err1=float(np.max(np.abs((pert(eps)-pert(-eps))/(2*eps)-c)))
    err2=float(np.max(np.abs((pert(eps)-2*y+pert(-eps))/eps**2-h)))
    batch=w.batch(j,np.random.default_rng(91));g=w.grad(j,batch);fd=[]
    for k in range(4):
        th=w.th.copy();th[j,k]+=eps;la=w.loss(batch,th)
        th[j,k]-=2*eps;lb=w.loss(batch,th);fd.append((la-lb)/(2*eps))
    errg=float(np.max(np.abs(g-fd)))
    e=np.array([.3,-.7,.9]);cc=np.array([.4,.8,-.3]);d=np.array([.2,-.5,.6])
    exact=float(np.mean(.5*e*e-.5*(e+cc*d)**2))
    expansion=float(np.mean(-e*cc*d-.5*cc*cc*d*d))
    assert err1<1e-6 and err2<1e-6 and errg<1e-6
    assert abs(exact-expansion)<1e-12
    perm=np.random.default_rng(99).permutation(10)
    renamed=w.copy();renamed.th[perm]=w.th;renamed.teacher[perm]=w.teacher
    renamed.edges=[(int(perm[a]),int(perm[b]),s) for a,b,s in w.edges]
    renamed.mem=[[] for _ in range(10)]
    for k in range(10):renamed.mem[int(perm[k])]=w.mem[k].copy()
    w.remember(j,batch);renamed.mem[int(perm[j])]=w.mem[j].copy()
    ff=w.features(j,np.ones(9)*.1)[0]
    rf=renamed.features(int(perm[j]),np.ones(9)*.1)[0]
    rename_error=float(np.max(np.abs(ff-rf)))
    assert rename_error<1e-12
    emit('derivative_and_identity_checks',first_error=err1,second_error=err2,
         parameter_error=errg,affine_identity_error=abs(exact-expansion),
         id_renaming_feature_error=rename_error)

def train_data(seed):
    X=[];Y=[];cost={};allrecords=[]
    for n in range(36):
        r=np.random.default_rng(seed*100+n)
        w=World(seed*1000+n)
        for j in range(10):w.remember(j,w.batch(j,r))
        for t in range(24):
            i,packet=source_step(w,r,t)
            j=int(r.choice([k for k in range(10) if k!=i]));a=float(r.choice(AMP))
            th,d,resid=w.transport(j,packet,a)
            feat,first,second=w.features(j,d)
            # Prediction features frozen BEFORE fresh labels and utility are obtained.
            batch=w.batch(j,r)
            sham=w.loss(batch);tmp=w.th.copy();tmp[j]=th
            edited=w.loss(batch,tmp);gain=sham-edited
            X.append(feat);Y.append(gain)
            w.th[j]=th;w.th[j]-=.25*w.grad(j,batch);w.remember(j,batch)
            emit('training_audit',seed=seed,world=n,event=t,source=i,recipient=j,
                 amplitude=a,propensity=1/27,sham_loss=sham,edited_loss=edited,
                 gain=gain,distill_residual=resid,requested=(a*packet).tolist(),
                 realized=d.tolist(),features=feat.tolist())
        for k,v in w.cost.items():cost[k]=cost.get(k,0)+v
    return np.array(X),np.array(Y),cost

def rollout(seed,world_id,mode,model):
    r=np.random.default_rng(seed*30000+world_id)
    w=World(seed*40000+world_id)
    for j in range(10):w.remember(j,w.batch(j,r))
    xs=np.linspace(-.995,.995,128)
    initial=w.evaluate(xs);gains=[];errors=[];stales=[];nohist=[]
    harm=0;abstain=0;identity=0;stale_mem=None
    for t in range(30):
        if t==15:
            stale_mem=[v.copy() for v in w.mem];w.change()
        source_rng=np.random.default_rng(seed*700000+world_id*100+t)
        i,packet=source_step(w,source_rng,t)
        recipients=[j for j in range(10) if j!=i]
        candidates=[]
        if mode not in ('local','random'):
            for j in recipients:
                for a in AMP:
                    th,d,resid=w.transport(j,packet,float(a))
                    feat,fo,so=w.features(j,d)
                    candidates.append(dict(j=j,a=float(a),th=th,d=d,resid=resid,
                                           feat=feat,first=fo,second=so))
            feats=np.array([v['feat'] for v in candidates])
            pred=model.predict(feats)
            w.cost['score_calls']+=len(candidates)
            w.cost['mlp_multiplies']+=2*len(candidates)*(len(feats[0])*48+48)
            # A generic contextual learned optimizer with this value head is identical.
            identity+=int(np.array_equal(pred,model.predict(feats)))
            if mode=='learned':scores=pred
            elif mode=='gradient':scores=np.array([v['first'] for v in candidates])
            elif mode=='curvature':scores=np.array([v['second'] for v in candidates])
            else:
                scores=[]
                for v in candidates:
                    tmp=w.th.copy();tmp[v['j']]=v['th']
                    scores.append(w.loss(w.mem[v['j']])-w.loss(w.mem[v['j']],tmp))
                scores=np.array(scores)
            if r.random()<.1:
                idx=int(r.integers(27));q=.1/27
                if idx==int(np.argmax(scores)) and scores[idx]>0:q+=.9
                chosen=candidates[idx]
            elif np.max(scores)>0:
                idx=int(np.argmax(scores));q=.9+.1/27;chosen=candidates[idx]
            else:
                chosen=None;q=.9/9
        elif mode=='random':
            j=int(r.choice(recipients));a=float(r.choice(AMP))
            th,d,resid=w.transport(j,packet,a);feat,_,_=w.features(j,d)
            chosen=dict(j=j,a=a,th=th,d=d,resid=resid,feat=feat);q=1/27
        else:chosen=None;q=1/9
        if chosen is None:
            j=int(r.choice(recipients));a=0.;th=w.th[j].copy();d=np.zeros(9)
            feat,_,_=w.features(j,d);resid=0.;abstain+=1
        else:
            j=chosen['j'];a=chosen['a'];th=chosen['th'];d=chosen['d']
            feat=chosen['feat'];resid=chosen['resid']
        prediction=0. if a==0 else float(model.predict(feat[None,:])[0])
        if a!=0:w.cost['mlp_multiplies']+=len(feat)*48+48
        # The selected proposal is frozen here, before the audit batch is generated.
        audit_rng=np.random.default_rng(seed*800000+world_id*100+t)
        batch=w.batch(j,audit_rng);sham=w.loss(batch)
        tmp=w.th.copy();tmp[j]=th;edited=w.loss(batch,tmp);gain=sham-edited
        gains.append(gain);errors.append((prediction-gain)**2);harm+=int(gain<0)
        if t>=20 and stale_mem is not None:
            stale_feat,_,_=w.features(j,d,stale_mem[j])
            stale_pred=0. if a==0 else float(model.predict(stale_feat[None,:])[0])
            stales.append((stale_pred-gain)**2)
            empty_feat,_,_=w.features(j,d,[])
            empty_pred=0. if a==0 else float(model.predict(empty_feat[None,:])[0])
            nohist.append((empty_pred-gain)**2)
            if a!=0:w.cost['mlp_multiplies']+=2*(len(feat)*48+48)
        w.th[j]=th;w.th[j]-=.25*w.grad(j,batch);w.remember(j,batch)
        if (t+1)%5==0:ev=w.evaluate(xs)
        else:ev=None
        emit('transfer_event',seed=seed,world=world_id,mode=mode,event=t,
             source=i,recipient=j,amplitude=a,propensity=q,prediction=prediction,
             sham_loss=sham,edited_loss=edited,gain=gain,
             requested=(a*packet).tolist(),realized=d.tolist(),distill_residual=resid,
             untouched_evaluation=ev,cost=w.cost.copy())
    final=w.evaluate(xs)
    return dict(seed=seed,world=world_id,mode=mode,initial_loss=initial[0],
                final_loss=final[0],unaffected_loss=final[1],gain_sum=float(sum(gains)),
                harm_count=harm,abstentions=abstain,prediction_mse=float(np.mean(errors)),
                late_current_mse=float(np.mean(errors[20:])),
                late_stale_mse=float(np.mean(stales)),late_nohistory_mse=float(np.mean(nohist)),
                contextual_optimizer_identity_checks=identity,cost=w.cost,
                memory_observation_slots=sum(3*len(m) for m in w.mem))

emit('serialization_smoke',seeds=SEEDS,timeout_seconds=120,
     evaluation_isolation='instruction_only; generated targets not passed to learner',
     status='exploratory, not confirmatory or novel')
checks()
results=[];training=[]
for seed in SEEDS:
    if seed==3101:
        previous=[json.loads(s) for s in (OUT.parent/'raw.jsonl').read_text().splitlines()]
        audits=[v for v in previous if v['kind']=='training_audit']
        fit=next(v for v in previous if v['kind']=='acquisition_fit')
        X=np.array([v['features'] for v in audits]);Y=np.array([v['gain'] for v in audits])
        cost=fit['cost']
        emit('reused_observations',seed=seed,count=len(audits),source='../raw.jsonl',
             missing_weights=True,refit_counted=True)
    else:
        X,Y,cost=train_data(seed)
    model=Predictor(X,Y,seed)
    trainmse=float(np.mean((model.predict(X)-Y)**2))
    if seed==3101:
        assert abs(trainmse-fit['mse'])<1e-12
    np.savez(OUT/('predictor-'+str(seed)+'.npz'),mean=model.mean,std=model.std,
             ym=model.ym,ys=model.ys,**{'p'+str(k):v for k,v in enumerate(model.par)})
    training.append(dict(seed=seed,n=len(Y),mse=trainmse,zero_mse=float(np.mean(Y*Y)),
                         predictor_bytes=model.bytes,cost=cost))
    emit('acquisition_fit',**training[-1])
    for world in range(4):
        for mode in ['local','random','gradient','curvature','replay','learned']:
            row=rollout(seed,world,mode,model);results.append(row);emit('rollout_summary',**row)
summary=dict(status='completed',seconds=time.monotonic()-START,training=training,
             results=results,scope='synthetic constructive pilot; no novelty or superiority established')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
emit('completed',seconds=summary['seconds'],rollouts=len(results))
