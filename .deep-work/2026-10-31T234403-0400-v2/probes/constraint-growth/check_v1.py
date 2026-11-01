"""Finite exploratory path tracker; independent roots used only to verify."""
import json
import time
from pathlib import Path
from fractions import Fraction
import numpy as np

OUT=Path(__file__).resolve().parent
RAW=(OUT/'raw.jsonl').open('x')
START=time.monotonic()

def emit(kind,**kw):
    row=dict(kind=kind,**kw)
    s=json.dumps(row,allow_nan=False)
    assert json.loads(s)['kind']==kind
    RAW.write(s+'\n');RAW.flush()

def packed(z):return [[float(v.real),float(v.imag)] for v in z]
def q(z,u,e):return (z-3)*(z*z-u)-e
def qz(z,u):return 3*z*z-6*z-u
def roots(u,e):return np.roots([1.,-3.,-u,3*u-e])

class Tracker:
    def __init__(self):
        self.steps=0;self.corrector=0;self.oracle=0;self.halves=0
        self.minsep=1e30;self.maxsens=0.;self.maxcorr=0.;self.maxoracle=0.

    def observe(self,z,u,e):
        sep=min(abs(z[a]-z[b]) for a in range(3) for b in range(a))
        self.minsep=min(self.minsep,float(sep))
        den=qz(z,u)
        self.maxsens=max(self.maxsens,float(np.max(np.abs((z-3)/den))))
        corr=float(np.max(np.abs(q(z,u,e)/den)))
        self.maxcorr=max(self.maxcorr,corr)
        rr=roots(u,e);self.oracle+=1
        err=max(min(abs(a-b) for b in rr) for a in z)
        self.maxoracle=max(self.maxoracle,float(err))
        assert sep>1e-7 and np.isfinite(sep)
        assert err<1e-7

    def step(self,z,u,e,v,f,depth=0):
        self.steps+=1
        if self.steps>200000:raise RuntimeError('external finite stepcall budget')
        den=qz(z,u)
        predictor=((z-3)*(v-u)+(f-e))/den
        if np.max(np.abs(predictor))>.08:
            if depth>=12:raise RuntimeError('predictor subdivision exhausted')
            self.halves+=1
            mid=(u+v)/2;em=(e+f)/2
            zz=self.step(z,u,e,mid,em,depth+1)
            return self.step(zz,mid,em,v,f,depth+1)
        zz=z+predictor
        good=False
        for _ in range(12):
            self.corrector+=1
            delta=q(zz,v,f)/qz(zz,v)
            zz=zz-delta
            if np.max(np.abs(delta))<1e-11:
                good=True;break
        if not good or not np.all(np.isfinite(zz)):
            if depth>=12:raise RuntimeError('corrector subdivision exhausted')
            self.halves+=1
            mid=(u+v)/2;em=(e+f)/2
            half=self.step(z,u,e,mid,em,depth+1)
            return self.step(half,mid,em,v,f,depth+1)
        self.observe(zz,v,f)
        return zz

    def migrate(self,e):
        z=np.array([-1.+0j,1.+0j,3.+0j]);u=1.+0j;last=0.
        ts=np.linspace(0,1,65)
        # Unsafe endpoint4 crosses a resting-control degeneracy on the REAL
        # coefficient segment. The declared complex detour keeps migration live.
        es=np.asarray(e*ts,dtype=complex)+(0.3j*np.sin(np.pi*ts) if e==4. else 0.)
        es[-1]=complex(e)
        for f in es[1:]:
            z=self.step(z,u,last,u,complex(f));last=complex(f)
        emit('coefficient_migration',epsilon=e,complex_detour=e==4.,
             initial=packed(np.array([-1.+0j,1.+0j,3.+0j])),final=packed(z))
        return z

    def path(self,z,path,e,tag,save=False):
        z=z.copy();u=complex(path[0]);trail=[]
        for k,v in enumerate(path[1:]):
            z=self.step(z,u,e,complex(v),e);u=complex(v)
            if save and k%16==0:trail.append(dict(index=k,u=[u.real,u.imag],z=packed(z)))
        if save:emit('path_trace',tag=tag,epsilon=e,trace=trail)
        return z

def circle(center,radius,seed,segments=256,phase=0.):
    t=np.linspace(0,2*np.pi,segments+1)
    rr=np.random.default_rng(seed)
    a,b=rr.uniform(-1,1,2)
    perturb=.025*radius*(a*np.sin(2*t)+1j*b*np.sin(3*t))
    ans=center+radius*np.exp(1j*(t+phase))+perturb
    ans[-1]=ans[0]
    return ans

def lollipop(center,radius,seed,segments=256,handle_segments=128):
    c=circle(center,radius,seed,segments=segments,phase=np.pi)
    handle=np.linspace(1.+0j,c[0],handle_segments+1)
    return np.r_[handle,c[1:],handle[-2::-1]]

def permutation(z,base):
    p=[int(np.argmin(np.abs(base-a))) for a in z]
    err=max(abs(z[i]-base[j]) for i,j in enumerate(p))
    assert sorted(p)==[0,1,2] and err<1e-6
    return p,float(err)

assert Fraction(49,64)>Fraction(1,5)
assert Fraction(105,64)>Fraction(49,64)
emit('serialization_smoke',epsilon=[.005,.05,.2],seed=[4401,4402,4403],
     status='analytic growth primitive; no neural acquisition',
     analytic_old_margin=str(Fraction(49,64)),new_margin=str(Fraction(105,64)))
results=[]
for e in [.005,.05,.2,4.]:
    tracker=Tracker();base=tracker.migrate(e)
    # Critical points of Q: 2 z (z-3)^2 + epsilon=0.
    zs=np.roots([2.,-12.,18.,e]);crit=3*zs*zs-6*zs
    upper=complex(crit[np.argmax(crit.imag)])
    near=min(abs(upper-c) for c in crit if abs(upper-c)>1e-8)
    radius=min(.2,float(near)/5)
    cases=[]
    for seed in [4401,4402,4403]:
        a=circle(0.,1.,seed)
        b=lollipop(upper,radius,seed)
        za=tracker.path(base,a,e,'A',save=seed==4401)
        zb=tracker.path(base,b,e,'B',save=seed==4401)
        zab=tracker.path(za,b,e,'AB')
        zba=tracker.path(zb,a,e,'BA')
        zaa=tracker.path(za,a,e,'AA')
        pa,ea=permutation(za,base);pb,eb=permutation(zb,base)
        pab,eab=permutation(zab,base);pba,eba=permutation(zba,base)
        paa,eaa=permutation(zaa,base)
        # A refinement comparison uses an independently finer contour.
        fine_a=circle(0.,1.,seed,segments=512)
        p_fine,_=permutation(tracker.path(base,fine_a,e,'A_fine'),base)
        fine_b=lollipop(upper,radius,seed,segments=512,handle_segments=256)
        pb_fine,_=permutation(tracker.path(base,fine_b,e,'B_fine'),base)
        assert p_fine==pa
        assert pb_fine==pb
        if e<49/64:
            assert pa==[1,0,2] and pb[2]!=2 and pab!=pba and paa==[0,1,2]
        else:
            # Independent critical-value count: the unit disk is simply connected
            # and has no critical values, so every sheet returns to itself.
            assert all(abs(c)>1 for c in crit)
            assert pa==[0,1,2]
        row=dict(epsilon=e,seed=seed,A=pa,B=pb,AB=pab,BA=pba,AA=paa,
                 max_endpoint_error=max(ea,eb,eab,eba,eaa),
                 root_anchors=packed(base),critical_values=packed(crit),
                 new_loop_radius=radius)
        emit('gate_result',**row);cases.append(row)
    result=dict(epsilon=e,cases=cases,metrics=tracker.__dict__.copy(),
                old_margin_condition=e<49/64,constructor_critical_root_solves=1)
    emit('epsilon_result',**result);results.append(result)

# No-coupling control: added branch is exactly inert on the old operation.
t0=Tracker();z0=t0.migrate(0.)
p0,_=permutation(t0.path(z0,circle(0.,1.,4401),0.,'inert'),z0)
assert p0==[1,0,2]
summary=dict(status='completed',seconds=time.monotonic()-START,results=results,
             no_coupling_old_gate=p0,no_coupling_metrics=t0.__dict__,
             scope='growth and retention numerical check under analytic sufficient bound; not learned topology or advantage')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
emit('completed',seconds=summary['seconds'])
