import json
import time
from pathlib import Path
from fractions import Fraction as Q
import numpy as np

out=Path(__file__).resolve().parent
raw=(out/'raw.jsonl').open('x')
def emit(kind,**kw):
    row=dict(kind=kind,**kw);s=json.dumps(row,allow_nan=False)
    assert json.loads(s)==row
    raw.write(s+'\n');raw.flush()
emit('serialization_smoke',seeds=[221,222,223],steps=80)
started=time.monotonic()

def fun(th,x):
    a=th[0]*x[:,0]+th[1]*x[:,1]
    b=th[2]*x[:,2]
    c=th[3]+th[4]*x[:,0]
    y=a*b+c*c+th[5]*x[:,1]*x[:,2]
    jac=np.column_stack((x[:,0]*b,x[:,1]*b,a*x[:,2],2*c,
                         2*c*x[:,0],x[:,1]*x[:,2]))
    return y,jac

fd_th=np.array([.2,.3,-.1,.4,-.3,.2]);fd_x=np.array([[.4,.2,-.5],[-.3,.6,.7]])
_,j=fun(fd_th,fd_x);fd=[]
for k in range(6):
    a=fd_th.copy();b=fd_th.copy();a[k]+=1e-6;b[k]-=1e-6
    fd.append((fun(a,fd_x)[0]-fun(b,fd_x)[0])/2e-6)
fd_error=float(np.max(np.abs(j-np.array(fd).T)))
assert fd_error<1e-5
emit('call_derivative_check',max_error=fd_error)
cases=[]
for seed in [221,222,223]:
    rng=np.random.default_rng(seed)
    owner=np.array([0,0,1,1,1,2]);phi=np.array([.2,-.1,.3])
    theta=phi[owner].copy();mx=0.;birthmx=0.
    for step in range(80):
        x=rng.uniform(-1,1,size=(9,3))
        target=.2*x[:,0]*x[:,2]+.1*x[:,1]**2
        if step>=40:target=target+.1*x[:,0]
        if step in [20,45,65]:
            index={20:1,45:4,65:3}[step]
            y_before=fun(phi[owner],x)[0]
            phi=np.r_[phi,phi[owner[index]]]
            owner[index]=len(phi)-1
            # Full call-state theta does not change on a tying-kernel switch.
            berror=float(np.max(np.abs(fun(phi[owner],x)[0]-y_before)))
            birthmx=max(birthmx,berror)
            emit('fission',seed=seed,step=step,index=index,
                 owner=owner.tolist(),birth_output_error=berror)
        y1,j1=fun(phi[owner],x);y2,j2=fun(theta,x)
        g1=np.mean((y1-target)[:,None]*j1,axis=0)
        g2=np.mean((y2-target)[:,None]*j2,axis=0)
        # Owner-coordinate route: sum all occurrence contributions per owner.
        own_grad=np.bincount(owner,weights=g1,minlength=len(phi))
        phi=phi-.001*own_grad
        # Call-coordinate route: independently apply same-owner blocks.
        kernel=(owner[:,None]==owner[None,:]).astype(float)
        theta=theta-.001*(kernel @ g2)
        error=float(np.max(np.abs(phi[owner]-theta)))
        mx=max(mx,error)
        assert error<1e-10 and np.isfinite(error)
        emit('step',seed=seed,step=step,parameter_error=error,
             output_error=float(np.max(np.abs(y1-y2))),
             owner_count=len(phi),kernel_rank=int(np.linalg.matrix_rank(kernel)))
    cases.append(dict(seed=seed,max_parameter_error=mx,
                      max_birth_output_error=birthmx))

t0=Q(-1);t1=-t0
assert t0*t0==t1*t1==1 and t1==1
assert 2*t0!=0  # local tangent 2*theta*dtheta=0 implies dtheta=0
t=Q(-1,10)
old_gradient=4*t*(t*t-1);new_gradient=2*(t-1)
assert old_gradient*new_gradient<0
assert (t1*t1-1)**2==0 and (t1-1)**2==0
counter=dict(initial=str(t0),finite_edit='theta -> -theta',
             final=str(t1),old_output_before='1',old_output_after='1',
             new_target_satisfied=True,local_feasible_tangent='zero',
             conflicting_gradient_at_minus_point_one=[str(old_gradient),str(new_gradient)],
             common_zero='theta=1')
emit('finite_tied_repair_countermodel',**counter)
result=dict(status='PASS',cases=cases,finite_difference_error=fd_error,
            countermodel=counter,controller_acquisition_tested=False,
            representation_equivalence_scope='given alias schedule, plain SGD',
            stronger_policy_novelty_excluded=False,
            elapsed_seconds=time.monotonic()-started)
p=out/'result.json.tmp';p.write_text(json.dumps(result,indent=2)+'\n');p.replace(out/'result.json')
emit('FINAL_PASS',elapsed_seconds=result['elapsed_seconds']);raw.close()
