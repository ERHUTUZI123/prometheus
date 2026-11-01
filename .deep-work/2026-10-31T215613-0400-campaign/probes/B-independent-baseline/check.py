import numpy as np, json, time
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
out=Path(__file__).parent; start=time.perf_counter()
A=np.array([[1.,0.],[-1.,1.]])
def rate(k):return float(np.linalg.eigvalsh(A.T@np.diag(k)@A)[0])
opt=minimize_scalar(lambda t:-rate([2-t,t]),bounds=(1e-7,2-1e-7),method='bounded',options={'xatol':1e-12})
conditions={'uniform_parents':(A,np.array([1.,1.])), 'optimized_parents':(A,np.array([2-opt.x,opt.x])), 'shortcut_v0':(np.vstack([A,[0.,1.]]),np.array([.5,.5,.5]))}
wrap=lambda x:np.angle(np.exp(1j*x))
rng=np.random.default_rng(8801);starts=rng.uniform(-np.pi,np.pi,(32,2));records=[];summaries={}
for name,(N,k) in conditions.items():
 times=[];success=0
 for qi,x0 in enumerate(starts):
  f=lambda t,x: -N.T@(k*np.sin(N@x))
  sol=solve_ivp(f,(0,80),x0,rtol=1e-8,atol=1e-10,max_step=.05)
  e=np.max(np.abs(wrap(sol.y)),axis=0); ids=np.where(e<.01)[0];settle=float(sol.t[ids[0]]) if len(ids) else None
  ok=bool(e[-1]<1e-4);success+=ok
  if settle is not None:times.append(settle)
  records.append({'condition':name,'query':qi,'initial':x0.tolist(),'settling_time':settle,'final_error':float(e[-1]),'success':ok,'nfev':sol.nfev})
 H=N.T@np.diag(k)@N
 summaries[name]={'strengths':k.tolist(),'local_eigenvalues':np.linalg.eigvalsh(H).tolist(),'success_count':success,'query_count':32,'median_settling_successes':float(np.median(times)) if times else None,'cost_proxy':float(np.dot(k,[4.,4.] if len(k)==2 else [4.,4.,8.]))}
N=np.eye(4)
for j in range(1,4):N[j,j-1]=-1
N=np.vstack([N,[0.,0.,0.,1.]])
f=lambda x:-N.T@np.sin(N@x)
xstar=np.arange(1,5)*2*np.pi/5
h=1e-5;J=np.column_stack([(f(xstar+h*np.eye(4)[j])-f(xstar-h*np.eye(4)[j]))/(2*h) for j in range(4)])
sol=solve_ivp(lambda t,x:f(x),(0,150),xstar+np.array([.001,-.002,.003,-.001]),rtol=1e-10,atol=1e-12,max_step=.05)
counter={'force_norm':float(np.linalg.norm(f(xstar))),'fd_jacobian_eigenvalues':np.linalg.eigvalsh(J).tolist(),'final_output_error':float(abs(wrap(sol.y[-1,-1]))),'final_state_distance_to_wrong_equilibrium':float(np.linalg.norm(wrap(sol.y[:,-1]-xstar)))}
result={'scope':'exploratory coordinator check, not blind rebuild','conditions':summaries,'winding_countermodel':counter,'elapsed_seconds':time.perf_counter()-start}
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');(out/'query-records.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(result,indent=2))
assert all(c['success_count']==32 for c in summaries.values())
assert rate([1.2,.8])>.399999
assert counter['force_norm']<1e-10 and max(counter['fd_jacobian_eigenvalues'])<0
assert counter['final_output_error']>1.
