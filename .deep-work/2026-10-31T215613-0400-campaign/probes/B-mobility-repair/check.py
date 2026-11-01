from pathlib import Path
import numpy as np,json,time
p=Path(__file__).parent;c=json.loads((p/'config.json').read_text());t0=time.perf_counter();rng=np.random.default_rng(c['seed'])
N=np.array([[1.,0.],[-1.,1.]])
def mobility(U):
 P=np.eye(2)+U@U.T
 return 2*P/np.trace(P)
def E(X,N=N):return np.sum(1-np.cos(X@N.T),axis=-1)
def rollout(X,P,steps):
 X=X.copy()
 for _ in range(steps):X-=c['step']*(np.sin(X@N.T)@N)@P.T
 return X
train=rng.uniform(-.8,.8,(c['train_queries'],2));U=rng.normal(0,.5,(2,2));trace=[]
def loss(U):return float(np.mean(E(rollout(train,mobility(U),c['train_steps']))))
initial=loss(U)
for ep in range(c['train_epochs']):
 g=np.zeros_like(U)
 for idx in np.ndindex(U.shape):
  d=np.zeros_like(U);d[idx]=c['finite_difference'];g[idx]=(loss(U+d)-loss(U-d))/(2*c['finite_difference'])
 U=np.clip(U-c['learning_rate']*g,-c['parameter_clip'],c['parameter_clip'])
 if ep%20==0 or ep==c['train_epochs']-1:trace.append({'epoch':ep,'loss':loss(U)})
P=mobility(U);H=N.T@N;P_oracle=2*np.linalg.inv(H)/np.trace(np.linalg.inv(H));rng=np.random.default_rng(c['evaluation_seed']);starts=rng.uniform(-np.pi,np.pi,(c['evaluation_queries'],2));out={};records=[]
for name,Q in [('identity',np.eye(2)),('learned',P),('hand_inverse_hessian_diagnostic',P_oracle)]:
 X=starts.copy();settle=np.full(len(X),-1,dtype=int);increases=0;prev=E(X)
 for k in range(c['evaluation_horizon_steps']):
  X-=c['step']*(np.sin(X@N.T)@N)@Q.T
  now=E(X);increases+=int(np.sum(now>prev+1e-12));prev=now
  err=np.max(abs(np.angle(np.exp(1j*X))),axis=1);new=(settle<0)&(err<.01);settle[new]=k+1
 finalerr=np.max(abs(np.angle(np.exp(1j*X))),axis=1)
 out[name]={'P':Q.tolist(),'linear_rates':np.sort(np.real(np.linalg.eigvals(Q@H))).tolist(),'success_count':int(np.sum(finalerr<1e-4)),'queries':len(X),'median_settling_steps_successes':float(np.median(settle[settle>=0])) if np.any(settle>=0) else None,'energy_increase_count':increases,'mobility_extra_scalar_multiplies_per_step_per_query':0 if name=='identity' else 4}
 for i in range(len(X)):records.append({'condition':name,'query':i,'initial':starts[i].tolist(),'settle_steps':int(settle[i]),'final_error':float(finalerr[i])})
N4=np.eye(4)
for j in range(1,4):N4[j,j-1]=-1
P4=np.zeros((4,4));P4[:2,:2]=P;P4[2:,2:]=P;xstar=np.arange(1,5)*2*np.pi/5;X=xstar+np.array([.001,-.002,.003,-.001]);drift=float(np.linalg.norm(-P4@N4.T@np.sin(N4@xstar)))
for _ in range(3000):X-=.05*P4@N4.T@np.sin(N4@X)
winding={'drift_at_wrong_shortcut_fixed_point':drift,'final_output_error':float(abs(np.angle(np.exp(1j*X[-1])))),'final_parent_energy':float(np.sum(1-np.cos(N4@X))),'scope':'block reuse is specified, not learned generalization to four nodes'}
r={'scope':'exploratory learned mobility repair; not new principle or full architecture discovery','config':c,'U':U.tolist(),'P_eigenvalues':np.linalg.eigvalsh(P).tolist(),'training_initial_loss':initial,'training_final_loss':loss(U),'training_loss_evaluations':c['train_epochs']*9+1,'training_phase_steps_estimate':(c['train_epochs']*9+1)*c['train_queries']*c['train_steps'],'evaluation':out,'winding':winding,'elapsed_seconds':time.perf_counter()-t0}
(p/'results.json').write_text(json.dumps(r,indent=2)+'\n');(p/'learning-trace.json').write_text(json.dumps(trace,indent=2)+'\n');(p/'evaluation-records.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(r,indent=2))
assert min(r['P_eigenvalues'])>0
assert all(v['energy_increase_count']==0 for v in out.values())
assert winding['drift_at_wrong_shortcut_fixed_point']>0 and winding['final_output_error']<1e-4
