"""Inspected finite replay of quotient-defined nonlinear promotion; no network."""
import json
import time
from pathlib import Path
from fractions import Fraction as F
import numpy as np

OUT = Path(__file__).resolve().parent
CFG = dict(seeds=[9101, 9102, 9103], m=2, n=3, width=128,
           batch=24, steps=100, eta=0.005, promotions=[8, 55])
start = time.monotonic()
raw = (OUT / 'raw.jsonl').open('x')

def emit(kind, **kw):
    row = dict(kind=kind, **kw)
    s = json.dumps(row, allow_nan=False, sort_keys=True)
    assert json.loads(s) == row
    raw.write(s + '\n')
    raw.flush()

emit('serialization_smoke', config=CFG)

def gram(st):
    return st['H'] if st['kind'] == 'compact' else st['Z'] @ st['Z'].T

def forward(st, x):
    h = gram(st)
    y = x @ h[:2, 2:].T
    for v, u, c in st['paths']:
        z = x @ u
        y = y + (z + c * z*z)[:, None] * v[None, :]
    return y

def selector(x, error):
    # Current observations only; no target program or hidden unit coordinates.
    a = -np.mean(error, axis=0)
    if np.linalg.norm(a) < 1e-12:
        a = np.array([1.0, 0.0])
    a = a / np.linalg.norm(a)
    score = -(error @ a)
    cmat = np.einsum('b,bi,bj->ij', score, x, x) / len(x)
    vals, vecs = np.linalg.eigh(cmat)
    b = vecs[:, np.argmax(np.abs(vals))]
    j = np.argmax(np.abs(b))
    if b[j] < 0:
        b = -b
    return np.r_[a, b] / np.sqrt(2.0)

def promote(st, t):
    h = gram(st)
    q = float(t @ h @ t)
    assert q > 1e-12
    r_h = h @ t / np.sqrt(q)
    if st['kind'] == 'compact':
        r = r_h
        st['H'] = h - np.outer(r, r)
        st['H'] = (st['H'] + st['H'].T) / 2
    else:
        z = st['Z']
        s = z.T @ t / np.sqrt(q)
        assert abs(float(s @ s)-1) < 1e-10
        r = z @ s
        if st['kind'] == 'minimum_factor':
            # Ordinary orthogonal factor rotation, then delete extracted column.
            e = np.zeros(len(s)); e[-1] = 1
            w = e-s
            if w @ w > 1e-24:
                zrot = z-2*np.outer(z @ w, w)/(w @ w)
            else:
                zrot = z.copy()
            assert np.max(np.abs(zrot[:, -1]-r)) < 1e-10
            st['Z'] = zrot[:, :-1]
        else:
            st['Z'] = z - np.outer(r, s)
    assert np.max(np.abs(r-r_h)) < 1e-10
    st['paths'].append([r[:2].copy(), r[2:].copy(), 0.0])
    return q

def derivatives(st, x, target):
    pred = forward(st, x)
    err = pred-target
    g = err.T @ x / len(x)
    gs = []
    for v, u, c in st['paths']:
        z = x @ u
        ev = err @ v
        dv = err.T @ (z+c*z*z) / len(x)
        du = x.T @ (ev * (1+2*c*z)) / len(x)
        dc = float(np.mean(ev*z*z))
        gs.append((dv, du, dc))
    return pred, err, g, gs

def apply(st, g, gs):
    eta = CFG['eta']
    if st['kind'] == 'compact':
        mat = np.eye(5)
        mat[:2, 2:] = -eta*g
        mat[2:, :2] = -eta*g.T
        st['H'] = mat @ st['H'] @ mat.T
        st['H'] = (st['H']+st['H'].T)/2
    else:
        # Direct OLD-factor updates, independent of covariance update code.
        v = st['Z'][:2].copy(); ut = st['Z'][2:].copy()
        st['Z'] = np.vstack((v-eta*g @ ut, ut-eta*g.T @ v))
    for path, (dv, du, dc) in zip(st['paths'], gs):
        path[0] = path[0]-eta*dv
        path[1] = path[1]-eta*du
        path[2] = path[2]-eta*dc

def path_distance(a, b):
    assert len(a['paths']) == len(b['paths'])
    return max([0.0]+[float(np.max(np.abs(np.r_[*p[:2],p[2]] -
                                                    np.r_[*q[:2],q[2]])))
                      for p, q in zip(a['paths'], b['paths'])])

def target_for(x, phase):
    y = np.column_stack((x[:,0]**2+0.5*x[:,1]*x[:,2],
                         x[:,1]**2-0.3*x[:,0]*x[:,2]))
    return y if phase == 0 else y[:, ::-1]*np.array([0.8, 1.2])

# Independently evaluate analytic explicit-path derivatives by finite differences.
fdst = dict(kind='compact', H=np.eye(5)*0.2,
            paths=[[np.array([0.4,-0.3]), np.array([0.2,0.5,-0.4]),0.6]])
fdx = np.array([[0.2,-0.7,0.1],[-0.5,0.4,0.8],[0.7,0.3,-0.6]])
fdy = target_for(fdx,0)
_, _, _, fdgs = derivatives(fdst,fdx,fdy)
analytic = np.r_[*fdgs[0][:2],fdgs[0][2]]
theta = np.r_[*fdst['paths'][0][:2],fdst['paths'][0][2]]
numeric=[]
def loss_at(w):
    s=dict(kind='compact',H=fdst['H'],paths=[[w[:2],w[2:5],float(w[5])]])
    return float(np.sum((forward(s,fdx)-fdy)**2)/(2*len(fdx)))
for j in range(len(theta)):
    p=theta.copy();p[j]+=1e-6
    q=theta.copy();q[j]-=1e-6
    numeric.append((loss_at(p)-loss_at(q))/2e-6)
fd_error=float(np.max(np.abs(analytic-numeric)))
emit('finite_difference_path_gradient', max_error=fd_error)
assert fd_error < 1e-5

# An old-interface-equivalent source pair cannot support arbitrary neuron ReLU.
relu_a=F(1)
relu_b=F(3,5)*max(F(3,5),F(0))+F(-4,5)*max(F(-4,5),F(0))
assert relu_a == 1 and relu_b == F(9,25)
emit('non_descending_action', same_H=True, relu_outputs=[str(relu_a),str(relu_b)])

cases=[]
for seed in CFG['seeds']:
    rng=np.random.default_rng(seed)
    z=rng.normal(size=(5,CFG['width']))/np.sqrt(CFG['width'])
    h=z @ z.T
    vals,vecs=np.linalg.eigh(h)
    assert np.min(vals)>1e-10
    minimal=vecs*np.sqrt(vals)[None,:]
    states=[dict(kind='source',Z=z.copy(),paths=[]),
            dict(kind='compact',H=h.copy(),paths=[]),
            dict(kind='minimum_factor',Z=minimal,paths=[])]
    maxerr=dict(output=0.0,boundary_gradient=0.0,residual=0.0,
                path_parameters=0.0,birth_output=0.0,selector=0.0)
    births=[]
    for step in range(CFG['steps']):
        x=rng.uniform(-1,1,size=(CFG['batch'],3))
        target=target_for(x,int(step>=50))
        if step in CFG['promotions']:
            before=[forward(s,x) for s in states]
            ts=[selector(x,y-target) for y in before]
            serr=max(float(np.max(np.abs(t-ts[0]))) for t in ts)
            maxerr['selector']=max(maxerr['selector'],serr)
            rank_before=int(np.linalg.matrix_rank(gram(states[0]),tol=1e-9))
            qvals=[promote(s,t) for s,t in zip(states,ts)]
            berr=max(float(np.max(np.abs(forward(s,x)-y))) for s,y in zip(states,before))
            maxerr['birth_output']=max(maxerr['birth_output'],berr)
            rank_after=int(np.linalg.matrix_rank(gram(states[0]),tol=1e-9))
            assert rank_after==rank_before-1
            assert berr < 1e-10
            dcs=[derivatives(s,x,target)[3][-1][2] for s in states]
            birth=dict(step=step,q=qvals[0],output_error=berr,
                       residual_rank_before=rank_before,residual_rank_after=rank_after,
                       nonlinear_gate_derivative=dcs[0],
                       minimum_residual_width=states[2]['Z'].shape[1])
            births.append(birth);emit('promotion',seed=seed,**birth)
        drs=[derivatives(s,x,target) for s in states]
        oe=max(float(np.max(np.abs(d[0]-drs[0][0]))) for d in drs)
        ge=max(float(np.max(np.abs(d[2]-drs[0][2]))) for d in drs)
        for s,d in zip(states,drs):apply(s,d[2],d[3])
        he=max(float(np.max(np.abs(gram(s)-gram(states[0])))) for s in states)
        pe=max(path_distance(s,states[0]) for s in states)
        for key,value in [('output',oe),('boundary_gradient',ge),('residual',he),('path_parameters',pe)]:
            maxerr[key]=max(maxerr[key],value)
        assert max(maxerr.values()) < 1e-9
        assert float(np.min(np.linalg.eigvalsh(gram(states[1])))) > -1e-9
        emit('step',seed=seed,step=step,output_error=oe,boundary_gradient_error=ge,
             residual_error=he,path_error=pe,
             source_loss=float(np.sum(drs[0][1]**2)/(2*len(x))),
             explicit_gates=[float(p[2]) for p in states[0]['paths']])
    probe=np.vstack((np.zeros((1,3)),0.7*np.eye(3),-0.7*np.eye(3)))
    yp=forward(states[1],probe)
    nonlin=float(np.max(np.abs(yp[1:4]+yp[4:7]-2*yp[0])))
    # Removing explicit nonlinear edit state is deliberately wrong, not a fair rival.
    folded=gram(states[1])[:2,2:].copy()
    for v,u,c in states[1]['paths']:folded+=np.outer(v,u)
    wrongfold=float(np.max(np.abs(yp-probe @ folded.T)))
    assert nonlin > 1e-8
    assert wrongfold > 1e-8
    assert any(abs(b['nonlinear_gate_derivative']) > 1e-8 for b in births)
    path_scalars=2*(2+3+1)
    case=dict(seed=seed,max_errors=maxerr,births=births,
              nonlinear_witness=nonlin,wrong_fold_output_error=wrongfold,
              state_scalars=dict(source=5*CFG['width']+path_scalars,
                                 compact_symmetric=15+path_scalars,
                                 minimum_factor=int(states[2]['Z'].size)+path_scalars),
              initial_factorization='one symmetric eigendecomposition of5x5H',
              selector_calls_per_state=2,selector_eigendecomposition_dimension=3)
    cases.append(case);emit('case_complete',**case)
result=dict(status='PASS',config=CFG,cases=cases,finite_difference_error=fd_error,
            strongest_ordinary_alternative_matched=True,
            learned_neural_constructor=False,novelty_established=False,
            measured_speedup=False,elapsed_seconds=time.monotonic()-start)
tmp=OUT/'result.json.tmp';tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
tmp.replace(OUT/'result.json')
emit('FINAL_PASS',cases=len(cases),elapsed_seconds=result['elapsed_seconds'])
raw.close()
