#!/usr/bin/env python3
"""Independent finite rebuild; offline, fixed inputs and finite resources."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
import scipy
from scipy.integrate import solve_ivp

OUT = Path(__file__).resolve().parent
N = 4
B = np.eye(N) - np.eye(N, k=-1)
H = B.T @ B
DT = .025
T = 160.
GRID = np.linspace(0., T, 641)
PHASE_TOL = 1e-3
ENERGY_TOL = 1e-6
SEEDS = dict(training=2026103101, evaluation=2026103102,
             initial_vector=2026103103, random_control=2026103104)


def energy(y, weight=0.):
    x = np.asarray(y) @ B.T
    return np.sum(1. - np.cos(x), axis=-1) + weight * (1. - np.cos(np.asarray(y)[..., -1]))


def gradient(y, weight=0.):
    y = np.asarray(y)
    g = np.sin(y @ B.T) @ B
    g[..., -1] += weight * np.sin(y[..., -1])
    return g


def unit(v):
    return v / np.linalg.norm(v)


def metric(v):
    return .5 * np.eye(N) + 2. * np.outer(v, v)


def phase_residual(y):
    return np.max(np.abs(np.arctan2(np.sin(y), np.cos(y))), axis=-1)


def summarize_trajectory(y, weight):
    e = energy(y, weight)
    residual = phase_residual(y)
    ok = (e <= ENERGY_TOL) & (residual <= PHASE_TOL)
    persistent = np.logical_and.accumulate(ok[::-1])[::-1]
    ix = np.flatnonzero(persistent)
    settling = float(GRID[ix[0]]) if ix.size else None
    return dict(energy_final=float(e[-1]), phase_final=float(residual[-1]),
                energy_at_10=float(e[40]), energy_at_25=float(e[100]),
                energy_at_60=float(e[240]), settling_time=settling,
                censored=settling is None, max_sampled_energy_increase=float(np.max(np.diff(e))),
                endpoint=y[-1].tolist())


def main():
    started = time.perf_counter()
    # All seeds, inputs and learning choices are fixed before test execution.
    train = np.random.default_rng(SEEDS['training']).uniform(-np.pi, np.pi, (16, N))
    test = np.random.default_rng(SEEDS['evaluation']).uniform(-np.pi, np.pi, (64, N))
    v0 = unit(np.random.default_rng(SEEDS['initial_vector']).normal(size=N))
    vrandom = unit(np.random.default_rng(SEEDS['random_control']).normal(size=N))
    y = train.copy()
    late = []
    late_steps = set(range(1000, 2401, 200))
    # Baseline, vectorized RK4. Counts below are per state gradient evaluations.
    for step in range(1, 2401):
        k1 = -gradient(y)
        k2 = -gradient(y + DT * k1 / 2)
        k3 = -gradient(y + DT * k2 / 2)
        k4 = -gradient(y + DT * k3)
        y = y + DT * (k1 + 2*k2 + 2*k3 + k4) / 6
        if step in late_steps:
            late.append(gradient(y))
    # Shape starts,times,coordinates; deterministic declared observation order.
    observations = np.stack(late, axis=1)
    v = v0.copy()
    norm_observations = []
    for obs in observations.reshape(-1, N):
        norm = np.linalg.norm(obs)
        if norm == 0.:
            raise RuntimeError('Unexpected zero late gradient; no silent sample replacement')
        u = obs / norm
        norm_observations.append(u)
        v = unit(v + .2*u*np.dot(u, v))
    eig, vec = np.linalg.eigh(H)
    vslow = vec[:, 0]
    metrics = {'baseline': np.eye(N), 'learned': metric(v),
               'analytic_slow': metric(vslow), 'random_orientation': metric(vrandom)}
    training_cost = 16 * (2400 * 4 + 8)
    frozen_learning = dict(initial_vector=v0.tolist(), learned_vector=v.tolist(),
                           random_vector=vrandom.tolist(), analytic_slow_vector=vslow.tolist(),
                           training_gradient_evaluations=training_cost,
                           learning_updates=128, train_starts=train.tolist(),
                           observations=observations.tolist(),
                           learned_slow_abs_cosine=float(abs(np.dot(v, vslow))),
                           metrics={k:m.tolist() for k,m in metrics.items()})
    # Persist learning before evaluation (the script never reads outcomes to update v).
    (OUT / 'frozen_learning.json').write_text(json.dumps(frozen_learning, indent=2))
    checks = {}
    rng = np.random.default_rng(908817)
    pts = rng.uniform(-np.pi, np.pi, (10, N))
    for weight in [0., .8]:
        fd = np.empty_like(pts)
        eps = 1e-6
        for j in range(N):
            shift = np.eye(N)[j] * eps
            fd[:, j] = (energy(pts+shift, weight) - energy(pts-shift, weight))/(2*eps)
        checks[f'gradient_fd_error_weight_{weight}'] = float(np.max(np.abs(fd-gradient(pts,weight))))
    critical = []
    for bits in itertools.product([0,1], repeat=N):
        x = np.pi*np.array(bits)
        theta = np.linalg.solve(B, x)
        hess = B.T @ np.diag(np.cos(x)) @ B
        lam = np.linalg.eigvalsh(hess)
        critical.append(dict(bits=bits, grad_norm=float(np.linalg.norm(gradient(theta))),
                             negative_eigenvalues=int(np.sum(lam<0)), expected_negative=sum(bits),
                             hessian_eigenvalues=lam.tolist()))
    checks['critical_points'] = critical
    checks['metric_spectra'] = {k:np.linalg.eigvalsh(m).tolist() for k,m in metrics.items()}
    checks['parent_H_eigenvalues'] = eig.tolist()
    assert all(c['negative_eigenvalues'] == c['expected_negative'] for c in critical)
    assert max(c['grad_norm'] for c in critical) < 1e-12
    assert max(checks[k] for k in checks if k.startswith('gradient_fd')) < 1e-8
    assert all(np.min(np.linalg.eigvalsh(m)) >= .499999 for m in metrics.values())
    cases = []
    samples = {}
    dop_started = time.perf_counter()
    for weight in [0., .8]:
        for name, m in metrics.items():
            for idx, start in enumerate(test):
                sol = solve_ivp(lambda t,q: -m@gradient(q,weight), (0., T), start,
                                method='DOP853', rtol=1e-10, atol=1e-12, t_eval=GRID)
                if not sol.success:
                    raise RuntimeError(sol.message)
                record = summarize_trajectory(sol.y.T, weight)
                record.update(weight=weight, metric=name, start_index=idx, nfev=sol.nfev)
                cases.append(record)
                samples[f'w{weight}_{name}_{idx}'] = sol.y.T
    dop_seconds = time.perf_counter()-dop_started
    # Independent explicit-Euler execution, vectorized over the 64 held-out starts.
    cross = []
    euler_started = time.perf_counter()
    for weight in [0., .8]:
        for name,m in metrics.items():
            y = test.copy()
            stored = [y.copy()]
            for step in range(1, 6401):
                y = y - DT * (gradient(y,weight) @ m.T)
                if step % 10 == 0:
                    stored.append(y.copy())
            arr = np.stack(stored)
            for idx in range(64):
                record = summarize_trajectory(arr[:,idx,:],weight)
                ref = samples[f'w{weight}_{name}_{idx}']
                phase_diff = np.arctan2(np.sin(arr[:,idx,:]-ref),np.cos(arr[:,idx,:]-ref))
                record.update(weight=weight,metric=name,start_index=idx,gradient_evaluations=6400,
                              max_trajectory_phase_error=float(np.max(np.abs(phase_diff))),
                              terminal_phase_error=float(np.max(np.abs(phase_diff[-1]))))
                cross.append(record)
    euler_seconds = time.perf_counter()-euler_started
    aggregate = []
    for weight in [0., .8]:
        base = [r for r in cases if r['weight']==weight and r['metric']=='baseline']
        base_t = np.array([T if r['censored'] else r['settling_time'] for r in base])
        for name in metrics:
            rows = [r for r in cases if r['weight']==weight and r['metric']==name]
            ers = [r for r in cross if r['weight']==weight and r['metric']==name]
            observed = [r['settling_time'] for r in rows if not r['censored']]
            restricted = np.array([T if r['censored'] else r['settling_time'] for r in rows])
            nfev = np.array([r['nfev'] for r in rows])
            time_disagreements = [abs(r['settling_time']-er['settling_time'])
                                  for r,er in zip(rows,ers) if not(r['censored'] or er['censored'])]
            aggregate.append(dict(weight=weight,metric=name,n=64,successes=len(observed),
                                  median_settling_successes=float(np.median(observed)) if observed else None,
                                  mean_restricted_settling=float(np.mean(restricted)),
                                  median_paired_restricted_speedup=float(np.median(base_t/restricted)),
                                  faster_than_baseline=int(np.sum(restricted<base_t)),
                                  slower_than_baseline=int(np.sum(restricted>base_t)),
                                  mean_nfev=float(np.mean(nfev)), total_nfev=int(np.sum(nfev)),
                                  max_final_phase=float(max(r['phase_final'] for r in rows)),
                                  max_final_energy=float(max(r['energy_final'] for r in rows)),
                                  max_sampled_energy_increase=float(max(r['max_sampled_energy_increase'] for r in rows)),
                                  euler_successes=sum(not er['censored'] for er in ers),
                                  euler_max_settling_discrepancy=max(time_disagreements,default=None),
                                  euler_max_terminal_phase_error=max(er['terminal_phase_error'] for er in ers)))
    result = dict(specification='branches/rebuild/specification.md', seeds=SEEDS,
                  versions={'numpy':np.__version__,'scipy':scipy.__version__},
                  tolerances={'phase':PHASE_TOL,'energy':ENERGY_TOL,'T':T,'grid_spacing':.25,
                              'DOP853_rtol':1e-10,'DOP853_atol':1e-12,'Euler_h':DT},
                  test_starts=test.tolist(), learning=frozen_learning, checks=checks,
                  aggregate=aggregate, cases=cases, euler_cases=cross,
                  resource={'training_gradient_evaluations':training_cost,
                            'training_metric_vector_updates':128,
                            'dop853_state_gradient_evaluations':sum(r['nfev'] for r in cases),
                            'euler_state_gradient_evaluations':len(cross)*6400,
                            'DOP853_wall_seconds':dop_seconds,'Euler_wall_seconds':euler_seconds,
                            'total_wall_seconds':time.perf_counter()-started})
    (OUT / 'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    np.savez_compressed(OUT / 'dop853_samples.npz', **samples)
    print(json.dumps({'learned_slow_abs_cosine':frozen_learning['learned_slow_abs_cosine'],
                      'aggregate':aggregate,'resource':result['resource']},indent=2))


if __name__ == '__main__':
    main()
