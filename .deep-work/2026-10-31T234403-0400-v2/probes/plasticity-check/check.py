import json
import time
from pathlib import Path
import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
CONFIG = {'seeds': [413, 414, 415], 'hidden_width': 512, 'output': 3,
          'module_input': 2, 'external_input': 4, 'steps': 180,
          'eta': .015, 'growth_steps': [60, 120]}
# Serialization check precedes any trajectories.
assert json.loads(json.dumps(CONFIG)) == CONFIG
(ROOT / 'config.json').write_text(json.dumps(CONFIG, indent=2) + '\n')
started = time.monotonic()
cases = []
with (ROOT / 'steps.jsonl').open('x') as raw:
    for seed in CONFIG['seeds']:
        rng = np.random.default_rng(seed)
        m, n, h = CONFIG['output'], CONFIG['module_input'], CONFIG['hidden_width']
        A = rng.normal(size=(m, h)) * .2 / np.sqrt(h)
        B = rng.normal(size=(h, n)) * .2 / np.sqrt(h)
        J = rng.normal(size=(n, CONFIG['external_input'])) * .2
        K = J.copy()
        H = np.block([[A @ A.T, A @ B], [B.T @ A.T, B.T @ B]])
        targets = rng.normal(size=(3, m, CONFIG['external_input'])) * .3
        maxerr = {'output': 0., 'block_state': 0., 'upstream': 0., 'growth_value': 0.}
        injections = []
        for step in range(CONFIG['steps']):
            if step in CONFIG['growth_steps']:
                Lo = rng.normal(size=(m, 2)) * .1
                Li = rng.normal(size=(n, 2)) * .1
                old_W = A @ B
                A = np.concatenate([A, Lo, np.zeros((m, 2))], axis=1)
                B = np.concatenate([B, np.zeros((2, n)), Li.T], axis=0)
                D = np.block([[Lo @ Lo.T, np.zeros((m, n))],
                              [np.zeros((n, m)), Li @ Li.T]])
                H = H + D
                growth_error = float(np.max(np.abs(A @ B - old_W)))
                maxerr['growth_value'] = max(maxerr['growth_value'], growth_error)
                injections.append({'step': step, 'new_paths': 4,
                                   'trace_D': float(np.trace(D)), 'current_value_error': growth_error})
            x = rng.normal(size=CONFIG['external_input'])
            target = np.tanh(targets[step // 60] @ x)
            xm, xc = np.tanh(J @ x), np.tanh(K @ x)
            Wm, Wc = A @ B, H[:m, m:]
            ym, yc = np.tanh(Wm @ xm), np.tanh(Wc @ xc)
            dm, dc = (ym - target) * (1 - ym**2), (yc - target) * (1 - yc**2)
            Gm, Gc = np.outer(dm, xm), np.outer(dc, xc)
            dJ = np.outer((Wm.T @ dm) * (1 - xm**2), x)
            dK = np.outer((Wc.T @ dc) * (1 - xc**2), x)
            eta = CONFIG['eta']
            old_A, old_B = A, B
            A = old_A - eta * Gm @ old_B.T
            B = old_B - eta * old_A.T @ Gm
            M = np.block([[np.eye(m), -eta * Gc], [-eta * Gc.T, np.eye(n)]])
            H = M @ H @ M.T
            J, K = J - eta * dJ, K - eta * dK
            oracle = np.block([[A @ A.T, A @ B], [B.T @ A.T, B.T @ B]])
            errs = {'output': float(np.max(np.abs(ym-yc))),
                    'block_state': float(np.max(np.abs(oracle-H))),
                    'upstream': float(np.max(np.abs(J-K)))}
            for key, value in errs.items():
                maxerr[key] = max(maxerr[key], value)
            raw.write(json.dumps({'seed': seed, 'step': step, 'errors': errs,
                                  'loss_micro': float(np.sum((ym-target)**2)/2),
                                  'loss_macro': float(np.sum((yc-target)**2)/2)}) + '\n')
            raw.flush()
        assert max(maxerr.values()) < 1e-10, maxerr
        cases.append({'seed': seed, 'max_errors': maxerr, 'injections': injections,
                      'final_width': A.shape[1], 'micro_block_scalars': A.size+B.size,
                      'dense_macro_scalars': H.size, 'symmetric_macro_scalars': (m+n)*(m+n+1)//2})

# Independently exact/algebraic factor dynamics for the two-step witness.
eta = sp.Rational(1, 4)
G1, G2 = sp.diag(1, 2), sp.Matrix([[0, 1], [0, 0]])
I, Z = sp.eye(2), sp.zeros(2)
exact = []
for s, t in [(2, 2), (3, 1)]:
    A = (sp.sqrt(s)*I).row_join(Z)
    B = Z.col_join(sp.sqrt(t)*I)
    H = sp.diag(s*I, t*I)
    first = None
    for g in [G1, G2]:
        old_A, old_B = A, B
        A, B = old_A - eta*g*old_B.T, old_B - eta*old_A.T*g
        M = I.row_join(-eta*g).col_join((-eta*g.T).row_join(I))
        H = sp.simplify(M*H*M.T)
        assert sp.simplify(A*B-H[:2, 2:]) == Z
        if first is None:
            first = A*B
    exact.append({'initial_S': s, 'initial_T': t, 'initial_value': 'zero',
                  'initial_gradient_response': f'{s+t} * G', 'first_W': str(first),
                  'second_W': str(sp.simplify(A*B)), 'second_W_12': str(sp.simplify((A*B)[0, 1]))})
assert exact[0]['first_W'] == exact[1]['first_W']
assert exact[0]['second_W_12'] != exact[1]['second_W_12']
result = {'config': CONFIG, 'cases': cases, 'exact_two_step_witness': exact,
          'status': 'finite micro/macro equivalence and exact witness verified',
          'automatic_state_discovery_tested': False, 'meta_growth_learning_tested': False,
          'novelty_verified': False, 'elapsed_seconds': time.monotonic()-started}
temp = ROOT / 'results.pending.json'
temp.write_text(json.dumps(result, indent=2) + '\n')
temp.replace(ROOT / 'results.json')
print(json.dumps(result))
