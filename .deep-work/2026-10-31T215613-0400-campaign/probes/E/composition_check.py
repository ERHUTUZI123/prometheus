"""Exact check of hand-specified downstream composition, not learning.
Old 12 contacts retained exactly. Added one site, one specified kinetic gate,
and two material units (+50%). Nine timing combinations are held out of training.
The ideal catalytic model gives zero back-action by construction; a separate
binding model in kinetic_gate_probe.py challenges that assumption.
"""
import json,itertools
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from kinetic_gate_probe import EPS,MU,GAMMA,ALPHA,NAMES,EDGES
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'results.json').read_text())

def run(c,template,g1,g2):
    # template has external ligand in A slot / B slot / C slot
    phases=[(template[0],1.),((0,0,0),g1),(template[1],1.),((0,0,0),g2),(template[2],1.)]
    p=np.zeros(8);p[0]=1.;old_at_B=None
    for phase,(u,dt) in enumerate(phases):
        Q=np.zeros((8,8))
        for s in range(8):
            bits=[s&1,(s>>1)&1,(s>>2)&1]
            vals=[u[0],u[1],bits[0],bits[1]]
            for e,(j,sources) in enumerate(EDGES):
                g=np.prod([vals[k] for k in sources]);rate=ALPHA*c[e]*g*(1 if not bits[j] else np.exp(EPS-MU))
                dest=s^(1<<j);Q[s,dest]+=rate;Q[s,s]-=rate
            # New q is set by z AND C; its single scaffold is supplied, not discovered.
            rate=8.*bits[1]*u[2]*(1 if not bits[2] else np.exp(EPS-MU))
            Q[s,s^4]+=rate;Q[s,s]-=rate
            for j in range(3):
                rate=GAMMA*(1 if bits[j] else np.exp(-EPS));Q[s,s^(1<<j)]+=rate;Q[s,s]-=rate
        p=p@expm(Q*dt)
        if phase==2:old_at_B=float(sum(p[s] for s in range(8) if s&2))
    return float(sum(p[s] for s in range(8) if s&4)),old_at_B

templates=[('ABC',[(1,0,0),(0,1,0),(0,0,1)],1),('BAC',[(0,1,0),(1,0,0),(0,0,1)],0),
           ('AB_no_C',[(1,0,0),(0,1,0),(0,0,0)],0),('A_no_B_C',[(1,0,0),(0,0,0),(0,0,1)],0),
           ('no_A_BC',[(0,0,0),(0,1,0),(0,0,1)],0)]
out=[]
for seedrec in r['learning_seeds']:
    c=np.array([seedrec['final_material'][n] for n in NAMES]);timings=[]
    for g1,g2 in itertools.product([.5,1.,1.5],repeat=2):
        probs={name:run(c,seq,g1,g2)[0] for name,seq,y in templates}
        ba=.5*probs['ABC']+.125*sum(1-probs[name] for name,seq,y in templates if not y)
        timings.append({'gap_AB':g1,'gap_BC':g2,'balanced_accuracy':ba,'probabilities':probs})
    out.append({'seed':seedrec['seed'],'mean_balanced_accuracy':float(np.mean([t['balanced_accuracy'] for t in timings])),
                'minimum_balanced_accuracy':min(t['balanced_accuracy'] for t in timings),'timings':timings})
(OUT/'composition_results.json').write_text(json.dumps({'new_material':2,'old_material':4,'new_state_count':1,
 'new_gate':'z AND C -> q, supplied explicitly','learned_new_composition':False,'ideal_no_back_action_is_assumed':True,'seeds':out},indent=2))
for s in out:print(s['seed'],s['mean_balanced_accuracy'],s['minimum_balanced_accuracy'])
