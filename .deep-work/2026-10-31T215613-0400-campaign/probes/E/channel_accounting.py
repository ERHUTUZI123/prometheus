"""Independent exact expected counts for every physical reaction direction.
No additional training trajectories or learning. Checks channel totals against
previous independently integrated work and heat.
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from kinetic_gate_probe import NEDGE,NAMES,CASES,DURATION,STATES,TARGET,DEST,make_tables,MU,EPS
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'results.json').read_text())
labels=[name+':fuel_forward' for name in NAMES]+[name+':fuel_reverse' for name in NAMES]+['m:passive_down','z:passive_down','m:passive_up','z:passive_up']

def counts(c,case):
    tables=make_tables(c);v=np.zeros(4+len(labels));v[0]=1
    for inp,dt in zip(case[1],DURATION):
        rates,_,_=tables[inp];Q=np.zeros((4,4));A=np.zeros((len(labels),4))
        for s in range(4):
            for e,rate in enumerate(rates[s]):
                Q[s,DEST[s,e]]+=rate;Q[s,s]-=rate
                if e<NEDGE:ch=e+(NEDGE if STATES[s,TARGET[e]] else 0)
                else:
                    j=e-NEDGE;ch=2*NEDGE+j+(0 if STATES[s,j] else 2)
                A[ch,s]+=rate
        H=np.zeros((4+len(labels),4+len(labels)));H[:4,:4]=Q.T;H[4:,:4]=A
        v=expm(H*dt)@v
    cc=v[4:];net=float(sum(cc[:NEDGE]-cc[NEDGE:2*NEDGE]));passive=float(sum(cc[2*NEDGE:2*NEDGE+2]-cc[2*NEDGE+2:]))
    return {'expected_channel_counts':dict(zip(labels,map(float,cc))),
            'chemical_work_kT':MU*net,'heat_kT':(MU-EPS)*net+EPS*passive}

configs=[]
hand=np.zeros(NEDGE);hand[NAMES.index('A->m')]=2;hand[NAMES.index('B+m->z')]=2
configs.append(('hand',hand,r['hand_constructed']))
for ss in r['learning_seeds']:configs.append((str(ss['seed']),np.array([ss['final_material'][n] for n in NAMES]),ss['final']))
out={};errors=[]
for tag,c,ev in configs:
    out[tag]={}
    for case in CASES:
        cr=counts(c,case);out[tag][case[0]]=cr
        for key in ['chemical_work_kT','heat_kT']:errors.append(abs(cr[key]-ev['cases'][case[0]][key]))
report={'configurations':out,'maximum_discrepancy_from_previous_work_heat':max(errors),
        'excluded_costs':['input preparation and chemostats','read and reset','scaffold turnover','learning controller and trace']}
(OUT/'channel_counts.json').write_text(json.dumps(report,indent=2))
print('Maximum independent accounting discrepancy',max(errors))
for ss in r['learning_seeds']:print(ss['seed'],'net fuel',ss['net_training_fuel'],'work kT',ss['chemical_training_work_kT'])
