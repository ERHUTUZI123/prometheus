"""AE finite construction/insulation witness. No networking or host execution.
Six-state source probabilities, coupled to explicitly finite sensor/load pools.
New buffer rates are designed, never described as learned here.
"""
from __future__ import annotations
import json, math, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
START=time.perf_counter()
EDATA=json.loads((ROOT/'probes/E/results.json').read_text())
LEARNED=EDATA['learning_seeds'][0]
assert LEARNED['seed']==1301
CNAMES=EDATA['contact_names']
C=np.array([LEARNED['final_material'][n] for n in CNAMES])
EDGES=[]
for target,sources in [(0,[0,1,3]),(1,[0,1,2])]:
    import itertools
    for order in [1,2]:
        for pair in itertools.combinations(sources,order): EDGES.append((target,pair))
STATES=[(m,z) for z in range(3) for m in range(2)]
INDEX={s:i for i,s in enumerate(STATES)}
ZFREE=np.array([z==1 for m,z in STATES],float)
ZBOUND=np.array([z==2 for m,z in STATES],float)
MSTATE=np.array([m for m,z in STATES],float)
CASES=[('AB',[(1,0),(0,0),(0,1)],1),('BA',[(0,1),(0,0),(1,0)],0),
       ('A_only',[(1,0),(0,0),(0,0)],0),('B_only',[(0,0),(0,0),(0,1)],0),
       ('empty',[(0,0),(0,0),(0,0)],0)]
WEIGHTS=np.array([.5,.125,.125,.125,.125])
EPS=8.;MU=16.;GAMMA=.03;ALPHA=4.
YSUM=100.;SSUM=.01;S_KON=100.;S_KOFF=10000.
L_KON=10.;L_KOFF=1.;Y_EPS=16.;Y_MU=32.

def source_table(inp):
    a,b=inp;Q=np.zeros((6,6));work=np.zeros(6)
    for si,(m,z) in enumerate(STATES):
        vals=[a,b,m,int(z==1)]
        def jump(dest,rate,w=0.):
            Q[INDEX[dest],si]+=rate;Q[si,si]-=rate;work[si]+=rate*w
        for (target,pair),material in zip(EDGES,C):
            gate=float(np.prod([vals[k] for k in pair]))
            if target==0:
                jump((1-m,z),ALPHA*material*gate*(1 if m==0 else math.exp(EPS-MU)),MU*(1-2*m))
            elif z<2:
                jump((m,1-z),ALPHA*material*gate*(1 if z==0 else math.exp(EPS-MU)),MU*(1-2*z))
        jump((1-m,z),GAMMA*(1 if m else math.exp(-EPS)))
        if z<2:jump((m,1-z),GAMMA*(1 if z else math.exp(-EPS)))
    return Q,work

def run_case(case,mode,load=0.,gain=50.,fuel=True,tol=1e-8):
    v=np.zeros(10);v[INDEX[(0,0)]]=1.
    ts=[];ys=[]
    for slot,inp in enumerate(case[1]):
        Q,w=source_table(inp)
        def rhs(t,vv):
            p=vv[:6];y,d=vv[6:8];bound=float(ZBOUND@p)
            dp=Q@p
            if mode=='direct':kon=L_KON;koff=L_KOFF;pool=load
            elif mode=='buffer':kon=S_KON;koff=S_KOFF;pool=SSUM
            else:kon=0.;koff=0.;pool=0.
            for m in [0,1]:
                free=INDEX[(m,1)];complex_=INDEX[(m,2)]
                flux=kon*max(0.,pool-bound)*p[free]-koff*p[complex_]
                dp[free]-=flux;dp[complex_]+=flux
            dy=dd=wy=0.
            if mode=='buffer':
                # Kd=100 and a small sensor pool give small input loading.
                # kcat is finite; rapid carrier turnover is fueled, not a read oracle.
                kcat=gain*(S_KOFF/S_KON)/(SSUM*YSUM)
                mu=Y_MU if fuel else 0.
                y0=max(0.,YSUM-y-d)
                drive=kcat*bound*(y0-y*math.exp(Y_EPS-mu))
                reset=gain*(y-y0*math.exp(-Y_EPS))
                loadflux=L_KON*max(0.,load-d)*y-L_KOFF*d
                dy=drive-reset-loadflux;dd=loadflux;wy=mu*drive
            return np.r_[dp,dy,dd,float(w@p),wy]
        grid=np.linspace(float(slot),float(slot+1),101)
        sol=solve_ivp(rhs,(slot,slot+1),v,t_eval=grid,method='BDF',rtol=tol,atol=tol*.01)
        if not sol.success:raise RuntimeError(sol.message)
        if slot:ts.extend(sol.t[1:]);ys.extend(sol.y.T[1:])
        else:ts.extend(sol.t);ys.extend(sol.y.T)
        v=sol.y[:,-1]
    tr=np.array(ys);p=tr[:,:6]
    assert np.max(abs(p.sum(axis=1)-1))<2e-6
    assert np.min(p)>-2e-7
    port=tr[:,6] if mode=='buffer' else p@ZFREE
    return {'time':np.array(ts),'source_total':p@(ZFREE+ZBOUND),'source_free':p@ZFREE,
            'memory':p@MSTATE,'port':port,'work_source':float(v[8]),'work_buffer':float(v[9]),
            'sensor_or_source_bound_final':float(v[:6]@ZBOUND),
            'carrier_free_final':float(v[6]),'carrier_bound_final':float(v[7])}

ISOLATED={case[0]:run_case(case,'isolated') for case in CASES}
def condition(mode,load=0.,gain=50.,fuel=True):
    details={};score=[];errors=[];source_errors=[]
    for case in CASES:
        tr=run_case(case,mode,load,gain,fuel);base=ISOLATED[case[0]]
        err=abs(tr['port']-base['port']);serr=abs(tr['source_total']-base['source_total'])
        terminal=float(tr['port'][-1]);score.append(np.clip(terminal,0,1) if case[2] else 1-np.clip(terminal,0,1))
        errors.append(float(np.mean(err)));source_errors.append(float(np.max(serr)))
        details[case[0]]={'port_final':terminal,'ideal_final':float(base['port'][-1]),
              'mean_abs_port_error':float(np.mean(err)),'max_source_total_error':float(np.max(serr)),
              **{k:tr[k] for k in ['work_source','work_buffer','sensor_or_source_bound_final','carrier_free_final','carrier_bound_final']}}
    return {'mode':mode,'load':load,'gain':gain if mode=='buffer' else None,'fuel':fuel,
       'mean_abs_port_error':float(np.mean(errors)),'max_source_total_error':max(source_errors),
       'analog_balanced_terminal_score':float(np.array(score)@WEIGHTS),'cases':details}

def growth(starts,radius=2):
    # The local transition law has no component labels or prebuilt object slots.
    n=81;r=np.full(n,-1,int);r[starts]=radius;history=[r.tolist()]
    previous=np.zeros(n-1,bool);wall_synthesis=0;wall_removed=0
    for tick in range(radius+1):
        occupied=r>=0;walls=occupied[:-1]!=occupied[1:]
        wall_synthesis+=int(np.sum(walls & ~previous));wall_removed+=int(np.sum(previous & ~walls))
        previous=walls.copy()
        new=r.copy()
        new[1:]=np.maximum(new[1:],r[:-1]-1)
        new[:-1]=np.maximum(new[:-1],r[1:]-1)
        r=new;history.append(r.tolist())
    occupied=r>=0;comp=[];current=[]
    for i,a in enumerate(occupied):
        if a:current.append(i)
        elif current:comp.append(current);current=[]
    if current:comp.append(current)
    return {'starts':starts,'radius':radius,'components':comp,'active_sites':int(occupied.sum()),
      'final_wall_segments':int(np.sum(occupied[:-1]!=occupied[1:])),
      'wall_synthesis_events':wall_synthesis,'wall_disassembly_events':wall_removed,
      'state_carriers_per_type_if_one_per_active_site':int(occupied.sum()),
      'history_remaining_growth_budget':history,
      'interpretation':'Supplied local constructor witness. Component labels are computed after growth only.'}

def assembly():
    rng=np.random.default_rng(55109);p=C/C.sum();dom=np.argsort(p)[-2:];obs=[]
    for count in [100,1000,10000]:
        draws=rng.multinomial(count,p,size=64);rates=4*draws/count
        obs.append({'scaffolds_per_instance':count,'copies':64,
          'mean_L1_recipe_error':float(np.mean(np.sum(abs(rates-C),axis=1))),
          'dominant_contact_names':[CNAMES[i] for i in dom],
          'dominant_contact_empirical_relative_std':(rates[:,dom].std(axis=0,ddof=1)/C[dom]).tolist(),
          'dominant_contact_binomial_relative_std':np.sqrt((1-p[dom])/(count*p[dom])).tolist(),
          'min_probability_contact':CNAMES[int(np.argmin(p))],
          'chance_missing_rarest_contact':float((1-p.min())**count)})
    return obs

res={'model':'six-state source mean-field finite pools, AE v1','inherited_E_seed':1301,
   'inherited_recipe':dict(zip(CNAMES,C.tolist())),
   'parameters':{'source_epsilon':EPS,'source_mu':MU,'source_alpha':ALPHA,'source_gamma':GAMMA,
        'Y_total':YSUM,'sensor_total':SSUM,'sensor_kon':S_KON,'sensor_koff':S_KOFF,
        'load_kon':L_KON,'load_koff':L_KOFF,'Y_epsilon':Y_EPS,'Y_mu':Y_MU},
   'isolated':condition('isolated'),
   'direct':[condition('direct',load) for load in [0.,1.,10.]],
   'buffer':[condition('buffer',load,gain) for gain in [1.,10.,50.] for load in [0.,1.,10.]],
   'buffer_fuel_off':condition('buffer',10.,50.,False),
   'growth':growth([8,27,52]),'growth_collision':growth([8,11]),'assembly':assembly()}
direct=res['direct'][-1];buffer=res['buffer'][-1]
res['frozen_criterion']={'half_direct_port_error':buffer['mean_abs_port_error']<=.5*direct['mean_abs_port_error'],
    'source_error_at_most_0p02':buffer['max_source_total_error']<=.02}
res['frozen_criterion']['both']=all(res['frozen_criterion'].values())
tr1=run_case(CASES[0],'buffer',10.,50.,True,1e-8)
tr2=run_case(CASES[0],'buffer',10.,50.,True,1e-10)
res['tighter_tolerance_check']={'case':'AB, buffer G=50 load=10','max_port_difference':float(np.max(abs(tr1['port']-tr2['port']))),
    'max_source_total_difference':float(np.max(abs(tr1['source_total']-tr2['source_total'])))}
res['runtime_seconds']=time.perf_counter()-START
(OUT/'results_v1.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({'runtime_seconds':res['runtime_seconds'],'frozen_criterion':res['frozen_criterion'],
   'direct_error':direct['mean_abs_port_error'],'buffer_error':buffer['mean_abs_port_error'],
   'buffer_source_error':buffer['max_source_total_error'],'results':'results_v1.json'},indent=2))
