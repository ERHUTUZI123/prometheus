"""Exact local test of constructed learning-state closure; stdlib, no external data."""
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path

ROOT = Path(__file__).parent

def emit(r):
    print(json.dumps(r, sort_keys=True), flush=True)

def clean(p):
    return {m:c for m,c in p.items() if c}

def padd(a,b):
    r=dict(a)
    for m,c in b.items(): r[m]=r.get(m,F(0))+c
    return clean(r)

def pscale(a,c): return clean({m:v*c for m,v in a.items()})

def pmul(a,b):
    r={}
    for m,c in a.items():
        for n,d in b.items():
            k=tuple(x+y for x,y in zip(m,n))
            r[k]=r.get(k,F(0))+c*d
    return clean(r)

def var(n,i):
    m=[0]*n; m[i]=1
    return {tuple(m):F(1)}

def embedded(p,nc): return {m+(0,)*nc:c for m,c in p.items()}

def pullback(p,subs,ns,nc):
    total={}
    for mon,c in p.items():
        term={(0,)*(ns+nc):c}
        for j,e in enumerate(mon):
            for _ in range(e): term=pmul(term,subs[j])
        total=padd(total,term)
    groups={}
    for mon,c in total.items():
        control=mon[ns:]; state=mon[:ns]
        groups.setdefault(control,{})[state]=c
    return groups

def serial_poly(p):
    return [{'powers':list(m),'coefficient':[c.numerator,c.denominator]}
            for m,c in sorted(p.items())]

def discover(initial,subs,names,nc,tag):
    ns=len(names); basis=[]; pivots=[]; events=[]
    def reduce(p):
        r=dict(p); coordinates=[F(0)]*len(basis)
        for j,(b,pivot) in enumerate(zip(basis,pivots)):
            c=r.get(pivot,F(0))
            coordinates[j]=c
            r=padd(r,pscale(b,-c))
        return clean(r),coordinates
    def insert(p,origin):
        r,_=reduce(p)
        if not r: return False
        pivot=min(r); scale=r[pivot]
        r=pscale(r,1/scale)
        basis.append(r); pivots.append(pivot)
        events.append({'index':len(basis)-1,'origin':origin,
                       'statistic':serial_poly(r)})
        return True
    for j,p in enumerate(initial): insert(p,{'output':j})
    j=0
    while j<len(basis):
        if len(basis)>ns*(ns+1)//2:
            raise AssertionError('quadratic closure exceeded ambient dimension')
        groups=pullback(basis[j],subs,ns,nc)
        for control,coef in sorted(groups.items()):
            insert(coef,{'pullback_of':j,'control_powers':list(control)})
        j+=1
    updates=[]
    for j,b in enumerate(basis):
        record=[]
        for control,coef in sorted(pullback(b,subs,ns,nc).items()):
            r,coordinates=reduce(coef)
            assert not r
            record.append({'control_powers':list(control),
                           'state_coefficients':[[x.numerator,x.denominator]
                                                 for x in coordinates]})
        updates.append(record)
    result={'tag':tag,'source_variable_names':names,'dimension':len(basis),
            'events':events,'closed_updates':updates,
            'total_statistic_monomials':sum(len(b) for b in basis)}
    (ROOT/(tag+'-closure.json')).write_text(json.dumps(result,indent=2)+'\n')
    emit({'stage':'closure_discovered','tag':tag,'dimension':len(basis),
          'source_dimension':ns,'statistic_monomials':result['total_statistic_monomials']})
    return result

def scalar_discovery():
    m=3; ns=2*m; nc=1; n=ns+nc
    states=[var(ns,i) for i in range(ns)]
    full=[var(n,i) for i in range(ns)]; control=var(n,ns)
    subs=[]
    for i in range(m): subs.append(padd(full[i],pscale(pmul(control,full[m+i]),-1)))
    for i in range(m): subs.append(padd(full[m+i],pscale(pmul(control,full[i]),-1)))
    output={}
    for i in range(m): output=padd(output,pmul(states[i],states[m+i]))
    r=discover([output],subs,['a'+str(i) for i in range(m)]+
               ['b'+str(i) for i in range(m)],nc,'scalar')
    assert r['dimension']==2

def matrix_discovery():
    # Input is only the factor update program and output coordinates, not Gram targets.
    do=di=h=2; ns=do*h+h*di; nc=do*di; n=ns+nc
    names=['v'+str(i)+str(k) for i in range(do) for k in range(h)]
    names+=['u'+str(k)+str(j) for k in range(h) for j in range(di)]
    sv=[var(ns,i) for i in range(ns)]; fv=[var(n,i) for i in range(ns)]
    g=[var(n,ns+i) for i in range(nc)]
    vid=lambda i,k:i*h+k
    uid=lambda k,j:do*h+k*di+j
    subs=[]
    for i in range(do):
        for k in range(h):
            p=fv[vid(i,k)]
            for j in range(di): p=padd(p,pscale(pmul(g[i*di+j],fv[uid(k,j)]),-1))
            subs.append(p)
    for k in range(h):
        for j in range(di):
            p=fv[uid(k,j)]
            for i in range(do): p=padd(p,pscale(pmul(fv[vid(i,k)],g[i*di+j]),-1))
            subs.append(p)
    outputs=[]
    for i in range(do):
        for j in range(di):
            p={}
            for k in range(h): p=padd(p,pmul(sv[vid(i,k)],sv[uid(k,j)]))
            outputs.append(p)
    r=discover(outputs,subs,names,nc,'matrix')
    assert r['dimension']==10
    return r

def mat(a): return [[F(x) for x in row] for row in a]
def zeros(r,c): return [[F(0) for _ in range(c)] for _ in range(r)]
def ident(n): return [[F(i==j) for j in range(n)] for i in range(n)]
def tr(a): return [list(row) for row in zip(*a)]
def add(a,b): return [[x+y for x,y in zip(r,s)] for r,s in zip(a,b)]
def scale(a,c): return [[x*c for x in row] for row in a]
def mm(a,b):
    bt=tr(b)
    return [[sum((x*y for x,y in zip(row,col)),F(0)) for col in bt] for row in a]
def serial(a): return [[[x.numerator,x.denominator] for x in row] for row in a]
def blocks(a,b,c,d): return [x+y for x,y in zip(a,b)]+[x+y for x,y in zip(c,d)]

def factor_state(v,u):
    w=mm(v,u); a=mm(v,tr(v)); b=mm(tr(u),u)
    return w,a,b

def source_step(v,u,g,eta):
    # Ordinary factor-space gradient descent, using old values on both right sides.
    return add(v,scale(mm(g,tr(u)),-eta)), add(u,scale(mm(tr(v),g),-eta))

def compact_step(w,a,b,g,eta):
    wt=tr(w); gt=tr(g)
    wn=add(add(w,scale(add(mm(a,g),mm(g,b)),-eta)),scale(mm(mm(g,wt),g),eta*eta))
    an=add(add(a,scale(add(mm(g,wt),mm(w,gt)),-eta)),scale(mm(mm(g,b),gt),eta*eta))
    bn=add(add(b,scale(add(mm(gt,w),mm(wt,g)),-eta)),scale(mm(mm(gt,a),g),eta*eta))
    return wn,an,bn

def factor_diag(counts,do):
    width=sum(counts); z=zeros(len(counts),width); col=0
    for row,count in enumerate(counts):
        for _ in range(count): z[row][col]=F(1); col+=1
    return z[:do],tr(z[do:])

def witness():
    va,ua=factor_diag([2,2,2,2],2)
    vb,ub=factor_diag([3,3,1,1],2)
    wa,aa,ba=factor_state(va,ua); wb,ab,bb=factor_state(vb,ub)
    assert wa==wb
    for i,j in product(range(2),repeat=2):
        e=zeros(2,2); e[i][j]=F(1)
        assert add(mm(aa,e),mm(e,ba))==add(mm(ab,e),mm(e,bb))
    eta=F(1,4); g1=mat([[1,0],[0,2]]); g2=mat([[0,1],[0,0]])
    for step,g in enumerate((g1,g2)):
        va,ua=source_step(va,ua,g,eta); vb,ub=source_step(vb,ub,g,eta)
        wa,aa,ba=compact_step(wa,aa,ba,g,eta)
        wb,ab,bb=compact_step(wb,ab,bb,g,eta)
        assert (wa,aa,ba)==factor_state(va,ua)
        assert (wb,ab,bb)==factor_state(vb,ub)
        if step==0: assert wa==wb
    assert wa[0][1]==F(-37,32) and wb[0][1]==F(-77,64)
    assert wa!=wb
    emit({'stage':'same_function_same_initial_kernel_witness',
          'first_step_equal':True,'second_step_a':str(wa[0][1]),
          'second_step_b':str(wb[0][1]),'difference':str(wb[0][1]-wa[0][1]),
          'initial_source_width_each':8,'compact_independent_entries':10})

def replay():
    eta=F(1,8)
    v=mat([[1,F(1,2),0,1,0,0],[0,1,1,0,F(1,2),0]])
    u=mat([[1,0],[0,1],[F(1,2),1],[0,F(1,2)],[1,0],[0,1]])
    w,a,b=factor_state(v,u)
    gradients=[mat(x) for x in ([[1,0],[0,-1]],[[0,1],[2,0]],
                 [[F(1,2),-1],[0,F(1,4)]],[[1,1],[-1,0]])]
    for j,g in enumerate(gradients):
        if j==2:
            newv=[F(1,2),F(1,4)]
            v=[row+[x] for row,x in zip(v,newv)]; u=u+[[F(0),F(0)]]
            a=add(a,[[x*y for y in newv] for x in newv])
            assert (w,a,b)==factor_state(v,u)
        v,u=source_step(v,u,g,eta); w,a,b=compact_step(w,a,b,g,eta)
        assert (w,a,b)==factor_state(v,u)
        emit({'stage':'sgd_replay','step':j,'passed':True,
              'source_width':len(u),'output':serial(w)})
    # Rectangular boundary dimensions are not assumed square in the law.
    v=mat([[1,0,1]]); u=mat([[1,2],[0,1],[F(1,2),0]])
    w,a,b=factor_state(v,u)
    for g in (mat([[1,2]]),mat([[-1,F(1,2)]])):
        v,u=source_step(v,u,g,eta); w,a,b=compact_step(w,a,b,g,eta)
        assert (w,a,b)==factor_state(v,u)
    emit({'stage':'rectangular_replay','passed':True,'dimensions':[1,2,3]})

def momentum():
    eta=F(1,8); mu=F(1,2); do=di=2; p=4
    v=mat([[1,0,1],[0,1,F(1,2)]])
    u=mat([[1,0],[0,1],[F(1,2),1]])
    pv=mat([[F(1,4),0,0],[0,0,F(-1,4)]])
    pu=mat([[0,F(1,4)],[0,0],[F(-1,4),0]])
    c=v+tr(u)+pv+tr(pu)
    h=mm(c,tr(c))
    gradients=[mat(x) for x in ([[1,0],[0,-1]],[[0,1],[1,0]],
                              [[1,-1],[F(1,2),0]],[[0,1],[-1,1]])]
    for step,g in enumerate(gradients):
        pvn=add(scale(pv,mu),mm(g,tr(u)))
        pun=add(scale(pu,mu),mm(tr(v),g))
        vn=add(v,scale(pvn,-eta)); un=add(u,scale(pun,-eta))
        k=blocks(zeros(do,do),g,tr(g),zeros(di,di))
        n=blocks(add(ident(p),scale(k,-eta)),scale(ident(p),-eta*mu),k,scale(ident(p),mu))
        h=mm(mm(n,h),tr(n))
        v,u,pv,pu=vn,un,pvn,pun
        c=v+tr(u)+pv+tr(pu)
        assert h==mm(c,tr(c))
        emit({'stage':'momentum_replay','step':step,'passed':True,
              'compact_rows':len(h)})
    # A source implementation that updates U using already-updated V changes the law.
    g=gradients[0]
    correctv,correctu=source_step(v,u,g,eta)
    wrongu=add(u,scale(mm(tr(correctv),g),-eta))
    assert mm(correctv,wrongu)!=mm(correctv,correctu)
    emit({'stage':'wrong_order_negative_control','detected':True})

smoke={'fraction':[3,7],'stage':'serialization_smoke'}
assert json.loads(json.dumps(smoke))==smoke
emit(smoke)
scalar_discovery()
matrix_discovery()
witness()
replay()
momentum()
result={'all_checks_passed':True,'neural_proposer_trained':False,
        'meta_learning_performance_tested':False,
        'scope':'symbolic construction and finite exact learning-state replay'}
(ROOT/'plasticity-result.json').write_text(json.dumps(result,indent=2)+'\n')
emit(result)
