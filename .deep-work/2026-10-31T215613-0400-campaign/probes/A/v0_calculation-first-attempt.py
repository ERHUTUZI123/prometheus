"""Offline algebra checks for Branch A. Not an end-to-end learning experiment."""
from itertools import product
import json
import math
from pathlib import Path
import time

started = time.perf_counter()

def matrix_reaction(code, data, theta):
    # The 4-bit special case, canonical row-major fold; data in two length-2 blocks.
    p = (code[:2], code[2:])
    return tuple(int(sum(p[i][j] * data[2*k+j] for j in range(2)) - theta >= 0)
                 for k in range(2) for i in range(2))

strings = list(product((0,1), repeat=4))
literal_products = {matrix_reaction(a,b,0) for a in strings for b in strings}
assert literal_products == {(1,1,1,1)}
repaired_example = matrix_reaction((1,1,0,1), (1,0,0,1), 1)
assert repaired_example == (1,0,1,1)

sigma = 1.0
lam = 0.001
def g(d): return math.exp(-d*d/(4*sigma*sigma))
def task_loss(d): return 0.5*g(d)**2
def loss(d): return task_loss(d) + 0.5*lam*d*d
def derivative(d): return d*(lam - g(d)**2/(2*sigma*sigma))
def difference_derivative(d, h=1e-5): return (loss(d+h)-loss(d-h))/(2*h)

checks = []
for d in (0.0,0.1,0.8,2.0,3.5):
    analytic = derivative(d)
    finite_difference = difference_derivative(d)
    assert abs(analytic-finite_difference) < 1e-9
    checks.append(dict(d=d,analytic=analytic,finite_difference=finite_difference))

dstar = sigma * math.sqrt(2*math.log(1/(2*lam*sigma*sigma)))
assert abs(derivative(dstar)) < 1e-14
trace = []
def adapt(initial, steps=500, step_size=0.5):
    d = initial
    for t in range(steps):
        d -= step_size*derivative(d)
        if t in (0,4,19,99,499):
            trace.append(dict(initial=initial,step=t+1,d=d,loss=loss(d)))
    return d
zero_final=adapt(0.0)
split_final=adapt(0.05)
assert zero_final == 0
assert abs(split_final-dstar) < 0.001
assert loss(split_final) < loss(0)

# Generalization over values here means only the same two keyed contexts; keys are
# supplied scalar parameters. This does NOT test learned context recognition.
value_cases = []
for a,b in ((1.0,2.0),(-3.0,1.0),(0.0,4.0)):
    value_cases.append(dict(memory_a=a,memory_b=b,query_a=a+g(dstar)*b,
                            query_b=b+g(dstar)*a))

# Von Neumann/Burks copy-consume accounting. A consumes one description.
def reproduce(copy_outputs):
    descriptions=copy_outputs
    automata=1
    descriptions-=1
    automata+=1
    return dict(automata=automata,descriptions=descriptions,
                complete_parent_and_child=(descriptions>=2))
assert reproduce(2)['complete_parent_and_child'] is False
assert reproduce(3)['complete_parent_and_child'] is True

# Minimal JK fast-dynamics reconstruction: C=[[0,2],[1,0]]. The positive
# eigenpair is lambda=sqrt(2), x=(sqrt(2)/(1+sqrt(2)), 1/(1+sqrt(2))).
# Weighted edges here are OUR generalization of the book's binary adjacency.
root2=math.sqrt(2)
x=(root2/(1+root2),1/(1+root2))
cx=(2*x[1],x[0])
flow=sum(cx)
dx=tuple(cx[i]-x[i]*flow for i in (0,1))
assert max(map(abs,dx)) < 1e-15

result = dict(
    classification='algebra/finite checks, not end-to-end acquisition evidence',
    matrix_source_literal=dict(pairs_checked=256,outputs=list(literal_products),
                               theta=0,step_at_zero=1),
    matrix_explicit_repair=dict(theta=1,code=[1,1,0,1],data=[1,0,0,1],
                               product=repaired_example),
    copying=dict(double_copy=reproduce(2),triple_copy=reproduce(3)),
    scope=dict(sigma=sigma,lambda_cost=lam,dstar=dstar,cross_affinity=g(dstar),
               task_loss=task_loss(dstar),total_loss=loss(dstar),
               gram_condition_number=(1+g(dstar))/(1-g(dstar)),
               hessian_at_collision=lam-1/(2*sigma*sigma),
               exact_symmetry_final=zero_final,perturbed_final=split_final,
               updates_per_case=500,trace=trace,derivative_checks=checks,
               value_cases=value_cases),
    jk_generalized_weighted_two_species=dict(x=x,lambda_pf=root2,dx=dx),
    elapsed_seconds=time.perf_counter()-started,
)
destination=Path(__file__).with_name('v0_calculation.json')
destination.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
