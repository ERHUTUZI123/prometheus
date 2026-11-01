# Stable wrong computation from individually exact composable relations

AGENT PROPOSAL / RECONSTRUCTION by root after reading saved dissipative v0; not a published novelty claim. This tests a necessary execution step, not learned factor formation.

## Same information and task
Three signs a,b,c in {-1,+1}. Four known relations ab=1, bc=1, ca=1, abc=1 are given equally to both executors. Pair constraints imply all signs equal; the triple relation selects (+1,+1,+1) uniquely. A conventional GF2 solver maps sign=(-1)^bit and solves the same four equations; there is no answer oracle or hidden intermediate labels.

Use candidate's exact sum-of-squared residuals, no learned error:

 E(a,b,c)=1/2[(ab-1)^2+(bc-1)^2+(ca-1)^2]+beta/2(abc-1)^2, beta=1/10.

The triple factor has lower positive confidence but exactly the same zero set. Continuous execution is synchronous x <- clip_[−1,1](x-h grad E), h=1/10; no quantization for the initial necessary-condition test. All candidate factors are already perfect. No training is required or claimed. A stable incorrect local minimum would defeat a claim that correct relation factors alone suffice for reliable composition. It would not defeat a restricted task/initialization/noise-dependent claim, nor all implicit solvers.

## Analytic construction before execution
At a=b=c=t, each gradient component is

 g(t)=2t(t²−1)+beta t²(t³−1).

Besides t=1 and t=0, a negative root near −1 satisfies q(t)=2(t²−1)+beta*t*(t³−1)=0. With beta=.1, q(−1)=.2 and q(−.9)=−.22439, so the intermediate-value theorem gives a stationary point t_* in (−1,−.9). Here q'(t)=4t+beta(4t³−1)<0 throughout the interval, hence the root is unique there.

The Hessian at a symmetric point has diagonal d=2t²+beta t⁴ and off-diagonal o=2t²−1+beta(2t⁴−t). Eigenvalues are d+2o and d−o (multiplicity2). For t in (−1,−.9),

 d−o=1+beta(t−t⁴)> .8;
 d+2o=6t²−2+beta(5t⁴−2t)> 2.86.

Thus the negative stationary point is a strict stable wrong minimum, not merely zero-gradient initialization at an unstable saddle. Eigenvalues are also below 4.7 on that interval, so h=.1 makes the local discrete fixed point attracting. Its sign output violates abc=1 although the valid assignment has E=0. This is a fully specified counterexample before the numerical test.

The finite numerical test should compute the negative root by bracketed bisection, inspect analytic gradient/Hessian, run fixed initializations (near negative point, asymmetric perturbation, positive point, and zero), count actual executed steps and energy evaluations, and compare the same clause system with GF2 elimination. It must not hunt seeds, tune beta/h or infer global worst-case runtime from a finite run. The missing acquisition issue remains separate.
