# Frozen independent reconstruction certificate

**Judgment — VERIFIED finite numerical result:** The learned metric substantially accelerates this parent relaxation, but a no-training analytic preconditioner produces identical sampled settling outcomes. On the altered energy the inherited metric helps most starts and hurts 12/64. Including training, it does not save total gradient calls over 64 full-horizon parent evaluations. This is finite preconditioning evidence, not construction of new computations or global innovation.

## Independence and execution

This evaluator received only the neutral mechanism and training recipe. Before completion it read root AGENTS.md, the evaluator section of deep-work/ROLES.md, and the allowed isolated runner; it did not read author branches, other probes, reports, state/events, or expected curves. Input choices and criteria were written in specification.md before execution. Isolation of agent context is by instruction, not ACL. Same-model reconstruction is not independent mathematical proof.

One inspected script, probes/rebuild/reconstruct.py, executed successfully through probes/infrastructure/run_isolated.py with --timeout120 and run-only outputs. Actual script wall time was 7.75s. No mechanism-changing rerun or parameter search occurred. A later read-only formatting command extracted checks and failing cases from the frozen data; it ran no trajectories. Only branches/rebuild/ and probes/rebuild/ were intentionally written by this agent; the infrastructure runner also owns its pre-existing tmp area. No external services, network experiments, package installation, or research-document edits occurred.

## RECONSTRUCTION: exact mathematical claims

Let B be the lower bidiagonal 4x4 matrix with diagonal1 and subdiagonal-1. The map x=Btheta is a torus automorphism because B has integer entries and determinant1. The parent gradient is B^T sin(Btheta). Since B^T is invertible, its zeros are exactly x_i in {0,pi}: 16 points modulo2pi. The Hessian is B^T diag(cos x) B; by congruence its negative index is exactly the number of pi-edges. Thus only x=0 is a local minimum; the other critical points are saddles or the maximum. This does not say every initial condition converges to zero: initializing exactly at another critical point is an immediate counterexample.

For any unit v, M=.5I+2vv^T has eigenvalue2.5 along v and.5 on its orthogonal complement. It is SPD with trace4, matching trace(I). Along the exact continuous flow dE/dt=-g^T M g<=-.5||g||^2, with equality exactly at a critical point. Invertibility preserves critical points. At a critical point MH is similar to M^(1/2) H M^(1/2), so Hessian inertia and the local stable/unstable dimension are unchanged. SPD alone does not imply faster convergence.

The parent minimum Hessian eigenvalues are 0.1206147584,1,2.3472963553,3.5320888862. The analytic control takes v along the first eigenvector. Since this metric commutes with H, local linear decay rates become 0.3015368961,.5,1.1736481777,1.7660444431: the slowest linear rate is2.5 times baseline. This predicts a local rate change, not a global settling-time guarantee.

Training repeatedly multiplies v by I+.2uu^T and renormalizes. Late normalized gradients near a nondegenerate minimum can be dominated by the slow mode; their signs do not affect uu^T. The measured absolute learned/analytic cosine is0.999999999967925. The reconstruction therefore supports slow-mode identification as the simple explanation. The global origin of the energy, anchor, observations, update and rank-one family are manually supplied; the learned object is four real orientation parameters, then frozen during testing.

## VERIFIED numerical comparison

Training:16 iid uniform torus starts, RK4 h=.025 to60,8 observations per start at25:5:60. Independent PCG64 seeds, initial v, all observations and learned metric are preserved in frozen_learning.json. Evaluation:64 fresh iid starts; common starts for every metric and both energies; no evaluation feedback into v. DOP853 uses rtol1e-10/atol1e-12, T160. Settling requires max absolute wrapped phase<=1e-3 AND task energy<=1e-6 at every remaining .25-spaced sample through160. All cases succeeded; no censoring was removed.

| Energy | Metric | Median settling | Mean settling | Faster/slower than baseline | Full-horizon gradient calls |
|---|---|---:|---:|---:|---:|
| Parent | I |65.000|61.488|—|140018|
| Parent | Learned |27.125|26.160|64/0|88997|
| Parent | Analytic slow |27.125|26.160|64/0|88595|
| Parent | Random same-spectrum |67.375|65.520|16/48|168587|
| Parent+.8(1-cos theta4) | I |21.750|22.121|—|134684|
| Changed | Learned |17.375|17.762|51/12 (1tie)|98831|
| Changed | Analytic slow |17.375|17.762|51/12 (1tie)|98471|
| Changed | Random same-spectrum |31.750|32.676|2/62|162341|

Median paired baseline/learned settling ratios are2.418 (parent) and1.254 (changed). These differ from ratios of medians. A single random orientation control shows the metric spectrum alone does not guarantee improvement; it does not estimate a distribution over random metrics.

Preserved countercases to universal transfer improvement (zero-based held-out index, baseline->learned time):0,9->15.5;6,21->21.5;16,16->16.75;17,24.5->31.5;23,13.75->14.75;24,25.25->29.5;26,16->17.75;34,16->19;42,11->15;45,12.5->12.75;46,16.25->17;52,15.5->17.75. Their inputs and full sampled trajectories are retained. “Retention” here means performance under one changed energy with a frozen metric; it does not test forgetting after later learning, composition, or discovery of new functions.

## Numerical checks and resource boundary

Centered finite differences at10 independent points give maximum gradient error6.71e-10 for both energies. Enumeration checks all16 parent critical points, residual<1e-12 and correct Hessian inertia. Every tested metric has expected SPD spectrum. The largest sampled DOP853 energy increase is1.11e-16, consistent with roundoff, while the exact energy-descent claim comes from the derivation above.

A scientifically distinct explicit-Euler execution at h=.025 on all512 cases gives the same64/64 terminal success in each condition. Sampled settling discrepancies are <=1 time unit on parent learned/analytic and <=.25 elsewhere. Maximum terminal torus disagreement over all cases is2.28e-9. However, maximum intermediate coordinate disagreement reaches3.062 radians, so these integrations do not uniformly reproduce entire paths near sensitive basin passages. Matching endpoint outcomes is weaker than certified trajectory error bounds. No interval arithmetic or formal proof assistant was used; the changed energy's full critical-point structure was not classified.

Training requires153728 per-state gradient evaluations and128 normalized vector updates. Parent baseline full-horizon evaluation costs140018 calls; trained evaluation costs88997, so training+evaluation costs242725 calls, greater than baseline. Reusing the observed per-start savings gives an arithmetic break-even estimate of about193 parent starts, not a validated generalization result. The no-training analytic control needs88595 calls for these64 tests, excluding its small eigendecomposition cost. The call counts measure neither equal wall time nor equal matrix-operation cost; rank-one multiplication differs from baseline. Full-horizon evaluation counts also continue after settling and cannot be read as measured cost-to-first-success. DOP853 uses980524 calls over all conditions; the numerical Euler check uses3276800, separate verification cost. At fixed Euler h, full-horizon call budgets are exactly matched between conditions. Dense M was used even for I, so no optimized computational-efficiency claim is warranted.

## Certificate boundary

CLAIM: Frozen learned orientation accelerates this parent and usually transfers to one changed energy. STATUS: VERIFIED for these seeds, thresholds, integrators and finite comparisons. ARTIFACTS: specification.md; reconstruct.py; frozen_learning.json; results.json (all per-start cases, checks, starts and costs); dop853_samples.npz; execution.log. WEAKEST LINK: one training realization,64 evaluation starts, numerical rather than certified trajectories, one changed task and no novel computation. FAILED BRANCH/ATTACK: universal improvement under task change is contradicted by12 recorded starts; training-inclusive savings at64 queries is contradicted by accounting; a special learned benefit beyond ordinary analytic slow-mode preconditioning is unsupported. NEXT SINGLE DECISION: whether that narrow relaxation/preconditioning question has enough relevance to the Fundamental Question to warrant further work. No follow-on experiment is authorized by this certificate.
