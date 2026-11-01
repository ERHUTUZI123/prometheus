# Post-freeze comparison with BF

**Judgment:** The independently rebuilt mechanism agrees with BF's narrow positive-SPD-mobility and parent-relaxation claims. It adds an important rival explanation: a no-training Hessian slow-mode metric matches every sampled settling time of the learned metric in the rebuilt parent and altered-energy tests. The data support learned preconditioning, with task-dependent finite benefits. They do not establish construction, composition, or discovery of an internal computational mechanism.

## Access boundary

Before author access, specification, code, full results, samples and certificate were hashed in blind-stage-sha256.txt. That manifest records certificate SHA256 1f207c13489b39bd4b4c8cb4dcdeca3c7b427c4b3d5ddb8d11b983ab6a760c8a, code9fa9a71a58410bd052fccffd05deae06cd9991ed286ee47f54c08f14aedc7055, and results4d03a2deea6ab11ee5cb7d61b70afcf24e2dbfe5babbaa8c8ae203d2bab853c7. The frozen files remain unchanged. Only after root authorization did this evaluator open branches/BF/candidate-v1.md, probes/BF/probe.py, and probes/BF/results.json. This comparison adds no simulation and pools no measurements.

## What independently reproduced

Both implementations use the same anchored four-edge parent energy, up to an additive constant. BF uses forces f=-grad E; the rebuild uses gradients. Their updates normalize(v+.2u(u^T v)), so replacing u by -u has exactly no effect. Both use the same SPD metric .5I+2vv^T and trace4. Independent incidence-matrix derivation and checks establish the16 parent critical points, one local minimum, exact energy descent and preserved equilibrium/stability classification. Finite runs agree qualitatively: all64 parent starts settle and all64 settle earlier with the learned metric. The learned vector aligns almost exactly with the slowest minimum-Hessian eigenvector (BF cosine0.999999999772; rebuild0.999999999968).

Both training integrations are RK4 h=.025 to60,16 uniform torus starts, samples25:5:60,128 updates in start-major then chronological order, with153728 counted state-force evaluations. The rebuild uses separate frozen seeds for starts, initial v and evaluation, whereas BF uses one seeded generator sequentially. This is replication on independent pseudorandom inputs, not reproduction of BF's exact samples. Same-model agreement does not substitute for the mathematical derivations or numerical checks.

## What differs and cannot be pooled

| Choice | BF | Independent rebuild |
|---|---|---|
| Main solver | Fixed RK4 h=.025 | DOP853 rtol1e-10/atol1e-12 |
| Main horizon |180 |160 |
| Success observable | Max wrapped adjacent parent residual<1e-6 | Max wrapped absolute phase<=1e-3 AND energy<=1e-6 |
| Settling | First threshold crossing; stop that path | First .25-grid point satisfying thresholds for every remaining sampled point through160 |
| Reported force cost | Actual calls until first crossing | Actual calls through fixed horizon160, including after settling |
| Numerical check | Half-step RK4 on5 preselected starts | Euler h=.025 on all512 cases |
| Added controls | Unsafe five-factor shortcut at perturbed winding | Analytic slow-mode and random same-spectrum metrics; unequal-weight altered energy |

BF's parent median times111.8375 and46.0375 differ from the rebuild's65 and27.125 because inputs, thresholds, observables and solvers differ. Their broadly similar time ratios are qualitative corroboration, not an agreement test on the numeric values. The rebuild's sampled persistent definition is not continuous-time settling certification. Euler terminal successes agree, but intermediate trajectory disagreement reaches3.062rad; the certificate explicitly preserves that limit.

BF stops at success and records1122032 baseline versus467332 learned evaluation calls, before153728 training calls. Its operation-weighted conclusion changes with sine cost. The rebuild deliberately continues to160 and records140018 versus88997 calls, before the same training cost; thus242725 training-inclusive calls exceed140018 baseline calls. Those cost conclusions are compatible because they measure different execution protocols. The rebuild's rough193-query amortization arithmetic must not replace BF's cost-to-success accounting. Neither equal trace nor raw call counts establish matched arithmetic, communication, wall time or energy. BF's explicit rank-one operation/communication accounting is more informative than the rebuild's dense-matrix call count on those dimensions.

## What the controls and altered energy add

The rebuild's analytic control chooses the parent Hessian's slowest eigenvector, without training. Its parent and changed-energy sampled settling times match the learned metric for every one of64 starts at.25 resolution. This does not refute BF's claim that useful orientation can be learned without an eigensolver; it shows no demonstrated behavioral advantage over ordinary analytic preconditioning when the small fixed chain is known. The random same-spectrum control is slower than baseline in48/64 parent and62/64 altered-energy starts. Orientation matters; SPD and matching trace do not by themselves confer acceleration.

The rebuilt altered energy is sum_i(1-cos(theta_i-theta_(i-1)))+.8(1-cos theta4), with the same exact compatible zero target. It is transfer across an altered solver/factor graph, not discovery of a new function or a fully new task. The frozen parent metric is faster in51 starts, slower in12 and tied in1. Countercases include index0,9->15.5, and17,24.5->31.5. This refutes universal acceleration under this single factor perturbation, while retaining a positive finite average result. It does not test later learning or forgetting.

This altered energy differs from BF's unsafe shortcut, where all five factors have weight.8: .8[sum_i(1-cos diff_i)+(1-cos theta4)]. The rebuild did not independently execute BF's perturbed winding intervention, so its altered-energy successes cannot be presented as a direct replication or refutation of that intervention. After unblinding, the displayed BF winding equilibrium can be checked analytically: equal edge phases alpha=2pi/5 balance gradients, and its Hessian is .8cos(alpha)(B^T B+e4e4^T), positive definite because cos(alpha)>0. That is a sound post-freeze mathematical check, distinct from blind numerical evidence.

## Decision boundary

The concrete result is a safe execution-geometry change relative to adding a potential factor, with learned slow-mode identification and finite relaxation speedups. Exact semantic critical-point preservation does not imply useful later capability acquisition. The analytic equivalence, resource dependence and12 transfer slowdowns should appear in any synthesis. One useful next human decision is whether further work on learned execution geometry closes a meaningful gap in the Fundamental Question; this comparison authorizes no further experiments.
