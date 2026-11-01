# Independent reconstruction: frozen specification

Provenance: AGENT PROPOSAL from a neutral mechanism, without author code or expected curves. Context separation is instruction-only. No claim of distinct model or ACL isolation.

QUESTION: Does a frozen rank-one metric learned from late gradients improve finite-horizon convergence of the anchored four-phase parent energy, and what survives one explicitly changed energy?

GIVEN: theta0=0; x=B theta with B the invertible adjacent-difference matrix; parent energy sum(1-cos x); flow theta'=-M grad E. Learned M=.5I+2vv^T, unit v, updates normalize(v+.2u(u^T v)).

NON-GOALS: Neural emergence, novelty, open-ended evolution, learned function construction, universal improvement, or asymptotic global convergence inferred from finite starts. This reconstruction supplies no literature novelty claim.

FROZEN INPUTS: NumPy PCG64 seeds 2026103101 (training starts), 2026103102 (held-out starts), 2026103103 (initial unit v), 2026103104 (random fixed metric control). Train 16 iid uniform starts on [-pi,pi]^4 with RK4 h=.025 to60, sampling times25,30,...60. Process observations in start-major then time-major order. Normalize each nonzero gradient; freeze v before sampling evaluation outcomes. Evaluate64 fresh iid uniform starts.

COMPARISONS: I; learned metric; fixed analytic metric aligned to the smallest eigenvector of B^T B (no training); fixed random metric with the same spectrum. The fixed analytic alternative is a rival explanation: hand-designed preconditioning could explain any speed change. The random condition separates orientation from the shared spectrum.

VERIFIER: Main evaluation SciPy DOP853 (rtol1e-10, atol1e-12), horizon160, grid spacing.25. Success is max absolute wrapped theta<=1e-3 AND E<=1e-6. Sampled settling time is the first grid point after which both thresholds stay satisfied through160; right-censor other cases at160, preserve all per-start records. Report continuous-time settling separately from gradient-call counts and training amortization. Check analytic gradient by centered finite differences, all16 parent critical points and Hessian inertias, SPD spectrum, observed sampled energy increase. Cross-check main trajectories using fixed-step Euler h=.025 under the same starts/criteria. Optional transfer: add .8(1-cos theta4), keeping all metrics frozen, the same64 starts, tolerances, horizon, and comparisons. Parent and transfer thresholds are task-specific energies but identical numerical values.

BUDGET: One main script, one <=120s isolated execution, at most one corrective rerun for code/verifier error; no network, no new packages, no root/reference writes. No hyperparameter tuning, start resampling, or outcome-based metric selection. The existing runner enforces network denial, credential-minimal environment, run-only writes and timeout. It does not enforce disjoint branch write ACLs.

STOP: Complete finite verification and freeze code/specification/results/report hashes before any author comparison. Failure/censoring is retained. Any failed numerical verifier is a limitation, not permission to broaden the experiment.

OUTPUT: Compact certificate with exact metrics, derivation, preserved cases and limits. Compare author only after freeze and root authorization.
