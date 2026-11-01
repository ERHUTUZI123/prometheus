# Frozen independent comparison contract

QUESTION: Does the B shortcut gain survive optimizing the parent coupling allocation under the same declared coefficient budget? Can an independent implementation reproduce the advertised winding failure?
INPUT: Frozen B v0 equations and its qualitative countermodel; no author probe code is used. This is a coordinator check, not the later blind rebuild.
BASELINES: parent strengths (1,1); optimized positive parents under 4(K_A+K_B)=8; proposed shortcut strengths (.5,.5,.5). No wall-clock/physical energy claim. The stronger parent's optimal allocation is chosen analytically/numerically before dynamic measurements.
MEASUREMENTS: minimum local Hessian eigenvalue; 32 identical initial phase pairs from seed 8801, static zero clamps, RK45 max step .05, rtol1e-8, atol1e-10, horizon80; first max wrapped coordinate error below .01 and final error below1e-4. Count all failures. Uniform start distribution spans [-pi,pi]^2. No parameter tuning based on these results.
COUNTERMODEL: 4 free phases in an anchored chain plus endpoint shortcut; initialize at the 5-cycle twist plus a fixed small perturbation; horizon150; finite-difference Jacobian eigenvalues and final wrapped output error.
COST/SCOPE: one small optimization plus fewer than100 ODE solves, CPU only, 120 second outer timeout. This tests the specified substrate/improvement claim, not general learning, novelty, hardware energy or discovery of useful coordinates. Evaluation is exploratory: contexts separate, shared filesystem is not enforced isolation.
DECISION: if gain disappears against optimal parents, withdraw resource-normalized improvement; if winding persists, forbid a global correctness claim for entailed shortcut addition.
