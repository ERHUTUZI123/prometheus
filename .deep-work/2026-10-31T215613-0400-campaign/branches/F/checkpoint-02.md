# F protected checkpoint 2 — after frozen v0 and first execution

v0 SHA256: 9f309ab74e075cdb59cc17b052e1f936bcb2ab915edcdaf63b6ecc1244101679.
No cross-branch exposure. Additional nearest-prior research began only after freeze.

EXECUTED OBSERVATION: The bounded probe completed. All 5 seeds learned the two designed transport maps in 145–156 epochs, then the length-first search found A,B,A^-1,B^-1 at candidate index 67. All 160 reduced words of length <=4 were evaluated for the minimality audit. Max held-out error across seeds was 4.36e-14. The physical realization expands to 18 path edges and 4 primitive loops. A separate RK4 implementation of q,h dynamics agreed with analytic transport to 5.33e-15 and had zero observed q-closure error. The commuting negative control could not realize C (best basis error 1.0).

RECONSTRUCTION / EXECUTED OBSERVATION: General weight perturbations alter retained C. At sigma=0.01 median C basis error was 0.00971. However, perturbations confined to the central E31 entries change the primitive maps while canceling from the commutator; C stayed at floating-point training residual. This confirms a narrow algebraic robustness mechanism, not generic stability under learning. The central-noise hypothesis was added to config before running; it was not in the original frozen v0 stability proposal.

SOURCE: Substantive postfreeze reading of Zanardi & Rasetti 1999, arXiv PDF pp. 2–4, establishes a close predecessor: parallel-transport computation, loop composition, inverse paths, connection curvature, and generic-gate accessibility were explicitly discussed there. Their physical setting is quantum/adiabatic/unitary; our v0 is classical, nonunitary and trainable. These differences do not make the loop-to-operation mechanism new. Their complexity discussion also blocks treating generic accessibility as an efficient compilation result.

CURRENT JUDGMENT: Mechanically executable classical learned holonomy is viable in a heavily designed instance. No evidence of benefit relative to cached action sequences; the identical matrix-action baseline is exactly equivalent in this probe. The novel-learning claim remains unsupported. Continue only through closest neural prior-art inspection and explicit analysis of the one useful robustness distinction; then end with certificate rather than escalating to larger benchmarks.
