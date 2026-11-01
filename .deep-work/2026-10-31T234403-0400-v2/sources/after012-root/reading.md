# Root original reading after review012

Actual entry during this continuation; no experiment or novelty claim. Primary online originals, not library filenames. Source claims below are limited to entered passages.

## Neural CDE (Kidger et al., NeurIPS 2020)
SOURCE: https://papers.nips.cc/paper/2020/file/4a5876b450b45371f6cfe5047ac8cd45-Paper.pdf . Actually read printed pp2–4 §§2–3.4, especially Eq2–5. Their hidden state solves dz=f_theta(z)dX; with differentiable input, this is the ODE zdot=f_theta(z)Xdot. Neural vector fields, learned initialization and readout receive ordinary loss/gradient training. They already learn executable continuous responses to incoming controls. Appendices B/C formal universal-approximation/comparison proofs NOT checked; those generic claims are NOT our dismissal argument. A proposed learned connection is required to identify a specific additional acquisition/organization rule.

## Neural RDE (Morrill et al., 2021 v4)
SOURCE: https://arxiv.org/html/2009.08295v4 . Actually entered §§1–3.3, Eq1–10, definition of iterated integrals and the learned log-signature update. Method: compute fixed-depth local log-signatures; evolve a learned hidden vector field against them, compressing sequence length at a channel cost. Their implementation learns the lifted vector field directly rather than imposing the exact bracket structure. Adjoints/training, preprocessing and channel-depth costs belong in a comparison. Convergence Appendix B and raw experiments NOT checked. No published speed/energy result adopted here.

## Loops, Holonomy and Signature (Alonso, Burgos, Paternain 2024 v2)
SOURCE: https://arxiv.org/html/2411.12881v2 . Read §§1–3, including Theorems1.2/2.2/2.3 statements, path equivalences, and Lemma3.2 proof. Authors construct a connection whose holonomy is Chen signature, distinguishing retrace/thin relations from ordinary homotopy. The infinite-dimensional Banach-group construction/proof in §4 NOT yet read. This is a warning that holonomy and signature proposals may be two representations of one mechanism, not independent computational principles. Ordinary homotopy must not be silently substituted for thin equivalence.

## Dynamical Learning in Deep Asymmetric Recurrent Neural Networks (Badalotti et al.)
SOURCE: https://arxiv.org/html/2509.05041v2 ; journal https://doi.org/10.1103/lbr6-91sn . Actually read main-text model/learning passage, Eq1–6 and surrounding assumptions. Binary recurrent dynamics use self-coupling; training transiently injects true labels through W_back, removes them, then stabilizes the reached fixed point. Local rule is J_ij <- J_ij+eta s_i*s_j* 1[s_i*h_i<=kappa]1[J_ij!=0]. Thus local plasticity does not eliminate label feedback or grow edges. Large-N accessibility/statistical assumptions are separate from the rule. SupplementC and benchmark data NOT checked. Potential inspiration is shaping accessible stable dynamics, not evidence for a new operation-acquisition principle.

## Teaching Recurrent Neural Networks to Modify Chaotic Memories by Example (Kim et al., preprint2020 / published2021)
SOURCE: https://arxiv.org/html/2005.01186 ; published https://www.nature.com/articles/s42256-021-00321-2 . Actually read §§II–V and VI through Eq8. Fixed reservoir rdot/gamma=-r+g(Ar+Bx+d); learn readout W by least squares, feed output back to give recurrent matrix A+BW. Supplied transformed trajectory examples plus control values teach transformations of attractor dynamics; quadratic Taylor model explains nearby response. This already turns learned outputs into executable recurrent transformations. It does not autonomously discover operation classes from an unlabeled path stream. Remaining VI, supplements, raw data NOT checked.

RECONSTRUCTION / next constructive pressure: continuous executable transformations, learned local weight rules and path composition are established ingredients. A successor needs an explicit way to acquire/revise the path equivalence or state organization with a consequential learning effect. Neither geometry terminology nor a neural vector field alone supplies it. This is a constructive constraint, not a universal-simulability veto or proof of topology exhaustion.

## Root actually enters the dissipative nearest originals

SOURCE: COMET https://arxiv.org/html/2111.03042 , §§3.1–3.2, Eq1–2 and Algorithm1; root read these body passages directly. Component-conditioned energies are added; finite state-gradient steps reconstruct observations and parameter gradients train the encoder/energy. This supports a specific mechanism overlap, not a result on general algorithm acquisition. Experiments not audited.

SOURCE: IREM https://proceedings.mlr.press/v162/du22d/du22d.pdf , printed pp3–4 §§3.1–3.2, Eq2–6/Algorithm1; root read directly. It trains finite energy-descent outputs against supplied solutions and discusses inconsistency when test-time descent runs longer. A compact verifier and an efficiently executed solution are distinct claims. The complexity discussion entered afterward is not adopted as a learner superiority result; no source experiments were replicated.
