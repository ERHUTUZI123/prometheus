# after012-a — original reading after constructive v0 freeze

The v0 equations/example were saved before these sources were entered. No experiment or numerical literature result is adopted as our result. File/context isolation is instruction_only; actual server effective mode and model are unknown. Post-freeze project reads authorized by coordinator: root reading notes, candidate R v0, candidate after012-b v0. Those are comparison inputs, not primary evidence. No earlier catalog was read.

## 1. Kidger, Morrill, Foster, Lyons — Neural CDE (NeurIPS 2020)

Original: https://proceedings.neurips.cc/paper/2020/file/4a5876b450b45371f6cfe5047ac8cd45-Paper.pdf . Entered printed pp2–3, §2 Eq.2 and §3 Eq.3–5; also the adjoint/training discussion and §3.1 theorem statement. Their state solves z_t=z_0+∫fθ(z_s)dX_s, with learned initialization and terminal readout. Differentiable X turns this into an ODE driven by Ẋ. Our q-dependent connection embeds by augmenting state with q. **Changed assumption:** executable continuous task-learned response is established machinery, not the candidate contribution. Universal-approximation proof/appendices and empirical results were not independently checked. No universality result is used as a same-resource comparison.

## 2. Lou, Li, Ni — Path Development Network (TMLR 2024)

Original accepted paper: https://arxiv.org/pdf/2204.00740 (v2, 8 September 2024); UCL mirror https://discovery.ucl.ac.uk/id/eprint/10201445/1/2024_Path_Development_Network_.pdf . Entered printed pp5–6 §2.2 Definition2.2, Example2.1, Lemma2.1; pp9–12 §3, Definition3.1 Eq.6, Theorem3.1, Proposition3.1, Algorithms1–2, Remark3.4. They learn z_(n+1)=z_n exp(Mθ(Δx_n)); concatenation multiplies developments. Orthogonal/unitary updates make the relevant differential an isometry; backward transport is explicit. **Changed assumption:** multiplicative learned group state, finite-dimensional path compression and stable gradient transport are very close existing mechanisms. Their constant linear map differs from a spatially varying edge field; that does not produce a new learning rule. Full convergence/separation theory and raw experiments were not checked. OpenReview primary fetch was blocked by browser verification; the original PDF was read through arXiv/UCL instead.

## 3. Bodnar et al. — Neural Sheaf Diffusion (NeurIPS 2022)

Original: https://proceedings.neurips.cc/paper_files/paper/2022/file/75c45fca2aa416ada062b26cc4fb7641-Paper-Conference.pdf . Entered printed pp2–4 definitions, orthogonal-bundle interpretation, transport products and harmonic-section loop condition; pp7–8 Eq.5 and Sheaf Learning passage. Restriction matrices are learned from local node features; their products define transport, and the connection changes the diffusion operator. **Changed assumption:** learned geometric transport and nontrivial loop actions are established even in graph learning. Our execution follows a single input path, rather than diffusing over all nodes, and proposes relations on faces; neither difference is a proven advantage. Their asymptotic classification proofs, normalization appendices and experiments were not independently audited.

## 4. Gao, Brodzki, Mukherjee — Geometry of Synchronization / Learning Group Actions

Original: https://arxiv.org/pdf/1610.09051 (v3, 14 May 2019). Entered printed pp2–5 §1.1 edge/vertex potentials and bundle interpretation; printed p24 Eq.49 and following synchronizability discussion. Edge potentials ρ_ij with inverse reversal are matched by vertex potentials when ρ_ij=g_i g_j⁻¹; the checked discussion identifies global trivial holonomy with synchronizability and formulates approximation by frustration. **Changed assumption:** gauge consistency, flat-bundle interpretation and fitting globally coherent local transformations are prior machinery. The candidate needs useful nontrivial computational holonomy as well as learned nuisance identities, and cannot call synchronization itself an acquired computation principle. The later clustering algorithm and full classification theorem proofs were not checked.

## 5. Laird, Hsu, Bapat, Walters — MatrixNet (NeurIPS 2024)

Original: https://arxiv.org/html/2501.09571 (v1, January2025 posting of NeurIPS2024 work). Entered §§3.1–3.2 and 4.1–4.3, especially matrix block construction, inverse/free-group propositions and relation secondary loss. The model embeds generators as matrices, uses exponentials/products and handles inverse words; additional group relations are encouraged by a loss. **Changed assumption:** relation penalties on acquired matrix operations are direct prior coverage. Correct relations are supplied to that loss; unsupplied relation choice is not resolved by this checked method. A matched comparison must remove any asymmetric disclosure of the true relation list. Experiments and categorical braid-action mathematics were not audited. No reported numerical advantage is adopted.

## 6. Wei, Wang, Rui, Chen — Network Morphism (ICML 2016)

Original: https://proceedings.mlr.press/v48/wei16.pdf . Entered printed pp2–4 §3.1 Eq.1–6, §3.2 algorithms and §3.3 nonlinear extension setup. Linear function-preserving refinement requires G=F₂F₁; convolutional morphism generalizes that equation. **Changed assumption:** retaining H while opening factors BA=H, including nonidentity factors, has clear existing coverage at the single-operation level. Our exact compensation during later constrained updates and overlapping-loop constraint geometry are more specific uses, not an established different acquisition principle. Published training gains and algorithmic general claims were not checked or imported. The paper's own convergence qualifications remain separate from our exact invertible factor identity.

## 7. Littman, Sutton, Singh — Predictive Representations of State (NIPS2001)

Original: https://proceedings.neurips.cc/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf . Entered §1 and §2 (PDF pages5,7,9 in the scanned file), Eq.1–4 and Theorem1 statement/construction setup. Predictive state is a vector of conditional future-test probabilities; linear PSR updates normalize linear predictions. The accompanying belief update is b′∝bT_a O_(a,o). **Changed assumption:** compression of histories by their future consequences is not new. Our endpoint-only task observations do not automatically supply the test queries required for a particular PSR learner. The existence theorem is not a proved efficient acquisition advantage and is not used as one.

## 8. Alonso, Burgos, Paternain — Loops, Holonomy and Signature (2024)

Original: https://arxiv.org/html/2411.12881v2 . Entered §§1–3: loop equivalences, Theorem1.2 statement, definitions/theorems in §2.2, and Lemma3.2 proof. The constructed holonomy equals Chen signature in their stated loop class; retrace/thin equivalence and ordinary homotopy are distinct issues. **Changed assumption:** calling a path product “holonomy” does not automatically make it a different principle from path-development/signature methods. Our selected *filled face* identities intentionally impose a coarser, learned relation. We have not read their infinite-dimensional construction in §4, and do not claim it proves finite learner efficiency or correct task-relevant quotient acquisition.

## 9. Additional directly entered originals; no load-bearing claim adopted

- Huh, *Discovering Abstract Symbolic Relations by Learning Unitary Group Representations*: https://arxiv.org/html/2402.17002 , §§4.1–4.3 and §6.2 mechanism. Matrix-symbol factorization and trace products provide another close operational-algebra learning precedent. Full proof/benchmarks not checked.
- Grover/Bourgerie, *Do Sheaf Neural Networks Use Holonomy?*: https://arxiv.org/pdf/2607.19514 , v3 September2026, pp1–2 plus model/intervention descriptions. Search snippets reflected older claims; the entered v3 explicitly separates learned-connection sensitivity from establishing a beneficial causal holonomy mechanism. No numerical claim from either version is used here.
- OpenReview bZnERPFWpe search suggested a ConnectionNet/curvature model, but both original PDF and forum fetches were blocked by verification. **Not read, not counted as checked coverage.**

## Reading consequence

The broad ingredients are covered. The exact first-mask reduction is our algebraic check, not a literature inference. The hard-relation repair is a concrete change of learning parameterization, but its currently stated acquisition procedure remains equation/model selection under constraints. “No new principle obtained” is supported by the absence of a residual mechanism in this construction, **not** by an exhaustive novelty search or a claim that topology research is exhausted. No training or large implementation should be started merely to demonstrate these already expressible rotations.
