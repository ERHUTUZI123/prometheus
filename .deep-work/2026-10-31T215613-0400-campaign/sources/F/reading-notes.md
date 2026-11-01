# F primary-source reading events

The independent mechanism was frozen before the postfreeze readings below. Exact passages are identified by page, section and equation; summaries are paraphrases. Only the downloaded lawful originals were used for technical attribution. Search snippets motivated retrieval, not verification. No source's reported experiments were independently reproduced.

## Initial constructive reading: Murray and Sastry 1993

Richard M. Murray and S. Shankar Sastry, *Nonholonomic Motion Planning: Steering Using Sinusoids*, IEEE TAC 38(5), 700–716. [Author-hosted original](https://people.eecs.berkeley.edu/~sastry/pubs/OldSastryALL/MurrayNonholonomic1993.pdf). Local `murray-sastry-1993.pdf`.

Read PDF pp. 3, 6–9, printed pp. 702, 705–708. Load-bearing passage: printed 702, Section II-A, equation (3), interprets a signed four-flow excursion as second-order bracket displacement. Its assumptions include smooth driftless control-affine fields and availability of signed controls. Section II-B distinguishes the immediate distribution from its bracket closure. Printed 705 Hall-basis conditions impose antisymmetry/Jacobi constraints, preventing arbitrary independent claims for every nested expression. Printed 706 Algorithm 1 restores immediate coordinates while changing higher coordinates; 707–708 treats higher-order frequency selection and states limits of the general approach.

What changed: stopped equating the number of directly actuated fields with the number of available effective operations. Also stopped treating bracket accessibility as a claim of efficient discovery or unlimited operator growth. The implemented two-hat payload connection is our reconstruction, not their architecture.

## Postfreeze nearest prior: Zanardi and Rasetti 1999

Paolo Zanardi and Mario Rasetti, *Holonomic Quantum Computation*. [Original arXiv record](https://arxiv.org/abs/quant-ph/9904011). Local `zanardi-rasetti-1999.pdf`, retrieved arXiv version, 5 PDF pages.

Read PDF pp. 2–4. Page 2 equations (1)–(2) specify the ordered transport exponential and connection; nearby properties state composition, identity, path reversal and reparameterization. Page 3 equation (5) relates curvature to the operator algebra; the two-loop proposition is existential rather than a compilation recipe. Page 4 explicitly discusses unfavorable loop-count scaling and local structural restrictions needed for efficiency. Their assumptions include degenerate eigenspaces, adiabatic control and unitary quantum transport.

What changed: the loop-as-instruction idea, composition and inverse-path principle are close established precedent, not this branch's novelty. A classical learned nonunitary connection is a different physical/learning setting, but those differences need a substantive benefit. Their complexity boundary reinforces full execution-cost accounting.

## Postfreeze neural overlap: Sung 2026

Ilmo Sung, *Robust Reasoning as a Symmetry-Protected Topological Phase*, arXiv:2601.05240v1. [Original](https://arxiv.org/abs/2601.05240). Local `sung-2026.pdf`, 29 PDF pages.

Read specifically PDF pp. 21–24: equation (6) gives input-selected multiplicative state transport; equations (9)–(10) compare recurrent updates; printed 24 parameterizes each trainable generator by a skew matrix and matrix exponential. The source directly proposes learned holonomic operator composition in a neural setting. The reported experiments and physical/topological interpretations are SOURCE claims, not VERIFIED results here.

What changed: even the neural learned-operator language is occupied prior ground. I do not import an inference from norm preservation to protection from accumulated operator error; the explicit rotation counterexample in `repair-and-attack.md` shows why that implication needs additional assumptions. My smooth controller-space loops and joint geometry-learning proposal differ operationally, but an advantage is still unestablished.

## Postfreeze causal caution: Grover and Bourgerie 2026

Ankit Grover and Rémi Bourgerie, *Do Sheaf Neural Networks Use Holonomy? A Measure--Intervene--Control Study*, arXiv:2607.19514v3. [Original](https://arxiv.org/abs/2607.19514v3). Local `grover-bourgerie-2026.pdf`.

Read PDF pp. 2–5. Section 2 defines learned edge transports from restriction maps. Page 4 equation (2) gives loop products; propositions distinguish finite-depth return effects from the global-section constraint. Page 5 Table 1 distinguishes synchronized-flat, shuffled and identity checkpoint interventions. The main system is tied SO(2) sheaf diffusion, rather than our nonabelian sequential transport.

What changed: a measurable connection, sensitivity to destroying that connection, and evidence of a useful holonomy-specific computation must be kept separate. Our successful transport probe supplies only the first mechanistic level. It does not identify a unique learning advantage or prove that geometry improves the task.

## Access limitation

The OpenReview PDFs `rsbDkbX8h0` and `pH1KuDsEIL` were identified by search but returned a browser-verification page when opened. No access barrier was bypassed. Their search snippets were not used as checked technical evidence; the original Sung arXiv paper provided the relevant neural predecessor instead.

## Final nearest-form check: Kidger et al. 2020

Patrick Kidger, James Morrill, James Foster and Terry Lyons, *Neural Controlled Differential Equations for Irregular Time Series*, NeurIPS 2020. [Conference original](https://proceedings.neurips.cc/paper/2020/file/4a5876b450b45371f6cfe5047ac8cd45-Paper.pdf). Local `kidger-et-al-2020.pdf`.

Read PDF pp. 2–4: equation (2) defines controlled differential equations; equation (3) supplies a learned matrix-valued vector field; equations (4)–(5) reduce differentiable driving paths to an ODE. Conditions include a continuous bounded-variation control path and Lipschitz fields for global existence/uniqueness. Their controls encode observed data, rather than self-generated retained internal instructions.

What changed: the generic neural differential execution law is also established prior ground, independently of holonomic gate literature. By augmenting z=(q,h), our law has columns F_i(z)=(e_i,A_i(q)h), giving dz=F(z)dq. The particular bilinear field and internal learned closed-path repertoire are structural restrictions/extensions of a known execution formalism, not a new differential primitive. The only open claim is the learning/retention advantage of jointly shaping geometry and executable loops.
