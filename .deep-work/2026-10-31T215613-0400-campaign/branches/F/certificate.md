# F first-stage technical certificate

**Judgment:** A classical neural connection can realize an internal operation as a closed controller trajectory, and a composed operation can retain its semantics under some changes to its constituents. The tested mechanism works. **The v0 advantage claim over cached action sequences is closed by exact equivalence.** Whether learning geometry can make useful operations cheaper to discover and retain remains OPEN; this result does not close that broader route.

**Mode and independence:** Independent free-dissent context, inherited current user-authorized Ultra route. The tool surface provides no reliable concrete backend-model or effort telemetry, so no stronger host-mode attestation is made and this is not called Pro. Only the supplied seed, contract and Independent researcher role were read from campaign/workflow material. `food/` filenames were inspected; unfamiliar geometric-control originals were selected lawfully outside the library. No other branch or earlier architecture report was read. No nested agents. v0 was frozen before nearest-prior attack and probe execution; SHA256 `9f309ab74e075cdb59cc17b052e1f936bcb2ab915edcdaf63b6ecc1244101679`.

## Opportunity and complete mechanism

The Fundamental Question asks how neural networks can construct, compose and improve internal computational mechanisms. F questions the assumption that immediate dynamical update directions are already the complete useful instruction set. A learned system can move an internal controller q around a closed path while its data payload h undergoes a nontrivial transformation. Store that path as an executable instruction; compose stored paths to synthesize further transformations. Learning may change both the connection defining transport and the path descriptions. External task feedback is permitted and explicit.

**RECONSTRUCTION / AGENT PROPOSAL:** Full persistent/execution state is `(q,h,theta,L,search_cursor)`. For a supplied piecewise control u:

`dq_i/dt = u_i; dh/dt = sum_i u_i A_i(q;theta) h`.

The matrix fields A_i are outputs of a finite neural network. Parameters theta and library L are fixed during execution. Reversing a path implements an inverse only for the stated driftless, reversible law. A closed polygon is parameterized by displacements whose sum is zero; this enforces controller closure without providing the payload operation. Adapt theta and polygon coordinates using specified symmetric finite-difference loss evaluations and bounded gradient steps. Search retained-loop compositions by length-first enumeration of signed indices, execute every proposal, and retain successful words with complete expansion cost. This is designed conventional optimization/search, not an emergent controller or hidden self-organization oracle.

The executed reduction uses q=(x,y), h=(a,b,c), and

`A_x = y[phi_0(x) M_0 + phi_1(x) M_1]; A_y=0`,

where phi_0 and phi_1 are fixed disjoint ReLU hats of unit integral, centered at x=1 and x=3. Each M is an arbitrary trainable strictly lower-triangular 3x3 matrix. Two fixed closed controller loops traverse the respective hats at height y=1 and return along y=0. Their exact payload maps are `H_A=exp(M_0)` and `H_B=exp(M_1)`. The nilpotent matrix exponential is evaluated exactly as `I+M+M^2/2`. Six initially random scalar weights are trained from basis-vector input/output feedback to produce A:(a,b,c)->(a,b+a,c) and B:(a,b,c)->(a,b,c+b). The subsequent composition search is given only target input/output feedback for C:(a,b,c)->(a,b,c+a), a four-operation length bound and signed concatenation grammar.

**Worked operation:** A,B,A^-1,B^-1 sends `(2,5,7)` through `(2,7,7)`, `(2,7,14)`, `(2,5,14)`, `(2,5,9)`. The intermediary is restored. Every loop restores q. No operation receives free execution merely by being named.

## Checked evidence

**EXECUTED OBSERVATION:** Five deterministic seeds learned the two designed primitive maps in 145–156 epochs. Training used 1,874–2,017 exact-transport loss evaluations per seed, comprising 11,244–12,102 basis-input primitive-loop executions. The fixed reduced-word enumeration first accepted the commutator at candidate 67. A full audit evaluated all 160 reduced words of length <=4; no shorter accepted word exists within that grammar. Max held-out input error across seeds was 4.36e-14 on 100 independent Uniform[-2,2]^3 inputs. This is the same linear target, not a held-out task or new function family.

**Independent computational path:** Direct RK4 integration of the original q,h differential law on the learned four-loop execution agreed with analytic transport to 5.33e-15 across the checked seed/input cases; observed controller closure error was zero. This checks the integrated law in this designed instance. It is not an independent empirical architecture evaluation.

**Negative control:** Restrict both fields to scalar multiples of E21. All loop actions commute and no word changes the third register. The same 160-word search fails with minimum max-basis error 1.0. This is a mechanistic necessity check in the chosen representation, not an equal-capability evolvability control.

**Costs:** First acceptance costs 201 input/output evaluations and 588 primitive-loop basis executions per seed. The retained instruction expands into four primitive loops, 18 path edges and control-path length 32. The checked RK4 realization uses 1,024 integration steps per input. A direct cached-action implementation needs the same four matrix actions. Analytic training uses exact integrated transport; it does not perform RK4 for every candidate. These costs are distinct and explicitly recorded.

## What survives the retention attack

Write

`M_0=alpha E21+beta E31+gamma E32`,
`M_1=delta E21+epsilon E31+zeta E32`.

**RECONSTRUCTION:** Direct finite matrix multiplication gives the composite map

`C=I+(alpha*zeta-delta*gamma) E31`.

Thus beta and epsilon cancel. Composite semantics can survive genuine changes to primitive maps. More generally, preserving the determinant of `[[alpha,delta],[gamma,zeta]]` preserves C. This is a known elementary-matrix identity reconstructed for this candidate, not a novel theorem.

**EXECUTED OBSERVATION:** At isotropic weight-noise sigma=0.01, median retained-C error was 0.00971: generic drift is not protected. Central-entry-only perturbations changed primitive outputs but left C at floating-point training residual. That specific additional hypothesis was recorded in config before execution, after v0 freeze. No criterion from the original probe was replaced.

**OPEN:** An actual learner would have to identify which composite semantics deserve retention and find updates preserving them while learning something useful. All constituent tasks need not survive such an update. The existence of an algebraic invariant is not evidence that a task learner can exploit it efficiently.

## Source-grounded changes in assumptions

**SOURCE:** Murray–Sastry 1993, printed 702 and 705–708: bracket motion, constrained operator growth and sinusoidal steering. This supplied the load-bearing construction. I changed from counting immediate fields to studying their executed compositions, while rejecting unlimited independent growth from arbitrary nested brackets. [Author original](https://people.eecs.berkeley.edu/~sastry/pubs/OldSastryALL/MurrayNonholonomic1993.pdf).

**SOURCE:** Zanardi–Rasetti 1999, arXiv PDF 2–4: loop transport, composition, inversion, curvature and complexity. This defeats broad novelty for the loop-as-operation principle. Quantum adiabatic/unitary assumptions differ from our classical trainable instance; that difference is not itself an advantage. [Original](https://arxiv.org/abs/quant-ph/9904011).

**SOURCE:** Sung 2026, PDF 21–24: learned input-selected matrix exponentials and ordered neural transport. Even neural holonomic operator composition has direct prior art. The source's broad physical/topological interpretations and numerical claims are not independently verified here. A rotation-error counterexample shows why isometry alone does not establish indefinite semantic fidelity. [Original](https://arxiv.org/abs/2601.05240).

**SOURCE:** Grover–Bourgerie 2026v3, PDF 2–5: learned connections, loop products and separate causal interventions. This sharpened the distinction between exposing a connection, reliance on it, and a useful holonomy-specific computation. Our check establishes the designed transport operation, not unique task-level value. [Original](https://arxiv.org/abs/2607.19514v3).

**SOURCE:** Kidger et al. 2020, PDF 2–4, equations (2)–(5), gives the established neural controlled-differential execution formalism. Augmenting z=(q,h) makes our law `dz=F(z)dq` with columns `(e_i,A_i(q)h)`. Their driving path represents observed data; our proposed learned internal loop repertoire is a learning-level difference requiring its own evidence. [Conference original](https://proceedings.neurips.cc/paper/2020/file/4a5876b450b45371f6cfe5047ac8cd45-Paper.pdf).

Exact passages, assumptions and reading-induced changes are recorded in `sources/F/reading-notes.md`. Two OpenReview documents returned verification barriers and were not treated as substantively checked sources. No barrier was bypassed.

## Strongest attack, failed repair and boundary

The instance manually supplies disjoint spatial supports, triangular matrix family, reversible execution, primitive loop paths, task teachers and external enumeration. Only six numerical field weights are learned. An identical cached-word system using H_A,H_B has the same candidate scores and outputs by construction. This closes any v0 performance or primitive-discovery advantage claim. Its payload maps lie in a fixed three-dimensional unipotent group; higher commutators vanish. No universal computation, open-ended evolution, expanded global reachability or new learning principle follows.

A constructive repair replaces smooth transport by signed matrix updates at branch-cut crossings in a punctured controller plane. It gives explicit limited invariance to path deformations avoiding punctures. It also becomes a geometrically routed gate catalog with designed punctures and crossing semantics. Learning the topology or obtaining a learning advantage is still missing. This unexecuted repair and the smooth-loop deformation counterexample are preserved in `repair-and-attack.md`.

**Hardest remaining link:** Does learning a connection together with reusable closed paths improve later operation discovery/retention compared with the identical cached-action learner, after matching representation, external feedback, physical execution and search costs?

**Next single decision:** Pass that concrete geometry/retention gap to a new protected constructive-repair context if the coordinator chooses to pursue it. Preserve the exact cached-word equivalence as a baseline constraint, not as a veto on all geometry-learning mechanisms. This first-stage branch is complete; no additional benchmark or background work is implied.

## Artifacts and reproduction

- `branches/F/v0-frozen.md`: full pre-attack mechanism and declared probe.
- `branches/F/checkpoint-01.md`, `checkpoint-02.md`: changes in epistemic state.
- `branches/F/repair-and-attack.md`: exact invariant, causal attacks, failed repair.
- `sources/F/reading-notes.md` and five original PDFs: substantive source provenance.
- `probes/F/config.json`, `probe.py`: locked finite parameters and executable law/search.
- `probes/F/results.json`, `summary.json`, `cost-accounting.json`, `stdout.json`: raw observations, summaries and costs.

Reproduce from repository root:

`.deep-work/2026-10-31T215613-0400-campaign/probes/infrastructure/venv/bin/python .deep-work/2026-10-31T215613-0400-campaign/probes/F/probe.py`

Python 3.13 and installed NumPy in the isolated campaign environment; exact versions are in results.json. The probe itself has no network, subprocess, arbitrary execution or secret access. Measured inner-probe runtime was approximately 0.67 seconds on this host; source-reading and research time are not inferred from that measurement. Original library, project instructions, canonical NOW and all other branches were left unchanged.
