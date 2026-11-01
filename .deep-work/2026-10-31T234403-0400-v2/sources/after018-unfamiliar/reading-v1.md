# after018 v1 original reading — nonholonomic control and neural bracket mechanisms

## Entry source actually read

Richard M. Murray and S. Shankar Sastry, **Nonholonomic Motion Planning: Steering Using Sinusoids**, *IEEE Transactions on Automatic Control* 38(5), 700–716 (1993). [Author-hosted original PDF](https://people.eecs.berkeley.edu/~sastry/pubs/OldSastryALL/MurrayNonholonomic1993.pdf); [Murray publication page](https://www.cds.caltech.edu/~murray/papers/1993d_ms93.html).

Read substantive original body: §II.A/B, printed pp.701–703, including control-affine system (2), local bracket motion (3), distribution filtration and regularity; §II.D, pp.704–705, Hall construction; §III.A/B, pp.705–708, canonical system (8), Algorithm 1 and first/higher-order sinusoidal steering; §IV opening and §V discussion, pp.708 and 713–714. Some equations in the scan's text extraction are garbled, so the candidate independently fixes its sign convention and derives its four exact shear updates. No figure was relied on as visually inspected. Do not claim to have read the whole 17-page paper or its cited originals.

SOURCE summary: the paper represents constrained dynamics as controlled smooth vector fields, describes commutator motion and bracket filtrations, and constructs steering controls for specific canonical/chained structures. Full controllability depends on assumptions. Its discussion identifies restrictions and growing path complexity outside favorable structure. It does not supply the candidate's terminal-gradient-residual operation-allocation rule.

What the reading changed: **RECONSTRUCTION** — global reachability and the present instantaneous control span differ. This suggested asking a separate learning question: can acquiring a named composite change the first-order update span seen by an optimizer? The derivative comparison in candidate §3 is our consequential local derivation, not attributed to a source theorem on neural learning.

## Closest neural mechanism actually entered

Benjamin Walker, Andrew D. McLeod, Tiexin Qin, Yichuan Cheng, Haoliang Li and Terry Lyons, **Log Neural Controlled Differential Equations: The Lie Brackets Make a Difference**, arXiv:2402.18512v3 (28 October 2024), ICML 2024 paper. [Original HTML](https://arxiv.org/html/2402.18512v3). PMLR PDF access failed; OpenReview requested browser verification; Oxford PDF returned 403. The original arXiv body was accessible.

Read §§1.2–1.4; §§2.3–2.5, especially equations (13), (19)–(20); §§3.1–3.5, equations (21), (26)–(28). Did not independently audit its appendix proofs or experiments.

SOURCE summary: Log-NCDE constructs the Log-ODE vector field recursively from a neural vector field's brackets, weights these with input log-signature components, evaluates them with JVPs, and discusses regularity and increasing high-depth computation costs. Neural bracket fields and their JVP implementation are already concrete prior work. The candidate's task-triggered word promotion is not the inspected training organization.

Sign discipline: their equation (13) uses `J_a*b - J_b*a`; the candidate defines the chronological f,g,-f,-g leading field as `Dg*f - Df*g`. These are opposite notational conventions. All candidate signs use its explicit four-step chronology, not an unexamined transplant of (13).

## Additional direct control-learning precedent

Peyman Morteza, **Parameter-Efficient Generative Modeling with Controlled Vector Fields**, arXiv:2605.28267v1 (27 May 2026). [Original HTML](https://arxiv.org/html/2605.28267v1).

Read original §2 definitions, §3.1 Lemma 8 and Example 9, §§3.2–3.6 and Algorithm 1. Did not audit distribution-transport proofs or treat their proof-of-concept results as established general performance.

SOURCE summary: ChowFlow fixes bracket-generating vector fields and learns scalar state/time-conditioned neural controls with a CNF likelihood objective and backpropagation through an ODE solver. A fixed two-field geometric substrate plus learned neural controls is existing work, as is its Heisenberg example. This candidate's three-state arithmetic check is therefore a known geometry sanity check, not a newly invented computation.

## Unread boundaries and epistemic status

Sussmann's original orbit theorem, Lie-extension learning, adaptive motion-primitive acquisition, higher-order optimization and constructive/boosted network feature growth have not yet been substantively entered in this pass. Search snippets are not evidence that these fail to cover the formation rule. Source coverage is **describable difference in inspected originals / broader priority OPEN**.

No source claims a benefit for this candidate. No code, symbolic script, training or numerical experiment was run. The scope is original reading plus mathematical/algorithmic construction. Parent disclosed prior Lafont overlap before the v0 freeze; after014 was read only after that freeze. V1 is intentionally a post-comparison new theory entry, not a pristine blind context. Same run, instruction_only isolation and unknown effective mode.

## Post-v1-freeze actual-body comparison and disposition

Chronology: candidate-v1 and this source record were both written and hashed before the parent's second overlap disclosure arrived. The candidate hash was `d09b0adab1f32a5c9897f003cfccb8ea4cdcf474506a42106cb8b6c884752ea3`; the source record's **pre-append freeze** hash was `ddebae36e866302d6174c82bc9fa33734de37e58e555933aeaee82bc81ddf777`. Candidate-v1 remains unchanged. This section is a transparent later append, so the source record's current whole-file hash differs.

The parent then named, and this worker read their actual complete bodies:

- `candidates/resumed-r/candidate.md`, especially formation/update laws (1)–(5), exact cancellation calculations (6)–(8), and actual options/model-controller rivals.
- `candidates/resumed-r/nullspace-rival.md`, including its six-action identified-response controller against the ten-action nested loop.
- `candidates/after012-a/candidate-v0.md`, including the near-identity commutator zero-gradient obstruction in §6 and its post-freeze source/rival comparison.

**Exact comparison / RECONSTRUCTION.** In v1's worked substrate the old loop already gives `C(a,b):z<-z+a*b`. Fix any admissible h!=0 and use `a=h, b=c/h`; its endpoint is exactly `z<-z+c` and its coefficient derivative is 1. Alternatively choose `a=sqrt(|c|), b=sign(c)*sqrt(|c|)`; this is precisely v1's balanced normalized coefficient construction, allowing the appropriate inverse order for negative c. Balanced amplitudes make prefix excursions tend to zero with c, but an old parameterized loop can use the same amplitude law. Thus this does not establish a different first-order learning interface. Resumed R's equation (8) already supplies the higher-order version with nonzero fixed side amplitudes and a linearly tunable remaining argument.

The residual/JVP score in v1 is a more explicit **derivative-rich proposal heuristic**, not yet a different computational organization. V1 exposes known neural fields, exact derivatives and terminal backpropagation; resumed R originally learns unknown response functions from probes. A matched controller must receive v1's richer information. Ordinary derivative-aware control, adaptive bracket-basis growth or feature addition can then use the same candidate direction. No proof equates all acquisition costs, but there is presently no derived advantage either.

The second-order rival is concrete, not just a simulation objection: the primitive-coordinate loss has a mixed Hessian term corresponding to the bracket effect, and a nonzero pulse/perturbation can escape the zero-control stationary point. Restricting that rival to zero initialization and first-order descent would create the desired result by exclusion. After012 had already isolated the near-identity obstruction, so v1 must not claim that observation as a fresh campaign discovery.

**Judgment:** retain v1 as an explicit conditional learning-coordinate reconstruction and a paid operation-allocation proposal. Do not promote it as a new principal mechanism. The finite-work threshold, exact-limit derivative caution and high-depth proof obligation sharpen its limits; they do not rescue novelty or significance. No test or further blind theory pivot follows from this comparison. The result returned to root is that **this entry obtained no substantiated new principle**. Root retains the next allocation decision.
