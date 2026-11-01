# after012-a v0 — learning a quotient of paths through noncommutative transport

Status: **FROZEN CONSTRUCTIVE v0, before original-source comparison.** This is an AGENT PROPOSAL, not a novelty claim or an executed result. The predecessor supplied only the phrase-level seed reported by the coordinator: continuous learned holonomy, local connection learning, compositional error transport, learned symmetry quotient. No predecessor derivation or saved evidence is assumed.

## Scope, provenance and execution boundary

- Read for this task: root `AGENTS.md`, `README.md` including Research Vision, and `deep-work/WORKFLOW.md`.
- This task owns only this file and `sources/after012-a/reading.md`. No earlier candidate, catalog or campaign result was read before this freeze. Original-source comparison is appended below only after this freeze.
- Isolation is **instruction_only**. The delegated context includes the supplied standing instructions and the coordinator's task; no technical filesystem or context isolation has been independently established. Independence of source reading or mathematics is not inferred from a second model call.
- Requested model/mode: the delegated task says to continue the existing V2 campaign; its authoritative run request is not independently visible here. Observed: an assistant task with shell, web and collaboration tools. Effective server mode, actual model and effort: **unknown**. No model identity is self-certified.
- Authorized: construction, reading, these two artifact writes, existing CPU resources; no experiments in this task. No installations, paid API, external release, or online-organization branch. No experiment was run.

## 1. The scientific question and success counterfactual

**Question.** Can task learning discover which continuous deformations of an input history should be computational identities, while retaining the noncommuting effects of the remaining paths as reusable internal operations?

This addresses the Fundamental Question through **the acquisition of an internal operation algebra**, rather than assuming a library of named operations. An input is a continuous control trajectory; a learned local transport field turns the trajectory into a state transformation. Learned near-identity relations identify nuisance path variations. Nontrivial holonomies are operations available to arbitrary incoming states, so their chronological composition performs computation.

The important candidate limit is that a recurrent system may learn many geometrically different trajectories separately, with no constraint relating their effects. If a learner discovers an appropriate quotient of paths, it can share an operation across a whole class of deformations and compose it without retaining the trajectory. The ambition is an **acquired computational equivalence relation**, not merely a smoother recurrent model.

**Success counterfactual.** If the learner, from the same endpoint labels and compute as a strong recurrent/controlled-ODE rival, acquires correct operations on held-out shapes *and previously unseen combinations*, a mechanism that changes which histories count as the same computation would become credible. The structural reason for extension is that local relations entail families of global path equivalences; this is stronger than memorizing finitely many demonstrated loop shapes. Orthogonal transport gives a concrete, nonexpansive error law for that entailment.

What would remain different from a generic learned connection is the learned relation/defect organization and its measured effect on acquisition and composition. If the only result is ordinary regularization, stable integration, or resource allocation, this candidate does not earn the main scientific line. No claim of universal computation, autonomous open-ended evolution, or a new mathematical theory follows.

The weakest step is **learning the useful relations rather than collapsing all loops or protecting accidental ones**. Endpoint supervision may not identify them. This is not delegated to a verb such as “discover.” Sections 3–6 give an executable candidate rule and expose its limitations.

## 2. State and execution: an actual transport machine

### 2.1 Initial finite version

Let a finite, oriented cell complex K have vertices V, edges E and oriented polygonal faces F. It discretizes an observed compact control domain B; initially the model is *given the local geometry*, not the task-relevant holes, generators, word decomposition, homology class or target operators. An observed continuous trajectory is sampled into a path of adjacent vertices. Sampling costs and mesh resolution count against every rival. A learned observation-to-base map is an OPEN extension, not part of v0.

Each undirected edge stores one matrix U_e ∈ SO(d). The reverse traversal uses U_eᵀ exactly. At a vertex v the running state is h ∈ R^d; the complete execution state is (v,h). For a path γ=(e₁,…,e_L),

    h₀ = Eψ(x),                     h_k = U_{e_k} h_{k−1},
    P_γ = U_{e_L} ⋯ U_{e₁},        ŷ = Dψ h_L.

Task episodes start and end at a specified root o. x is an independent problem input; Eψ is shared across all paths, and Dψ only sees the final fiber state. Neither encoder nor decoder receives the trajectory, a loop word, relation choices, or task identity. Thus the transport is the only path-dependent computation. More expressive shared encoder/decoder networks are allowable but fully counted and matched in rivals.

The operations P_γ are acquired through their edge factors; no routine extracts a word and calls a critical-loop compiler. Every path segment updates h directly. The same closed trajectory acts on every incoming h, and sequential loops compose by matrix multiplication. This is linear in the fiber and consequently limited; the dimension d is a resource parameter, not a hidden infinity.

Memory and work are O(|E|d²) and O(Ld²), respectively; the discrete relation mask below uses |F| bits. A dense training step also pays for all face products used in the regularizer. No operation table exponential in trajectory length is available to this model. A rival may use its same total parameter and compute budget in a different form.

### 2.2 Continuous interpretation, not an extra implemented system

For q(t)∈B and skew-symmetric A_i(q;θ)∈so(d),

    d h(t) / dt = −Σ_i A_i(q(t);θ) q̇_i(t) h(t).

This is a controlled differential equation with linear action on its fiber. U_e is its path-ordered transport across a short edge. Its curvature is

    F_ij = ∂_i A_j − ∂_j A_i + [A_i,A_j].

Sign conventions depend on whether the transport equation uses −A or +A; none of the finite v0 proofs depends on that convention. Flatness on a simply connected region makes transport path-independent there under the usual smoothness assumptions. A flat connection on a multiply connected region may have nontrivial holonomy. v0 does not assume that a visual loop must carry useful computation: the training objective must decide which local loops should be identities.

### 2.3 Gauge freedom is not new capability

For vertex frames G_v∈SO(d),

    h_v ↦ G_v h_v,       U_(u→v) ↦ G_v U_(u→v) G_uᵀ.

A root loop operator conjugates by G_o, and root encoder/decoder transform accordingly. Such a change of frame preserves behavior and is **functional equivalence**, not discovery. Rotating a drawing or changing a fiber basis cannot count as computational innovation. Only changed input-output capabilities, generalization, or acquisition cost can count.

## 3. What structure is learned

For an oriented face f with boundary path ∂f, let C_f=P_(∂f). An optional learned relation bit r_f∈{0,1} says that the boundary of f is proposed to be a nuisance identity. Formally, filling the selected faces yields a 2-complex K_r, whose path relations include ∂f≈identity when r_f=1. Faces with r_f=0 are potential computational defects, not pre-labelled operations.

The finite learner minimizes

    J = average task loss
        + μ Σ_f r_f ||C_f−I||_F²
        + λ Σ_f (1−r_f),                  μ,λ > 0.

The last term charges for leaving a local loop unresolved; otherwise the vacuous choice r=0 everywhere has no quotient effect. The task term prevents indiscriminate flatness when a loop effect is required. Holding U fixed, the exact coordinate minimizer is

    r_f = 1  if μ||C_f−I||_F² < λ,
          0  if μ||C_f−I||_F² > λ,

with a fixed tie convention. Holding r fixed, take the Lie-group gradient steps in Section 4. Alternate on a schedule fixed before any evaluation. This is nonconvex alternating optimization; no global convergence or successful relation discovery is asserted.

**Critical honesty.** With finite μ, r_f=1 does not impose C_f=I exactly. K_r is the model's proposed quotient; its realized transport is only an approximate representation of that quotient. Calling this exact learned topology would be incorrect. The bound in Section 5 quantifies the gap. An exact flatness parameterization or justified constraint solver is a genuinely missing possible revision, not an already implemented compiler. Arbitrary-length exact compositionality is not claimed.

The regularizer may choose to concentrate necessary holonomy in a few faces. That is a learned organization of the connection, but it may merely exploit the discretization. Mesh changes and geometry-held-out trajectories are therefore central counterconditions, not cosmetic robustness checks.

## 4. Learning signal, transport of error and costs

The only supervised observations are examples (x,γ,y), with y an endpoint task outcome. No target edge matrices, curvature map, defect position, generating loops, homotopy labels or full operator is supplied. If an environment supplies those, the comparison is a different, easier scientific question.

For differentiable loss ℓ(Dh_L,y), set δ_L = Dᵀ ∂ℓ/∂ŷ. If U_k is the matrix actually used at step k, the exact adjoint recursion is

    δ_{k−1} = U_kᵀ δ_k.

For a left variation U_k(ε)=exp(εΩ)U_k with Ωᵀ=−Ω,

    dℓ/dε |₀ = δ_kᵀ Ω h_k
              = ⟨skew(δ_k h_kᵀ), Ω⟩_F.

Thus a one-use edge update is

    U_k ← exp(−η skew(δ_k h_kᵀ)) U_k,

before adding face-loss and shared-parameter contributions. If an edge is reused, sum all of its occurrence gradients in a common chosen edge orientation; for a reverse occurrence differentiate U_eᵀ, rather than pretending it is an independent parameter. The face term has the same product-rule/transport structure on a short loop.

This gives a concrete local-in-state outer-product rule **after an endpoint error has been transported backward**. It is ordinary exact reverse-mode credit assignment, not biologically local learning. It requires either storing h_k or recomputing the trajectory; O(Ld) saved states and O(Ld²) reverse work are a direct implementation. Orthogonality preserves ||h_k|| and ||δ_k||, but does not make optimization easy or eliminate the supervision cost. No free online organization is claimed.

Task-only supervision identifies an operator only on the states it probes. If every episode starts at one h₀, two matrices agreeing on h₀ are observationally indistinguishable. To claim a reusable operator, the training inputs must excite enough fiber directions or the held-out composition test must reveal those directions. Even d linearly independent probes only identify an operator if the input/output observation map exposes them. Sparse scalar feedback can require many more examples; its sample cost must be measured rather than silently replaced by full operator labels.

## 5. A checked compositional derivation

**RECONSTRUCTION, finite exact statement.** Suppose all edge matrices are orthogonal. Let γ′ be obtained from γ by replacing one contiguous subpath α from u to v with β from u to v. Write the prefix transport Q and suffix transport S. Then

    P_γ  = S P_α Q,       P_γ′ = S P_β Q,
    ||P_γ−P_γ′||₂ = ||P_α−P_β||₂
                   = ||P_βᵀP_α−I||₂.

The first equality uses orthogonal invariance of operator norm. The replacement-loop holonomy P_βᵀP_α is based at u. If a sequence of N elementary replacements transforms γ into γ′, each corresponding local loop having residual ε_j, the triangle inequality gives

    ||P_γ−P_γ′||₂ ≤ min(2, Σ_(j=1)^N ε_j).

Therefore ||P_γh−P_γ′h||₂ ≤ ||h||₂ Σ_j ε_j. A decoder of Lipschitz constant L_D multiplies this bound by L_D. The bound is independent of the lengths of unchanged prefixes/suffixes. This is a concrete mechanism by which local near-identities constrain novel globally deformed trajectories; it is not a claim that every pair has a short filling.

For a one-face replacement, ε_j=||C_f−I||₂ (up to inverse/conjugation, which preserve the norm). For a multi-face deformation, choose an explicit elementary filling. Repeated use of a face counts repeatedly. If each selected face has ε≤ε₀ and the held-out deformation needs at most N faces, the bound is Nε₀. Small average face penalty alone does not imply a uniform ε₀, a small N, or good task accuracy.

**ATTACK: finite flatness tolerance is not indefinite memory.** A face with residual rotation angle ε has C_f^N rotating by Nε. Taking N of order 1/ε produces order-one error although the single-loop penalty is tiny. Orthogonality prevents exponential norm growth but not accumulated semantic error. Unlimited composition would require exact relations or another correction mechanism; v0 has neither.

## 6. Worked noncommutative computation and a learning obstruction

Consider a base with two task-relevant holes. Let learned transports around based loops a,b be

    A = R_x(π/2) = [[1,0,0],[0,0,−1],[0,1,0]],
    B = R_y(π/2) = [[0,0,1],[0,1,0],[−1,0,0]].

For h₀=e_z, executing a then b gives BAe_z=−e_y; executing b then a gives ABe_z=e_x. The order is retained in the internal state although both histories end at the same base point. A readout along e_x distinguishes them. Deforming either loop through selected flat faces preserves its operation exactly when the face holonomies are identity, and approximately by Section 5 otherwise. The input trajectory itself need not be remembered once its transport has acted.

This is a **worked possible mechanism**, not a trained solution. These particular quarter-turn matrices generate only a finite rotation group, so the example cannot store arbitrary distinct words or demonstrate unbounded computation. Equal operators define functional equivalence even when words or geometric paths differ. SO(3) is chosen for transparent arithmetic, not declared the best representation.

Learning from commutators alone is already difficult. If A=exp(X), B=exp(Y), then

    ABA⁻¹B⁻¹ = I + [X,Y] + terms of degree ≥3.

At X=Y=0, the first derivatives of the commutator effect vanish. Initializing every edge to identity can therefore give zero first-order task signal for data that only distinguishes order through commutators. A near-identity random initialization or a curriculum exposing individual actions may help, but each is an additional assumption with charged examples and no guarantee. This obstruction prevents a false inference from representability to learnability.

## 7. A constructive retention extension: neutral defect birth

This is a proposed extension to study, not smuggled into the trained v0 result. Let R be a currently flat disk-shaped subcomplex. Preserve every edge outside R and on its boundary. Add internal connectivity/refinement within a fixed maximum mesh budget. There exist edge assignments with two inner based defect loops carrying H and H⁻¹, with the paths to their common base accounted for, whose aggregate boundary transport remains identity. In a gauge where a shared connecting tree has identity transports, this is simply H⁻¹H=I.

All computations whose executed edges stay outside the interior of R are preserved exactly because none of their factors changes. A newly available trajectory entering R can isolate H and therefore access a new operator. This supplies a possible *mechanism* for expanding an operation algebra without altering previously executed external traces; it is not merely a renamed claim that any new feature is retained.

The limitation is sharp: an old trajectory that crossed R internally is unprotected; an external boundary observation cannot identify H; a task has to probe the new internal route. Keeping the same boundary holonomy alone does not preserve every boundary-to-boundary internal path. No general protection against forgetting follows. This may reduce to identity-preserving network expansion or frozen sparse modules. Its value would require a nontrivial acquisition/generalization consequence beyond that rival. It is not a warrant to work on the parked online-organization branch.

## 8. What is and is not self-organizing here

| Dimension | v0 mechanism |
|---|---|
| Manually designed | observed base mesh, fiber group SO(d), transport law, alternating loss, capacity/compute bounds |
| Executed locally | one edge transport per input segment; no precomputed loop word |
| Optimized externally | edge matrices, optional shared encoder/decoder, relation mask via labelled episodic loss |
| Acquired organization | proposed local identity relations and locations/values of remaining nontrivial transports |
| Retained | edge parameters/mask across episodes; no evolutionary inheritance |
| Possible emergent operation | a task-useful holonomy acting on varied incoming states, if demonstrated |
| Not established | autonomous primitive invention, efficient learning, exact quotient, continual retention, universality, novelty |

Nontrivial holonomy by itself is not scientific evidence of new computation. It can be an imposed operation, a spurious defect or a gauge description of an already known recurrent mechanism.

## 9. Strongest pre-source attacks and what could reverse the candidate

1. **Generic controlled dynamics.** The continuous form is already a linear-fiber controlled differential equation. A learned connection plus endpoint reverse mode is not the contribution. The learned relation objective must explain an actual difference under matched data and resources.
2. **The base supplies the answer.** If the task-relevant punctures or loop words are supplied, operation discovery has been replaced by fitting a representation. The initial mesh must include nuisance faces without revealing which relations are useful. Even then, the choice to encode inputs as paths may be substantial prior knowledge.
3. **Underidentification.** Endpoint labels can fit many incompatible holonomies. Held-out probes and compositions, not training accuracy, must expose operator reuse.
4. **Collapse or discretization artifacts.** λ and μ may select arbitrary concentrated curvature rather than meaningful computation. A single face of huge curvature can game the sparsity term. Mesh refinement and equivalent base coordinate changes could destroy the organization.
5. **Bayesian predictive state.** A compact belief or predictive-state filter may solve exactly the task equivalence needed, without transport geometry. If its matched finite-budget acquisition is as good, geometric elegance is insufficient.
6. **A fixed equivariant prior does the work.** When true homotopy labels are known, a fixed flat connection/representation learner may win. v0 only matters if useful quotient relations are learned where that prior is unavailable or partly wrong.
7. **Norm stability is not computational power.** SO(d) excludes dissipative updates and can require larger d or a readout to implement ordinary noninvertible operations. All capacity and sample cost must be charged. Stable composition can coexist with a useless hypothesis class.

## 10. Finite next decision; no execution authorization inferred

The next task is **original-source comparison**, appended below after freezing this constructive part. It must decide whether the relation/defect learning rule has a substantive remaining difference from learned connections, Neural CDEs, gauge/topological models and Bayesian/predictive-state learning. If covered, stop or reconstruct the specific covered result honestly; do not invent an experiment to keep the candidate alive.

If a meaningful difference survives, a later scoped validation contract would have to freeze: endpoint-only supervision, mesh and total matrix budget, trajectory sampling, geometry-held-out *and combination-held-out* tasks, total formation/maintenance cost, rival optimizers, relation-removal intervention, and finite length/filling bounds. That contract and all experiments are outside the present task.

`CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

Learned approximate path quotients could yield reusable noncommuting internal operations | HYPOTHESIS, constructive v0 frozen | explicit equations, finite norm derivation, SO(3) calculation in this file | acquisition/identification of useful relations from task labels; nearest prior work still unread at freeze | identity initialization fails on commutator-only first-order signal; finite curvature tolerance cannot imply indefinite exact composition | read original load-bearing rivals and decide whether a consequential difference remains.

## Post-freeze original-source comparison

This heading is the boundary between independent construction and source-informed revision. The frozen v0 above is not retroactively attributed to sources read later. After freeze, the coordinator authorized reading its `sources/after012-root/reading.md`, `candidates/R/candidate-v0.md`, and `candidates/after012-b/candidate-v0.md`; those were read. This ends any independent-candidate comparison claim. The comparison below also uses my own entry into original texts, recorded with locations in `sources/after012-a/reading.md`.

### 11. An exact attack on the first learning rule

The coordinator independently supplied this attack; I also derived it. Eliminating the independent relation bits in Section 3 gives exactly

    min_r J(U,r) = L_task(U) + Σ_f min(μ||C_f(U)−I||_F², λ).

Consequently a conventional orthogonal linear recurrent transport trained with this saturated curvature penalty has the **same states, accessible information, loss and updates**. The mask has no causal role in execution or parameter sharing. This is an exact same-mechanism reduction, not a universal-simulability objection. It defeats a claim that this particular mask already acquires a new computational organization.

The original-source pressure is also concrete. [Path Development Network](https://arxiv.org/pdf/2204.00740) has the trainable multiplicative Lie-group recurrence and manifold backpropagation; [MatrixNet](https://arxiv.org/html/2501.09571) already supplies a relation-residual loss for learned matrix representations. Their checked mechanisms cover the broad claims of learned executable transformations, order-sensitive matrix composition, inverse handling and norm-preserving error transport. Their exact architectures differ from a position-dependent edge connection; that difference does not rescue the v0 mask.

**Present status of Sections 2–6:** coherent connection learner and bounded error reconstruction; **not a supported new learning principle**. Section 7's neutral defect pair also does not rescue the contribution. The H/H⁻¹ pair provides just one freely selectable operator and its inverse, so that pair alone generates a cyclic subgroup and cannot itself create a noncommuting pair of new operations.

### 12. Constructive repair: accepted relations become exact parameter sharing

This is a **central mechanism revision**, not a new name for the soft mask. Keep direct path transport at execution, but let a finite set R of accepted loop equations define its parameterization:

    M_R = { U∈SO(d)^|E| : P_γ(U)=H_γ for γ∈R }.

For a proposed nuisance relation, H_γ=I. For a retained acquired operation, H_γ is the *learner's own* previously learned operator, stored before a later update. It is not an oracle target matrix. Freezing that H preserves the previous model's operation, including its previous mistakes. Endpoint-only supervision and the excitation limitations in Section 4 remain unchanged.

The repaired state is (v,h; R,Φ_R,ξ), where Φ_R maps the free coordinates ξ to every edge matrix U. Every forward pass uses U=Φ_R(ξ). A relation has an execution effect because it removes independent factors and ties all their occurrences to the remaining factors. A task gradient cannot change an accepted identity merely because its penalty coefficient is finite.

**A concrete exact tying rule.** If an equation contains a pivot edge once and can be written Q U_e S=H, solve

    U_e = Q⁻¹ H S⁻¹.

Substitute this expression into **every** other retained equation and every use of U_e, including neighboring faces and reverse traversals. Local, independent per-face repair is invalid: changing a shared edge changes its neighbors. New relations can leave residual equations on the nonpivot variables, rather than eliminating another free factor. If the substitution dependency graph has a cycle, an expression exceeds its externally fixed size bound, or a residual system lacks a supplied feasible parameterization, the proposed edit is rejected or kept OPEN. No general noncommutative equation solver or word-problem oracle is assumed.

This is an algebraic tying step, not a loop-word compiler for task execution: all incoming path segments still act on h. However it is a substantial symbolic/structural scaffold. Charge the expression DAG, stored H matrices, relation evaluation, all substitutions and gradients. Materializing products of d×d factors can cost O(T_R d³) per parameter update for expression size T_R, before the O(Ld²) path execution. Numerical floating-point residuals are not mathematically exact equalities; no numerical implementation or tolerance guarantee has been checked.

### 13. Geometry of the allowed learning update

The following derivation shows what the hard rule gains and what it costs. It does not invent a new optimizer.

Write a left edge variation as δU_e=Ξ_e U_e, Ξ_e∈so(d). For a loop P=U_L⋯U₁, differentiating the product and multiplying by P⁻¹ gives

    δP P⁻¹ = Σ_k Ad_(U_L⋯U_(k+1))(Ξ_k),
    Ad_G(X)=GXG⁻¹.

For a reverse occurrence of a stored edge, the variation to insert is −Ad_(U_e⁻¹)(Ξ_e). Thus all accepted loop equations define a concrete linear map D_R from edge variations to loop variations. Feasible infinitesimal updates obey D_R Ξ=0. On a regular smooth stratum, an edge-gradient g has orthogonal feasible component

    g_feasible = g − D_R* (D_R D_R*)† D_R g.

Here * denotes the adjoint in the charged Frobenius metric and † the Moore–Penrose inverse. This is ordinary equality-constrained/Riemannian projection. A finite exponential step in this tangent direction does **not** in general satisfy the nonlinear relations; exact tying through Φ_R or a justified feasible retraction is still required.

Root-fixed gauge variations Ξ_(u→v)=φ_v−Ad_(U_(u→v))(φ_u), with φ_o=0, leave every rooted loop action unchanged. They lie in ker D_R. Therefore the candidate's functionally usable deformation space is ker D_R modulo these gauge directions, not its raw count of trainable parameters. A simply connected, fully flat region has only such gauge freedom; a learning signal cannot obtain a new rooted holonomy there while all its relations remain fixed.

This is a real obstruction criterion: if task loss is high but its feasible gradient has only gauge directions, an accepted relation or representation assumption must change. It does not itself decide which relation is false. A small feasible gradient can also mean an ordinary optimization saddle or poor excitation. Calling every such event “need new topology” would be false.

**Singularity attack.** At A=B=I, the derivative of ABA⁻¹B⁻¹=I vanishes even though a generic finite pair of nearby changes violates commutation. First-order nullspace size can therefore overstate actual freedom. The smooth-stratum assumption is load-bearing; the v0 commutator obstruction also attacks this repair.

### 14. A stronger operation-refinement rule, and its coverage boundary

Instead of inserting only an identity pair, retain an already acquired coarse operator H while learning its internal factors:

    BA=H,        B=HA⁻¹.

For any finite skew update Ξ,

    A′=exp(ηΞ)A,       B′=B exp(−ηΞ),       B′A′=BA=H.

Thus the incoming task error can train a new accessible partial-path action A while its compensating action B preserves H for **all incoming fiber states**, not just a finite set of old examples. This can create noncommuting actions with the retained H. It is stronger than the neutral pair when H≠I. It is a precise finite update, not a small-angle retention heuristic.

Overlapping retained operations constrain this freedom. If both BA=H and AB=K are preserved, then AHA⁻¹=K. Given one solution A₀, all solutions are A=A₀C with C in the centralizer Z(H). Consequently a previously learned pair of overlapping macro-actions can leave fewer usable deformation directions than one macro-action, even at the same edge/fiber count. This is an explicit reason that *organization of retained operation constraints*, rather than parameter count alone, can affect later acquisition. It is not evidence that the effect helps.

The source boundary matters: [Network Morphism](https://proceedings.mlr.press/v48/wei16.pdf), §3.1 Eq.3, already formulates function-preserving linear refinement as a factorization G=F₂F₁. Our BA=H special case is covered at that level. Exact preservation throughout training and overlapping noncommutative path constraints are a more specific proposed use, but no independently consequential acquisition rule has yet been established. This is a construction to examine, not a novelty escape hatch.

### 15. Relation acquisition and revocation: an explicit but presently conventional rule

A finite structural search can propose an unsupplied face identity, an equality of already observed rooted paths, or removal of a provisional identity. For each candidate R′:

1. Construct a bounded Φ_R′ by the global elimination rule; reject edits whose parameterization is unresolved.
2. Refit free coordinates using the **same endpoint examples**, charging all optimization work.
3. Score a held-out endpoint likelihood plus an actual description/execution charge for Φ_R′, free coordinates and stored operations; accept only the specified improvement. No target operator, loop class or selected relation is disclosed.
4. Keep rejected alternatives/evidence and permit revocation if a new same-budget endpoint contrast contradicts the relation. A false relation must be removable; finite agreement is not a proof of an identity.

This differs causally from the original saturated penalty: accepted identities now determine the feasible update parameterization and exact induced equivalences. But the rule itself is model selection with equality constraints. A generic equation learner with the same proposal class, equality parameterizations, labels and cost can perform the same procedure. A Bayesian rival with a point-mass component on each accepted constraint manifold and a continuous alternative can make the same structural choice while retaining uncertainty; the hard learner's MAP score is not a new inference principle.

The unresolved research possibility is narrower and more consequential than “learn a connection”: can the **organization of already acquired path actions supply an effective, noncatalogued rule for choosing which operation to refine or which equivalence to revoke**, with a demonstrable future-acquisition effect after all constraint/search costs are included? The projected-gradient obstruction and centralizer calculation expose information such a rule might use. They do not yet provide the rule, and merely ranking normal-gradient magnitudes would reduce to active-set optimization.

### 16. Matched alternatives after repair

| Rival | Same available information | Actual shared or different mechanism | Required cost comparison |
|---|---|---|---|
| Orthogonal edge RNN with saturated curvature penalty | Identical endpoint examples and mesh | Exactly v0 Sections 2–4 after eliminating r | Exactly equal; v0 has no advantage to test |
| Position-dependent Neural CDE / development layer | Same trajectories, inputs and endpoint labels | Continuous state response and multiplicative operation learning already available; no new-law claim from these ingredients | Include interpolation, matrix factors, integration/adjoint work, precision and optimization |
| Matrix representation learner with relation loss | Same observed paths, with **no supplied true relation list** | Directly covers soft learned operator products; supplying correct relations only to the rival or candidate changes the question | Count relation search and all failed structural fits |
| Exact equation learner + constrained SGD | Same relation proposals, Φ construction and examples | Same hard-constraint repair unless another acquisition rule is supplied | Same expression DAG, solve/retraction, gauge handling and backward work |
| Function-preserving network refinement | Same learned coarse H, new endpoint labels and free-factor budget | Covers the single BA=H retention construction | Charge new factor, routing/access, constraint maintenance and task training |
| Bayesian connection/structure learner | Same endpoint-only likelihood, same hypotheses and model capacity | May retain uncertainty over relations instead of committing to one | Charge every posterior particle/structural alternative; do not give it free oracle topology |
| Predictive-state learner | Only observations/queries actually available in this task | History equivalence is already a predictive-state idea; standard discovery may require additional queries absent here | Do not hand it intermediate observations or target full-state transitions |

There is no current finite-budget result that defeats these matched alternatives. Their ability to express the mechanism is not by itself an impossibility theorem for a future acquisition advantage. It does prevent claiming an already obtained different principle.

### 17. Judgment after the constructive pass

**Do not promote after012-a to main on the present evidence.** The main obtained result is an explicit reduction of its first learning rule to a known kind of recurrent transport regularizer. The serious repair changes the parameterization, derives exact globally coupled error compensation and exposes a gauge/constraint obstruction to new operation learning. It remains an incomplete mechanism hypothesis whose available acquisition rule is covered by constrained equation/model learning.

What might reverse this judgment is an actual noncatalogued structural update rule using the obstruction and retained-operation organization to acquire a useful new action with a total-cost advantage, or a theorem showing a distinctive learnability effect under a stated restricted source class. Neither has been obtained. Running more SO(3) rotations, increasing the dimension, adding nonlinearities, or measuring holonomy would not answer that gap.

`CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

v0 relation mask changes computation beyond a regularized orthogonal transport learner | FALSIFIED in exact stated sense | min-r derivation in §11 | no stronger claim implied | soft relation mask | do not test it as a new-principle mechanism.

Hard acquired relations can constrain future transport learning and preserve acquired loop actions | RECONSTRUCTION / incomplete implementation | pivot tying, derivative, exact BA=H update in §§12–14 | bounded global parameterization and unsupplied useful relation acquisition | independent face repairs; tangent-only finite update; neutral identity pair as noncommutative discovery | decide whether a genuinely different structural acquisition rule exists before training.

A consequential new learning principle has been obtained | OPEN / unsupported | source comparison and exact matched rivals in §§15–16 | no residual rule beyond equation/model learning | continuous holonomy alone; norm-stable adjoints alone; saturated mask; simple factorization retention | coordinator should re-anchor or formulate one new theoretical gap; no experiment is warranted by this artifact alone.
