# after017 outside-principle — candidate v0

## Status and exposure boundary

AGENT PROPOSAL; DWS-E reserve, not MAIN. This is a serious mechanism construction with an unresolved representational step, not a novelty, performance, or paradigm claim. No experiment or training was executed. This file is to be frozen before comparison with current candidates or old reports.

Read before this freeze: effective root AGENTS.md, README.md Research Vision, deep-work/WORKFLOW.md, and the primary material recorded in the paired reading.md. Did not read NOW/water/now.md, current book index, current candidates, old reports, or reviews. Other agents' architectures were not supplied. Context separation is instruction_only on a shared filesystem. Requested subtask: fresh-context constructive reserve inside the existing V2 run. Server effective mode and precise model/effort are unknown; no Pro claim. No installation, new paid route, model change, experiment, or publication.

## Judgment

**HYPOTHESIS:** a neural system could acquire an internal mechanism as an admissible relation among trajectories, then produce different executable operations by changing which variables are specified, composed, or hidden. The learned object would preserve a family of possible computations instead of preserving only one trained input-to-output use.

The concrete proposal is **behavior-first mechanism acquisition**: traces → a learned law → composition and elimination → a boundary-specific executable operator → later traces that correct the law. Learning the law and selecting the task boundary are different changes. The intended limitation is that acquiring one directed use of an internal process does not automatically yield its other computational uses; training a forward predictor can leave smoothing, inverse use, completion, and composition as separately learned skills.

The finite linear construction below works algebraically, but it is already matched by conventional identification plus linear solving. That is a real prior-work/alternative boundary. The high-risk scientific wager is whether a neural system can learn internal representations and port choices in which useful behavioral laws become compact enough that this loop constructs operations efficiently across substantial nonlinear computations. That step remains OPEN and cannot be smuggled into the worked example.

## What the unfamiliar original theory changed

SOURCE: Willems's behavioral approach starts from sets of allowed trajectories; inputs and outputs may be derived later. Interconnection is variable sharing. His linear differential-system elimination result removes latent variables without leaving that class. The original also explicitly restricts this closure and warns that nonlinear elimination need not work. See the paired ledger for exact locations and scope.

RECONSTRUCTION: an operation need not be remembered first as a callable directed map. It can be remembered as a law plus a procedure for imposing a boundary. That changes the proposed unit of acquisition: the learner acquires something from which several directed mechanisms can be made. It does not imply the learner has acquired the ability to solve every boundary problem efficiently.

This is not a claim that existing computers cannot represent relations or that fixed arithmetic primitives must emerge. The matrix and solver substrate below is designed. The candidate's acquired objects are the laws, their port scopes, and the compiled operators they entail.

## Explicit system state

Use real-valued, finite-length trace windows first; no infinite-trajectory or nonlinear theorem is assumed. A descriptor c chooses a subset of currently observable internal ports and a lag L. Its flattened window z belongs to R^d, d=q(L+1). The mechanism state is

    S = (theta, {C_c,n_c,D_c,R_c}_c, {program_m}_m, active_boundary, active_z).

- theta: a neural proposal policy over a finite, declared set of port/lag descriptors and composition/elimination proposals. It is not an oracle for the correct law.
- C_c=sum_i z_i z_i^T, n_c: accumulated uncentered second moment and count for descriptor c. D_c is a bounded archive of original windows and task witnesses; retaining the archive costs memory.
- R_c: a row-orthonormal basis of ker C_c in the exact noiseless case, or a declared near-null eigenspace in the approximate case. Its proposed behavior is B_c={z:R_c z=0}. Constants require an explicitly appended 1 coordinate; affine and homogeneous cases are not silently conflated.
- program_m: a law, manifest port names, validity conditions, the compiler's elimination record, and executable arithmetic for one or more requested boundaries. It contains a reusable relation, not only the answer to the trace that produced it.
- active_boundary: a selector P and value b supplied by the current task; active_z is the current candidate completion.

The finite descriptor vocabulary, arithmetic, rank tests, memory cap, proposal network, and compiler are manual design. What can be learned is which internal ports/lags exhibit a usable law, its coefficients, what previous operations it can join, and whether a compiled boundary operation earns its full formation and maintenance cost.

## Actual state transitions and learning law

### 1. Acquire a law from actual execution

An existing neural computation or environment produces a window z. The system records that window, not an invented counterfactual output, and makes

    C_c <- C_c + z z^T;  n_c <- n_c+1.
    R_c <- orthonormal_basis(ker C_c).

For finite exact arithmetic this is an explicit learning rule. If the observed windows span the true local linear behavior B*, then ker C_c=(B*)^perp, hence ker R_c=B*. Proof: a^T C_c a=sum_i(a^T z_i)^2; this is zero exactly when a annihilates every observed window. The conclusion requires the span assumption. With fewer directions, the inferred behavior is only span{z_i}; it can impose false laws on unobserved behavior.

For floating point/noise, replace exact zero by a predeclared spectral threshold, store residual bounds, and mark the result approximate. No exact preservation follows from a small empirical eigenvalue. This version has not been tested.

An increment C<-C+zz^T can only shrink its exact kernel, so the modeled behavior can expand as false restrictions are refuted. All archived windows remain admissible in exact arithmetic. This proves replay consistency for those windows, not preservation of all earlier capability.

### 2. Compose and construct an acquired operation

Suppose two learned laws are R_1 z_1=0 and R_2 z_2=0. A proposal identifies some corresponding ports by equality J_1 z_1=J_2 z_2. Stacking those equations gives

    L_w w + L_h h = 0,

where w are the ports to retain and h are the proposed hidden internal ports. Compute N whose rows span the left nullspace of L_h. Install

    R_new = N L_w;    B_new={w:R_new w=0}.

**RECONSTRUCTION, exact finite linear proof:** an h exists iff -L_w w lies in im L_h, iff it is orthogonal to ker L_h^T, iff N L_w w=0. Thus this elimination preserves exactly the manifest relation of the composed system. It can remove an entire internal execution from the representation used by later tasks. No execution-speed gain follows automatically: fill-in and dense coefficients can make the new operation costlier.

The elimination record is the causal derivation of the new operator. It does not certify that the learned input laws describe their domains correctly. Composing false laws preserves a false model exactly.

### 3. Execute under the current boundary

Given R and a task boundary Pz=b, form A=[R;P] and rhs=[0;b]. A rank-revealing factorization first checks consistency. For target selector T, target values are unique exactly when ker A is a subset of ker T. If so, compile the corresponding linear map from b to Tz. Otherwise return the unresolved degrees of freedom, or use a separately declared task objective to choose among them; silently selecting a pseudoinverse solution would invent a preference.

An explicit iterative execution, when exact compilation is not used, is

    z_(k+1)=z_k-eta [R^T R z_k + P^T(P z_k-b)].

For 0<eta<2/||A||_2^2 this converges to a least-squares minimizer; a zero residual and target uniqueness are still separate checks. Each step, residual check, and factorization counts as execution cost. The state-transition rule is fixed; the learned R changes what internal operation that rule executes.

### 4. Form/select proposals using feedback

For a sampled proposal c~pi_theta(c|summary(S)), fit/compose/compile as above. Obtain a task loss from actual task feedback and charge all trace collection, fitting, validation, factorization, runtime, and retained bytes. Let

    J(c)=task_loss(c)+lambda*formation_work(c)
         +mu*future_execution_work(c)+nu*retained_bytes(c).
    theta <- theta-eta_theta [J(c)-baseline] grad_theta log pi_theta(c|summary(S)).

The future-work term is estimated from a declared query distribution, not knowledge of held-out tasks. A candidate replaces an installed operator only if the original witness archive remains within its declared residual tolerance and its measured score improves on the selected tasks. These are executable acceptance and policy updates; they are not a proof that the chosen port vocabulary makes the search affordable. The neural proposer is optional in the linear example. It is not presented as the source of the algebraic property.

## Fully worked acquisition, composition, and transfer

This is a hand derivation, not a run. The learner is offered two length-three internal traces, z_1=(1,2,3), z_2=(2,3,5), without being told their recurrence coefficients. Their uncentered moment is

    C = [[5,8,13],[8,13,21],[13,21,34]].

The two traces are independent, both annihilate r=(1,1,-1), and rank C=2. Therefore ker C=span{r}. The learned relation is

    a+b-c=0.

The system can install a directed one-step operation c=a+b, but keeps the relation. It then proposes to share ports between three copies:

    x0+x1-x2=0,
    x1+x2-x3=0,
    x2+x3-x4=0.

Eliminating x2 and x3 yields

    2*x0+3*x1-x4=0.

If a later task fixes x0=1 and x4=8 and asks for x1, rank checking finds a unique solution and the compiler produces

    x1=(x4-2*x0)/3=2.

It can recover x2=3 and x3=5 through the stored composition. No examples labeled with x1 as an output were used to identify r. The same acquired behavior produced a different computational use. A more distant terminal yields coefficients through ordinary recurrence composition; this is not a claim of novel fast powering.

At a broader level, arbitrary chains of finite linear constraints can be joined and hidden using the same construction. This structural extension, rather than the numbers above, is the reason to consider the idea. But all of this linear extension is also available to the conventional alternative below. The example verifies neither a new learning principle nor useful nonlinear internal mechanisms.

## Strongest conventional alternative under the same information

Give a conventional learned linear state-space/constraint model the identical traces, port descriptors, task boundary masks, arithmetic budget, memory budget, and proposal opportunities. Let it fit the same relation and use sparse QR/Gaussian elimination, with compiled solver reuse. It reproduces every algebraic step above. Calling the candidate relational, local, neural, or self-organizing cannot distinguish them. A separate baseline with multiple independently trained directed networks is weaker and cannot be the decisive comparator.

A broader conventional alternative is a learned factor model or implicit network plus a generic constrained solver and representation learning. I have not read its closest full primary literature in this pass and make no source claim that it covers every proposed extension. Mechanistic overlap is already strong enough that a claimed advantage must specify what information, representation, credit, or amortization cost it changes. Universal representability is not an equal-resource equivalence, but the explicit linear construction here really is equal-information algorithmic coverage.

## Success counterfactual and its one fragile step

If the **strong** candidate succeeds, it would show that a neural learner can deliberately organize its own intermediate execution so that compact, behavior-preserving laws can be extracted and turned into new operators under unfamiliar boundaries. Past learning would then provide reusable computational laws, including inversion or completion uses not separately supervised. The potential change is in what learning leaves behind and how later computations are constructed, not in adding a larger cache or a fixed module library.

The missing step is **compact law-rich internal representation formation with honest total cost**. The present policy can search for it; no argument shows it will find it, that useful nonlinear computation admits it, or that doing so beats ordinary latent modeling and a solver. This is one load-bearing open step, not a result inferred from linear elimination. A fixed collection of nonlinear feature functions would merely move manual design into that collection unless acquisition of useful features is itself explained and charged.

Broader-than-example prediction: useful internal operators, if acquired this way, should survive changes in which of their ports are supplied, provided target uniqueness and conditioning hold. The gain should occur on held-out boundary patterns and compositions after all construction costs; equal performance only on familiar forward queries is insufficient. A phase where the nonlinear laws become dense or the solver dominates would reverse the positive interpretation.

## Specific attacks and counterexamples

1. **Observational ambiguity:** suppose true traces obey c=a+2b, but the available observations have a=b. Every observed triple also obeys c=3a and c=3b. The kernel learner cannot identify the true law from these observations. Changing the task to a!=b exposes failure. Novel boundary masks do not manufacture missing independent information. Deliberate interventions or naturally diverse traces must be paid for and supplied equally to the baseline.
2. **Nonlinear projection is not just more rows:** relation x*h=1 has manifest x!=0 after eliminating h. A compiler that keeps only polynomial equalities in x loses this exclusion and falsely admits x=0. Adding inequalities, disjunctions, or solver code changes the representation class and can grow its cost. The linear closure argument cannot be exported unchanged.
3. **Inverse conditioning:** c=epsilon*a admits a=c/epsilon when epsilon!=0, but uncertainty is amplified by 1/|epsilon|. Input-output symmetry at the level of a set of solutions does not imply symmetric learnability, robustness, or cost.
4. **Preserved examples are not preserved mechanisms:** if the neural substrate changes between acquisitions, old port values may no longer denote the same variables. The explicit scheme requires versioned port meanings or revalidation. Replay tolerance on a finite archive does not prove the old computation is retained off archive.
5. **Outer learning can cheat:** favoring short laws may produce unused zero channels, copy channels, or redundant coordinates. A low-rank law is valuable only when its elimination performs task-relevant work that survives the matched solver alternative. The current construction has no theorem preventing this degeneracy.

## Decision boundary

Do not promote the finite linear result as new. Keep this as a high-risk constructive reserve only if comparison with the existing V2 body reveals a genuinely different unresolved mechanism and a plausible way to acquire task-relevant law-rich representations. If its best plausible success is merely learned constraints plus a standard solver, close it as a known reconstruction rather than enlarge the apparatus.

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION

Learned trajectory laws can yield multiple boundary-specific executable operations through exact finite linear composition/elimination | RECONSTRUCTION, hand-derived; conventional alternative matches | This file's moment-kernel, left-nullspace proof, and worked recurrence; paired primary reading ledger | Whether neural computation can acquire compact useful laws beyond known linear modeling with a total-cost advantage | No experiments; strong linear novelty interpretation rejected by the explicit baseline; nonlinear equality-only elimination fails on x*h=1 | After v0 hash freeze, coordinator compares against the existing V2 body and decides whether the nonlinear representation-formation gap is materially different and worth retaining.
