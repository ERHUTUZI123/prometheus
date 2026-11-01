# R v1 — Construct an operational theory instead of inferring a hidden one

**Judgment:** the R v0 construction is coherent, but its strongest scientific gap was misstated. Internal actions can be *defined* by acquired rules. Finite feedback selects useful internal semantics; it does not prove those rules about an unknown external world. This removes the false requirement to infer a universally true law from finite samples. It also exposes the closest strong rival: ordinary inductive program/abstract-data-type synthesis. The current work has not established a distinction from that rival sufficient for a flagship claim.

This file preserves v0 unchanged. It is a serious constructive pass, not an executed acquisition claim.

## 1. A transparent construction with no data-structure compiler oracle

Use a persistent, typed term DAG as the designed substrate. A node contains a symbol tag and a bounded list of child handles; atom nodes hold externally supplied payloads. Matching a finite pattern, allocating a node, following an edge and replacing a term persistently are designed primitive operations. The network is allowed to invent symbol tags and arities within external budgets. A constructor catalogue such as stack/queue/map is not supplied, but a term DAG is already a powerful general data representation.

An acquired mechanism is `(Sigma, R, interface)`: a finite typed signature, finite first-order rewrite rules over variables, and callable operation ports. An input event builds an interface term containing the current state handle and input payload. Execution repeatedly matches and reduces this term, allocating nodes and sharing old subterms. It returns the normal-form state/output term. No procedure converts a desired stack law into a stack program: **the rule is the program**.

A trusted checker accepts only a restricted, explicit class:

1. Rules are well typed; RHS variables occur on the LHS; unknown payload values cannot be smuggled into constants.
2. Each rule decreases under a specified substitution-stable well-founded ordering. A very conservative subclass erases at least one function-symbol node and never duplicates any variable; a larger allowed class can use the explicit Knuth–Bendix ordering checked from its source. No general termination oracle is assumed.
3. Enumerate the finitely many nonvariable rule overlaps by first-order unification. Normalize both one-step descendants of every critical overlap, recording explicit joining traces. Reject if a trace fails or exceeds the compilation budget. Termination plus the checked overlap condition provides canonical normal forms. Merely checking a few ground terms is insufficient.
4. Retain a runtime step/allocation cap even for accepted rules, because finite does not mean affordable. Rejection/timeout is a recorded candidate failure, not a reason to expand resources.

This checker is classical rewriting machinery, not a new scientific result. The currently executed string probe does **not** implement this first-order checker and is not evidence that this v1 compiler exists. A full implementation would have to handle variable overlaps, occurs checks, fresh-variable renaming, ordering and persistent DAG sharing correctly. Destructive rewrites of a shared node would change other contexts and invalidate the functional-term semantics.

An accepted relation is true for the resulting **internal algebra by construction**. This gives no guarantee that its task outputs are correct. The term algebra's quotient generally has many states; “minimal memory” does not follow. It is the least congruence imposed by the chosen rules, not the smallest state machine compatible with observations.

## 2. A worked resource, with the target laws withheld from learning

For explanation only, call external interface operations `p(v,s)`, `r(s)` and `d(s)`; they have opaque numeric tags in the proposal grammar. Suppose a learner selects the following rules from ordinary task feedback:

`r(p(v,s)) -> v`

`d(p(v,s)) -> s`

`r(e) -> blank`, `d(e) -> e`.

Here `p` can itself be the retained constructor. After payloads alpha and beta arrive, state is `p(beta,p(alpha,e))`. Calling r returns beta. Calling d returns `p(alpha,e)`, after which r returns alpha. Each rule is variable-general, so the same transition applies to new payloads and arbitrarily long finite terms within the runtime budget. Payloads are data, not training-set constants. This is an acquired stack law **only if these rules were selected by learning**; these equations cannot appear as seed rules, privileged proposal templates, synthetic labels, or hidden compiler cases in the acquisition test.

The same interface can support other semantics through other rules. A queue-like retrieval, for example, must recurse toward the old end and need not return v. Thus the external interface alone does not prescribe the newest-first law, although its arities and available operations strongly shape the problem. Testing only newest-first tasks would still give the designer a strong implicit inductive bias.

For this worked stack, the term DAG allocates one node per p call and d reuses the saved tail pointer. Root-pattern matching is constant in the depth, conditional on a bounded, indexed rule set. These are inspectable resource consequences of the rule, not a neural generalization theorem. A generic rewriter that scans the entire DAG need not realize those costs; its actual cost must be counted.

Construction, memory and execution are separated as follows: the network selects the finite rules and which rule module to call; the runtime supplies term matching and allocation; terms preserve payloads; accepted rules define future transitions. A memory organization is encoded in the acquired program instead of a designer-selected special memory module. This is a legitimate route toward the Fundamental Question, but it is close to program synthesis, rather than a demonstrated alternative to it.

## 3. Explicit acquisition law and its strongest simple baseline

Let D contain task episodes with inputs and observable outputs/rewards. Let q_phi(M|D) propose a bounded finite mechanism M, including its signature. The checker returns either a certified admissible rewrite system with a cost C_compile or a concrete rejection. Evaluate an accepted M by executing its interface terms on D and additional task queries, with the same budgets as every competitor.

More concretely, q_phi is an autoregressive network over a finite serialization grammar. An edit can add a symbol with an arity and input/output sorts, add a rule whose terms use declared symbols and indexed typed variables, or finish a new immutable module version. The external envelope bounds signature size, arity, rule size, edit count and compile steps. These bounds define a finite candidate set for each acquisition attempt; they are not scientific claims about the natural size of a mechanism. Payload atoms cannot be named as training-specific rule constants: they enter through typed variables or fixed, openly specified equality operations. The same grammar and this relabeling bias are available to every baseline. A candidate changes the finite rules rather than receiving generated target equations from the checker.

For each proposal, execute the parser/checker, retain rejection cost and cause, replay the specified task episodes, and score actual task/cost objective values. Selection retains the lowest measured objective among admissible candidates including the incumbent; a changed module is retained as a new version rather than mutating the old one. The score-function update trains q_phi on the measured objective, with a baseline estimated from past proposals. A separate controller pi_theta is optional: fixed task-interface events can call the learned ports directly. If a controller learns routing, preserving a module's semantics does not preserve old task performance when routing changes; that requires a separate constraint or comparison.

One objective is

`J(phi,theta) = E_M~q_phi [ L_task(M,pi_theta;D) + lambda*C_total + beta*description(M) ]`,

where `C_total` includes proposal, compilation/unification, failed candidates, replay queries, neural training, pattern matching, allocation, execution and retained data. A discrete structural update uses the score-function gradient of this actual measured objective. Neural parameters that supply numerical leaves can receive ordinary gradients only along actually defined differentiable operations. No gradient through an unimplemented discrete compiler is assumed.

This is executable as a learning algorithm in principle, but it is not enough to claim a new acquisition mechanism. The early comparator is **best-first enumerative/CEGIS synthesis over exactly the same admissible signatures, rule grammar, checker, history, task queries and reuse operations**. It receives the same previous tasks and may retain discovered subprograms. A fixed-memory catalogue with the same total resource budget is a second comparator. A neural proposer may only amortize search; counting its pretrained knowledge or training compute as free would invalidate that comparison.

The unresolved meaningful hypothesis is narrower: selecting reusable *state constructors and transition laws* together can alter the later search landscape in a way that ordinary program-library reuse or a generic history interpreter does not capture economically. There is currently no derivation establishing this distinction and no frozen task family that makes it measurable without privileging R. Calling the object a law, type, algebra or mechanism does not settle it.

## 4. Retention under growth: an exact constraint, and its limitation

**RECONSTRUCTION.** Let an old partial execution system be `(X,T_a)` and the enlarged one `(X',T'_a)`. To preserve every old defined computation through an injective embedding i, require

`i(T_a(x)) = T'_a(i(x))`

for every old state x and every old generator a where the old transition is defined. Induction on a finite action word gives preservation for every old defined composition. This states the necessary semantics-preserving interface; it is not a new theorem about learning.

Lazy allocation under unchanged rules satisfies this on already defined graph edges. One cheap syntactic discipline also preserves old modules: retain their rules and introduce new symbols whose rules have a new symbol at the root, so they cannot reduce old terms. New modules can call old modules. Ordinary rule indexing/versioning suffices; no global neural retraining is needed to retain their exact semantics.

Changing old rules is harder. Adding an explicitly derived shortcut preserves the old equality theory if a proof trace verifies it is an old consequence, but this is classical semantics-preserving optimization. Replacing an old representation requires a checked migration/intertwiner or explicitly limited empirical evidence. A false old merger cannot always be repaired by an injective state embedding: one old node cannot map to two newly distinguished states. Retained histories/versioned modules, not a claim of monotone semantic preservation, are required in that case.

Thus the candidate currently gives a clear route for retention and composition by keeping learned programs. It does not yet solve discovery of useful new laws or their safe replacement without growing a library indefinitely.

## 5. Why the finite-dimensional obstruction is useful but insufficient

The v0 exact linear obstruction is representation-specific. A stronger quantitative version makes the issue precise. For m operators C_j,D_i on a real or complex d-dimensional space, form a block matrix A with blocks `A_ij=D_i C_j`. It factors through dimension d, so rank(A)<=d although A is md by md. On the at-least `(m-1)d` dimensional kernel, `(A-I)v=-v`. Extending an orthonormal kernel basis gives

`sum_ij ||D_i C_j - delta_ij I||_F^2 >= (m-1)d`.

This applies for d>0 and finite m; it does not assume adjoints or bounded operator norms. It rules out driving the **full uniform block relation loss** to zero in a fixed finite linear space. Testing only visited states does not supply that loss. A sparse growing term representation escapes by allocating new distinguished states instead of asserting an everywhere-defined finite matrix solution.

The executed probe provides a complementary negative witness. A single truncated shift has mean full-basis inverse-relation error 1/d and zero error on all tested low-depth states, yet collapses at its boundary. Therefore small or zero sampled relation error is not a resource-acquisition certificate. The result does not establish that the rewrite approach learns better, and no neural acquisition was executed.

## 6. Current decision

The constructive repair is **internal semantics chosen by learning, directly executable as rules, with a classical finite admissibility checker**. This avoids an opaque compiler and an invalid inference from finite task data to a universal law. It also strips away much of the apparent novelty: the remaining system is a plausible form of neural-guided inductive ADT/program synthesis.

Do not spend a training campaign obtaining a favorable stack example. The next scientific decision is whether a distinct acquisition mechanism remains after direct comparison with same-grammar program/ADT synthesis and library learning. If not, close R as a re-expression of existing mechanisms. If yes, state the specific acquisition advantage and its causal source before implementation. Current status is **OPEN, sharpened construction; no substantiated original learning mechanism or novelty claim**.

## Audit boundary

No P/K candidates, project library, historical research report or NOW was opened during this pass, although cross-reading was later authorized. After v0 freeze I inspected the coordinator's run-local numerical runner; it contains the pathname of an older virtual environment but no old scientific result was read. This is an infrastructure path exposure after the freeze, not an enforced isolation guarantee. All code, results and source notes written by R are within its assigned directories. The only numerical execution was the frozen, stdlib-only supplied-law probe and its serialization smoke under a 30 s external sandbox/CPU/wall bound.
