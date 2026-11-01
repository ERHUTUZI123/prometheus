# R v0 — Acquire relations that construct the space of computation

**AGENT PROPOSAL; HYPOTHESIS.** A neural learner should be able to acquire relations among its own elementary actions, then use those relations to construct the memory geometry on which subsequent actions execute. The learned object is neither a task answer nor only another weight vector: it is a finite presentation of an executable internal action algebra, together with an on-demand representation of it. This is an unfinished construction, not a novelty claim or an executed neural result.

## Opportunity and success counterfactual

The division between “memory architecture” and “learned program” may be an unnecessarily frozen division. Today a designer normally chooses a stack, slots, a grid, attention memory, or a fixed recurrent vector; learning chooses how to use it. Here the same substrate would acquire operational laws, and the laws would determine whether its internal memory behaves like a stack, independent counters, reversible registers, or something outside that catalogue.

If this works beyond selected examples, a network could acquire a new reusable computational resource by learning a short set of operational relations instead of adjusting a fixed state vector until it imitates the resource over training lengths. The important prospect is **acquiring and preserving an algorithmic organization with costs that scale with the actual computation**, not increasing universal computability. A successful stack demonstration alone would not establish this prospect: it could be a disguised stack built into the scaffold.

Connection to the Fundamental Question: the acquired relations change the transitions that future internal computation uses; composition of acquired generators creates callable operations; refinement of relations changes memory and execution costs. No evolutionary or open-endedness claim is made.

## Explicit machine

Let A be a finite set of initially opaque action names. Composition is ordered as operator composition: uv means first v, then u. The fixed scaffold supports graphs, bounded term rewriting, exact pointer identity, allocation of fresh nodes, and reversible versioning. Those are substantial designed computational resources.

At learning time t, state is

`S_t = (theta_t, phi_t, R_t, G_t, z_t, H_t, B_t)`.

- `theta`: neural controller, choosing action names or compositions from observations and an embedding of the current internal state.
- `phi`: neural proposer for finite relations `u = v`, possible inverse/zero laws, and macro definitions `q := u`; words have a bounded proposal length K.
- `R`: versioned, provisional rewrite relations. Each records its supporting trials and the rewrites that used it.
- `G`: a sparse executable representation of the acquired relations. Nodes are encountered internal configurations; labelled edges implement generators. It is constructed lazily rather than allocated to a fixed dimensionality.
- `z`: current node or a finite sparse vector over active nodes. The discrete case suffices for v0; differentiable mixtures are optional and may be costly.
- `H`: replayable action histories and observations. Histories are retained before quotienting so false mergers can be undone.
- `B`: external allocation, action and rewrite budget. Exhaustion returns a failure; there is no autonomous resource expansion.

Execution is a specified transition, not a similarity score:

1. Controller emits `a` or a macro word.
2. Follow the corresponding existing edge when present.
3. Otherwise form the symbolic successor of the current normal form, apply R under a bounded normalizer, and intern its normal form. An unseen normal form allocates a node. A relation that identifies it with an old node reuses that node.
4. Emit observations through the task interface and retain provenance. A rewrite timeout or inconsistent relation does not silently become an answer.

Relations must act in all syntactic contexts. Merely identifying two histories with equal current outputs is insufficient: their successors may differ. A canonicalizer is not automatically available for an arbitrary finite presentation. In v0 the runtime either obtains a terminating, locally checked rewrite orientation, or retains an explicitly bounded nondeterministic equivalence search and reports its cost and unresolved pairs. No general word-problem solution is assumed.

Learning has two distinct signals. Task feedback trains action use; replay interventions train the relations that make action use reusable. Given replay states s and continuation probes c, a proposed relation u=v receives a violation witness when

`observe(c(u(s))) != observe(c(v(s)))`.

Equal observations on finite probes only support a provisional relation. They do not prove equality. The proposal objective is task loss plus actual execution/allocation/rewrite cost and a description cost for relations, with a penalty for demonstrated violations. One explicit discrete update is

`phi <- phi - eta * (L(R') - baseline) * grad_phi log q_phi(R' | H)`.

Controller parameters can use ordinary policy gradients or ordinary differentiation through a fixed chosen graph. Changing R is a separate structural action; no fictitious gradient through an unimplemented discrete edit is claimed. Counterexamples retract responsible relations and split affected states using retained histories. This part resembles active automata learning; the intended additional object is a reusable law that generates an unbounded family of configurations, rather than another finite state per distinguishing history.

## Worked construction: acquire a creation/deletion resource

Use two opaque creation-like actions C_a,C_b, two deletion-like actions D_a,D_b, an empty state e_empty, and a zero invalid state. Explanatory subscripts are not semantic labels supplied to the learner. Suppose the acquired laws are

`D_i C_j = delta_ij I`, for i,j in {a,b},

`D_i e_empty = 0`, and `C_i 0 = D_i 0 = 0`.

These laws have a concrete representation. Let e_w be an orthonormal basis vector for a finite word w over {a,b}. Define

`C_i e_w = e_(iw)`;

`D_i e_(jw) = delta_ij e_w`; `D_i e_empty = 0`.

This is a stack-like resource constructed by the acquired relations. An actual operation is

`e_empty --C_a--> e_a --C_b--> e_ba --D_b--> e_a --D_a--> e_empty`.

The algebra says that `D_b C_b C_a` is the same operation as `C_a` for every admissible starting state, so that cancellation can be reused in unseen compositions. Applying `D_a` to `e_ba` returns invalid zero rather than silently popping the wrong symbol. For a basis state, the top symbol can be observed through `||D_i z||^2`; a matching D then returns the tail. This read interface and the Euclidean norm are designed, not emergent.

The representation can grow only where execution visits it: a missing `C_i e_w` allocates e_iw and records the corresponding inverse edge, as demanded by the acquired relation. A finite run visits finitely many nodes and terminates on its external budget. This is not a claim to have physically instantiated an infinite Hilbert space.

**RECONSTRUCTION, elementary obstruction.** No nonzero finite dimensional linear space can carry even two everywhere-defined operators C_a,C_b satisfying all `D_i C_j = delta_ij I`. Each C_j is injective. The two ranges are linearly independent: if C_a x=C_b y, applying D_a gives x=0 and applying D_b gives y=0. Therefore their direct sum has dimension 2d inside a d-dimensional space, impossible for d>0. This is a restriction on the stipulated exact linear representation, not a no-go theorem for nonlinear networks, approximate finite-horizon implementations, or arbitrary-precision encodings.

The obstruction offers a constructive instruction: do not compromise the acquired law by squeezing another needed configuration into a fixed vector; allocate a new distinguished configuration. For a real neural learner, learning the correct law is the unresolved step. Giving the above equations to the runtime is not learning them.

The same generative runtime should produce a different resource from different learned laws: commuting increments should produce a lattice-like configuration space, idempotent writes should collapse repeated operations, and involutions should produce reversible toggles. These are prospective scope tests, not results. A uniform interpreter restricted to a prelisted collection of stack/counter/toggle templates would weaken the claim to architecture selection.

## Designed versus acquired

**Designed:** symbolic graph substrate, finite action vocabulary and proposal language, rewrite executor and budget rules, neural optimizer, replay/intervention access, observation interface, cost accounting, and all safety boundaries. Dynamic allocation itself is designed. The free-history substrate already supplies significant storage and execution power.

**Acquired if achieved:** which action identities and inverse relations hold; which histories can safely share state; which new relations support reuse across contexts; which macros are worth retaining; how to use the resulting resource for tasks. “Stack emerged” is false if a stack normalizer or the specific equations were supplied in code, examples, rewards, or privileged labels.

**Inherited/retained:** accepted relations, their versioned evidence and executable representation; controller weights. **Executed:** graph transitions and rewrites above. **Externally selected:** task/cost feedback. There is no evolutionary inheritance regime in this candidate.

## Strongest matched alternatives and ways this fails

1. **Active automata learning with neural perception.** Angluin's source construction already turns observational equivalence into states and uses distinguishing experiments to refine them. Merely reproducing that with a network is insufficient. A meaningful difference would be acquiring a finite generative law that supports novel state counts and composition, with equal feedback access and accounting for all query costs.
2. **A controller plus a designer-supplied stack and other memory types.** This will probably win on stack tasks. The meaningful target is an acquired law outside a held-out catalogue, or a demonstrated reduction in total acquisition cost across genuinely different organizations. Matching only parameter counts would be misleading; match executable primitives, accessible feedback, memory, and search/normalization work.
3. **Generic history storage plus a neural interpreter.** The proposed graph/rewriter must beat or explain something beyond this baseline. A syntax engine already contains some of the necessary computational capability. Calling the selected quotient “constructed computation” cannot by itself establish a new capability.
4. **Program-library learning, algebra induction and term rewriting.** These are plausible close coverage, not decorative related work. Public primary comparison is incomplete at this freeze. Novelty is OPEN.

The hardest missing step is obtaining a correct and useful relation from finite feedback without building its intended data structure into the proposer or compiler. Finite probes cannot identify unrestricted action semantics: an alternative system can agree for all tested depths and fail at the next depth. Therefore the strongest attainable acquisition claim must state a restricted hypothesis class or accept provisional, falsifiable relations. Exact algebra after a guessed law is not evidence that the guess was learned or true.

A second danger is that the generic compiler solves the entire interesting problem. A decisive demonstration must withhold the target law and architecture, test unseen compositions/depths, and compare a fixed catalogue under the same feedback and total compute. A third is intractable normalization; the law language must earn its cost rather than hide it.

## Present judgment and next single decision

**OPEN.** The explicit construction survives as a coherent route from acquired operational relations to an expandable executable memory. Its exact linear obstruction is elementary and can be independently checked. It is not yet evidence of neural acquisition, general improvement, novelty, or paradigm-level significance. The root scientific decision is whether learning the *law that creates a resource* is a sufficiently distinct mechanism gap after the closest algebra/automata/program-induction precedents are checked. Do not start a large training system before resolving that boundary.

## Independence and source boundary

Before this v0 freeze I read only this run's seed.md and contract.md among project files. I did not inspect project bibliography names, indexes, categories, historical NOW, prior campaigns, or other candidates. Public sources were selected independently. I first considered partial evaluation/fast weights, then moved to algebra-generated memory; those source readings do not establish this proposal's novelty. The PDF skill was read outside the project. Filesystem isolation is instruction-only, not an enforced blindness guarantee. No accidental restricted-project exposure is known. Requested/observed/provider-effective model fields are left to the coordinator; this worker does not certify model identity or provider mode.
