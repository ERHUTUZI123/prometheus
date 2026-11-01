# D v0 — constructing editable representations

Frozen independent candidate, before cross-branch exposure and before the external nearest-prior search. AGENT PROPOSAL; this is a mechanism hypothesis, not a novelty claim or an empirical result.

## Scope and provenance

QUESTION: Can a learner construct a persistent representation together with the executable operations that make that representation useful, rather than only fit values or select supplied modules?

GIVEN: The campaign seed/contract; primary passages recorded in sources/D/reading.md. Input is raw finite symbolic observations and task input/output examples; a fixed, resource-bounded interpreter executes all learned code. No access to other branches or prior Prometheus architectures.

NON-GOALS: Open-ended evolution, autonomous modification of the host, a full cognitive architecture, and a claim that changing encoding creates a new computable function. No evolutionary interpretation is needed.

VERIFIER: Actual bounded synthesis/semantic checks on a small string domain, plus primary-source mechanism comparison. This can check generated view/writeback semantics and expose scaffold dependence. It cannot establish neural discovery, generality, or novelty.

BUDGET: One main construction, one nearest-prior/causal attack, one local deterministic probe with necessary diagnostic variants. All files stay under branches/D, sources/D, probes/D. No paid services or release.

STOP: Deliver the constructed mechanism, its strongest failure, executed check and limits; do not start a larger benchmark.

MODE: Inherited current route, explicitly authorized as Ultra by the campaign contract. The supplied system identifies the agent as GPT-6-based; no independent provider/model-ID or reasoning-route telemetry is exposed here. No Pro claim.

## Opportunity

Copycat changes which objects and relations are perceived, but the permanent concept topology is supplied and does not learn across runs (local PDF 226 / printed 216). SICP makes even the evaluator a modifiable program (PDF 307–308), and its amb evaluator changes the control meaning of evaluation through explicit continuations (PDF 360–367). Koza already evolves truth-table primitives as well as call structure (PDF 599–605 / printed 577–583). Therefore, neither changing concepts temporarily nor learning a primitive function is by itself a useful novelty claim.

The candidate gap is **learning an editable representation**: a new internal object must come with both a program that finds its instances and a program that implements changes to those instances in the previous representation. Recognition alone is insufficient. An inferred group-length variable, for example, becomes computationally consequential when changing it actually edits the underlying sequence while preserving everything its interface promises to preserve.

## Computational state

At learning step t the machine state is

    M_t = (K, L_t, theta_t, D_t, Q_t, H_t).

K is the fixed sandboxed interpreter over finite trees/lists/integers and explicit state; L is the persistent collection of executable views and task programs; theta is a neural proposal distribution over code edits; D contains raw task examples and subsequently obtained observations; Q is a metered candidate queue; H is the executed causal trace, including rejected code. The neural network is an amortized proposer, not a semantic oracle. The first probe will substitute deterministic enumeration for theta, and will say so.

A learned representation object is not a symbol with a label. It is

    R = (Get_R, Put_R, Valid_R, C_R, cost_R),

where Get maps source states to typed finite structured states, Put takes an old source and edited view and returns a new source, Valid describes the domain where the edit laws are promised, and C is a checked certificate or finite test record. The source may be a raw observation or a previously learned representation. Program source and the cost of checking it are retained.

The core edit laws, for the declared domains, are:

    Put(s, Get(s)) = s                     (no-op round trip)
    Get(Put(s, v)) = v                     (the requested edit takes effect)
    Put(Put(s, v1), v2) = Put(s, v2)        (overwriting does not drift)

These laws are HYPOTHESIS-level obligations until actually proved or checked. A finite test record is not a proof for all strings. Partial views must retain a complement in the old source; lossless views can decode directly. Edits that change the number or identity of view objects can violate the laws and must be rejected, normalized with an explicit changed interface, or trigger a revised view. Silent identity changes are forbidden.

## Constructing the available operations

For a newly synthesized view-state program u: V -> V, the machine generates—not retrieves by semantic name—the source operation

    Lift(R,u)(s) = Put_R(s, u(Get_R(s))).

The generated code is simply the AST `put(s,u(get(s)))`; its entire execution is charged. It is admitted only on the declared valid domain. The outer learner can now express further programs over new objects, fields and relations. Consequently, L changes the future proposal language, object addresses, and operator arguments together. It is not just a new embedding of a fixed action vector.

For nested views R:S->V and T:V->W, construction is explicit:

    Get_{T o R}(s) = Get_T(Get_R(s))
    Put_{T o R}(s,w) = Put_R(s, Put_T(Get_R(s),w)).

When the component laws hold on the actually reached domains, the composite has the corresponding laws. This is a mechanical route from learned recognition to executable composition. It does not ensure a useful view can be discovered.

## The learning/construction law (no hidden organizer)

The boot language contains scalar equality/order/arithmetic, list/tree constructors and selectors, conditional, bounded fold/while, lexical closure, raw read/write/insert/delete, and a metered call mechanism. It contains no `run`, `group`, `object`, `successor-group`, `sort`, or task-specific view constructor. The representational inductive bias is nevertheless substantial: finite programs, list/tree shape, equality, explicit loops and a preference for short reusable code.

1. Generate Get/Put/Valid code edits from the boot grammar and existing L. Generation has a maximum AST size and execution fuel. In a neural implementation q_theta(code | raw task examples, failure traces, L) supplies order; in the local check, enumerate.
2. Reject nonterminating/ill-typed candidates. Check the edit laws on a declared challenge set, including source states not used as task examples and adversarial edits. Store counterexamples, never silently remove them.
3. For each surviving representation, synthesize task programs over its state using the same metered code grammar. A task answer is accepted only after lowering and executing it on raw inputs.
4. Evaluate an explicit total account, not only the short task program:

       J = DL(Get,Put,Valid) + sum_j DL(p_j | L,R)
           + lambda * raw execution steps
           + mu * candidate generation/execution/checking steps.

   Wrong task outputs are inadmissible. All exploratory costs are charged even to the winning representation. For a long-running learner, representation acceptance must be on held-out task savings sufficient to repay this construction debt. A perfect but unused view has no demonstrated value.
5. Add a genuinely new executable view only if it improves the predeclared total account within budget. If neural proposal learning is used, update theta by code-level likelihood on accepted programs, or a score-function gradient using *held-out* metered benefit and a baseline. This is ordinary learning of a proposal distribution; its success is OPEN.
6. A later counterexample can split a view's validity domain or refine Get/Put. The old code/certificate remains versioned. Existing callers continue using their declared version or are explicitly rechecked. There is no magical automatic semantic migration.

The only task information comes from supplied raw I/O and allowed subsequent raw executions. The checker knows the interpreter and stated algebraic edit laws, not which semantic organization is correct. Designing the domain/examples and selecting an AST grammar still supplies information and must be reported.

## Worked construction, without a semantic group primitive

Assume the generic code search eventually constructs this pair:

    Get(s):
      out := []
      for c in s:
        if out is empty or c != out[-1][0]: append(out, [c,1])
        else: out[-1][1] := out[-1][1] + 1
      return out

    Put(s,v):
      out := []
      for [c,n] in v:
        while n > 0: append(out,c); n := n-1
      return out

The newly discovered view type is a list of `(symbol, positive integer)` records, with unequal adjacent symbols. The reader can recognize run-length encoding; the machine need never receive that name. Positive count edits preserve this canonical domain; arbitrary symbol rewrites can merge neighbors and are outside this interface.

Raw training I/O might include multiple tasks whose outputs alter whole equal-symbol stretches, e.g. `abbc -> aabbbcc` and `deef -> ddeeeff` for one task, and `aabccc -> aaaabbbbbb` for another (double each count). These examples are hand-designed scaffolding; they do not prove a natural task distribution. The last example is a schematic I/O contract and must be checked carefully in the probe rather than assumed correct.

The constructed Get executes on `mrrjjj` as follows:

    m        -> [(m,1)]
    mr       -> [(m,1),(r,1)]
    mrr      -> [(m,1),(r,2)]
    mrrj     -> [(m,1),(r,2),(j,1)]
    mrrjj    -> [(m,1),(r,2),(j,2)]
    mrrjjj   -> [(m,1),(r,2),(j,3)].

A separately synthesized arithmetic/list program u changes the integer in the last record from 3 to 4. Put then emits `mrrjjjj`. The new action is not one of six raw token insertions indexed in advance; it accepts a constructed object's address and a constructed field. Its raw implementation, however, is entirely explicit and costs a parse plus decode plus output writes. There is no computational free lunch from calling that whole operation one step.

This is not a claim to have induced Copycat's analogy from `abc -> abd`. The example only reconstructs a concrete change of executable representation. The source's length insight must not be smuggled into the task supervision or counted as discovered.

Persistence is code persistence across tasks. Composition is lens/program composition with domain rechecks. Improvement is replacing a costly view/program by one with lower measured total account while preserving declared behavior. Inheritance and population selection are absent.

## Protected high-upside claim and fragile step

HYPOTHESIS: On a suitable nontrivial distribution, jointly learning recognizer, edit semantics and task code can make later useful mechanisms accessible at lower *total* discovery cost than learning flat programs with the same primitive interpreter and neural guide. The improvement need not be a new computable function; it must be a reproducible change in finite-budget construction, with current capability matched.

The strongest plausible advantage is that an inferred representation supplies **new causal intervention coordinates** and compositional validity conditions, not only stored call patterns. A neural system could build and revise what counts as one editable internal object. This directly serves the fundamental question about constructing internal computation.

The hardest unresolved step is joint discovery under honest cost. Naively enumerating Get, Put and task programs multiplies the search space. The construction may simply move human intelligence into the boot grammar, validation set or examples. The protected phase therefore ends with a concrete mechanism, not a declaration that it will outperform ordinary neural program synthesis.

## Planned adversarial check, fixed before results

Compare a small learned editable view against (a) no view; (b) its Get alone with arbitrary ungrounded field edits; (c) a fixed hand-supplied correct view; (d) wrong grouping/count field interfaces. Use raw correctness and an explicit operation count. A coherent view must survive novel symbol identities and run lengths, and its valid domain must reject identity-merging edits. If the effect is solely a supplied run/count schema, call that scaffold dependence, not representational discovery. A failed total-cost comparison closes only this small route, not the high-upside hypothesis.
