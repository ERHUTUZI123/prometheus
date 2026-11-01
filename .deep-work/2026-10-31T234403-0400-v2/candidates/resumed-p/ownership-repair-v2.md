# Portability under shared owners: attack and a constructive repair law

RECONSTRUCTION / AGENT PROPOSAL. This is the independent portability subtask requested by root. It supplements candidate-v2, not the separate R macro construction. No experiment was executed. The mathematics below is a semantic check, not a claim of a new theorem or an established learning advantage.

## Judgment

A learned finite edit sequence is generally **not** portable merely because a newly constructed routine has the same forward interface, uses isomorphic cells, or returns to the old outputs at the end. Its parameter-owner binding is part of its semantics. A serious repair is to learn a conditional **program for re-specializing a macro on an owner-incidence graph**, with all affected callers present, instead of declaring the old open-loop sequence reusable under arbitrary aliases. This can remain constructive: learn reusable conflict-resolution procedures, not only additional scope restrictions. But it makes the direct rival graph-aware MPC/optimization with learned repair options; no advantage follows from the typing discipline itself.

## 1. Exact endpoint condition for transporting a role-space action

Let a macro use n formal parameter roles q in R^n. A binding rho maps these roles to m persistent physical owners theta in R^m. Every owner is used; P_rho is the n-by-m replication matrix with (P_rho theta)_r=theta_{rho(r)}. Its image D_rho is the subspace where all aliased role slots have equal values. P_rho is injective.

Given an endpoint action T:R^n->R^n, a physical endpoint action T_rho satisfying

    P_rho T_rho(theta) = T(P_rho theta)                     (1)

exists exactly when T(D_rho) is a subset of D_rho. Necessity follows because the left side lies in D_rho. Sufficiency: because P_rho is a bijection between R^m and D_rho, set T_rho=P_rho^{-1} T P_rho. This also gives uniqueness. State-dependent targets/context are included among T's arguments and held fixed in this statement; changing those arguments on rebinding is a different action.

This condition concerns transporting the *same role-space endpoint semantics*. An owner-aware optimizer is free to define a different action from the same high-level demand. Equation (1) does not forbid that and does not imply ordinary shared-parameter gradient descent is invalid: it sums role requests before making one owner update, thereby defining the shared action anew.

### Nonlinear failure and a finite alternative

For two distinct roles, the finite action

    T_s(a,b)=(exp(s)a, exp(-s)b)

preserves the response r(a,b)=ab exactly. If a and b become one physical owner, D={a=b}. For nonzero a and s, T_s(a,a) leaves D, so there is no physical action with the same endpoint semantics. Writing the owner twice, taking the last write, or averaging the two requested values does not repair this fact; those define other endpoint actions and generally fail to preserve a^2.

The finite action J(a,b)=(-a,-b) does preserve both product and D, hence descends to theta->-theta. On the tied constraint theta^2=1, this changes branches while preserving the old response. Local tangent infeasibility does not establish global incompatibility. Conversely, existence of this finite branch change does not establish that a learner will discover it cheaply. This example is a semantic countermodel, not the proposed scientific endpoint.

## 2. Endpoint descent is still insufficient for a finite program

Suppose a macro is a program with successive role states q_0,...,q_H and intermediate observations. To transplant that *same executed program*, every reachable prefix must lie in D_rho and every intermediate observation must match the physical execution. A final q_H in D_rho is insufficient.

For example, two source instructions a<-a+s followed by b<-b+s have an endpoint that maps the diagonal to itself. But the first source prefix leaves the diagonal. Under one physical owner, the first write changes both reads immediately; the second write can produce either one or two increments depending on whether it uses a staged absolute target or a current-state increment. If the program probes after the first write, its source observation is also unavailable. Compiling the whole endpoint into one owner write can be valid, but requires a new semantics-preserving compilation step and a new physical cost; it is not the old trace by substitution.

For R's proposed signed invertible **physical** flows, every prefix is already a valid physical transition on the original binding. This avoids the problem within that scope. Rebinding a flow to a new shared-owner graph still changes its denotation: generators may collapse, previously separate read/write sets may interact, and the old commutator effect need not survive. Actual bound-flow execution is a legitimate new experiment, not proof of portability.

## 3. A relation transports more honestly than one chosen witness

Let R_k(q,q') denote the set of feasible endpoint changes satisfying a high-level demand k, rather than only its previously chosen witness. Its binding-specialized relation is

    R_{k,rho}(theta,theta') := R_k(P_rho theta,P_rho theta'). (2)

This exactly restricts the relation to alias-consistent endpoints. Cost must then be recomputed on physical owners, counted once; the sum of formal-role write costs can be wrong. Restriction can remove a preferred witness while retaining another, such as J above. It can also make the relation empty. Neither feasibility nor cost is inherited without this restriction.

For **trace** relations, impose the same equalities at every prefix, not only endpoints. Pulling back an already existentially hidden intermediate state can keep a role-space path that was illegal at its hidden prefixes. Thus storing only endpoint success, or minimizing over role-space paths before imposing aliases, loses information needed for a lawful physical transplantation.

This is why a set-valued learned repair interface can be useful: it can expose alternative procedures with different ownership requirements. It is not sufficient to retain a single cheap endpoint and a scalar predicted cost.

## 4. Concrete acquisition/repair rule: quotient first, then learn a specializer

The proposed repair of P is an explicit amendment to candidate-v2's abstraction and conflict steps.

**Normalize before acting.** On a new routine, resolve all parameter roles to physical owners. Build the owner-incidence graph B: parameter owners, instructions/cells that read or write them, and affected forward callers. Normalize role references to these owners before executing any edit. This is a designed compiler operation; it supplies alias facts, not a successful edit.

**Close the affected set.** For the macro's possible write set W, compute every protected caller output whose static dependency cone intersects W. These outputs and their specified data/conditions form the repair contract. Traced influence on a few inputs is insufficient to certify absence of an effect through dormant branches. In a static neural graph, conservative source dependency is available; dynamically constructed/control-flow code needs a conservative bound or an explicitly finite-trace claim. If hidden external callers exist, scope the claim instead of calling retention passed.

**Try lawfully portable variants.** A macro library entry stores a small set of conditional programs/effect models, their owner/read/write requirements and actual versioned witnesses. Bind each on B and run a bounded validity/feasibility check. Use disjoint macro pieces unchanged only where their contracts and physical trace semantics still hold. A learned heuristic can rank variants; an unverified effect estimate cannot certify a new binding.

**Resolve only the connected conflict component.** When a binding merges the read/write dependencies of previously independent pieces, take the union of their written owners and affected callers. Within that component, execute a bounded joint repair search initialized by the old variants; permit alternatives, finite sign/branch changes, and ordinary curvature/constraint optimization. Outside the component, retain already validated pieces. If the component spans the whole graph, charge the global solve; no locality is claimed. If no variant succeeds, return failure rather than average incompatible writes.

**Acquire a conditional specializer from these episodes.** Each successful or failed joint solve records (B, current owner state, demanded effects, retained contracts, old variants, physical trace, cost). Train/synthesize a bounded repair program

    Specialize_psi(B,state,k,variants) -> staged physical edit program

using these records and actual effect loss. Its program language includes iteration over owners and affected callers, calls to existing repair programs, probes and bounded numerical correctors. Owner labels are arbitrary; messages aggregate over the complete set of aliased role incidences before emitting one owner write. The architecture/compiler enforces owner renaming equivariance and consistent storage; effect correctness comes from execution/proof, not equivariance.

**Retain the method, not just the patched instance.** Freeze a candidate specializer and evaluate it on held-out owner-incidence patterns, checkpoint states and demands. The new object is acquired only if its procedure reduces total later repair work, including joint-solve teacher work, failures, checks, code/memory and compilation, against the same solver receiving the same history. Training on a tied pair and testing the same pair with different numeric IDs is only renaming, not structural transfer. A useful held-out test changes which constituent repairs overlap and requires new cross-caller coordination.

This rule is fully executable in principle without supplying an inverse, an algebraic commutator effect or a success witness for the new composition. It may use AD and standard solvers. The acquired part is the conditional choice/composition/correction procedure that reuses earlier repair programs under newly encountered ownership structures. The graph quotient, action language, training objective and endpoint validator are designed.

## 5. Why this could matter, and why it may fail

Later learning in a growing neural computer is not just repeated optimization on a fixed vector: constructing a routine changes which existing parameter owners participate together. A compact repair specializer could make a repeated *type of ownership conflict* cheap to resolve across new mechanism graphs, retaining useful sharing instead of always cloning everything or rediscovering a global joint edit. This would transfer an algorithm for learning a new composition, with causal evidence in the actual persistent writes.

The important limitation is not removed by more labels. Alias patterns can grow combinatorially; new ties can merge all conflict components; arbitrary nonlinear effect relations can require global information. A graph-conditioned neural specializer can overfit familiar overlaps. Then source-aware MPC/SQP plus options may match or beat it after acquisition cost. The one constructive fragile step is whether a scientifically meaningful family of constructed neural mechanisms has compact reusable **coordination procedures over owner graphs**, not merely a finite menu of endpoints.

Exact graph-aware optimization already normalizes aliases and can synthesize the same repair; a learned graph optimizer can learn the same specializer. The hypothesized distinction is a growing hierarchy of acquired, callable repair computations versus a fixed policy/action vocabulary. A program synthesizer allowed the same growing update DSL is the direct rival and may be organizationally identical. In that event P contributes no separate principle merely by naming the procedure a repair interface.

## 6. Small decisive audit before any training

For any proposed R/P macro-transfer result, record:

1. The exact owner binding and all primitive physical transitions, including intermediate observations.
2. Whether portability means identical physical trace, identical endpoint, satisfying the same finite contract, or succeeding at a new task. These are distinct.
3. Every affected protected caller and whether its coverage is source-conservative or only tested.
4. The cost of discovering/rebinding/correcting the macro, and the strongest solver given the same source/history.
5. One genuinely new overlap pattern plus a case where the library correctly rejects or replaces an invalid transplant.

These are finite scientific obligations for the particular result, not a new campaign workflow. No duplicate code execution or training is requested by this artifact.

CLAIM | STATUS | EVIDENCE | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION
--- | --- | --- | --- | --- | ---
Old role-space finite macros do not automatically survive new aliases | RECONSTRUCTION, exact | Equations (1)–(2), nonlinear scaling countermodel, prefix countermodel | Scope is transporting the same semantics, not forbidding a redesigned owner-aware solver | Endpoint-only portability rejected | Include binding/trace semantics in the frozen R test
An alias-conditioned specializer is a concrete acquired repair law | AGENT PROPOSAL | Section 4 state/input/output/training law | Compact transfer across unseen overlap topology untested | Ad hoc averaging/last-write repair rejected | Choose one consequential mechanism family for this transfer question
The specializer is a new principle beyond learned optimization/options/library synthesis | OPEN | Direct rival explicitly has the same information and macro abstraction rights | Possible organizational identity | No novelty or advantage claimed | Require a nontrivial difference before major implementation
