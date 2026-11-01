# P resumed v2 — acquire programs for changing owned computations

AGENT PROPOSAL / HYPOTHESIS. Frozen before execution; no executable probe or neural training has been run for this version. This replaces the affine-response restriction in construction-v1a. It keeps P v1's forward/repair coupling but gives the acquired object and its training rule a sharper meaning: **a conditional, closed-loop edit program over a graph of persistent parameter owners**, rather than an output-to-input inverse, a cost table, a fixed vector, or a symbolic derivative supplied for each new operation.

## Judgment and success counterfactual

The substantial possibility is that a neural computer can acquire a growing language of ways to alter its own mechanisms. A repair program can coordinate several owners, probe its changing computation, call earlier repair programs, and construct/reroute a bounded part of its source graph. A later mechanism would inherit both executable constituents and programs for adapting them. This could change the search depth of later learning, including nonlinear finite changes that a local gradient cannot specify at the current state.

The contribution would have to be an acquired organization of subsequent learning, under measured acquisition cost. Neither finite-update noncommutativity nor the existence of a repair language suffices. A strong graph-aware solver or a learned optimizer supplied the same source, owner identities, history and persistent memory can simulate this system. No computability separation is proposed. A significant result would be systematic reuse of short repair programs on independently new mechanism graphs and demands where rediscovering the same coordinated change consumes substantially more total work. Whether such a class exists beyond standard optimizer/library learning is OPEN.

## 1. State and allowed transitions

The persistent computer state is

    S=(G,theta,O,L,A,E,psi).

G is a bounded executable graph of neural cells and calls. O maps each parameter read in G to exactly one persistent owner. theta stores each owner's value once. L stores forward routines with their source graphs and versions. A stores acquired repair programs. E stores actual executed repair episodes and counterexamples. psi parameterizes a neural program proposer and its continuous argument heads. Scratch activations are not persistent parameters.

An edit program p is bounded code in a designed typed language. Its inputs are (a graph-role binding rho, a desired behavioral displacement d, retention conditions C, a current execution context). Its instructions are READ_OWNER, WRITE_OWNER, CALL_FORWARD, CALL_REPAIR, CREATE_CELL, REDIRECT_READ, SNAPSHOT, RESTORE and bounded control flow. Arithmetic, automatic differentiation and standard numerical routines are permitted instructions, with their cost counted. Raw owner writes and a generic optimizer are always available to the control; the library does not conceal a direct easy action from the baseline.

rho binds graph roles to actual owner IDs and nodes. Repeated uses of a role remain aliases of the same owner. Each READ reads the current staged state, not a separate copy of the initial parameters. A CALL_REPAIR returns a staged edit transaction and a witness. Conflicting writes are composed sequentially or reconciled in one joint solve; independently adding owner-disjoint cost estimates is not assumed valid.

For a finite task/retention suite T, Execute(p,S,rho,d,C) returns a proposed S', a transcript and the measured physical work. The transaction commits only after actual forward replay satisfies C and the requested acceptance threshold. Otherwise it restores S and records the failure. This makes finite-suite retention checked; it is not a guarantee on all unobserved inputs or later learning. A graph rewrite with a separate exact semantic proof may obtain a stronger statement, explicitly labeled as such.

Programs and G are versioned together. A repair program is not silently carried over after changing the semantics of a primitive or the owner binding.

## 2. The actual acquisition law

There are two learning time scales. Ordinary current-task optimization may alter theta/G. Acquisition alters A and psi, changing the subsequent update action vocabulary.

**Obtain an episode.** At a common checkpoint S, solve one authorized repair demand using the current library plus bounded primitive search/optimization. Record the actual instruction trace tau, all queried forward values/gradients, owner bindings, failures, old/new task data and execution cost. A successful endpoint supplies a feasible example; no inverse label or optimality certificate is supplied. A trusted solver's answer is charged as teacher work, including failures and the computations it used.

**Propose a conditional program, not a stored endpoint.** The neural proposer emits a typed program skeleton p whose calls may reference existing A. An abstraction procedure replaces concrete owner IDs by graph roles, numerical update constants by typed arguments, and repeated trace fragments by calls. The skeleton must be executable at a new state; it may not read a table indexed by the training instance ID. Proposer supervision comes from successful traces, while continuous argument heads are trained by executing p at perturbed states and demands. A clone of an entire trace is allowed as an unproven candidate but is not counted as acquisition of a reusable mechanism.

**Train the procedure on its effects.** For a distribution of stored checkpoint/demand pairs (S_i,k_i), unroll at most H instructions and minimize

    J(psi,p) = mean_i [ L_new(Execute(p_psi,S_i,k_i))
                       + lambda_R L_retained
                       + lambda_W Work
                       + lambda_F 1[rollback] ]
                 + lambda_P DescriptionLength(p).

The simulator/interpreter supplies execution, feedback and owner provenance, not the correct finite update. Differentiable paths may use meta-gradients. Discrete program changes use bounded proposal/ranking from actual outcomes. Description length penalizes uninterpreted trace memorization but is not evidence of semantic generalization.

**Close interaction conflicts.** If p and q each work alone but their sequential use gives a failed composition, record both orders on cloned common checkpoints. A discrepancy identifies an interaction to model. Follow actual read/write provenance to its shared owners; train a correction program c(rho,d_p,d_q,state) on the residual failure, or refine a previous program's state inputs. A generic four-step loop p,q,p^{-1},q^{-1} may generate useful training data only when the inverses have themselves been learned/verified in the declared scope. An algebraic inverse is never presumed. The correction may call p,q and numerical correctors, but its endpoint is replayed. This is an acquisition trigger, not a claim that a Lie bracket is novel or that every conflict admits a compact correction.

**Promote on subsequent learning, not training fit.** Freeze the candidate program and use disjoint graph/state/demand episodes. Compare complete work to solve these demands with and without the program from identical checkpoints. Promotion to A requires actual held-out benefit after charging acquisition, storage, failed calls and verification, with an explicit horizon. The controller is then trained to select it. A program can remain useful to one distribution without being declared a universally valid repair. Failure examples update its precondition/gating or remove it.

**Construct new computations with these programs.** When G gains a new routine composed from library entries, the constructor can also build a repair program from the constituents' existing repair programs and shared-role bindings. It is initialized by code composition, then refined on the new routine's actual repair episodes. New repair programs can themselves call earlier repair programs. Thus acquisition can recursively shorten later *learning computations*, rather than only forward computations. The designed compiler and acquisition objective are fixed; the useful coordinated procedures, arguments, call hierarchy and binding choices are acquired.

## 3. Why shared ownership is essential

Suppose two constructed operations call the same neural cell. A repair selected for one operation can change the other's behavior through an aliased parameter even if their forward ports appear separate. A program containing two writes to that parameter does not acquire a second independent degree of freedom. Before commit, all callers affected by the owner's write set must enter the relevant retention/effect suite; unknown callers make the retained-behavior claim incomplete.

There are three legal responses to a conflict: find a coordinated nonlinear parameter change, change the computation's ownership by an explicit clone/reroute action, or report failure within budget. These are different hypotheses and their resource costs must not be conflated. Changing ownership can create genuinely new independent parameters, so it must be matched in the baseline's representation/growth budget.

## 4. Worked nonlinear case: an acquired finite repair at a dead first-order state

Consider a shared neural cell

    h_{u,v}(x)=v tanh(u^T x)

called by two constructed operations F_i(x)=f_i(x)+a_i h_{u,v}(x). u, v have one owner each; a_i belong to the callers. At u=0 and v=0, all first derivatives of h with respect to u and v vanish. First-order optimization confined to those parameters cannot start this path from the exact origin, irrespective of the output error. This is a direct calculation, not an observed training result.

A finite edit u=s w, v=s b gives

    delta F_i(x)=a_i s b tanh(s w^T x).

For small s the leading effect is a_i s^2 b w^T x; finite s produces a nonlinear feature. This coordinated two-owner effect is absent from the local first derivative but visible in actual paired perturbations. It is not an arbitrary two-bit truth table: the computation acts on a vector input and its finite change is nonlinear over an input distribution.

What the learner must acquire is a *conditional procedure* that selects w,b,s from the current residual/evidence, then checks/repairs every relevant caller. It is not given this formula as an inverse. Exploration or a teacher optimizer can supply successful paired perturbation traces. The acquired program may learn to propose a joint birth, fit caller coefficients, evaluate old traces, and either compensate, clone the shared cell, or rollback. Reusing one sampled (w,b,s) without conditioning on new demands is not success. New graph-role bindings must preserve the fact that u and v have a single shared owner.

If F_1 and F_2 have independent callers, a repair targeting only F_1 can still change F_2 by a_2 delta h. When a_2 is nonzero, a program trained only on the first caller is invalid. If old functional constraints prohibit that change, the learner needs a genuine compensator or a paid ownership split; there is no free retained-capability theorem.

**Strong actual attack.** This example is readily handled by negative-curvature/Hessian methods, nonzero initialization, random feature insertion, or graph-aware function-preserving network growth. The mixed second derivative at the origin exposes the same interaction. It is therefore only an acquisition semantics example and a negative control for weak first-order comparisons. It is not the scientific endpoint and cannot justify a significance claim. A test that supplies w or the right feature basis to the proposer would also remove the difficult acquisition problem.

## 5. Noncommuting edits: legitimate use and forbidden inference

For smooth acquired edit flows P_epsilon and Q_epsilon, executing a verified inverse loop can have an O(epsilon^2) endpoint residual given by a Lie bracket under the usual smoothness/order convention. The loop can expose an interaction hidden by a summary that stores only each action's first-order endpoint effect at one state. It does not by itself supply a portable finite correction or a cheap search direction.

If both fields preserve an old smooth constraint r at every state (Xr=Yr=0), then [X,Y]r=0. Brackets cannot escape a true retention constraint. If all physical coordinates are directly editable, a graph-aware solver may construct the same endpoint more cheaply without executing the loop. Consequently a Heisenberg-style demonstration is neither a learning advantage nor a contribution here. The potentially useful object is an acquired finite repair program that changes the cost of later search, with the full-state solver as control.

## 6. Strongest controls and exact novelty boundary

1. **Graph-aware joint optimization/control:** give constrained SQP/trust-region/continuation or a trajectory optimizer the entire G, shared O, the same gradients/execution oracle, permissible structural edits and all retained data. It can coordinate persistent updates and see curvature. Count all derivative/linear-solve/proposal work. AD being exact composition is not a weakness to withhold from it.
2. **Graph-aware learned optimizer:** a recurrent/hierarchical or graph neural update model gets the same owner graph, task history, outputs, gradients and a matched persistent-memory budget. It can already acquire state-dependent nonlinear multi-step updates. Compare a fixed update vocabulary to one that grows callable repair programs; a coordinatewise LSTM alone is not an adequate strongest control.
3. **Library-learning program synthesizer over the SAME update DSL:** it gets the same episodes, role bindings and ability to invent macros. This is the strongest direct organizational rival. It may reproduce the proposed mechanism. If the only difference is calling the learned programs neural, there is no separate principle.
4. **Supplied safe growth/untie transformations:** matched structural-growth budget and existing network-morphism transformations. They expose whether results come merely from using a known safe expansion rule. Learned selection/acquisition and the cost of deriving that rule are the only remaining differences.

Checked primary sources in sources/resumed-p/source-record-v2.md establish that composing parameter updates/requests with weight tying, learning nonlinear update rules, and learning libraries that improve later search are existing ideas. No novelty is claimed for these ingredients or their broad conjunction. The unverified proposal is the specific recursively acquired, alias-aware repair-program organization and a possible total-cost effect on later neural mechanism acquisition. A complete closest-work boundary is not yet established.

## 7. A consequential discriminating test, before any large implementation

The intended task family is growing neural mechanism graphs with tied cells, not a fixed low-dimensional synthetic objective. One concrete setting is a library of small bounded neural state-machine controllers that learn sequence transformations, share internal cells, then must acquire new held-out compositions while retaining specified earlier traces. The repair program sees the real graph/owners. New tests vary graph topology, amount of sharing, nonlinear cell state and required sequence behavior; role renamings alone do not count as new functions.

Before training, independently establish that the selected demands require changing an internal computation, that the old and new functions are not solved by appending an unconstrained readout, and that success changes a meaningful capability outside the training combinations. Data, direct edit options and structural growth are matched across controls. Outcome is correct acquisition under a fixed complete compute/evaluation budget, with retention and held-out compositions separate. No continuous open-endedness claim follows.

The decisive causal intervention freezes all forward mechanism state/data and proposer memory, then allows/withholds only execution of the acquired repair programs, with the equivalent expanded primitive work charged. A second control supplies the same stored programs to the strongest solver: if it matches, the effect belongs to the acquired program content, not a special representation. A third allows the solver to learn its own macros from the same history. If that matches under total costs, the present candidate has no demonstrated distinction from existing optimizer/program-library learning.

No larger prototype should begin merely because this test is programmable. First freeze a task family for which the best plausible success would change an important acquisition limitation, and identify a nontrivial difference from the macro-learning control. That remains the one fragile constructive step: **can nonlinear repairs admit compact programs that transfer over newly constructed owned graphs, rather than just amortizing a fixed family of numerical objectives?**

## Status ledger

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION
--- | --- | --- | --- | --- | ---
A concrete closed-loop acquisition rule changes both forward and update vocabularies | PROPOSAL, fully specified at algorithm level; not executed | Sections 1–2 | Discovery/generalization and fair total-cost behavior untested | P affine law insufficient for arbitrary nonlinear interactions | Select one consequential neural mechanism family against the direct macro-learning rival
Finite coordinated nonlinear changes may be invisible to first derivatives at a birth state | RECONSTRUCTION | Exact tanh-cell calculation in section 4 | Does not imply an advantage over curvature/growth methods | First-order-only baseline rejected | Use as negative control, not endpoint
Noncommutativity supplies a new learning principle | NOT ESTABLISHED | Section 5 attack | Known control/optimization effects; direct solver can match | Bracket-only candidate rejected | Retain only as an observed-conflict acquisition signal
The proposed system outperforms graph-aware learned optimization/synthesis | OPEN | No experiment | Potential organizational equivalence | No positive result claimed | Require an actual residual distinction before costly training

Provenance: AGENT PROPOSAL. Root independently mentioned a related path-transport/bracket direction; that agreement is not verification. Original P/K/R files unchanged. No experiment executed. Current effective provider mode remains unknown; no Pro claim.
