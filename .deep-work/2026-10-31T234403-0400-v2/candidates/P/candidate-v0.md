# P v0 — acquiring callable state transformations from neural execution

**Judgment: HYPOTHESIS / AGENT PROPOSAL.** A promising causal gap is the passage from *having performed a useful computation* to *having acquired an independently callable and improvable internal mechanism*. I propose to learn this passage as a factorization of neural state transitions under counterfactual changes of the surrounding state. The proposal is incomplete. Its formal algebra is substantially covered by causal abstraction; its possible research contribution is the acquisition process and ensuing change in learnability. No novelty or empirical success is claimed.

## Opportunity and connection to the fundamental question

A recurrent network can repeatedly solve a problem using distributed, context-dependent activity without acquiring a callable object that later computation can name, bind, and replace. Learning another solution in the same weights is therefore different from constructing a new computational operation. A central opportunity is to make a successful stretch of neural execution acquire an interface *because it survives changes in the rest of the neural state*, rather than because an engineer assigned a module or because a repeated symbolic subtree was already available.

This serves the Fundamental Question directly: construction is promotion of an executed transition family into the available operations; composition is calling acquired transformations in new sequences; improvement is replacing an implementation behind an acquired interface. It makes no claim about evolution or open-endedness. The anticipated gain is finite-budget accessibility and preservation of useful computation, not enlarged mathematical computability.

The important limitation is a proposed one, not an empirical universal claim about all neural networks. Standard networks, program learners and modular learners already achieve kinds of reuse. The issue to investigate is whether *learned intervention-stable interfaces* can make internally discovered computation more reliably reusable and improvable than their strongest alternatives.

## State and designed scaffold

Take a bounded recurrent neural computer with inspectable state s in a bounded subset S of R^d, neural controller parameters theta, primitive transition cells, external task feedback, and an execution limit. Its ordinary execution is

    s[t+1] = U_theta(s[t], input[t], action[t]).

The controller may call primitives or existing library entries. It records replayable internal traces. A candidate segment has an entry condition, a mode/argument token q, an executable stopping rule, and a cap H. Its total state update is T_q(s). A segment is a real bounded computation, not an output-only association. Initially, segments can be proposed by a learned start/stop selector, rewarded for reuse of the eventual factorization; a simple exhaustive span selector is a diagnostic scaffold, not the intended scalable solution.

The entire machine state relevant to resuming the segment must be included in s: scratch memory, the necessary controller state, and any continuation information. If that makes a small interface impossible, the proposed module fails. A recorded action sequence can define an initial segment proposal, but is not silently equated with the state-dependent policy that generated it.

The designed ingredients are the trace recorder, bounded interpreter/call stack, invertible-chart architecture, factorization objective, optimizer, replay/intervention scheduler, promotion rule and task evaluator. Gradients are allowed. No theorem requires the scaffold to arise autonomously.

## What is acquired

For a proposed family {T_q}, learn a coordinate chart

    E_phi : S <-> A x C : D_phi,

where a in A is an active interface and c in C is the context that should be preserved. E and D are inverses on the admitted domain. A practical parameterization could use an invertible neural map followed by a learned dimension gate; it need not assign mechanisms to contiguous original neurons. Invertibility is a scaffold, not an observed property of a model we trained.

Jointly learn a small executable recurrent function f_psi(a,q), including its local scratch state and capped stopping mechanism. The target relation is

    E_phi(T_q(D_phi(a,c))) = (f_psi(a,q), c).                 (1)

Equivalently, the full-state transition is D_phi o (f_q x id_C) o E_phi. The learned object is the tuple (entry domain, E, D, active dimensions, f, stop rule, bounds, version), not merely a vector embedding given a suggestive name. q is a learned invocation token and real numerical arguments if needed; neither its eventual interpretation nor the active coordinates need be supplied.

A provisional training objective samples states s_i,s_j, extracts (a_i,c_i)=E(s_i), forms mixed states D(a_i,c_j), actually runs the bounded source segment there, and minimizes

    L_factor = E[i,j,q] ||E(T_q(D(a_i,c_j))) - (f(a_i,q),c_j)||^2.

Include natural states, newly reached states and difficult valid mixed states. Factorization compares complete poststates; matching the original task output alone can conceal destructive side effects. A fixed source segment supplies counterfactual transition labels, so no supplied symbolic algorithm is necessary. This is self-supervision from actual execution, not proof that the teacher's behavior on mixed states is meaningful.

The chart and active dimension are optimized jointly with description and execution costs, including *the full cost of E and D*. A useful proposal must improve later task acquisition or computational cost after accounting for those costs. Compression on its own is not the scientific endpoint. Zero active dimensions cannot represent a nonidentity T; all-state activity always gives a trivial solution and must pay its real cost. A small active dimension does not by itself establish usefulness or semantic correctness.

Replay teacher execution is frozen during an extraction pass. On failure, keep the counterexample; revise or reject the entry domain or factorization. A learned validity classifier alone is insufficient to certify a universal domain. Promotion is provisional on the tested domain and evidence boundary.

Once promoted, an actual call becomes

    (a,c) = E(s)
    a_next = execute_bounded(f, a, q)
    s_next = D(a_next,c).

The controller now has an additional operation and learns when to invoke it. Newly successful traces can include such calls and are eligible for later extraction. Thus the available computational vocabulary changes through learning. We store old chart/interface versions so ordinary controller learning cannot silently reinterpret a retained call.

## A worked consequence, and exactly what it does not show

Consider a finite illustrative state s = H(a,c), with a in Z_3 and c in {0,1}, where H is an unknown scrambled encoding. Two executable segment modes happen to implement

    T_inc(H(a,c)) = H(a+1 mod 3,c),
    T_reset(H(a,c)) = H(0,c).

No label tells the learner which original states mean 0,1,2 or which coordinate is c. The transition table contains two disjoint isomorphic action systems. The reset operation selects an origin in each; successive inc transitions select coordinates 1 and 2. This gives an executable discovery route to E and f from transitions, without a provided high-level arithmetic model. Swapping arbitrary c while holding a fixed then leaves f unchanged. This is a mathematical reconstruction; it has not yet been executed as of v0.

After extraction, a controller can compute a new sequence of these calls in either context, including contexts absent from particular training sequences. For a word w=(q_1,...,q_n), exact relation (1) on a domain closed under the calls gives

    T_qn o ... o T_q1
      = D o ((f_qn o ... o f_q1) x id_C) o E.               (2)

Proof: substitute (1) into each factor and cancel adjacent E o D. This elementary composition fact is not a novel theorem. It isolates the load-bearing assumption: the common chart and *the complete reached domain* satisfy (1). It does not turn finite training agreement into such a guarantee.

If the implementation of f_inc is replaced by a cheaper computation with the same transition on every admitted a, every closed-domain caller retains the same full state behavior. The acquisition of a reusable algorithm is therefore separable from acquisition of a faster implementation. A capability-changing version must receive a new contract/version; it is not justified by same-function replacement.

With a uniform single-call approximation error epsilon and a physical-state Lipschitz bound L for each target transition, length-n composition error is bounded by epsilon sum_{k=0}^{n-1} L^k. This follows by a standard error recurrence. Expected training loss provides neither the uniform bound nor a useful L; expansion can destroy approximate compositional reuse.

The Z_3 construction is a sanity example, not the scientifically consequential task. If the mechanism worked for large distributed recurrent computations with unknown variable bindings, it could permit a network to turn its own hard-won algorithms into transportable, separately trainable operations. Examples worth the ambition include constructing a learned constraint-propagation routine and reusing it on a new relational domain, or acquiring a data-structure update and reusing it under new internal encodings. Those examples remain unimplemented prospects, not evidence.

## Improvement rather than just caching

An acquired operation should have several admissible implementations behind one state interface: its original neural trajectory, a shorter learned neural implementation, or a newly composed implementation using other acquired operations. Later optimization may alter local scratch dynamics and call structure while preserving admitted input/output state behavior. The stored original operation and counterexamples anchor that change.

The harder and more interesting possibility is representational: discovering E makes an expensive distributed operation a simple f, and the simple f becomes easier to modify, compose and search around than the original dynamics. This is the prospective principle: *discovering the coordinates in which computation is local can change the subsequent learning problem*. A module count or a compressed trace does not establish that effect. A meaningful success would need matched subsequent-learning comparisons, not just lower extraction loss.

## Strongest plausible simple rival

The strongest rival is a neural controller over an ordinary growing program library with explicit typed arguments and memory slots, optionally distilled from successful traces. It already supplies stable interfaces, exact semantics and independent replacement. Its manually provided type/memory discipline may be a powerful and entirely legitimate inductive bias. A second relevant rival is a fixed modular recurrent architecture trained on the same multi-task experience; learning weights and routing may suffice without post-hoc extraction.

If either rival acquires and composes the same capabilities at matched total data, replay, parameter, interpreter and execution cost, the proposed mechanism has not earned a new organizational principle. An advantage confined to hiding chart overhead or granting extra counterfactual labels is invalid. A serious favorable result would involve a reusable boundary not representable as an initially supplied slot/subtree, followed by better later acquisition under a fair budget. That counterfactual is why this is more than a proposed accelerator.

## Prior-work boundary

**SOURCE:** DreamCoder already learns hierarchical callable libraries and neural search; its technical extraction and representation are detailed in the source record. The novelty cannot be “learning reusable primitives.” **SOURCE:** IIT already optimizes counterfactual neural behavior toward a high-level causal model. **SOURCE:** the 2025 causal-abstraction theory already formalizes distributed mechanisms through invertible translations and interventions. Relation (1) should be treated as a reconstruction within that landscape, not a new theory of causal abstraction. **SOURCE:** RIMs already learns selectively interacting recurrent mechanisms.

The prospective difference is a loop that discovers a state-update interface and its computational content from actual neural execution, without a supplied high-level program, then makes that discovery change the system's executable operation set. Whether that difference is new, learnable and useful is OPEN. These readings are targeted, not a complete novelty survey.

## Strong attacks and hardest missing step

1. **Identifiability and vacuity.** An invertible re-encoding exists trivially with A=S. Cost pressure can prefer arbitrary small factors unrelated to useful computation. Many equivalent charts remain; factorization need not identify the intended semantics.
2. **False context freedom.** On natural trajectories, a and c can be correlated. A decoder can generate mixtures far outside the valid state manifold. Executing the teacher there gives a real result but does not make it a meaningful reusable computation. Conversely, rejecting every new mixture can protect a vacuous contract.
3. **History leakage.** A useful computation may not have a small Markov interface at any available segment boundary. Putting all continuation and memory into a may erase the hoped-for advantage.
4. **Chart cost and incompatible localities.** Each module can require a different global chart. Repeated decoding and encoding can cost more than the computation, and different useful operations need not admit any common low-dimensional factorization. Equation (2) uses a common chart; independently learned charts can still compose physically, but the cheap cancellation and local reasoning do not follow.
5. **Discoverability.** The objective specifies what a reusable operation would be, not why gradients should find it before task-specific fitting or all-state solutions. Chart learning, segment discovery and operator learning can mutually hide their failures.
6. **Retention illusion.** Freezing a tuple retains an implementation, but a changing outer representation can make its input domain unreachable or useless. Retained bits are not retained capability. Versions and replay prevent silent redefinition, not this scientific failure.

**Hardest missing step:** finding a small, task-relevant, counterfactually meaningful interface in a genuinely distributed neural computation *without supplying the desired abstraction*, and showing that its acquisition changes the difficulty of learning new computation. A hand-scrambled finite transition table is insufficient to establish this.

## Next discriminating action and stopping boundary

First reconstruct, with an inspected bounded standard-library probe, the finite example and an ambiguity attack: inc alone leaves independent origin gauges in the two contexts, while reset anchors the chart; correlated observations may hide a context-dependent transition. This can expose exactly what transition evidence the construction requires. It is a mechanism check, not a neural-learning result. Substantive numerical work would require vetted infrastructure from the coordinator and a new explicit test specification. Do not automatically launch it.

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION
--- | --- | --- | --- | --- | ---
A neural execution could acquire a callable update by learning an intervention-stable state factorization | HYPOTHESIS | This construction; targeted primary reading | Acquisition without supplied semantics and later-learning benefit | None executed; several explicit degeneracies above | Check finite extraction and its nonidentifiability, then decide whether the neural acquisition gap is significant enough to pursue

## Provenance and isolation

Initial inputs read: this run's inputs/seed.md and inputs/contract.md. No old campaign files, historical NOW, library taxonomy, other candidate files or migration reports were opened before v0. Public primary technical literature was read as listed in source-record.md. Filesystem isolation is instruction-only on a shared filesystem; same-model branches are not independent mathematical or empirical verification. Requested profile=pro; parent-reported host selection=gpt-6.1-sol, effort=ultra; provider-effective model/profile is unknown. These fields are not self-certified identity.
