# Active interaction-law learning — v0 constructive pass

AGENT PROPOSAL / HYPOTHESIS. Same V2 continuation after review014. Effective provider mode unknown; file isolation instruction_only. This worker owns only this file and `sources/after014-higher-order/reading.md`. No experiment or training has been run.

## The computational limitation chosen

Can learning acquire the *local execution law* by which representations become active higher-order computations, rather than learn a program or module within an already supplied evaluator? The consequential possibility is that binding, duplication, recursion and the resulting causal graph become outcomes of learning. Universal simulation is irrelevant to this question; the issue is acquisition and resource organization under matched access.

The initial finite-table version is already too close to the sealed R branch's learned relations plus neural-guided synthesis. The stronger construction below makes agent codes generative and continuous, so neither a finite list of operations nor their interaction table is enumerated in advance. That change alone does not establish a useful difference from graph-program synthesis.

## Explicit provisional machine

An agent has one principal port, arity `r in {0,1,2}`, a principal polarity `s in {+,-}`, and a code `z in [-1,1]^d`. A finite port graph is a perfect pairing of its ports and boundary ports, together with a finite multiset of closed wire components with no ports. The latter must be retained because gluing can create a closed wire; silently dropping it would change the graph state and the resource account. An *active pair* joins opposite-polarity principal ports. The polarity orders the pair (+ first), so the neural emitter does not obtain node IDs or an accidental left/right ordering. Same-polarity principal pairs are stuck unless another interaction changes their connections. No universality claim is made for this restricted variant. The scaffold provides exact pointer wiring, bounded allocation, and a local rule emitter. It contains no primitive `apply`, `lambda`, `substitute` or `copy-function`. Arity, polarity, graph wiring and allocation are substantial designed resources.

For two descriptors `(r,+,z),(s,-,w)`, a neural kernel emits a bounded open replacement graph with the `r+s` old auxiliary wires as its boundary:

`K_theta(r,z;s,w) = (k, (r_i,s_i,z'_i)_(i=1..k), M), 0 <= k <= 4`.

Here M is a perfect matching of the `r+s+sum_i(r_i+1)` ports. Impossible odd-port descriptors are masked. Arity/polarity descriptors and M are categorical emissions; new codes are bounded neural outputs. The categorical space is finite for each local emission, but its entries are unlabeled port graphs, not named task operations or a lambda grammar. Deterministic execution uses fixed argmax and a fixed tie break. The input ordering comes from principal polarity. The early checkpoint's all-trivalent convention was unnecessarily restrictive: it required even k and could not directly represent the ground constructors used below. This is an explicit model correction.

State is `S=(G,theta,B,H)`, where B separately bounds transitions, cumulative fresh allocation and peak live agents, and H records exact emitted edits and terminal feedback. A transition removes the two active agents, reconnects their outside wires to the emitted graph, decrements the appropriate budgets, and records the edit. An externally maintained active-pair queue costs O(1) pointer operations per modified bounded-size neighborhood; the neural emitter's actual arithmetic is additional. No global graph search is hidden inside a local step. Exhaustion returns failure. Graphs remain finite; no infinite material or uncharged parallel hardware is assumed. Code precision b, d, parameter precision, neural width and allocation limits must be frozen before a run. Infinite real-valued code capacity is not a resource claim.

For training, sample discrete emissions from the categorical kernel, use terminal task loss plus measured work/space, and update

`theta <- theta - eta * [(L-b) sum_events grad_theta log p_theta(k,M | z,w) + pathwise_terms]`.

Pathwise terms apply only to differentiable dependence of subsequent code values on earlier code values within the realized finite trace, including a declared differentiable ground-value decoder. Discrete arity/polarity/graph decisions use likelihood-ratio terms; no derivative through a graph edit is invented. This is an ordinary stochastic-computation-graph estimator, not a new credit algorithm. Training samples and frozen argmax execution differ; successful training does not certify the argmax machine. Its actual performance must be checked separately. Exact trace probabilities, encoder work, all failed executions and training computation enter the account. The source of semantic truth is external input-output feedback, not self-consistency or confluence. The proposed system learns externally between finite episodes; it is unrelated to the PARKed online-organization line.

## Early derived invariant

For frozen theta and deterministic pair-local emissions, two distinct active pairs cannot share an agent. In the **uncensored semantics**, replacing either first produces the same graph as replacing the other first, up to fresh-agent renaming. Their boundary wires may meet, so the argument uses port-graph gluing, not spatial separation. Fresh IDs do not enter K. Codes on the two pairs are unchanged by the other's rewrite, their emitted open graphs are identical, and attaching both open graphs to the same remaining wires is associative. The standard interaction-net local diamond therefore survives generated codes and RHS graphs. This is a reconstruction of the structural argument, not a new theorem. It says nothing about termination or task correctness.

The bounded machine needs a separate qualification. Both rewrite orders must have enough remaining budget to complete both events. A peak-live-agent cap can permit shrink-then-grow but reject grow-then-shrink, so uncensored confluence is **not** schedule-independent bounded success. A canonical recorded scheduler gives an executable baseline; another schedule is a separately charged intervention. No theorem currently removes this resource dependence. Root explicitly caught this missing condition in the early checkpoint.

The precise architectural promise is local semantic compatibility under uncensored independent reductions. Whether it improves acquisition of useful computational organizations beyond existing local rewriting is OPEN.

## Fully worked higher-order process

This is a hand reconstruction using explanatory names for three code values. It is **not an executed learning result**. Let L have principal + and two auxiliary ports `(body,variable)`; A have principal - and auxiliaries `(argument,result)`. An L/A encounter has four boundary wires `(b,v,a,r)`. One possible acquired kernel output is the empty replacement with pairing

`beta: (b,r), (v,a)`.

The input graph represents an affine lambda expression: each bound variable occurs once, so no duplication or erasure rule is required. Binding is a physical wire from L.variable to the occurrence site. Application connects A.principal to its function's root, A.argument to its argument's root, and A.result to the surrounding context. A lambda connects L.body to its body root and exposes L.principal as its value. **The encoder already supplies binding/scope and exact alias identity in this worked case. Only the local execution law is varied.** Calling this complete acquisition of binding would be false.

Take `((lambda f. lambda x. f x) (lambda y.y)) a`. The active nodes initially are outer A_1 paired with L_f; outer A_2's principal is connected to A_1.result. The inner application A_fx has principal connected to L_f.variable, argument connected to L_x.variable, and result connected to L_x.body. L_f.body connects L_x.principal. A_1.argument connects the identity L_y.principal; L_y.body and L_y.variable are the same wire.

1. Rewrite `(L_f,A_1)`. Connecting b-r joins L_x.principal to A_2.principal. Connecting v-a joins A_fx.principal to L_y.principal. Two active pairs now exist: `(L_x,A_2)` and `(L_y,A_fx)`.
2. Rewrite `(L_x,A_2)`: its variable/argument pairing connects A_fx.argument to the ground value a; its body/result pairing connects A_fx.result to the external output.
3. Rewrite `(L_y,A_fx)`: the identity's internal body-variable wire makes A_fx.argument and A_fx.result one wire. The external output is therefore connected to a.

Steps 2 and 3 can be interchanged. Both delete two nodes and allocate none, so sufficient event budget avoids the peak-space caveat here. The process uses three interaction events and dependency depth two. It does not invoke a `call` procedure or invoke a host substitution traversal: active binding wires *become* the next execution sites. This is a real execution-organization distinction from a module-call interface, but it is established graph-reduction organization, not our invention. All initial parse/binding work is additional.

## A worked formation discrimination, including a misleading success

If the emitter for this one pair is temporarily restricted to allocating zero agents, four boundary ports have exactly three perfect matchings:

`M0={(b,r),(v,a)}`;

`M1={(b,v),(a,r)}`;

`M2={(b,a),(v,r)}`.

For identity-only examples `(lambda x.x) a -> a`, the input already identifies b with v. All three matchings connect argument to result at the observed boundary. M1 additionally leaves a disconnected cyclic wire. With a boundary-value observer that ignores unreachable garbage, every rule passes. A garbage penalty can reject M1, but M0 and M2 still agree. Therefore impressive identity generalization to unseen atoms would supply no evidence that beta was acquired.

Use instead `(lambda x.S(x)) a -> S(a)`, where S is an opaque arity-one ground constructor whose principal port is its read root and whose auxiliary port is its child. The input connects b to S.principal and v to S.child. M0 produces the target S(a). M1 makes a the output and leaves S in a detached self-loop. M2 exposes S.child at the output while connecting S.principal to a.principal; a read requiring a constructor principal root rejects it. Thus one nonidentity boundary test distinguishes these three zero-node matchings. This uses a designed terminal constructor/read interface. It does not establish that the unconstrained emitter will choose the zero-node class or learn all higher-order rules.

An ordinary finite synthesizer given the same three candidates, ground-value readout and one query can identify the desired pairing in at most three candidate executions. The neural learner has no identified advantage here. This finite matching demonstration must not be run or promoted as a new computational-learning mechanism.

The source of truth also needs care: once M0 is chosen it **defines internal execution**. Finite task examples are evidence of usefulness, not proof of equality to an unknown external evaluator. This correction already appeared in sealed R v1 and must not be rediscovered as a new contribution.

## Serious repair: acquire the representation as well as the local law

The pre-encoded example leaves the hardest representational work supplied. A stronger machine adds an encoder `E_phi` that constructs the initial port graph from task inputs, using the same descriptor/allocation/matching API. There is no lambda parser or binder-wire constructor. E_phi can be an autoregressive neural graph emitter over input tokens/observations, with STOP and an external node cap. Its discrete emissions use the same score-function objective; it receives terminal task outputs, not intermediate binding graphs or correct rewrite traces. Costs include all encoding, dynamic graph execution and learning.

For each task episode, execute

`x --E_phi--> G_0 --local K_theta reductions--> G_T --ground decoder--> yhat`.

An explicit formation objective is

`J(phi,theta)=E[loss(yhat,y) + lambda*W + mu*N_peak + nu*DL(phi,theta)]`.

W includes encoder emissions, each local neural evaluation, pointer rewiring and decoding. A malformed graph, budget exhaustion or unreadable terminal output receives a stated finite failure loss. Local graph topology may grow; **no module catalogue, finite operation signature or target interaction table is stored**. The learned code generator can emit an agent code not present initially, whose next interactions are determined by the same learned kernel. New codes are not automatically new functions: at b-bit precision they are finite machine states, and their meaning may merely select old rule cases. The kernel defines their entire prior execution family. This removes a fixed discrete vocabulary but does not create a new physical primitive or prove acquisition of a new operation.

This repair is substantive: input representation, active control topology and operational semantics are jointly learned. It is also substantially harder. No loss term or construction presently explains why E_phi/K_theta should discover a coherent execution organization from terminal feedback instead of a task-specific shortcut, a code-indexed rule table, or an opaque graph RNN. Specifying two neural networks is not a solution to that gap.

## Closest original mechanisms and the strongest same-information rival

The unfamiliar theory actually entered was Lafont's **Interaction Combinators**, especially the original local graph model and the code/copier/decoder construction. Its crucial warning is that a naive duplicator does not duplicate arbitrary executable packages. The affine example above avoids that obstruction; extending it to arbitrary reusable functions cannot simply add an unexamined `COPY` rule. A learned mechanism would have to acquire a correct representation/copy protocol or pay the supplied protocol as scaffold. The primary reading boundary is in `sources/after014-higher-order/reading.md`.

The original Neural Production Systems construction already learns sparse pair-conditioned rules and rule/entity selection by end-to-end training. In the inspected mechanism it updates entity slots; it does not implement the local port surgery specified here. This is a concrete difference from that source, not a novelty certificate. DreamCoder already couples neural search with learned reusable programs and abstractions. Merely allowing this learner to retain graph fragments would repeat its broad learning story and sealed R/D.

The strongest rival is **induction of the same local port-graph evaluator**, not a DreamCoder forced to receive a ready-made lambda interpreter. Give it identical opaque codes, d, b, graph emission vocabulary, observations, ground decoder, allocator, task queries, previous tasks and all total cost terms. It may synthesize or neurally parameterize the pair-local emitter, invent graph encodings, and reuse discovered rules. The zero-node case above gives an exact tiny equality of the learning problem. The full code-generating case is not reduced to a three-choice search.

One concrete execution translation compiles K_theta's finite neural arithmetic to a straight-line numeric graph program and wraps its categorical decision plus bounded allocation/rewiring in the same local evaluator API. With matrix arithmetic supplied equally, it performs the same emitter arithmetic, allocations and rewires per event; it need not simulate an exponential table over codes. A synthesizer can search this representation, and a neural kernel learner can optimize the same arithmetic parameters. This construction rebuts an execution-capability separation. It does **not** prove equal acquisition cost or claim that released DreamCoder implements this learner. Generic universal simulation is not being used as a veto.

The remaining candidate question is consequently about **a specific acquisition bias and its costs**: does forcing representations to acquire pure local interaction semantics make systematic higher-order organization accessible with fewer total examples/search/training work than the matched evaluator-induction rival? The present kernel objective supplies no distinct explanation yet. A future comparison must freeze an actual rival rather than exclude graph rewriting from its grammar.

## Success counterfactual and present scientific decision

If the repaired system learned computational encodings and local execution laws that transfer across materially different task programs, it could change what architecture learning acquires: reusable execution/binding/duplication organization itself, rather than fitted modules or selected call sequences. Local uncensored compatibility is structural and can apply at unseen graph sizes. The plausible broader relevance is learning an organization in which data structures, active operations and causal concurrency are one mutable representation. This is a reasoned prospect, not evidence of a new principle.

The fragile step is not whether neural weights can express beta or whether local rewriting can compute. It is the missing **formation principle** that preferentially discovers useful, transferable execution organization under honest total costs and distinguishes the result from ordinary induction of a graph evaluator. The construction currently names that learning problem and gives an executable search space; it does not solve it. Arbitrary terminal performance or a newly generated code would not fill the gap.

Judgment: **OPEN construction, not promoted to MAIN and not admitted to experiment.** Do not run the three-matching toy: its strongest success would be established local-rule identification. Do not call continuous codes a novel mechanism by themselves. Equally, do not close the broad repaired idea just because a program can implement it; a substantive acquisition law could still change the comparison. The next single scientific decision is whether a concrete formation law with a checkable causal advantage over same-interface evaluator induction can be constructed. If none is available, archive this as a sharpened boundary and redirect invention.

## Evidence/provenance certificate

`CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

- Pair-local generated rewrites have the uncensored local diamond. | RECONSTRUCTION, manually derived | gluing argument above; original structural precedent checked | no independent formal/empirical verification; budgeted success differs | early bounded-success overreach corrected after root review | retain as scaffold, not novelty.
- The printed affine higher-order graph executes to a via three reconnections. | RECONSTRUCTION, manually worked | full wiring trace above | no machine execution; scope encoded in input | no acquisition claim | no toy implementation needed.
- Identity examples identify the beta matching. | REFUTED under the stated boundary observer | all three matchings worked above | observer/scaffold explicitly stated | identity-only acquisition argument | preserve counterexample.
- Joint encoder/kernel learning can acquire useful higher-order organization. | HYPOTHESIS, unexecuted | explicit state, emission law and objective | formation advantage and arbitrary function duplication remain open | finite-table version overlaps sealed R; code novelty unsupported | construct a distinct formation principle before code.

Relevant original old bodies were actually read before any experimental commitment: old A's neural graph/local-construction proposal; AE-v2's finite installer; old D's `branches/D/v0.md` and primary comparison; current sealed R v0/v1/v2 and disposition. D/R already cover learned internal transition semantics via a grammar and neural proposal. This pass claims neither those ideas nor primitive-meaning evolution as new. Root separately reports original Koza primitive-semantics reading; that report is not this worker's independent SOURCE reading. No old artifact was modified. No experiment, install, paid/cloud execution, permission change or publication occurred.

## Root-seeded repair: stochastic causal organization rather than a deterministic rule table

PROVENANCE: root proposed causal-ancestry-keyed stochastic emissions after reading the first concrete v0. The following statement and proof are this worker's reconstruction of that proposal. It preserves the earlier finite-rule failure; it does not retroactively make that toy novel. This is episodic generative execution, not the PARKed online-organization branch.

### An exact finite-precision version

For this statement make all emitted codes b-bit finite values. A neural autoregressive distribution emits the whole local replacement R, including descriptors, wiring and code bits:

`R_e ~ p_theta(. | r,z;s,w)`.

It uses a bounded number of local categorical decisions, each with an explicit normalized probability. There is no enumerated truth table over the code space. Shared neural weights factor this potentially huge discrete conditional distribution. Freezing d and b exposes rather than hides finite representation capacity. There is no pathwise differentiation through code quantization in this version: the likelihood includes every sampled code bit. The outside graph, allocation address, wall time, scheduler, remaining budget and trace length are not inputs to p_theta. Parameters remain frozen throughout an episode.

Give initial agents distinct symbolic IDs fixed by the initial encoding. An event consuming agents i+ and i- has key

`e=(id(i+),id(i-))`.

An emitted child j receives ID `(e,j)`, with j its local emission position. Each pair can fire at most once because both agents are consumed. IDs are symbolic ancestry terms, not allocator addresses. Note that an event can be enabled by rewiring two previously existing agents: its causal predecessors include those wire-producing events, even when they are absent from its ID's agent-creation ancestry. The key is a unique schedule-stable identity, not a complete causal-order encoding.

Let `U_(e,l)` be independent uniforms for each event key and local draw index l. The emitter samples its local categorical decisions from this field by inverse cumulative probabilities in a fixed local enumeration. No scheduler may inspect draws of an event before selecting it. Conditional on a fixed random field, the machine is deterministic. Generated codes/rules/graph sizes are still random; only their causes, rather than their serial allocation positions, choose the draws.

### Exact causal-law statement

Fix theta, the encoded initial graph and the random field. Ignore resource censoring and require a schedule to fire an enabled active pair at each nonterminal step. If **one** finite reduction reaches a graph with no active pair, then:

1. Every maximal reduction reaches the identical terminal labeled graph, modulo representation of symbolic IDs, in the same finite number n of events.
2. Complete reductions contain the same causal event/emission set and differ only by commuting independent adjacent events. Any sum of fixed local event costs is identical.
3. Therefore, for independent local draws, the terminal law and the distribution of total local event cost do not depend on the nonanticipating scheduler on the normalizing random fields. If almost every field normalizes, this is a full probability distribution. Without that assumption it is a terminal subdistribution; no normalized-output claim is made by silently conditioning away divergence.

**Derivation.** Two distinct currently enabled pairs are disjoint. Their descriptors/codes and random keys remain unchanged by the other rewrite. They use the same predetermined U values in either order, and their independently emitted open graphs glue to the remaining wires in either order. Thus there is a labeled one-step diamond, including all emitted codes and IDs, not merely an equality after forgetting randomness.

To extend this local fact, induct on the length n of a fixed normalizing witness. For n=0 no event exists. For n>0, compare any alternative first event with the witness's first event. If they agree, use the induction hypothesis on the remainder. If they differ, the diamond lets the witness's first event commute past the alternative. The witness's first successor has a normalizing path of length n-1, so the induction hypothesis applied to its alternative first event gives a common successor with a path of length n-2. Hence the alternative first successor normalizes in n-1 events. Repeating proves that every choice decreases the remaining count by one; an infinite maximal sequence cannot coexist with the finite witness. The same induction explicitly permutes first events into place, preserving the event/emission set, and therefore additive local costs.

This is a manually checked mathematical reconstruction, not a formal proof artifact or executed verification. Its structure is the known strong-diamond/random-descent argument specialized to this emitter. It is not a new general confluence theorem.

### Learning consequence, without a new gradient claim

Let C denote a complete causal event configuration, including all emitted bits, rather than a serialization. Under any fixed valid scheduler, its probability is

`P_theta(C | G_0)=product_(e in C) p_theta(R_e | descriptors_of_e)`.

One complete local emission sequence defines each R_e; its probability is the product of that emitter's categorical decisions. The outside scheduler chooses which independent event to reveal next but does not choose its local distribution. The event products commute. Unvisited random variables integrate to one. Choosing a canonical scheduler defines this normalized probability measure when termination is almost sure; no factorial sum over serializations is required.

For a terminal-and-additive objective `F(C)=loss(decode(NF(C)),y)+lambda*sum_e c_e`, and a fixed theta-independent initial encoder, take a uniform finite event bound T, bounded F and differentiable positive local categorical probabilities. These sufficient conditions justify differentiation under the finite sum. Then

`grad_theta E[F] = E[(F-b) * sum_(e in C) grad_theta log p_theta(R_e | descriptors_of_e)]`,

where b is independent of the sampled configuration (it may depend on the input). The score evaluates conditional probabilities with sampled descriptors fixed. If the encoder is learned, add its ordinary score terms; if the decoder/cost depends directly on theta, add its direct derivative. There is **no schedule-policy gradient term** for this objective and fixed scheduler. Causal-key coupling gives the same F and the same score sum under every complete reordering of the same sampled causal configuration. Schedule augmentation therefore supplies no additional information about this terminal objective in the ideal class. This does not prove lower estimator variance or cheaper learning than a competent fixed-schedule rival.

The sufficient finite bound is a real missing condition for an unrestricted growing emitter. Ordinary p_theta can reproduce an active pair forever. Almost-sure termination without integrability does not automatically justify the gradient identity. The practical resource-limited machine is always terminable by the host, but that operational fact is distinct from the theorem's uncensored normalizing semantics.

This gives a substantive architectural consequence: learned stochastic graph operations may acquire a meaning independent of the serial schedule *for every parameter value satisfying the stated local model*, without learning an ordering policy or testing every interleaving. It concerns preservation of the generated computation's law under asynchronous execution. It does not establish acquisition of arbitrary binding, useful duplication, a task-efficient local law, or a new learning estimator. No factorial search-speedup is claimed: a conventional rival can also use a canonical schedule.

### Concrete failure boundaries

- **Allocator IDs or global time entering the kernel.** Two independent pairs e,f feed distinct output positions. A rule that emits 0 on the first global event and 1 on the second produces ordered output (0,1) if e runs first and (1,0) if f runs first. Its behavior is no longer a function of only the interacting pair. Remaining-budget-dependent probabilities cause the same problem.
- **Sequential RNG needs a precise distinction.** A shared *ideal iid* random stream still gives the same terminal distribution under nonanticipating schedules; swapping which independent event receives U1 versus U2 preserves their product law. It need not give identical results for the same seed: e.g. fixed stream values (.2,.8) assigned to two Bernoulli(.5) events swap their observed bits when the schedule swaps. Causal keys repair that pathwise comparison. Distributional failure needs more: e.g. a nonexchangeable shared stream with U1 a fair bit and U2 constantly zero gives (Z,0) versus (0,Z), or a scheduler that can peek at future draws. Merely saying “shared RNG breaks confluence” would be false.
- **Peak-live-space censoring.** Start with four agents in two independent pairs. One rewrite deletes two and emits four, changing live count by +2; the other deletes two and emits none, changing it by -2. With peak cap four, grow-first fails at six; shrink-first succeeds and ends at four. The uncensored labeled diamond does not make these censored outcomes equal. Physical parallel memory peaks, deadlines and scheduling overhead remain relevant.
- **A hard event cap is narrower.** When a coupled field has a normal form of n events, all maximal schedules need n. A step cap F admits it iff n<=F, if rejection is one uniform failure and no partial output is accepted. This does not imply equal partially incurred event costs, peak space, or failure diagnostics. Cumulative-allocation and variable-cost cutoffs require their own accounting.
- **Random-field implementation is not free.** Exact independent uniforms on countably many symbolic ancestry keys are the mathematical coupling. An implementation must pay ID/DAG storage and lookup or use a stated finite pseudorandom construction. Finite hashes can collide and are not a proof of exact independence. No RNG implementation, collision experiment or resource claim has been made here.

### Original-source comparison and what survives

Primary originals checked after this repair was proposed: Díaz-Caro–Martínez's distribution-confluence criteria; Faggian's probabilistic random descent; Varacca–Yoshida's confusion-free probabilistic event structures, local probability construction and fair-scheduler Theorem 6.2; Abbes–Mairesse's trace-monoid probability definitions and their warning about naive word sampling; Schulman et al.'s score-function/pathwise framework. Exact passages and the read/unread boundary are in the companion reading file.

These are real prior coverage. Schedule-independent local probabilistic computation is not new. Trace-equivalence priors also require care: counting all serial words can over-weight a concurrent configuration by its number of linearizations. The local normalized event construction above samples each reached causal choice once. This is **not** a claim to have constructed Abbes–Mairesse's uniform Bernoulli measure on an arbitrary trace monoid. Dynamic graph-enabled independence is not the fixed finite independence alphabet in that particular source; probabilistic event structures are the closer semantic rival.

The strongest matched rival now is a **learned probabilistic local graph-rewrite kernel with the same causal-key randomness, code precision, graph-growth API and event objective**. It can be understood as neural parameter learning of a confusion-free generative event structure. It inherits the same theorem, gradient and execution organization. A deterministic finite rewrite-table baseline would be too weak for this repair. A general neural graph learner that updates overlapping nodes in arbitrary order is a meaningful architectural comparator, but beating it would show the value of an existing confluence discipline, not automatically a new foundational principle.

What remains OPEN is a concrete contribution at the intersection: the neural emitter can generate the computational encoding and future interaction behavior while the schedule-invariance guarantee is structural. Does this remove an actual acquisition barrier for higher-order mechanisms, beyond applying known probabilistic-concurrency semantics to a learned kernel? A useful next construction must explain how transferable interaction behavior forms, including the function-copying obstacle, with the same prior and total cost available to the event-structure/rewrite rival. The exact causal law sharpens a consequential execution principle; it does not yet solve the foundational formation problem.

Updated decision: retain this **root-seeded high-risk OPEN repair** as a concrete architecture/derivation, not merely finite rule synthesis and not a novelty result. No experiment admitted yet. The next single decision is whether a mechanism for acquiring coherent generated interaction laws can exploit this exact causal organization in a way not already captured economically by the same-interface probabilistic evaluator learner.
