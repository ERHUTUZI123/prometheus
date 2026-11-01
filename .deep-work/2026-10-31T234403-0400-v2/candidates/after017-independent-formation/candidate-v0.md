# Independent formation v0: learn an exact routing law by coherent rule-table search

Status: **AGENT PROPOSAL / HYPOTHESIS; acquisition not executed.** This is an independent constructive pass on the neutral mechanism target, in the existing V2 campaign. It is not a campaign restart. No current candidate, review, implementation, or old campaign report was read before this freeze. Isolation is `instruction_only` on a shared filesystem, not access control. The effective provider model/mode is unknown; no Pro claim is made. Authorized work here was primary-source reading and reasoning; no experiment, training, installation, paid route, or external publication occurred.

## Judgment and exact question

Exact mutable communication can be a meaningful inductive bias if learning discovers a small, reusable **routing law**, while input-dependent pending relationships reside in exact links. Universality and the availability of rewiring do not establish this. The formation problem is to acquire the law that allocates, revisits, distinguishes, and retires those relationships from endpoint feedback.

My candidate couples the proposed local graph substrate to **rollout-consistent search over latent local rewrite tables**, followed by neural distillation of the successful tables. This removes a particular source of incoherent exploration: the learner does not independently guess a different rule every time the same local situation recurs. It also makes the complete acquisition object inspectable. This is an executable learning specification, not a claim that its search will scale. Its most fragile step is discovering the codes and routing conventions that make the correct action locally identifiable.

Connection to the Fundamental Question: the neural kernel's outputs construct subsequent computation and memory paths; these paths are not supplied as a parser, binder, stack, or interpreter. What must be acquired is an internal operational organization, not merely parameters of a supplied semantic module. The contribution would be consequential only if that organization lowers the acquisition cost of later, structurally harder behavior against a pointer learner given the same feedback and structured exploration.

## A success counterfactual worth testing

Suppose a kernel learned on short identity-dependent interaction streams acquires an organization that supports new combinations of scope depth, aliasing, and delayed use with a small increase in training examples and search. The important change would be: the complexity of pending contexts grows mainly in an exact, input-dependent communication graph, rather than requiring the controller to learn a separate numerical representation for every context size or to rediscover a long sequence of routing decisions at every use.

The mechanism is not generic modularity. An edge retains a particular pending dependency, and the same acquired local rule processes that dependency at arbitrary graph positions. This could matter beyond a toy because variable binding, suspended computations, sparse relational state, and delayed matching all require retaining many identity-specific obligations. The structural prediction is transfer with the number and arrangement of obligations, after charging for graph formation and maintenance.

The strongest simple alternative is an exact pointer-memory learner with equally coherent program exploration. It already has the relevant representational capacity. If it learns the same organization as cheaply, the proposed substrate may remain useful engineering, but the central learning-bias claim loses its reason for primary-line allocation.

## State and transition: no semantic host shortcut

Freeze a finite code alphabet `C={0,...,2^b-1}`. A node has a code, polarity `+/-`, one principal port, zero to two ordered auxiliary ports, and one optional atom reference. One reference per node is a v0 restriction, making the local descriptor space explicit. An atom is opaque: equality and copying an encountered reference are allowed; numerical identity, order, hashing, arithmetic on identity, and arbitrary atom synthesis are unavailable to the kernel.

A state is a finite collection of nodes, a perfect matching of their ports and external interface ports, and the remaining external resource counters. A principal-to-principal edge joining opposite polarities is an active pair. The scheduler chooses one active pair without reading task truth. The kernel observes only the two node descriptors and the equality pattern of their optional atom references. It cannot inspect neighboring labels, addresses, boundary identities, the rest of the graph, wall time, or task truth.

An action specifies zero to four fresh nodes, their descriptors, selectors copying the old atom references or the empty reference, and a perfect matching over the fresh ports and the old auxiliary boundary stubs. Every boundary stub occurs exactly once. The old pair is deleted. Splicing means composition of matchings along alternating wire paths, including the case where old boundary stubs were already connected. Closed wire loops must have an explicit representation or be discarded by a declared substrate convention. This wiring operation is host substrate work, not a learned binder or parser. No host routine follows a name, copies a function, recognizes a scope, or executes a token instruction.

For fixed `b`, each descriptor `d` has a finite set `A_d` of legal actions. Legality checks only arity, copied-reference selectors, matching completeness, and fresh-node bounds. Syntactically legal deadlocks and unproductive growth remain learnable failures. A table `T` assigns one action `T[d]` to every descriptor. Evaluation fixes `T` throughout the rollout. Neural `K_theta` parametrizes a distribution over these actions during learning; deployment takes a deterministic table or deterministic kernel. The graph may grow, but it is externally capped in nodes, rewrites, and elapsed process resources. A cap hit is a failed output, not successful termination.

Input is an ordinary linear chain of raw token nodes plus Start; any instruction categories are token observations, not host actions. Output is a raw token/atom sequence at a declared external interface, with fixed framing only. Task truth is checked externally after execution. The kernel has no access to that truth within the rollout.

## The actual acquisition law

The trainable object is a distribution over **complete local operational laws**, not a supplied list of semantic instructions. Let

`q_theta(T) = product_d q_theta(T[d] | d)`.

The factors can share neural parameters and autoregressively generate legal fresh-node descriptors and matchings. For each sampled table, lazy sampling stores the first sampled action for each descriptor and reuses it on every subsequent encounter, across the whole scored task batch. Lazy sampling is only an implementation of sampling a table; it is not online learning or memory available to the task solver. It must be charged as training machinery.

For a training batch `B={(x_i,y_i)}`, define

`F_B(T) = - sum_i endpoint_error(Exec_T(x_i),y_i)`

`          - lambda_r * sum_i rewrites_i - lambda_n * sum_i peak_nodes_i`

`          - lambda_c * encoded_size(T_used)`.

The primary endpoint error is exact task failure. Optional token-wise error is permitted only if the rival receives the identical signal. Resource terms cannot buy acceptance of wrong output; use lexicographic ranking, correctness first, then costs. `T_used` is the union of encountered entries, and the encoding convention must be fixed before comparing systems. Charging each dynamic occurrence as a separate new rule would erase the intended reuse bias. Charging graph pointers as zero bits would falsely favor the graph.

One outer update has a finite particle budget `S`, each rollout has a fixed cap, and the search has an external termination path:

1. Sample `S` tables from `q_theta`, score each on the same training batch, and retain the exact tables and actual traces. No partial semantic program is supplied.
2. Reweight by `w_s proportional exp(beta * (F_B(T_s)-max_t F_B(T_t)))`, or use a declared elite quantile. This is a specified search heuristic, not an unbiased likelihood gradient.
3. Fit the neural proposal by one or more bounded steps on

   `L(theta) = - sum_s w_s sum_{d in U_s} log q_theta(T_s[d] | d)`.

   Here `U_s` is the set of distinct entries encountered, counted once. Fresh code assignments, allocation decisions, wire matchings, copying, and retirement behavior are all variables in this law. It does not start with resolver/stack/call rules.
4. Mix the updated proposal with a nonzero prior component to keep alternatives accessible. Replay previously solved endpoint tasks and preserve the best exact tables; account for replay, search, and distillation. An archive is not a claim of immunity to forgetting.
5. Expand the training distribution only according to a declared input-length and interaction-combination schedule. Short cases can supply a reward foothold, but there is no guarantee that a curriculum crosses a sparse-reward barrier. Evaluation uses fixed tables, held-out combinations and names, and no parameter update.

This law addresses consistency of exploration and repeated-use credit. It does **not** solve global code invention by fiat. If early tables never produce useful behavior, reweighting and distillation have no semantic foothold. The proposal's advantage is therefore conditional: once a local convention becomes useful, it is reused coherently and its cost is counted once. Episode-consistent exploration is already established in other neural learners; this cannot be claimed as novel in itself. See [Plappert et al., pp. 2–3](https://arxiv.org/pdf/1706.01905).

Across training episodes, the retained object is the table/proposal and its tested endpoint history. Graph topology is transient within a rollout in v0. No online weight learning, evolutionary population, autonomous task generation, or cross-episode graph inheritance is assumed.

## One substantive reconstruction: finite local observations do not imply a finite-context learner

**RECONSTRUCTION, mathematical argument; no computational execution claimed.** Let atoms form a countably infinite set `A`. Consider the endpoint language

`L = { w # a : a occurs in the finite atom sequence w }`.

Take any deterministic, one-pass, atom-equivariant learner/executor whose complete retained state after the prefix has nominal support at most `k`. It may have arbitrarily many atom-independent numerical bits. It receives no external replay of `w`, atom identity encoding, or hidden memory beyond this state. Then it cannot recognize `L` for all input lengths.

Proof: present `k+1` distinct atoms in `w`. At least one, `a`, is outside the support of the retained state `s`. Choose a fresh `z` outside both `w` and the support. The permutation swapping `a` and `z` fixes `s`. Equivariance of the continuation computation, whose output is a Boolean with empty support, forces the same answer on suffixes `#a` and `#z`. One must be accepted and the other rejected. Contradiction. Randomness does not remove the deterministic zero-error impossibility by merely adding atom-independent random bits; finite-error tradeoffs require a separately stated model.

Why this bears on formation: a local descriptor can have only two atom references and therefore finitely many equality-pattern orbits, yet the **whole graph** can retain arbitrarily many references and have unbounded support. A small local rule alphabet is not a small global residual state space. A dynamic graph can retain the input chain and scan it with a query-carrying node; that is a representability witness, not a learned solution or a proposed result. Importing a nominal-automaton learning guarantee solely because local observations are orbit-finite would be wrong. The checked primary source's learning setting has orbit-finite global state and active oracle assumptions; see [Moerman et al., §§2–4](https://arxiv.org/pdf/1607.06268). Our graph machine need not satisfy those assumptions.

There is a complementary finite counting boundary. Store a bijection between `m` fixed distinct query atoms and `m` fixed distinct response atoms, erase the original presentation, and later accept any single query. There are `m!` different residual behaviors. A deterministic zero-error representation must distinguish them, requiring at least `ceil(log2(m!))` bits of association information somewhere. Keeping both atom sets without an association does not suffice. Bounded-degree exact links can carry this information with `O(m)` nodes and pointers; those pointers themselves cost `Theta(m log m)` bits under an ordinary index encoding. A pointer RAM can carry the same information. Thus topology is a genuine state carrier, but this argument establishes no special learning advantage over exact pointer memory.

These are not supplied-rule-choice tests. They locate the actual burden: acquire a rule for managing **unbounded relational state**, not merely a small classifier over the local equality cases.

## Confluence is useful, with a strict boundary

**SOURCE:** deterministic, boundary-preserving interaction rules with one principal port have strong confluence up to fresh-name renaming; the checked proof treats distinct redexes as disjoint. [de Falco, Proposition 5.3 and Corollary 5.4, printed pp. 13–14](https://lmcs.episciences.org/1108/pdf).

**RECONSTRUCTION:** the same argument applies when a fixed table additionally reads node-local atom equality and copies only encountered atoms. Disjoint redexes cannot change one another's descriptors; replacing both gives the same glued net modulo fresh node IDs. This removes scheduling order as a hidden semantic search variable for completed terminating execution. Sampling independent actions at repeated encounters, reading global counters in K, or mutating node-local state outside the rewrite invalidates that premise.

This is not an unconditional budget claim. A rollout interrupted before normalization may expose different partial outputs under different schedules. A complete test must define external termination and compare total rewrites, peak storage, and wall-clock costs. No free parallelism or physical energy advantage is claimed. A learned rule can diverge or retain useless graph components despite confluence.

## The hardest formation gap and its strongest attack

The acquired local descriptor must distinguish contexts that require different actions. Two histories that reach the same pair of codes and atom-equality pattern are forced to use the same rewrite. Correct behavior may require propagating an extra code, storing a continuation in a wire, or postponing a decision until another node interacts. The learner must discover which of these to do. Calling them learned continuations does not supply them.

Endpoint-only table search can face an exponential barrier in the number of mutually dependent entries. Coherent exploration removes repeated stochastic inconsistency; it does not turn that barrier into a polynomial algorithm. The neural factorization can also correlate harmful changes, and canonical code relabelings create redundant hypotheses. Canonicalization of the training representation may merge literal code renamings, but must not introduce semantics or count re-encoding as new computation.

The closest alternative is neurally guided program induction with learned reusable abstractions; [DreamCoder, Eq. (1), printed pp. 4–7](https://arxiv.org/pdf/2006.08381) already makes acquired operational knowledge and neural search mutually reinforcing. The distinguishable proposal here is acquisition of the low-level communication law without a typed lambda evaluator or task DSL carrying the intended binding operations. Whether that weaker prior is useful or merely makes the search worse is open.

Likewise, [NRAM, §§3.1–3.4](https://arxiv.org/pdf/1511.06392) already restricts controller access, manipulates pointer memory, trains from endpoints, and discusses exact discretization. The comparison must not handicap NRAM by forcing soft pointers at deployment while giving this system hard wires. [Neural GPU, §2](https://arxiv.org/pdf/1511.08228) supplies a fixed-lattice, shared-update rival; its working state also scales with input size. Fixed lattice is not equivalent to fixed-capacity hidden state.

## A discriminating next decision, not an executed experiment

Use one family of raw token streams describing nested scope, rebinding, delayed lookup, and name aliasing. Host truth is the endpoint response sequence; neither learner receives a parse tree, scope stack, binding edges, execution traces, or intermediate truth. New atoms are renamed by arbitrary bijections. No result on this family alone establishes a general language learner.

Before expensive work, construct and inspect whether short streams admit any reward foothold under the declared action prior. That is a construction/coverage question, not an instruction to start training. A serious subsequent comparison must include:

- The graph learner above and an exact-pointer NRAM-like register controller, each with raw tokens, equality/copy access, the same endpoint signal, variable memory, and episode-consistent search. Pointer address arithmetic must be either available to both as paid substrate machinery or excluded from both when it grants unmatched information.
- A fixed-lattice shared-update learner with enough state to preserve the complete input. Its total cell updates and precision cost count.
- A synthesis rival allowed to learn new low-level routines, with no supplied binder or parser. If a full semantic interpreter is used as a separate upper-bound baseline, label its prior information and cost explicitly.

Freeze total training evaluations, endpoint labels, parameter bits, complete graph/pointer bits, proposal/search compute, runtime rewrites or primitive operations, peak state, and all unsuccessful searches. Hold out joint combinations of nesting, shadowing and delay, not only fresh atom spellings. The central metric is total acquisition cost to a predefined exact-success target on that held-out combination distribution, with censoring at the same budget.

The causal intervention is to preserve the acquired local table while disrupting only the supposed retained communication relation, with equal-cost sham rewiring. Today's task damage is a storage-function test. To test subsequent learning, replay the same pretraining checkpoint, initial endpoint capabilities, and training budget; otherwise lost initial capability confounds any acquisition effect. A matched explicit-pointer implementation of the learned routing law is particularly important: if it retains the same acquisition benefit, the result concerns acquired routing organization rather than the particular net notation.

Decision boundary: continue only if the candidate provides a plausible route through the code/routing formation barrier and a material difference against the exact-pointer learner. Do not declare novelty from this v0, exact execution from this prose, or learning from a hand-constructed rule table. No next experiment is authorized by this artifact itself.

## Frozen claim record

`CLAIM`: coherent latent-rule search plus exact, input-dependent relational storage may reduce acquisition cost for compositional identity-dependent computation.

`STATUS`: HYPOTHESIS; no kernel, training run, sample-efficiency result, or novelty certificate.

`EVIDENCE/ARTIFACT`: explicit state/transition/acquisition law; checked source passages; support and association-memory arguments above; confluence reconstruction conditional on deterministic local rules.

`WEAKEST REMAINING LINK`: discover useful context codes and routing conventions from sparse endpoint feedback before exponential joint-rule search dominates.

`FAILED BRANCHES`: no empirical branches attempted. Rejected inference: finite local descriptor orbits imply a nominal-automaton acquisition guarantee. Rejected comparison: hard graph links versus only soft-pointer or fixed-hidden-state rivals.

`NEXT SINGLE DECISION`: after comparison inputs, assess whether the formation mechanism offers anything beyond known structured exploration and pointer/program induction; if not, treat this as a clarified substrate hypothesis rather than a new primary principle.
