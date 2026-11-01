# P v1 — acquire an operation together with a composable way to change it

**AGENT PROPOSAL / HYPOTHESIS.** The stronger candidate is a neural computer that constructs both executable routines and their *intervention interfaces*: learned procedures describing how persistent changes can make a routine meet a new demand. Constructing a routine composes its constituents' repair procedures. Failed compositions refine the repair interface; successful repairs become reusable internal change operations. Consequently, learning a capability can change the algorithms by which the system acquires later capabilities.

This continues P's interface-acquisition question. It does not require an exact conserved context in arbitrary neural state, and it is not a rescaling repair to v0. The new causal interaction is **forward routine acquisition -> acquisition/composition of update semantics -> structured changes to later routines -> new executed evidence and new update semantics**. All scientific capability claims below are separated from this proposed mechanism.

## 1. Why this could matter

An executable library stores what to compute. A learning system also needs to know *how its computations can be changed*. A desired output generally corresponds to several incompatible internal changes, some impossible under current bindings, some cheap, and some destructive to retained behavior. A point inverse or a useful forward abstraction need not expose those distinctions.

The opportunity is to make this controllability structure an acquired computational object. A newly learned neural routine should enter future computation with more than a forward call: it should carry a procedure for allocating interventions among its inputs, local parameters, routing choices and constituents. This procedure can be reused on new combinations rather than rediscovered from terminal task reward each time.

Success would change finite-budget acquisition of composed computation. It need not enlarge the set of computable functions. Ordinary gradients, a fixed interpreter, arithmetic and explicit module calls are allowed. The intended result is neither a new definition of general intelligence nor a promise of continuous open-endedness.

## 2. The new distinction: same value, different possible change

Consider the currently executed circuit

    y = AND(z, OR(NOT(z), w)),       z=0, w=0.

z and w are persistent editable choices, not independently reset inputs on each execution. Each bit flip costs one. The forward output is 0. A new task requests y=1.

Separate local inverse choices would say:

    left AND input: make z=1, cost 1;
    right AND input: OR(NOT(z),w) is already 1, cost 0.

Combining these choices predicts cost 1. Applying the change to the actual shared z makes the right branch 0, so it fails. Algebraically y=z AND w; the true minimum is the two-edit intervention z=w=1.

The missing interface variable is **the proposed value of the shared intervention on z**, not another observation of its current value. Condition the right repair message on that proposed value b:

| proposed z=b | minimum additional w-edit cost to make right branch 1 |
| --- | --- |
| 0 | 0 |
| 1 | 1 |

The left branch demands b=1 and pays the single owning z-edit cost. The composed minimum is now 1+1=2. A port carrying only the current value 0 could not make this distinction. The execution/provenance record identifies that the two reads refer to one editable object; a learned interface initially hid this dependency and must expose it after the conflict.

This is a worked mathematical consequence, not an executed neural result. It motivates a specific interface change. The proposal does not claim that sharing or conditional inverse reasoning itself is new.

## 3. Machine state and concrete objects

At learning step t the computer holds

    S_t = (G_t, theta_t, L_t, I_t, P_t, R_t, D_t).

G is a bounded internal call graph/program, theta contains neural parameters and other persistent choices, L is the callable library, I the repair-interface schemas, P repair procedures and their witness records, R replay/checkpoint evidence, and D the current finite task data. The neural constructor proposes code/routing/module changes; a bounded interpreter executes them. Local neural cells and learned repair heads can use gradients. A repair is an actual persistent intervention e, applied by a specified edit operation theta' = Apply(theta,e), and may also change G within its allowed action grammar.

A library entry has:

* **Execute:** F_m(x;theta_m), a bounded neural computation or call graph. It runs in a designed local workspace; its scratch activity may change arbitrarily. Explicit inputs/outputs and edit ownership replace v0's unsupported universal context invariant.
* **Intervention identity:** persistent edit objects and ownership/provenance. Repeated reads of the same object remain the same potential intervention; a parameter is not silently duplicated at a call boundary.
* **Repair:** B_m(target, incoming repair messages, interface context), returning proposed consistent edits, their actually measured costs, and an execution witness. It may return several alternatives or fail within its budget.
* **Interface schema:** a learned selection/encoding of the distinctions needed to communicate repair alternatives, including proposed shared edits, conditional bindings and effect summaries. It is not assumed sufficient for every future operation.
* **Version:** forward and repair versions are recorded together. A changed implementation is retested; stored old execution and old repair behavior are not silently relabeled.

The designed scaffold supplies the bounded interpreter, edit grammar, provenance identities, data handling, optimizer, interface-extension operations, costs and validation. The acquired objects are routines, repair procedures, useful conditional interface channels, macro-edits, and the neural policy that uses them. None of this is described as fully autonomous emergence of a substrate.

## 4. The repair relation and its executable composition

For a module, task inputs X=(x_1,...,x_N), persistent base parameters theta, allowed edits e, and actual edit cost c(e), define the executed output profile

    Y_m(X,theta,e) = (F_m(x_j; Apply(theta,e)))_{j=1..N}.

The *same* persistent e is used on every example. Independently choosing a different edit for each example is prohibited: that would solve input inversion instead of learning a mechanism. A desired output condition Phi is evaluated on this whole profile and any required retained-task profiles.

The exact ideal repair object is a weighted relation

    K_m(Phi;context) = inf { c(e) : e admissible,
                                    Y_m(X,theta,e) satisfies Phi,
                                    shared edit bindings and retention constraints hold }.

It includes a witness generator, not only a cost estimate. Empty feasible sets have cost infinity. This is a mathematical definition; an unverified neural estimate is not an exact K. A finite search may find an upper bound with a real witness without proving optimality or impossibility.

For a tree whose child edit sets are disjoint and finite-valued interfaces a_i are explicit, a node with local edit q has the recurrence

    C_v(y) = min_{q,a_1,...,a_k:
                  F_{v,q}(a_1,...,a_k)=y}
                    [c_v(q) + sum_i C_child_i(a_i)].                 (1)

Store the minimizing q and child interface values to reconstruct an actual persistent intervention. Leaves' messages are the costs of the available persistent alternatives. For N examples, each a_i and y denotes a full N-example output profile; equation (1) does not remove that potentially enormous state space.

For shared edits, messages are conditional on a separator assignment Gamma of the shared proposed edits. An edit cost is counted at its unique owner. Child assignments must agree on shared identities. Minimizing them independently is the error in Section 2. A learned compressed message is permitted only as an approximation; the full proposed intervention is still replayed through Execute.

This is min-plus relation composition / structured inference, not claimed as new mathematics. It gives an operational semantics for the acquired repair channel. Autodiff supplies a different exact compositional change calculus in smooth settings; the proposal need not discard it. A repair procedure may invoke gradients locally, retain multiple discrete alternatives, and communicate conditional persistent edits that a single local derivative does not describe.

## 5. The actual acquisition loop

1. **Construct a forward routine.** The neural constructor composes current cells/routines or learns local parameters from task feedback. Its source computation and edit provenance are retained. Primitive routines need not already have a good repair head.
2. **Acquire repair evidence from execution.** Execute inspected bounded perturbations e of its actual persistent state. Record (X,theta,e,Y_m,c(e),bindings). Positive evidence is an executed feasible change. Failed demands and infeasible combinations are retained. No inverse oracle is supplied for the new routine.
3. **Train a repair head.** A neural conditional proposal pi_psi(e | Phi,context) learns from verified repairs and bounded exploration. An estimator V_psi predicts discovered costs. Labels from bounded search are named best-found upper bounds, never certified optima. Pairwise cost ranking and execution success train the proposal; a false proposal produces another real counterexample.
4. **Compose before rediscovering.** When a forward routine H is formed from existing entries, its initial B_H is the executable composition of their repair relations/proposals under (1) and shared-edit binding. This constructs a learning procedure for H immediately from acquired constituents. It is not a newly hand-authored inverse function. A neural head may later distill this composed procedure into a cheaper conditional computation, preserving an executable fallback and witness boundary.
5. **Refine an interface on repair conflict.** If separately feasible repairs cannot be combined, follow their executed edit provenance. Expose a previously hidden shared edit identity/condition or a discriminating effect predicate. Recompute the affected messages. More generally, if two edit situations have the same current summary but a new caller assigns them different feasibility/cost, extend that summary. A learned gate can choose among such trace-grounded distinctions and compress after replay; the trace-grounded candidate generator is designed.
6. **Retain useful repair computations as actions.** Repeated successful structured repairs can themselves be compiled into parameterized internal routines that emit edit packets. A new repair routine is a first-class computation with input conditions, a finite execution bound and tested effects. The constructor can invoke it during later learning. Thus the internal action vocabulary changes on both the execution side and the learning side.
7. **Use and test the change.** Apply a proposed persistent edit, execute on the specified task and retained inputs, and keep only outcomes supported by those checks. Current-task fit, retention, held-out composition and later discovery are reported separately.

This is a reciprocal training process: executing routines supplies data for their repair mechanisms; those repair mechanisms construct updates that change later execution and the distribution of learnable routines. Ordinary library extraction has no obligation to create such an update path. Ordinary representation learning has no obligation to make a newly constructed computational object arrive with a reusable learning procedure.

## 6. A refinement law that retains old interfaces without freezing future semantics

Let sigma(e) summarize an edit situation at a boundary and let K(sigma,t) be its exact best cost for a target t under the old caller's assumptions. A new caller distinguishes an additional trace-grounded property d(e), so define

    sigma_new(e) = (sigma(e), d(e)).

Old readers recover sigma by projection. In a finite exact relation, under the *same old caller assumptions* and with all feasible d alternatives retained,

    K_old(sigma,t) = min_d K_new((sigma,d),t).                         (2)

Proof: the old feasible edit set is partitioned by d; taking the minimum over its union equals taking the minimum of the block minima. This is elementary, not a novelty claim. It provides a concrete retention operation for the repair semantics rather than presuming one fixed interface is adequate forever.

The new caller can constrain d; the old one marginalizes it. The learned computer can therefore expand its internal operation/test language without discarding all old abstraction. This costs state and computation, and it can fail to stay compact. Equation (2) does not make an old coarse witness automatically valid under a new caller's extra constraint. It also does not license throwing away the unrefined edit source before refinement is possible.

This is a specific response to v0's dilemma. The invariance is a projection/marginalization relation between versions of *intervention semantics*, not conservation of every surrounding neural state coordinate or sufficiency for all imaginable future tests.

## 7. Conditional computational consequence

**RECONSTRUCTION, conditional exact claim.** Suppose an internal computation is an n-node tree; each node has arity at most d, finite interface alphabet size at most k, at most b local persistent edit choices, and disjoint owned edits. Complete local execution/repair relations are given or acquired. Then equation (1) computes an exact minimum-cost repair for one task instance in O(n b k^(d+1)) elementary table work, plus execution cost. A witness is reconstructed by following stored minimizers. Proof is induction on the tree: every feasible global edit decomposes into one local choice and independent child edits, and every compatible tuple of these edits is globally feasible.

Enumerating all b^n edit assignments is unnecessary in this class. That is a consequence of its conditional structure, not an empirical comparison or a new dynamic-programming theorem. The important organizational fact is that a newly constructed routine can inherit this repair algorithm from its constituents: acquisition creates a usable learning operation, not just a forward name.

The strong limits are explicit. For N learning examples the naive output-profile alphabet is k^N. Shared parameters/variables require joint messages and can produce exponential separator width. Neural continuous parameters do not come with complete finite relations. V1 does not magically evade any of these costs.

**HYPOTHESIS, possible significant class.** Rich neural routines may nevertheless admit a compact family of *intervention-cost summaries* across a task family, even when their forward computation and parameter space are large. Learning those summaries, adding the needed coordination channels and compiling successful repair programs could let a growing computer retain small, effective learning interfaces. Then a library's growth would transfer procedures for acquiring new mechanisms, not merely execution fragments. This might matter in modular planning, learned data-structure manipulation, inverse graphics or construction of discrete/continuous algorithmic routines. These are prospective classes, not demonstrated capabilities or a new benchmark proposal.

The intended principle is **organize computation partly by how its behavior can be changed**. Whether that yields a useful organization beyond existing learned optimizers, target propagation and symbolic inference is OPEN. It is not settled by universal simulation in either direction.

## 8. Strong actual controls and where they overlap

The targeted source record is sources/P/source-record-v1.md. The following boundaries are consequential.

* **Difference target propagation (Lee et al.).** This is a real neural control that learns feedback mappings and trains forward modules using local targets. It already couples a forward computation to an acquired learning path. V1 cannot claim that idea generally. Its proposed difference is a set-valued, cost-aware intervention procedure with persistent/shared edit identities, library construction and interface refinement; a single point inverse is not supplied these features for free. Backprop/autodiff with the same forward modules is also an essential actual control in smooth regimes.
* **FlashMeta/PROSE and NGDS.** These are real deductive/neural synthesis controls. They already use inverse semantics, conditional dependencies and neural search guidance. V1 cannot claim backward constraint propagation or conditional inverse composition as new. A comparison should give them the same primitive semantics, source graph, data and total execution budget. Their documented operator-specific witness requirements must be counted; newly created routine witnesses are not silently handed to them as if free.
* **Alford et al. 2021.** Its actual system combines learned abstractions and inverse-based search in separate approaches, and explicitly anticipates synthesizing inverses for newly learned abstractions as future work. Thus even the general ambition of a growing forward/backward library is not new. The precise proposal here is intervention-cost/interface acquisition and refinement, not that broad slogan.
* **Generic structured inference / program repair.** A solver that inlines the same source graph and builds extensional constraints from the same executed local tables is a strong control for the exact finite class. It may match (1) without neural repair learning. V1 earns a contribution only if acquired compact repair procedures or interface organization change acquisition under fair total costs. Giving this solver the learned compressed repair relations at zero acquisition cost would give it the coupling being tested; giving it the original graph and equal primitive calls is fair.

Useful ablations of v1 would keep forward routines, task evidence and budgets fixed while removing repair-head inheritance, removing interface refinement, or learning unstructured repair proposals directly. These are causal distinctions, not a request to run a benchmark campaign now.

## 9. Main failure modes and unresolved constructive step

* Most inverse relations may be too broad or too correlated to compress. A low-cost witness on training examples is not a reusable learning rule.
* Persistent edits must remain consistent across examples and across repeated module calls. Solving per-example input inversion would give a misleading success.
* Learned cost heads may hallucinate feasible edits or underestimate cost. Forward witness execution grounds feasibility; only separate exact reasoning can certify optimality/absence. Pruning on an unverified learned lower bound is not allowed.
* Interface refinement can grow until it effectively inlines the whole computation. Then it has acquired no compact learnability advantage. This is not repaired by counting more modules.
* Macro-edits can overfit the training task distribution or destroy later adaptability. Retained performance and a small edit count alone do not establish useful future search.
* Sharing a repair head across new routine structures may transfer errors and create mutually reinforcing bad targets. The composed procedure and recorded counterexamples remain inspectable rather than being replaced by model consensus.

**Hardest constructive step:** find a neural parameterization and acquisition pressure under which a family of reliable intervention-cost/coordination summaries remains compact as the machine constructs new routines, and show that reusing these summaries changes later learning rather than only matching known symbolic inference. V1 specifies an actual bidirectional mechanism and a conditional class, but has not solved this step.

## 10. Status and next single scientific decision

No new neural experiment or chart enumeration was run for v1. Its construction, exact tree consequence, shared-edit example and refinement identity are mathematical reconstructions with the stated assumptions. The proposed neural acquisition loop and broad transfer effect remain HYPOTHESES.

The next decision is which concrete structured neural computation could admit a compact learned repair interface while presenting a real difficulty to both target propagation and a graph-aware symbolic solver under equal primitive access. This decision serves the mechanism's ambition; it should not be replaced by optimizing a convenient toy score. The campaign remains active; this note does not recommend abandoning P or freezing research direction.

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION
--- | --- | --- | --- | --- | ---
Constructed routines can carry compositionally acquired learning/repair procedures | Explicit mechanism / HYPOTHESIS of useful neural acquisition | Sections 3–6 | Compact and reliable learned intervention summaries | V0 exact frame and latent metric attacks retained separately | Select consequential structured case where this coupling could matter
Exact finite tree repair composes without exhaustive global assignment enumeration | RECONSTRUCTION, proof above | Equation (1), stated disjointness/domain assumptions | Does not give neural acquisition or multi-example compactness | Independent shared-edit composition explicitly fails | Preserve separator identities; investigate learned compression
New repair interfaces can refine old ones with exact coarse semantics recoverable | RECONSTRUCTION | Equation (2), finite feasible-set partition | Source alternatives/provenance must remain recoverable | No universal future sufficiency asserted | Make refinement produce executable new coordination channels
V1 surpasses DTP, existing inverse synthesis or graph-aware inference | OPEN, not claimed | Targeted actual-control comparison | No experiment or novelty resolution | General forward/inverse library ambition already anticipated | Need a discriminating problem and matched acquisition accounting

## Lineage and isolation

v0, its freeze manifest, the original finite probe and postfreeze-analysis.md are unchanged. V1 was developed in response to new user steering conveyed by the coordinator, using P's own attacks and targeted primary sources. Frozen K/R were permitted but not read: no concrete cross-input gap required them. All writes stay in P's assigned current-run directories. Filesystem isolation is instruction-only; same-model branches are not independent mathematical or empirical verification. Requested profile=pro; parent-reported host selection=gpt-6.1-sol/effort ultra; provider-effective profile remains unknown.
