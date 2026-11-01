# Acquisition of an immutable local neural law — v2

**AGENT PROPOSAL / HAND CONSTRUCTION, not trained or executed.** Same campaign, same raw-stream MAIN. Root requested an acquisition mechanism and exposed v1's exponential owned-environment growth. The supporting worker exposed v1's constant-function reuse shortcut and independently audited its port equations. Those results are preserved. V1 and v0 are unchanged; clarifications are in `raw-stream-v1-erratum.md`.

The substantive revision has two parts: explore **whole immutable neural interaction laws**, so an exploratory hypothesis has stable semantics throughout recursive use; and admit **locally constructed selective closure capture**, so the hand witness need not duplicate every ambient binding. Neither parameter-space exploration nor closure conversion is claimed new. Their exact role in this architecture is specified below. The actual acquisition advantage over Neural GPU/NRAM/NLI is OPEN.

## 1. Fixed information and observable fitness

There is one execution kernel. Initial state is the raw token chain+Start of v1, with the root-seeded nominal equality/reference interface fixed explicitly. The kernel sees only its two structural descriptors, their opaque-atom presence/equality mask and its own bounded emission prefix. It never sees global raw input, expected atom, parser/type state, caller address, environment depth, elapsed time, remaining fuel or a correct intermediate graph. Parent/source programs are not parsed by a host loader. The host supplies pointer matching/allocation and external limits only.

An external task example is `(raw_token_chain, terminal_ground_atom)`. A task generator/reference may know types or interpretation to create the label, but none of that state or any teacher trace is exposed to either learner. Its generation/query cost is part of the shared study account. No intermediate FV set, variable binding, correct clone or continuation is labeled. The four raw source-tag tokens and token identity/order are the common input representation.

**Complete-drain protocol is fixed here.** After the output atom becomes connected, continue until there is no active pair, or the external event/allocation/space limit is reached. Correctness requires a ground atom at out after this drain. Budget exhaustion is failure even if the expected atom appeared earlier. Meter all copying, erasure, neural arithmetic, queue/rewiring operations and residual graph storage. Inert garbage can remain but is charged; no host garbage collector secretly completes a learned Erase protocol.

For a finite bank B of M raw I/O examples, define bounded loss

`F_B(W) = (#wrong_or_exhausted)/M + lambda * mean_B normalized_total_cost`,

where normalized cost is in [0,1] and `0<lambda<1/M`, e.g. `lambda=1/(4M)`. Thus one additional correct bank answer outweighs any possible aggregate resource saving. Exhaustion cost includes the work actually spent; the cap and schedule are part of the task contract. The bank is sampled identically for candidates/competitors and all queries count. A finite bank is not a semantic certificate. No concrete training bank, hyperparameter values or run have yet been authorized/frozen in this worker.

## 2. Compact emission parameterization, with no named interpreter roles

Let each structural code have b bits, and let W be the finite-precision weights of a shared local encoder plus bounded autoregressive emission head. Input contains the two b-bit codes, arity one-hots, principal polarity/order, atom-presence bits and the equality bit. Opaque atom IDs are not numeric features. The hidden vector can be produced by two ordinary width-h ReLU layers. No table indexed by every possible code pair is materialized.

The output head emits, in a fixed **local** order:

1. k in {0,...,4}.
2. For each new node: arity 0/1/2, principal polarity, b structural-code bits and an atom-reference selector (none, either input reference, or a bounded fresh reference).
3. A port matching: repeatedly take the smallest unpaired local port and choose its partner among the remaining ports. Invalid parity/descriptor combinations are masked. There are at most 16 ports and eight partner decisions.

All descriptors, code bits and port partners are learned outputs. Prefix state is reset for each collision and has constant externally bounded length; it is not a global parser or recurrent episode state. Actual per-collision work is the neural forward work times the number of local emission decisions, plus pointer operations. The head's graph syntax supplies no Lam/App/Env/Copy/FV constructors. The finite hand witness roles are points in its code space, not a supplied vocabulary of candidate rules.

For a fixed W execute every categorical head by deterministic argmax with a fixed local tie rule. Thus `K_W` is a pure deterministic pair-local law. The network can emit codes absent from the raw input, which future collisions interpret through the same W. At fixed precision this is a finite but factorized code family. A newly emitted code is not automatically a new operation, and its executable meaning is still determined by the acquired W.

## 3. Actual formation update: sample a law, execute it coherently, select by terminal outcomes

Represent W by D finite bits. Use a mixture of product-Bernoulli search distributions

`q_psi(l,W) = pi_l * product_j p_lj^(W_j) * (1-p_lj)^(1-W_j)`.

The mixture retains alternative whole-law hypotheses; it is an optimization distribution outside the executed graph, not another input encoder. Decode the bits into the fixed quantized weight format. Search capacities, mixture size, precision, exploration floors and external trial limits must be fixed before a run. An initial distribution contains no witness role labels or correct table rows.

One iteration is explicit:

1. Sample a complete `(l,W)` once. Freeze W for the entire finite bank of episodes. Every occurrence of the same local input gets the same emission, including all recursively generated codes.
2. Initialize each episode from its raw chain+Start, run the complete-drain protocol, and retain actual local events, cost and terminal result. No successful parse or correct local rule is inferred merely from a long trace.
3. Compute F_B(W) using only the bank's terminal labels and measured costs. Store failures as failures.
4. Update psi by the ordinary likelihood-ratio estimator

   `grad_psi E_q[F_B(W)] = E_q[(F_B(W)-baseline_B) * grad_psi log q_psi(l,W)]`.

   The baseline depends on the bank/history but is independent of this sampled law. Keep exploration probabilities away from 0/1 while search is active. The finite weights, finite bank and external execution bounds make this a finite sum; no derivative through graph edits or quantization is asserted.
5. Retain distinct actual candidate W versions and their bank evidence; rescore incumbents and challengers on new common raw episodes before preferring one. An unfinished candidate may survive in the hypothesis population; no rule is declared correct just because it helps one example. All candidate generation, forward trials and search-distribution updates are charged.

**Estimator distinction:** the score here is `grad log q_psi(l,W)`, once per sampled law. It is not the sum of fresh per-event emitter log probabilities from stochastic v0. The execution head makes deterministic choices conditional on W. Repeating a rule one thousand times does not mean that rule's exploratory draw was made one thousand times. A local-key random lookup would be another algorithm and ordinary finite rule-table induction unless a separate generalizing kernel were specified; it is not silently inserted here.

### Why coherence matters, and what it does not prove

In a deliberately restricted comparison, suppose a successful protocol requires R independent local choices and invokes choice r exactly n_r times, each with correct probability p_r. Fresh independent event dithering executes that protocol with probability `product_r p_r^(n_r)`. Sampling each local choice once and keeping it fixed gives `product_r p_r`. This is a direct elementary comparison conditional on that factorization and the same required protocol. It explains why fresh graph-rewiring noise can destroy long recursive hypotheses before terminal feedback can evaluate them.

Our neural weight distribution does **not** imply independent rule choices or that exact probability formula. Weight changes couple many local inputs and generated codes; successful-kernel mass may be extremely small. The concrete benefit of whole-W sampling is only semantic coherence of each tested hypothesis. Parameter-space/posterior exploration is established machinery. Neural GPU/NRAM/NLI can also sample and hold whole weights/programs fixed; the comparison does not privilege this architecture over them or establish an acquisition-cost separation.

The organizational hypothesis left to test is different: a learned local law can create exact persistent communication/binding interfaces, and then reuse that same small law at new graph depths and interaction configurations. Search is over the organization of those interfaces, not only over a supplied instruction sequence. Whether this bias makes a useful law easier to acquire remains OPEN.

## 4. Substantive execution repair: local selective capture

Root's `root-environment-growth.md` shows that v1's ambient-environment tree copying can be exponential on a linear-size closed typed source. This is a decisive execution-cost attack on that witness, not an acquisition theorem. V2 admits a concrete local protocol that computes a lambda's free names from its passive syntax, projects the current environment to those names, and stores only that projected tree. No host free-variable oracle or supplied set primitive is added. The analysis/capture controllers below are **hand witness target rules**, which the same emitter would need to acquire from terminal correctness and cost feedback.

Define the passive set representation by Nil[] and `Cons_x[tail]`, with atom equality used only when a control agent carrying x meets Cons_y. Ret is the same negative one-value return controller as v1; all continuation/set/data nodes are positive. FV/Remove/Insert/Union/Project and the unpacking controls are negative. Every displayed row has at most four new agents and arity at most two.

### 4.1 Preserve the body and initiate the analysis through four local nodes

Replace v1's Eval/Lam row. For old boundaries body,e,k, emit

`body=D[bKeep,bScan]; bScan=FV[q]; q=CaptureStage_x[p,k]; p=Pair[bKeep,e]`.

The D rule is still the learned passive-tree copying protocol. Analysis consumes bScan; bKeep will become the closure body. CaptureStage waits for the FV result. Its auxiliary payload Pair preserves the current environment and body physically, without a global E_phi or a name-indexed store.

When `Ret[S] / CaptureStage_x[p,k]` fires, emit `p=CaptureUnpack_x[S,k]`. On that control meeting `Pair[body,e]`, emit

`S=Remove_x[q]; q=AfterRemove_x[p,k]; p=Pair[body,e]`.

Removing x from FV(body) supplies the free names of the lambda. `Ret[Sfree] / AfterRemove_x[p,k]` emits `p=ProjectUnpack_x[Sfree,k]`. On meeting `Pair[body,e]`, emit

`Sfree=Project[e,q]; q=FinishClosure_x[body,k]`.

Finally `Ret[eCaptured] / FinishClosure_x[body,k]` emits

`k=Ret[v]; v=Closure_x[body,eCaptured]`.

These are finite local phases of one emitter. Capture is not a primitive supplied to learning.

### 4.2 Actual free-name analysis and finite-set operations

The intended mathematical recursion is `FV(Atom)=empty`, `FV(Var_x)={x}`, `FV(Lam_x b)=remove(x,FV(b))`, `FV(App(f,a))=union(FV(f),FV(a))`. The following collisions execute that recursion using generated continuations:

| Active pair | Replacement |
|---|---|
| `FV[k] / Atom_c[]` | `k=Ret[s]; s=Nil[]` |
| `FV[k] / Var_x[]` | `k=Ret[s]; s=Cons_x[n]; n=Nil[]` |
| `FV[k] / Lam_x[body]` | `body=FV[q]; q=FVRemove_x[k]` |
| `Ret[S] / FVRemove_x[k]` | `S=Remove_x[k]` |
| `FV[k] / App[f,a]` | `f=FV[q]; q=FVLeft[a,k]` |
| `Ret[Sf] / FVLeft[a,k]` | `a=FV[q]; q=FVRight[Sf,k]` |
| `Ret[Sa] / FVRight[Sf,k]` | `Sf=Union[Sa,k]` |
| `Remove_x[k] / Nil[]` | `k=Ret[n]; n=Nil[]` |
| `Remove_x[k] / Cons_y[t]`, x=y | `t=Remove_x[k]` |
| `Remove_x[k] / Cons_y[t]`, x!=y | `t=Remove_x[q]; q=ReCons_y[k]` |
| `Ret[S] / ReCons_y[k]` | `k=Ret[c]; c=Cons_y[S]` |
| `Union[S2,k] / Nil[]` | `k=Ret[S2]` |
| `Union[S2,k] / Cons_x[t]` | `S2=Insert_x[q]; q=UnionNext[t,k]` |
| `Ret[S2new] / UnionNext[t,k]` | `t=Union[S2new,k]` |
| `Insert_x[k] / Nil[]` | `k=Ret[c]; c=Cons_x[n]; n=Nil[]` |
| `Insert_x[k] / Cons_y[t]`, x=y | `k=Ret[c]; c=Cons_y[t]` |
| `Insert_x[k] / Cons_y[t]`, x!=y | `t=Insert_x[q]; q=ReCons_y[k]` |

These rules produce a unique-name list; insertion order is unimportant for capture because each name's nearest value is looked up before the old environment is discarded. The analysis copies names exactly through the nominal selector API. It does not read an atom's spelling or infer scope from numerical ordering.

### 4.3 Project the actual current environment, including captured values

Project consumes a unique-name set S and an owned environment tree e, returning a new environment containing only the requested nearest bindings. It uses v1's actual Lookup and D, not a host lookup/copy routine:

| Active pair | Replacement |
|---|---|
| `Project[e,k] / Nil[]` | `k=Ret[n]; n=Empty[]; e=Erase[]` |
| `Project[e,k] / Cons_x[t]` | `e=D[eFind,eRest]; eFind=Lookup_x[q]; q=ProjectFound_x[p,k]; p=Pair[t,eRest]` |
| `Ret[value] / ProjectFound_x[p,k]` | `p=ProjectNext_x[value,k]` |
| `ProjectNext_x[value,k] / Pair[t,eRest]` | `t=Project[eRest,q]; q=ProjectTail_x[value,k]` |
| `Ret[eTail] / ProjectTail_x[value,k]` | `k=Ret[n]; n=Frame_x[value,eTail]` |

The four-node Project/Cons row duplicates the environment, looks up one requested name on one copy, and preserves the other for the remaining requests. On a closed source, every free name required at closure creation has a binding; Lookup's nearest-frame rule preserves shadowing. This may be expensive: m free names can require O(m*|e|) copied data before accounting for captured value sizes. Arbitrarily shared or cyclic values remain outside the tree-copy invariant.

### 4.4 Exact repair of the checked growth family, with a remaining conventional gap

Use root's linear-size family `let f1=(lambda u1.C a) in let f2=(lambda u2.V f1) in ... let fk=(lambda uk.V f_(k-1)) in C a`, where let is raw application syntax, not a host construct.

Every function `lambda fi.remaining_body` is closed relative to its ambient environment: its only potentially needed previous name is its own bound fi. Thus selective capture gives it Empty. F1's argument closure stores an Atom body and Empty, so `|F1|=3`. For i>=2, Fi stores a Var body and just one binding of f_(i-1), whose owned value is a copy of F_(i-1):

`|Fi| = |F_(i-1)| + 4 = 4i-1`.

After applying the i-th let binder, its parent environment is Empty (the function's actual captured environment), so

`Ei = Frame_fi[Fi,Empty]`, `Ni=4i+1`.

This replaces v1's `Ni=4*2^i-3` on that actual family. The v1 Eval/App copying events now sum to `sum_(i=1..k)(4(i-1)+1)=2k^2-k`. Additional body copying, FV analysis, projection and complete erasure must be paid. Here relevant free-name sets have bounded size, suffix bodies have O(k-i) nodes, and those extra traversals also total O(k^2). Thus this explicit repaired tree protocol removes the exponential total-work growth on the family. It is **not** an O(k) implementation: the conventional shared immutable heap in root's attack still has the stronger execution bound. General selective capture does not solve arbitrary sharing, and this family alone is not an acquisition benchmark because its output is the constant a.

This is a hand derivation from the new rows, not an executed result or proof that cost feedback discovers them. The meaningful formation burden is now concrete: acquire a data-relevance analysis and the operational capture discipline jointly, retaining correct scope while avoiding ambient-environment replication. More fuel does not substitute for that acquisition.

## 5. Stronger concrete proposal operator: counterfactual law edits using own traces

Whole-weight sampling gives stable hypotheses but may put almost no mass on a coherent multi-role interpreter. A second explicit proposal operator uses the learner's **own** executed local events to aim its weight search. It does not receive correct rule labels.

For a retained candidate W and a failed/partially successful raw bank:

1. Group the recorded events by their observable local input descriptor (two codes/arities and nominal presence/equality pattern). Choose one visited group; a simple initial selector is uniform over groups. This is a metered analysis of a saved training trace, not a runtime result cache or a supplied semantic parser.
2. Propose a valid alternative emission R' using only the generic graph-emission API. Examples: switch partners of two wires, change one new code bit or an atom-reference selector, delete/add permitted local agents and resample a legal matching. No proposal is named “make a closure,” “copy a function,” “find a free variable” or “use nearest binding.” Its atom selectors refer only to the input pair's available references.
3. Obtain W' by a fixed, metered number S of ordinary neural optimization steps on

   `L_patch(W') = CE(emission_R' | local_input_x*) + kappa*sum_x KL(p_W(.|x)||p_W'(.|x)) + tau*||W'-W||^2`.

   The anchor x's are selected **own successful-trace** inputs. Remove conflicting copies of x* from this preservation term. They are not correctness labels: they preserve the incumbent's emitted behavior provisionally. CE includes R's local descriptor/code/reference/matching decisions. Quantize into the same finite W format after the declared distillation procedure. Failure to realize the proposed emission is a recorded failed proposal, not a passed compile.
4. Reinitialize every bank episode from the same raw input chain and execute the complete new W' law. A patch changes every recurrence of the affected local behavior, not just the chosen occurrence. It can also affect unseen inputs through the shared neural weights; nothing silently restricts that interference.
5. Terminal F_B(W') decides its value. Rescore old raw episodes as well as the new challenge. Successful-trace anchors are soft/revisable and may preserve a bad representation; only external terminal feedback and measured costs decide admission. Keep alternative whole-law candidates within the external population budget so a multi-rule construction can traverse temporarily unhelpful steps; do not assert that greedy single edits suffice.

This operator creates internal supervised optimization targets from *hypothesized edits*, not from a teacher's correct parse/reduction. The semantic test remains the full terminal replay. Its cost can be large: every patch pays distillation, quantization, all replayed episodes, failed attempts and retained anchors. No local counterfactual is claimed free or exact without replay.

If a learned proposal distribution rho chooses the group and graph edit, its own score-function update uses `grad_rho log q_rho(group,edit | W,H)` times the measured resulting fitness, conditional on the incumbent W and the deterministic distiller. That is distinct from the whole-weight distribution's once-per-W score. Mixing the two proposal routes requires recording which route generated a candidate; applying `grad log q_psi(W')` to an unscored distillation output would be wrong. A fixed uniform edit proposal needs no extra learned optimizer at all.

The algorithm thus has a meaningful but unverified acquisition route: small behavior-level hypotheses can be compiled into the shared neural law, tested globally and retained coherently. It does not search only a tiny prelisted matching set. It can introduce new codes and graph organization, but may still be too sparse/expensive to discover the joint parser/capture/copy discipline. No guarantee is hidden in the word “distillation.” Ordinary neural-guided graph-program induction is an appropriate description of this search until an actual organizational advantage is established.

## 6. A non-idempotent higher-order and scope discrimination

V1's constant `F=lambda y.x` does not extensionally require two uses; that limitation is preserved in the erratum. A stronger reader-level family uses Church booleans and a captured readout:

`T = lambda u. lambda v. u`,

`F = lambda u. lambda v. v`,

`N = lambda p. lambda u. lambda v. p v u`,

`H = lambda p. p x (C b)`.

T,F,N,H are explanatory abbreviations only. The actual initial chain contains their expanded raw L/A/V/C tokens, not supplied macros or primitives. N is non-idempotent on booleans: `N T -> F`, `N F -> T`. Let a,b,c be distinct ground atoms and consider the fully expanded raw version of

`P = (lambda x. (lambda h. (lambda f. (lambda x. h (f (f T))) (C c)) N) H) (C a)`.

H is formed in the outer x=a scope; at the call to h the caller's nearest x is c. The local closure protocol constructs H with captured binding x=a. It constructs two physical copies of the value N for f's two occurrences. The inner application gives `N T=F`, the outer gives `N F=T`, and H then evaluates `T a b=a`. Using the caller x would give c. Keeping only one N use gives `H F=b`. Thus this example simultaneously exposes a scope error and a one-use shortcut that v1's constant example could not distinguish.

Zero N uses also give a, so this one label still does not prove two executed applications. Vary the expanded use count k, initial Boolean B in {T,F}, Boolean transformation in {identity,N,constant-T,constant-F}, the captured atom pair and an independently shadowed caller binding. For N, the correct captured result alternates with parity and B. Raw forms, names and irrelevant syntax should vary so the test does not reduce to counting a reserved NOT token. These are prospective discriminations, not a frozen benchmark or claimed learned behavior.

Behavior alone cannot certify the intended copies. After a learner has actually formed an identifiable reusable function representation, a causal intervention replacing one actual N instance with an equal-interface identity instance should switch the P_{2,T} result from a to b; a sham replacement by another copy of N should preserve a. Redirecting only H's captured-x interface to c should change the relevant result to c. Such tests require finding the learner's real functional interfaces rather than supplying the witness labels to it. Matching state/cost and validating each intervention is further work. No intervention has been executed.

Selective capture matters within this family for a concrete reason: N and T are closed and should retain no caller environment, while H needs only x. Retaining every past scope is sufficient for correctness but can be prohibitively expensive. Dropping all environments makes some short cases cheap but fails capture. Terminal correctness together with measured construction/copy/cleanup cost can therefore select among genuinely different internal organizations. The hand FV protocol shows one viable target. It does not prove that search finds that target, that exact FV is the only useful organization, or that a learned alternative must look human-designed.

## 7. Curriculum, freezing, composition and retention — exact boundaries

An allowed curriculum selects **raw examples**, never correct local actions. It can begin with literal/identity/constant examples, then introduce closed higher-order transformations, shadowing/capture, repeated use and environments with irrelevant as well as relevant bindings. A source generator may construct well-typed tasks and labels externally; no type judgment, subtree boundary, free-name set or scope depth is an input to K. Both competitors get the identical example stream, label access and curriculum/search budget. A progression chosen by the human is still a designed bias and cannot be counted as discovered organization.

Only terminal equality/failure and total-cost fitness shape proposal selection. Complete-drain validity is a generic execution condition, not reward for “parser progress.” Own traces locate possible edits but do not say which step was wrong. Repeating old successful terminal examples supports continued acquisition; it is not a proof of correctness on all old programs. Every reset/replay/label query is charged.

For a finite evaluation freeze:

- Freeze the exact W bits, input initializer, nominal interface, scheduler, terminal observer, complete-drain rule and all external limits. Evaluate new raw programs with **no** weight/search updates.
- Composition means that this same frozen law constructs and executes newly nested programs/closures, including independently withheld combinations. It does not mean that a hand-constructed AST or correct closure is loaded for it.
- Within one correct frozen W, the witness's environment/copy induction gives reuse of generated closures. For an acquired W this must be established by actual behavior and mechanism evidence, not transferred from the handwritten table.
- Between learning phases, changing W can change all old code meanings. Keeping W_old permits replay/rollback of the old **whole law** and costs its storage. It does not show that W_new preserves old behavior, or that a closure made by W_old can interact correctly under W_new.
- Cross-law composition would require explicit version dispatch, semantics transport or cross-version rules. None is supplied or claimed here. The present study is about acquiring and reusing one operational organization; universal continual retention remains outside its demonstrated scope.

The broader Fundamental Question remains construction, composition and improvement of internal computational mechanisms. The lambda family is a demanding diagnostic for binding/copy/relevance organization, not the research goal itself. Turning this campaign into a parser/interpreter engineering exercise would be drift. A useful result must identify an acquisition or resource consequence of learned organization that survives strong conventional learners and the full formation cost.

## 8. Actual precedents and matched conventional alternatives

**Checked by this worker:** Plappert et al., *Parameter Space Noise for Exploration* (ICLR 2018), original §3 explicitly holds a sampled perturbed policy fixed throughout an episode. Therefore coherent whole-law exploration is known machinery. Sinot's token-passing originals cover the broad local eval/return/copy organization. Shao–Appel's original *Space-Efficient Closure Representations* (1994), pp.150–152, already distinguishes flat free-variable capture, linked closures, copying costs and safe sharing. The selective-capture reconstruction above is not new closure theory. The reading file records the exact passages and unavailable/garbled versions.

**Supporting original readings:** the actual conventional-rival artifact enters Neural GPU, NRAM, NPS and NTM bodies. Root separately entered NLI (Macfarlane et al., arXiv:2604.18907v1), §§2–3.4, equations (1)–(3) and Algorithm 1. These are source reports with their own provenance, not this worker's independent full-paper verification.

- Neural GPU already learns shared local neural execution over raw symbolic streams using final outputs, with a supplied fixed grid/CGRU update. It is a real architecture-level rival, not a “given evaluator” strawman.
- NRAM supplies READ/WRITE and arithmetic modules, then learns their control from I/O without scope labels. It can represent mutable addressed structures and immutable sharing. Exact links do not establish a separation from this comparator.
- NLI jointly learns latent primitive tokens and a neural executor, with output-based program/primitive acquisition. Jointly learning primitives, interpretation and control is therefore not a sufficient distinction. The inspected organization remains sequential latent-token execution rather than generated pair-local physical communication.

The candidate's live hypothesis is **acquisition of exact mutable communication and activation topology**, including which retained bindings/values are physically carried into a new computation, by the shared local neural law. It may provide a useful search bias or a resource law; there is no established advantage yet. A conventional program can implement the hand protocol, and a competent NRAM learner can use sharing. Neither is a proof of equal acquisition costs; neither may be excluded from comparison.

**Nominal interface matching is a real prerequisite.** V2 fixes opaque atoms plus copy/equality/reference selection. Original Neural GPU uses finite token embeddings and original NRAM exposes numerical values/addresses. A nominalized comparator must be declared as a variant: preserve its CGRU or fixed READ/WRITE/arithmetic organization, provide the same opaque-atom registers/equality/copy access, and prevent arithmetic/spelling access on atom identities. This must not add the new graph emitter or a scope oracle. Conversely, comparing numerical IDs to hidden opaque references without explaining the information difference would not be “same information.” This adaptation is not historical coverage proved by the original papers, and its exact implementation remains part of freezing a fair study.

The terminal-feedback choice also matters: if candidate selection receives only scalar exact-match/failure and cost, do not give a competitor teacher traces or token cross-entropy for free. If terminal target gradients are allowed, make the same terminal supervision available to both. Match persistent parameter bits, working precision, peak state including pointers/queues/nominal references, task queries, training/search work and failed tuning. Charge all dense grid updates or NRAM module work, not just nominal recurrent steps. The graph machine's copied environments and static analysis are fully charged too.

## 9. What this construction achieves, and the next scientific decision

The actual new work of this pass is inspectable: a finite local neural-emission parameterization; a correct once-per-law search estimator; a terminal-only counterfactual neural-law edit mechanism; a complete-drain/nominal information contract; non-idempotent repeated-use+capture discriminations; and actual local FV/set/projection transitions that repair the checked exponential family to polynomial work. These are **constructed methods/derivations**, not empirical findings or priority claims. They make the acquisition burden concrete and expose its strongest failure modes.

The hardest unresolved premise is still substantial: a bounded, terminal-only search can acquire a useful stable interaction law and a selective retention/copy organization at an acceptable **total** cost. The proposed law can express it; the hand witness shows how it executes; known optimizer machinery supplies a real update. None of these implies discoverability. Multi-rule coordination, bad successful-trace anchors, atom-selection shortcuts, graph-generation failures and residual owned-tree copying can defeat it. If matched Neural GPU/NRAM/NLI acquires the same capabilities more economically, this architecture may be an inferior organization despite its exact wiring.

**Current judgment: CONTINUE constructive research / OPEN acquisition, no experiment admitted by this worker.** There is now enough explicit content for an independent supervisor to judge a finite acquisition comparison. Do not spend compute showing only that the handwritten interpreter works or that three port matchings can be identified. The next single decision is whether one finite, genuinely discriminating acquisition study can retain scope, non-idempotent higher-order reuse and meaningful relevance/copy costs while giving the actual conventional rival identical terminal information and an honest total budget. If that cannot be designed without a supplied semantic scaffold, re-anchor before training.

`CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

- Stable law exploration has a once-per-law score, with the conditional p^R/p^N distinction. | DERIVED / existing optimizer principle | §§2–3 and checked Plappert original | no neural success-mass bound; rivals may use the same exploration | per-event score cannot be reused for whole-W draws | freeze an actual optimizer only in a future study.
- The checked growth family admits a local selective-capture protocol with Ni=4i+1 and O(k²) total work. | HAND RECONSTRUCTION, unexecuted | full FV/set/Project port rows and recurrence in §4 | independent wiring/cost audit; general sharing unsolved | v1 exponential owned-environment copy | retain as a paid target, not efficiency novelty.
- Repeated function use and capture can be tested together without the constant-F shortcut. | PROSPECTIVE DISCRIMINATION | Church N/B/H family in §6 | labels alone do not prove actual clones; no run | v1 F was constant/idempotent | freeze causal interfaces only after actual acquisition.
- This architecture makes new internal operational organization easier to acquire than conventional neural learners. | HYPOTHESIS | explicit formation mechanism and actual rivals | no acquisition evidence, fair nominal baseline not implemented | primitive/executor novelty and generic simulation veto both rejected | independent decision on one discriminating finite study.
