# Post-freeze comparison v1 — a worked transfer consequence and its pointer rival

**Judgment: PARK the specific v0 claim that coherent local-table search plus exact links is, by itself, an important new acquisition mechanism. REFRAME it as an untested architectural prior within graph-program induction.** This is not a claim that exact mutable communication cannot matter, not a priority verdict on an identical historical K, and not a decision to finish MAIN. The actual pair-local architecture remains OPEN. The present construction does not warrant another interpreter repair or a training commitment.

This comparison follows the substantive independent freeze. `candidate-v0.md` SHA-256 is `6123d5184d655651dc3b0d3f9f122f61780200a785a6fa7478aa9628c8b151d2`; frozen `reading.md` SHA-256 is `8da90fcefc6f28ac92b58b219cfafef6973137e4c0d6b9fecb9c0a92a7a8560b`. Both files are preserved byte-for-byte. Root authorized comparison afterward. No experiment, training, installation, or model override occurred. Provider-effective mode is unknown.

## 1. What the actual comparison changed

Read the complete actual `acquisition-v2.md`, `raw-stream-v1.md`, `root-acquisition-value-check.md`, `selective-capture-cost-audit.md`, review017, root's source-reading file, and sealed `candidates/R/candidate-v1.md`. An initial batched tool result was truncated; the affected files were then read individually in full, including the remaining acquisition-v2 sections. I also entered NLI's original method §§2–3.4, Eq. (1)–(3), and Algorithm 1. New source details are in `comparison-reading.md`.

V0 and v2 are different concrete algorithms but converge scientifically:

- V0 samples descriptor-indexed table actions independently conditional on a shared neural proposal and distills weighted tables. Its action realizations remain independent across distinct descriptors; neural sharing of their probabilities does not make them a common neural operational law. A huge local table is implicit even when lazily sampled.
- V2 samples a complete finite neural W, so its deterministic actions at different descriptors are coupled by W. Its trace-edit route proposes a local emission, distills it with successful-trace anchors, and judges the entire resulting law by terminal replay.
- Both are ordinary structured search over executable graph programs with endpoint selection, coherent repeated use, and neural proposal/behavioral distillation. V2 already addresses the useful coherence observation in v0. V0 supplies no demonstrated formation advantage over v2's representation of the search coordinates.

The support and association-memory arguments in v0 survive. Their negative scope is essential: a bounded-support summary cannot replace arbitrary identity relationships; an exact pointer heap or an input-sized grid is not such a summary. They do not select the graph architecture over those rivals. Confluence removes schedule choice as a completed-execution nuisance, but an ordinary sequential pointer program also need not learn a scheduler.

## 2. Worked interface that could actually form from allowed feedback

Use the existing raw grammar, not a new host API:

`e ::= C atom | V name | L name e | A e e`.

The initializer supplies only its linear token chain and Start. For explanation, abbreviate the following expanded source expressions:

`T = lambda u. lambda v. u`

`F = lambda u. lambda v. v`

`N = lambda p. lambda u. lambda v. p v u`

`H = lambda p. p x (C b)`.

No T/F/N/H token, host Boolean, call instruction, parse tree, closure object, or binding link is supplied. These reader abbreviations are expanded to A/L/V/C tokens. Atoms a,b,c are distinct. A pretraining bank may include two kinds of ordinary endpoint examples:

1. Closed transformed-readout examples, expanded versions of `(N B) (C a) (C b)`, for B=T and B=F. Their labels are b and a respectively. Also vary identity and constant transformations so one generic atom-selection or parity shortcut cannot satisfy the whole bank.
2. Capture examples, expanded versions of `(lambda x. (lambda h. (lambda x. h B) (C c)) H) (C a)`, for B=T and B=F. Their labels are a and b. Vary shadowed names, names of unused parameters, lexical depth, and ground atoms. A dynamic-scope answer c is wrong in the T case.

The external generator computes only final labels; those labels and full costs are the allowed feedback. The graph learner can search whole laws or hypothesized local edits under that feedback. There is no supplied correct interface target in the loss. The batch is an acquisition opportunity, not a proof that the desired mechanism is identified: many programs agree with it, and all failed searches count.

One useful organization that endpoint selection **could** retain is:

`function value = passive body + bound-name reference + captured-environment link`;

`application = current argument link + return-continuation link`;

`return = value link delivered to the waiting continuation`.

The current hand witness makes all functions use a single Closure/Apply boundary, not a separate N-to-H cross-rule. Its closure parent is the environment at formation. Its call allocates an argument frame onto that parent, rather than onto the caller's shadowed environment. Codes for these roles and all connecting rewrites must be acquired; the words above do not supply them to learning.

This is the exact formation burden. Endpoint examples can select between capture and caller scope, and between N and identity/constant behavior. They do not label which wire should be the captured parent or identify a unique internal program. A future claim that this interface formed must locate the acquired interface in actual traces and test it; one cannot rename a latent node Closure and import the hand invariant.

## 3. What changes on a genuinely new composition

Withhold the joint family

`P_(k,B) = (lambda x.
              (lambda h.
                (lambda n.
                  (lambda x. h (n^k B)) (C c)) N) H) (C a)`.

Here `n^k B` means k nested raw applications `n (n (... B))`, expanded before tokenization. It is not a supplied POWER/REPEAT operation. The input grows with k. In pretraining, the N transformation and H's captured readout occurred separately; H after repeated N and their nesting inside a shadowed caller are withheld jointly. The correct label is a when `N^k B=T` and b when `N^k B=F`. For example:

| Initial B | k=1 | k=2 | k=3 |
|---|---|---|---|
| T | b | a | b |
| F | a | b | a |

All entries remain independent of c. Include other transformations and varied source structures in an actual held-out family; these six labels alone do not certify a mechanism or prevent every shortcut.

**Conditional transfer consequence.** If the general call/return and lexical-parent organization above has actually formed, then increasing k introduces more instances of known interfaces, rather than a new semantic dispatch between N and H. Parser frames delimit new subexpressions; each application creates an existing kind of waiting continuation; each function value meets the same Apply interface; each result returns along a stored continuation wire. H's body uses its retained x=a relationship even though the caller has x=c. For a correct common protocol, this uses the same finite descriptor cases as the component behaviors. No new local table entry is required merely because the nesting is deeper or H now consumes N's result.

This is a nontrivial capability consequence: the body, environment, and continuation graph transports the input-dependent context, and the neural law need not store one policy per depth or one N-to-H adapter. It is a **conditional deduction about a formed interface**, not a claim that the finite bank forces that interface, that all acquired laws have it, or that there are no new descriptors in an arbitrary learned representation.

Closed source definitions N and H have bounded bodies, but that does **not** make all values produced by repeated N bounded. Under call-by-value, `N B` returns a closure for `lambda u. lambda v. B v u`, retaining B through p. Repeating N therefore builds a single-predecessor captured-value chain. If C_i is its owned passive size, the displayed witness has `C_i=C_(i-1)+c` for a fixed positive constructor constant c. At the i-th such closure formation, the actual Project/Cons row duplicates an environment containing the complete owned predecessor C_(i-1) before Lookup returns it. Thus these captures alone pay `sum_i Omega(C_(i-1))=Omega(k^2)` D/constructor collisions. No complete upper event count is asserted here. A previous working intuition that closed N/H implied O(k) owned execution would have missed this deferred captured chain; it is not used as evidence. The general Fibonacci overlap attack remains valid and is not needed for this smaller quadratic lower cost.

## 4. A concrete exact-pointer learner removes the same decisions

The proper rival is not a controller that receives a fresh list of all m live addresses and must guess among them on each call. Original NRAM already selects arguments from a fixed number of registers and prior gate outputs, and READ follows a selected pointer. A deep heap therefore need not expand the controller's action alphabet. Calling pointer selection depth-indexed would impose a false burden.

Use an explicitly declared nominalized, discretized NRAM-like variant. It has a fixed number R of registers, a bounded recurrent neural controller, variable-size addressed storage, and generic READ, WRITE, equality, copy, and address-allocation arithmetic. Ordinary addresses may be numeric; opaque task atoms have an equality/copy-only payload path and cannot be decoded by arithmetic. The atom adapter is an information-matching variant, not a claim about the original NRAM implementation. Uniform fixed-size memory records are permissible substrate layout; record tags and field roles are not supplied as Closure/Frame/Continuation semantics.

Its acquisition process samples/fixes whole controller weights or microprogram choices for a task batch, executes from the same raw chain, scores the same terminal labels and correctness-dominant total costs, and may propose/distill changes located in its own unsuccessful traces. The available modules remain generic READ/WRITE/equality/copy, not the new pair-local graph emitter. All neural work, register/gate choices, allocation, replay, failed search and exact memory storage count. This is a concrete adaptation of the existing pointer-controller organization with a permitted ordinary optimizer, not an invented historical implementation of K.

For explanatory reconstruction only, a successful learned pointer organization can use five named registers `expr, env, cont, val, scratch` plus finite controller state. Heap records can hold a tag and bounded fields. A learned parse loop builds syntax records and a parser stack from raw tokens. A learned execution loop can implement the following constant-size record processes:

- On an application, save the argument pointer, current environment pointer and return pointer in a continuation record, and set `expr` to the function body-expression pointer.
- On a lambda, return a record containing its body pointer, name reference and the current environment pointer.
- On a variable, follow environment records and use equality to find the nearest matching name.
- On applying a returned closure, allocate a frame for the argument whose parent comes from that closure's saved field. Set `expr` to the closure body pointer and retain the return record.
- On returning a value, READ the current continuation tag and its fields, update a constant set of registers, and continue.

Those are a hand representability witness, **not rules supplied to this learner** and not evidence of a trained RAM. They show where its subsequent decisions reside. Each step selects among a fixed number of register/gate sources and follows stored handles. Pointer bits carry the identity of the current environment/continuation; there is no new learned m-way address classifier as nesting increases. The controller must acquire field layout, parsing, lexical-parent choice and return discipline, just as the graph law must acquire codes and boundary routing.

After that organization forms, the same controller processes `P_(k,B)` without a new N-to-H adapter or depth-specific rule. It follows the heap fields just as the graph follows its wires. It can share immutable bodies and environments; the graph's owned-tree witness cannot be granted cheaper sharing without changing its acquired law. On this family the pointer process uses O(k) closure/frame/continuation construction and selected-call work with O(k) records: each returned N-composition closure adds a record pointing to its predecessor rather than unfolding it, and the final readout follows one predecessor per N. A straightforward collector visits its actual retained records; any collector work counts, and it cannot complete an unevaluated source computation. This favors the heap over the current owned-copy witness on execution cost; it still does not compare the learners' acquisition cost.

**What remains empirically different:** the graph prior permits only a bounded boundary matching at each interaction; RAM permits a bounded circuit of register-source choices and READ/WRITE effects. Their search geometries, error modes, training relaxations, representation constants, and useful-law probability mass can differ. We have no result that their acquisition costs match. But the worked transfer removes the claimed necessary burden of new context-specific routing decisions from **both**. That is enough to reject that argument for a graph-specific advantage, not enough to rule out an actual advantage.

## 5. Why the tempting formation revisions do not yet rescue the main claim

I considered three substantive routes rather than only renaming episodic tables:

**Preserve a learned interface and search only extensions.** Freeze old table entries or immutable learned modules and assign new codes to new operations. This can remove old semantic decisions from future acquisition. It also changes v0's shared neural fitting problem: distillation must demonstrably preserve all frozen entries, or an exact dispatch/store must become part of the runtime. Sealed R v1 already states a classical noninterfering extension discipline for retained rewrite modules. A pointer/program learner can freeze old subroutines and learn new callers. This is a real retention mechanism, but this pass supplies no additional graph-specific acquisition consequence.

**Use counterexamples to split overloaded contexts.** The nominal source shows why a failed suffix can reveal support or symmetry that the representation omitted. For the growing graph, however, the endpoint failure does not identify which internal contexts should be split, and there is no equivalence oracle or frozen finite residual-state bound. Graph codes and continuation links could be learned through such splitting, but treating the desired split as a training target would inject the answer. V2's trace-edit/replay already gives an honest, generic route. I did not derive a new constructive source of counterfactual truth.

**Quotient exploration by independent local execution.** Deterministic confluence avoids exploring many schedules. A fixed sequential RAM controller also executes one schedule without learning alternatives, so this does not yet change its acquisition burden. Turning partial-order reuse into a training acceleration could still be interesting, but it must count trace matching/cache costs and is not presently a demonstrated principle of operation formation.

NLI independently narrows the novelty claim. Its checked method learns reusable latent tokens and a recurrent neural executor from I/O, with leave-one-out output training and latent search. [Original method](https://arxiv.org/html/2604.18907v1). It does not provide the generated physical graph organization. It also operates in a different specification-to-program setup, so its published behavior is not an equal-information raw-lambda acquisition result. It suffices to block primitive discovery plus sequential reuse, by themselves, as the residual contribution.

R v1 was read as an actual construction with explicit acquisition and rival boundaries. I do not claim its finite probe learned rewrite rules or executed CEGIS acquisition; its own artifact explicitly says otherwise. Its hand-selected stack laws and consistency checker do not provide missing evidence for this graph learner.

## 6. Decision for this formulation

The worked case identifies an interface that endpoint learning could select, a genuine new-composition consequence, and the subsequent decisions it removes. It also shows an actual pointer organization removing those same decisions with constant-sized local control. My original v0 proposal therefore supplies a serious executable search specification and a clarified memory boundary, but **no remaining specific important acquisition prediction attributable to exact pair-local mutable communication rather than acquired pointer/program organization generally**.

That is why I recommend parking **v0's formation mechanism as the next primary implementation**, not parking exact topology as a universal direction. A future candidate may supply a different causal source: an actual task family where the bounded boundary prior changes acquisition at full cost, a nontrivial learning law tied to causal structure, or a new formation phenomenon. None is supplied by calling this interface autonomous or adding another closure optimization. This pass does not authorize generating or testing such a candidate automatically.

The narrower conditional prediction survives: if either learner acquires the common closure/return interface, it should handle the withheld H-after-repeated-N compositions without relearning an adapter. That can support a worthwhile capability/mechanism result after actual acquisition. It is shared with program/pointer learning and is not, by itself, the paradigm-level principle sought in MAIN.

`CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

| Claim | Status | Evidence/artifact | Weakest remaining link | Failed branch | Next single decision |
|---|---|---|---|---|---|
| The independent v0 supplies a different important formation mechanism from v2 | NOT ESTABLISHED | actual table-search vs whole-W/edit-distill comparison | no acquisition evidence; ordinary graph-program induction | coherence renamed as new learning | retain v0 as reference, park its primary implementation |
| A formed lexical call/return interface removes new depth- and N/H-specific rules | CONDITIONAL RECONSTRUCTION | explicit raw-input pretraining/test family and interface trace | actual interface not learned or identified | endpoint labels treated as unique mechanism labels | use only as a shared mechanism prediction |
| Exact-pointer learning necessarily faces new m-way routing choices at depth m | REJECTED ARGUMENT | fixed-register READ/WRITE process with stored env/continuation handles | acquisition-cost differences remain unknown | pointer memory conflated with global address guessing | do not use this contrast to justify graph training |
| Local support/association bounds exclude flexible RAM or an input-sized grid | FALSE APPLICATION | v0 proof assumptions and explicit pointer storage | none of those rivals has fixed global support | local orbit finiteness mistaken for finite global context | preserve restricted theorem only |
| Exact mutable pair-local organization may still change acquisition costs | OPEN ARCHITECTURAL HYPOTHESIS | distinct generic action/representation priors | no selected important comparative prediction | universality veto and identical-K prior-coverage claim both avoided | root decides whether another independently grounded mechanism merits allocation |
