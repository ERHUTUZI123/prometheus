# CD v1 — construct the editable state before constructing its repair law

**Judgment: a concrete bridge works on a completely observed finite machine, but this realization becomes state assignment plus circuit synthesis; its equally informed frozen-module/state-adapter baseline is identical.** The repair removes a separate omniscient learned critic and supplied semantic field names. It does not remove the need for old executable behavior and new-task feedback. The substantial remaining problem is learning a useful, incomplete causal interface in a neural circuit, rather than renaming its full state.

AGENT PROPOSAL / RECONSTRUCTION. Inherited current user-authorized Ultra route; no exposed backend/model identifier or compute telemetry; no Pro claim. Original C/D frozen files remain unchanged. Only the instructed C/D artifacts, the common contract/role sections, infrastructure, and the original sources listed in sources/CD/reading.md were read. No A/B/E/F or old architecture report was inspected. No nested agents.

## 1. Reconstruct the two actual mechanisms

**C:** execution is a recurrent elementary circuit. A learned trace energy, a circuit-edit proposer, retained teacher executions, and task loss guide bounded reconstruction. Previously acquired computations may change their implementations during composition while their tested boundary relations remain active. The hardest step is making a compact energy discriminate wrong computation rather than copying the whole old solver or only accepting short familiar traces. Its counterexample repair temporarily executes the already learned operation to find false acceptances; this is a real information source, with a cost.

**D:** Get identifies a view; Put makes an edit take effect in source state; a validity domain specifies promises. GetPut and PutGet establish coherence, not usefulness. They neither identify semantic fields nor prescribe task behavior. Its executable lens lifting/composition has established precedent. Its probe supplied the central run/count schema and decoder scaffolding. The interesting unbuilt step is joint construction of the schema, its interventions and its implementation from downstream raw feedback.

The conflict is precise. C needs semantic distinctions but risks embedding the solution in its critic. D supplies executable edits, but no facts about which edits preserve a computation. Simply adding a lens loss to C's energy cannot repair either gap: every bijective recoding satisfies those losses, including task-useless ones.

## 2. Change the retained object, learning target, and construction step

V1 retains an **editable causal interface** and finite executed evidence, not a learned scalar energy. The interface is a proposed encoder and executable state editor, plus the intervention footprints actually measured for existing operations. Field names and task-specific types are absent. A generic finite-product/tree grammar is still a designed inductive bias.

Let a source operation have executable state s, action/input a, transition F_a(s), and boundary output R(s). A proposed interface is

    A = (Get_g, Put_p, domain D, field product Z, local programs f_a, readout r, evidence T).

Get maps s to z. Put(s,z*) changes the represented fields and preserves its declared complement. In the executed finite special case g is bijective, Z is an induced product of finite enums, and Put(s,z*) = g^-1(z*). The source argument is unnecessary for that lossless special case. A partial neural interface would retain a complement and must check when different complements change future boundary consequences; that part is unbuilt.

An edit of one discovered field is the actual intervention

    I_j^v(s) = Put(s, replace_j(Get(s), v)).

No named instruction such as memory-write, run-length-edit or parity-toggle is supplied to the learner. It initially has generic scalar/finite-enum field replacement. The raw result is executed. An illegal or unreachable edited state is a counterexample to the proposed interface, not evidence that a new type exists.

The construction state is

    M = (source teacher/version, candidate graph, g,p,Z, local code library,
         T of source/counterfactual executions, proposal parameters, metered queue).

One construction transition is explicit:

1. Propose a finite state partition/codebook or a bounded Get/Put program edit. Neural q_theta may rank proposals; the actual check enumerates. Reject invalid round trips and invalid edits on the declared finite domain.
2. Execute the already learned source on actual and field-edited source states. Store (s,a,F_a(s),R(s)) and the charged intervention calls. Do not invent a symbolic high-level causal teacher.
3. Select an interface using measured intervention locality, implementation/storage cost, and task behavior. In the finite check, minimize total bit width of fields changed by each action, over every source state. This is an explicit, strong locality preference, not a discovered universal rule.
4. Encode an executed next-state target z* = Get(F_a(s)). For tentative circuit G, compute field residuals e_j = 1[z'_j != z*_j] (or a squared residual in a continuous version). This tells construction which editable field disagrees, instead of giving only C's scalar boundary-energy error. Synthesize or fit a replacement transition expression only for a failing field, using the same raw examples and primitive grammar. All proposed expressions and teacher calls are charged. A neural weight version would differentiate these residuals through the proposed circuit/encoder, but was not built.
5. Train any newly required transition or consumer from **new-task** raw feedback. Old interface laws can preserve old behavior and localize repairs; they cannot dictate the intended new computation. Check old behavior after the edit. Retain the realized graph and codecs; discard the source teacher and all checkers from normal execution.

In a general neural variant the explicit objective would include lens error, prediction/transition error on executed targets, boundary task error, interface/program description length, changed-field width, and all query/construction costs. Jointly learned g,p,f without fixed executed targets has a recoding/collusion ambiguity. Loss terms alone do not provide semantics; the executed source targets and downstream feedback do.

The retained finite T is a replay specification, not an asserted complete learned semantic law. In the small probe it covers the entire eight-state domain and is therefore already a complete finite behavioral description. That is both why the check is decisive there and why it is a weak model of neural discovery at scale.

## 3. The hardest discovery step is exposed rather than delegated

The finite implementation discovers the editable schema by searching codebooks from opaque source identifiers into generic products (8), (2,4), and (2,2,2). It tries 80,641 candidates for each source encoding. It receives no labels for the fields and does not receive the designer's generating variables. The three-field structure wins the predeclared locality objective over one- and two-field structures. This is genuine finite field/encoding inference; it is **not** synthesis of arbitrary types, object identity, variable-dimensional state, or a neural causal abstraction.

The designed apparatus supplies a known exhaustive eight-state domain, unrestricted source-state reset/read/write, complete old transition rows, a finite product grammar, a locality objective and Boolean gates. These are much stronger premises than learning an editable internal state in a large neural circuit. Replacing 8! enumeration with the phrase “the neural proposer finds a representation” would leave the central discovery problem unsolved.

For a useful incomplete interface, the learner must find states/interventions that are sufficient for the relevant future boundary behaviors while ignoring implementation detail. Too fine an interface recreates the old transition solver and restricts C's freedom to accommodate; too coarse an interface accepts destructive rewrites. No method here resolves that choice for rich neural mechanisms. This is the single most fragile step.

## 4. An executed construction, including the newly added operation

The experiment designer used hidden bits (m,p,n), opaque raw state IDs and four opaque actions. The source actions set m to zero, set m to one, toggle p and toggle n. Output is m XOR p. These semantic labels describe the generator only; learner functions receive numeric transition/output rows.

For seed 11, the induced Get codebook, indexed by raw state ID 0..7, is

    [0,1,3,5,4,2,7,6],

where a code is unpacked into three binary fields. Evaluation afterwards identifies those fields as (m,p,NOT n). The inference itself never sees that correspondence. Put decodes an edited tuple through the inverse codebook. All eight source states and all eight target tuples are valid.

The compiler generates these old executable transitions:

    event 0: (x0,x1,x2) -> (0,x1,x2)
    event 1: (x0,x1,x2) -> (1,x1,x2)
    event 2: (x0,x1,x2) -> (x0,NOT x1,x2)
    event 3: (x0,x1,x2) -> (x0,x1,NOT x2).

Eight new raw before/after task examples require a fifth event. From generic Boolean expression trees, the learner constructs

    event 4: (x0,x1,x2) -> (x0 XOR x1,x1,x2).

This transition is absent from the old four-action table; it is a changed executable state-transition law, not just a new label for an old raw state. XOR is an available elementary gate, not an installed event-4 instruction. The task examples cover all eight states, so subsequent execution is **not** evidence of unseen-state generalization.

Worked run: raw state 5 has discovered fields (0,1,0). Event 4 makes them (1,1,0), which Put materializes as raw state 7. The induced program, not the old teacher, performs this update. A frozen implementation of the four old events plus a new state adapter performs exactly the same update.

A synthetic XOR edge was also inserted into one old field output. State residuals identified that field, with four wrong rows. Resynthesizing that field restored the old program exactly. This is a useful diagnostic of a repair signal, but it uses complete old finite transition targets. The damage diagnostic was specified in implementation before the first run; it was additional to the earlier frozen protocol and is not called preregistered.

## 5. Executed result and honest cost

The pre-execution contract and hash are retained. One isolated execution used a 45-second wall cap; it completed in about 0.486 seconds. There was no failed run or criterion change after seeing results.

For **both** fixed opaque state encodings:

| Quantity | Executed observation |
|---|---:|
| Old transition / old output / new-event raw rows | 32 / 8 / 8 |
| Schema codebooks considered | 80,641 |
| Best locality score, shapes (8)/(2,4)/(2,2,2) | 12 / 6 / 4 |
| Binary codebooks attaining minimum | 48 |
| Checked transition rows after synthesis | 40 |
| Post-training sequence checks, lengths 0..4 | 6,248 |
| Post-training executed step checks | 23,440 |
| Candidate vs frozen-plus-adapter emitted code | identical |
| Candidate vs baseline gate-count sums | 4 vs 4; 6 vs 6 |

The 48 minimizing binary codebooks are consistent with field permutations/complements; only the chosen codebook was semantically audited. The code does not prove a general uniqueness theorem for all minimizers. Across both encodings the totals are 161,282 codebook candidates, 12,496 sequence checks and 46,880 step checks. The Boolean expression library considered 2,650 expressions once and represented 183 of the 256 three-input Boolean functions within the frozen seven-node tree bound. No coverage of all Boolean functions is claimed.

The gate count is a structural count of NOT/AND/OR/XOR nodes in emitted per-action/readout trees. It omits dispatch/multiplexing, registers, wiring, codecs and Boolean-node implementation costs; those omissions are identical for the two implementations and make this **not** a total runtime benchmark. The literal codebook has 24 payload bits before indexing/format overhead; storing its inverse separately doubles that payload. Replay contains the entire source transition table. These costs are not free. The 80,641-candidate discovery debt dominates this tiny computation; the lower locality score is not evidence of lower total discovery cost. No matched flat-search benchmark or neural proposer was run.

## 6. Strongest attack: the complete bridge is an existing-method reconstruction

The finite construction chooses a representation of symbolic machine states, then synthesizes Boolean next-state logic. This is substantively the classical **state-assignment and sequential-circuit-synthesis problem**, not merely something an ordinary computer could emulate. NOVA's primary passages explicitly describe symbolic state encodings, combinational next-state logic, and encoding-dependent circuit cost (sources/CD/reading.md). Our locality objective and exhaustive search differ from its PLA objective/algorithms; that does not create a new architectural principle.

At the neural end, interchange intervention training already uses aligned counterfactual state replacements to train neural causal structure. Its supplied high-level model provides the counterfactual labels. V1 attempts to infer a finite interface before using similar intervention consistency, but the complete finite transition table supplies the semantics. These checked precedents bound the claim; no novelty claim survives for the implemented core.

Most decisively, the honest frozen-old-functions-plus-adapter baseline generated **the same program**. Keeping every old function and adding the same new-field controller is sufficient in this example. Permitting state recoding/local repair does not force a computational interaction unavailable to that baseline. There is no measured advantage in subsequent construction or joint accommodation.

Finally, old causal laws do not imply a new useful composition. The executed adversarial extension replaces event 4's update by m <- m XOR p XOR n. It agrees with every old action/output observation and differs from the desired new event on four source states. Thus even perfect old editable-state discovery cannot select the intended extension without additional new-task information. This is an identifiability counterexample, not a claim that transfer is impossible.

## 7. What remains alive, and stop

**Narrow surviving result:** an induced editable state can turn prior executions into localized construction/repair targets without requiring a separately learned all-knowing critic or human-labelled field semantics. In the executed case the target local programs are as small as the corresponding computation; there is no demonstrated checker-versus-solver cost advantage. A finite replay table also carries the entire old solver's information. V1 avoids an oracle for the **new** task only by using actual new-task examples.

**Unresolved high-upside question:** can a neural learner infer an incomplete causal interface that is much cheaper than reproducing the source transition system, and use it to change a circuit's realization during composition at lower *total* cost than frozen modules with equally capable state/boundary adapters? It must not receive target variables, desired transition algebra, privileged state-surgery directions, or a high-level model secretly supplied by the researcher. A gain restricted to state recoding, replay, or a more helpful interface budget is insufficient.

Decision: **STOP this CD repair stage as an executed existing-method reconstruction with an informative baseline tie.** No larger neural benchmark is started. The central next decision is scientific prioritization of the incomplete-interface discovery problem, not another toy state-encoding experiment.
