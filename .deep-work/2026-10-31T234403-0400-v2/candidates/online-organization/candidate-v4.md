# Form procedures by cutting the live credit frontier

AGENT PROPOSAL / RECONSTRUCTION. Separate v4; v1–v3 unchanged. This is the primary support-closure formation construction requested by root. No experiment was run. The shared goal remains acquisition of computational mechanisms, not another test of the chain rule.

## Intended mechanism and important counterfactual

The learning pressure is the cost of the **executed transitive dependency closure of live values**, including their trainable parameter support. A network can have unchanged parameter count, hidden width and forward FLOPs but a very different exact online-credit cost. This pressure can therefore favor a computational organization invisible to a forward-compute penalty alone.

The constructor uses that information to propose a real code action: replace a contaminating live recurrent connection by an explicit call/return interface, move a return boundary, or rewire a bounded local computation. It then performs the paid finite consumer-learning challenge from v3 before retiring private state. Thus a prospective causal cut must preserve useful subsequent learning as well as current prediction.

Success would mean acquiring procedures whose returned objects support new composed computations and continuing plasticity while avoiding growth of the lifetime learning frontier. It would change the cost of building/improving larger neural mechanisms, not merely save inference activations. This is a concrete hypothesis, not a novelty or foundational-value result.

## 1. Exact support ledger on an executed program

For each live value instance v in the executed event graph, maintain a conservative parameter-owner support S(v). External fixed input values have empty support. If an executed instruction reads parent values v_1,...,v_d and trainable parameter owners B, propagate

    S(v_out)=B union S(v_1) union ... union S(v_d).         (1)

IDs here identify actual storage/versions; they are not hidden task labels. Repeated reads of one owner are counted once. A structural controller can use cardinalities, overlaps and program-role features without treating arbitrary owner numbering as a semantic signal.

For a specified exact sparse forward-sensitivity executor, the basic live-gradient storage proxy is

    M_support(t)=sum_{v live at t} |S(v)| * bytes_per_derivative.

Integrate this over actual event time, and add ordinary values, support-set bookkeeping, local tapes/restart inputs, parameter versions, optimizer memory, policy scores and any encoder derivatives. Track peak bytes and per-event derivative latency separately. A committed value remains live until actual consumption/release, not until a controller guesses there will be no future use.

This is a safe syntactic support bound, not a universal memory lower bound. Numerical cancellations can reduce the actual Jacobian; a factored AD representation can exploit common ancestry. Strong controls may choose sparse, factored, forward, reverse or checkpointed differentiation under the same allowed data/replay rules. The final resource score uses actual working state/work, not only equation (1)'s proxy.

Do not delete a trainable dependency because its weight or current gradient is zero. For a term a*x at a=0, the derivative in the trainable a direction is still x. Hard structural absence and a numerically zero plastic parameter are different. A gate/controller's own gradient state is also part of the ledger; hiding a large score/eligibility record would defeat the claimed saving.

## 2. Why forward work alone cannot represent this pressure

Take N scalar recurrent states, P=Nd trainable incoming-edge parameters, and identical scalar update/activation counts. Compare a globally strongly connected sparse ring-like graph with s disconnected equal-size sparse recurrent components. All weights remain eligible for native learning; no columns are permanently frozen.

After sufficient generic causal propagation, every global-ring state can depend on all P parameters, giving NP possible state-to-parameter entries. In the blocked graph, each state depends on only P/s parameters, giving NP/s entries. For N=4,d=1, a four-cycle can fill 16 entries, while two two-cycles fill 8, at the same four recurrent weights and per-round forward work. At shorter horizons, use the actual propagated supports rather than the asymptotic count.

This contrast proves only that equal forward counts need not mean equal credit-frontier cost. The two structures are NOT capacity-matched on all tasks and it is not a performance advantage. A task requiring continuous cross-component communication may defeat the blocked structure. Conversely, a global readout does not automatically force every recurrent state to share all parameter support unless that readout feeds back. These distinctions are precisely what the formation/probation rule must test.

Calls/returns permit a less permanent intervention than disconnecting weights. A private computation may exchange information through k committed output values and then retire N private rows; those k values can still have full parameter support P. Communication is retained through explicit objects rather than every private state remaining live. If those objects immediately feed back into all states and all remain live, support can spread globally again. No automatic saving is claimed.

## 3. A bounded support-contamination proposal rule

At a candidate execution boundary, identify a recurrent/message edge e that reads a value u with large support into a region whose other live inputs have smaller support. On the retained recent execution trace, compute a marginal contamination estimate

    Delta_e = estimated future byte-time with edge ancestry
              minus estimated byte-time with that ancestry cut.

This can be computed by replaying equation (1) on the existing trace, holding its control decisions fixed, or approximated by immediate union increments and measured remaining lifetimes. It is a proposal heuristic: a real cut can change numerical states, control decisions, task behavior and lifetimes, so the estimate cannot certify its benefit.

Choose one of the highest estimated contaminating edges/regions under a fixed proposal budget, using randomized exploration when tied/uncertain. Generate ONE bounded candidate surgery, rather than searching all partitions for free:

* **Per-call port cut:** make the source region a private frame; learn/propose return code C(h); replace ongoing cross-region reads by reads of a committed value; retire private state only at the selected return. The return code can use typed arithmetic/neural routines, rather than merely selecting existing coordinates.
* **Move a return:** change when a region emits its result, so a costly dependency is not imported into a long-lived private workspace earlier than needed, or so a completed workspace is retired sooner.
* **Local structural rewire:** reassign a bounded set of existing read edges, optionally preserving degree/parameter count for a diagnostic control. Preserve the actual parameter-owner identities and specify optimizer-state migration. Rewiring changes computation and is never presumed function-preserving.
* **Refactor a body/read set:** extract a new routine implementing useful returned functions with fewer genuine parameter reads, if it can actually be acquired from experience. A smaller learned implementation is not declared equivalent before task/learning validation.

These are event-/call-scoped organization actions, not necessarily permanent weight pruning. Existing recurrent parameter owners remain plastic; the method obtains no saving by stagewise freezing an old column. Usual parameter-version discipline during one differentiable trace is shared with the controls. If the proposed code introduces new encoder parameters, their support and derivative caches enter the cost immediately.

The surgery itself can be proposed by a learned controller with structural ledger features and feedback from previous trials. Its executable transition is still the bounded code action above. The learning signal is the measured paired result below. The constructor is not supplied a correct modular decomposition, future task label or an exact minimum-cut answer.

## 4. Paired finite-learning probation is the admission gate

Before destructive production use, fork a common versioned checkpoint into the original full continuation and the candidate code/port continuation. The cap specifies which original private values/tapes/restart data are retained, for how long and at what cost.

Each support/query instance in the candidate branch is materialized once at its actual return. Its consumer can subsequently read only the declared port/cache data. The full branch retains its original information solely because that probation expenditure was authorized and recorded. Both branches obtain the same legitimate external support observations and run the specified bounded native learning schedule. Then they predict on fresh external query/retention observations before those targets are revealed.

The challenge compares actual finite learning outcomes, not just current output or cotangent reconstruction. Source-aware translated consumer initialization and optimizer migration are explicit parts of the candidate. Where the parameterizations differ, compare observed function outcomes/resource cost rather than claiming identical raw gradients. The original full model is a diagnostic reference, not an infallible teacher; external performance is primary.

Accept or continue training a candidate only when a declared held-out challenge meets task and retention tolerances AND its measured resource tradeoff is worthwhile over the stated use horizon. A failing challenge can propose another return coordinate, keep the frame live, move the cut or reject the surgery. No missing private information is reconstructed after deletion. A probation that ends before its needed feedback is censored/unverified.

The exact learning rule can use a score-function update for discrete surgery choices or bounded proposal search. A scout encoder/body can use meta-gradients through the paid inner native updates. CoBRAS and v2's observed-cotangent residual can initialize coordinate proposals, but the challenge includes the learning continuation; a current-consumer span is insufficient. All VJPs, second derivatives, candidate forwards and full reference learning count.

## 5. Full formation cycle with a necessary plasticity correction

Consider a long-lived producer frame with h=(u,v,nuisance...), generated by many internal events. Its live support ledger and actual measurements identify a costly ancestry crossing into this frame. A first proposed cut returns only u and retires the rest.

The current consumer is

    q_w(u,v)=u+w+w^2v,      w_0=0.

The hard port C_0=u with translated consumer q'_w(z)=z+w preserves the current output u, current state cotangent (1,0) and current parameter derivative 1. Hence a current-output/first-order validator admits this wrong proposal.

Learning probation obtains a real support observation (u,v,y)=(0,2,1). With squared loss and native SGD rate 1/2, BOTH branches have gradient -1 and update to w_1=1/2. On a fresh query (u',v',y')=(0,3,1.25), the full branch predicts 1.25 while the port branch predicts .5, with loss .28125. Its next native gradient also differs. This failure is actual finite learning evidence that the candidate cut removed useful information; it is not a fictitious local target for v.

The constructor now proposes a new executable port containing the missing feature, for example C_1=(u,v), and tests it on new support/query episodes. In a dense neural producer, the analogous feature must be learned/proposed from actual challenge failures rather than given as a labeled hidden coordinate. The original symbolic witness is only a negative control and should also be solved by joint state/parameter derivatives or parameter-neighborhood sampling, since partial^2 q/partial w^2=2v.

If the corrected candidate passes fresh finite learning/retention tests and actually lowers the full measured learning-state objective, it can enter production: compute its port once, fold actual producer sensitivities into returned values, release retired private state, and continue native learning at all eligible producer/consumer owners. Future challenge failures revise only future calls. This completes contamination -> candidate surgery -> finite-learning failure -> corrected port -> validation -> physical retirement.

Information accounting is essential. If a trainable encoder C_phi=u+phi*v is used at phi=0 and its derivative cache retains partial C/partial phi=v, then v was NOT erased. Count that cache and allow the same information/operations to controls. A hard proposal avoids that particular side channel; a scout-only/frozen-within-challenge encoder is allowed only with its selective gradient scope explicit. General trainable ports must charge all their retained derivatives rather than pretending they are one scalar.

## 6. Concrete observations and cost counts for one proposal

For one candidate trial, declare a cap H on retained original events, K native learning updates, B query examples and R retention examples. The cost record must include at least:

| Item | Honest accounting |
| --- | --- |
| Original learning stream | Actual external training labels, obtained once and reused by both branches |
| Query/retention labels | B+R newly observed labels unless lawfully retained earlier; never visible before prediction |
| Paired computation | Full and candidate native updates/forwards, derivative work and any translation/search |
| Probation state | Full private state, retained/restart inputs, versions and differentiation records up to H |
| Port learning state | Values, exact sparse/factored derivatives, parameter read/support sets, optimizer/policy state |
| Proposal search | Support-ledger replays, candidate-code construction, failed candidates and controller training |
| Production benefit | Measured peak bytes, byte-time, total work and per-event latency over an explicit subsequent-use horizon |

A trace support estimate cannot be substituted for these measured counts. The old and candidate task distributions, precision and allowed replay operations must match. It may be cheaper to store raw inputs and replay the original producer; that alternative remains legal and charged. A permanent full shadow is not a successful low-credit frontier.

## 7. Strong actual controls and what an effect would mean

Compare the proposed formation cycle to weight-magnitude pruning and ordinary structural heuristics at the same proposal count; ACT/HM-LSTM-style gating and boundary learning; OML-style post-adaptation representation learning; CoBRAS/goal-oriented ports for actual learning continuations; and mixed AD/checkpoint/preaccumulation runtimes. Include a dynamic structural search driven by actual gradient-memory cost, rather than only a forward-FLOP baseline.

These controls receive the same source graphs, actual support/query observations, parameter/cache information and resource envelope. They may allocate it differently. Letting an ordinary modular learner adopt this exact support-contamination plus finite-learning-probation rule is a replication/shared-mechanism control; success there attributes the effect to the formation rule. It is not a novelty veto simply because another universal learner can run the same rule. Conversely, broad architecture-search expressivity is not evidence that the proposed rule already solves its scientific target.

Useful causal ablations remove the executed support-closure pressure while preserving forward counts; remove finite-learning probation while preserving current-output/cotangent tests; and keep the same resource objective but use ordinary forward/activation-based proposals. A ring contrast alone cannot establish a performance claim or matched function class. A win over ACT forward cost alone would identify that its objective missed credit cost, not by itself establish a paradigm change.

The prospective scientific content is an acquired organization in which expensive recurrent dependency growth is replaced by reusable, still-trainable procedural interfaces. The decisive unresolved step is whether this rule can learn those interfaces/read sets on consequential new compositions while preserving consumer plasticity, at an acquisition cost worth paying against the strong actual memory-aware alternatives. The primitive support union, RTRL cost arithmetic and post-adaptation loss are known tools; no novelty is claimed for them individually.

## Status and next single decision

This v4 supplies a concrete surgery/proposal rule and complete formation cycle, with explicit plasticity and information admission tests. It preserves all older artifacts and keeps the sealed stage007 mechanisms separate. No experiment has been run by this worker.

Root should choose a central acquisition/content test only if the best plausible success changes an important learning/resource limitation beyond pruning a convenient graph. The rule is now concrete enough to evaluate that judgment without another chain-rule-only test. This is a return to MAIN selection, not campaign completion.
