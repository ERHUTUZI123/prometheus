# A return interface must survive the consumer's learning

AGENT PROPOSAL / RECONSTRUCTION. Separate v3; v1/v2 unchanged. Incorporates root's paired learning-challenge repair and the supervisor's stronger witness. No experiment or neural acquisition result. MAIN continues.

## Judgment

Current prediction and current cotangent sufficiency can select a wrong physical return interface. The consumer is a learning system: changing its parameters can make an initially irrelevant producer feature useful. Destroying that feature can preserve today's computation and today's native gradient while disabling tomorrow's learning.

The operative repair is **capped paired learning probation**. Before committing a destructive interface, actually adapt full-state and port-only consumers from matched initial learning state using the same legitimately acquired training observations, then compare fresh external outcomes under the declared learning budget. Use failures to form executable ports and boundaries; do not certify a port from current gradient span alone.

Post-adaptation representation objectives already exist, including OML. The proposed mechanism is their use to form/revise physical program returns, parameter read sets and retained learning state under real byte-time/latency constraints. Its contribution must be judged against actual meta-representation and adjoint-aware methods with the same data, not by calling the objective novel or by comparing only to naive BPTT.

## 1. The exact counterexample is stronger than a current-gradient mismatch

Let private producer state be h=(u,v). The consumer is

    q_w(u,v)=u+w+w^2 v.

At w_0=0 the fixed hard port C_0(h)=u and port consumer

    q^port_w(z)=z+w

have all of the following equalities:

    q_0=u,
    partial q_0/partial h=(1,0),
    partial q_0/partial w=1.

Thus current outputs, current producer-state cotangents AND the current native parameter derivative agree exactly. With squared loss and a real observed whole-program training target y, both perform the same SGD step

    w_1=-eta (u-y).

On a subsequent query, their output difference is w_1^2 v, and their native derivatives are generally 1+2w_1 v versus 1. Current first-order agreement did not preserve finite learning response.

For a full numeric process, take one training observation u=0,v=2,y=1 and eta=1/2. Both branches predict zero, have identical gradient -1 and update to w_1=1/2. A fresh query u'=0,v'=3 with observed external target y'=1.25 then has

    full output = .5+.25*3=1.25, full loss=0,
    port output = .5, port loss=.28125.

The full query gradient in w is zero, whereas the port gradient is -.75. These labels are consistent with the same whole-program family at w_star=.5; they are not local targets supplied for u or v. The example exhibits a finite failure, not a result from a run.

Adding a hard second coordinate v repairs this specified family. It is not a foundational discovery: the second parameter derivative partial^2 q/partial w^2=2v already exposes the missing information, and a competent joint state/parameter derivative or parameter-neighborhood method can retain it. The example is an exact admission/negative-control check for the formation rule, not the scientific endpoint.

## 2. Count all retained information, including derivative caches

The witness uses a **fixed hard port** C_0=u. It has no trainable coefficient whose derivative secretly stores v. If instead C_phi=u+phi v and a full derivative cache is retained at phi=0, then partial C/partial phi=v. The variable has survived in the learning cache even if it was removed from the forward state. A claim that v was erased would be false.

Every candidate must declare:

* the port's actual forward values;
* all parameter versions, Jacobians/eligibilities, optimizer state and decision-score traces still stored;
* which of these the forward consumer and native learner can read;
* every byte and access cost, with the same information rights for controls.

A learning-state cache cannot be declared inaccessible to a strong control merely to create an apparent separation. If a cache contains enough information to repair the query, allow the same lawful read/compute operations and charge them. Conversely, choosing not to retain gradients for a frozen scout-only encoder is valid only if its learning scope is explicit. It does not preserve exact gradients through arbitrarily old production calls after the encoder is changed.

The first concrete formation version can use hard typed port proposals (select/compute specified scalar functions) and a bounded scout/meta learner. Production ports and their encoder version are frozen during each inner learning challenge. Encoder updates affect future calls; exact old-instance encoder gradients are available only on explicitly retained probation records. This avoids an uncounted information channel while remaining an actual learning/formation rule.

## 3. State and complete paired probation transition

A production library entry holds the producer body, a return program C, boundary policy, port-consumer translation/initialization rule, declared native learning rule U and the physical cache/read-set specification. State includes the active frames and returned values, consumer parameters w and optimizer memory, the producer/encoder versions, bounded probation records and the formation proposer.

At a sampled boundary the controller can choose KEEP_STATE, PRODUCTION_RETURN or LEARNING_PROBATION. The final option pays to retain the full boundary states and restart/derivative data needed for the declared comparison until its cap. It initializes two branches:

    full: original producer state h and native consumer state (w_0,o_0),
    port: materialized z=C(h), translated consumer/optimizer state (w'_0,o'_0).

The translation is a specified, executable part of the port proposal. It is not supplied a future optimal consumer. Where possible, verify current output/native-learning equivalence before the challenge; where architectures differ, report their actual capacities, initialization and compute. Compare outcomes in function space, not arbitrary equality of differently parameterized weight trajectories.

For k=0,...,K-1, both branches receive the same actually obtained external training observation/batch D_k and apply their declared native updates:

    (w_{k+1},o_{k+1})=U(w_k,o_k;h_train_cached,D_k),
    (w'_{k+1},o'_{k+1})=U'(w'_k,o'_k;z_train_cached,D_k).

Each support/query instance is encoded ONCE at its actual return time. The port branch cannot re-encode an old instance from h after each inner update or after a meta-update. Doing so would use the state it claims to have destroyed. Fresh query instances are likewise encoded at their real arrival, without their query target being visible before prediction.

After K bounded native updates, obtain a fresh external query batch and any specified old-task retention observations. Record the full/port losses, actual work/state costs and, when paid for, output/parameter cotangents after learning. No query labels or post-learning gradients enter the earlier return decision before being observed.

This initial comparison uses exogenous supervised/event streams. If branch actions change future observations, both branches cannot automatically use the same observational continuation; explicit paid counterfactual replay/simulation or a different causal protocol is required. A probation that times out without its learning/query evidence remains censored, not certified.

## 4. One operative formation/update law

For a candidate return code/boundary/read set, use the measured challenge score

    J_port = L_query_after_K_native_updates
             + lambda_R L_retained_after_K
             + lambda_M K_all_live_learning_state
             + lambda_C K_forward_backward_learning
             + lambda_A K_probation_and_code_acquisition.

A paired discrepancy from the original full continuation is a diagnostic/constraint, not the only objective: an old full consumer can itself learn badly. Actual externally observed task/retention performance is primary. Learning state includes the caches in section 2, and can be charged by byte-time, peak bytes and per-event latency as appropriate to the declared budget. Proxy width-times-parameter-support is useful for proposing cuts, but actual sparse/factored storage and restart alternatives must enter the measured score.

The proposer uses actual failures to choose a bounded next action: add/train a return coordinate, keep state longer, move a boundary, extract a new procedure with a smaller genuine read set, or retain the old implementation. For continuous scout encoders, differentiate through the bounded inner learning updates on the paid probation data. This is a meta-gradient, potentially requiring higher-order work; it is not free online gradient information. For hard program/boundary proposals use bounded search or a declared score-function estimator, with failures retained.

V2's future-cotangent residual and CoBRAS can initialize/propose a coordinate. Revise their target function to include the **actual learning continuation**, not only a frozen consumer's outputs. For instance,

    F_K(h, observed_learning_data, initial_consumer_state)
       = outputs after the specified K native updates.

Observed gradients of this function or its query output can expose information that current gradients miss. However, differentiating through it has the same honesty requirements as the paired challenge: actual training data, optimizer state, retained traces and bounded K. A local span match does not establish universal learning sufficiency or a globally realizable return program.

In the witness, a real post-update query output has state cotangent (1,w_1^2). Relative to the old port's row (1,0), this exposes the v direction. A candidate coordinate that retains v can then be tested as code on new challenges. The procedure does not need to be told a correct local v target; it can propose/select that coordinate from the actual failure and available producer computation. A dense neural case requires learning a useful executable feature rather than receiving an explicitly factored symbolic expression.

Promote a port for production only on held-out states, consumer episodes and declared learning horizons after paying acquisition. Continue occasional capped learning probation. New failures modify future calls; they cannot repair already erased old instances. Passing is evidence for the tested task/update/horizon distribution, not all possible later training.

## 5. Worked formation process after the failure

Initially the constructor favors C_0=u because it passes current outputs/gradients at w=0 and costs one forward coordinate. Its first legitimate learning probation produces the section 1 discrepancy. This supplies a counterexample to that candidate under the declared one-step learning challenge.

The next bounded proposal is C_1=(u,v), with port-consumer code q_w(z)=z_1+w+w^2 z_2 and the same scalar native parameter/SGD rule. This is an executable source transformation; no future w or target is installed. Both current behavior and the specified later updates can now be realized from the retained port, without a hidden state read. The new proposal is evaluated on independently obtained support/query episodes and its additional state cost is measured.

For this exact family, supporting outputs at both w=0 and a nonzero w requires two independent state sensitivities: (1,0) and (1,w^2). Any smooth port supporting both over an open state region must have width at least two, by the chain-rule rank bound. This is a scoped local obstruction, not a global theorem about minimum finite-precision code or all learning algorithms.

The important larger construction would have N private recurrent coordinates but a small, learned set of features sufficient for the encountered finite-budget consumer learning processes. Formation would discover those features and when to return them, rather than always returning the whole state or fixing a summary width/layer in advance. Real parameters/read sets, encoder derivatives and optimizer memory may erase the expected savings; those are measured rather than assumed.

This example demonstrates a different formation pressure from current-output compression. It does not show the learning algorithm discovers a difficult nonlinear feature, changes an important scaling limit, or improves on competent meta-representation learning. Those remain the substantive tests.

## 6. Actual precedents participate in the construction

OML §3 equation (2), Algorithms 1–2 explicitly trains a representation using loss after native online SGD steps on a prediction network, with representation parameters frozen during inner adaptation. This is directly reusable for a valid challenge objective and is an essential actual control. Learning a representation for future adaptation is not new here.

Likewise CoBRAS can reduce a function whose outputs include a finite learning continuation, if supplied the correctly obtained snapshots/adjoints and its stated assumptions. HM-LSTM already learns eject/reset boundaries; ACT learns adaptive computation using a task/resource objective; CCN changes recurrent organization to make learning efficient. V3 must not claim these broad ideas as absent.

The candidate's specific formation question is whether a system can acquire/revise **physical causal returns with complete live learning-state accounting** so that continuing consumer plasticity stays feasible across variable schedules and new composed computations. An OML-style fixed encoder supplies a strong formation objective but does not, merely by having that objective, establish the claimed dynamic port/boundary/readset mechanism or its resource behavior. Test its actual rule in the same runtime and allow it the same adaptation data and budget. If adding this formation rule to it reproduces the effect, that is shared content attributable to the rule; it is not proof of historical novelty.

Cheap strong controls include current-output CoBRAS, CoBRAS on legitimately sampled learning continuations, joint state/parameter derivative or parameter-neighborhood feature selection, ordinary OML/MAML-style representation training, memory-aware architecture/program search and source-aware liveness/checkpoint/preaccumulation. The quadratic witness should be solved by the derivative/neighborhood controls; a result defeating only current first-order compression is insufficient.

Every method receives the same actual support/query observations, parameter/cache information and replay rights. It may spend its complete resource allowance differently. Raw-input retention/replay is allowed and charged; if it is cheaper than the learned interface, report that result. No comparison can hide v in an eligibility and then deny another learner access to the same retained bits.

## 7. Important prospect, limits and central decision

The prospect concerns agents that must compute with large transient state, preserve many returned objects through variable waits, and keep learning their consumers under strict memory and per-event compute limits. Current-output compression can remove dormant learning features; retaining full state/caches can scale badly with outstanding objects. A successful formation rule would learn which information and credit records must cross a return to preserve useful *finite subsequent learning*, then acquire cheaper procedural implementations/read sets where possible.

That is more ambitious than choosing hidden width from an immediate loss curve. It is still not evidence of a new foundational principle. A finite challenge can miss other tasks, optimizer states, horizons and parameter regions. A learned feature may overfit the challenge, and the meta-training cost can exceed production savings. Moving a representation to parameters/caches is not deleting its information. Fully continuous exact learning with mutable shared parameter versions retains v1's outstanding-credit difficulties.

The central experiment, if root selects it, should test actual formation/content: expose a current-output-and-gradient-sufficient port to real bounded native adaptation, observe the failure, then acquire a port/boundary/readset that preserves new finite learning outcomes on held-out episodes under the full budget. The exact quadratic witness is a prerequisite negative control; a consequential result needs nontrivial neural procedure/consumer compositions where known higher-order/OML/model-reduction controls receive the same resources. No more chain-rule verification or hidden-dimension tuning alone would answer the mechanism question.

This v3 supplies an operative new pressure and complete information path while preserving the existing-method boundary. No experiment has been requested or executed by this worker. It returns to MAIN selection; it does not end the campaign.
