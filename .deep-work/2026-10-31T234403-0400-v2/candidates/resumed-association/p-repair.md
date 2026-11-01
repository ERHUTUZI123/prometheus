# P technical repair — acquire propagation from sparse finite-effect audits

AGENT PROPOSAL / RECONSTRUCTION. Implements the new MAIN steering in inputs/learning-association-2026-11-01.md. This is an information/interface correction to root's association construction, not a competing experiment or a novelty claim. No code or experiment was executed. Source checks are original technical reading; analytic numbers below are an illustration, not measured results.

## Judgment

The association can be acquired without a free matrix of future recipient labels if its training signal is **sparse, randomized, paired audits of particular proposed transfers**, followed by shared prediction over observable recipient histories. Unknown recipients get predictions, not certified labels. The central test must score each proposal before its audit label is revealed, including rejected proposals. Otherwise accepting only favorable transfers quietly turns the evaluator into a success oracle.

For portability, transmit a finite **change in observable behavior** and implement it in each recipient's independent parameters. Copying raw neural weights/gradients across unrelated hidden-unit coordinate systems is not a valid transfer interface. In the first concrete version, positions have the same semantic input/output interface and may have independent hidden coordinates. Transfer to arbitrary new input semantics is not assumed.

## 1. Concrete state and packets

There are M independent position models f_{theta_j}, a shared finite-effect predictor mu_psi, and a bounded per-position history h_j. Owners theta_j are physically independent in the first test; source position and recipient have no hidden tied parameters. Execution uses IDs to select storage, but predictors never receive position IDs, insertion order or absolute table rows.

A public unlabeled probe set P=(x_1,...,x_p) is sampled from the declared input distribution. Its inputs have common semantics across positions. It is not a set of labeled future recipient tests. After an ordinary source experience produces theta_i -> theta_i^+, obtain

    u_i = (context_i, source_observation_summary,
           f_{theta_i}(P), Delta_i(P), source_step_work),
    Delta_i(P)=f_{theta_i^+}(P)-f_{theta_i}(P).

These are actually evaluated before/after source responses. Their acquisition costs 2p forward evaluations in addition to the source learning step. The packet does not contain recipient outcomes or a true task-group label.

Each h_j contains only an explicitly acquired context descriptor, current responses f_{theta_j}(P), a bounded summary of its previously observed support/calibration examples, and previous sparse audit records/surprises. A recurrent history encoder or fixed reservoir may compress these records. Its memory budget is matched by replay/optimizer controls. Such compression is lossy; it does not reconstruct arbitrary discarded history or imply statistical sufficiency. Record the model version and intervening updates so stale effects are not treated as current measurements.

### The first transport mechanism

Use a finite amplitude set A, including 0 and positive/negative scales. For action a, form pseudo-targets

    y_tilde_j(P)=f_{theta_j}(P)+a Delta_i(P).

Starting from a snapshot theta_j^0, run at most r local optimization steps toward these pseudo-targets, obtaining theta_j^{a}. This bounded distillation uses no new recipient target labels. It may fail to realize the pseudo-target; the actual realized response Delta_j^a(P), solver work and failure flag enter the effect predictor and transcript. The amplitude set, local optimizer and r are designed scaffold. Which source experience propagates, to whom, in what context and amplitude are learned.

This interface avoids assuming neuron identity alignment. It does not solve arbitrary semantic alignment or guarantee behavior away from P. Root may replace scalar amplitude by a learned finite response transform, but that transform must consume the same observable packet/history and incur the same sparse audit cost. Supplying correct cross-position transformations is not permitted as hidden supervision.

## 2. One complete acquisition cycle

1. **Predict from observed state.** For each eligible recipient/action, predict the distribution of finite loss change and uncertainty:

       (m_j,a, sigma_j,a)=mu_psi(u_i,h_j,a,realized_transport_features).

   If obtaining realized transport features would require distilling every candidate, use a cheap pre-transport prediction for selection and compute those features only for the sampled candidate. Do not charge one transport while secretly running M transports. A complete descriptor scan costs O(M|A|) predictor evaluations and is counted; cached retrieval is also charged and cannot supply extra labels.

2. **Choose a sparse audit, not all recipients.** In the minimal test take b=1 recipient/action pair per source event. Sample (j,a) from logged propensity

       q_t(j,a)=(1-epsilon) q_exploit(j,a)+epsilon/(M|A|),

   with epsilon>0. q_exploit favors positive predicted finite benefit after work/harm costs. The uniform component gives an exploration/identification path and exposes mistaken routes. It does not make one sample an accurate estimate of all pairs. Larger b must be explicitly budgeted.

3. **Freeze the proposal before its new label.** Compute theta_j^a and predictions using only u_i,h_j. Snapshot the sham baseline theta_j^0 after exactly the same ordinary local learning/replay that both branches would have received. Only then obtain one fresh recipient audit minibatch V_j with its target labels from the allowed environment/data stream.

4. **Evaluate the paired finite outcome.** On the SAME V_j and matched randomness evaluate

       R_t=L(V_j;theta_j^0)-L(V_j;theta_j^a).

   This is an actual finite-update effect on sampled recipient data. It is not -eta g_j^T K g_i and not a predicted future-learning gain. A norm/work-matched sham transfer from a randomized unrelated source packet can be evaluated on the same labeled minibatch, costing extra forward work. Its purpose is to distinguish useful source association from generic extra optimization/regularization. The zero-transfer branch is the primary causal contrast.

5. **Learn from success AND rejection.** Store (observable packet, pre-audit h_j, chosen action, q_t, realized transport, R_t, sham outcome, work). Fit mu_psi with squared/proper probabilistic loss on observed R_t, retaining negative outcomes. For a declared reference distribution nu over recipient/actions, use importance weight nu(j,a)/q_t(j,a). Conditional on the current checkpoint/proposal, this corrects sampling in expectation; it does not create missing labels, solve variance, or validate unobserved contexts. If weights are clipped, report the resulting bias. The shared predictor learns general relations; no M-by-M truth matrix is installed.

6. **Revise and act.** A negative R_t increments the history's surprise/harm record and revises mu_psi, so similar future contexts can stop receiving this source update, receive a smaller/opposite action, or trigger additional paid audits. The online policy may reject/rollback the current transfer using V_j, but its pre-label proposal remains in the performance record. All controls receive the same newly obtained labels and may use them for ordinary recipient learning. A route is not reported as zero-label success merely because the audit rejected its failures.

7. **Retain compressed response experience.** Update h_j from the actual effect and context; bounded global replay may retain selected rare failures. New positions initialize h from their allowed descriptors/calibration observations and use the shared mu_psi. A later context change can revise this history through new audits. No deleted per-position state is assumed to reappear.

The main learning computation is therefore causal and inspectable: source update -> measured finite response packet -> proposed recipient transport -> sparse paired feedback -> changed future association policy. It does not require a fixed learned sharing matrix and does not collapse independent position parameters into one shared owner.

## 3. Worked process: useful source effect, harmful recipient copy, revised amplitude

Take f_w(x)=tanh(wx), squared loss (f-y)^2/2, and one probe x=1 purely to show the information flow. A source with w=0 observes y=0.8 and makes a finite update to w=2. Its source loss changes from 0.32 to about 0.01345; the measured packet contains Delta_i(1)=tanh(2)≈0.96403. The packet need not state why another position should share its experience.

A recipient currently has w=1, hence response about 0.76159. Suppose its context/history leads the current predictor to select amplitude a=0.2. Distillation toward the proposed response 0.76159+0.2(0.96403)≈0.95440 is a finite actual local computation. If it fits within the declared tolerance, the proposal has response about 0.95440. No recipient target has yet been queried for this proposal.

The sparse audit now happens to reveal y=0.8 for this recipient. Its sham loss is about 0.000738 and the transported loss about 0.01192: R≈-0.01118. This is a recorded failure, even if rolled back. The positive source effect was not universally transferable, and local gradient alignment would not have established finite-step safety. The first-order direction at the recipient points upward, yet this large update overshoots.

That audit teaches the state/history-dependent predictor to reduce propagation in this response/context regime. A later tested amplitude a=0.04 would request response about 0.800155 and have very small loss in this specific example; the learner is not given this successful amplitude as an inverse. It must select/explore it from observed episodes or learn it through audited loss gradients, charged to the chosen recipient. No outcomes for the other M-1 recipients or untried amplitudes were obtained.

An independently new position with similar observable response and comparable contextual/support history can receive the revised small-amplitude prediction through the shared encoder. This is a generalization hypothesis, not a deduction. If that new position instead targets -0.8 while all permitted observations are identical, the policy cannot know that before new evidence. It must carry uncertainty and audit/abstain; no descriptor trick eliminates this identifiability limit.

This one-probe scalar calculation is an information-flow illustration only. It is not the endpoint task, an executed result, proof of nonlinear function generalization, or evidence beyond ordinary line search/learned optimization.

## 4. Main information budget and fair success measurement

Per source event in the minimal construction:

| Resource | Explicit allowance/cost |
| --- | --- |
| Source labels | One ordinary source support minibatch; identical across methods |
| Source packet | 2p unlabeled probe forwards; packet size recorded |
| New-position calibration | A fixed declared number c of actually observed labels/examples; charged to every method |
| Recipient labels | One newly sampled audit minibatch for one (j,a); no all-j outcomes |
| Recipient transport | At most r distillation steps on p pseudo-labeled public probes for that sampled candidate |
| Paired evaluation | Two recipient forward evaluations per audit example; sham adds one more |
| Association memory | Shared predictor plus bounded h_j/replay, equal-byte memory controls |
| Prediction/search | Actual candidate scan/retrieval and action computation, counted |

Two separate outcomes are needed. First, **pre-audit transfer utility** scores every frozen proposal on the freshly obtained audit data, including rejected routes. Second, **later learning/retention utility** uses independently later labeled observations under equal cumulative label/work/memory budgets. Reporting accepted-only utility, auditing all positions while charging one, or comparing with a baseline denied the audit labels invalidates the claim.

New-position transfer must split by positions and contexts, remove IDs, randomize position order and independently initialize local model coordinates. The held-out audit labels are unavailable until after the proposed action. Descriptors must be functions of allowed observations; they cannot encode latent correct groups, target coefficients or future success labels. A null condition makes latent association independent of every allowed descriptor/history. The system should then express uncertainty and approach the no-information control, rather than appear to infer invisible relations.

Equal visible histories can support different unseen worlds, so arbitrary zero-shot association is impossible without assumptions. The constructive target is learnable structure in the declared task/context distribution. Test shifts where a familiar relation becomes harmful, with informative feedback arriving through the same sparse channel; check whether rejected routes revise subsequent propagation. Reusing a static correct group oracle would erase the scientific question.

For the cleanest initial comparison, use an exogenous randomized audit stream shared across methods, independent of any one learner's predictions. Its known propensities support off-policy estimates of the learned routing policy and common observed training evidence; a separate blinded execution phase tests the selected policy. If adaptive query selection is itself evaluated, allow each method its own equal-budget query policy under paired environment seeds and count all queries. Different adaptive policies need not receive identical recipient-label realizations; do not claim otherwise or silently give one method all labels collected by the others for free.

## 5. Strong controls and checked source boundary

**MSR §4 — SOURCE.** Original text: https://arxiv.org/html/2007.02933 , equations (2)–(4), Algorithm 1 and §4.3. It represents vec(W)=Uv, learns U through validation loss after inner updates of v, then freezes U for test-task inner adaptation. Learning sharing from experience is already present. The proposed independent owner states and contextual online revision differ from that documented test procedure; MSR-Joint also updates U and v in the inner loop, so the mere phrase “sharing can change” is not a distinction. The control should include an online/contextual extension with the same feedback and memory.

**TAG task affinity — SOURCE.** Original text: https://arxiv.org/html/2109.04617 , §4.1 equation (1) and §4.2. TAG evaluates the recipient task's loss before/after a source task's shared-parameter update, averages these effects, and selects task groupings for retraining. Thus finite measured cross-effects are established. Our sparse sampled outcomes, independent recipient transports and descriptor-conditioned continuing revision are proposed details, not a novelty certificate. Adapt TAG to the same sparse feedback and new-position descriptors; a dense affinity oracle is an informative charged upper control, not a fair equal-budget method.

**Learned kernels/optimizers — RECONSTRUCTION/control requirement.** At a fixed context and small update, this transport policy can reduce to a learned coupling kernel. A sufficiently general graph/context-conditioned learned optimizer can implement the whole procedure. Give it the same source packets, local histories, sparse labels, independent parameters and memory. Also compare a contextual bandit or Bayesian effect regressor choosing transport amplitude, and a line-search/validation method with the same queried labels and forward work. If these reproduce the effect, there is no demonstrated separate principle. The candidate's prospect is acquiring reusable associations over continuing experience and new positions under a strict feedback budget; it must earn significance through that behavior.

## Weakest remaining link and next discriminating decision

The repaired mechanism has a complete information path and revisable update law. Its weakest link is not obtaining a label for every pair; it is whether short observable response histories carry transferable causal information about useful propagation that a matched contextual learned optimizer does not acquire equally well. The next decision is to freeze one consequential nonlinear mechanism-learning family whose new-position/changed-context conditions make that distinction meaningful. Root owns that construction and its test; this artifact requests no independent execution.
