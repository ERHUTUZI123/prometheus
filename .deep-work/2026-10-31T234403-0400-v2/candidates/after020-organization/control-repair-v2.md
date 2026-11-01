# Control repair v2: learn a stronger write without undoing the acquired hold operation

**AGENT CONSTRUCTION / MANUAL ANALYSIS; no new code or experiment.** Same selected hypothesis and protected development lineage. Mechanism-v1, pilot_v1.py and the first raw run are unchanged. This worker read the actual pilot checkpoints, gradient-check receipt, final summaries and changed-write evaluation rows. Root executed that process once; this worker did not rerun it. Root owns any next feedback/process contract and MAIN decision. Provider-effective mode unknown.

## 1. Actual first-stage evidence and what it did not identify

The actual raw file `probes/after020-local-contrast/pilot-v1.raw.jsonl` contains 193 rows and a complete record reporting 2,048 updates, nine passed finite-difference checks, and about .09924 seconds of internal wall time. That is evidence about this one deterministic diagnostic, not a general efficiency estimate.

The endpoint+prefix+local-carry arm finished at

`A=1.3888860838531099, beta=.31505394251752883, v=1.2366219426063343`,

with task loss about `3.66e-16`. Prefix alone reduced its seed loss but retained task loss about `.49999975`; endpoint-only stayed at `.5`. The strong ordinary endpoint arm initialized at A=1.2 also succeeded, with final A about1.42056 and task loss about1.46e-25. Therefore the outcome does not show that local contrast is necessary or generally superior. It does establish a useful stage-specific fact: the mean local-carry update did learn a holding regime from the contracting checkpoint in this run; its possible dilution did not prevent that outcome.

Both persistent arms kept the first cue after a later opposite pulse. For the local-carry arm, a long positive hold had actual pre-pulse state about `.80865456`; the negative pulse left state about `.66852703`, still positive. Single-cue training had not distinguished first-cue from last-write task extensions. This is **unidentified control behavior**, not a retrospective failure against labels that were never supplied.

The first pilot used the quarter-normalized carry loss clarified by root; the older mechanism-v1 formula normalization remains frozen and is not silently edited here.

## 2. New legitimate feedback, and how to find the write cut

The next proposed information increment is four actual endpoint-labeled changed-write episodes, not labels inferred from the old evaluation columns:

`(p, 0^H, s, 0^L) -> y=s`, for p,s in {+1,-1}.             (1)

These are **specified prospective new observations** for the next process. No new endpoint query or acquisition of these four labels has been executed by this worker. They must be obtained from the declared task/label source and offered identically to every control. They select the last-write extension that the original two single-cue observations left ambiguous. Include all four combinations so that “negate the first cue” is not enough to fit the changed-write bank.

For each fixed p, pair the episodes whose final differing cue is s=+1 versus s=-1. The miner must check: same task context and length; identical entire prefix before the differing token; that observed token differs; and identical complete suffix after it. Locate the cut k by comparing the raw episodes, not by receiving an oracle annotation “this is a write.” In (1), k=H+2 and the common future has L zeros. The incumbent h=x merges one step later, at k+1.

The later merger still supplies an alias witness. But the necessary separation can be requested **immediately after the last differing input, at k**, before the extra zero transition. The two histories have a common remaining future and different endpoint labels, so a correct deterministic extension must distinguish them there. The identical prefix also gives a controlled local comparison: both episodes have the same actual new-mode predecessor before the input intervention.

The orientation of that distinction is a representation choice. In this scalar pass the actual retained readout has v>0, and the zero-input hold map preserves sign. Thus positive/negative endpoint class requires the corresponding sign at the cut. A positive margin is an additional robustness prior, not an observed latent-state amplitude. This justification is specific to the retained sign-preserving dynamics/readout. If the readout orientation changes or the common suffix transforms state differently, the cut target must be rederived; endpoint labels do not universally supply the same intermediate sign.

Raw history storage and reference replay remain necessary. Nothing recovers an old predecessor from an aliased h. With H=L=64, the four new episodes contain520 raw input tokens; their four new endpoint labels and all later reuses are distinct cost categories. The original two episodes, their labels and the acquired holding checkpoint remain available as retention data.

## 3. Remove the old exact-state target's ceiling

For the original symmetric cue pair, mechanism-v1's exact seed target was

`J_seed=(tanh(A*tanh(beta))-m)^2`, with m=.4.               (2)

At A>0,beta>0 the derivative of the inner state with respect to beta is positive. Once that state exceeds m, the derivative of (2) with respect to beta is positive too. Ordinary descent then opposes stronger writes even when stronger writes would help change an already occupied attractor. The first pilot ended almost exactly at the seed target, so this is a concrete conflict between an auxiliary amplitude convention and a new control demand.

Replace that exact-state loss by an oriented squared margin hinge:

`J_seed_margin=mean_i [m-o_i*r_tau(i)]_+^2`,               (3)

where o_i is the existing witness orientation and `[z]_+=max(0,z)`. Once the seed has its needed sign/margin, the loss stops pulling it back to an exact amplitude. This is a change in the learning rule justified by new control requirements; no claim that it was required for the old single-cue result is made.

Dropping the old seed loss entirely after its job is done is a reasonable same-signal control. The margin form retains a lower-bound constraint while avoiding the known ceiling; it is not a new optimization principle by itself.

## 4. Actual one-step write rule and its gradient

Keep the currently acquired state family

`r_t=tanh(A*r_(t-1)+beta*x_t)`, answer `v*r_T`.

Replay each new episode from its true reset to obtain its **actual current-model** predecessor at the mined write cut. Detach that predecessor for the local write gradient, then compute

`z_i=A*stopgrad(r_(k-1)^i)+beta*x_k^i`,

`r_write^i=tanh(z_i)`,

`J_write=mean_i [m-o_i*r_write^i]_+^2`.                   (4)

Here o_i is oriented from the newly obtained endpoint label using the retained positive readout convention. Neither that label nor future input is an input to recurrent execution. The local target is an offline task-derived constraint. Recompute the predecessor trajectories after every parameter update; detached predecessor values are not supplied correct latent states.

For an active hinge, with e_i=`m-o_i*r_write^i`,

`d J_write/d z_i = -(2/M)*o_i*e_i*(1-(r_write^i)^2)`,

`g_write_A=sum_i (dJ/dz_i)*rbar_i`,

`g_write_beta=sum_i (dJ/dz_i)*x_k^i`,

`g_write_v=0`.                                          (5)

These are **conditional one-step derivatives**. They omit the dependence of the replayed predecessor on A and beta. The true full-task RTRL derivative still includes the complete trajectory, including both writes and the hold suffix. Equation (5) must not be checked against a finite difference that replays a perturbed predecessor or called the true gradient of the full task.

For this newly labeled bank, `x_k^i=o_i`. Every unsatisfied local example therefore contributes a negative beta derivative. Its local update increases write strength, including when the current endpoint remains deep in the wrong basin and its terminal gradient is tiny. This sign statement uses actual observed label/input alignment in (1), not a claim that arbitrary future tasks label the last input.

## 5. Protect the acquired hold transition from the local write update

Increasing beta is not the only descent direction in (5). In an opposite-write case the predecessor has the wrong sign relative to the new label, so the A derivative is positive: a joint local descent also lowers A. That can weaken the very persistence learned in stage one. A new control loss should not silently destroy a retained computational operation.

After this worker derived the cut loss/write threshold, root supplied P's interference observation and a concrete retained-hold Jacobian projection. That is an external contribution, recorded here rather than attributed as independent invention.

For each actual retained zero-input hold transition, record predecessor a and next state `r'=tanh(A*a)`. Its parameter Jacobian, treating that actual predecessor as fixed, is

`J_H(a)=((1-(r')^2)*a, 0, 0)` for parameters (A,beta,v).   (6)

Stack the charged sampled rows and form

`P_H=I-J_H^dagger*J_H`.                                 (7)

If at least one row is numerically resolved and nonzero, the scalar family gives `P_H=diag(0,1,1)`. Applying it to (5) removes the new local write loss's A update while retaining its beta update. Because the zero-input transition is independent of beta and v in this family, this particular projected write component preserves the hold function **exactly**, not merely to first order on the sampled rows. It is a data/graph-derived control-parameter split, not a fresh high-gain initialization.

This exact statement is about the projected write component. A proposed combined pass is

`g = g_full_task + g_seed_margin + g_retained_carry + P_H*g_write`,

followed by the declared finite learning-rate/clipping/bounds rule. Full task and retained carry are not silently projected. They can still alter A and v; whole-pass retention therefore requires actual old-bank replay and the chosen admission criterion. Root will freeze coefficients, budgets and the readout-orientation handling before any process. No success is promised for that optimizer.

If every J_H row vanishes because states are zero, true saturation or numerical underflow, the data do not identify a protected hold direction. Do not silently set (7) to the desired mask anyway. A declared rank threshold, actual transition derivatives and a fallback/abstention record are needed. In more general gated families this projection may only give first-order preservation and can be costly; the scalar exact-independence argument does not automatically transfer.

An ordinary selective-plasticity baseline may simply keep the actually acquired A fixed and train beta, with the same labels and costs. That is a strong control for whether (7) adds anything beyond a familiar modular parameterization. There is no novelty claim for nullspace/constraint projection.

## 6. Worked control consequence from the actual learned checkpoint

At the recorded local-carry checkpoint, let a be the positive pre-write state, about .80865456. A negative input produces

`r_after=tanh(A*a-beta)`,

which the raw row reports as about+.66852703. For a single opposite pulse to reverse sign one needs

`beta>A*a`.                                             (8)

For the same pulse to establish negative margin at least m,

`beta>=A*a+atanh(m)`.                                   (9)

The current beta about.315 is far below even (8); A*a is about1.123. Persistence was learned, but control of the occupied attractor was not identified by single-cue feedback.

For that negative-write example, the local hinge in (4) is active: its scalar error is `m+r_after>0`. Its beta derivative is strictly negative, whereas its A derivative is strictly positive. The projected local update from (7) therefore increases beta without reducing A. The symmetric opposite-write example has the same beta-update direction. The two aligned-write examples already satisfy the sign margin and need no local correction.

The exact old seed (2) would resist this movement once its initial state exceeded m. The hinge replacement (3) exerts no such opposing gradient above margin. This gives a coherent write–hold learning pass: retain the acquired autonomous hold transition, train a newly demanded stronger input action, and validate the full terminal task after replay.

There is also a uniform representational witness, not a learned result. For any predecessor r in [-1,1], any cue s in {+1,-1}, and A>=0,

`s*(A*r+beta*s)>=beta-A`.

Thus `beta>=A+atanh(m)` guarantees the post-cue state has the requested sign and at least margin m regardless of the previous memory. With the actual A<1.4, beta=2 is sufficient for m=.4 and lies within the old beta capacity bound. No experiment has moved beta there; this is a hand capacity/control calculation. Zero-input dynamics then preserve the sign and approach their corresponding stable attractor. The resulting scalar mechanism can execute last-nonzero-sign classification for arbitrary finite cue/zero sequences under those exact deterministic assumptions.

That classification statement does not give exact numeric output amplitude immediately after every write. A fixed v and finite transient can give different regression outputs at short read delays. No clear/reset behavior, distractor-sensitive write selection, noisy execution theorem, or learned compositional reuse follows. Those remain genuine next control questions within this lineage.

## 7. Feedback, discovery and maintenance costs

All four new endpoint labels in (1) are additional information. The old two single-cue labels cannot supply them; the old raw evaluation's two possible target columns are not observations from a task oracle. The new examples, raw histories and task context must be shared with all competitors.

The miner pays raw-prefix comparisons, finding the last differing token, and complete common-future checks. Replay pays every recurrent step to obtain current predecessors; detached gradients shorten the **credit path**, not the physical forward history. Local write evaluation, all retained zero-input transition Jacobians, rank/projection work, full task/RTRL updates, old-data replay and rejected candidates count. For the scalar case the resolved projection is trivial after inspection, but its derivation and diagnosis are still explicit.

Parameter changes must be versioned. The old h/current-input computation stays one-way and unchanged. Modifying beta improves a new demanded control response of the acquired mode; it can change previously untested behavior, so it is not exact preservation of every old mode input history. The actual old single-cue bank must remain in the replay/admission test. Preserving a finite bank is not proof of universal old-task retention.

## 8. Strong same-information controls and next stage question

Useful constructive controls include full-task RTRL with the same new labels; endpoint learning with the old exact seed removed; a margin-seed-only variant; selective beta-only learning at the actual acquired A; a gated recurrent learner; and a model-based one-dimensional search over beta using the same visible state equations and endpoint bank. The last control can exploit the transparent scalar threshold cheaply and must not be replaced by a deliberately weak random-word search.

This is not a claim to beat those controls. It asks whether **task-derived local control constraints plus protection of an acquired operation** provide a useful learning organization, and which part causes the observed change. Existing optimization/control components are legitimate building blocks; naming them does not answer that empirical/mechanistic question.

The next informative construction/process, if root authorizes a frozen contract, should acquire the new labels, distinguish exact-seed interference from weak writes and hold/write interference, and inspect actual trajectories/conditional versus true gradients. No code or next process is executed or authorized by this note. The first pilot remains a completed first-stage observation, not a failed overwrite study.

`CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

`The first local-carry diagnostic acquired holding from its contracting checkpoint, while a persistent-init ordinary control also succeeded | OBSERVED, bounded one-run scope | actual raw final/checkpoint rows in §1 | one deterministic task/optimizer setting, not a general acquisition result | prefix-only lost its early separation | develop control under genuinely new feedback`

`Exact seed amplitudes can oppose newly needed stronger writes | HAND DERIVATION | (2) and positive beta derivative above margin | affects this auxiliary target convention, not all seed losses | none newly executed | use margin/no-seed control in a new frozen pass`

`A conditional write margin gives a direct basin-crossing beta signal, with the hold-Jacobian projection isolating it from A | EXPLICIT REPAIR / HAND DERIVATION | (4)–(9), scalar independence of hold from beta | full optimizer/readout effects, rank handling and actual acquisition untested | no new experiment | root freezes next feedback/learning contract`
