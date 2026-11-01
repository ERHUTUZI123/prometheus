# Constructive relevance to the existing gap

AGENT RECONSTRUCTION / HYPOTHESIS, no new experiment. These are refinements of the current open question, not replacement architecture names.

## Retained information versus newly acquired information

An exact compactor for old actions need not support a new action. Spikes, local updates and random birth do not change this fact. For the elementary update w'=w+eta*m*e, two states with identical retained execution variables but opposite e produce different next weights. Fresh random noise independent of the erased distinction cannot recover which old state was present. New observations correlated with it can help; alternatively the learner can construct a different mechanism without recovering the historical one. These are different scientific claims.

Do not make exact preservation of every old learning trajectory the scientific endpoint. The original ambition allows a deliberate new learning process after construction. Such a process must say which present capabilities it preserves, what new evidence it uses, what state it allocates, and what costs/retention risks it creates. The earlier thin-factor equivalence still limits the older K mechanism; it does not imply all alternative learning processes are equivalent.

## One meaningful mechanism opportunity

After inserting a new nonlinear computation using retained state, explicitly allocate its local dynamical/eligibility state and a plastic route for task feedback. New training inputs, measured output errors or rewards supply evidence; co-adapt the computation and its feedback route rather than assuming a precompiled derivative connection already exists. Eligibility keeps a local history until delayed feedback arrives; SAM is an actual prior for co-adaptive feedback circuitry, subject to its alignment assumptions. Event-driven execution is optional and only justified if temporal state/timing matters.

Possible concrete state is (z_old, theta_new, neuron_state, eligibility, feedback_weights, connectivity_mask, prior/proposal_state). None of those new components is free. If compaction later deletes feedback or eligibility, the claim must either preserve their future effects or explicitly pay for reacquisition. An edge that never existed has no historical eligibility unless counterfactual candidate-edge statistics were separately retained, with their cost counted.

The possible important change is reducing the need to prescribe a new module's credit-assignment organization in advance. It is not stronger universal expressivity, spontaneous recovery, or proof that the module learns a new operation merely because its synapses change.

## Real alternatives and the unresolved scientific step

Compare ordinary AD after graph growth; Kolen–Pollack/feedback alignment or direct feedback; e-prop/three-factor learning; existing structural sampling or dynamic rewiring. Match data, labels/rewards, topology opportunities, feedback precision, locality/communication restrictions and all calibration/storage costs. A digital comparison cannot give the candidate a locality restriction that the baseline is forbidden to exploit or ignore the latter's actual constraints.

Neither the source theorem nor this sketch proves that the pathway error E_p becomes small after a structural change. That reassembly step, feedback conditioning/rank and later-task usefulness are the serious unresolved conditions. If arbitrary new operations are required, SAM's fixed two-stream result does not automatically provide them. A larger feedback circuit can cost more than ordinary AD. A meaningful result would need an actual acquisition rule with a consequential difference under these matched assumptions; this note supplies a direction to inspect, not that missing result.

Provisional disposition: retain local temporal-credit state and SAM as substantive sources/constraints for P/R and future construction proposals; do not adopt SNN as the main architecture, run a replacement-neuron benchmark, or reopen the old covariance tests on this basis alone.
