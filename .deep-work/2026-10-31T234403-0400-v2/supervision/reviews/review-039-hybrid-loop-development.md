# Review 039 — executed hybrid loop learning, feasibility repair, and admission

**Decision: CONTINUE developing the same Monodromy method.** The executed V2 repair fixes a concrete coupling between a feasible neural calling update and an infeasible joint loop/controller update. It learns a new loop operation and its call from terminal examples, preserves the supplied old computation, and saves/reloads actual neural weights. This is a successful, bounded integrated mechanism diagnostic. It is not a six-stage acquisition result, a validated general Block, an independent topology advantage, or a publication-grade strong-baseline comparison. MAIN remains human-locked TOPOLOGY; the six fixed paper TODOs remain the governing scope.

## Evidence actually examined

I read the frozen executed sources in `probes/topology-architecture/versions/code-parameter-hybrid-loop-learning-v2/`, including `hybrid_loop_block.py`, `run_hybrid_loop_diagnostic.py`, and the original `loop_boundary_learning.py`, and compared V1/V2 source and trajectories. I independently verified SHA256 and byte counts for all **67 synchronized files: 39 outputs and 28 source files**. The coordinator receipt records CPU-only isolation, external120s/internal90s bounds, exit0, and runner elapsed8.944029545993544s. No GPU training, SSH, new experiment, downloaded code, or change to canonical Method was performed for this review.

The raw V2 arm records contain818/2360/2710 JSONL events for full/within/class respectively. Counts and curve rows agree with their summaries. I independently recomputed all372/1200/1200 recorded differentiable terminal feedback losses as negative log probability of the supplied terminal label; they match within1e-10. These are repeated accesses to four distinct new-task terminal examples, not372 independent supervised examples.

I inspected the actual owned final `.pt` ZIP metadata and numeric storages with the previously checked restricted, inert allowlist reader, adding only the known `DoubleStorage` descriptor in memory. No model code or Torch loading was executed. All final neural scalar values match the raw result, with **30 learned parameters and8 fixed feature-buffer scalars** per arm. Stored transaction counts, protected sets, configuration, and parameter hashes match the recorded results. Independently reconstructing the final soft neural calculation from those actual values and recorded executed maps reproduces all30 final held-out/old probability vectors, maximum absolute difference4.44e-16. This is independent arithmetic/data verification, not a rerun of numerical path continuation. The original execution separately records cold continuation of both restored primitives, matching their persisted maps and unchanged restored neural hashes.

## Actual mechanism and information

The implemented Block has trainable encoder/readout logits E,R, a shared2×2 linear body H, a1×2 loop head v, and a3×2 call head C. Operation and token features are the disclosed, nonorthogonal vectors x0=(1,0), x1=(1,1). The controller produces

    a_o = v H x_o,
    gamma_o = (0, a_o+i, a_o-i, 0),
    q_token = softmax(C H x_token).

The fixed cubic cover is z³−3z=u, with branch controls±2 and a three-sheet starting fiber. There is no trainable permutation-score matrix. Maps are obtained from charged numerical continuation or licensed same-class semantic reuse. For the supplied triangular family, a<−2 gives the old transposition B=(1 2), |a|<2 the identity, and a>2 the other transposition A=(0 1), in the fixed starting-sheet labeling. This exact scalar family is an important supplied geometric prior, not a discovered general representation.

Soft execution propagates a learned initial distribution through mixtures of skip, B, and the currently realized new map, then applies the learned readout. Hard execution uses the model's own encoder, argmax calls, realized maps, and readout. One slot is used in this diagnostic. All30 parameters, including the unacquired slot's controller/call parameters, are allocated and charged from initialization. There is **no workspace growth or increase in bank size** here: a new executable class forms in an already allocated slot.

The initial old loop/call and near-identity E/R frame are supplied. The old loop is actually executed to establish its receipt; six old checks, including empty words, are derived from that acquired starting mechanism. The new token's logits start exactly uniform across skip/bank0/bank1, not at the correct new call. New supervision consists only of four author-specified terminal examples, with lengths1–3 and noncommuting compositions. No target-permutation object, correct route, or intermediate correct state enters the learning rule. The fixed cover, branch geometry, affine loop family, full-fiber observations, and chosen features must nevertheless remain disclosed as architectural information available to conventional controls.

For the smoothed objective

    J(theta) = E_epsilon L(theta; rho(gamma(a_theta+epsilon))),

the actual update adds two terms:

    g_within = E_epsilon partial_theta L(theta; fixed executed maps),
    g_class = D_theta a_theta^T [sum_b jump_b(L) k(b-a_theta)],
    g_full = g_within + g_class.

The first uses BPTT through E/R/calls and the shared body, with executed maps detached. The second holds neural weights fixed when measuring terminal-loss jumps and chains the resulting action coefficient through the loop-producing network. In this one-dimensional setting, the event formula is a legitimate boundary derivative of the compactly supported smoothed piecewise objective. It does not differentiate an ordinary locally constant monodromy map as if that derivative were nonzero.

The actual code uses the same generic boundary estimator for `event` and `ordinary_event` (`hybrid_loop_block.py:366–437`). The stochastic neural term uses one raised-cosine noise sample per iteration; the action-event term queries both chamber representatives. Near the finite numerical-clearance tube, a noisy sample may be evaluated at an actually continued safe representative of the same scalar chamber. This is explicitly an ideal class-objective evaluation, charged and logged, not proof that the original noisy physical path was executable. Mean-path transaction/admission never uses that substitution.

## What V1→V2 actually changes

V1 tries the joint selected gradient at scales1,1/2,…,1/64, then accepts zero if no positive tested step preserves the contracts. V2 retains all those joint proposals but, before zero, tries

    theta' = theta − eta alpha M g_selected,

where M keeps only E/R and existing call-head entries, setting shared-body and loop-head directions to zero. It does not recompute a favorable gradient, obtain new labels, add parameters, or donate the other arm's gradient. Each attempted step still checks:

1. the finite NN/support/execution domain;
2. old-loop geometric displacement below its old clearance, plus numerical execution margin;
3. old input–call–operation–readout native relation closure;
4. old soft/native path mass above1/2 for protected words up to length16.

`hybrid_loop_block.py:490–615` restores the same Parameter objects from a common preproposal snapshot at every trial. Failed observations remain charged. The class-only arm has exactly zero selected E/R/call gradients, so the fallback is skipped rather than receiving the separately calculated within gradient. This is **a deliberate optimizer/feasibility-policy repair**, not a software bug fix, topology-specific novel primitive, or proof that every possible joint optimization scheme must fail.

The causal record is unusually clean for this small repair. All full-arm V1/V2 weights and sampled noise match exactly through the first28 iterations. At V2 iterations28,29,30, all seven positive joint proposals fail, then the first heads-only proposal succeeds. Actual shared/loop-head displacement is exactly zero for those three commits, while call-head displacements are .0629138,.0588132,.0548814. V1 instead performs72 final zero commits; its finite backtracking schedule remains stuck. Both ablation arms' entire100-update parameter/noise trajectories remain V1-identical. Thus the measured improvement is attributable to the changed feasible proposal policy in this version, rather than an undisclosed seed, new feedback, or changed initial route.

The new loop class A already appears after iteration1 in the full arm and also appears in class-only. The unresolved V1 issue was the calling/binding adaptation, not continued inability to cross the discriminant. The shared old loop moves from action−3 to−2.0159782 in the successful full arm while retaining B; it is not manually frozen throughout training. The new action moves from1 to3.0497013. Under the frozen finite line search, later joint directions disturb the nearly boundary-limited old loop; the masked head updates retain a feasible route-learning direction.

## Measured outcomes and the maintenance contribution

|Executed V2 arm|Gradient iterations|Parameter transactions|Final TRAIN neural/native|Held-out neural/native|Old neural/native|New operation+binding admitted|
|---|---:|---:|---:|---:|---:|---|
|Full coupled|31|32|1 /1|1 /1|1 /1|Yes|
|Within only|100|100|.25 /.25|0 /0|1 /1|No|
|Class only|100|100|.25 /.25|0 /0|1 /1|No|

The extra full-arm transaction is one own-native confidence maintenance, not a32nd terminal-gradient iteration. Every full gradient iteration changes parameters; class-only has only six nonzero parameter iterations and94 zero commits. Counters explicitly include accepted zero transactions, so “100 updates” must not be read as100 nonzero learning steps for that arm.

The full arm reaches TRAIN neural/native1 before maintenance. Its learned new-token call probabilities then are approximately(.445276,.054346,.500378) for skip/B/A: the correct hard choice was acquired from feedback, but its confidence is weak. The independently reconstructed combined-alphabet length16 path-mass bound is1.52886e-5. The single allowed maintenance uses only the current own argmax E/R/calls as targets, leaves shared and loop-head weights fixed, fits the existing head on current features, and preserves the hard graph. The fit residual is1.78e-15; it adds zero P and zero external target information. The same guarded transaction installs it at scale1, raising the combined bound to.999778834 and permitting admission of operation1 and token1. This sharpens an already learned choice; it does not discover or hand over the correct route. The resulting final E/R/call probabilities must not be attributed to terminal-gradient optimization alone.

The four held-out examples are accessed only after the final actual weights are frozen/saved. The code constructs and discards the helper's held-out tuple earlier, but never reads those targets in gradient construction, maintenance, selection, stopping, or retries. All four held-out combinations and the six old checks pass after freezing and after actual reload. They have lengths at most3. **There is no executed length16 test here**: length16 is the scope of the sufficient path bound and the protected native reference. Nor is there later learning after new admission to empirically test this new operation's retention through a second new acquisition. The conditional certificates are valuable but not replacements for those future measurements.

## Complete relevant cost scope

The full arm uses31 within and31 boundary VJPs,62 chamber representative evaluations,372 differentiable terminal-label accesses, and716 total recorded feedback/label accesses including evaluations. It tries127 proposals, rejects95, and commits32. Three commits use heads fallback. Actual execution includes122 program calls,11,526 accepted continuation steps,34,578 root-strand steps, and102,046 derivative/Newton/polynomial evaluations. The32 transactions require192 old-segment checks,720 new-domain segment checks,214 vertex comparisons,96 native relation pairs/token checks, and104 path-mass checks. These counts coexist with cached receipt reuse; they do not mean every changed old path was freshly lifted.

Final learned-weight storage is240 bytes (30 doubles), and fixed feature buffers add64 bytes. The full-arm derived cache contains118 maps:2,832 raw int64 payload bytes and64,463 serialized-cache bytes including receipts. The actual final `.pt` is33,783 bytes; all five delivered checkpoints total116,258 bytes. The different serialized cache and Torch checkpoint formats overlap rather than forming disjoint additive storage terms. Receipts, binding reference, configuration, and cache state remain real overhead;30 P is not the total system memory.

The full arm's recorded wall is.6955724s, with.2604296s continuation,.1381472s transaction scope,.0174760s checkpoint delivery, and.0022504s maintenance. Transaction/continuation/maintenance scopes overlap and must not be summed as separate total cost. The whole isolated runner reports8.9440s including startup and the other two arms. Both gradient components are computed and charged even in ablations; early stopping follows the prospective TRAIN+admission criterion. These data do not support claiming the ablations' compute efficiency, a hardware energy advantage, or comparison with the large six-stage Transformer experiments.

The checkpoint restores actual weights, receipts, binding references, class cache, and policy configuration. It contains no hidden optimizer tensors because this is explicit SGD. **The external Python smoothing RNG state is not saved in this format.** Inference reload and cold map reexecution are actually verified; bit-exact interrupted training continuation is not. Future long-lived runs should persist that RNG state and noise/transaction position, or explicitly reconstruct and verify them from the already saved traces before claiming exact training recovery.

## Claim judgments and constructive continuation

- **DIRECT, supported:** one supplied old computation plus one newly acquired noncommuting operation/call can coexist in an actual learned E/R/shared-controller/call model. Its new loop map is executed rather than learned as a free W matrix, and its call starts uniform. The targeted same-gradient block fallback repairs the observed joint-policy deadlock without adding P or feedback.
- **DIRECT, conditional:** old class retention under genuinely moving loops is explained by the clearance guard; old task binding by a separate native closure check; bounded soft agreement by the path-mass condition. These mechanisms are implemented together, rather than equating protection of a primitive with protection of a task.
- **PARTIAL:** full-versus-component ablations show that these two derivative components and the feasibility repair are useful in the tested initialization. They do not establish that no other optimizer, representation, or ordinary program learner could obtain the same behavior.
- **Not established:** independent Monodromy competitive gain, scalable state/primitive growth, task-independent Block generality, multiseed performance, or strong Transformer/DEN/HOPE superiority. The ordinary boundary optimizer can use precisely the same event formula, observations, guards, and repair. The earlier standalone ordinary/discrete matches remain valid; this integration does not erase them or convert correspondence into a universal impossibility argument.

The next useful within-Method development is to make this now-working loop-class/binding learner perform a second acquisition under its newly admitted combined contract, with the same feasible proposal policy and a counted representational budget, while retaining the same-information ordinary boundary/constraint control. This tests whether the current repair leaves meaningful future class-learning freedom rather than only completing one protected installation. An actual contrast must separate loop-class information, safe deformation, ordinary binding constraints, and own-class sharpening; do not force a win by withholding the ordinary optimizer's identical geometry or certificate. This is a constructive next research action, not an additional preapproval requirement or permission to replace the human-locked MAIN.

**CLAIM:** integrated terminal-feedback loop-class and call learning with a concrete guarded block-coordinate repair.

**STATUS:** actually executed and independently audited in a single fixed-D3 diagnostic; broader architecture/advantage claims remain open.

**EVIDENCE:** `probes/topology-architecture/runs/parameter-hybrid-loop-learning-v2/`, its complete `sync-hashes.json`, frozen `versions/code-parameter-hybrid-loop-learning-v2/`, coordinator receipt, original V1, and earlier review038/standalone controls.

**WEAKEST LINK:** whether additional protected acquisitions preserve usable learning freedom and improve a consequential resource/retention tradeoff beyond the concrete same-information ordinary constrained learner.

**FAILED VERSION:** V1's joint finite line search learns A but cannot complete its new call;72 zero commits and final TRAIN.25/held-out0 are preserved. Class-only and within-only V2 failures are unchanged, not relabeled as successes.

**NEXT SINGLE DECISION:** extend the working integrated learner through another genuinely new protected acquisition, with matched ordinary learning/cost, rather than treating this successful repair or this review as campaign completion.
