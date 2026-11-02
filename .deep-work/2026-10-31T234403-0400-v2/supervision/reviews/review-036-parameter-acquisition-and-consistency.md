# Review 036 — six-stage acquisition, actual dual-interface repair and Transformer failure

Decision: **CONTINUE construction and experiments inside human-locked TOPOLOGY.** The N8 stream acquired six correct executable task models in a compact neural architecture. The original trajectory failed its stricter neural-plus-deployed retention criterion. A subsequent teacher-free, parameter-preserving consistency refresh repairs all six current tasks and identifies a concrete soft-routing failure. This is substantive development; it does not retroactively pass continual retention or establish topology-specific parameter efficiency. The completed Transformers have an acquisition failure that deserves the already initiated diagnostic and repair, rather than being presented as strong solved-task rivals.

This review follows actual execution. I read the current parameter model, task generator, terminal-profile learner, study/restore code, Transformer implementation, consistency helper and diagnostic driver; original Topology and four completed Transformer raw logs/summaries; the refresh report, raw events and real checkpoint. I independently checked all 14,400 Transformer update events, all six stage batch-index arrays against Topology, 72 actual Transformer checkpoint SHA256 values and the refreshed checkpoint SHA256. I also reconstructed the path-mass argument from the actual forward rules. I did not run new training, inference, SSH or a geometric experiment. Executed reload/accuracy results below are checked against the actual recorded artifacts, not independently rerun.

## Actual acquisition and growth mechanism

All learners receive the same terminal-only files: six unknown reversible task processes, all public six-symbol words through length three represented in TRAIN, plus random words through length eight. Tasks 3–5 have an unlabelled hidden-state dependence; no mode, teacher transition or correct call is returned. This is a deliberately informative public observation-table interface, not merely random sparse terminal sampling.

The native learner forms observed history response profiles, checks continuation closure, aligns them to its current learned encoder/readout, solves a reversible transition completion, checks all original TRAIN examples, and installs resulting own executable actions and calls into real neural Parameters. It grows workspace only when the observed separated classes exceed current D. Both compact topology and ordinary constrained models have the same learner and representation (`terminal_profile_formation.py`, `parameter_native_models.py`).

Each expert stores D trainable coordinates xi rather than D² scores:

\[
W_{kij}=-8(j-\xi_{ki})^2.
\]

Every permutation is represented by integer destination coordinates; rearrangement identifies the hard matching. The D² generated scores are still real inference/training work, not stored Parameters. This compression is a genuine representation change but is explicitly available to the conventional constraint control. The continuous family adds four shared geometry parameters; no topology-exclusive parameter saving is built into this representation.

For the real N8 Topology lineage, 600 updates per task were executed, with stage0 preserved and only stages1–5 resumed. P is 1,596 / 2,232 / 2,868 / 3,946 / 4,630 / 5,314. The new task's VALID neural accuracy is already 1 immediately after terminal-profile formation and NN installation, before Adam, at every stage. Therefore actual task acquisition here comes from that data-driven formation/installation procedure. The 600 gradient updates are real but cannot be credited with first discovering these already acquired programs.

Deployed execution remains correct on all six task test suites. Original neural accuracy degrades on old tasks: after stage5 task4 is .578125 IID / .275391 OOD / .177734 length16, while deployed execution is 1. The original stage-end strict stable count remains one throughout; later parameter increments correctly yield `deltaP_per_new_retained_capability=None` when an old stable capability is lost. This score is stability at recorded stage checkpoints, not a guarantee about every intermediate neural update. Original failures remain evidence.

## Concrete repair and why it works

`refresh_native_consistency` reads only the model's own current encoder/decoder argmax, each expert permutation and **both existing slot routes**. It fits the same call head to those own targets with larger margins, sharpens encoder/readout to their own existing classes, and resets expert coordinates to their own existing hard destinations. It does not read labels, add an expert, change D/K, replace two-slot calls by a whole-operation lookup, or consume new task information.

For row-stochastic soft execution, a particular native computation path supplies the lower bound

\[
\Pr(\text{own native output})\ge p_Ep_R(p_{\rm route}p_{\rm op})^{sL}.
\]

Here each p is the minimum actual probability of the corresponding own hard choice over the declared domains, s is slots and L the maximum word length. All other paths have nonnegative mass. A bound exceeding one half makes that output the unique soft argmax. This argument uses the actual stochastic execution and preserves native semantics; it does not require teacher answers. The floating evaluation of the minima is not an interval-certified theorem about machine arithmetic.

Actual stage5 repair:

- P 5,314 → 5,314, K 38 → 38, D 16 → 16; all parameter shapes/flags and native task graphs checked unchanged.
- Minimum correct-slot probability .420014 → .998894. Minimum operation probability .999189 → .999330. The L16 path bound rises from 8.57e−13 to .944706.
- All six tasks × four recorded splits become neural=deployed=1; the real NN checkpoint reload reproduces all 24 split-interface results.
- Zero new labels and zero gradient updates; one SVD, one least-squares solve, 456 charged graph assignments and 144 relation pairs. Refresh synchronized time .076194 seconds; full before/after/reload diagnostic wall 10.985176 seconds. The report acknowledges possible baseline overlap, so these are not standalone throughput-superiority estimates.
- Parameter replacement requires a fresh optimizer; the diagnostic resets Adam state and records this. Future training cannot silently reuse the old optimizer objects or call this zero-maintenance learning.

The refreshed actual checkpoint SHA256 is `7f7266207b1a8381e76f73f1e9597faccd1a755519e8231c63d5396af13cc4ec` (288,501 bytes). Original historical `Qstable=1` is retained in the new payload and summary. Restoring six current capabilities after the fact is not evidence that none had been lost during the original trajectory. A prospectively integrated refresh policy can now be tested without treating this posthoc repair as that experiment.

This mechanism is available to the ordinary constrained model too. The intervention gives specific evidence for insufficient neural/native call margins and accumulated soft-path leakage, rather than incorrect native operation classes. It does not by itself isolate which of encoder, expert-coordinate and route sharpening is necessary, since all are changed together. The route probability is the conspicuous measured bottleneck; necessity remains a narrower causal question.

## Completed Transformer results and the next useful diagnostic

All four baselines completed six phases × 600 real updates on matched task files. Their saved current-batch index arrays exactly match Topology. Replay uses 32 current + 32 old examples per step versus 64 current without Replay: identical total batch size, different current-data allocation. All Transformer64 models store 103,752 P; Transformer96 models store 229,352 P. All recorded strict stable/current-passing counts are zero.

| Actual model | Task0 after first phase IID / OOD / L16 | Task5 after final phase IID / OOD / L16 |
|---|---|---|
| Transformer64 | .183594 / .123047 / .134766 | .115234 / .109375 / .101563 |
| Replay64 | .183594 / .123047 / .134766 | .119141 / .117188 / .105469 |
| Transformer96 | .134766 / .109375 / .103516 | .132813 / .140625 / .119141 |
| Replay96 | .134766 / .109375 / .103516 | .113281 / .128906 / .115234 |

TF64 phase0 mean CE is 2.10796 for its first100 updates and 2.07492 for its last100, near ln(8)=2.07944. Thus poor final performance is already an acquisition failure, not merely catastrophic forgetting. Code inspection finds ordinary start/token/task embeddings, position encodings, padding mask, shared attention/FFN and terminal head, with finite nonzero gradients. This inspection does not prove the optimizer or architecture is adequate.

The requested length-stratified **existing checkpoint** TRAIN/VALID diagnostic is now coded but not claimed executed here (`transformer_length_diagnostic.py`). It can distinguish failure to learn even one-step maps from failure of longer composition without new labels. The prospective fixed-budget short-to-long sampling repair is scientifically justified; its outcome remains unmeasured in this review. Larger width alone has not repaired the problem. Treating these failed baselines as demonstrated strong acquisition competitors would overstate the current evidence.

Native methods additionally inspect full TRAIN profiles, solve matchings and perform whole-TRAIN candidate evaluation before the 600 gradients. Their information is the same, but equal gradient counts do not mean equal total learning work. Those passes, CPU solves, temporary D² activation work, old contracts, replay, optimization state and geometry costs belong in the comparison. Full `.pt` sizes include persistent `model_extra`/materialization history; `reference_bytes + parameter_bytes` alone is not complete model storage.

## Scope and scientific decision

The ordinary compact control's initial run failed during checkpoint reload because the former generic stride helper assumed every model had continuous-cache invalidation. The current callable-guard fix is an engineering recovery; do not count that exception as a learning failure. Its completed continuation and the remaining growth-policy/scale/seed comparisons are not claimed here.

Supported: compact real NN operation formation on six terminal-feedback tasks; exact deployed retention in the observed stream; a practical zero-added-P consistency repair that restores current soft composition. Unsupported: an important topology-specific parameter-efficiency advantage, prospectively maintained dual-interface performance over a repaired stream, or M0 passage. The original four-geometry-parameter shear protects operation realization within its family; task acquisition still uses terminal profile learning and binding preservation still uses explicit graph relations.

**Next substantive execution:** finish the ordinary compact/control continuation, test the prospectively integrated consistency policy on the current stream, and run the fixed-budget Transformer acquisition repair. These are different live empirical obligations; this review neither gates their start nor ends the campaign. MAIN remains human-locked TOPOLOGY, M0 current and M3 final.

Evidence locations: `probes/topology-architecture/runs/parameter-n8-seed0-topology-{through0,resumed}/learn/`; `.../parameter-n8-seed0-topology-refresh-diagnostic-v1/learn/`; the four `.../parameter-n8-seed0-{transformer64,replay64,transformer96,replay96}-full/learn/` directories; corresponding raw records, actual checkpoints, frozen source snapshots and runner receipts.
