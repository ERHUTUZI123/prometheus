# Five-arm acquisition contribution: what produced the N16 improvement

**The successful intervention is the joint history/state/continuation construction, installed as real neural operations. Extra explicit old-action proposals made no difference in this instance.** Success was already present before the additional200 Adam updates. MAIN=TOPOLOGY remains human locked; M0 is unpassed and Transformer remains the primary architecture baseline.

This analysis directly read each arm's `records.jsonl`, `candidate-audit.json`, materialization report and all three saved curves under `runs/n16-acquisition-contribution-v1/learn/arms/`. It did not run new training or read a teacher function. Root raw SHA256: `67d3821ba48f337a28342b31e16ef829ee035683d8003d87e3764c778d94a2f9`.

## Actual controlled execution

All five arms restored the same actual V8 source parameters, RNG and old task0/1 references. The raw layout reports verify source strides as well as values. Every arm performed200 updates using the same saved batch indices, batch64, LR.01, terminalCE+.01geometry, with no old-data SGD. **No arm protected the current task during those200 updates.** Final validation admission occurred only afterwards. Thus an earlier current-task consolidation is not an explanation for the successful arms.

All old-task train/valid/IID/OOD/length16 native and actual continuous accuracies were1 at steps0,100,200 in every arm. Their common old protection does not distinguish a topology-specific retention benefit.

| Arm | New IID, step0→200 | Final OOD / length16 | Distinct witnessed histories, step0→200 | Final parameters |
|---|---|---|---|---:|
|continue_only|.464844→.464844|.253906 / .162109|16→16|10,984|
|native_recompile|.464844→.464844|.253906 / .162109|16→16|10,984|
|one_step|.412109→.476563|.240234 / .160156|16→16|12,058|
|history_full|1→1|1 / 1|32→32|16,354|
|history_no_explicit_reuse|1→1|1 / 1|32→32|16,354|

The history-state counts above were recomputed from the actual saved encoder and symbol6 transition, for the32 histories `(x,empty)` and `(x,[6])`; they are not teacher state labels. Continuous output agreed with native graph output on every evaluated batch.

## Where acquisition happened

`continue_only` improved soft train CE from1.26787 to1.16436 while native behavior and aliases remained unchanged. Another200 gradient updates did not resolve the structural error.

The native recompile control introduced zero new expert classes and preserved the same wrong native graph. Its step0 train CE rose to7.37643 after sharpening/head rewrite, then fell to1.78316, while native accuracy stayed fixed. This control establishes that this particular recompile alone was insufficient. It is a real soft-parameter intervention, not an inert placebo.

`one_step` used42 actual single-symbol6 terminal rows, added one expert, and retained only16 distinct history states. Its final native train accuracy was.49902. Terminal compatibility of one operation's immediate output did not provide the missing recurrent-state distinction in this version.

Both history arms verified496 actual pair witnesses, built4,106 observed continuation-profile entries, and considered two residual history alignments. Only one alignment was feasible. It separates all32 histories. Continuation/decode constraints globally forced all32 transitions for operations1 and2; the corresponding forced-row counts for operations3–6 were16,24,19,30.

Both arms then evaluated exactly two complete training-word candidates. Candidate0 made0/4096 errors; candidate1 made92/4096 errors. Candidate0 was selected using training labels only. Real materialization installed five new full32-state expert classes, preserving old input/call/readout behavior. **At step0, train, validation, IID, OOD and length16 were already1, before any of the extra200 gradient updates.** The gradient continuation reduced the already tiny soft CE to reported float zero and retained the acquired program; it did not cause the initial acquisition jump.

## Why the explicit-reuse ablation also succeeded

The full arm tested125 layer-wise combinations of old learned actions. Five were feasible. All five duplicated the already available learned-score constrained matching for the corresponding operator. Consequently:

- Full and no-explicit-reuse had the same two unique complete candidate programs, same selected history assignment, same five new permutation classes and same32-state native transitions.
- All200 logged CE/action/gradient-norm/shared-omega values were exactly equal between these arms.
- The no-explicit-reuse arm removed only extra old-action candidate generation. It retained the common pretrained neural parameters and their learned operator scores.

Thus this instance supports the sufficiency of continuation constraints plus the existing learned scores. It supplies no added acquisition benefit from the explicit old-action proposal search. It also does not show that prior learned knowledge was unnecessary: neither arm reset the neural score prior or representation.

## Resource and attribution boundaries

| Arm | Formation seconds | Final checkpoint bytes | Guard full / backtracked / rejected |
|---|---:|---:|---|
|continue_only|.000040|244,213|56 / 37 / 107|
|native_recompile|.024731|248,373|18 / 5 / 177|
|one_step|.028485|276,405|47 / 10 / 143|
|history_full|.221105|531,253|200 / 0 / 0|
|history_no_explicit_reuse|.212088|530,421|200 / 0 / 0|

Each history arm additionally replayed8,192 training episodes across its two candidates and checked35,176 native transition uses; both recorded7 assignment calls. Full made125 explicit reuse-source checks; the ablation made0. No new external labels were bought. All arms then used12,800 current training episodes in SGD. Equal SGD budgets therefore do not mean equal preprocessing/search work.

The history arms used five more full expert matrices than the no-formation source; one_step added one. A capacity-matched control is absent. The experiment attributes the observed improvement to the **joint history alignment/continuation-constrained operator-formation package**, with extra native capacity, not independently to state assignment alone, continuation closure alone, or a new topology-specific learning law.

The history arms'200 full guard accepts occurred after acquisition was already perfect and CE almost zero. They do not establish more freedom for nontrivial later capability acquisition. All arms used the same topological execution family and old graph guards; there was no topology-removal contrast here.

Checkpoint storage contains optimizer/RNG/reference state and materialization provenance, including stored profile/label audits. Parameter bytes or `representation_state_bytes` alone therefore understate the actual saved footprint. Retaining old data as audit evidence is distinct from replaying it in SGD, but it must still be counted in storage. The measured checkpoint bytes above include that burden; a stripped deployment state has not been measured in this study.

The strongest existing-method account is passive state/automaton identification and constrained permutation/program learning, followed by neural parameter installation and continuous geometric execution. The successful program does exhibit actual learned hidden-state use, executable composition, retention and checkpoint reload. This five-arm component test complements the Transformer comparison; it does not replace a costed primary-baseline comparison or establish M0 by itself.

## Direct evidence locations

- `runs/n16-acquisition-contribution-v1/learn/common-training-indices.npz`: common actual batches.
- `arms/*/records.jsonl`: per-update losses/guards, steps0/100/200, source layout and end-only admission.
- `arms/history*/candidate-audit.json`: source-row witnesses, profile constraints, alignments, candidate origins and costs.
- `arms/*/topo-stage2-step1000-contribution-final.pt`: actual final neural state; each arm's raw record confirms real reload and continuous execution equality.

No core or canonical method file was modified by this analysis. The next cost question concerns protected ongoing shared adaptation across growth checkpoints, not another claim that the existing acquisition result was a fresh paradigm-level success.
