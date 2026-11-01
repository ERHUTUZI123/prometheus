# Review032 — completed continuous-primary v7 results at N8 and N16

2026-11-01. **M0 NOT PASSED.** Continuous monodromy now participates from stage0 in actual three-stage training/evaluation at two operand scales. N8 learns/retains well. N16 exposes a significant acquisition/admission failure, including a protected arm whose final new-task result is below the tested Transformer. The record supports continued construction inside the same human-locked topology MAIN, not a positive M0 declaration or a claim that topology as a direction is false.

The human currently requests the full completed results. No new experiment or new mechanism development was launched for this review. The unimplemented direct-neural-loop proposal remains OPEN only. MAIN=TOPOLOGY USER_LOCKED/HUMAN_ONLY; M0 current; M3 ultimate; Transformer primary. Online USER-PARK and standalone Persistent State pause remain unchanged.

## Actually examined and independently checked

Read `continuous_native_model.py`, v7 changes to `train.py`, inherited neural materialization/protection/workspace code, `continuous-primary-protocol-v7.json`, three completed runner receipts, both primary raw files/summaries and the N16 Transformer/Replay/Adapter raw file/summary. Read the stored N8 supplemental baselines for comparison; those are reused completed runs, not claimed as new v7 executions.

Actual primary source hash train.py `401d57d065fde2894407aa52add9ae14694c5e841ac8769dcc95c76b7f2092a4`; continuous model `aaf3aa828fed79f7211170222cfd9f93f9d23ebf08c0c099e4e591a804096105`. Runner source/receipts match prospective protocol. Actual three runner receipts report exit0/completed, full process times152.2690s (N8 ON+OFF),180.5318s (N16 ON+OFF),75.0318s (N16 three baselines), under the existing900s external bounds. These include startup and multiple model arms, not per-model training-only time.

Independent read-only replay of the saved native graphs against the saved original NPZ arrays checked **864 curve/task/split accuracy entries** across both scales and ON/OFF. Every replay matched its actual raw hard accuracy, including poor outcomes. Independently traversed75 recorded ON old-reference closures, using only actually admitted reference graphs; all passed. No neural training or repeated geometry experiment was run to obtain this audit.

Matched all15 N16 data-array hashes and all3 minibatch schedule/old-index hashes between primary and baselines. Matched the same15 dataset and3 schedule hashes between N8 v7 and each reused N8 Transformer/Replay/Adapter/program-v2 run. Information and sample schedules therefore match the declared conditions; equal update counts still do not imply equal total computation or representation budgets.

## What v7 actually executes

The primary model is instantiated with four trainable shared geometry weights at stage0. Existing terminal CE, temporary router MI, native operation formation, old behavior contracts and data-triggered workspace growth remain in the same neural learner. Native hard evaluation actually obtains its primitive maps by cold continuation through unordered configurations, then caches those maps for the learned encoder/calls/decoder to use. Exact cache signatures include D, all assigned bank permutations and actual omega bytes. Hard evaluation is not secretly bypassing the continuous realizer.

Geometry training uses the supplied normalized kinetic action and its four-parameter gradient. It does not supply the terminal semantic acquisition gradient: permutation classes still originate in the learned soft operation/call system and materialization. The mathematical shear protection and finite continuation bound reviewed in031 remain valid. Current topological scope is a real continuous realization of a permutation quotient, not full pure-braid-sensitive computation or a physical quantum experiment.

Every arm has800 updates/stage, batch64, three stages, seed0, common terminal-only task data and public task ID. Primary LR .01; tested Transformer/Replay/Adapter LR .003. Replay splits its total batch32current/32old; other arms consume64current. Both native arms use continuous geometry and old-preserving formation/growth transactions; `topo_off` disables only step/temperature guards. It is not an all-topology-removed ablation.

## N8 — stage-end IID acquisition and retention

Entries are new-task IID / old-task IID in chronological order. Primary columns are actual continuous-primary v7; Transformer/Replay are actual stored same-data LR.003 supplemental runs.

| Stage | Native ON | Native OFF | Transformer | Transformer+Replay |
|---|---|---|---|---|
|0|1.0000 / —|1.0000 / —|.955078 / —|.955078 / —|
|1|1.0000 /1.0000|1.0000 /1.0000|.617188 /.695313|.589844 /.988281|
|2|.974609 /1.0000,1.0000|.974609 /1.0000,1.0000|.408203 /.437500,.291016|.398438 /.970703,.576172|

N8 native new-task OOD8–12=.906250, length16=.835938. Old tasks0/1 remain1.0 on their tested IID and longer compositions. Both native arms grow D8->16 from the actual16-history terminal witness and finish K10,3,610 parameters. Formation occurs at stage0 step100, stage1 step200, stage2 step200. All three stage references are actually admitted in both arms. ON accepts all2,400 updates fully, with no backtrack/reject or temperature rejection. Thus the additional stepguard has no demonstrated benefit here.

Stored N8 Adapter final new=.328125, old=.984375,.476563; ordinary program-v2 final new=.515625, old1.0,1.0. The earlier **separate** witness repair and neural continuation reached all tested splits100%, but that was extra discrete search plus200 updates on an earlier saved checkpoint. It is not secretly applied to v7 and is not an equal800-step result. The later200-step continuous extension is another separately charged completed experiment; Review030/031 delimit those facts.

## N16 — acquisition failure appears

| Stage | Native ON | Native OFF | Transformer | Transformer+Replay | Transformer+Adapter |
|---|---|---|---|---|---|
|0|1.0000 / —|1.0000 / —|.943359 / —|.943359 / —|.947266 / —|
|1|.593750 /1.0000|.955078 /1.0000|.562500 /.730469|.533203 /.970703|.402344 /.947266|
|2|.314453 /1.0000,.593750|.460938 /1.0000,1.0000|.388672 /.341797,.382813|.320313 /.960938,.558594|.228516 /.947266,.402344|

Final new-task soft IID is .400391 ON, .509766 OFF. Final new-task OOD is .113281 ON, .253906 OFF; length16 is .072266 ON, .158203 OFF. Thus ON's final new-task hard IID is lower than the tested Transformer (.314453 vs .388672), though old task0 retention is much better. OFF has higher final new-task IID than the tested Transformer and much better old IID, but is slower and retains the same continuous topology family/transactional preservation, so this contrast cannot establish a causal topology advantage.

Both N16 arms actually grow D16->32 from32 distinguishable terminal histories. ON ends K7,8,836 parameters; OFF ends K9,10,984. State growth happened, but it did not solve new-task acquisition. No N16 program-v2 arm was executed in this completed set; do not invent its result.

### The crucial ON/OFF interpretation

N16 ON stage1 never materializes or admits task1: native valid=.619141, final joint proposal valid=.845703, below the .95 formation threshold. It therefore finishes task1 native IID=.593750 (soft=.750000). Its stage1 guard accepts **all800** proposals at alpha1, with **zero** temperature rejection. It is factually wrong to say the poor stage1 acquisition was caused by guard rejection or that learning was blocked by repeated projection.

OFF reaches joint valid=.953125 at stage1 step800, materializes two new native classes, and admits task1. After stage2, its task1 native IID becomes1.0. The observed branch point is therefore late proposal quality plus threshold-triggered formation; **assignment/admission sensitivity is a hypothesis, not an experimentally identified cause**. In particular, the alpha1 ON path reconstructs parameters as `before+(proposal-before)` whereas OFF returns without that copy; floating-point/numerical trajectory differences are a concrete possible source of later divergence, alongside normal GPU numerical sensitivity. No counterfactual test has isolated that explanation, and none was run for this report.

N16 ON protects only actually admitted task0. Its task1 native IID happens to stay .593750 after stage2, but task1 was never admitted. That unchanged score is **not** an all-word protected capability theorem. ON stage2 has799 full accepts and one positive-alpha backtrack, no rejection; this cannot retroactively explain the earlier stage1 failure. N16 OFF has admitted tasks0/1 but no stepguard; its preserved old behavior is observed, with formation/growth transaction checks retained.

## Resource results and accounting scope

Per-model total wall includes training, evaluation, maintenance and checkpoint work for that model; synchronized training-only time is a narrower recorded category. They are not interchangeable, and the model-wall totals do not include all earlier development/tuning/failed runs.

| Scale / method | Three-stage training s | Model total s | Final parameters | Final optimizer bytes | Final retention resource |
|---|---:|---:|---:|---:|---|
|N8 native ON|28.233|80.005|3,610|28,912|1,728 relation bytes + up to768 map bytes|
|N8 native OFF|27.928|66.826|3,610|28,912|1,728 relation bytes + up to768 map bytes|
|N16 native ON|43.261|93.709|8,836|70,720|640 relation bytes + up to1,536 map bytes|
|N16 native OFF|38.299|80.956|10,984|87,904|1,536 relation bytes + up to1,536 map bytes|
|N16 Transformer|16.389|23.847|41,056|328,580|no replay/old relation|
|N16 Transformer+Replay|14.876|18.988|41,056|328,580|1,376,256 replay tensor bytes|
|N16 Transformer+Adapter|20.823|26.739|52,224|35,760|old task modules included in parameters|

Map byte entries above are to be read from the actual final/cumulative report; graph/reference byte figures count serialized arrays, not complete Python object overhead or physical RAM. Parameters and optimizer tensors are not a complete memory/energy comparison. Adapter's final active-after-mask parameter count is4,080, versus total52,224; reporting only its total as trainable would be misleading. No physical hardware energy was measured.

**Important growth cost:** N16 ON geometry moments are rebuilt335 times cumulatively. Stage2 alone adds323 rebuilds,63,003,648 quadrature point-strands and12.958855 seconds of moment construction (cumulative13.030334s). N16 OFF stage2 adds174 rebuilds,36,731,392 point-strands and7.612429s (cumulative7.689350s). Churning hard classes repeatedly invalidates the current all-active-class quadrature summary. This is actual measured overhead, not a theoretical bound. The observed extra parameters are cheaper than the Transformer totals, but native elapsed time is worse; the full resource frontier cannot be reduced to parameter count.

Final cumulative cold realization: N8 ON31,418 snapshots/1.512287s, OFF31,396/1.542356s; N16 ON42,993/2.511991s, OFF51,359/3.069354s. All reported cold lift attempts succeed. Warm cached execution totals are about1.45–1.55s per native model across its repeated evaluations; these are aggregate batched evaluation work, **not** a measured single-query p50/p95 latency. Moment/class-cache counters are cumulative across stages; do not sum stage-end cumulative values as if disjoint.

Matched schedules and source inspection support the fairness of stated information/update comparisons. The tested Transformer is a real2-layer d48 model with learning-rate repair and replay/adapter variants, but these measurements do not establish that all competitive Transformer sizes/optimizers have been exhausted. Single-seed development outcomes are not a confirmatory superiority test.

## Generalization and M0 A–E

IID datasets are not wholly novel strings: N8 task0/1/2 have512/291/156 of512 IID rows also present in training; N16 has499/238/122. These overlaps were recorded prospectively and identically shared across arms. Longer8–12 and16 evaluations provide a more demanding separate composition check. Do not label the entire IID score unseen-combination generalization.

| M0 condition | Current evidence judgment |
|---|---|
|A Acquisition|Supported for actual finite operation formation and strong N8 stages; N16 fails to master later stages. Partial overall.|
|B Retention|N8 admitted old capabilities and N16 admitted task0 are preserved in the checked scope. No protection claim for unadmitted N16 ON task1. Partial overall.|
|C Competitive advantage|N8 gives a strong finite accuracy/parameter contrast, but slower total execution; N16 ON new acquisition is below tested Transformer and OFF performs better. Important reproducible protected advantage not established.|
|D Scaling|Two operand scales actually tested, only seed0. Capability degradation appears atN16; no multi-seed reproducibility or acceptable general scaling law. Not met.|
|E Mechanism|Actual continuous monodromy and class-preserving shear mathematics/execution are established. Causal contribution to an important continual-acquisition advantage is not isolated; OFF keeps topology and common preservation. Not met.|

**M0 remains NOT PASSED; M1–M3 unpassed.** This reports specific versions, not a verdict against all topology. The outstanding technical bottleneck is acquisition/formation at larger state spaces plus costly geometry-class maintenance, with no identified architecture-level protected-learning advantage yet. These are concrete observations to inform later development after the human has received the completed results, not an excuse to rename ordinary compilation as innovation.

## Exact evidence

Relative to the same V2 run:

- `probes/topology-architecture/runs/continuous-primary-v7-n8-seed0/learn/{records.jsonl,summary.json}`;
- `probes/topology-architecture/runs/continuous-primary-v7-n16-seed0/learn/{records.jsonl,summary.json}`;
- `probes/topology-architecture/runs/continuous-primary-v7-n16-seed0-baselines/learn/{records.jsonl,summary.json}`;
- all three `.runner-status.json` receipts and local hash-sync manifests;
- `continuous-primary-protocol-v7.json`, actual source snapshots under `versions/`, and the previously preserved v6b baseline/repair/fusion evidence.

Raw SHA256 respectively: N8 `3b96bcd0b2c5f7ebd42b39338d239661e2eaf5cc32e823e57b8bf48264566653`; N16 `f30ba20a49d41b6bc79d856721766e0274f8b0be4672320994765d8258e33484`; N16 baselines `c7cea619dc27bb72061cb4f694ab4ef2fd3295e5d772a627498b3a1738d0cc9d`. Their summaries SHA256: `cf4ad8c34702dfae0237b2e1975a7da8f86d73d1229cbd5828db23b85d2f12c5`, `3fc9f80e7cd989ecd173a5a365173c28f38452b2273c3beae9266c7cdd40a7bd`, `3050401ac1cbd7d3d0fda672753264cf58d76262e29974c42fc7af180bebecb8`.

CLAIM: completed end-to-end continuous-primary training at two scales with actual Transformer/replay/adapter comparison. STATUS: actual executed development evidence; M0 not passed. EVIDENCE: checked code/raw/receipts and independent864 native replay/75 admitted-reference checks. WEAKEST LINK: robust acquisition and identified protected-learning benefit beyond supplied permutation/geometry scaffolding. FAILED VERSION BOUNDARIES: N16 ON later-stage acquisition, scale sensitivity, geometry maintenance churn; previous positive/negative raw remains unchanged. DECISION: report the complete results now; preserve all evidence and OPEN constructive work. This is not campaign completion or a change to MAIN.
