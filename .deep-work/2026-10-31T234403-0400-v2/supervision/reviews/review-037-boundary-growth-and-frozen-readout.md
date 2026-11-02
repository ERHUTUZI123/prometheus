# Review 037 — actual boundary-consistent growth and repaired frozen baseline

**Decision: CONTINUE under the fixed six-TODO paper scope, with TOPOLOGY human-locked MAIN.** The N8/seed0 stream now has reproducible within-run six-stage acquisition with both neural and deployed interfaces correct at every recorded task boundary. A conventional constrained learner matches that result with four fewer parameters. Frozen and modular controls also work after a justified, fully charged readout-capacity repair. These are useful mechanism and engineering results; they do not establish an original topology-driven parameter-efficiency advantage. This review neither ends the campaign nor gates N16/N32, DEN/HOPE or subsequent primary runs.

## Independent evidence examined

I read the actual current study, shared compact/native model rules, boundary refresh and `PrivateModels.ensure_frozen_readout`, save/restore alias metadata, original freeze failure and completed resumed raw evidence. I waited for final `sync-hashes.json` markers and did not read temporary sync directories. Independent SHA256 checks cover **62 source/output files for ordinaryC-consistent,69 for modular-consistent,48 for repaired Freeze**. The ordinaryC and modular raw records contain respectively 3,000 resumed and3,600 actual updates; Freeze's repaired resume contains1,800, exactly600 per executed stage. Existing stage0/first-three-stage checkpoints supply the earlier actual work, which is not relabelled or rerun.

Beyond summaries, I decoded the owned PyTorch ZIP metadata and numeric storage with a restricted, local allowlist reader rather than loading Torch or executing a model. It resolves only known tensor/storage descriptors, NumPy RNG metadata and ordinary container/path objects. This verifies physical storage sharing and values; it is not independent neural inference. **All21 stored model-state entries from the real pre-repair stage2 checkpoint are byte-identical in the final stage5 checkpoint**, and the source SHA matches the actual resume record: `e33b4a6fdc9ee3a49b51bc7adf93d192180e1599166951992225440b7918f578`. The final unique stored model-tensor scalar count is independently6,384, matching the resource report. Actual GPU reload/output checks are recorded execution evidence, not rerun by this reviewer. No new training, SSH or geometric experiment was performed.

## What actually grew

All methods in this comparison use the same raw terminal task files. Native variants share the terminal-profile formation procedure, full-TRAIN closure/candidate checks and a real neural installation; none receives teacher state, target operator tables or correct call routes. All600 updates/task remain counted. Boundary consistency maintenance uses only own current hard classes/routes, adds zero learned parameters and is available to the ordinary controls.

| Completed N8/seed0 policy | P after stages0–5 | Stage-boundary strict retained tasks | Final four-split seen-task interfaces |
|---|---|---|---|
| Topology compact + boundary refresh |1596,2232,2868,3946,4630,5314|1,2,3,4,5,6|all neural/deployed1|
| Ordinary constrained compact + same refresh |1592,2228,2864,3942,4626,5310|1,2,3,4,5,6|all neural/deployed1|
| Frozen shared backbone + priced compatible readout |1536,2480,3424,4496,5440,6384|1,2,3,4,5,6|all neural/deployed1|
| Separate compact task modules + same refresh |1352,2704,4056,5592,7128,8664|1,2,3,4,5,6|all neural/deployed1|

The first successful capability pays the entire initial allocation. Later increments for Topology are636,636,1078,684,684 P; ordinaryC has the same increments. Freeze pays944,944,1072,944,944; the exceptional1,072 includes the128-P readout repair. Modules pay1,352 per new non-memory task and1,536 per hidden-memory task. These are actual policy differences, not solely different execution caches.

This supports an economical shared/compact representation relative to the tested whole-module policy. The matched ordinaryC result is the necessary stronger explanation: destination-coordinate compression, shared encoder/token/call structure and data-driven program formation explain the current parameter saving without Monodromy. OrdinaryC is not excluded or weakened. Conversely this exact version-level correspondence does not prove topology cannot help some other growth/learning mechanism.

## Boundary correctness is not uniform neural correctness during training

The repaired stream's strict score is evaluated at task boundaries, after the fixed refresh, under IID≥.95 and OOD/L16≥.90 for both interfaces. It is correctly1…6 there. Topology's intermediate100/300-step curves have old neural minima about .348/.309/.252/.242/.178 across later stages before returning to1 at step600. Observed deployed native behavior remains correct. Thus the supported statement is **boundary-consistent neural growth with observed deployed retention during training**. It is not evidence that the soft neural interface never temporarily forgets. A stronger always-online soft guarantee would need another explicit maintenance schedule and charged cost.

The original unrefreshed stream's strict count1 and failed deltas remain historical evidence. The prospective boundary-maintained continuation is a distinct version and has not retroactively rewritten it. As in review036, new-task validity already reaches1 after terminal-profile formation/NN installation at step0; the gradient updates are real but are not the first discovery of those already installed programs.

## Why the frozen readout repair is fair

The original frozen policy succeeds on the first three tasks, then its fixed decoder lacks enough latent states carrying some observable output labels for the new terminal response profiles. That is a specific representation bottleneck, not evidence that parameter freezing generally fails.

The new rule counts the observed output multiplicity required by distinct TRAIN continuation profiles. It first searches existing **learned** decoders for one satisfying both the old input encoding and those counts. If none fits, it allocates a private D×N decoder initialized to a compatible output-label assignment inferred from the same terminal profiles. It does not supply transition maps, routes or hidden-mode labels. Candidate formation must still solve the executable transition model and pass full TRAIN checks.

Actual stage3 requires output counts `[2,2,2,2,2,2,2,2]`, creates one16×8 decoder and charges **128 P**. Stages4/5 reuse that compatible task3 decoder with **zero extra readout P**, not another adapter each time. Current trainable counts are1,072 /944 /944, while total stored counts remain4,496 /5,440 /6,384. All old encoder/token/decoder weights and prior private task weights are unchanged in the byte audit.

Saving records `decoder_shared_with`: final owners are `[None,0,0,None,3,3]`. The actual serialized decoder storage keys are `[1,1,1,15,15,15]`; all six encoders and all six token weights share their respective original storage. Restore first verifies saved values, then reattaches the intended Parameter aliases. Its actual reload resources recover exactly the original P/trainable counts at every stage. This rules out both uncharged copied readouts and accidental double-counting of shared ones in this run.

Boundary refresh uses `preserve_endpoints=True` for Freeze. It preserves the encoder/decoder Parameter objects, so it cannot silently detach aliases or overwrite frozen endpoints. The driver checks unchanged global P after maintenance, rebinds existing Adam state to the same ordered parameter shapes, and separately logs that maintenance. The rule is a teacher-free neural/native margin repair available to all these native policies, not a topology-exclusive mechanism.

## Resource comparison is a frontier, not a single parameter number

Actual final-stage records:

| Policy | P tensor bytes | Extra-state JSON bytes | Optimizer tensor bytes | Actual saved checkpoint bytes |600-step old-guard seconds|
|---|---:|---:|---:|---:|---:|
|Topology|21256|208326|42544|341373|47.4276|
|ordinaryC|21240|104105|42508|202737|45.6787|
|Freeze+readout|25536|83598|7568|146821|0|
|Modules|34656|81825|12316|160265|0|

The JSON representation includes receipts/history, and is not a minimal information-theoretic representation. Reference-array bytes are a subset of extra-state metadata and cannot simply be added as a disjoint term. Checkpoint bytes also include RNG, optimizer and container overhead. These measured formats nevertheless show why lower P alone cannot establish lower current storage cost. Cold path realization, cache metadata and checkpoint reload work remain real costs; the earlier certified-cache improvement does not erase them.

Final-stage boundary refresh wall times are roughly .03248s Topology, .03208s ordinaryC, .00723s Freeze and .01090s Modules. Old binding checks dominate shared-policy maintenance in this instance. Training wall times are approximately9.62/9.47/7.49/7.60s respectively, but executions can overlap other authorized GPU jobs. No replicated end-to-end throughput or hardware-energy ranking follows from these isolated timings. Full formation/search/failed-run/development costs remain in their separate real records; matching600 gradients does not match every method's total learning computation.

## Claim judgment and constructive continuation

**Supported finite result:** six terminal-feedback task models can be installed in compact neural weights, composed at unseen lengths and retained at stage boundaries with useful parameter sharing; a concrete zero-added-P consistency rule fixes a measured soft-route bottleneck. A priced readout adapter strengthens the frozen comparison rather than preserving an avoidable baseline failure.

**Unsupported current claim:** Monodromy has reduced P per retained capability relative to the strongest same-information ordinary constrained policy. Here that policy matches accuracy and growth with four fewer parameters. Current topology protects the realization family and enables certified semantic reuse across geometric updates; the profile learner, ordinary binding guard and consistency refresh explain the measured acquisition/retention score. An independent topology-specific causal learning advantage still requires a distinct tested rule.

Under the human's fixed six-TODO scope, this is one basic synthetic N8/seed0 result serving reliable growth and strong comparison. N8/N16/N32×three seeds, harder composed programs, a general noninvertible/continuous Block, real-data transfer and applicable Transformer/HOPE/DEN/Mamba comparisons are **not completed by this result**. They are the stated broader evidence coverage, not a new seventh gate or a reason to halt development. N16 ongoing work, N32 factorization and the new baseline harness are outside this review until their actual evidence exists.

Continue the already running scientific work. The useful next mechanism development must connect topology to acquisition/growth beyond shared coordinate compression and ordinary graph preservation, while these working ordinary controls remain equally capable. This review does not prescribe a new MAIN or an untested constructor as the answer.

Evidence: `probes/topology-architecture/runs/parameter-n8-seed0-topology-{through0,resumed-consistent}/learn/`; `.../parameter-n8-seed0-constraint_compact-{full,resumed-consistent}/learn/`; `.../parameter-n8-seed0-modular-full-consistent/learn/`; `.../parameter-n8-seed0-freeze-{full-consistent,resumed-consistent}/learn/`; completed raw records, real `.pt` files and each final synchronization manifest. Current source and frozen executed versions retain the original failures and incremental fixes.
