# Review030 — terminal-witness repair installed in the neural executor

2026-11-01, actual post-execution review completed after source, raw records and checkpoint metadata inspection. **CONTINUE_TOPOLOGY.** The experience-driven repair is a real new discrete learning step, is installed in the actual neural model, and survives 200 further terminal-trained updates. It resolves the specific third-stage state alias without adding parameters. This closes the claim that the saved native-v6 alias was unlearnable with its available capacity; it does not close the topology hypothesis or the campaign.

MAIN=TOPOLOGY USER_LOCKED/HUMAN_ONLY; M0 NOT PASSED; M3 ultimate. Online remains USER-PARK; standalone Persistent State remains paused/reserve. Transformer is the primary architecture baseline. This review is constructive supervision of an executed increment, not a pre-execution gate or a direction change.

## What was actually checked

Read the complete current `witness_operation_repair.py` and `integrate_native_repair.py`, their prospective protocols, the relevant native graph/materialization/protection implementation, all repair candidate events, integration update/curve/install events, both summaries, runner receipts and checkpoint event metadata. The inspected source hashes agree with the protocols and actual runner receipts:

- witness repair: `12e4c0b63a181e08faeba48044af04988a23c39baef3783bb9648d4521c219f4`;
- neural installation/training: `076ec180fc6331b21359935028ab64f933be6c741853a8edee4ab6ba1e86c613`;
- topology model: `48f8f7aa5efe96297bef04e29ca6c52b8bd8bc38a2af409ce772619679e77acd`;
- semantic promotion: `d569785c90e02189ced289d043563ce8c08d97ecda04903e77494771f256035e`.

Independent read-only arithmetic used the existing campaign NumPy environment. I directly verified all 120 pair witnesses against the original task2 training NPZ records; replayed all 15 repaired saved-graph splits against their original NPZ targets, obtaining zero errors; and traversed both old closed state relations independently, obtaining 8 related reachable pairs for each old task. This was evidence replay, not retraining or a new experimental arm. The default system Python initially lacked NumPy; using the already-existing campaign environment required no installation and did not alter evidence.

I also checked that integration raw events contain exactly updates 1–200, all accepted at alpha=1; all four saved checkpoint files match recorded SHA256/byte sizes; and their ZIP pickle streams contain the explicit integration provenance/extra-update metadata keys. Pickle opcode inspection was non-executing. I did not import PyTorch or independently deserialize/forward the saved neural tensors; actual neural forward/update evidence comes from the inspected executed program, its regenerated graph/soft metrics, matching receipts and saved files. That boundary is not concealed.

## Mechanism and information discipline

The actual original histories `(start=1,prefix=empty)` and `(start=1,prefix=[6])` were mapped to one learned state even though terminal training suffixes distinguish them. The data witness already establishes at least 16 distinguishable histories, and 16 states were already available. Thus this case calls for operation/representation repair rather than more capacity.

Candidate generation (`witness_operation_repair.py:211–242`) uses the currently aliased history prefix and its learned expert. It considers all other destination states, prioritizing equal currently learned decoder output, and swaps two images of that expert. This maintains bijectivity. The proposed bank edit propagates to every saved call of that shared expert (`:201–208`). Candidate selection uses only current training terminal errors and actual history-alias reduction; old task graphs must remain closed-equivalent. Validation is a single final no-degradation gate. IID/OOD/length16 arrays are opened after the repair is frozen.

The executed 15 proposals produced one admissible edit: expert10 previously fixed states 1 and 9; it now swaps their images. The chosen permutation is obtained by the declared generator/selection rule, not supplied as a correct teacher toggle. All 120 observed history witnesses are checked from terminal records; no intermediate teacher states or generator functions are imported. The learned decoder makes both states currently output 1, so the edit preserves that immediate readout while enabling the distinguishing continuation to act on separate states.

The repair search excludes the current task's contradicted provisional reference but retains tasks 0/1. That is the intended scientific correction: protecting an imperfect current estimate must not make contrary terminal feedback unusable. It does not rewrite historical raw results or withdraw an older task's preservation contract.

## Installation and continued actual learning

`integrate_native_repair.py:321–370` regenerates the real checkpoint's native graphs and requires exact correspondence with the repair's declared source before installation. It changes expert10's actual operation-logit tensor, leaves encoder/decoder/call parameters unchanged during the installation transaction, regenerates the graph, and checks the admitted maps and all old reference relations. It revises only task2's provisional reference after the admitted repair. Actual installation changes all 256 expert10 logits because the whole corrected map is sharpened; the discrete structural edit itself is two images. These are different counts.

Fresh Adam then trains every available parameter on the original task2 terminal training examples, 200 updates of batch64 at LR .01, with no old-label replay and no router auxiliary. There are 12,800 additional current training sample uses. The all-word guard remains active on tasks0/1 and the repaired current contract. All 200 updates are fully accepted; no backtrack or reject occurs.

Important causal separation: native and soft accuracy are already 1.0 immediately after installation. The extra Adam steps do not discover the missing edge; the discrete terminal-driven repair did. The subsequent training demonstrates that the corrected neural implementation can continue learning while preserving its installed native semantics. Task2 IID CE falls from .0142066 to 6.77252e-6, and length16 CE from .0882555 to 4.24442e-5. All three tasks, all five saved splits, have native and soft accuracy 1.0 at recorded post-installation steps 0/50/100/200. Parameter count stays 3,606; no additional workspace or expert is allocated by this repair.

## Executed results, costs and failed attempt

Repair v1 first failed before search because its supplied dataset path did not exist. That failed receipt/stderr remains preserved. v1a fixes the input path and runs the same source hash, exit0; no result is reconstructed for v1.

Repair v1a performs 15 candidate attempts, 30 old-relation checks/240 relation pairs, two training scores, 46,080 terminal episode evaluations and 214,334 native transition uses including final evaluations. It buys zero new external labels. Original task2 errors are train111/4096, validation19/512, IID13/512, OOD48/512, length16 84/512; all become zero. Search plus evaluation takes .1083185 seconds inside the script; sandbox process wall time is 4.177862 seconds. These timings have different scopes.

Neural integration exits0 with sandbox process wall time 17.962157 seconds and script wall time 12.340629 seconds. Recorded synchronized work: installation .022853s, additional training 3.117772s, guard .655021s, evaluation 4.977530s. Guard records contain 600 relation checks/6,400 relation pairs. Evaluation uses 92,160 soft episodes, including pre-installation and four post-installation evaluations. These costs supplement the original stage's 800 updates; they are not retroactively included in an equal-800 comparison.

Checkpoint hashes independently checked:

- installed step800: `1724fe9f7220703a82ca7f8282822e62ef653d2426fc2c9eaf812ca605baa1b9`;
- trained step850: `dcad212d233dbffafaab09e78b4afe0f0f2bd5524527d15dda5bedefab50cdb1`;
- trained step900: `8bafcd4335aad177faab5d8d310d00ef9501a4544f6c296136089a17a57c6435`;
- trained step1000: `c441a2c6e49fbf43c21665076440a4e8c7f03f30f43693c58e40309454418028`.

## Scientific assessment and constructive continuation

The supported new mechanism is **terminal-experience-driven revision of an already formed executable primitive**, followed by actual neural installation and continued shared training. It adds an acquisition/repair operation to a prototype that previously only grew capacity and protected its partly incorrect estimate. The existing hard representation had sufficient capacity; SGD's current route/materialization dynamics did not exploit it completely. This execution gives a concrete way to use witnessed history distinctions rather than declaring acquisition impossible or adding arbitrary dimensions.

The strongest current conventional explanation remains finite-state/permutation program repair with a learned neural proposal and semantic installation transaction. This result does not identify a causal benefit from continuous topological protection: no continuous path executor participates in these two runs, the extra stepguard never rejects, and expert10's repaired call is specific to the current task. Previous baseline curves must remain visible; a final repaired100% versus Transformer800-update score is a development comparison with extra resources, not matched-budget superiority. Researcher-level test inspection has already occurred, so these old heldout splits are developmental evidence, not a fresh confirmatory sample.

Continue the SAME architecture. The next consequential integration is to make its actual learned primitive/encoder/call interface execute through the continuous layer and train a charged class-preserving shared implementation objective. Preserve the clear split between (i) terminal-driven creation/revision of classes and (ii) useful class-preserving realization learning. Compare each effect against its applicable conventional control, retaining Transformer as the architecture baseline. A new path representation alone is not the claimed M0 advantage; its useful update and lifecycle cost effect must be measured. This recommendation does not require stopping current experiments or a new round of pre-approval.

The current canonical method's row-state convention is consistent with code: `alpha <- alpha Pi`, with matrix entry `Pi[q,Pi(q)]=1`; successive first/second calls multiply in that written order. Write state-map closure as `T_a(q)` to avoid silently treating the state as a column vector. The two-image update in method section5.2 preserves bijectivity under that convention. Its complete loop should mark continuous execution/training according to actual implementation status and keep this executed repair distinct from the earlier unmodified800-step evidence.

## Evidence locations and decision

Relative to the same V2 run:

- `probes/topology-architecture/runs/native-v6-witness-repair-v1/` and its failed runner receipt;
- `probes/topology-architecture/runs/native-v6-witness-repair-v1a/repair/{config.json,events.jsonl,summary.json,frozen-native-graphs.json}`;
- `probes/topology-architecture/runs/native-neural-repair-integration-v1/learn/{records.jsonl,summary.json,repair-training-batch-indices.npz,topo-stage2-step1000-repair-trained.pt}`;
- both `.runner-status.json` receipts;
- prospective `witness-repair-protocol-v1.json` and `neural-integration-protocol-v1.json`; actual immutable source snapshots under `versions/code-witness-repair-v1/` and `versions/code-neural-integration-v1/`.

**CLAIM:** actual terminal-driven primitive repair, real neural installation, continued terminal SGD and preserved saved old behavior. **STATUS:** supported within the single N8/seed0 development instance; M0 unpassed. **EVIDENCE:** independently checked terminal witnesses/saved-graph replay, inspected executed code, original events/receipts/checkpoints above. **WEAKEST LINK:** no demonstrated important topology-specific acquisition/retention/resource benefit or multi-scale reproducibility yet. **FAILED VERSION:** initial wrong input-path invocation, preserved before any search; original native-v6 alias/partial task2 scores remain historical facts. **DECISION:** CONTINUE_TOPOLOGY with actual continuous-layer integration and charged learning, not campaign finish or direction reassessment.
