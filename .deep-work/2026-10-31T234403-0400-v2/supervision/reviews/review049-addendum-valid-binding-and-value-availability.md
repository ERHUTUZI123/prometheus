# Review049 dated addendum — frozen VALID binding and value availability

2026-11-02. This adds actual diagnostic evidence to the historical review049 without changing that review or its original claims. **The observed RAW .625 VALID result is consistent with weak query-role binding, supported by paired predictions rather than the score alone. Literal memory availability also degrades between writes and reads. Neither read-mass normalization nor a universal representational impossibility follows.** These are single-development-seed diagnostics; no architecture advantage, Primary freeze, new novelty or TEST selection is established.

The independent reviewer used only saved JSON, NPZ/NPY payloads, original source bytes and SHA, with standard-library arithmetic. No model, Torch/NumPy, checkpoint deserialization, SSH, compiler or training was executed by this reviewer. ROOT executed all original CPU/GPU diagnostic jobs. The posterior/value class and CPU checks were originally authored in this agent branch; that source authorship is not independent evidence of their correctness. The independent contribution here is recomputation of raw predictions, formulas and conditional memory statistics, with explicit limits.

## Exact conditional ambiguity and source-role evidence

The original source computes the final operation as `(3*q + cells[address]) % 8`. The saved VALID truth covers all 64 ordered value pairs, both query orders, both write orders and two fixed noise groups: 512 rows. The verifier checked the original VALID SHA `7d8c81c0e41bcea94785fc63593f781b6042b0d2b87844d841db2d0a1875a31b` and privileged original counterfactual truth SHA `7f08f29f671a483dd6b979723dc81359408fade781cd72a7ee2d6ef6afb08418`. Query-flip pairs have byte-equal prefixes and differ only in the two query KEYs; write-order-flip pairs preserve the suffix, noise and target. Physical slot index is never equated to logical key.

Let `y0=(3*v0+v1)%8`, `y1=(3*v1+v0)%8`. Equality holds iff `2*(v0-v1)=0 mod8`: 8 equal-value and 8 difference-four pairs. A predictor insensitive to the query selector, conditional on the other matched features, can therefore achieve `16/64 + (48/64)/2 = 5/8 = 320/512`. This is an exact conditional ambiguity bound on this balanced grid. A model scoring .625 need not be query-insensitive, and an arbitrary predictor can exceed it by using the selector.

| Frozen final NN | Correct /512 | Prediction in {y0,y1} | Changed prediction on 192 query flips with different targets | Changed prediction on 256 target-preserving write flips |
|---|---:|---:|---:|---:|
| original POSTERIOR | 135 | 215 | 19 | 140 |
| original RAW | 320 | 512 | 20 | 58 |
| input-history shared-address RAW | 319 | 508 | 11 | 61 |

For original RAW, only 10 of the 192 nondegenerate query pairs are correct at both query orders; 172 have exactly one correct, 10 neither. The shared-address NN has 6 both, 179 one, 7 neither. Both get all 64 same-target query pairs correct and unchanged. The original RAW agrees with fixed logical0/logical1 formulas on 318/322 rows and temporal first/last on 310/330; the shared-address NN agrees on 317/319 and 337/299. No one of these formulas describes every prediction. Two-candidate coverage plus weak correct query response and unwanted write-order sensitivity localize a role-binding gap; they do not prove that the NN explicitly decoded both operands or selected one fixed erroneous rule.

Every old per-row prediction/formula field was recomputed from raw final cached logp and original truth, then compared with the prior binding JSON. The new shared-address prediction vector came from the same already-executed stage3 VALID forward. Full per-value-pair, noise-group and pair counts are in [independent saved binding recheck](../../probes/topology-architecture/saved-binding-independent-recheck-plus-shared-address-v1.json), SHA `e368ffd78a1695e00c5923e16767efe039d7b4a9c1bc3eae7cf10fd2ccb9033b`, produced by [stdlib verifier](../../probes/topology-architecture/verify_saved_operand_role_binding.py).

## Value-coordinate availability and survival

For each saved event, the analysis reconstructs `u = attention @ Vp`, slot masses `m_j=sum(Vp_j)`, read mass `m=sum(u)` and conditional `q=u/m` only when `m>1e-6`. The conditional mixture weights are `attention_j*m_j/m`; raw attention alone is insufficient. It reports both `max_j Vp_j[y]/m_j` with the winning slot's mass and `max_j Vp_j[y]`. Best-any-slot is an oracle diagnostic upper bound, not a learned retrieval policy. Read norm is recomputed from saved `u` and E; entropy/prototype cancellation can reduce norm even with mass near one.

| NN/event | Cached current q correct | Read argmax true selected cell | Some slot argmax true selected cell | Read mass mean | Conditional true probability / best slot conditional mean |
|---|---:|---:|---:|---:|---:|
| POST stage2 READ | 64 | 177 | 399 | .556681 | .312074 / .519776 |
| POST stage3 MIX | 135 | 116 | 265 | .648510 | .212814 / .393035 |
| RAW stage3 MIX | 320 | 64 | 114 | .990460 | .124036 / .214961 |
| shared-address stage2 READ | 59 | 64 | 100 | .937903 | .138557 / .320921 |
| shared-address stage3 MIX | 319 | 66 | 96 | .989058 | .131568 / .214694 |

Original POST stage2 emits the true q at both WRITE events on 512/512 rows, but first-value best-any-slot argmax availability falls from 512 after WRITE1 to 410 after WRITE2 and 351 before READ. In the new shared-address stage2 NN it falls from 512 after WRITE1 to 120 **before** WRITE2. The second written value is available in some slot on 512 rows after WRITE2, then only 100 before READ (412 lost, none gained). In shared-address stage3, first-value availability is 512 after WRITE1, 224 before WRITE2 and 94 before READ; the second value is available on 144 after WRITE2 and 96 before READ, while its cached WRITE q is correct on only 72 rows.

These are availability changes between sparse snapshots, not attribution to one token: continual writes during intervening SET/STEP/noise may overwrite or reencode content. They quantify literal task-value coordinates, not all information in h/controller/keys or indirect recurrent reconstruction. Prefix/WRITE probabilities received only terminal supervision. Low alignment does not prove that relevant information is absent. Stage2 excludes its unseen MIX token; its final KEY-step read is not an actual MIX-conditioned read. Conditional normalization leaves read argmax/entropy unchanged, and the newer RAW mass is near one; the observations do not establish mass attenuation as the sole bottleneck.

Original four-NN raw analysis is preserved unchanged in [value/slot analysis](../../probes/topology-architecture/posterior-valid-value-slot-analysis-v1.json), SHA `7195e90904cd4f3dde99c2fda39a5af04a9be4b1082f2513e8bdfc7aa6902d76`. The new two-NN extension is [shared-address value/slot analysis](../../probes/topology-architecture/shared-address-valid-value-slot-analysis-v1.json), SHA `147d710fbd44568f78fbb518e8470fbc8b5a3f1116969d1b4ddfc70e0091e97d`. Both retain derived per-row NPZs, all/equal/distinct-value strata, E Gram/norms and attention-mass identities; new reconstruction error is below 1e-7.

## Diagnostic costs and evidence boundary

The privileged prefix truth was generated once, checked on all 17,664 prefixes and reused: CPU algorithm wall .125164s, remote runner4.444564s, local coordinator5.327880s, output58,569B. Its source SHA is `8943cd6dd4d638224dcc7be62df82e505b871b32bbcf1be5de3be3afadb92726`; truth NPZ SHA `bff1c36f7b7951f9dc2ad012e847a7b2300666283fde61b65af6eef27368d176`. The truth is privileged audit material. NN raw files were completed before the audit loaded it, but the same UID could read the file: this is a trusted source-ordering boundary, not OS-enforced blindness.

| Additional diagnostic scope | NN passes / active transitions | Science wall / CPU s | Remote runner / local coordinator s | Saved raw payload |
|---|---:|---:|---:|---:|
| original POST/RAW stage2+3 | 4 /67,584 | 4.146853 /4.147315 | 8.864531 /9.828137 | original audit logical output payload15,427,501B |
| shared-address stage2+3 | 2 /33,792 | 1.897561 /1.900076 | 6.459168 /7.466744 | event NPZ files4,103,609B; decoded array payload11,898,880B |

Both audits use 248 logical FP32 carry scalars/example, include raw event stores/D2H/serialization and perform zero SGD. Original stage2/3 P are13,381/13,397; new shared-address10,789/10,805. New audit CUDA peak allocated10,808,320B/reserved25,165,824B and CPU peak RSS1,164,926,976B; original audit CPU peak RSS1,172,881,408B. These allocator/RSS values are not complete driver/context or provider memory. Full FLOPs, GPU busy time, energy and provider-effective compute remain unknown. Nested times must not be added together; prior training/admission/failures remain separately charged.

Independent saved-array binding arithmetic cost .053252s wall/.046143s CPU; original slot analysis .798175s/.785935s, peak RSS81,870,848B; new slot analysis .386136s/.380687s, peak RSS61,931,520B. They add zero NN queries or learner feedback. The new 2400-update run was separately checked using all12 saved TEST vectors (12,288 soft/native row predictions), all2400 update logs, original20 input hashes, four current/three replay arrays, eight source/checkpoint files and 28 raw all23-gradient gates: [raw verification](../../probes/topology-architecture/actual-shared-address-fourstage-independent-raw-checks-v1.json), SHA `0224c38fb2ca6bb035d04efb01ae242de0c800f0bd6240b4b53a34cc22e46b65`. Stage0/1 VALID per-example snapshots were not saved, so their curve values remain checked producer aggregates. Its P10,805, update24.481718s, science50.282021s and local coordinator55.896138s are a capability–cost tradeoff, not a win from lower parameters alone.

`CLAIM | STATUS | EVIDENCE | WEAKEST LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

`Probabilistic VALUE / shared-address history yields executable binding | ADVERSE on this acquisition grid; exact conditional ambiguity identity and saved-array localization verified | original fourNN + newer twoNN events, paired binding and full raw/source/cost chain | sparse events and terminal-only codes do not identify total hidden information or a unique causal repair | projection and shared-address results retained, no mass-only remedy inferred | ROOT has prospectively selected one persistent learned input-history address repair; this addendum executes no new model or variant and does not tune TEST.`
