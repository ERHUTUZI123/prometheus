# Frozen AB: fixed-strength content-direction controls

**Judgment: the frozen soft AB computation depends on the directions of its nonchosen-write contents, beyond the retained write schedule and source-vector norms. These controls reject an amplitude/coverage-only explanation in this conditional pathway, but establish neither semantic binding nor a learning cause.** Candidate-answer computation is strongly affected; correct query-role selection remains scarce in every condition. Continuous soft computation remains legitimate. Native performance is a separate observation, not an acceptance requirement imposed by this review.

This is an executed zero-SGD mechanism audit serving TODO3 and the Fundamental Question of how learned Blocks construct and use internal operations. The original three research arms remain exactly 7,200 updates. There is no new learner, parameter variant, architecture selection, training, TEST use, or broad-advantage claim here.

## Exact controlled contrast

ROOT executed `parameter-ab-write-direction-controls-cpu-v1` with the one completed AB checkpoint: 10,351 parameters, checkpoint SHA `912abc79925e9014146fafdec7f538282198bbddcc44034a8d7852021ebfbf85`. The driver and core are frozen at SHA `e065a2b0539c30f03fc57dda6411f1a63451b173fef4ed347c2fd8d49ded925e` and `93bdfea778c2f079ec04a6696ab8df77dc8bcc52b7f3fabdc57c05cbcad5250f`. All four raw conditions and replacement signals precede target loading under the checked source ordering; this is a dataflow restriction, not a claim of OS-enforced blindness.

The reference captures every row/time/slot weight w = alpha_STORE a and OLD h. Each replay retains exactly those weights, coverage recurrence, and saved reference OLD h at unchanged/chosen writes. The caller, h, controller, address and output otherwise evolve from each replay's actual incoming state. Thus this is a reference-conditional memory-content pathway intervention, not the total effect of changing an endogenous policy.

Both treatments use the same mask M. A public identical-prefix query pair must be active and reference-nonchosen STORE in both rows; eligible groups receive a fixed nonzero cyclic group derangement. The canonical first row of a donor group supplies one direction for both recipient query twins. After query divergence this is correctly called canonical-direction replacement, not a permutation of the full row-content multiset. The other condition substitutes the direction of the last strictly past reference-selected STORE content, using fixed e0 before the first such event or if the cached direction is zero. Every substituted direction is rescaled to the recipient's reference OLD-h norm. Unchanged writes also use saved reference contents.

The matched quantities are w, its per-row/per-slot sums, z, retention/decay coefficients, and each injected source-content norm. Therefore each cell's injected term norm ||w u|| is matched. The resulting ||V||, ||V/z||, net update ||w(u−V)||, read norm and temporal/vector coherence are not matched. Norm scalars, schedules and unchanged reference contents deliberately retain factual NN information. A directional effect cannot be called a pure semantic-information effect or complete information removal.

## Independent raw checks

The reviewer used only standard-library ZIP/NPY arithmetic: no neural execution, Torch/NumPy import, checkpoint deserialization, SSH, training or canonical-document edit. Reproducible checker: [`verify_ab_write_direction_controls_coverage.py`](../../probes/topology-architecture/verify_ab_write_direction_controls_coverage.py). Machine-readable proof: [`ab-write-direction-controls-independent-coverage-checks-v1.json`](../../probes/topology-architecture/ab-write-direction-controls-independent-coverage-checks-v1.json), SHA `c5ac176a7861ef47306d7b02bf6816eb156a81004a54e47d182a7426c4f38ef0`. Actual result SHA is `6491dd29446709108d403dbfdd1089177f4f108d7c9e2a051cf09004bbb29cc4`.

`VERIFIED, DIRECT`:

- Every sham saved array is bitwise equal to reference, including full h/read/action trajectories, coverage, all final carry fields and outputs; the two complete NPZs have the same SHA. This is stronger than merely equal final accuracy.
- w and full coverage trajectories are bitwise identical across reference/sham/direction replacements. The paired mask, group derangement, canonical donor choice, strict-past cache, first-selected times, e0 cases and untouched contents were independently reconstructed.
- All 16,912 reference nonchosen events are treated; excluded events and excluded write mass are zero. Permutation changes actual contents at 16,892 events; cached replacement changes all 16,912. Source norms match to maximum absolute error 1.7635e-7 and 2.3830e-7, respectively. Donor zero-norm fallbacks: zero. Cached/e0 cases: 12,472; 272/512 rows never have a reference-selected STORE.
- Explicit backward coefficients lambda_i = w_i product_(j>i)(1−w_j) reconstruct final V with maximum absolute error 7.3702e-7 and final z with error 1.7267e-7. The independent linear identity Delta V_T = sum_(i in M) lambda_i (replacement_i − reference_h_i) agrees with actual final-memory differences to 7.9529e-7. This independently checks the targeted pathway without rerunning a model.

The producer's effective-mass inventory computes `(1−w)` in FP32 before multiplying a float64 accumulator. Its saved masses were separately reconstructed to 3.33e-16; exact real-weight lambda masses differ from that mixed-precision inventory by at most 1.1084e-7 per row/slot. This small numerical distinction is recorded rather than treating saved arithmetic as an exact symbolic identity. It changes no reported score or causal contrast. The first local verifier invocation had an invalid 1e-12 comparison between those different arithmetic conventions; the corrected verifier preserves both reconstructions, without changing any experiment or model tolerance.

The true-weight final retained strength is approximately 54.11965 from nonchosen writes versus 138.75858 from selected writes. Of the former, 52.84304, or 97.64%, precedes the first reference-selected STORE. This is a widespread early feedback pathway, not a small set of excluded or later exceptional events. Effective strength here denotes summed recency coefficients, not semantic information or memory-vector magnitude.

The previous native/drop event raws were separately SHA-checked against their retained producer result, SHA `18895561da465bca752343e81d5dda79fbfd4b63651cf4653323cb60cd505ad2`. They were parsed, not rerun. The new reference differs from the earlier CPU soft logp by at most 1.9074e-6. Compared with the saved original GPU VALID outputs it has maximum logp difference 2.2888e-5 and zero argmax differences; numerical identity across row shapes/devices was not presumed.

## Frozen VALID outcomes

All counts below were recomputed from raw outputs and the original public full-factorial programs. Query columns use the 192 pairs with distinct target answers; each endpoint must be correct for BOTH. Relation-only consistency is reported separately and is never counted as endpoint success. Gaps are percentage points against the one fixed previous native VALID observation, not an additive causal decomposition.

| Condition | Correct /512 | NLL | Query BOTH /192 (rate) | Query relation /192 (rate) | Accuracy minus fixed native, pp |
|---|---:|---:|---:|---:|---:|
| Reference /sham | 320 | .531687 | 7 (3.646%) | 11 (5.729%) | +41.992 |
| Canonical donor direction | 215 | 3.504600 | 6 (3.125%) | 11 (5.729%) | +21.484 |
| Cached-selected direction + e0 | 200 | 2.235468 | 2 (1.042%) | 3 (1.563%) | +18.555 |
| Previous native, reused | 105 | 4.738303 | 1 (.521%) | 1 (.521%) | 0 |
| Previous endogenous drop, reused | 153 | 3.831787 | 0 (0%) | 0 (0%) | +9.375 |

| Condition | In either arithmetic candidate /512 | Write-order BOTH /256 | Value key0/key1 BOTH /1792 | Nondegenerate query-pair mean probability TV |
|---|---:|---:|---:|---:|
| Reference /sham | 512 | 109 | 669 /669 | .010647 |
| Canonical donor direction | 328 | 57 | 299 /285 | .009019 |
| Cached-selected direction + e0 | 320 | 47 | 274 /274 | .018991 |
| Previous native | 176 | 38 | 83 /88 | .020668 |
| Previous endogenous drop | 234 | 52 | 165 /163 | .021084 |

Maximum query-pair probability TVs are respectively .043795, .102460, .043517, .199206 and .087521. Probability-level response remains nonzero even when few argmax answers change; small TV alone does not prove erasure of a reference. A larger response can accompany incorrect direction, as the cached/native/drop results illustrate. The proof retains query logp differences as well as these probability distances.

The reference scores exactly at the finite-grid 5/8 query-invariant ceiling, with seven both-correct and seven both-wrong nondegenerate query pairs. The replacements mainly damage membership in the two-answer arithmetic candidate set; they do not reveal successful query binding hidden behind aggregate accuracy. Cached replacement has worse accuracy but better NLL than donor replacement, so neither should be called a universally better control. These overlapping pairs from two templates and one frozen training seed are not independent experimental replications.

The controlled replacements reduce accuracy by 105/512 and 120/512 despite matching w/z/decay/source norms. Thus a theory that only those matched quantities matter is insufficient for this frozen execution. The result supports dependence on content direction/geometry in the specified pathway. It does not identify which directions encode operands, prove correct logical key binding, isolate a learning defect, show that the remaining gap is attributable to a particular action, or establish selected-STORE/native sufficiency. Cached compensation still performs dense writes at reference nonchosen times. The old endogenous deletion changes its future schedules and source contents; its 153/512 is not a factorial level that may be subtracted to partition amplitude versus information effects.

## Complete available cost and decision boundary

The actual audit paid one frozen 10,351P/41,404 B NN restoration and four full 512-row passes: 68,608 active NN transitions, 168 transition calls, zero SGD. Measured science wall/CPU time is 1.702655/1.701930 s, local-cold coordinator time 7.029691 s, peak CPU RSS 580,460,544 B. These clocks are nested, not additive. Carry remains 328 FP32 scalars (1,312 B/example); live optimizer bytes are zero after checkpoint restoration, while the 516,787 B checkpoint includes historical training state. Four raw trajectory files plus replacement signals occupy 20,048,751 compressed bytes and 40,102,912 serialized array bytes. Sham/reference duplicate payloads are still separate retained files and paid I/O.

The independent successful saved-array check adds 3.06119 wall/3.05038 CPU seconds; the earlier failed arithmetic-convention check paid approximately .395 s local command wall, with its CPU time unmeasured. Further source/provenance reading and agent compute are not fully measured. Complete audit FLOPs, provider-effective compute, energy, exact allocator/cache/driver-storage attribution and a separate full compilation bill remain unknown. Native/drop costs are the previously paid audit, not charged as new NN executions here; all earlier architecture/training/admission costs remain retained.

`CLAIM | STATUS | EVIDENCE | WEAKEST LINK | NEXT SINGLE DECISION`: nonchosen writes matter only through amplitude/coverage | rejected for the two fixed, norm/schedule-matched reference-conditional replacements | bitwise sham/w/z, exact replacement reconstruction, lambda/DeltaV checks and frozen endpoint/pair outcomes above | direction changes also alter geometric coherence and downstream state/read amplitudes; retained reference signals can carry information | record this scoped content-direction dependence and keep the remaining question on acquisition of correct query roles, without selecting a new learner or expanding the completed 7,200-update budget. Primary remains unfrozen; this review does not end the campaign.
