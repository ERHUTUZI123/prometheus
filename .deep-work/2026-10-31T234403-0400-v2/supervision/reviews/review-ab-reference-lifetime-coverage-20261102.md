# Frozen AB: observed KEY lifetime and OWN-address preservation

**Judgment: the reference's first SET0 transition almost removes the KEY-induced address contrast, but preserving the NN's own post-KEY address through both value-construction intervals does not repair the frozen model's query binding.** The address intervention physically succeeds and the endpoint/pair result remains adverse. This does not establish a missing HOLD action as the cause, test a learned HOLD rule, or refute all future reference-preservation mechanisms. No training, additional architecture or campaign allocation is selected by this review.

## Saved-reference token localization

The pure standard-library analysis pairs 256 raw VALID programs having the same initial state and full active word except the two write-KEY tokens. Temporal first/second values, noise, suffix, and length are identical. Before the first WRITE, the two programs differ only at the first KEY. There are 32 pairs for each temporal first value 0…7. Attention is reconstructed from saved per-cell weights as w/sum(w), with no zero denominator observed; no target/truth arrays enter this token analysis.

Mean paired attention TV is .311823817 after KEY, .311824669 after CLEAR, .003184959 after SET0, and .003185476 immediately before the first WRITE. It stays approximately .0031855 after each STEP. The same SET0 contraction occurs for temporal value zero, when no STEP executes, so the observed loss is not progressive decay over a longer STEP sequence.

| First construction event | Mean ADDRESS mass | Mean COMPUTE mass | Mean LOAD mass | Own argmax counts | Mean executed h-change L2 |
|---|---:|---:|---:|---|---:|
| KEY | .970032 | .018553 | .011407 | ADDRESS 512/512 | .077609 |
| CLEAR | .0000161 | .998619 | .001362 | compute0 512/512 | 6.383185 |
| SET0 | .995570 | .000889 | .003525 | ADDRESS 512/512 | .022683 |
| first STEP | .0000211 | .994123 | .004813 | compute0 448/448 | 10.154521 |

The executed source makes the SET0 address proposal token-dependent and common across the KEY-flip pair, while ADDRESS has almost all action mass. Thus the observed contraction is consistent with the actual address update. It identifies where the frozen reference contrast is attenuated; it does not identify the learning cause or establish that small remaining distinctions are unusable. Mean read norm grows from .189856 at SET0 to 5.478973 at the first WRITE. Fractional content processing remains active.

Token-analysis artifact: [`ab-reference-key-lifetime-token-posthoc-v1.json`](../../probes/topology-architecture/ab-reference-key-lifetime-token-posthoc-v1.json), SHA `8f951714f7345458b87b71d2c81404cbdf40e7b3851957202ea3d96bdc3dd461`; reproducible source [`analyze_ab_reference_key_lifetime_saved.py`](../../probes/topology-architecture/analyze_ab_reference_key_lifetime_saved.py), SHA `294959d88709335310ffd371b62f9b88c978ff89341fd463fdd6089bb8372f73`. Its measured local wall/CPU cost is approximately .14581 s, zero NN/SGD.

## Actual lifetime intervention and independent checks

ROOT executed `parameter-ab-reference-lifetime-cpu-v1`, two zero-SGD passes of the same 10,351P AB checkpoint in soft/native mode. At each of the two construction KEY events, the driver saves that condition's OWN updated address. For every subsequent token through and including its WRITE, it overrides only the new address with that saved vector. It supplies no correct slot or value. Public KEY→WRITE timing is privileged diagnostic information, and the query suffix receives no override. All other updates and subsequent calls remain endogenous. Consequently “suffix untouched” means no suffix override and the same transition rule; its incoming state and outputs need not equal the original execution.

Independent standard-library verification checks the actual source/file/checkpoint hashes and raw arrays. All 6,656 override events per mode match the parsed public intervals exactly. Every held before/after address, including both WRITE events, equals its OWN saved updated-KEY snapshot bitwise. Active address continuity is exact. Outside these intervals, actual after-address equals the source-proposed address exactly, and the entire query suffix mask is false. Captured read attention agrees with softmax of the actual incoming address to maximum absolute error 1.7650e-7 soft and 1.3096e-7 native; write weights match the specified execution modes.

The desired contrast is therefore preserved at physical use: first-WRITE attention TV across the KEY-flip pairs is .311823818 soft and .319855732 native, versus .003185476 in the original soft reference. This is a successful intervention on address lifetime, not a successful acquisition result.

| Frozen condition | Correct /512 | NLL | Query BOTH /192 | Query relation /192 | Either arithmetic candidate /512 | Write-order BOTH /256 | Value key0/key1 BOTH /1792 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original soft, reused | 320 | .531687 | 7 | 11 | 512 | 109 | 669 /669 |
| OWN-address held soft | 314 | .538771 | 4 | 9 | 512 | 110 | 641 /646 |
| Original native, reused | 105 | 4.738303 | 1 | 1 | 176 | 38 | 83 /88 |
| OWN-address held native | 103 | 4.791690 | 2 | 5 | 176 | 36 | 79 /83 |

All counts were recomputed from saved outputs and the public factorial program grid. BOTH requires correct endpoints; relation-only consistency is separate. Soft query-pair mean probability TV increases from .010647 to .020851, yet correct query pairs decrease. Native TV increases from .020668 to .038049 while accuracy falls. Increased response or preserved addresses must not be substituted for correctly oriented query use. These overlapping pairs are not independent seeds or significance trials.

The result rules out this exact frozen address-lifetime patch as a sufficient rescue. It does not prove that address lifetime is irrelevant: the frozen model may be adapted to its original memory/content geometry, and changing a reference without retraining changes that operating distribution. Nor does it show that a learned neutral transition would succeed. Reliable binding remains unaccepted, and the stronger content-direction audit remains a separate controlled-path finding.

Independent proof: [`ab-reference-lifetime-independent-coverage-checks-v1.json`](../../probes/topology-architecture/ab-reference-lifetime-independent-coverage-checks-v1.json), SHA `5462890f4a6603aaad02c819a28fd39dfa9d6d975e3f57355a4aa87624cd0d8a`; checker [`verify_ab_reference_lifetime_saved_coverage.py`](../../probes/topology-architecture/verify_ab_reference_lifetime_saved_coverage.py). Actual producer result SHA is `2554c3fae0ba62970081cf8bbb2bb7272d6c0b35234ac1c082c3c92c9f7808a1`. The reviewer performs no NN, checkpoint deserialization, Torch/NumPy import, SSH, training or canonical-document edit.

## Cost and decision boundary

Actual audit cost: two 512-row passes, 34,304 active transitions and 84 transition calls, zero SGD; science wall/CPU .690028/.689970 s, local-cold wall 5.944205 s, peak CPU RSS 565,080,064 B. NN payload remains 41,404 B, carry remains 328 FP32 scalars (1,312 B/example), checkpoint is 516,787 B, and live optimizer bytes after restoration are zero. The diagnostic additionally pays an eight-FP32-address holder, 16,384 B at 512 rows, and two-address snapshots, 32,768 B per mode. Shared hold masks are 21,504 B and public key/write/event positions 49,152 B; these are diagnostic state, not free model carry. Two raw files occupy 7,534,329 compressed bytes and 23,810,048 serialized array bytes. The source-preserved temporary-state sidecar distinguishes shared versus per-mode data.

The independent saved-array verification adds .33187 wall seconds; its CPU time is retained in the proof. Other review/agent compute, complete FLOPs, provider-effective compute, energy, exact allocator/cache attribution and a separate compilation bill remain unknown. Timings are nested and should not be added together; old baselines are reused outputs, not new executions. The original three training arms remain exactly 7,200 paid updates, with all earlier failed/source/admission costs preserved separately.

`CLAIM | STATUS | EVIDENCE | WEAKEST LINK | NEXT SINGLE DECISION`: preserving OWN construction-reference lifetime is sufficient to repair this frozen AB model | not supported; contrast preservation succeeds but binding/accuracy do not improve | exact raw snapshot/interval/attention checks and paired outcomes above | acquisition of correct content/query roles is still unresolved; inference surgery does not test a learned rule | retain the adverse intervention and choose no architecture or learning-rule change from the HOLD explanation alone. TODO3 remains development and Primary remains unfrozen; the campaign is not closed.
