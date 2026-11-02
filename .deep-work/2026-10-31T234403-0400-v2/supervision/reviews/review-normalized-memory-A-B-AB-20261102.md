# Normalized-memory A/B/AB: independent scientific node review

**Judgment: the fixed three-arm study improved acquisition of a two-candidate output, but did not establish reliable query-conditioned binding, native execution, or broad capability–complete-cost superiority. TODO3 remains development; these models do not justify Primary freeze.** A gives the best observed native new-task accuracy and removes 640 parameters. B/AB give higher soft accuracy while introducing a large soft/native discrepancy. The new human causal criterion makes the distinction decisive: query changes must select the correct role, rather than merely cause an output change or leave an answer inside the two-candidate set.

This review follows the current human priority of broad, significant, reproducible capability–parameter–complete-resource advantage. Monodromy remains optional; this review changes no strategic allocation. It connects to the Fundamental Question by testing whether a learned Block acquired an internally executed, query-controlled operation, rather than only a useful differentiable recurrent path. It closes neither that long-term question nor this campaign.

## Evidence actually inspected

The reviewer read the executed core, masked helper, Graph engine and four-stage trainer, all three actual `learn/result.json` files, the three independent-training-check JSONs, the saved TEST counterfactual parser/metrics and table v2, and the actual CPU v2 failure/v3 result. The frozen per-run source maps were checked against their eight source files. The three `records.jsonl` hashes and all 7,200 actual update rows were independently checked. No Torch/NumPy import, model execution, checkpoint deserialization, SSH, training, Method/state edit, or additional experiment was performed for this review.

`VERIFIED, DIRECT` within its numeric scope: independently decoded all 84 saved gradient NPZ files, rather than relying on their PASS summaries. Each contains the same 20 parameter names, finite reference/candidate values, and output/loss/gradient agreement at the frozen tolerances. These cover 929,488 scalar pairs: A 303,856; B 321,776; AB 303,856. Maximum gradient absolute differences are respectively 7.1526e-7, 2.9802e-7, 3.2783e-7; maximum output difference is 5.7220e-6 for each arm. There are 28 fixed-current-NN checks per arm. These checks establish sampled eager/masked Graph correspondence; they are not independent mathematical verification of every derivative or every training step.

The reviewer also independently decoded the saved prediction NPZs and raw programs for all three arms × three TEST splits × soft/native. Raw token structure, initial state and distractor sequence determine each template; targets verify the decoded arithmetic only after that grouping. All reported accuracies, query-pair endpoint successes/changes, write-order endpoint successes and candidate-set coverage match direct recomputation. A further 64,512 value-pair comparisons independently reproduce both-endpoints-correct plus modular-relation counts and relation-only counts. These operations reuse frozen outputs, perform zero NN forwards/SGD, and introduce no training labels or acceptance feedback into the completed runs.

Actual evidence roots are under [`probes/topology-architecture`](../../probes/topology-architecture/):

- `runs/parameter-typed-normalized-memory-{a,b,ab}-n8-seed0-fourstage-graph-v1/learn/` contains results, records, checkpoint files, gradient raws and same-pass frozen TEST outputs.
- `typed-normalized-memory-{a,b,ab}-independent-training-raw-checks-v1.json` provides the separately checked source/sampler/checkpoint/cost inventory.
- `saved-test-counterfactual-original-plus-A-B-AB-metrics-v1.json` and `saved-test-counterfactual-original-plus-A-B-AB-table-v2.csv` retain the posthoc intervention arithmetic and its provenance.

Result SHA256s are A `79f673926df1d4464e9d3a21d67654ec67b86f8d4abb3707de0fb95d2fbce270`, B `2d1e6f6641ccd9d6a39cac07f29975305e82bae6d19ef32f54e46fa836899187`, AB `ad8b0d6105f4848a68f63d79676075f2871397029010d0a168136b4228ba3421`.

## What the fixed intervention changed

All arms actually executed four fresh 600-update stages, with the original terminal-NLL objective, Adam .003, clip 1, sampler, Replay128, growth from 15 to 16 token rows, and two fixed Graph captures. The three research paths total exactly 7,200 updates; no historical NN was loaded or extra training purchased. A and AB share their exact initialized parameter hash; B shares the historical unpruned initialization. Coverage adds no parameter or RNG draw.

A changes only the address proposal to an affine function of the raw learned token embedding: it retains the original full-proposal factory/RNG sequence, then takes columns 48:64. It does not remove context from the learned ADDRESS action probability. The caller still receives old h, current token-only GRU control, token plus slot embedding, and the memory read; its seven-way action budget, conditional four-expert mixture and G8 gates remain intact. A cannot be interpreted as a learned general content-addressing rule: its proposal is token-dependent, while contextual invocation remains learned separately.

B retains the value write exactly and adds per-cell coverage, initially zero. With w_m = alpha_STORE a_m, the executed soft equations are

\[
V'_m=(1-w_m)V_m+w_m h_{old},\qquad
z'_m=(1-w_m)z_m+w_m,\qquad
r=\sum_m a_m\frac{V_m}{\max(z_m,10^{-4})}.
\]

Both read and STORE use OLD address attention; STORE uses OLD h; ADDRESS changes later-token access. B normalizes each cell before attention. It is not posthoc normalization of the summed read. Native hardens the seven-way action/expert choice while retaining soft cell attention. The actual learning runs used soft forward execution, not native or straight-through training.

`RECONSTRUCTION`: above the coverage floor, the first nonzero write gives V_m = w_m h and z_m = w_m, hence V_m/z_m = h in every covered cell. A small fractional STORE may therefore create a full-amplitude shared memory path despite not being the selected native action. It does not itself separate logical keys. Its immediate normalized-slot derivative with respect to that first-write amount/attention is zero away from the clamp boundary, because numerator and denominator cancel. Below the floor, the slot remains attenuated by w_m/epsilon. Later distinct writes produce cell-dependent recency-weighted averages and can still supply address/strength credit; this is neither an impossibility proof nor a diagnosis established for the trained trajectories. The source admission tests verify these exact local equations and finite derivatives, including the floor boundary, without assigning teacher action roles.

## Capability and the query-blind alternative

The new IID results below are counts on the frozen 512-row grid. Query columns use only the 192 pairs for which the correct answer changes. Write-order pairs must preserve the answer. Value-pair counts require both endpoint answers correct and the correct modular difference; relation alone is not counted as success.

| Arm/mode | P | Correct /512 | In two-candidate set /512 | Query both correct /192 | Query changes /192 | Write both correct /256 | Value key0/key1 both correct /1792 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A soft | 10,351 | 301 | 480 | 1 | 1 | 130 | 577 /577 |
| A native | 10,351 | 298 | 469 | 2 | 4 | 126 | 564 /578 |
| B soft | 10,991 | 320 | 512 | 1 | 2 | 115 | 698 /698 |
| B native | 10,991 | 37 | 66 | 0 | 1 | 15 | 4 /7 |
| AB soft | 10,351 | 321 | 512 | 5 | 9 | 100 | 674 /673 |
| AB native | 10,351 | 93 | 152 | 1 | 34 | 32 | 64 /74 |

The original fresh typed control has 196/512 soft and 159/512 native correct. A's improvement is therefore substantive within this finite developmental contrast. Nevertheless A still almost never switches correctly when only the query role changes. B/AB improve candidate computation without meeting the query criterion; native performance deteriorates sharply. Soft/native argmax disagreements are A 57/512, B 470/512 and AB 392/512. Native simultaneously changes all seven action/expert decisions; these numbers alone cannot attribute the discrepancy specifically to STORE.

`VERIFIED finite-grid reconstruction`, also consistent with the earlier paired-tail review: the two query answers are y0 = 3v0+v1 mod8 and y1 = 3v1+v0 mod8. They agree exactly when v0−v1 is 0 or 4 mod8. Thus 64 query pairs have equal labels and 192 have distinct labels. A query-invariant predictor can get at most

\[
\frac{2\cdot64+192}{512}=\frac58=0.625.
\]

This is a ceiling for the stated query-invariant policy on this grid, not chance accuracy, a baseline gate, an architecture bound, or a conclusion drawn from proximity to .625 alone. Actual pair evidence supplies the diagnosis: B has one both-correct and one both-wrong nondegenerate query pair; AB has five both-correct and four both-wrong. Their full candidate coverage makes their scores exactly 320 + both-correct − both-wrong. AB therefore gains only one net correct row above the query-invariant ceiling on IID. Its 321/512 long and 322/512 unseen scores have correspondingly small net query gains (6 versus 5, and 4 versus 2). Query changes or relation-only consistency must not be described as binding successes.

The value and write-order results are complementary, not substitutes for query binding. B soft has 742 relation-consistent key0 pairs but only 698 both-correct; AB has 703 versus 674. Write-order both-correct is only 115/256 and 100/256, respectively. Even answer preservation is incomplete. These overlapping pairs come from only two initial/distractor templates per split and one training seed; they are not independent trials supporting a significance or reproducibility claim.

Retention also remains incomplete. From pre-stage3 to post-stage3 VALID, old task2 soft accuracy falls A .949219→.816406, B .964844→.843750, AB .933594→.855469 despite Replay. The final old-task TEST IID soft triples are A .830078/.857422/.845703, B .900391/.953125/.873047, AB .955078/.966797/.845703. These are developmental capability/retention tradeoffs, not a full acquisition/retention/composition/discovery result across the five formal layers.

## Resource judgment

| Arm | Model /Adam bytes | Carry bytes/example | Update wall s | Driver wall s | Local cold/SSH wall s | B64 soft/native inference p50 ms | Recorded CUDA peak allocated bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 41,404 /82,888 | 1,280 | 24.8291 | 51.0496 | 57.2141 | 40.557 /38.954 | 68,557,824 |
| B | 43,964 /88,008 | 1,312 | 26.9177 | 54.7462 | 60.6894 | 48.009 /45.330 | 71,948,288 |
| AB | 41,404 /82,888 | 1,312 | 26.9164 | 54.4208 | 60.3359 | 44.329 /42.658 | 71,932,928 |

A saves 640 parameters (2,560 model bytes, 5,120 Adam bytes) relative to the historical typed model. B adds eight FP32 carry scalars, 32 B/example, plus real divisions/clamps and activation work. Its added state is not free because it adds no P. Each B/AB run pays 943,718,400 padded training coverage divisions and 29,491,200 clamps; these operation counts exclude backward and other passes and are not measured full FLOPs. The allocator peaks include diagnostics/audits and are not exact private-pool, cache, driver-context or deployment footprint measurements.

The three runs together pay 78.6632 update-wall seconds, 160.2166 driver-wall seconds and 178.2394 local-cold seconds. These are nested scopes, not additive. Driver subscopes include 6.15165 graph-setup seconds, six total warmups per arm, all current-NN diagnostics, 19.1131 VALID seconds, 23.3669 TEST seconds, 6.10441 inference-report seconds, checkpoint/reload, replay maintenance, copy and logging. Eight safety/final checkpoints per arm total 12,957,640 compressed bytes across this family; each live Replay array payload is 55,296 B plus metadata. Padded forward product proxies total 116.6672 G MAC and active proxies 57.1968 G MAC, with their omissions preserved. Neither substitutes for measured training FLOPs.

Shared CPU admission costs are charged once: failed v2 local-cold 9.98373 s plus successful v3 11.38901 s, total 21.37273 s, alongside the actual discarded synthetic AB Adam step. The v2 assertion incorrectly demanded gradients for the encoder/token tables in a detached-carry/direct-raw fixture. V3 records exactly those two expected unused names and finite connected gradients; full-sequence all-20 checks remain mandatory. Core/helper and tolerances did not change. Historical source/configuration/paired-tail/data-formation costs remain separately paid evidence; these three runs are not the whole-campaign bill.

Full forward/training FLOPs, provider-effective compute, energy, separately attributable compilation/JIT costs, exact private-pool/cache/driver bytes and parts of formation/search cost remain unknown. Current times are single-process observations, with possible overlapping workloads. Transformer32's published table includes its inherited 1,800 updates (effective training wall 19.8123 s); its 600-tail time is not a complete training comparator. None of the three arms establishes full-resource superiority over Transformer or the required strong architecture families. Parameter savings and better frozen accuracy are evidence to retain, not permission to erase runtime/native/causal failures.

## Current causal diagnostic and next single learning-rule question

At review time ROOT is preparing one already scoped frozen AB VALID diagnostic: soft, native, and `soft_drop_nonchosen_STORE`. The third condition runs the same frozen soft transition, then restores both V and z for rows whose own current argmax action is not STORE; h/controller/address/logp stay as computed for that step. Restoring both fields preserves the normalized-memory invariant. Any saved source trace of proposed new coverage must be distinguished from the actually retained coverage after intervention.

This is a well-defined closed-loop policy intervention. A large drop would show that these fractional writes are necessary for this frozen model under that intervention; little change would count against that particular explanation. It cannot by itself explain the whole native gap, show query-role binding, identify a learning cause, establish write sufficiency, or prove discrete acquisition impossible. Future read/action choices may change because the altered memory feeds them. No outcome has been assumed here, and no new training is authorized by this review.

`AGENT PROPOSAL, HYPOTHESIS`: if the scoped diagnostic establishes dependence on nonchosen fractional writes, the minimal consequential next learning-rule question is: **can the same parameterized Block acquire correctly oriented query operations when its forward training executes its existing discrete seven-way action rule, using the already defined one-step straight-through soft-transition Jacobian, instead of obtaining memory through a dense soft forward path?** This changes the execution/credit rule, not a generic head, token semantics, teacher action, auxiliary truth or task. Its plausible benefit is to make useful memory depend on selected executed operations; its strongest rival is ordinary continuous recurrence being easier to optimize and genuinely useful. The straight-through estimator is biased and its local Jacobian does not equal the gradient of an entire all-soft trajectory; it guarantees neither convergence nor binding. A future prospectively bounded comparison would need correct-versus-wrong query orientations, value/write controls, soft/native behavior, old-task retention and complete costs at the same frozen budget. Accuracy near the 5/8 ceiling would not be a successful answer. This is a next decision for ROOT/human scope, not an executed or newly authorized experiment.

`CLAIM | STATUS | EVIDENCE | WEAKEST LINK | FAILED VERSION | NEXT SINGLE DECISION`: A/B interface repairs yield reliable learned query binding | not supported for the three actual seed0 models | exact source, 84 raw gradient files, frozen full-factorial endpoint/value/query/write controls and complete available costs above | internal causal dependence and acquisition credit remain incompletely identified; no multi-seed/strong-family validation | current terminal-NLL soft-forward A/B/AB implementations as binding solutions | inspect the already scoped frozen fractional-STORE intervention before selecting one learning-rule change. TODO3 remains unaccepted; TODO2 may remain optional/negative; formal TODO1/4/5 follows eventual TODO3 acceptance and Primary freeze; TODO6 retains all positive, adverse and failed-attempt evidence.
