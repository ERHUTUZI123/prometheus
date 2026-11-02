# Typed binding and fractional-write assessment

**Judgment:** the fixed A/B experiment improved some terminal accuracies, but did not establish query-role binding or a broad architecture advantage. The soft model performs valid continuous computation; its failure under a different hard execution rule is not, by itself, a rejection criterion. The direct negative evidence is the soft counterfactual result.

This mechanism question remains connected to the Fundamental Question: a learned computational Block must retain which value belongs to which reference while executing subsequent value transformations. Producing the two candidate arithmetic answers without choosing the queried operand does not demonstrate that organization.

## Executed evidence

ROOT selected and executed the three prospective fresh arms, each with the same original four stages × 600 updates, sampler, replay, terminal NLL and optimizer. A uses input-only address proposals; B adds per-cell coverage; AB combines them. The historical typed none arm was retained. I implemented the new source and independently checked saved source, schedules, all 28 × 20 gradient gates, checkpoint bytes, reloads and costs using only standard-library arithmetic. No model, SSH or training was executed by this reviewer.

| Arm | P after growth | TEST IID soft correct | Nondegenerate query both correct | TEST IID native correct | 2,400-update wall |
|---|---:|---:|---:|---:|---:|
| historical none | 10,991 | 196/512 | 3/192 | 159/512 | 24.807 s |
| A | 10,351 | 301/512 | 1/192 | 298/512 | 24.829 s |
| B | 10,991 | 320/512 | 1/192 | 37/512 | 26.918 s |
| AB | 10,351 | 321/512 | 5/192 | 93/512 | 26.916 s |

The original GRU has 315/512 correct and 8/192 query pairs both correct; the original Transformer has 159/512 and 2/192. Their historical training and later raw-output extraction have distinct cost scopes. These finite, correlated program pairs are not independent seeds or evidence of broad superiority.

AB's predictions belong to the two arithmetic candidates for every TEST row, but its nondegenerate query flips are only 9/11/6 across IID/long/unseen. Those flips contain 5/6/4 both-correct versus 4/5/2 both-wrong pairs. Consequently its 321/321/322 correct counts equal the query-blind 320 plus only one/one/two net correctly directed flips. The small excess over 5/8 does not establish acquisition of the binding rule.

## Frozen-NN causal intervention

ROOT executed the selected AB VALID comparison: soft, native, and soft execution that reverts **only** V and coverage whenever the NN's own top action is not STORE. All other immediate soft updates remain unchanged, and future execution proceeds in closed loop. All three raw files precede target/truth analysis.

Independent saved-array verification reproduced:

- soft: 320/512, NLL 0.531687;
- native: 105/512, NLL 4.738303;
- drop non-winning writes: 153/512, NLL 3.831787, mean probability TV 0.569719 and 322 changed predictions relative to soft.

This establishes causal dependence on fractional non-winning writes for this frozen execution. It does not decompose the native gap additively: changing memory changes subsequent caller decisions, addresses and experts. Whole-program STORE argmax counts are 240, 460 and 552 respectively. In the soft path, before the first public WRITE, raw V norm is only 0.004535 but normalized read norm is already 5.479, with mean coverage 0.0001209. The memory is an active continuous feedback channel before the explicit write command. That is a legitimate computation, not evidence that stable key/value roles were acquired.

For consistent state, coverage gives

\[
z_T=\sum_i w_i\prod_{j>i}(1-w_j),\qquad
V_T=\sum_i w_i\prod_{j>i}(1-w_j)h_i.
\]

Above epsilon, V/z is a convex historical-value average. A single weak write recovers h independently of its amplitude, so the soft model can use a write that does not win argmax. Normalization does not recover overwritten distinctions or ensure key separation.

## The remaining concrete obstruction

The two AB input KEY proposals are distinct: their softmax TV is 0.319856. Yet the two soft write-attention footprints have mean TV only 0.003609; both writes and READ have the same top slot in all 512 rows. In native execution the two write-attention vectors are exactly identical despite ADDRESS being chosen at the KEY events. The distinctions exist in the proposals and disappear before use. Slot indices were never assumed to equal logical keys.

The next mechanism question should therefore concern **reference lifetime during value construction**, rather than generic head capacity or whether continuous execution is legitimate.

One specific unexecuted proposal is a neutral HOLD transition for h/V/z/address while the input-history controller still updates. The current action set offers resource-mutating STORE/ADDRESS as cheap ways to leave h unchanged; COMPUTE must learn an identity through G8/LayerNorm. An explicit HOLD is a real neutral transition, not a supplied token route. It would add one caller row, 33 parameters, and no carry state. However, COMPUTE can approximate an identity already, and HOLD could simply absorb action mass without improving binding. Current evidence does **not** establish missing HOLD as the cause. No extra training, head variant or budget extension has been selected or executed by this review.

## Complete resource boundary

The three new arms consumed 7,200 updates: measured training wall 78.663243 s; science wall 160.216642 s; separate cold coordinator wall 178.239414 s. Failed CPU v2 plus successful CPU v3 cost 21.372732 s once for the family, giving 199.612146 s of these disjoint new process clocks. The AB causal audit additionally cost 2.352359 s science / 7.582571 s cold, 51,456 transitions, zero SGD. The original GRU raw extraction additionally cost 0.422079 s science / 6.166571 s cold, zero SGD. Historical none and all earlier failed/development branches remain separate retained costs.

A/AB NN payload is 41,404 B; B is 43,964 B. Adam payloads are 82,888/88,008 B. All use 55,296 B replay arrays. A carry is 1,280 B/example; B/AB carry 1,312 B. B and AB together executed 1,887,436,800 training coverage divisions and 58,982,400 clamps, in addition to their measured compute. The family padded forward-MAC proxy is 116.667187 G versus 57.196796 G active. These are **not training FLOPs**. Complete FLOPs, provider-effective compute, driver storage, exact graph pools, shared caches, energy and agent compute remain unknown.

Evidence bindings:

- family source/schedule/gradient/cost evidence: `typed-normalized-memory-factorial-family-cost-v1.json`, SHA `7ecbfd58941ab49fda793889c2a7a8432b996995f8cf5dbbec2c7f80db6c51e9`;
- all-six counterfactual evidence: `saved-test-counterfactual-original-all-six-metrics-v1.json`, SHA `c96ac6f81426604b33764e2678ccfe7f7d8a8a50a1776032db4bfb436028e5e5`;
- AB causal raw verification: `ab-fractional-store-independent-raw-checks-v1.json`, SHA `8682a57f5fc73f7947c57aee2ad080080cdc35aceeee04b981325e06b35e8b1d`;
- AB address posthoc arithmetic: `ab-fractional-store-address-posthoc-v1.json`, SHA `18f6f72c4dea3b734b28b0130b6ae00ed41cd18e13faf6d079e513d0f78e8017`.

`CLAIM`: these changes establish reliable learned operand binding. `STATUS`: not established by the fixed experiment. `WEAKEST LINK`: preservation and use of the reference during intervening computation. `FAILED BRANCHES`: all three fixed arms fail the query-pair criterion; native collapse is a separate execution-mode finding. `NEXT SINGLE DECISION`: select a reference-lifetime mechanism question before authorizing more training. Primary remains unfrozen; this does not close the campaign or the broader hypothesis.
