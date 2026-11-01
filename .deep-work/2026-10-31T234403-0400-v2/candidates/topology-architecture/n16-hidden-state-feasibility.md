# Frozen diagnostic: terminal experience can support the missing state split

Same V2; MAIN=TOPOLOGY is human locked; M0 remains unpassed and M3 the final ambition. Transformer remains the primary architecture baseline. This is a diagnostic/construction artifact, not a second living method and not a new training result.

## Actual evidence inspected

Source: `probes/topology-architecture/runs/n16-repaired-stage2-v8/learn/records.jsonl`, SHA256 `c159858b20985dbda7fb61d5cf30d7f803d3ae5a7db774fef647bd3241338517`.

Original observed training data: `learn/datasets/task2-train.npz`, SHA256 `afcbb2490941213239baf5a479f0b5c650b209a6a83b5a6dbec1af84fbd23f19`.

The final actual stage2 model has D=32, IID native accuracy 0.46484375, soft accuracy 0.51953125, old native task accuracies 1 and 1. Its guard made 764 full accepts, 32 backtracked accepts and 4 rejections. Hard encoder maps x to x for the 16 inputs. Decoder fibers each have exactly two states, y and y+16. The new symbol's native route is `[0,5]`; its transition maps the first16 states only within the first16.

The recorded valid32-history clique comprises `(x,empty)` and `(x,[6])` for each x. **All32 histories currently occupy only16 MODEL states.** Every new-prefix history aliases an anchor history. None of states16..31 is used by these histories. This is a failure to use the added capacity, not evidence that D=32 cannot represent the observed lower bound.

Existing training coverage, counted by exact observed word and start (no teacher function called):

| Exact word | Observed rows | Distinct starts |
|---|---:|---:|
| `[6]` |42|14; missing2,4|
| `[6,1]` |144|16|
| `[6,2]` |161|16|
| `[6,3]` |4|3|
| `[6,4]` |7|5|
| `[6,5]` |9|9|
| `[6,6]` |9|7|

For the14 actually observed starts in `[6]`, the terminal label equals the observed start. This sentence makes no claim about the two missing starts.

## State assignment from experience

Let H be the32 witnessed histories and O(h)[s] the actual terminal label recorded for history h followed by suffix s. These dictionaries use original task2 training rows only. The historical clique gives pairwise distinguishability, so any deterministic32-state realization must assign distinct states to H.

Construct a bijection μ:H→{0,…,31}, anchoring μ((x,empty)) to the model's already learned E(x). An observed empty suffix constrains R(μ(h))=O(h)[empty]. Choose among remaining feasible assignments by the model's soft prefix-state/continuation scores; do not add an unobserved output constraint.

With the actual fixed encoder/decoder, the14 `[6]` output observations uniquely assign the corresponding histories to states x+16. The other two histories, for starts2 and4, can be assigned to `{18,20}` in either order at this step. Thus this simple state-assignment problem is feasible but still has two assignments; a model score or additional observed continuation consistency is required. Neither assignment has been executed in a new neural model here.

## Operation formation through recorded continuation compatibility

Direct empty-suffix constraints alone leave many possible transitions. Use the existing longer terminal episodes constructively. Define

    C_a(h,h') = 1 iff every suffix s observed in BOTH O(h a) and O(h')
                    has equal terminal labels in those two records.

Each forbidden edge has a concrete pair of observed terminal records. An allowed edge is finite observational compatibility, not a proof of equivalence on every possible continuation. Do not silently replace missing observations with a teacher query.

Data-analysis result: all six resulting32×32 bipartite graphs admit a perfect matching. Exact counts from the original training arrays:

| Symbol a | Edges using only common empty-suffix labels | Edges after all observed common suffixes | Rows with exactly one compatible target |
|---|---:|---:|---:|
|1|604|35|29|
|2|630|38|27|
|3|774|322|13|
|4|762|257|14|
|5|696|282|10|
|6|750|187|17|

The row-uniqueness values are counted before any global matching propagation; the number of feasible perfect matchings was not enumerated. In particular, all16 anchor histories followed by symbol6 have a uniquely compatible recorded `[6]` history, so the empirical history distinction can be turned into an executable transition requirement rather than remaining an offline capacity certificate.

For a chosen μ, select a permutation σ_a of H by constrained assignment using current learned soft operator A_a:

    maximize sum_h log A_a[μ(h), μ(σ_a(h))]
    subject to C_a(h,σ_a(h))=1 and σ_a bijective.

Its native operation is Π_a=μ∘σ_a∘μ^{-1}. The computation uses bounded matching, not enumeration of all possible programs. The normal materialsation/shared-call/old-contract pipeline can install these candidate whole operations in the SAME neural model while retaining old task contexts. A candidate must then be evaluated on complete original training/validation words before acquisition is admitted; finite pairwise compatibility alone does not establish that the whole learned program is correct.

No candidate was installed, trained or tested by this diagnostic. No claim is made that every sparse operation will be resolved: symbols3–6 still have substantial observational ambiguity. The concrete remaining construction is to couple the experience-derived μ with operation assignment and actual end-to-end feedback, retaining unknown entries as unknown.

## Further actual joint feasibility: state alignment and two operations identified

The simple μ ambiguity above can be checked against transitions, rather than guessed. For an actual terminal record ending immediately after h followed by a, also require the already learned decoder of the candidate destination:

    Ctilde_a(h,h') = C_a(h,h') AND
                    [empty not observed in O(h a), OR R(μ(h'))=O(h a)[empty]].

Only two μ assignments remained after anchor and direct `[6]` constraints. Both were checked in finite matching calculations. Exchanging the assignments of histories for starts2 and4 makes operators1 and2 infeasible (some rows have no allowed target), and also makes operator3 globally infeasible. The other assignment is feasible for every operator. Thus, **conditional on the current fixed learned encoder/decoder, these witnessed histories and this32-state realization, existing terminal feedback uniquely aligns the32 history states.** This is not an unconditional uniqueness claim about every representation or every possible hidden process.

The surviving μ assigns the two previously unknown histories to states18 and20 respectively. That conclusion used recorded continuation outputs and global matching feasibility; it was not assumed from an unobserved `[6]` label or a supplied hidden mode.

Under that surviving μ, matching support was checked using alternating-cycle reachability relative to a perfect matching. An allowed edge can participate in another perfect matching exactly when it lies on an alternating cycle (matched edges are always supported). Counts:

| Symbol | Ctilde edges | Initially unique rows | Rows forced across ALL feasible perfect matchings | Supported matching edges |
|---|---:|---:|---:|---:|
|1|35|29|32|32|
|2|36|28|32|32|
|3|279|14|16|147|
|4|221|16|24|80|
|5|177|13|19|60|
|6|162|18|30|34|

Operators1 and2 therefore have unique full32-state permutation assignments under these finite constraints. The symbol6 assignment retains two choices; the other sparse operators also retain ambiguity. Learned soft scores can rank the remaining feasible choices; complete task rollouts must still decide actual acquisition and composition. These are data-analysis/finite-matching results, not a trained neural implementation, a formal proof-assistant certificate, or a topology-specific advantage.

## Interpretation and accounting

The strongest conventional interpretation is passive finite-state/automaton identification with learned assignment scores. Its same-information implementation is an essential control; this diagnostic does not establish topology-specific advantage. The topology mechanism would carry the resulting reusable permutation actions through the existing protected continuous execution layer; actual benefit still requires measured acquisition, retention and cost comparisons.

The present calculations only parsed the already saved raw/NPZ data and ran finite32-node matching feasibility checks. No GPU, SSH, new training, new label acquisition, teacher operation, intermediate target or dependency installation was used. The next actual experiment remains under Root's control.
