# AE-v2 fresh attack addendum — acquisition is now executed, load-aware installation remains unestablished

**Judgment:** AE-v2 repairs one substantive gap in the earlier v1-only review: a neural policy now acquires a useful **component-inventory allocation strategy** from delayed scalar reaction-model feedback. All three frozen neural seeds meet the declared partial-acquisition criterion. The evidence does not establish learned load-aware construction, joint acquisition of chemistry and installation, or an advantage requiring a nonlinear neural constructor. The strongest remaining rival is a nearly fixed assembly priority combined with supplied feasibility masks.

Scope/provenance: read probes/AE-v2/contract.md, acquire.py, recovery-contract.md, recover_evaluation.py, results_recovered.json, reference.npz metadata and all six policy-array shapes/finiteness. No retraining, policy evaluation, new reward query or chemical simulation was run by this review. Static aggregation and an inventory-only reconstruction are saved in probes/final-attack/ae-v2-static-audit.json. The original certificate remains explicitly v1-only.

## What the policy actually learns

The source chemistry and twelve-contact recipe are frozen from E seed 1301. A deterministic assembly transition increments one of six inventory coordinates or chooses a direct/driver connector; masks enforce a supplied mass budget and caps. Chemistry runs only after assembly. No construction chemistry, spatial growth, wall, collision, synthesis time, binding during installation or stochastic recipe-copy fidelity is executed. Missing source/driver parts can make the resulting assay useless, so the masks do **not** give a completed functional installation. Learning which parts to purchase is real, although its action semantics and possible physical interfaces are supplied.

The width-16 tanh policy receives load, budget, remaining budget and its own counts. It does not receive A/B pulses, task labels, internal chemical state, good installation targets or evaluator derivatives. Its score-function gradients are multiplied by one terminal scalar computed from five reaction assays, material, action count and driven-work penalties. Thus it cannot directly emit the answer sequence; the installed inherited E law executes the task. Five deterministic expected-response assays per training episode are substantially richer, lower-variance feedback than one sampled Boolean outcome.

The recovered observations are:

| Policy / seed | Training-context reward gain | Held-out reward | Held-out analog task score |
|---|---:|---:|---:|
| Neural 631 | .373215 | .813217 | .839453 |
| Neural 632 | .346012 | .812292 | .838498 |
| Neural 633 | .318255 | .813217 | .839453 |
| Linear 631 | .337978 | .805001 | .836475 |
| Linear 632 | .325880 | .801649 | .834301 |
| Linear 633 | .319191 | .806755 | .836276 |

The three-neural mean held-out task score is .839135. This is a finite success for the declared .05 improvement/.75 held-out criterion, not a stochastic Boolean success rate. The held-out contexts are new load/budget values for the same component grammar and same five task assays, not new computations or unknown load laws.

## Strongest attack: a fixed priority plus masks explains the visible adaptation

Across **every physical load and all three neural seeds**, each budget has the same most frequent final inventory. In coordinates `(m,z,recipe,sensor,Y,turnover,port)` these are:

| Budget | Common final inventory |
|---:|---|
|35|`[1,1,1,2,1,4,2]`|
|50|`[1,1,2,2,1,4,2]`|
|65|`[1,1,2,2,2,4,2]`|
|90|`[1,1,2,2,3,4,2]`|
|115|`[1,1,2,2,4,4,2]`|

A posthoc fixed priority—m, z, recipe, sensor, Y, turnover, driver, recipe, sensor, turnover, turnover, turnover, Y, Y, Y—**skipping infeasible purchases under the identical masks**, reproduces all those inventories without reading either load or budget as policy features. This was checked by deterministic inventory arithmetic only, without inspecting fitness for the reconstruction or executing a new assay. It is an explanatory null fitted after observing the learned modes, not a preregistered zero-query performance baseline. It does not explain the full action distributions or prove the weights ignore context. It does show that the observed budget-varying configurations alone cannot identify a learned resource-sensitive construction computation: the supplied mask can generate that dependence.

The neural driver fractions over all contexts are 1, .997917 and 1. Replacing the observed load by zero while retaining physical load changes mean held-out reward by approximately -.000567, -.002276 and 0. These observations provide no positive evidence that load sensing is materially used for the held-out installation choice. Because the ordinary and masked evaluations consume **different stochastic draws**, this is not a paired counterfactual estimate or a proof of exact load invariance. One seed/context difference reaches .01323; the modal inventory still stays unchanged. A stronger attribution would require distinguishing learned contextual policy effects from stochastic tail choices and feasibility-mask effects. No further test is executed here.

## What the baselines do and do not establish

The matched linear policy has 99 parameters versus the neural policy's 329, identical observations/actions/masks and the same 2,000-episode budget. Its held-out task scores are within .0020–.0052 of the corresponding neural seeds; no inferential analysis establishes a neural-specific advantage. All six learn closely related dominant allocations. This is already a strong simpler explanation and should not be hidden behind the weaker random baseline.

The ordinary random allocator uses 2,000 terminal-inventory reward queries per seed, retains feasible observed candidates, and uses a **specified nearest-training-context rule** at test time (acquire.py:242–255). Its held-out task scores are .779512, .672929 and .772260; its training rewards are .854519, .813785 and .852510. Neural mean held-out performance beats that particular transfer heuristic, while random search exceeds neural training reward in two seeds. This does not demonstrate superiority over ordinary learned allocation, stronger interpolation, a fitted priority policy, or other equally informed nonneural transfer rules. The query budgets match; their proposal/retention/transfer procedures differ.

The exhaustive evaluator's mean held-out task/reward are .852864/.835023, above all neural outcomes, at the explicitly additional full-table cost. It is an upper reference, not a matched-budget competitor. The gap cannot be called a chemical impossibility or an inaccessible optimum.

## Leakage, numerical evidence and full cost boundary

I found **no reward-table argmax or held-out reward/gradient access in Policy.act/update**. Training selects only the first nine context rows (acquire.py:225–228); the exact table is used as a scalar environment query after the assembly trajectory. Initial evaluation also visits test contexts but updates no weights and returns no feedback to the subsequent learner except advancing its sampling stream. No result-conditioned seed selection or retraining is present in the inspected recovery.

However, the full training **and held-out** table is precomputed before training and resides in the same Python process/global namespace and a saved accessible NPZ. Separation is by inspected implementation and instruction, **not an independently enforced evaluator boundary**. This is not observed leakage, but it limits what the setup certifies. Cached table access is also not a free physical assay: the reference build contains 2,700 inventories × 15 contexts × 5 cases = 202,500 model trajectories. Training adds a static logical 60,000 assay visits across six policies; ordinary allocation has 6,000 reward queries. Pre/post/masked evaluations and recovery are additional table visits. Repeated exact cache hits are not fresh stochastic evidence; unique queried installations are unavailable.

The original process saved all six final policies before final JSON serialization failed. Program order and finite saved arrays support completion of the six specified training loops; their per-episode curves, training material/work totals and unique-query counts were held in memory and lost. Those measured costs cannot be reconstructed from final weights or replaced by the code budget. Recovery loads the saved weights, uses newly fixed evaluation streams and performs **no update**. Thus final recovery values are new frozen-policy observations, not recovered original evaluation samples. There is no full recorded training audit trail or total acquisition-energy result.

The recovered sampled checks pass: maximum terminal-port discrepancy versus BDF .0015313; checked reward-error bound .0016014; step-halving port discrepancy .0012445. Those checks are eight specified configurations/cases, not a uniform bound on the full reference table or learned-policy distribution. The recovery's conservation/nonnegativity `physics` summary also uses its eight-configuration batch. The .05/.75 criterion has comfortable margin relative to these sampled errors, but the close neural/linear differences should not acquire precision the verifier has not established. Model chemical work excludes physical stock synthesis, input/reset preparation, sensor/controller computation and the disclosed mass proxy's conversion to actual resources.

CLAIM | STATUS | WEAKEST REMAINING LINK

- A neural policy can acquire a useful finite inventory allocation for an inherited kinetic program. | **SUPPORTED partial acquisition**, three fixed frozen seeds, repaired serialization transparently | instruction/code-defined cache boundary; missing full training trace/costs; sampled numerical checks.
- The policy acquired useful load-aware construction. | **NOT ESTABLISHED** | modal choices are load-independent; load masking has near-zero aggregate effect; feasibility masks explain budget variation.
- The system learned its installer and computational law jointly. | **NOT EXECUTED** | source law, driver chemistry, assembly operators and compartment are supplied/frozen.
- Nonlinear neural construction is needed or outperforms equally informed simpler learners. | **NOT ESTABLISHED** | matched linear outcomes are similar; ordinary random search is limited by its chosen transfer heuristic.
- AE-v2 closes v1's complete absence of installer learning. | **YES, in the restricted inventory-allocation sense** | it does not close physical construction, copying fidelity, collision handling or learned interface invention.

NEXT SINGLE DECISION: retain this as an honest partial allocation result and decide whether learning a genuinely consequential installation/load contract remains the target. The current finite result does not warrant expanding an already supplied component library and counting the resulting installations as mechanism discovery.
