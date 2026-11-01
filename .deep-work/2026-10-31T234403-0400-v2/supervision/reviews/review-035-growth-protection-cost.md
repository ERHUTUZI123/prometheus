# Review 035 — actual homotopy-cache growth-cost diagnostic

Decision: **CONTINUE inside human-locked TOPOLOGY; accept a specific maintenance-cost improvement, without claiming parameter efficiency, acquisition advantage or M0.** The fixed-family certificate removes repeated numerical lifting during shared geometry updates. Cached direct execution remains cheaper. The next human-prioritized question is retained-successful-capability per added parameter under genuinely different growth policies, not another comparison of these cache implementations.

This is an actual post-execution review. I read the v1/v1a driver, cache, integration and loader; original failure, recovery, raw arm records, summaries, runner/coordinator receipts and nine real checkpoint files. I independently recomputed the nine checkpoint SHA256 values, checked all 1,800 raw update hashes against summaries, equality of all 200 post-update hashes between backends at each size, source/final neural-state hashes, all recorded curves and old-contract results. I did not run new training, SSH, checkpoint forward execution or a new geometric simulation. The existing script's reload checks are execution evidence, not independently rerun here. No unavailable old review is reconstructed.

## Actual experiment and recovery

Sources are three **already acquired** N16 models, of 2,642 / 3,254 / 16,354 parameters, not a fresh nine-arm capability-acquisition stream. Each size starts its three arms from the same model, exact parameter strides and RNG, with 200 shared minibatch indices, fresh Adam and terminal CE + .01 geometry. Existing encoder, calls, learned operations, old binding guards and geometry objective are shared. The arms change only repeated path realization: legacy exact-omega cache; same-family certified cache; ordinary cached direct permutation execution. See `growth_protection_cost_study.py:94–198` and frozen `versions/code-n16-homotopy-growth-cost-v1a/`.

All recorded train/valid/IID/OOD/length16 accuracies at steps 0/100/200 are 1 for soft, native and actual-backend interfaces on every seen task. All old relation checks pass. All 200 accepted parameter states are byte-identical between the three backends at each size. All nine arms accept all 200 guard proposals in full. Consequently this diagnostic identifies execution/verification overhead, not differences in learning, interference or admission.

v1 completed stage0 legacy and certified, then failed with `FileExistsError` while attempting to create the same append-only partial-summary filename again. v1a's first two results are exactly equal to those in the real v1 failure artifact; they were not retrained. Its root records explicitly preserve those two arms, and seven new arm directories contain the remaining 1,400 updates. Cache maps had not been persisted: recovery pays for three fresh actual origin lifts, 475 snapshots, in 0.024224 seconds, recorded separately. The v1a runner exits 0 in 63.094569 seconds externally; script wall is 57.656389 seconds. Failed v1 work and recovery are part of the campaign ledger, not erased savings.

## Measured cost improvement and its boundary

The following counts/times are **new work relative to each source's inherited counters**, for scheduled online maintenance/evaluation. They are not lifetime acquisition or physical execution costs.

| Learned size | Parameters, every arm | Legacy → certified actual lifts | Legacy → certified cold seconds | Legacy / certified / cached-direct monitor seconds |
|---|---:|---:|---:|---:|
| stage0 | 2,642 | 33 → 3 | .276936 → .022701 | .272505 / .021453 / .017279 |
| stage1 | 3,254 | 55 → 2 | .908570 → .057614 | .865409 / .038581 / .028824 |
| stage2 | 16,354 | 110 → 10 | 5.912195 → .539337 | 5.446336 / .053740 / .042348 |

Certified family hits are 30 / 53 / 100, all across changed omega. Global cache growth is 3 / 5 / 15 entries and 6,074 / 10,158 / 36,290 packed payload bytes. Direct cache grows to the same number of maps, with 384 / 640 / 3,200 numerical-array bytes plus serialized keys; neither figure is measured process RAM. Aliases in the model's inner map cache must not be added again as independent arrays. Python object overhead is bounded by finite capacity but not measured by the packed-byte proxy.

This is useful removal of repeated self-imposed lifting overhead, increasingly visible at the larger acquired representation. It does **not** reduce the number of neural parameters or the common 200 gradient updates. Measured gradient times do not consistently improve. Ordinary all-word binding checks still cost approximately .23 / .43 / .84 seconds per arm: this optimization does not reduce that protection cost. Direct map execution has no online origin lift and lower monitor times than certified execution at all three sizes.

Timers have nested scopes: cold realization is included in hard-evaluation time; family-get time includes validation and actual misses. They must not be summed as disjoint costs. Wall times depend on order, startup, source loading, checkpoint serialization and numerical audits; no replicated end-to-end speedup is established.

There is one accounting limit in the current driver: checkpoint reload at lines 217–226 constructs **fresh** caches and calls actual hard evaluation. Those additional cold lifts are included in `arm_wall_seconds`, but the reloaded model's counters and reload component time are not separately recorded. Therefore the cold-count table must not be called the full lifetime count including reload. A separate final fresh trajectory audit is paid in every arm, including direct, but covers only the selected one primitive, not the entire library at every new geometry. Neither omission reverses the observed online overhead reduction; both limit total-cost claims.

## Why the certificate is mathematically relevant

For the implemented family, at time t,

\[
\Phi_{\omega,t}(x,y)=\left(x,y+a(t)\sum_{\ell=1}^{4}\omega_\ell\tanh(v_\ell x+b_\ell)\right),\qquad a(0)=a(T)=0.
\]

Each map is a global shear diffeomorphism, inverted by subtracting the same shear. Linear interpolation of finite omega in the box [-8,8]^4 is an endpoint-fixed collision-free homotopy for a fixed canonical braid skeleton. Thus its marked-cover endpoint action is constant. The uniform box bound in this implementation gives clearance at least 4/(1+3.75) = .842105. This is the substantive reason a map can survive shared geometric change, rather than freezing omega.

The cache's key includes family fingerprint, D, base anchors and the actual learned permutation; misses perform the original lift and check its endpoint action. Changes of D, action class, compiler/cover/basepoint namespace require new entries. Runtime code identities/constants, finite bounded omega, permutation validity and deadlines are checked (`homotopy_lift_cache.py:55–223`). Map backing bytes are immutable, capacity/payload are bounded, and checkpoint policy records `maps_persisted=False` (`:224`). Loader integration validates the current namespace and recreates derived caches. The scope is **ideal real-arithmetic semantic reuse after an actual origin lift**, not a new floating-point trajectory certificate for each omega, free physical control or topology-derived task acquisition. Encoder/call/decoder changes still need their separate binding contract.

## Claim-level judgment

- **VERIFIED IN RECORDED EXECUTION:** same-family reuse sharply reduces repeated numerical lifts while real shared omega changes and all observed abilities remain correct.
- **NOT ESTABLISHED:** reduced parameter growth, gradient-learning cost, BFS binding-maintenance cost, or advantage over the cached direct control. Current backends share their growth policy and identical neural states, making them a useful null control for the next parameter-efficiency question.
- **M0:** A/B have finite earlier evidence; this diagnostic adds a concrete maintenance improvement only against legacy implementation. C/D/E for an important architecture-level advantage over strong alternatives remain unpassed. Three acquired checkpoints, one lineage and 200 maintenance updates do not establish a scaling law or reproducibility across training seeds.

The new human priority is therefore well-targeted: compare different permissible growth/retention policies on the same longer terminal-feedback stream, at fixed successful-retention thresholds, counting every persistent representation and all formation/search work. Keep certified-vs-direct with identical growth as a negative control. No additional legacy-cache experiment is warranted by these results; no MAIN change is authorized or recommended here.

## Direct evidence

- `probes/topology-architecture/runs/n16-homotopy-growth-cost-v1/learn/failure.json` — real failure and first two completed results.
- `probes/topology-architecture/runs/n16-homotopy-growth-cost-v1a/learn/summary.json` — nine completed arm results, initial/final counters, curves and paired hashes.
- Both run directories' `stage*-*/records.jsonl`, actual `.pt` files, coordinator receipts and frozen v1/v1a source snapshots.
- v1a driver SHA256 `bb7668d3698c91296780b3c15e2b1233a04c0dfd5a0f0d1234bb9fa8a962a5f4`; cache SHA256 `026c0453f2ac7369f680446f84b861d1133a6c2ebd248d9799627861b2f42c23`.

CLAIM: certified fixed-family reuse lowers repeated numerical realization cost of current learned operations.

STATUS: supported within the measured maintenance scope; no parameter-efficiency or general learning advantage claim.

WEAKEST LINK: no different topology-enabled acquisition/growth policy has yet reduced the representation needed per successfully retained new ability.

NEXT SINGLE DECISION: execute the human-prioritized parameter-efficiency stream with matched information and genuinely competitive growth policies; preserve these cache results as completed evidence, not the new experiment's endpoint.
