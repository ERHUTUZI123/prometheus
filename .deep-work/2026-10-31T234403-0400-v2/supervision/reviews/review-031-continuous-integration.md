# Review031 — actual shared continuous-monodromy training in the native model

2026-11-01, post-execution independent review. **CONTINUE_TOPOLOGY.** A continuous configuration-cover executor is now actually connected to the existing learned primitive bank, calls, model parameters, optimizer and checkpoint/reload flow. Four shared path parameters undergo real gradient updates while the three previously acquired tasks remain correct. This is a substantive implementation and mathematical integration result. It is not yet a run in which all three acquisition stages use continuous execution from the start.

MAIN=TOPOLOGY USER_LOCKED/HUMAN_ONLY; M0 NOT PASSED; M3 ultimate. Online USER-PARK; standalone Persistent State remains paused/reserve. Transformer remains the primary architecture baseline. This review records a completed consequential node and supports further construction, not a gate requiring research to stop.

## Evidence actually inspected

Read complete `continuous_monodromy.py`, `train_monodromy_extension.py`, the reused source-model/checkpoint/protection/evaluation interfaces, prospective `continuous-integration-protocol-v1.json`, the actual local continuous integration runner receipt, full update/curve/continuous-lift/reload records, all summary metrics and all 32 saved geometric traces. Inspected code hashes agree with prospective protocol, source snapshots, receipt and recorded provenance:

- `continuous_monodromy.py`: `ad1852bba88393b85db8aa44553d5c85183a0dce5ada4a62460f52b3a52c2936`;
- `train_monodromy_extension.py`: `8d0f3a3ce0e4bcd3added02bb69f1dc8aef060f1527011dcae5ebef88f090706`;
- reused integration source `076ec180fc6331b21359935028ab64f933be6c741853a8edee4ab6ba1e86c613`, topology model `48f8f7aa5efe96297bef04e29ca6c52b8bd8bc38a2af409ce772619679e77acd`, semantic promotion `d569785c90e02189ced289d043563ce8c08d97ecda04903e77494771f256035e`.

The source checkpoint SHA is exactly Review030's actual final repaired checkpoint, `c441a2c6e49fbf43c21665076440a4e8c7f03f30f43693c58e40309454418028`. No old stage was rerun for this extension. Actual runner exit0, external900s/internal850s bounds, L40S/CUDA isolation and authorized own-process thread metadata exception are retained.

Independent checks used read-only arithmetic in the existing campaign NumPy environment. I checked all 200 consecutively numbered fused updates, the quadratic action gradients against stored moments and pre-update weights, all 32 geometric traces/8,474 snapshots, all three evaluation curves'15-way consistency, all four checkpoint byte sizes/SHA256/metadata keys and the actual reload-verification event. No training, new experimental arm, dependency installation or downloaded code execution was performed for this review. I did not independently import PyTorch/deserialise/forward the neural checkpoint tensors; actual tensor reload equality comes from the inspected executed source and its saved event, not an invented second neural execution.

## Mathematical construction independently checked

Each model-learned permutation is translated to adjacent exchanges of base points at spacing4. On a swap segment, the two moving points are `m +/- 2 exp(i*pi*h(u))`, with `h(u)=3u²-2u³`. Their mutual distance is4; every stationary point is at distance at least4. The unordered set returns to the base configuration at each segment boundary. The coordinate-array ordering can reset there because it is not a marked-sheet label; the actual marked point is transported by the lift algorithm.

The shared deformation is

    Phi_(omega,t)(x,y) = (x, y + a(t) h_omega(x)),
    a(t) = sin²(pi*t/T),
    h_omega(x) = sum_l omega_l tanh(v_l*x+b_l).

For each finite omega, `Phi` is a global diffeomorphism with explicit inverse `(x,y)->(x,y-a(t)h_omega(x))` and determinant1. At operation endpoints, a=0. Interpolating omega gives a based homotopy of unordered configuration loops without collisions. Thus the primitive's monodromy is unchanged under these shared realization updates. This is an actual topology-based protection statement for the path realization, not a rebranding of the finite graph guard.

Let `B=sum|omega_l v_l|`, `H=sum|omega_l|`. Both the shear and its inverse have operator-Lipschitz bound at most1+B, giving pair clearance at least `4/(1+B)`. On a swap segment, radius2 and max h'=3/2 give base speed at most3pi. Since `|a'|<=pi/T`, the supplied bound

    V <= (1+B) 3pi + pi H/T

is sufficient; the identity loop omits the base-speed term. Choosing `V*dt < clearance/4` makes nearest continuation unique: the correct next point is within clearance/4 of the previous marked point, while every other next point is more than3clearance/4 away. The code includes every segment boundary and rejects a nonbijective continuation or violated numeric bounds.

I independently checked that all saved traces satisfy their actual per-step speed and sampled separation bounds, and that final marked-point positions reproduce each recorded/model-learned permutation. Minimum sampled pair distance across traces is3.69540932587. The maximum observed step divided by its sufficient bound is .95358832194. The analytical bound is a real-arithmetic sufficient theorem; sampled floating-point traces are numerical checks, not a rigorous interval certificate or a physical noise measurement.

## Learning rule, gradient and real integration

The model registers `path_omega` as a real four-dimensional `nn.Parameter`; it is part of the same `state_dict` and Adam optimizer as the existing encoder, operations and calls. For a single fixed native class skeleton, the supplied execution-work objective is the midpoint approximation to

    A(omega) = integral sum_j ||d_t Phi_(omega,t)(z_j(t))||² dt
             = c + 2 b^T omega + omega^T H omega.

Writing the baseline velocity `(vx,vy)` and `f_l(x)=tanh(v_l*x+b_l)`, the added vertical-velocity basis is `a' f_l + a f'_l vx`. Therefore the implementation's linear term is the integral of `vy*basis`, and H is the Gram integral of that basis. H is positive semidefinite by construction. Stored H has eigenvalues approximately .272621,1.918872,12.951567,253.791215, so it is positive definite in this run.

Actual combined objective is

    terminal_CE + .01 * A(omega)/max(1,c).

Native classes are extracted from current learned parameters and quadrature moments refreshed if the active class signature changes. In this run they rebuild once; the six active native classes remain unchanged. The path gradient is the explicit work derivative, not a derivative through the discrete monodromy label. Terminal CE does not depend on omega. This separation is the correct current architecture: class acquisition/revision occurs through the already implemented terminal-feedback mechanism; class-preserving realization learning uses an explicitly supplied execution-work model.

Using all 200 recorded pre-update states, I independently reconstructed `.01*(2b+2H omega)/c` with any recorded global clipping factor. Maximum difference from saved path gradients is2.88e-11. Reconstructed double action differs from float32 recorded action by at most .000820. The 16 midpoints per segment are the actual training quadrature; no convergence proof or higher-resolution integration claim is made.

The continuous hard evaluation actually constructs the shared learned bank by lifting through shuffled, unlabeled geometric snapshots. It checks lifted results against the learned bank only after continuation, caches the resulting maps, then runs the REAL learned encoder, two-slot calls and decoder with those maps. Thus the continuous realization participates in actual evaluation. Warm task episodes use cached lift-derived maps, not fresh point transport at every token; cold path work, cache and warm calls are separately charged. This distinction is explicit and scientifically appropriate.

## Actual execution result

The extension starts from the actual repaired stage2 step1000 checkpoint and performs200 more batch64 updates at LR.01, seed9002, original current-task terminal training data only. No extra teacher information, intermediate state or correct call relation is supplied. Parameter count is3,610, adding exactly four shared geometric weights.

    omega_initial = [.699999988,-.5,.400000006,-.300000012]
    omega_final   = [.179184064,-.052155528,-.031577431,.093362413]

Actual omega L2 movement is .9015581. Shared native call/token/task weights also change: call.weight .8148703, call.bias .1888294, token.weight .8339369, task.weight .5910137; operations move2.47649e-5. Encoder/decoder movement happens to be zero in this execution, despite remaining trainable. These are observed movements, not a guarantee that every allowed parameter receives useful credit.

All original three tasks' five saved splits have soft/native/continuous accuracy1.0 at steps0/100/200. The final checkpoint is actually loaded through `extended_model`; inspected code verifies exact neural tensor and native graph equality, then recomputes continuous lifts and all15 split results. The actual reload event reports unchanged continuous results. Every one of the200 guarded parameter updates is accepted at alpha1; there are600 relation checks/6,400 relation pairs and no backtrack/reject.

Quadrature action decreases14774.00390625 ->14591.3974609375. The canonical zero-shear action c is14591.417960417157. The bulk of the1.236% decrease from the chosen initial shear is its removal; gain over zero shear is only about1.405e-6 relative. This is valid realization-objective learning, not yet an important comparative resource gain. The strongest same-information direct control is explicit: the ordinary four-variable quadratic solver using the SAME c,b,H, together with the zero-shear controller, can be evaluated without denying it the supplied work model. This is a constructive baseline, not a reason to abandon the integration.

Costs in the actual script: training3.419649s, guard .655426s, neural evaluation3.048018s, geometry moment construction .014792s, class refresh .366325s, cold realization .674649s, warm cached native execution .068441s. There are39,680 quadrature point-strands,32 cold class realizations/8,474 snapshots/4,346,880 distance-matrix entries and699,696 warm primitive calls; peak map cache1,024bytes. Additional training uses12,800 current sample episodes, no old-label replay. Script wall11.641536s; full runner17.510752s. The reported component timings have their documented overlaps, so they must not all be summed as disjoint costs. All this is extra work after the original800-step stages and the earlier200-step repair integration.

Checkpoint hashes independently checked:

- step1000 trained: `aecf83f6e1410cc3ac227823d05f9e74750253c26f9aea1a90141f06a2688775`;
- step1100 trained: `e5c0a57c35d0bcab5889c3b84381770c046a6925285da4bbd3ff05e4cb6ce22e`;
- step1200 trained: `0d24a612a6c49a8d14eea1b24390073a1798473d7285c568d2885d6ebb3be995`;
- step1200 final/reloaded: `1d9b30f001a2fbdd00dfa6793732c0cb21c81653ac39c0002757dea241fcdd35`.

## Decision and the next constructive obligation

CONTINUE the same topology architecture. The previously missing continuous realizer is now implemented, learned and connected to the saved neural architecture; it is no longer accurate to describe all topology as an unexecuted geometric interpretation. Its tested topological role is **preserving an acquired operation's monodromy while shared continuous execution parameters learn**. The current discrete creation/refinement rule and binding guard remain separately inspectable.

The next primary test should integrate this execution/protection layer into the ongoing acquisition stream, including primitive formation and experience-driven workspace growth, rather than attach it only after all capabilities are already acquired. It should couple useful new-task acquisition or a meaningful execution/resource constraint to the protected implementation freedom, then compare its full costs with the applicable direct/geometric control and strong Transformer continual-learning baselines. Fixed-class path optimization alone does not show that topological protection improves acquisition. A class-preserving shear also does not protect arbitrary changes to the learned primitive skeleton or neural call bindings; those remain governed by their explicit formation and old-behavior contracts.

This is one measured construction step toward M0, not a completed M0 architecture evaluation, a paradigm-level result, a campaign ending, or authorization to change MAIN. Preserve the positive integration evidence and the small measured work improvement without inflating it. No new pre-approval cycle is required merely to continue authorized development.

Evidence lives in `probes/topology-architecture/runs/native-continuous-integration-v1/learn/`, its runner receipt, `continuous-integration-protocol-v1.json`, the inspected source files and immutable snapshots. Actual summary SHA `f4f70c5a0dc63feefde532946512e7e21eda7b7f2b98e20f70182f6e7fc21fb3`; actual records SHA `4fc88728fcda37e29b7bb99e29f5f66c220571cfa4e116bc6da85356fd1f9ec0`.

CLAIM: actual continuous cover-lift realization, real shared geometric parameter training, native neural integration/reload and retained previously acquired tasks. STATUS: supported in the declared N8/seed0 development checkpoint. EVIDENCE: independent path/gradient/metadata arithmetic, actual inspected code/raw/receipts/checkpoints. WEAKEST LINK: no demonstrated important protected-acquisition/resource advantage or acquisition-from-start continuous run. FAILED VERSION BOUNDARIES: none hidden; ordinary zero-shear/quadratic control explains the present small work improvement. NEXT SINGLE ACTION: continue the actual acquisition stream with this integrated layer and a consequential priced learning objective; M0 remains unpassed and M3 remains the goal.
