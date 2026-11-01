# First-stage bottleneck ablation — frozen after observed failure, before new runs

Same V2, MAIN topology human-locked, M0 unpassed/M3 final. User prioritizes learning stage1 before the full grid and requests separate credit/discretization/protection-timing ablations. This is a development diagnostic, not a substitute for full M0 evidence.

Actual previous matched six-arm/three-stage records are preserved. Initial80-update run failed to learn hard programs. Credit-repair240-update run achieved soft99.8% but componentwise hard48.4% on N4 stage0. A same-checkpoint16-way component ablation established: hard calls alone99.8%, hard encoder/decoder99.8%, hard operation banks48.4%. Probability versus log-semiring gradients differ only5.2e-7 relative with cosine1 at this checkpoint. Joint latent or external learned-map compilation gives100% on that held-out IID set; it uses learned matrices and no teacher program/operation matrices. This is post-training inference evidence, not new training or M0 passage.

Next controlled training configurations share exact data, initialization, minibatch order,400 updates, batch64, LR.003, latent factor1, same soft training objective and all sampling/temperature schedules:

1. log credit / component compilation / admitted-after-learning protection;
2. probability credit / component compilation / admitted-after-learning protection;
3. log credit / JOINT learned-operation compilation / admitted-after-learning protection;
4. log credit / JOINT compilation / EARLY unvalidated protection atstep0;
5. log credit / JOINT compilation / protectionOFF.

Start N4 seed0 and N8 seed0. They are first-stage focused runs (--stages1), with unchanged train/validation/IID/OOD/length16 splits. Two independent seeds may follow to check whether the first-stage fix repeats, without claiming M0. Report hard and soft learning curves separately, gradients, operation/call entropy, guard rejection, compile/maintenance resources. Config3 versus1 changes only compilation; config2 versus1 changes only numerical credit; config4/5 versus3 isolate timing/protection. Since admitted protection starts after this stage,3 and5 are expected to have identical training; this is an explicit negative-control prediction, not unexplained equal results.

Joint compilation forms each complete token-operation matrix from the model's trained call probabilities and learned operation matrices, then extracts a permutation assignment. It does NOT insert the teacher's correct route. It changes the granularity of the learned operation; it does not prove the old component program was correct, make mixture a physical coherent superposition, or establish a unique topology advantage. A conventional program compiler may use the same procedure.

If discrete stage1 learns reliably, return to3stage mandatory baselines and planned scale/seed grid. If not, inspect which learned component is at fault and execute a targeted next correction. Do not bundle several unexplained changes or stop at the ablation report. Existing runner isolation/process limits and raw/checkpoint preservation remain in force; actual CUDA bootstrap compatibility repair is separate engineering, not a learning result.
