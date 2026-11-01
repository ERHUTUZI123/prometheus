# pilot_v1.py — pre-execution implementation/inspection notes

This worker wrote the script and inspected its text only. **It has not been imported, compiled, run, finite-difference checked or trained by this worker.** Root owns the existing isolated runner, contract, execution and receipt. The script uses only Python's existing standard library; no dependency or installation is needed. Existing frozen mechanism-v1 is unchanged.

## Contract binding

Read the actual `contract.md` before this write. Root explicitly fixed `TRAIN_HOLD=64` (65 total training inputs) before execution and confirmed all four arms include task loss; prefix/carry coefficients are one. Carry uses transitions strictly after the mined merger through the terminal time. The fixture should mine tau=2, yielding 63 carry terms, but the miner obtains tau from actual full-state and remaining-word comparisons rather than assigning it.

Fixed four arms: endpoint at A=.5; endpoint+seed at A=.5; endpoint+seed+carry at A=.5; endpoint-only strong-initialization control at A=1.2. All beta=.01, v=1. Each has exactly512 updates unless a check/budget/error stops the process. Learning rate.05, m.4, global norm clip10, projections A[0,2.5]/beta[-4,4]/v[-3,3]. No performance-dependent stopping, tuning, seed search or arm changes.

The initial and every16th/final checkpoint recompute the current trajectories and record all component losses/gradients, even inactive auxiliary diagnostics. Those inactive components never enter updates, and checkpoint overhead is counted separately. No held-out evaluation occurs during learning. Final evaluation is exactly12 episodes per arm: single cues and changed-write pairs at holds32,128,512. Both first-cue and last-write target interpretations are emitted for changed writes. No one is selected as the learned task.

## Objective and derivative inspection

Task: mean `.5*(v*r_T-y)^2`. Analytic RTRL tracks

`dA_t=(1-r_t^2)*(r_(t-1)+A*dA_(t-1))`,

`dbeta_t=(1-r_t^2)*(x_t+A*dbeta_(t-1))`.

The seed is mechanism-v1 Eq2, `(error_positive^2+error_negative^2)/2`, with no additional half factor. It uses the actual candidate state and true RTRL derivative at the mined checkpoint. Prefix values are already available in the full forward pass; this implementation does not claim less total replay work for the seed arm.

Carry is the average over63 transitions of

`max(0,2*m-abs(r_plus-r_minus))^2/4`.

For an active hinge, its derivative with respect to the difference is `-gap*sign(delta)/2`, then averaged. Each one-step derivative treats both previous candidate states as constants; those are actual on-policy states recomputed every iteration. v has zero auxiliary gradient. At exactly zero difference, the declared absolute-value subgradient is zero. Seed/nonzero initialization, not this subgradient, provides symmetry breaking.

Before training, central finite differences use one fixed interior point A1.05/beta.12/v.8, step1e-6, atol1e-7/rtol2e-4. Task/seed finite differences rerun the complete candidate trajectories. Carry finite differences use a predecessor-pair bank frozen ONCE at that point for both +/- perturbations. They do not compare the conditional gradient against a new full-trajectory objective. A failed check aborts and retains raw rows. No claim that these checks pass is made before execution.

## Information and costs

Training inputs and the two endpoint labels are explicit contract fixtures. No r_t teacher labels are supplied. The miner derives orientation from ordered endpoint labels only after checking same context, actual h=x state equality, and the complete remaining raw input word. It records all original traces/comparison counts. Replay never receives a future word or endpoint label as recurrent input.

Counters separate optimization, checkpoint audit, finite differences, mining and final evaluation. They include actual mode transitions, RTRL sensitivity steps, conditional local transitions, episode replays, cached-label uses and pair uses. Two received endpoint labels are not relabeled as new environment queries on every update. Timers include all Python overhead. The notes/counters are not exact FLOP, physical energy or whole-runtime-memory accounting; native peak RSS and a labeled Python history-object estimate are recorded separately.

Full training-state traces are logged only at checkpoints; final held-out episodes log the final state and changed-write boundary states, avoiding an uncharged large trace dump. `pilot-v1.raw.jsonl` uses exclusive creation, immediate flushes, JSON finite-number validation and a strict below2MiB cap with reserved failure-row space. Rerunning into existing raw data fails rather than overwrites it. No other script-created file, subprocess, network operation or extra thread is present.

The script's25s internal wall guard is a backup termination check; it does not replace root's external30s CPU/wall/network/filesystem controls. It reads only its own code and the adjacent contract for SHA256, and writes its one raw JSONL in the probe directory. Exceptions attempt a small flushed failure row, retain earlier rows and return nonzero.

## Scientific limits retained

This is the ungated scalar subfamily, not a complete architecture/novelty test. Old h remains the specified separate h=x computation. The strong-initialization control diagnoses the importance of starting dynamics rather than supplying a matched-initial causal comparison. The local mean may dilute useful updates; no repair is applied after results. An observed write response or changed-write evaluation cannot establish learned overwrite/clear from single-cue training. Root should inspect and execute once under the existing contract, then use the actual outcome to guide the next within-lineage construction.
