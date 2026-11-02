# Task0 Transformer repair: anchored competence curriculum

**SOURCE/ACTUAL RECORD:** same six-task parameter benchmark, task0 N8, original Transformer64/96 attention architectures. No new training or GPU inference was performed while writing this artifact. Root owns execution. This supports TODO5; topology MAIN remains human-locked.

## What the existing execution actually says

Entered the actual [length-diagnostic summary](../../probes/topology-architecture/runs/parameter-n8-seed0-transformer-length-diagnostic-v1/learn/summary.json), original uniform [TF64 records](../../probes/topology-architecture/runs/parameter-n8-seed0-transformer64-full/learn/records.jsonl), and the [600-update curriculum records](../../probes/topology-architecture/runs/parameter-n8-seed0-transformer64-through0-curriculum/learn/records.jsonl). Also read `torch_baselines.TinyTransformer`, `GrowingTaskTransformer`, the exact task generator and training runner. They use real three-block, four-head attention/FFN computation and terminal CE; no code path was found that silently substitutes a program learner or supplies teacher transitions.

Uniform600: TF64 full TRAIN accuracy `.1478271484375`, one-symbol TRAIN `.3125`; TF96 full TRAIN `.1448974609375`, one-symbol TRAIN `.22916666666666666`. Their long-word CE is near `log(8)`. These are weakly trained specific implementations, not evidence against the whole Transformer family.

The stronger clue is inside the existing curriculum. Updates101–150, still drawing length<=1, had mean observed current-minibatch CE `.0029313934920355677`, mean preclip gradient `.011216333732008935`; update150 CE `.002206273842602968`. On updates151–300, after the jump to length<=3, mean CE rose to `2.1864353307088216` and mean preclip gradient to `1.4677189763387044`. These are actual changing-model minibatch losses, **not** a measured all-short-input checkpoint accuracy. The new driver measures that explicitly at entry.

TRAIN length counts are `8,48,288,1728,6120` for `0,1,2,3,4..8`. Thus the first curriculum transition reduces one-symbol sampling from `48/56` of its initial pool to `48/2072`, followed by `48/8192` under uniform training. The learned short-word scaffold can be overwhelmed by much larger gradients from unsolved long-word examples. That causal explanation is a **HYPOTHESIS**, not established merely by the coincident loss/gradient change.

## One concrete repair, same architecture

[transformer_competence_curriculum.py](../../probes/topology-architecture/transformer_competence_curriculum.py) leaves the actual Transformer, its parameter count, raw labels, input/output interfaces and attention computation unchanged. There is no width increase, auxiliary intermediate-state teacher, supplied operation table or manual correct route.

At frontier length `c`, each64-example minibatch contains32 existing TRAIN rows of exact length `c` and32 shorter rows, balanced across lengths `0..c-1`. Advance only `1→2→...→8`. The permanent anchor makes earlier terminal relations continue to receive direct gradient feedback; it does not freeze any weights or claim topology protection.

Every100 added updates, measure **all original TRAIN and VALID**, including per-length CE/accuracy. A frontier passes only if each TRAIN length<=c has accuracy>=`.99` and CE<=`.10`, and each sampled VALID length<=c has accuracy>=`.95`. The empty-program contract uses TRAIN because original VALID samples length>=1. Earlier lengths stay in every later gate. No progress is assumed from a low mean loss alone.

This is one constructive optimization policy. It does not isolate the contribution of permanent anchors from gradual promotion and additional training. A positive result would show a more competent actual Transformer baseline under its reported cost, not an independently identified new learning mechanism.

## Prospective finite protocol

- One fixed configuration: actual width64 Transformer, existing Adam LR `.003` unless an explicit alternative is recorded; no automatic LR/width search. Prefer the original curriculum's actual stage0-step100 checkpoint if available. Restore both neural weights and actual Adam state; count its100 updates/6400 example exposures. If unavailable, start the same architecture from the recorded seed, explicitly charging the fresh run.
- Maximum6000 **additional** updates,64 terminal examples/update; maximum1200 updates at one stalled frontier. Internal850-second process limit and15-second checkpoint reserve. The external authorized CUDA isolation/timeout remains required. Stop once all original length0..8 TRAIN/VALID contracts pass, or record the exact censored frontier/budget.
- Candidate selection uses only full TRAIN/VALID: prefer a complete0..8 contract, then higher whole VALID accuracy, then lower whole VALID CE; exact ties retain the earlier checkpoint. All selection screens are counted. No test split is opened during development.
- The selected actual `.pt` and hash are frozen before optional IID/OOD/length16 evaluation. Those results never select weights or hyperparameters. Passing this task0 check does not establish continual retention; the same policy must subsequently be carried into the full task stream.

The existing600-update failures, losses and checkpoints remain unchanged. Original prefix wall time is not inferred from missing per-update timings; its original receipt remains authoritative. The new run records all additional update/measurement wall and CPU time, CUDA stream elapsed time, real parameter/Adam/allocator bytes, exact batch indices, repeated terminal-label exposures and real neural checkpoints. More optimization is an explicit cost, not a secretly matched600-step result.

## Executable entry

```text
python transformer_competence_curriculum.py \
  --datasets ORIGINAL_N8_DATA --out NEW_RUN \
  --scale 8 --width 64 --checkpoint ORIGINAL_CURRICULUM_STAGE0_STEP100_PT \
  --max-updates 6000 --per-length-budget 1200 --gate-every 100 \
  --max-seconds 850 --evaluate-tests
```

Omit `--checkpoint` for an explicitly fresh start. Omit `--evaluate-tests` to keep the initial diagnostic entirely TRAIN/VALID-only. Actual code execution remains pending; AST parsing is the only executed code check here.
