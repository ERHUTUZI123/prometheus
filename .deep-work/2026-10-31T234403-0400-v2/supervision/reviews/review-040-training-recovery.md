# Review 040 — actual next-step training recovery

**Decision: the new CPU checkpoint path passes this exact-next-step integrity check; continue interface development.** This fixes a prospective recovery omission identified in039. It does not recover the original missing V2 RNG history, add an acquisition result, establish a topology advantage, validate the general Block, or freeze the publication Primary. Earlier prototypes remain historical/internal evidence under the latest human policy.

**Reviewer disclosure:** I authored the two new recovery helpers after reviewing the separately authored model. Root inspected, froze, and executed them. This review independently checks the execution's original records and numeric archives; it is not an independent reimplementation of my helper or external replication. I performed no model execution, training, SSH, or file changes outside this review.

## Checked provenance and failed attempt

SHA256 and byte-size verification passes for all9 synchronized V1 files and13 V2 files. The requested source, both staged source copies, and the actual hybrid-V2 `full_coupled/step-030.pt` all have SHA256 `806e4d591f6129db2a6947b3bc11df9800d5a35a8c9778f63e5bbf65b67bfbc6`. The recovery driver/helper and actual model/cover sources are identical between attempts. The source is a real pre-admission checkpoint; no alternative source was selected by accuracy or held-out performance.

V1 actually exits1: code-directory `source.pt` is unreadable under the existing sandbox. The traceback occurs in the initial owned-input load, before a session or gradient is created. Its output contains no completed training step. V2 stages the same input into the existing read-only `/opt` allowance and exits0. The recorded UID/GID65534, empty supplementary groups, Landlock ABI4, enforced seccomp, no-new-privileges, and resource limits match across attempts. These records support an input-placement correction, not a scientific retry, model change, or permission-relaxation claim. Runner durations are3.8749135s failed and3.7771675s successful; both remain charged historical execution work.

## What was actually tested

The successful run explicitly creates a **new seed3901 audit fork** from the old30-transaction checkpoint. It enables the new fork's model gradients openly because original inference checkpoints did not preserve those flags. It executes only:

1. one new warm learning step;
2. a full training-state save;
3. one next step in the continuous branch;
4. actual disk reload and the same next step in the restored branch.

There are exactly **three newly computed gradients**, represented by two continuous and one restored raw `coupled_gradient`/step records. All use the same four TRAIN terminal examples and own-old reference checks. No held-out evaluation, selection, admission, or confidence maintenance is performed. All three commits use the already executed V2 heads fallback at scale1 after seven rejected positive joint proposals. The model retains30 parameters plus8 fixed feature-buffer scalars.

The wrapper protocol includes inherited parameter-study fields such as `batch64`, VALID admission, a generic parameter-efficiency metric, and `transaction_policy=joint`. Those fields do not describe this integrity experiment. The dedicated scope, actual CLI/driver, restored session configuration, and raw records show a four-example objective, no validation admission, and `block_fallback`. This is a metadata-template limitation to avoid propagating into scientific summaries; it does not change what the trace actually executes.

## Independent data and archive checks

I decoded the three new owned PyTorch ZIPs with the checked restricted inert descriptor reader, including actual double and byte storages, without importing Torch or running model code. The saved boundary and both endpoints contain38 model/buffer scalars, recorded flags/control/source hashes, external Python RNG, and Torch CPU RNG bytes.

The saved boundary has next-iteration1, model transaction31, and logical audit-update1. Both endpoints have next-iteration2, model transaction32, and logical audit-update2. Thus30 original transactions are not relabeled as new work, and the shared warm step is not counted twice. Whole-archive bytes/hashes differ, as expected from branch provenance and timing; actual model values, shapes, strides, RNG contents, and relevant semantic state match.

All ten reported comparison categories are independently supported by original data: noise/event gradient records; within/class/combined gradient tensor hashes; TRAIN output vectors; own-old output vectors; model tensor bits; execution flags; RNG states; family/reference/receipt/cache/license state; chosen transaction; and next-step work counts. Continuous/restored endpoint model hash is `72671352d4b3651b7e6ba37368a29e2b1af1c36503d158490abd3f097cb61a22`. The two raw ledgers contain54 and27 events; their last step/gradient records equal the corresponding summary objects.

From the actual saved Python RNG state, an independent standard-library MT19937/rejection-sampling reconstruction produces the recorded next noise **−0.10983252466704929** in one proposal and reaches the endpoint's exact saved RNG state. This is a read-only sampling audit, not another training step. Both endpoint Torch CPU RNG byte arrays also match. The tested gradient uses Python sampling; a stochastic Torch layer, CUDA/NumPy generator, data loader, or optimizer with hidden state was not exercised and receives no broader recovery guarantee.

## Costs and remaining scope

Each new step accesses22 labels/check outputs, computes one within and one boundary VJP, and evaluates two chamber representatives. The three physical steps together use66 label accesses, three new continued programs,288 accepted continuation steps, and24 proposal trials:21 rejected and three committed. These are new audit costs, not extra benchmark optimization. Both next branches separately incur87 continuation steps and779 derivative/Newton/polynomial evaluations.

The logical ledger returns to the saved boundary so subsequent count limits and work deltas match continuous execution. Physical restoration is separately recorded: one additional constructor restore,0.0127462s restore wall, and0.0051503s boundary save wall. This overhead is not erased by logical counter restoration. The total new successful audit driver wall is0.1099640s, nested within its3.7771675s isolated runner; these are overlapping scopes, not additive totals. The new boundary/continuous-final/restored-final checkpoints are47,181/47,309/47,693 bytes. They contain RNG, references, receipts, caches, and provenance beyond the model's parameter payload.

**Supported boundary:** one actually saved CPU explicit-SGD state resumes through an identical stochastic next step under the same source/runtime configuration and ample stated limits. This closes039's prospective omission for this new helper path. The old inference-only artifacts still lack their original RNG states. Multi-step interrupted-process recovery, other generators/devices/optimizers, and future architecture variants need their own appropriate integrity coverage; none is claimed here. Continue the substantive fused-interface development rather than multiplying this finite bookkeeping test.

Evidence: `probes/topology-architecture/runs/parameter-hybrid-training-recovery-v1/` and `...-v2/`, their synchronization manifests, `versions/code-parameter-hybrid-training-recovery-v1/` and `...-v2/`, and the corresponding coordinator receipts/protocols. This review changes no old result, Method, paper, constitution, MAIN priority, or evaluation policy.
