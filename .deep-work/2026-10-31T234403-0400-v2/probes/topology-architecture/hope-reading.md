# HOPE source check and callable reconstruction

**Status: original-source reading plus unexecuted implementation. No HOPE training result is claimed.** This artifact supports the current strong-baseline work; it changes neither the human-locked topology MAIN nor the canonical Method.

## Original reading actually performed

Checked the [Google Research announcement](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/) and the extended [arXiv v1 paper](https://arxiv.org/pdf/2512.24695v1), with its [equation-preserving HTML](https://arxiv.org/html/2512.24695v1). The OpenReview manuscript requests the extended version. Read the load-bearing CMS and self-modifying sections, rather than inferring an implementation from the announcement.

Primary mechanism locations: §7.1, printed pp.27–28, Eq.70–74; §8.1–8.3, pp.30–34, particularly Eq.83–85, 88, 90–97; and §9.6/Table6. CMS schedules different MLP update frequencies. Self-modifying memories produce their own targets and adaptive projections/update gates. The paper supplies an explicit linear/L2 recurrence. Full experiments instead use residual two-layer memories and additional momentum. The explicit fixed-query statement and the adaptive-query ablation do not give the same specification. Appendix C and the full experimental recipe have not been independently reconstructed here.

Original snapshots and hashes are in [sources/hope](../../sources/hope/source-receipt.json). The downloaded PDF/HTML are source evidence, not executed code. PDF equations were read through original text and HTML LaTeX; no visual PDF-render audit is claimed.

## Official implementation check

The author's [publication entry](https://alibehrouz.com/publications/) links paper, blog and presentation, without a code link. The public author repository [ABehrouz/Titans](https://github.com/ABehrouz/Titans) reports size zero; its tree endpoint returned HTTP409. Checked the author's public repository list and Google Research repository root/search; no official HOPE implementation was obtained. The saved receipt includes author-page and repository snapshots. This is a checked-search boundary, not proof that no official implementation exists anywhere. Third-party replicas were not treated as author code and were not downloaded or executed.

## Implemented reconstruction and exact assumptions

[parameter_hope_baseline.py](parameter_hope_baseline.py) exports `SelfModCMSReconstruction`. **It must be labelled “Self-modifying linear memory + CMS reconstruction”, not “official/full HOPE”.** A result for this implementation cannot establish a performance ranking against the complete paper architecture.

For each block, five learned matrix initial states become per-example fast states: key, value, learning-rate, retention, and retrieval memory. A slow learned query projection follows the explicit fixed-query text. All five matrices change within a sequence. Using column-vector notation and chunk size one, the implemented Eq.93 specialization is

\[
k_t=\operatorname{norm}(M_{k,t-1}x_t),\quad
v_t=M_{v,t-1}x_t,\quad
\widehat v_{j,t}=M_{j,t-1}v_t,
\]
\[
M_{j,t}=\alpha_t M_{j,t-1}
-\eta_t\bigl(2M_{j,t-1}k_t-\widehat v_{j,t}\bigr)k_t^\top.
\]

Both appearances of `M k` are intentional: one is DGD's multiplicative rank-one term, the other the L2 residual. The half-squared-L2 gradient convention matches the displayed linear recurrence. The inner gradient treats the generated target as fixed; outer terminal-loss BPTT differentiates the entire explicit target-generation and recurrence. No detached surrogate replaces that outer computation.

Engineering choices, not claimed author settings: bounded scalar gates from the mean of their vector-memory outputs; `eta_max=.1`, `decay_max=.01`; normalized q/k; learned depthwise causal window4; residual wiring; random orthogonal half-scale initial matrices; one to three stacked blocks. The default width64/two-block choice is adjustable and must be reported, not selected using test data. There is no invented residual-MLP right-preconditioner or undocumented momentum hook; those are remaining fidelity gaps.

CMS is a sequential chain of residual MLPs. Its periods default to `(1,4,16)` **terminal-feedback minibatches**. Each level accumulates its own real terminal-loss gradients, updates its own AdamW only at its scheduled event, and retains pending gradients/moments across stages. Averaging accumulated gradients explicitly fixes the Eq.71 learning-rate normalization. This benchmark supplies terminal labels, not tokenwise labels, so using feedback minibatches as the CMS context unit is a disclosed adaptation. No claim of implementing the paper's full cross-context meta-initialization or M3 optimizer is made.

## Information flow and use

The forward interface is identical to the existing Transformer baseline:

```python
from parameter_hope_baseline import make_hope_reconstruction, make_optimizer

model = make_hope_reconstruction(n_states=8, width=64, n_tasks=6).to(device)
optimizer = make_optimizer(model, lr=3e-4, cms_lr=3e-4)
# Keep this optimizer, its clock and pending gradients across all six stages.
for stage in range(6):
    model.set_stage(stage)
    # Caller selects exactly the common, frozen training minibatch indices.
    optimizer.zero_grad(set_to_none=True)
    logits = model(tokens, starts, task_ids)
    loss = cross_entropy(logits, terminal_targets)
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
# Save BOTH model.state_dict() and optimizer.state_dict().
```

Only public start/task/symbol IDs enter forward. Fast memory receives no answer, teacher operation, hidden mode, or correct route. Its generated self-targets contain no new task information; terminal CE is the external success signal. Fast matrices reset to learned initial states for independent examples, including evaluation; persistent shared matrices/CMS weights learn across stages. An extra model READ token retrieves the terminal representation. PADs advance neither fast memory nor convolution history. No replay is silently added.

Do **not** use ordinary `Adam(model.parameters())`: that removes the multi-frequency mechanism. The default six preallocated task codes are charged fully in P. If task codes are dynamically grown, rebuilding the optimizer loses nested state unless explicitly migrated; that is an additional policy, not silently equivalent. `set_stage` itself does not reset or freeze any learned weight.

## Accounting and execution boundary

`model.resource_counts(optimizer)` reports real P, charged initial fast matrices, ephemeral fast-state shape, CMS P, optimizer tensors and a bounded forward dense-product proxy. `optimizer.resource_report()` includes each level's update count, clock and pending-gradient bytes. Registered tensors, learned fast-state initial conditions and Adam/CMS state are distinct costs. With width `d`, `k` CMS levels, `L` blocks and `T` task codes, the exact P check is

\[
P=L[(7+4k)d^2+(6+5k)d]+(V+2N+T+4)d+N.
\]

Runtime fast matrices scale as `5*B*L*d²`; autograd saves additional tensors. Logical shape/proxy counts do not establish allocator peak, full FLOPs, energy or wall time. The experiment runner must measure CUDA-synchronized wall time and allocated/reserved peaks, count identical terminal labels and steps, and retain full learning curves. No hidden persistent inference history crosses examples.

**Checked here:** AST parsing and source-level recurrence/interface/accounting review. **Not executed here:** Torch import, forward/backward smoke test, optimizer-cycle test, checkpoint round-trip, training or SSH. Before the bounded six-stage run, the coordinator should run one real minibatch/backward; confirm all five initial memories receive gradients, CMS event counts at steps1/4/16, output/optimizer round-trip equality, finite loss and resource counts. Those executions must have their own actual receipts.
