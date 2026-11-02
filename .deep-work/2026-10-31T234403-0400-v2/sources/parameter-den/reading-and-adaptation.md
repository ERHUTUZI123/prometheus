# DEN original mechanism and explicit terminal-sequence adaptation

Purpose: current TODO5 competitive baseline and TODO2 causal comparison. MAIN=TOPOLOGY remains human-locked. Historical M0–M3 records are preserved; they are not the current progress axis. This document is source evidence for one baseline, not another living Method or research plan.

## Sources actually read

- Yoon, Yang, Lee and Hwang, *Lifelong Learning with Dynamically Expandable Networks*, ICLR2018. Requested [OpenReview PDF](https://openreview.net/pdf?id=Sk7KsfW0-); an initial browser challenge was bypassed by reading the substantive primary [arXiv PDF](https://arxiv.org/pdf/1708.01547). Local `den-arxiv-1708.01547.pdf`, 11 pages, SHA256 `16182ee5c29e3d22095a663e93f351f05ba852c4bf56209e408ddafdb5077d92`. Entered printed pp3–7, especially pp4–6, equations2–6 and algorithms1–4. Exact source text extraction is preserved; missing optional fontTools produced font-decoding warnings, not invented equations. The later empirical plots/appendix were not independently reconstructed.
- [Author repository](https://github.com/jaehong31/DEN), commit `52ebb5da255a6225500d8a56b872ba643dd41572`: actual full `DEN.py`, `ops.py`, `DEN_run.py`, README. Exact Git blob hashes were checked against the downloaded bytes; pinned URLs/raw SHA256 are in `provenance.json`. Downloaded code was **not executed or imported**. GitHub connector supplied contents; pinned raw bytes corrected an intermediate newline-normalized local copy before these sources were used as byte-identical evidence.
- The `hf` paper-read fallback was attempted using the installed CLI; that paper was absent from the Hub. No dependency was installed and no cloud or paid job was created.

## Source mechanism reconstructed precisely

Use ordinary feedforward incoming-weight matrices with a task-specific output head. New-task data is available; previous datasets are not needed by the algorithm (paper p3). The article simplifies exposition to binary tasks and later evaluates image/feature classification; it is not a recurrent program-learning architecture.

1. **Sparse initial fit and selective retraining** (p4 Eq2–3, p5 Eq4, Algorithm2). First learn a sparse network. For a later task, hold all feature layers fixed and fit only its new L1-regularized head. Start at nonzero head connections and traverse nonzero incoming edges backward. Retrain that selected shared subnetwork while leaving its complement unchanged. In our notation:

   \[
   w_t\leftarrow\arg\min_w L_t(w;W^{t-1})+\mu\|w\|_1,
   \quad S_t=\operatorname{Ancestors}(\operatorname{supp}w_t),
   \]
   followed by task loss plus weight regularization on \(W_{S_t}\), with \(W_{S_t^c}\) fixed. Paper Eq4 prints an L2 norm; author code uses `tf.nn.l2_loss`, i.e. half the squared norm, alongside the drift term.

2. **Loss-triggered sparse expansion** (p5 Eq5, p6 Algorithm3). If the fitted task loss exceeds tau, temporarily append k units at each feature layer. Optimize new weights using L1 plus a sum of incoming-unit group norms; delete groups driven to zero. The reconstruction is

   \[
   W_N\leftarrow\arg\min L_t(W_N;W_{\rm old})+
      \mu\|W_N\|_1+\gamma\sum_{g\in G}\|W_{N,g}\|_2.
   \]

   Expansion is conditional, not one permanently reserved task module per task. Old tasks receive no connections from newly appended hidden units during timestamped inference.

3. **Drift-based splitting** (p6 Eq6/Algorithm4). Fit the current task with a penalty toward the prior weights, measure each old neuron's incoming-vector displacement \(\rho_i=\|w_i^t-w_i^{t-1}\|_2\), and duplicate units with \(\rho_i>\sigma\). Copies initialize a separate current representation; old versions can retain previous incoming weights. Retrain after duplication. Weight drift is a heuristic semantic proxy, not a proof that old task functions are protected.

4. **Timestamped inference** (p6). Each new or split unit records its birth task. When evaluating an old task, omit units born later than that task. Unsplit shared weights may still change, so timestamps alone are not a guarantee of no forgetting. This is different from freezing all old parameters.

## Actual author-code details and differences from the paper

Pinned local line references use `author-DEN.py` and `author-DEN_run.py`.

- `DEN.py:282–342`: Adam optimizes task loss plus L2 weight cost and prior-parameter deviation. The L1 update subtracts the literal `l1_lambda` each optimizer step, not eta times that value. Group shrink likewise uses literal `gl_lambda`, on newly introduced tensors. The two proximal assignments both depend on Adam but are not explicitly ordered relative to each other; a deterministic port must choose an order and disclose it.
- `DEN.py:371–442`: head fitting, backward nonzero selection, selected-parameter fitting and reinsertion are real separate phases. However, the top selection tests only output column0 and selected-head slicing uses `[0]`; this cannot be copied unchanged into an N-class task.
- `DEN.py:376–377,572–601`: selective-head call starts the iteration counter at `early_training`, then the selective stop condition is already met after its first update. This is an apparent implementation/control-flow discrepancy from the intended one-tenth warmup, not an empirical result of running that code.
- `DEN.py:444–538`: this implementation performs splitting inside the expansion branch. Paper Algorithm1 places splitting after the conditional expansion. Its variable `cosim` is an L2 displacement, not cosine similarity; `argsort(cosims)[:ex_k]` takes the smallest threshold-exceeding distances. A faithful-to-paper port need not reproduce those choices silently.
- `DEN.py:233–245,539–565`: inference slices feature widths using per-task stamps. The example appends an output width10 and freezes named first/second layer scopes between graph rebuilds. These are example-specific details, not a general sequence interface.
- `ops.py` supplies ROC/AUC helpers, not the expansion/splitting rules.

The example defaults (`DEN_run.py:9–21`) are: iterations4300; Adam initial LR .001 with exponential .95 decay per dataset-scale step; batch256; feature widths312,128; L1=1e−5; L2=1e−4; group shrink=1e−3; prior-deviation coefficient .5; k=10; loss threshold .01; drift threshold .05. Those numbers accompany a sigmoid/BCE multiclass example and AUROC reporting. They are **not established tuned values for N-way terminal sequence CE**.

## Implemented new baseline interface — not yet executed

`probes/topology-architecture/parameter_den_baseline.py` implements `DENSequenceBaseline`, explicitly named **DEN-RNN terminal adaptation**. It does not claim official reproduction. A fixed-position flattened MLP would have untrained input coordinates at held-out lengths9–16, so the shared input feature layer is instead recurrent:

\[
h_0=\operatorname{ReLU}(U e_x+b),\qquad
h_{j+1}=\operatorname{ReLU}(R h_j+A e_{a_j}+b),
\qquad \ell_t=H_t h_L+c_t.
\]

PAD leaves the state unchanged. Task ID chooses the public output head; it supplies no hidden state, operator or correct program. All symbols share the same recurrent computation. The task-specific head is created on demand and counted as a real parameter increment. All previous heads remain stored and counted. A single shared recurrent body is selectively reused; this is not an independent network per task.

For a fixed600-step task budget, task0 trains the initial sparse network. Later tasks use 60 sparse-head updates,120 selected-subnetwork updates,240 expansion-or-stability updates, then180 post-split updates. The source phases' work is therefore included in the common600 rather than receiving extra unreported training passes. Trigger loss is the mean observed current TRAIN CE during the phase, not a test-selected criterion. A runner may use a prospectively specified common curriculum; the class itself does not read datasets or labels.

The selected recurrent subnetwork is the transitive predecessor closure of nonzero R edges from nonzero head units. It can encompass the entire recurrent body. Gradient masks plus actual restoration after Adam preserve inactive entries despite optimizer momentum. Dense forward/backward computation is still performed; `active_after_DEN_masks` must not be called a measured sparse-kernel speedup.

Expansion appends up to k recurrent channels, trains new rows/columns with group shrink on the new incoming vectors, and prunes zero incoming-weight groups. Drift combines incoming token/start/recurrent weights on the pre-task coordinates. Splitting copies current rows, restores old versions' prior incoming weights, appends new versions and repeats fitting. Current outgoing duplication can change the new task's function, making post-split fitting necessary; no function-preserving initialization or all-old-task correctness is assumed. Largest drift-first selection with an explicit split cap is a bounded adaptation rather than the author's ascending-sort behavior. Every censored split/expansion is reported.

Old-task inference crops both recurrent axes and the corresponding start/token/state/head channels to that task's timestamp. Shape changes and phase changes return `optimizer_rebuild_required`; the author graph-reset behavior is made explicit. Shared unsplit neurons remain plastic under the prior-deviation penalty.

Persistent parameter count at width h with k allocated N-way heads is

\[
P(h,k)=h^2+h(\mathrm{vocab}+2+N+kN)+kN.
\]

The class enforces an explicit global `parameter_budget` as well as a hidden-width safety cap. At N8, widths32/64 with one head cost1800/5640 P, useful capacity points near or above the current compact native model. Formal comparisons must pass the same cap explicitly; the default250000 is only a finite safety/configuration default. A new task head that cannot fit is recorded as censored acquisition rather than deleting old stored weights.

`resource_counts()` reports all P, active masked scalars, previous-body snapshots, gradient masks, per-step inactive snapshots, timestamp/birth buffers, optimizer tensor storage and growth counters. `extra_state_dict()` retains actual phase state, masks/snapshots, counters and structural history; full serialized metadata bytes still need the runner's accounting. `den_model_from_checkpoint` restores own weights and metadata; the coordinator must restore saved RNG after neutral factory construction when continuing training.

## Runner integration and remaining evidence obligation

```python
model.set_stage(task, total_steps=600)
optimizer = torch.optim.Adam(model.parameters(), lr=chosen_lr)
for step, batch in enumerate(the_same_600_training_batches):
    optimizer.zero_grad(set_to_none=True)
    ce = cross_entropy(model(batch.tokens, batch.starts, batch.task_ids), batch.targets)
    (ce + model.regularization_loss()).backward()
    model.mask_gradients_()
    clip_grad_norm_(model.parameters(), 1.0)  # mask before clipping
    optimizer.step()
    event = model.after_optimizer_step_(float(ce.detach()), step + 1)
    if event['optimizer_rebuild_required']:
        optimizer = torch.optim.Adam(model.parameters(), lr=chosen_lr)
```

This code has passed AST inspection only. No unit smoke, training, numerical reconstruction, accuracy or fairness victory is claimed. The next execution obligation is bounded shape/mask/timestamp/checkpoint testing in the authorized runner, followed by task0 acquisition before interpreting a six-stage comparison. LR, width, normalized loss threshold and phase allocation require the same disclosed validation/HPO budget as other baselines; defaults are source-informed starting points. If it fails even a short task, diagnose and repair the adaptation rather than labeling random performance a DEN result.

This baseline tests whether an ordinary sparse, dynamically growing shared neural architecture explains a putative parameter-efficiency gain. It does not replace the Transformer primary baseline, use topology, inherit native terminal-profile formation, or establish that DEN is the strongest possible conventional learner.
