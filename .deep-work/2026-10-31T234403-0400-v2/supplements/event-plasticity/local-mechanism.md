# Local plasticity after compaction: mechanism judgment

**Judgment:** local eligibility is relevant as explicitly retained learning state and as a possible interface for acquiring new repair branches. It does not supply a missing learning signal, allocate topology by itself, or reconstruct discarded information. The useful question is which small set of owned edit traces should survive compaction or be built from later observations, and how task feedback should select their structural consequences. No conversion to an SNN is indicated by this reading.

Source facts and exact locators are in `sources/local-reading.md`. The following are our RECONSTRUCTIONS/PROPOSALS, not results claimed by those papers.

## Information and credit boundary

Separate four transitions:

| Transition | State/action required | What it does not establish |
|---|---|---|
| Signal propagation | Current inputs, weighted connections, neuron/membrane state, delays | That a useful computation was learned |
| Local eligibility | Stored pre/post or voltage/sensitivity traces with explicit ownership | Whether the marked event helped the task |
| Plastic weight update | Eligibility multiplied by an error/reward-dependent learning signal | Allocation of a nonexistent edge or new operator |
| Structural birth/deletion | Explicit mask/endpoints, allocation/proposal/deletion rule and budget | Recovery of erased historical structure |

The missing learning-signal problem is concrete. Two networks can have the same local event/eligibility at a unit, yet opposite signs on its downstream readout. Correct deterministic loss gradients can then have opposite signs. Local temporal memory alone cannot choose between them. Stochastic reward learning can estimate useful credit from additional trials, but those observations and their variance/time costs must be charged.

If two pre-compaction systems map to the same retained state and receive the same later observations, feedback and coupled randomness, a local update rule operating only on that retained state cannot distinguish their erased histories. Traces, timestamps, adaptation variables, endpoint identities or retained random seeds can make the states different; then the information was retained. Later informative observations can support new learning or inference; that is additional evidence, not spontaneous recovery.

For plasticity-aware compaction, preserving current outputs is insufficient. The retained state must support the allowed future update/query actions. In R language this includes edit ownership and transported credit; in K language H alone may cease to be a complete state once additional local traces are retained. A fair thin-factor/rate-network rival receives those same traces, feedback and candidate slots, with their full memory/update costs.

## Opportunity 1 — retained edit-port eligibility for new nonlinear repairs

**PROPOSAL.** Couple a bounded set of explicit repair ports to persistent eligibility, rather than keeping all deleted source neurons or claiming to recover them. A port owns a concrete candidate computation, its parameters and its trace. For P's zero-born nonlinear repair, consider

`f_t = f_compact,t + c * d_t`, `d_t = v phi(u^T x_t)`, `c=0` at birth.

The feature parameters u,v must be retained or newly proposed; the feature d_t must actually be computed. Hold the branch parameters fixed while accumulating this trial's trace and update at feedback. If an output integrates this branch with leak kappa, the exact gate sensitivity in this restricted model is

`e_t = kappa e_(t-1) + d_t`.

A delayed supervised output error delta_T updates `c <- c - eta delta_T^T e_T`. For a stochastic policy one instead needs a policy-score eligibility and an appropriate return/critic estimator; multiplying arbitrary Hebbian coincidence by reward is not automatically that estimator.

This gives a genuine new *allowed action* after compaction: a new nonlinear branch can acquire influence from newly arriving evidence while the compact executor remains available. It does not reconstruct the deleted factor network. A branch first allocated at time tau starts with e_tau=0 unless the necessary earlier feature/history was explicitly retained. Zero output gate preserves output at birth, but inner u,v gradients through this gate initially vanish; candidate-feature computation and later activation are real costs.

R's compact edit carrier could represent a small collection of such traces and their transports. Replacing full sensitivities with purely local traces still requires an honest account of the missing learning signal and any approximation. This is a constructive intersection with R's ownership problem, not proof that its closure stays small.

**Why it could matter:** streaming tasks with delayed feedback and a small number of useful reusable repair mechanisms might avoid retaining complete activation histories. **Strong ordinary rival:** the identical candidate branches with standard forward sensitivities/eligibility and the same feedback. The unresolved contribution is acquiring/selecting a useful, reusable set of ports and transports under total cost, not the familiar trace recursion.

## Opportunity 2 — explicit exploratory edge/branch birth, then local evaluation

**PROPOSAL.** Use a bounded structural proposal process to allocate a candidate edge or small repair branch; only then collect local eligibility and let task feedback determine its subsequent strength/retention. The active edge list must actually change. At deletion, specify whether the slot's identity/latent parameter is retained or discarded. A parameter changing from small to large is not sufficient evidence of structural construction.

A concrete state is `(active endpoints, candidate endpoints, theta, local traces, prior, proposal clock, budget)`. A prior/noise-driven birth process chooses an endpoint pair from current available ports. Once active, a reward-modulated eligibility rule changes its weight; deletion removes it under an explicit sparsity/utility policy. If a new nonlinear branch is attached with output gate zero, birth can preserve the old output, but shadow feature/state computation is still allocated and counted. Otherwise even a small new weight perturbs current behavior and requires retention checks.

Crucially, a truly disconnected candidate cannot be assigned a useful task-specific eligibility for pre/post events it did not observe. Either its dynamics were kept as a shadow slot, incurring storage/computation, or its birth is exploratory and later observations supply the evidence. Keeping all possible-edge latent variables would reintroduce a potentially quadratic hidden reservoir. Sparse waiting-time proposals avoid that reservoir only by accepting the corresponding change in state/history and search behavior.

**Why it could matter:** it gives a concrete route to new interactions after an exact finite-rank reservoir is exhausted, using fresh observations rather than erased directions. **Strong ordinary rival:** reward-driven sparse rewiring or generic candidate-branch growth with the same proposals, traces and feedback. The source mechanism is already known; a contribution would require a demonstrable advantage from acquired port organization/composition, not relabeling rewiring as self-assembly.

Both opportunities remain OPEN. Neither supplies a foundational principle yet. No experiments, training, dependency changes, purchases, source-code execution, main-report edits or frozen-artifact edits were performed in this supplement.
