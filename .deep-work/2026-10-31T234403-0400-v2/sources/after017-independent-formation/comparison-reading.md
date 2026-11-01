# Post-freeze original reading

Frozen candidate-v0.md and reading.md were not edited. Root authorized current-artifact comparison and this additional source record afterward. No source PDF, book, old artifact, or current candidate was modified. No experiment or training occurred. Provider-effective mode remains unknown.

## Macfarlane, Bonnet, van Hoof, Lelis — Gradient-Based Program Synthesis with Neurally Interpreted Languages

Primary body actually read: https://arxiv.org/html/2604.18907v1 . HTML original, §§2–3.4 in full, including Eq. (1)–(3), Algorithm 1 and the latent-search objective. The returned page also exposed part of §4; no numerical claim is adopted or independently reproduced. A PDF page number was not used; section/equation/algorithm references are the exact available locations.

Original variables entered: specification S, inductor q_phi, interpreter p_theta, latent tokens z_t, execution state s_t, output distributions pi_t, token and I/O embeddings, temperatures, skip gate, and test-time latent embeddings e.

Checked mechanism: leave-one-out I/O reconstruction and token-reuse regularization train an inductor plus shared recurrent executor. The executor predicts a token distribution and re-embeds it for the next step. Program search optimizes relaxed latent tokens; §3.4 describes retaining the soft representation at final decoding. This covers learned primitive semantics and reusable neural execution, not generated exact pair-local graph surgery.

Effect on comparison: semantic primitive acquisition/recombination is not a sufficient distinction. The setup infers latent programs from I/O specifications, so it is not silently equated to learning a raw-source evaluator under the candidate's information contract. Unread/unchecked: code, complete appendices, empirical resource accounting, all ablations, quantitative comparisons and external cited source bodies.

## Current artifacts actually read after authorization

Complete bodies: after014-higher-order/acquisition-v2.md; raw-stream-v1.md; root-acquisition-value-check.md; selective-capture-cost-audit.md; supervision/reviews/review-017.md; sources/after014-root/reading.md; sealed candidates/R/candidate-v1.md. The first parallel tool result was truncated, so affected files were subsequently opened separately, including acquisition-v2's later sections. These are campaign artifacts with their own evidence labels, not independent source verification.

The comparison uses no experimental claim from R. Its actual v1 explicitly distinguishes a supplied-law numerical probe from unexecuted rewrite acquisition and an unimplemented general checker. No CEGIS, learned rewrite table, or neural training was assumed to have run.

The graph and pointer call/return traces, fixed-register observation, single-predecessor owned-value capture cost, and O(k) shared-pointer execution for the N/H family are new manual reconstructions. The graph witness's actual Project row already gives an Omega(k^2) copying cost on repeated N; no total graph upper bound is asserted. No runtime, measured acquisition cost, equality of learners, or neural convergence is claimed. V0's already recorded primary reading of NRAM's controller/register/READ/WRITE/discretization method is the basis for the concrete pointer organization; no identical-K historical learner is invented.
