# Pivotal check for proposed ownership fission

QUESTION: Does function-preserving copy-on-write by itself change learning beyond a changed call-space kernel, and does local gradient conflict certify that a finite tied repair is unavailable?

GIVEN: N scalar call occurrences, literal alias incidence S, call parameters theta=S phi. For any differentiable composed execution/loss, owned SGD gives theta_next=theta−eta S S^T g, where g is the full call-coordinate gradient. Fission copies the old scalar and changes S, preserving current theta. A normalized owned update instead has kernel S diag(1/block_size) S^T. This is a mathematical reconstruction to distinguish the contribution, NOT an assertion that a future learned ownership policy is already prior art.

TEST: three fixed seeds221/222/223;80 small nonlinear SGD batches,6call parameters, ownership splits at20/45/65. Compare independent alias-state owner updates with call-coordinate block-kernel updates, including birth outputs. Check the actual analytic call gradient by finite differences. Tolerances1e-10 equivalence,1e-5 derivative. No learning claim for the handchosen split schedule.

DECISIVE COUNTERMODEL: tied scalar theta, old observable theta²=1 at theta=−1, desired new observable theta=1. The old local feasible tangent is zero, but the finite flip theta→+1 satisfies both. Also at theta=−.1 the two zero-compatible losses(theta²−1)² and(theta−1)² have opposite gradients. Require exact Fraction checks. A gradient conflict can propose a split; it is not a proof that new capacity or ownership is necessary. Equal-budget replay of tied finite alternatives must precede such a claim.

BUDGET: one inspected offline isolated process <=45s CPU/wall,3*80steps; no sweep, retries, policy training or new dependencies. Serialization before calculation, incremental JSONL and final summary. Stop on mismatch/timeout/nonfinite state. This supports a mechanism boundary, not foundational novelty, full learned fission usefulness or a campaign endpoint.
