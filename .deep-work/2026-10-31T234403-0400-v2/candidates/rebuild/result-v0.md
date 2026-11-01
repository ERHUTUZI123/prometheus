# Frozen independent result v0

## Judgment

The supplied simultaneous-SGD micro-law has an exact width-independent
sufficient state: S=[A;B^T][A;B^T]^T, with S_next=H S H^T. Its real PSD
factorization permits a replacement hidden width at most m+n. W alone loses
learning information. Even W plus the entire initial gradient-response
operator loses future behavior in dimensions beyond the scalar exception.

The analytic derivation was frozen before execution and remains unchanged.
The one inspected exact-Fraction executable check passed all frozen criteria
through the supplied isolated runner with timeout 90 seconds and process exit
0. Serialization smoke preceded numerical tests; each raw result was fsynced
to `probes/rebuild/raw.jsonl`, which ends in FINAL_PASS. No retry or code
revision was performed. No author artifacts had been opened at this freeze.

## Evidence and scope

- Exact closure: 20 fixed-seed cases, dimensions (1,1),(1,2),(2,1),(2,3),(3,2),
  widths 1,2,5,13; six steps each, 120 full-state comparisons. Direct OLD-factor
  updates agreed exactly with independently represented Gram congruence,
  product and sampled-input output. Every final case state roundtripped.
- Current-function counterexample: same w=1, exact next-product difference
  -3/8 at the frozen g and eta.
- Initial-response counterexample: W=0 and T(G)=4G for both blocks, with
  common first product [-2,0] and second products [-2,-9/4],[-2,-17/8].
  The statement for every initial G follows analytically; four numerical
  probes alone are not being treated as a proof of that quantifier.
- Scalar exception: exact (w,p+q) recurrence and two differently factored
  states agreed across eight commands.
- Zero-product appends: exact PSD block-diagonal state increment, predicted
  first response difference, and four subsequent direct/summary comparisons.
  A separately appended all-zero path stayed zero throughout those commands;
  invariance for all commands follows from the stated update equations.

These are checks of exact finite-dimensional state evolution, not evidence
of novelty, learned internal invention, neural task success, better hardware
performance, open-ended evolution or a foundational AI contribution. The
spectral compaction existence argument is analytic and was not separately
implemented by the executable check. Floating-point stability is untested.

## Required state record

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION
--- | --- | --- | --- | --- | ---
Width-independent sufficient state for the exact stated SGD law | VERIFIED in analytic reconstruction plus bounded exact-arithmetic checks | candidate-v0.md; check.py; raw.jsonl; summary.json | Not formal proof-assistant checked; applicability limited to exact stated law | None in frozen check | Compare independently frozen author equations and evidence when coordinator authorizes
Initial function and response do not generally specify future learning | VERIFIED by explicit realizable two-step counterexample | candidate-v0.md; TWO_STEP_PAIR_PASS records | Does not imply gains from manipulating the retained state | None | Same comparison decision
Scientific novelty/significance or learned mechanism construction | OPEN / not claimed | No relevant empirical or literature study in this reconstruction | Correct sufficient-state accounting alone is not a foundational mechanism | Not investigated | No additional experiment authorized by this result
