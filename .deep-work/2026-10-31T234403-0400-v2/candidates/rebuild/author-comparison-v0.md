# Author comparison after own full freeze

## Judgment

No algebraic discrepancy was found between the author's main exact
simultaneous-SGD learning-state construction and the independent micro-law
reconstruction. A separately frozen interpreter now also executes the actual
generated scalar/matrix JSON programs correctly on the fixed finite replay.
The previous evidence gap was real: the author replay used manually derived
block laws. The author technical receipt disclosed that gap correctly.

This comparison began only after the independent v0 derivation, code,
execution, result and certificate were frozen in `full-freeze.sha256`, whose
manifest SHA256 is 638c8cad5db5864e991bf028844c4d0d2f8e6f35d8a08d58ff21a255cce88f41.
That original freeze was checked unchanged after the new replay.

## Equation comparison and useful refinement

The author's V,U,A,B,H,M correspond to the reconstruction's A,B,P,Q,S,H.
Under this naming correspondence, the stacked update, PSD congruence, all
three block laws, width-at-most-m+n factor realization, and unilateral-zero
path append law agree exactly. The independent scalar reduction to (w,p+q)
agrees with the author's two generated scalar observables.

The two independently chosen initial-response witnesses differ in shape and
commands, but expose the same missing state. In the independent reconstruction,
equal W and response T imply D_P=cI_m,D_Q=-cI_n, and a first command changes
the response difference to

    D_T_next(K)=c eta^2(KG^TG-GG^TK).

For the author's c=1, G=diag(1,2), K=E12, eta=1/4 this gives
D_T_next(K)=3/16 E12 and second-product difference -3/64 E12, exactly the
author's stated result. The reconstruction's smaller rectangular witness has
second-product difference +1/8 E12. Neither requires initial response
information to be approximately matched.

The independent scalar exception refines any possible minimality reading:
full Gram state is sufficient, but for m=n=1 its difference p-q is invisible
to every permitted future sequence. The author does not contradict this;
its scalar generated closure retains just two coordinates.

## Why the generated tenth matrix statistic matters

The emitted matrix event statistics, read directly from the JSON, are:

index | statistic in the reconstruction's notation
--- | ---
0,1,2,3 | W00,W01,W10,W11
4 | P01
5 | Q01
6 | P00+Q00
7 | P00+Q11
8 | P11-P00
9 | P00

The first nine determine W and the full initial response operator. They
are invariant under P->P+cI,Q->Q-cI. The final emitted coordinate P00 removes
that ambiguity. It was generated from pulling back event 4, at quadratic
control exponent [0,1,0,1], consistent with the need to preserve future
response rather than just current response. This explanation is derived from
the independently frozen characterization, not inferred solely from the
reported state count.

The author's discovery code inserts all linearly independent coefficient
programs exposed by pullback. Its expected dimensions are post-construction
assertions; inspection found no supplied list of target Gram statistics or
dimension-based basis selection. This is a code-inspection statement, not a
re-execution of the closure discovery algorithm. Width-2 matrix discovery has
eight source variables and ten algebraically constrained emitted coordinates;
the author explicitly disclaims a storage advantage in that particular case.

## Actual generated-program execution

After schema inspection, `generated-replay-contract.md` and the independent
`generated_replay.py` were frozen before execution. The interpreter reads the
actual `events` and `closed_updates` arrays and retains all emitted coordinates;
state count comes from the events, with JSON `dimension` used only for a
consistency assertion. It imports no author Python and substitutes no
handwritten Gram recurrence.

One isolated process with external timeout 90 seconds exited 0. Its smoke
serialization preceded calculations. At all six own scalar commands and all
six own matrix commands, every interpreted coordinate equalled its emitted
statistic evaluated on separately updated OLD-factor source loops. Output
origin coordinates also equalled the independent source output. Initial and
final exact states roundtripped. Both intentional generated-coefficient
mutations were detected. Raw step states and source factors were fsynced
incrementally; `generated-replay-raw.jsonl` ends in FINAL_PASS.

Observed coordinate counts were 2 and 10. These were outcomes/read-schema
facts, not selection criteria. No retry, tolerance or post-result code change.
All author input hashes and pre-execution hashes remained unchanged.

## Evidence boundaries and discrepancy record

No equation mismatch, generated-program mismatch, or frozen-input hash change
was found. The material evidence addition is independent execution of the
emitted programs, closing a disclosed finite-replay gap. It is not a universal
proof of the generator or all programs it could emit. The symbolic construction
algorithm was inspected, not independently rerun or formally verified here.

The author momentum extension, neural/meta-learning proposal, adaptation-benefit
calculation, hardware/cost claims and literature/novelty boundaries were outside
the neutral micro-law check and this scoped replay. This worker does not
independently certify those claims. No neural proposer, meta-controller,
task-performance advantage or nonlinear internal closure was tested.

CLAIM | STATUS | EVIDENCE | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION
--- | --- | --- | --- | --- | ---
Main exact SGD/Gram state laws agree with independent reconstruction | VERIFIED in stated analytic/finite scope | unchanged own v0 freeze; equation mapping above | No proof-assistant certificate; exact-law assumptions matter | None | Coordinator assesses scientific significance and prior-work boundary
Actual emitted scalar/matrix updates replay their specified source factors | VERIFIED for the frozen finite replay | generated_replay.py; generated-replay-raw.jsonl; generated-replay-summary.json | Finite inputs do not prove all program identities universally | None; mutations intentionally rejected | No further numerical replay needed for this finite objective
Learned discovery/use of consequential internal mechanisms | OPEN / untested | Author proposal only | Correctness and construction of these toy closures do not establish that ambition | Not investigated | Return the result to coordinator; no automatic training or expanded experiment
