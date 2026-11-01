# Frozen independent check plan

One inspected standard-library Python process through the existing network-
denied, write-restricted executor, wall timeout 90 seconds. No new dependency,
network call, training run or sweep. Exact Fraction arithmetic, with no numeric
tolerance. Seed 742019 determines a fixed small synthetic check set; no seed
selection or retries after results.

Before any numerical claim, write/fsync an opening JSONL record; serialize a
rational matrix state, read it back, assert exact equality and persist the
smoke result. Flush/fsync every subsequent result immediately. Unhandled
exceptions or missing final certificate are failures, not partial passes.

1. For shapes (m,n)=(1,1),(1,2),(2,1),(2,3),(3,2) and h=1,2,5,13, generate
   fixed rational A,B with entries numerator in [-3,3], denominator 5. Across
   six fixed-seed arbitrary G commands with entries numerator in [-3,3],
   denominator 7, and eta=1/(step+3), compare direct old-factor simultaneous
   updates against the independently represented congruence S_next=H S H^T.
   Require exact full Gram agreement, product agreement, and output agreement
   on a generated rational x after every step. Roundtrip the final state.
2. Check the scalar current-product counterexample at g=2/3,eta=1/4 and require
   the exact predicted product difference -3/8.
3. Check the frozen m=1,n=2 initial-response pair. Require W and T agreement
   on four listed G, equal first updated product [-2,0], and exact second
   products [-2,-9/4] and [-2,-17/8].
4. Check scalar reduction for (a,b)=(1,2) and (2,1) across eight generated
   gradients and rates. Require exact agreement of direct products and (w,r)
   recurrence, and equality of the two trajectories.
5. Append one nonzero A/zero B path and one zero A/nonzero B path to a frozen
   m=2,n=3,h=2 block. Require exact block-diagonal Gram increment and equal
   subsequent states under four commands; check the exact first response
   difference. Also check one all-zero path remains zero through four steps.

Frozen inputs and all checks are in `check.py`. A single final PASS is emitted
only if every assertion passes; raw JSONL records support the certificate.
