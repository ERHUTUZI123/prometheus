# P finite factorization check — exploratory contract

Purpose: test only the finite transition reconstruction and ambiguity consequences in frozen candidate-v0.md. This is not a neural-training, utility, novelty, or evolvability evaluation.

State space: six symbolic states, secretly constructed as Z_3 x {0,1} and permuted with a fixed standard-library PRNG seed 83. Probe inference sees the resulting transition tables and enumerates all 6! bijections from physical states to active/context coordinates. It receives the proposed factor sizes 3 and 2; it does not infer those sizes. A factorization is accepted exactly when every observed transition preserves c and induces a deterministic f(a,q) independent of c.

Cases, fixed before execution:

1. Full inc transitions: predicted 36 charts. The independent origin phase of each three-cycle is not aligned by inc alone.
2. Full inc and reset transitions with one common chart: predicted 12 charts. Reset removes the relative-origin gauge, leaving equivalent global active relabelings and context swapping.
3. A nonproduct transition: first context has a three-cycle; second has a transposition and a fixed point. No complete 3x2 product factorization exists, because product dynamics duplicate the cycle counts. Predicted complete-chart count zero.
4. Observe only the a=0 transition in each context in case 3. Predicted some spurious charts fit; exact count deliberately not predicted. Inspect one witness and its complete-state counterexample.
5. Exact composition using any chart from case 2: all words of lengths 0 through 6 over the two operations, all six starting states. Predicted zero mismatches against physical table execution. This tests implementation of the equation, not learned generalization.

Budget: one substantive execution, external wall timeout 30 seconds, CPU bound through the coordinator's runner. 720 charts per case, fewer than 1,000 finite words. No third-party imports, network, downloaded code, subprocesses or credentials. Any exception, failed assertion, mismatch or timeout is retained in raw log. Write results incrementally to results.jsonl; JSON roundtrip checked independently before substantive execution. Freeze code and contract SHA256 first. One correction rerun only if an implementation failure is diagnosed; retain both logs and versions if needed.

Controls/limits: full-data nonproduct negative control; partial-data contrast; fixed encoding shuffle; no scientific inference from this single six-state construction. Outputs explicitly name designed factors, teacher rules, exhaustive domain and scope.
