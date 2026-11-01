# Neutral reconstruction task — do not open author artifacts first

A neural computation contains a trainable linear factor block F(A,B,x)=ABx,
A in R^(m*h), B in R^(h*n). Its use inside the rest of a network supplies a
differentiable loss L(AB) and matrix G=dL/d(AB). One simultaneous SGD step is

    A_next = A - eta G B^T,
    B_next = B - eta A^T G.

Both right sides use the OLD factors. G and positive eta may change arbitrarily
between steps; a task sequence can change the loss/data. A nonlinear network
may surround this block, provided its loss depends on the block only through AB.

Question: what information about this block is needed to reproduce its future
execution and learning after compaction? Can a summary with size independent of
the hidden width h suffice? Derive a construction or an obstruction. Do not
assume that equality of the current product is enough or insufficient; determine
it. Compare current-function information, initial gradient-response information
and the state needed under a sequence of later updates.

Also consider appending columns/rows with one side of each new hidden path zero,
so its immediate contribution to AB is zero. Determine what, if anything, this
changes for subsequent learning and how a proposed summary should handle it.

Do actual symbolic/analytic derivation plus one independent bounded executable
check. Choose/freeze your own inputs, numerical method, criteria and seeds before
execution. A different method from the authors is encouraged, but their method
is deliberately not supplied. You may use existing run-local isolated numerical
executor; maximum120s per inspected process, incrementally persist raw results
and check serialization. No meta-training, performance sweep or novelty claim.

Before a complete own specification/code/result/certificate freeze, do NOT open
candidates/, other probes/results, selection/checkpoints/state/events or old
campaign reports. Allowed project inputs are this file, inputs/contract.md and
the executor code only. Initial originals/curated index are not needed. Use a
fresh fork_turns=none context; shared filesystem isolation remains instruction-
only, not data ACL secrecy. Write only candidates/rebuild, probes/rebuild and
sources/rebuild in the current run. Notify root after hashing your artifacts;
only then compare author equations/code/results and preserve any discrepancy.

This asks whether a concrete learning-state mechanism is internally correct,
not whether it is new, automatically discovered, efficient on hardware, or a
foundational AI breakthrough. State those limits explicitly.
