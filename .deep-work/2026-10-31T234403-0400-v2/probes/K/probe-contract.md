# K reconstruction check, frozen before execution

Purpose: independently check the four-port algebra, complete boundary-preserving gauge move, defect handling and a specific false-compression control. This is a finite correctness check of the reconstructed laws, not evidence for neural acquisition, novelty, speedup or scientific significance.

Inputs: the exact rational tensors and Q specified in frozen candidate-v0. No random data, downloaded code, network or secrets. No baseline performance claim. An exact direct scalar contraction is the independent reference; a separate transformed contraction is compared with it. Exact cut-rank calculation checks the attack that a gauge does not lower unfolding rank.

Checks: JSON roundtrip before substantive calculation; identity residual 0 for A and 1 for its specified perturbation; scalar 156 and 164; gauge exact equality for both; inverse gauge reconstructs every tensor entry; rank invariance; deliberately dropping the exception must yield an incorrect scalar, thus preserving a negative control.

Execution: inspected stdlib code only, through root's vetted isolated runner with external wall/CPU timeout 20 seconds, network denied, environment scrubbed, writes limited to probes/K. Incremental JSON records go to a new exclusive raw log and a final JSON result. Stop on assertion or external timeout. No retries or parameter search planned. At freeze this is unexecuted.
