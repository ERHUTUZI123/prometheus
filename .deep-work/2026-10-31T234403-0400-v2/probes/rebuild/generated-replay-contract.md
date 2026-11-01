# Generated-program replay: frozen independent contract

Distinct authorized verification after the completed independent v0 freeze.
Read-only author inputs: candidates/K/candidate-v2.md, v2-technical-receipt.md,
probes/K/check_plasticity.py, matrix-closure.json, scalar-closure.json.
No author artifact is imported as Python or modified. Author input hashes are
recorded in `generated-replay-inputs.sha256`.

The independent interpreter executes the actual emitted `closed_updates`.
For event statistic i, evaluate its monomial list in the ordered source vector
using `powers` and rational `coefficient=[numerator,denominator]`. Obtain the
entire initial state from all events in sequential index order; no target
Gram basis or expected state count is used to select retained state.

Each generated successor coordinate is interpreted as

    sum_over_terms product_j(control_j^control_powers_j)
                   * sum_k(state_coefficients_k * OLD_state_k).

All emitted events and updates are retained. `dimension` is cross-checked
against their lengths after loading, rather than used as a selection target.
Controls are eta*G, in row-major matrix order, as established by inspection of
the author's source-variable/control construction. All arithmetic is Fraction.

Independent source factors, fixed before execution:

    scalar a=[1/2,-2/3,3/5], b=[-5/7,7/11,2/9]
    scalar controls=[1/7,-2/9,0,1/5,-1/11,2/13]

    matrix V=[[1/2,-2/3],[3/5,4/7]]
    matrix U=[[5/11,-1/2],[2/3,7/13]]
    matrix controls, flattened row-major:
    [1/7,0,-2/9,1/5], [0,1/6,1/8,0], [-1/5,2/7,0,-1/9],
    [1/11,-1/13,1/17,1/19], [0,0,0,0], [2/9,-1/4,-2/11,1/3].

Source updates are independent direct simultaneous loops over OLD factors,
with controls already absorbing eta. At each step compare every interpreted
state coordinate exactly with its emitted statistic evaluated on the updated
source factors. Separately compare emitted output-origin coordinates with
the direct scalar dot product or matrix factor product. Require all six steps
in both sources, initial/final source/state roundtrips and a final PASS.

Negative control: deep-copy each JSON program, increase by one the first
nonzero state coefficient in its first closed update, and require its first
interpreted state to differ from the actual direct source statistics for the
first fixed command. This verifies the replay can detect a program error.
No mutation is written into author files.

Before calculations, fsync an OPEN record, roundtrip a rational smoke state
and fsync its PASS. Persist inputs and every step's factors, actual generated
state, independently evaluated reference state and readout incrementally in
`generated-replay-raw.jsonl`. Save a final summary. One inspected process
through the existing network-denied/write-restricted runner with external
timeout 90 seconds. Missing records, mismatch, exception or timeout means
failure. No retry, seed search, tolerance, training or performance claim.

This finite replay checks emitted-program execution, not universal proof of
closure, the correctness of every possible generated closure, discovery of a
useful neural mechanism, novelty, efficiency or scientific significance.
