# Precision corrections from review002 original-code inspection

Append-only correction to hashed execution-001.md; previous record and frozen candidate/code stay unchanged. No rerun was needed.

1. The recorded `compact_symmetric=27` is15 independent symmetric residual coordinates plus12 explicit-path parameters. The Python implementation actually allocates a dense5x5 array for H:25+12=37 scalar array slots. The minimum-factor implementation has15+12=27. Neither count is a hardware-memory/time measurement. Do not report27vs27 as actual allocated memory or as a demonstrated resource advantage.
2. `wrong_fold_output_error` is measured AFTER nonlinear gates are nonzero. It directly demonstrates destruction of PRESENT nonlinear computation by folding it away. It is not a matched-current-function later-learning deletion intervention. Separately, recorded nonzero gate derivatives at zero-gate births show a currently silent trainable direction exists; the argument that dropping its interface can lose future behavior is mathematical. A dedicated matched-current zero-gate discard/relearning experiment was not performed.

These corrections came from the independent supervisor reading actual promotion code, rather than approving the coordinator's summary. The exact source/compact/thin-factor equality and nonlinear construction conclusions remain in their stated scope.
