# Root original reading: targeted memory update repair

SOURCE: Yang, Kautz, Hatamizadeh, Gated Delta Networks, arXiv:2412.06464v3 (6 March2025), https://arxiv.org/html/2412.06464v3 . Root actually entered §§2.2 and3.1, equations3–10 and their immediate mechanism text through arXiv HTML. OpenReview PDF returned a browser challenge; it was not treated as read. No claim of reading the complete paper or appendix.

The original delta recurrence targets an association error rather than independently replacing every memory coordinate. Equation10 adds learned scalar decay to the generalized rank-one transition. The original chunkwise algorithm relies on its stated linear recurrence and representations; it does not establish parallel execution for our history-dependent keys, gates, nonlinear primitives or caller.

RECONSTRUCTION: for current slot matrix C=[K,V], original update is C'=C+g diag(a)(1 ĉᵀ−C). A targeted candidate is C'=C+g a(ĉ−aᵀC). Under uniform a, both have the same slot-mean update, but only the original shrinks every slot contrast. Under one-hot a, they agree. This is a proposed local use of an existing delta principle, not an originality or superiority claim.

OPEN: actual VALID traces must determine whether diffuse writes and slot-contrast loss are relevant. Changed forward semantics may hurt old tasks at step0. Norm growth, feedback, learning dynamics and resource costs require execution; no full-Jacobian protection or linear-time parallel-training claim follows.
