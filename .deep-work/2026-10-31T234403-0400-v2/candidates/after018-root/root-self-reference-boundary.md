# A restricted algebraic boundary of self-generated weight updates

RECONSTRUCTION by root from the checked SRWM Eq5–8; support for new candidate comparison, not a MAIN proposal or a learning impossibility claim. No code or experiment.

Let W in R^(p by d), a=phi(q), b=phi(k) be probability vectors and 0<beta<1. The displayed shared-rate update is

    W+ = W + beta (Wa-Wb)b^T = W M,
    M = I + beta(a-b)b^T.

Since 1^T(a-b)=0, 1^T M=1^T. The matrix determinant lemma gives

    det M=1+beta b^T(a-b) >= 1-beta ||b||² >= 1-beta >0.

Thus M is invertible and both M and M^-1 preserve column sums. Columns of W+ are affine combinations of columns of W; the converse holds too. The affine hull of the columns is exactly unchanged. The column space/rank is also unchanged. Entries of M can be negative, so convex hull and norm bounds do **not** follow.

The paper's practical four-rate variant applies a different M_j to each block W_j. The argument survives separately for each block. A single common right factor for the whole stacked matrix need not exist, so its joint column-space invariance must not be asserted. Multi-heads and subsequent nonlinear/residual layers require their own analysis.

A finite illustrative separation for a **pure block** is W0 with columns e1,e2 in R³. Every self-generated update above stays on their affine line; no probability query yields e3. An ordinary delta writer receiving an external value e3 can leave that line. This comparison identifies which information enters the write, not a learning advantage for the external writer.

The restriction may be vacuous when initial columns already affinely span the required output space. Even codimension-one logit constraints can be immaterial after a softmax quotient; a nonlinear readout/residual also changes the question. New observations still influence a,b and hence the coefficients: invariant hull is **not** absence of new information or adaptation. An expanded initial basis can remove the illustrated obstruction. Therefore this is no refutation of SRWM, recursion, fast-weight learning or the Fundamental Question.

Constructive lesson: a claim of creating a new operation must specify the actual update and source of new task information, rather than infer it from self-reference. Adding external value writes is already a standard fast-weight mechanism, not a new principle here. No toy numerical confirmation of these algebraic identities is warranted as foundational evidence.
