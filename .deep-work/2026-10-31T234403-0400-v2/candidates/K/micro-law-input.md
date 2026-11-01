# Neutral source-program reconstruction task

This packet contains source laws only. Derive any useful reduced learning state independently. Do not read the originating candidate, its implementation, expected dimensions, other solutions or raw outputs before freezing your reconstruction.

Use real finite matrices U in R^(h x n), V in R^(m x h). The observable module is

    y=V U x, x in R^n.

The environment may place the module inside any differentiable surrounding program that accesses the module through this input/output interface. At each learning event it supplies a matrix G in R^(m x n). In ordinary task training G is the derivative of the task loss with respect to the effective matrix VU. For analysis, allow any finite sequence of supplied G matrices and step sizes eta.

One simultaneous source SGD event is

    V_new=V-eta G U^T,
    U_new=U-eta V^T G,

with both right sides evaluated at the old state. Source hidden factors are not otherwise exposed to the environment.

Questions: Can this source module be replaced by a smaller executable state that preserves every finite future output/learning trajectory under the stated interface? What current-function information is insufficient? State dimensions, constraints, arithmetic costs and the equivalence relation explicitly. If deriving sufficient statistics, derive them from the source rather than assuming an expected form. Distinguish exact reduction from a local first-order approximation and from changing the learning algorithm.

Optional specified source extensions, only after freezing the basic derivation:

1. Function-preserving physical growth appends hidden paths with new outgoing vector v and zero incoming row, or zero outgoing vector and a new incoming row. Count construction costs and any norm/trace budget. Determine whether and how the reduced learning state can represent this action.
2. Momentum source state also contains P_V and P_U, of the same shapes as V and U. Given mu,

       P_V_new=mu P_V + G U^T,
       P_U_new=mu P_U + V^T G,
       V_new=V-eta P_V_new,
       U_new=U-eta P_U_new.

   All gradient right sides use the old factors. Derive what optimizer memory a faithful replacement requires.

Implementations, if authorized by the coordinator, must be independently written and bounded. A proof or a failed reduction is acceptable. This packet supplies no macro-state answer, expected dimension, numerical witness, expected curve or preferred reduction method.
