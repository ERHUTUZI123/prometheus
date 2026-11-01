# K consequences and self-attack after v0 freeze

No cross-branch or historical research input has been opened. This supplements the immutable first candidate; it does not rewrite its provenance.

## A necessary constructive repair: differentiate the semantics, not a pointwise replacement

RECONSTRUCTION. Suppose a tensor at the current parameters satisfies a matchgate relation, and the compiler replaces it exactly. That equality at one parameter point does not justify differentiating an executor that silently constrains future tensors to the same matchgate chart. It gives only tangent derivatives if the chart parameterization is differentiated. A task-relevant direction outside the chart can disappear.

For the actual multilinear semantic computation, however,

    dZ/dF_v(a) = sum_{s : s|ports(v)=a} product_{u != v} F_u(s|ports(u)).

This is a pinned environment contraction: remove v and fix its incident edge bits. The derivative tensor insertion is the basis tensor e_a, a product of unary pinning tensors. Hence if the other gates are a realizable planar matchgate core plus k arbitrary exceptions, it is computed by the same pinned-core evaluator with no hidden dense global tensor. A conservative all-entry bound is

    O(N * 2^d * 2^(dk) * poly(N, bitlength))

for fixed maximum degree d, charging the exception enumeration anew for each derivative. This can probably be improved by sharing environments, but no such implementation or bound is claimed. This is ordinary multilinearity and pinning closure, not a new theorem about matchgates.

The derivative can point outside the core. In the frozen four-port example,

    Z(A + epsilon*e_1111, A) = 156 + 8*epsilon.

The exact full semantic derivative is 8 at epsilon=0, although a compiler that simply keeps the original core replacement would report no change. Updating only that tensor entry by a rational step gives one explicit arbitrary exception. Its execution is still exact and bounded by the new exception interface. A full unconstrained gradient update at every gate would generally create N exceptions and destroy the useful bound; that is a substantive limitation.

This supplies a concrete acquisition transition missing from the initial sketch: compute full local semantic derivatives through pinning, let a neural policy select a sparse local semantic update, retain the resulting exception, and search for exact region recompilation that later frees the exception budget. It permits an acquired operation outside the existing tractable manifold. It does not guarantee that new capabilities will ever become absorbable or that the policy discovers useful updates.

## What the scalar sanity check actually established

VERIFIED (finite exact arithmetic only). `probes/K/check_exact.py` ran once under the vetted runner with external wall/CPU timeout 20 seconds, network denied and scrubbed environment; exit code 0. JSON roundtrip completed before calculation. Raw incremental records are in `probes/K/exact-raw.log`.

The direct and gauge-transformed core scalar both equal 156. The perturbed scalar equals 164 in both representations. Dropping the exception incorrectly returns 156 and is detected. Inverse transformation reconstructs every entry. The balanced unfolding rank remains 4, while the entry count changes from 8 to 16. Thus visible density/rank alone does not diagnose the available algebraic executor. No Pfaffian evaluator, neural learner, compiler search or acquisition benchmark was implemented. These checks establish neither an asymptotic advantage nor the scientific hypothesis.

## Strong attacks that survive the constructive repair

1. **Task gradient cannot teach a pure gauge.** For an exact coordinated gauge move, Z(x) is constant for all B, so dL_task/dB=0 wherever differentiable. The compiler needs its own cost/identity signal. A claim of ordinary task-gradient emergence of the basis would be false under these laws.
2. **Exact membership is a thin target.** An exact algebraic identity is typically satisfied on a lower-dimensional set. A continuous residual heuristic alone neither reaches it exactly nor certifies a cheaper executor. Parameterizing a known tractable manifold plus a residual solves this operational issue but supplies the algebra as scaffold. Rational reconstruction is not guaranteed for arbitrary algebraic bases.
3. **A known algebra is allowed, but does not settle contribution.** The constitution permits fixed primitives and evaluators. A meaningful result could be a new acquisition or retention advantage inside this scaffold. It need not invent every algebraic law. The required distinction is a demonstrated change in learnability/resources/organization against the strongest rival, not an arbitrary purity test.
4. **The deterministic rival is very strong.** Give it the same core, pinning gradients, exception budget, exact signatures, rewrites, input examples and compute. Neural proposal amortization might help, but that alone could remain a routine learned compiler result. A claimed difference must be operational, not “neural versus symbolic” vocabulary.
5. **Compiled gradients are not free exits forever.** One sparse step is tractable; many independent semantic updates accumulate exceptions. Whether region compilation repeatedly restores capacity while preserving useful acquired behavior is the central OPEN mechanism question.
6. **Bit complexity matters.** Independent arbitrary algebraic bases can create large extension fields. A first implementation should use bounded-bit rational port transforms, e.g. Q(t)=(1+t^2)^(-1)[[1-t^2,-2t],[2t,1-t^2]] for rational t, and a single fixed extension field for any realization constants. Unit-cost complex arithmetic cannot be silently assumed.
7. **Gauge freedom does not change cut rank.** For every unfolding, transformed matrix = invertible left Kronecker factor * original matrix * invertible right Kronecker factor. Rank is preserved. Any gain must use algebraic identities/cancellation, not a fictitious rank reduction.
8. **No broad NN conversion claim.** The starting system is a neural-controlled finite-arity tensor program. Arbitrary continuous activations or large nonlinear gates need an explicit representation with its full cost. Calling it a general NN compiler would overextend v0.

## Decision interface

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION

Exact port re-expression and local defect calculation | VERIFIED in one finite example; algebraic reconstruction stated | frozen v0, exact raw log and code | unimplemented scalable evaluator/certificate implementation | exception-dropping control fails as expected | no additional identity tests needed.

Sparse semantic learning can exit an exactly compiled core with computable full derivatives | RECONSTRUCTION, proof obligation explicit above | multilinearity/pinning derivation | useful multi-step acquisition may accumulate extensive exceptions | no empirical learning run | independently assess the derivative and sparse-update construction.

An adaptive algebra/interface process enables consequential acquisition and later composition beyond a fixed core plus deterministic compiler | OPEN | candidate-v0 plus present consequences | no discriminating phenomenon or task acquisition evidence yet | simple hidden-basis recovery rejected as sufficient evidence | determine whether a concrete acquisition/absorption distinction survives matched-rival analysis; otherwise stop K as a reconstruction.
