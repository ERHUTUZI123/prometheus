# K v1: acquire outside the core, retain a learnable compiled family

Status: explicit constructive hypothesis plus mathematical reconstruction. No new neural acquisition effect demonstrated. This version preserves v0 and incorporates its off-manifold-gradient repair.

## Judgment

There is a real, specified acquisition-and-release path: a learner can use a few temporarily expensive operations, construct an entire family of cheap operations, compile that family, and release the exception budget while retaining an editable mechanism. Full semantic derivatives can be computed without incorrectly constraining updates to the cheap family.

The most consequential requirement found is that compilation of a learned mechanism must preserve its **parameterized behavior**, or explicitly retain full semantic update directions. Equality of the current output alone does not preserve what the mechanism can learn next. An exact example below separates those cases.

However, the same deterministic algebraic compiler plus ordinary task learner can implement this entire path. This v1 has not found a distinct acquisition/composition effect beyond that rival. A potentially new organizing principle remains OPEN; the present contribution is a precise construction, a mathematical failure mode of pointwise consolidation, and a narrower research boundary. Existing components do not disqualify an eventual contribution, but renaming their combination does not establish one.

## State and transition law

Retain the finite tensor-program semantics of v0. Each retained mechanism is now

    M = (boundary type, parameter domain T, F(t), compiled C(t), certificate Pi,
         parameter map h, cost/bit bounds, optional uncompiled fallback).

The certificate must establish F(t)=C(h(t)) as a complete boundary tensor for every t in its stated domain. A certificate only at t=t0 licenses inference replacement at that point; it does not license deleting the editable family F. For real parameter coordinates, the family identity implies

    J_F(t) = J_C(h(t)) J_h(t),

wherever differentiable. Thus both present behavior and its declared first-order semantic directions survive. Equality of current tensors alone implies neither equation. The optimizer state must also be translated or restarted honestly; exact function identities do not imply identical SGD/Adam trajectories under arbitrary parameter changes.

The current program contains immutable calls to retained versions, editable general local tensors, a compiler representation, and a finite exception interface budget b_max. A task-learning event performs:

1. Compute the full local semantic gradient by pinned environments, including directions outside the core.
2. A neural policy chooses a sparse gate/coordinate update under the charged interface budget. It may propose a new module rather than change a retained old version. Semantic updates can change the function; compilation cannot.
3. Use the general executor with the resulting explicit exceptions. If the update is useful, seek a connected-region representation with fewer exceptions or smaller exception interface.
4. Search for an exact parameterized realization, not merely a matching current scalar. Retain the parameter map and certificate in the library; replace the implementation and release only the exception cost actually removed.
5. Future tasks may invoke or edit the compiled family. Previously retained versions remain available. This provides versioned functional retention, not a claim of improved future discovery or free archive storage.

The neural policy can use task improvement plus measured future execution savings as its reward. The deterministic compiler is an allowed scaffold. Discovery, verification, retained-module storage, tensor encoding and all task evaluations are charged. No task-gradient signal acts on a pure function-preserving gauge: that gradient is zero.

## Exact full-gradient cost

For one local tensor entry a,

    dZ/dF_v(a) = sum_{s: s|ports(v)=a} product_{u!=v} F_u(s|ports(u)).

Remove v and pin its incident bits. Product unary pins preserve a realizable planar matching core. If the remaining arbitrary exceptions have maximum degree d and at most k vertices, their enumeration is bounded by 2^(dk). A conservative computation of all local entry derivatives costs

    O(N * 2^d * 2^(dk) * poly(N,L)),

where L bounds explicit coefficient bit lengths in a fixed field. A finite batch, output/readout derivatives, the controller Jacobian and task loss add their actual costs. This is a deliberately unshared bound, not a claim that an optimal adjoint implementation is needed or implemented. A dense simultaneous semantic update generally creates N exceptions and forfeits the bound; sparse selection is essential to this particular acquisition path.

For real rational tensors and rational learning rates, a finite gradient step can be represented exactly as rational arithmetic when the task loss and labels are rational polynomial data. General neural controllers require stated numerical tolerances or verified rounding. A frozen local tensor identity does not certify approximate controller outputs for all inputs.

## An explicit release path, not just a gauge cancellation

Use the Majorana algebra as an illustrative operator representation on n>=3 modes:

    gamma_i^*=gamma_i; gamma_i^2=I; gamma_i gamma_j=-gamma_j gamma_i (i!=j).

This example is a hand-constructed mechanism, not a task-discovered result. Gaussian operations are generated by quadratic Majorana products and admit the known polynomial classical representation. Quartic gates below are temporary non-Gaussian exceptions. The classical simulator, not a quantum device, executes the program.

Let

    P=gamma_1 gamma_2 gamma_3 gamma_4,
    Q=gamma_1 gamma_2 gamma_3 gamma_5,
    R=gamma_4 gamma_5.

Then P and Q are Hermitian involutions, PQ=R, QP=-R, and R^2=-I. Set

    A=(I+iP)/sqrt(2),
    c(t)=(1-t^2)/(1+t^2),  s(t)=2t/(1+t^2),
    B(t)=c(t)I+i s(t)Q,   t in Q.

Since A Q A^*=iPQ,

    A B(t) A^* = c(t)I - s(t)R =: G_45(t).

The last expression is a quadratic-generator Gaussian rotation. An entire editable family built with three generally non-Gaussian gates therefore has an exact cheap realization. All signs follow from the stated anticommutation algebra; no empirical fitting or matching on examples is involved. The only field extension is a single fixed sqrt(2), which cancels in the final expression. c and s are rational functions with positive denominator 1+t^2. A symbolic certificate can clear denominators and compare coefficients of I and R; it need not evaluate exponentially large matrices.

At t=0 the current operator is I. A pointwise compiler could replace the three-gate region with a constant identity and delete its parameter. That preserves the current function but removes

    dG_45/dt at t=0 = -2R.

A parameterized compiler retains G_45(t), with derivative

    c'(t) I - s'(t)R,
    c'(t)=-4t/(1+t^2)^2,
    s'(t)=2(1-t^2)/(1+t^2)^2.

For a simple normalized operator target G_45(t_*), the loss is

    ell(t) = ||G_45(t)-G_45(t_*)||_F^2 / 2^n
           = 2 - 2[c(t)c(t_*) + s(t)s(t_*)].

With t_*=1/2, ell'(0)=-16/5. The editable compiled family has a nonzero task-learning direction at the same present function for which the constant identity has none. The task target and circuit here are supplied only to exhibit the distinction; this is not a discovered new capability or a fair learning benchmark.

For repeated acquisition, substitute an analogous shared triple of Majoranas and two distinct endpoint modes to form another Gaussian family G_ij. Each new family can be built with at most three temporary exceptions, certified, and retained. Composing the retained Gaussian families remains Gaussian, and does not accumulate those original non-Gaussian gate implementations. Repeated calls on the same plane also compose through the rational circle law; retain (c,s) as the chart representation to avoid the singular coordinate t=(t1+t2)/(1-t1*t2) when its denominator vanishes. This gives an actual sequence of construct -> compile family -> release -> compose events with a rolling exception budget. It establishes possibility of that path, not that neural task feedback will find it or need it.

## What can and cannot be reabsorbed

A genuinely outside-core complete boundary function, with the boundary contract fixed and outside the allowed gauge/realization orbit, cannot be replaced exactly by a core function. The example above evades no such restriction: its intermediate implementation is outside the core, but the final complete family is inside it.

For genuinely new outside-core boundary behavior, exact reorganization can only remove redundant implementation hardness, concentrate it, lower its interface, or move to a larger execution family. Arbitrary nonlinear function acquisition does not imply eventual disappearance of exceptions. Restricted-context specialization is another possible route, but would require explicit context guards/fallbacks and their verification costs; it is not silently assumed in this version.

## Bit complexity and conditioning

The finite tensor calculation can use rational arithmetic or a single fixed algebraic extension. Allowing each learned basis an unrelated algebraic extension can make field degrees grow exponentially and is excluded from the stated polynomial bound. Bounded-bit rational orthogonal port transformations avoid arbitrary gauge scaling. Their tensor unfoldings retain exactly the same ranks.

The Cayley formulas above make every finite t exact. Large |t| may make t an inconvenient optimizer coordinate near the missing chart point; storing a rational unit-circle pair and changing charts avoids interpreting that as a new computational resource. The family certificate is symbolic. A numerical residual below a tolerance cannot substitute for it. Task-learning roundoff remains a separate explicit issue.

## Strongest rival, and where the attempted distinction disappears

Rival D receives exactly the same semantic tensors, initial gate grammar, task learner, full pinning gradients, exception budget, family-equivalence checker, algebraic rewrite rules, costs and storage. It runs a deterministic compiler after the same candidate constructions and retains the same parameterized families.

D compiles A B(t) A^* into G_45(t), releases the same budget, obtains the same derivative, and composes the same retained modules. No special neural compiler property is required. Giving only K a retained family while D gets a constant snapshot would compare different compilation contracts. Giving D direct access to every useful Gaussian primitive may further strengthen it; it must be included when assessing a claimed search advantage rather than prohibited as “too symbolic.”

Possible remaining distinctions are amortized discovery of useful families, compiler-informed choice of new semantic constructions, or the acquisition of a useful execution algebra not already in the supplied recognizer. Those are specific open hypotheses, not results of the construction. A generic neural proposer with cost reward does not yet make any one of them a distinct mechanism. The full-gradient derivation is ordinary AD through a tractable-core/exception simulator.

## Success counterfactual and present decision

If neural construction plus learned editable family retention repeatedly finds capabilities or useful compositions at lower total acquisition cost than D, while compiler/core knowledge and task information are matched, this would bear directly on the Fundamental Question. It would show a learned way to create future editable computational primitives rather than merely fit current outputs. That possible contribution does not require every evaluator or algebraic primitive to emerge.

The hardest missing step is a concrete task family where compiler-guided semantic construction and retained editable mechanisms change finite-budget acquisition/composition after giving D the same compiler and learner. No such discriminator has survived this constructive pass. The demonstrated release example is covered by deterministic compilation; a neural training run on it would not change that judgment. Therefore no additional numerical “decisive” test was run on this manufactured example.

Current decision: preserve the off-manifold gradient and parameterized-retention constructions as checked research ideas/explicit proof obligations; retain the scientific candidate as OPEN, with no established novelty or acquisition advantage. Do not expand K into more basis recovery demos. A future continuation requires one concrete remaining acquisition mechanism or phenomenon, not just a larger simulation of the present laws.
