# Root derivation: retaining forward information is not the same as retaining its future credit

RECONSTRUCTION / AGENT PROPOSAL. Same active V2; no sealed stage007 artifact changed. This is a new causal clarification for the physical-frontier construction, not a claim of a new gradient identity, an executed experiment or a necessary exact-trajectory objective for every learning system.

## 1. A complete cache can still fail a later gradient request

Let a private neural computation return h=f_theta(x) in R^n with Jacobian J=partial h/partial theta in R^(n by p). An interface forwards z=Wh and retains the exact current learning records

    partial z/partial W = h,
    partial z/partial theta = WJ.

Therefore the original h may be recoverable from the encoder's derivative cache. It would be false to call it erased solely because the private activation allocation was freed. This retained information and its legal access must be included in the cost and given to strong controls.

Yet even retaining ALL of h and WJ does not generally permit a new upstream gradient lambda^T J. For an otherwise unknown J, recovery from WJ for every J is possible exactly when lambda is in the row span of W. If lambda^T=a^T W, use a^T(WJ). Otherwise choose n0 in ker(W) with lambda^T n0 nonzero and DeltaJ=n0 b^T. Then WDeltaJ=0 while lambda^TDeltaJ is nonzero. Identical stored h,WJ can require different answers. This is a finite linear information statement conditional on the declared cache, not a universal impossibility result for retained source programs/inputs/replay.

## 2. Same program, same parameter version, different discarded instances

One program suffices; no ambiguity about a hidden source implementation is needed:

    f_theta(x) = (theta_1, theta_2 x),  theta=(0,0).

For past input x1 versus x2, the complete forward value is h=(0,0) in both cases. Let W=(1,0). The port z0, encoder derivative h, and producer derivative WJ=(1,0) are identical. The program and theta are known. If x is not retained anywhere else, these records cannot identify it.

Use the native downstream consumer

    q_w(h)=h_1+w+w²h_2,  w0=0.

On a legitimate squared-loss training observation with target1 and SGD rate1/2, both instances and both current-gradient-preserving representations update w to1/2. Their next prediction remains1/2 even if the consumer reads the entire recovered h. On a new external query target0, the consumer's state cotangent is (1/2,1/8). Correct native producer gradients are

    (1/2, x1/8) versus (1/2, x2/8).

Thus ALL current forward values, current derivatives, and the first downstream learning step matched; the later producer credit did not. The cache must retain x, J's missing row, a source restart/input permitting real recomputation, or an equivalent information-bearing state to answer exactly. Adding a forward copy of h_2=0 by itself is not enough unless its derivative also crosses the boundary. A cache-readable control can repair some forward omissions without solving this credit request.

The example has no resource advantage: raw x is the cheapest obvious sufficient record, and full sparse J is tiny. Its purpose is to identify the actual formation problem rather than manufacture a benchmark victory. If x survives in another cache or audit input channel, that information must be declared and the alleged ambiguity disappears.

## 3. A prospective explicit credit interface

A module could retain its forward object h and a learned finite future-cotangent basis B in R^(n by r), together with

    G=B^T J.

For a later actual request lambda, test its residual from col(B). If lambda=B a, return the exact versioned producer gradient G^T a. If not, report unsupported credit rather than hallucinating a missing J row. To repair it, use genuinely retained/replayed inputs or the next honestly paid full probation call; an already erased instance cannot be retroactively differentiated. Unbiased/random approximate alternatives are possible but must state their variance and different guarantee.

The basis, parameter read sets and return location can be acquired from ACTUAL finite native-learning continuations, rather than only current task cotangents. Capped full probation provides the information. It is a possible operative refinement of the return-formation rule, not yet an advantage: AD preaccumulation, Jacobian sketches, goal-oriented adjoint reduction, synthetic-gradient and RTRL methods are close actual controls. CoBRAS on the appropriately augmented learning process can already use future sensitivities; a learned matrix alone is not a new principle.

A strict contract might intentionally restrict future consumers to functions whose adjoints lie in B, or might accept incomplete/approximate old-instance credit and optimize actual later task acquisition instead. Those are different learning systems. Exact matching of every full-state learning trajectory is not required by the scientific goal; incorrect claims of exact matching are prohibited. External performance after the declared learning rule remains the primary criterion.

## 4. Concrete organization pressure, beyond a forward-cost count

For every live executed value v, track actual parameter-owner support S(v). Parent support union and parameter reads propagate it conservatively. A sparse forward-credit representation has proxy storage sum_live |S(v)|, but actual factorization, metadata, raw replay inputs, optimizer/policy state and all retained gradients are charged.

Two sparse recurrent graphs can have the same N states, P=Nd weights and forward arithmetic: one globally strongly connected ring, versus s disconnected equal-size rings. Under generic causal propagation, possible influence entries grow to NP versus NP/s. This is NOT matched task capacity or an architectural advantage; it only shows that forward FLOPs/width cannot substitute for learning-state cost. Calls/returns can replace a lasting private frontier by a smaller actual interface without permanently freezing its parameter owners. A factored AD control can reproduce that algebra; acquiring a useful source/return organization is the scientific hypothesis.

## 5. The next probe must measure formation, not repeat these equalities

The proposed pilot should use real neural private producers, once-materialized ports, actual finite consumer adaptation, actual changed return/cache objects and complete cost records. Current-only, OML with cut search, learning-continuation adjoint reduction, replay/checkpoint and cache-readable controls share the information/constraints. A formation-evidence reassignment should preserve native labels and budgets while changing which interface receives the learned record.

Positive results would support the specific organization in its measured scope; they would not establish historical novelty or foundational importance. If savings merely migrate into dense encoder/credit caches or scouts, or a simple existing rule explains the whole fixed-cut result, reconstruct the mechanism rather than enlarge the same loss experiment.
