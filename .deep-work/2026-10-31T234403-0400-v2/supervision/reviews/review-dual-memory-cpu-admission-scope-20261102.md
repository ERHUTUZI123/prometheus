# Dual-memory CPU admission: passed equations, retained numerical risk

**Judgment:** the single dual-memory rule is implemented and passed its bounded CPU admission. It has not been trained, evaluated for task capability, admitted on GPU, or shown stable over learning. The previous 7,200-update budget remains complete; this check used no optimizer or SGD.

ROOT executed `parameter-typed-dual-memory-cpu-v1`. The actual result is bound by SHA `ec6ac98cbf62480645f50e44f0d4dbc7a4c7897afcbed56263292c7ce95dc970`. Its core is `1374fd3ca62f64faec0daf6982a632f1ff8cd447e3c11d43a39aaacf3e712c6c`; the checker is `56c9b8656a8428860f7303f1f6f029bc64368a1e57c02cfb7ce37cc212e22aac`.

All 20 AB parameter tensors were restored exactly from the OWN checkpoint. Episodes start with zero C and V; no old coverage state is reinterpreted as key composition. P remains 10,351. Carry is 384 FP32 values / 1,536 B per example, plus a shared nonpersistent 256 B identity buffer. New schema/reload, C/V update equations, identical-key counterexample, empty read and finite read-input derivatives passed. B64L16 ragged eager/masked forward/backward and repeat agreed for all 20 gradients: maximum output difference 1.90735e-6 and maximum parameter-gradient difference 2.62260e-6. These are finite-test findings, not training-stability guarantees.

The independent SVD ridge reference agreed with the executed Gram solve. In the consistent two-key fixture, read error norms 0.000119734 / 0.000126235 match predicted ridge bias; solve residual norms are approximately 1.55e-12. Condition is 11,061.27, and minimum dual coefficient is −0.109217. This verifies an actual signed read that escapes a convex-reader restriction, within the fixture's assumptions. It does not establish learned address binding.

## The large derivative is real evidence

The `tiny` fixture has C entries 1e-10, V entries on a −1e-3 to 1e-3 ramp, and reports

\[
\|\nabla_C L_{\rm fixture}\|=333{,}244{,}768.5012.
\]

Here the declared diagnostic objective is

\[
L_{\rm fixture}=\operatorname{mean}(r^2)
+0.01\operatorname{mean}(b^2)+0.001\operatorname{mean}(z^2),
\]

not terminal training NLL, a full Jacobian norm, or a measured learning trajectory. The derivative is finite but large and must be retained as a numerical sensitivity warning. No ridge, test tolerance, clipping rule or fixture was changed to hide it.

For a physically formed history with a *specified* finite bound \(H=\max_i\|h_i\|\), the shared positive update weights imply

\[
\|V_j\|\le z_jH.
\]

The tiny fixture has \(z_j=8\times10^{-10}\) while V is around 1e-3, so it lies outside a moderate-H consistency domain. It is **not established to be globally unreachable**: the signed reader can amplify h, and no global recurrent h bound has been proved. Nor is it established that all coupled parameter gradients cancel this partial derivative. Even ordinary formed histories can be poorly conditioned.

## A narrow coupled-path fact

ROOT derived, and I reconstructed, the following single-write result. It applies only to the direct formed-memory path with fixed key k, fixed written h, and a later identical query k. From zero state,

\[
d_j=\frac{\beta k_j}{\max(\beta k_j,\epsilon)},\quad
A=d k^\top,\quad \bar V=d h^\top,\quad
r=f h,\quad
f=\frac{s}{s+\lambda},\quad s=\|d\|^2\|k\|^2.
\]

For a STORE one-versus-rest logit g with \(\beta=\sigma(g)\), the fully sub-floor case gives

\[
\frac{df}{dg}=2f(1-f)(1-\beta)\le\tfrac12.
\]

Mixed clamped/unclamped entries satisfy the same upper bound because their saturated terms add no derivative. Thus this direct coupled scalar path does not inherit the arbitrary partial-C fixture's magnitude. This is **not** a bound on the full caller gradient: changing shared parameters also changes other action probabilities, keys, hidden values and future states. Multi-key histories, address collisions, recurrent signed feedback and long-sequence backpropagation remain unverified.

The general dual coefficient bound \(\|b\|\le50\|q\|\) at ridge 1e-4 bounds coefficient magnitude, not its derivatives or the recurrent system's gain. Likewise, the conditional Gram condition bound of 80,001 for row-mass-bounded nonnegative C is not a stable-training certificate.

## Costs and next technical boundary

Actual `result.costs.wall_seconds` is **0.27813325508031994 s**, CPU **0.278036953 s**; the separate cold coordinator wall is **5.865543166990392 s**. Peak RSS is **557,817,856 B**. NN parameters occupy 41,404 B; the historical AB optimizer payload loaded with the OWN checkpoint is 82,888 B, but no optimizer was instantiated and live optimizer bytes are zero. The complete checkpoint is 516,787 B. Import/startup time is represented by the cold clock, not silently included in the smaller science clock.

Dense-solve forward/backward FLOPs, factorization workspace/cache and GPU behavior remain unknown. `torch.linalg.solve` may synchronize or impose capture/runtime costs; no GPU Graph or learning claim follows from this CPU result. The next minimum technical uncertainty is conditioning and upstream gradients on multi-key **formed** histories, together with actual GPU solve implementation/cost, if separately authorized. No additional test, NN run, training arm or regularization variant is introduced by this assessment.
