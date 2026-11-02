# Key-composition dual memory: independent CPU admission review

**Judgment: the single candidate passed its bounded CPU equation, gradient and reload admission. It remains untrained, GPU-unadmitted, and unverified for query binding or capability. Finite derivatives do not establish stable learning.** No optimizer was instantiated, no SGD occurred, and the completed three-arm training allocation remains 7,200 updates. No ridge/configuration change is selected by this review.

The candidate preserves the 20 AB parameter tensors (10,351P) and replaces episode coverage z[8] by C[8,8]. The matched updates C_j+=(w_j(k−C_j)) and V_j+=(w_j(h−V_j)) preserve V=CB when all historical values satisfy h_i=k_i^T B. With A=C/max(C1,epsilon), the ridge solve gives b=(AA^T+lambda I)^−1 A k and r=k^T T_lambda B, where T_lambda=A^T(AA^T+lambda I)^−1 A. Its singular-direction gains are sigma²/(sigma²+lambda). A query in the observed row space is recovered as lambda→0 under that consistency premise. Identical keys with conflicting values, overwritten history, rank deficiency and ridge suppression of weak contrasts remain real limitations. This is ordinary regularized linear associative retrieval, not an originality or learning claim.

For arbitrary A, ||b||≤||k||/(2 sqrt(lambda)) follows from sigma/(sigma²+lambda)≤1/(2 sqrt(lambda)). At lambda=1e-4 the multiplier is 50. For nonnegative C whose normalized rows have mass at most one, ||A||_F²≤8, yielding Gram condition bound ≤80001. These finite bounds neither make the bound tight nor bound derivatives of the whole recurrent NN.

## Actual evidence and independent verification

ROOT executed `parameter-typed-dual-memory-cpu-v1`, using exactly the completed AB checkpoint and newly zeroed episode C/V. The CPU checker compares an overlapping, independent two-key fixture with ordinary SVD ridge least squares, including the nonzero ridge bias, normal-equation residual, rank, conditioning, signed coefficients and norm bound. It preserves a separate identical-key/incompatible-value counterexample and empty-memory zero read. One synthetic B64/L16 eager/masked full-NN comparison checks all 20 parameter gradients, masking, repeated buffer zeroing, strict new checkpoints and theta-only restoration. These are implementation checks, not capability/accuracy gates.

Actual overlapping-key read errors are .000119734/.000126235, matching predicted ridge biases. Normal-equation residuals are about 1.5e-12; condition number is 11061.27. Dual coefficient norms .6074/.6193 are below the applicable 23.919 bound, with a negative coefficient −.109217. These facts demonstrate the stated synthetic algebra, not useful learning on the research tasks.

The reviewer independently parsed all saved numeric `.pt` artifacts using an explicit allowlist and inert Python tensor records, without importing Torch/NumPy, executing a model, deserializing a model class, SSH or training. Raw artifact/source hashes match; all 20 eager/masked gradient pairs are finite and satisfy the unchanged absolute/relative tolerances, maximum gradient difference 2.6226e-6. The saved CPU theta hash matches its receipt, and C is episode state while the 256 B identity is absent from theta. Equality with OWN AB theta is additionally checked by the executed, inspected strict-restore source; the reviewer did not execute another NN restoration.

An independent standard-library pivoted Gaussian solve reconstructs the synthetic ridge read within 8.33e-17. Its effective ridge scalar records the FP32 identity multiplication before promotion in the double-precision fixture, rather than silently presuming bitwise equality with a real-valued constant. Full-NN cached output difference is 1.9073e-6 and NLL difference zero. None of these checks imply GPU capture compatibility.

Reproducible saved-array verifier: [`verify_typed_dual_memory_cpu_saved_coverage.py`](../../probes/topology-architecture/verify_typed_dual_memory_cpu_saved_coverage.py). Proof: [`typed-dual-memory-cpu-independent-coverage-v1.json`](../../probes/topology-architecture/typed-dual-memory-cpu-independent-coverage-v1.json), SHA `f90e34db64570e17a67cfa14605126a0aa6f54ad7a5750290302eb69372faff7`. Actual CPU result and four raw artifacts remain in `probes/topology-architecture/runs/parameter-typed-dual-memory-cpu-v1/check/`.

## Large partial derivative and the coupled-write distinction

The deliberately arbitrary tiny-state fixture is retained unchanged: C entries 1e-10, V entries around 1e-3. Its C-gradient norm is **333,244,768.5**, finite and large. It is not evidence of stable normalization. For any history formed by these nonnegative writes, if ||h_i||≤H throughout that history, the recurrence implies ||V_j||≤H z_j. This fixture would require H≥6,647,012.37 to satisfy that norm inequality. No global bound on the actual model's h is established, so this observation is a conditional reachability comparison, not proof that such a state is universally unreachable. Independent C/V variations also need not follow the coupled formation path.

ROOT's targeted one-write calculation is correct. Hold h and nonnegative simplex k fixed, use the same query q=k, and write once from zero with w_j=beta k_j. Define d_j=w_j/max(w_j,epsilon), s=||d||²||k||². Then A=d k^T, Vbar=d h^T and the rank-one ridge identity gives r=f h with f=s/(s+lambda). For beta=sigmoid(g), away clamp kinks,

\[
\frac{ds}{dg}=2\|k\|^2(1-\beta)\sum_{j\;unsaturated}d_j^2
\le2s(1-\beta),\qquad
0\le\frac{df}{dg}\le2f(1-f)(1-\beta)\le\frac12.
\]

Thus ||partial r/partial g||≤||h||/2 for this fixed single-write fixture. The same beta derivative holds for a seven-way STORE logit with other logits fixed. This explains why a huge unconstrained partial C derivative does not alone imply a huge upstream write-gate derivative along that consistent path. It establishes no multikey, coupled h/k/caller, recurrent-Jacobian, convergence or general training-stability guarantee. The large fixture and the unresolved stability debt both remain evidence.

## Resource and next-decision boundary

The actual admission paid .278133 science-wall/.278037 CPU seconds and 5.865543 local-cold seconds, peak RSS 557,817,856 B. Model parameter payload is 41,404 B. Episode carry rises from 328 to 384 FP32 scalars: +56 values/+224 B, total 1,536 B/example. The shared nonpersistent identity adds 256 B. The historical checkpoint is 516,787 B and contains 82,888 B of optimizer tensors, but no optimizer is instantiated or inherited for updates. The independent saved-array check adds .02070 wall seconds; review/agent compute is not fully measured.

C storage/update is O(S²); Gram formation and a dense S×S solve occur on every token, with nominal O(S³) factorization work plus normalization, value combination and backward derivatives. Solver workspace, library caches, complete forward/backward/training FLOPs, provider-effective compute, energy and GPU launch/capture cost remain unknown. The reported forward-product proxy explicitly excludes solver internals and is not a complete efficiency measurement. Keeping P fixed does not remove this resource debt.

`CLAIM | STATUS | EVIDENCE | WEAKEST LINK | NEXT SINGLE DECISION`: the fixed dual rule has an executable, differentiable CPU implementation of its stated algebra | passed in the bounded synthetic sense only | original raw CPU artifacts, independent storage/gradient/Gaussian checks and conditional proofs above | acquisition of separated references, multikey recurrent stability and complete-resource behavior remain untested | retain CPU admission as implementation evidence; do not count it as binding, a trained arm, GPU admission, Primary freeze or permission for more training. The campaign continues within existing human authorization.
