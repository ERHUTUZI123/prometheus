# Candidate: preserve key composition and retrieve with a learned dual basis

**Status: AGENT PROPOSAL, unimplemented and unexecuted.** This is the single next constructive candidate from the completed 7,200-update segment. It authorizes no additional NN run, optimizer step, configuration study or Primary freeze. It uses ordinary linear associative memory and ridge/dual-basis retrieval ideas; no novelty is claimed.

The specific gap is retrieval contrast. For the requested first-written operand, the observed original memory's literal contribution is at most 0.1883 on average over *any* positive attention vector, versus actual 0.1723. Holding the model's own KEY address raises that upper bound to 0.4827, but actual retrieval reaches only 0.2410. These are history-contribution bounds, not limits on the nonlinear caller, hidden state or decoder. A new head or more arithmetic capacity does not address that particular averaging operator.

## One state/update change

Keep the AB learned parameters, input-only address proposal, input-history controller, four learned compute operators, G8, caller, hidden-state update, terminal NLL and learned STORE/ADDRESS decisions. Replace coverage `z[8]` by a **key-composition matrix** `C[8,8]`; retain the existing value matrix `V[8,32]` and address logits `l[8]`. Reset C and V to zero at episode start. C is ephemeral executed state, not a learned parameter or a supplied key label.

At each step let the model's own current key/query be

\[
k=\operatorname{softmax}(l),\qquad \eta=\alpha_{\rm STORE},\qquad w_j=\eta k_j.
\]

The writes use exactly the existing OLD-h rule, with matching key metadata:

\[
V_j^+=V_j+w_j(h-V_j),\qquad
C_j^+=C_j+w_j(k-C_j).
\]

All other transitions remain the same. In particular, ADDRESS remains learned, and its updated address applies only to later tokens. No correct KEY/WRITE/READ route, intermediate value, semantic class or task loss is supplied.

Coverage is now derived: `z_j = sum_i C[j,i]`. Because k sums to one, this obeys exactly the old coverage recurrence. Thus the additional state preserves *which keys compose each cell*, rather than retaining only their total weight.

## Dual read

For the OLD memory state, form

\[
z_j=\sum_i C_{ji},\qquad
\bar C_j=C_j/\max(z_j,10^{-4}),\qquad
\bar V_j=V_j/\max(z_j,10^{-4}).
\]

Use one fixed ridge coefficient, provisionally \(\lambda=10^{-4}\), with no search:

\[
(\bar C\bar C^\top+\lambda I)b=\bar Ck,
\qquad r=b^\top\bar V.
\]

Feed r into the same caller/features and LOAD update as before. The dual coefficients b may be signed; they can cancel interference between overlapping cells. The empty memory reads zero. The write gate remains learned, and weak-write amplitude normalization remains available.

The consistency reason is explicit. If historical key/value pairs admit a linear association \(h_i=k_i^\top B\), then the matched V/C recurrences preserve \(V=CB\), including their cellwise decay. In the zero-ridge limit, a query in the observed row space can retrieve its associated value even when its positive write footprints overlap. The current convex reader instead averages those mixtures again. This is a conditional algebraic property, not a learning or capability result.

## Feedback and decisive failure predictions

Trainable parameters receive only the existing terminal NLL through the differentiable read solve and executed C/V transitions. The intended learning effect is to make preserving and distinguishing an address useful at retrieval, without requiring a positive reader to isolate an already pure cell.

This **does not solve reference lifetime by definition**. If SET0 again overwrites both references with the same key, C loses rank and the rule cannot reconstruct the missing binding. A nearly singular row space can amplify noise; ridge suppresses precisely those weak contrasts and may remove the hoped-for benefit. Repeated writes of different values to the same key remain subject to the original soft EMA overwrite ambiguity. Neither a full Gram matrix nor a signed read recreates discarded history.

Accordingly, any future authorized run must fail this candidate's mechanistic claim if reference separation remains collapsed, if the solve mainly amplifies noise, or if improved arithmetic accuracy again lacks counterfactual query consistency. The frozen lifetime intervention's adverse result is retained: preserving a reference alone was not a sufficient repair of the already trained NN. It is not evidence that a newly trained dual rule will work.

## Complete resource debt

P remains **10,351**, with 20 parameter tensors. C replaces eight coverage floats with 64 floats: **+56 FP32 values / +224 B per example**. Total carry becomes 384 floats / 1,536 B. This introduces no trainable address labels or additional model width.

Each read adds row normalization, an 8×8 product and regularized solve, and a signed value combination. Each write updates 64 key-composition entries rather than eight coverage entries. Factorization workspace, backward cost, numerical conditioning and GPU launch overhead are material debts; their actual latency, training FLOPs, caches and storage have not been measured. No inference-speed or total-efficiency advantage follows from unchanged P. An eventual test would use the same complete capability–parameter–cost accounting and the existing terminal-only protocol, with a separately authorized finite budget.

Evidence basis: `ab-convex-read-selectivity-saved-v1.json` (ROOT, SHA `591bdaf947ae645c177728d0ba5703c5149c4ac672ccd5f3c231aa4bc36071e0`); `ab-reference-read-history-decomposition-v1.json` (SHA `5fe36e3b52f619bd176c2e20a4e52732d498c9f3062eaa6a8c0d5727ac51e634`); `ab-reference-key-lifetime-token-posthoc-v1.json` (SHA `8f951714f7345458b87b71d2c81404cbdf40e7b3851957202ea3d96bdc3dd461`). All completed negative branches and their costs remain valid evidence. The campaign continues; this segment's three-arm training budget is complete.
