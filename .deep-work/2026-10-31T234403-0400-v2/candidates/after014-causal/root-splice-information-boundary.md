# What failed-trace abundance does and does not provide

RECONSTRUCTION by root after reading actual causal v0 and its count argument; finite assay, no claimed new theorem or general neural impossibility. No experiment.

Assume 0<p<1 and integer k>=2. Suppose k hidden component outcomes each independently match a desired target with probability p. Two independently sampled executions A,B are combined by a fixed mask of b components from B and k-b from A. Before conditioning, the assembled result is entirely correct with probability p^k: exchanging independent coordinates does not improve the base rate. Conditional on both original trajectories failing, for0<b<k the exact success probability is

 P(success splice | A fails,B fails)
 = p^k (1-p^b)(1-p^(k-b))/(1-p^k)^2.

Proof: the used coordinates of A/B must all be correct (factor p^k); A must fail in its unused b coordinates, B in its unused k-b coordinates. These four disjoint coordinate events factor. Divide by the two original failure probabilities. This model deliberately grants a meaningful coordinate partition. It says nothing about arbitrary distributed activations, correlated errors, task-aware donor selection or learned masks. An adaptive controller can exploit structure only if observations/rewards reveal it.

With p=1/2 and uninformative hidden target, q/2^k equality-query counting in the candidate is the relevant information boundary. If repeated trace values have target-correlated biases (p not1/2), they themselves may reveal the target; do not carry the equality-only lower bound into that different assay for free.

A further qualification to the deterministic-target condition: Var(delta | z,j)=0 is necessary for exact squared-state imitation of all patch targets, NOT for existence of a successful donor-free operation. Two different valid latent patches may both solve every task; one deterministic selector could be sufficient. A mean target may be invalid. Therefore predictive conflicts must distinguish missing runtime information from harmless multimodality, and only actual task tests can ground that distinction. A conditional policy/mixture is an available conventional repair, not automatically a new mechanism or a reason to reopen the parked energy family.

This sharpens the causal candidate's burden: acquire informative ordinary task contexts and compile a consistent OR valid-selection transition at lower total cost than active stitching/self-imitation with the same observations. Merely storing abundant failed activations does not establish that prospect.
