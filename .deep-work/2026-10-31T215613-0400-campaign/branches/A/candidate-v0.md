# A v0 — learn the causal scope of an internal computation

**Freeze state:** independent first candidate, before any cross-branch candidate or report. Author: AGENT PROPOSAL. Route: inherited user-authorized gpt-6.1-sol / Ultra per task metadata, not independently instrumented and not labelled Pro. Sources and code are frozen alongside this file. No end-to-end architecture training has been run.

**Judgment:** A network's learned procedure is difficult to reuse if calling it twice also aliases its mutable state or its learning updates. A potentially consequential primitive is therefore an acquired *causal scope*: a region that shares an operation with other regions while controlling which activation, parameter-update and construction events can cross its boundary. The exact worked case below shows how task loss can destabilize a single mixed memory mode into two independently addressable modes. That case alone is close to established differentiable memory. The higher-upside proposal is to learn the same boundary across execution, local learning and the construction of additional instances, with no external module extractor. Its acquisition and distinctiveness are OPEN.

## Computational opportunity

An operation and an instance of that operation are different things. The same recurrent transition f can maintain several accumulators, several partial parses or several working hypotheses only if each instance has the appropriate independent state. Sharing weights is useful; accidentally sharing state or plasticity can be destructive. Conversely, hard-coding a module slot, an allocator and variable-binding conventions gives away part of the construction problem.

The proposed shift is to make *where a learned transformation is allowed to act and reproduce* an adapted object, not just the transformation's input/output map. In a bounded neural substrate, a learned scope can be constructed by changing a causal connectivity relation. This is not a claim of an expanding universal state space; host capacity is finite and fixed.

Kauffman's aggregate machines already establish composition and altered kinetic accessibility as known ideas. Banzhaf's matrix chemistry already makes produced objects executable. The unresolved opportunity here is neural acquisition of selective boundaries around a reusable mutable mechanism. The source comparison and its limits are in `sources/A/reading-and-reconstruction-v0.md`.

## 1. An explicit minimal model: a state field with learned affinity

Let r be a spatial/affinity coordinate, and let K_sigma(r-k) be an L2-normalized Gaussian profile. The neural substrate's mutable activation field is m_t(r). A context encoder k_phi(c) produces an address from an observable cue c. At runtime, a scalar write x and later query execute

    m_{t+1}(r) = m_t(r) + x_t K_sigma(r-k_phi(c_t)),
    y_t = integral K_sigma(r-k_phi(c_query)) m_t(r) dr.

This is an actual state update, with the field reset at episode start. The Gaussian self-overlap is 1 and its cross-overlap is

    g(d) = exp[-d²/(4 sigma²)],
    d = |k_phi(c_A)-k_phi(c_B)|.

In the finite implementation, r is a fixed grid of N neural cells and the integral is a weighted sum. An alternative particle implementation stores field contributions; it has explicit memory growth per write and must not be credited as free storage. Neither implementation has been run here. The probe evaluates the exact two-context reduction.

Assume two independent zero-mean accumulated quantities M_A,M_B, each with variance 1. After their writes,

    m(r) = M_A K(r-k_A) + M_B K(r-k_B),
    y_A = M_A + g(d) M_B,
    y_B = M_B + g(d) M_A.

Each query should return its own accumulated value. The average half-squared error is

    L_task(d) = 1/2 exp[-d²/(2 sigma²)].

Use a coordinate-spread cost lambda*d²/2. The adaptation law between episodes is ordinary gradient descent:

    L(d) = L_task(d) + lambda*d²/2,
    d_{n+1} = d_n - eta*d_n*(lambda - exp[-d_n²/(2 sigma²)]/(2 sigma²)).

No scope assignment oracle appears in this update. In this reduction the context encoder has only two trainable outputs; the observable cue-to-output mapping is supplied as the minimal model, so this is **not** an acquisition result for arbitrary context recognition.

At exact collision d=0, the gradient is zero. Its curvature is

    L''(0) = lambda - 1/(2 sigma²).

If lambda < 1/(2 sigma²), collision is unstable to a nonzero perturbation but is preserved by an exactly symmetric deterministic initialization. Symmetry breaking is a required ingredient, not an emergent miracle. A fixed small random perturbation to encoder parameters at initialization is sufficient for this reduced system; no special task-dependent separator is supplied. If cost exceeds the threshold, splitting is not favored.

The stationary separated state satisfies

    d*² = 2 sigma² log[1/(2 lambda sigma²)].

With sigma=1 and lambda=0.001:

    d*=3.5255093528,
    g(d*)=0.04472135955,
    L_task(d*)=0.001,
    L_total(d*)=0.00721460810.

The memory Gram matrix is [[1,g],[g,1]]. At d=0 it has rank one and cannot represent the difference between the two accumulated values. At nonzero d it has rank two; at d* its condition number is 1.093629979. Thus the constructed object is an additional independent mutable mode and better conditioned access to it, not new arithmetic. A host with more modes was already available.

### Executed observation and a preserved failure

`probes/A/v0_calculation.py` actually executed the scalar adaptation for 500 updates at eta=0.5. Starting at d=0 it stayed exactly zero with loss 0.5. Starting at d=0.05 it reached d=3.505188641 with total loss 0.00721722139. The analytic derivative matched central finite differences at five points to <1e-9. The original assertion that 500 updates would reach |d-d*|<0.001 **failed**; the gap was 0.0203207121. The first script/failure is retained, and training was not silently extended. The result demonstrates the local instability and loss reduction at this small budget, not the originally assumed convergence tolerance.

For an explicit runtime computation at d*, let M_A=1 and M_B=2. Query A returns 1.089442719 and query B returns 2.044721360. This exposes remaining cross-talk. It must not be reported as exact independent scoping. In particular, overwriting B can still affect A by g(d*).

If exact barriers are required, use a compact-support profile rather than a Gaussian. For the normalized triangular profile of support radius R, the overlap for u=d/R is

    g(u)=1 - 1.5u² + 0.75u³,       0<=u<=1,
         (2-u)³/4,                 1<=u<=2,
         0,                        u>=2.

Then sufficiently separated fields have zero causal cross-talk. This is an unexecuted analytic extension. Positive spread cost can still favor nonzero overlap, and ordinary gradient descent need not reach the exact cutoff in a finite budget. It is not included among the verified probe results.

### What is designed and what is acquired in this case

- Designed: field memory, write/read protocol, Gaussian affinity, episode reset, scalar accumulation, supervised targets, coordinate-spread penalty, optimizer and two identifiable cue features.
- Acquired in the executed reduction: separation between two encoder outputs, from a small nonzero initialization, as a response to cross-talk loss.
- Not acquired: addition itself, general procedure code, a variable-binding algorithm, a new task, an allocator for unlimited fresh contexts, an endogenous local learning law or a constructor. The calculation does not establish inheritance/evolution.

**ATTACK:** on a fixed grid, this is kernel-weighted differentiable memory. NTM erase/add and read operations can express it after normalization/rescaling. The literature check prevents treating a Gaussian key field alone as a new architecture. A causal “membrane” metaphor is insufficient.

## 2. The protected higher-upside construction: a learned boundary governs three causal processes

The more ambitious proposal makes the construction law explicit. It removes a designated centralized module-discovery or graph-compression controller. All material is in a finite local neural graph; a learned local law decides occupancy, connections, local parameter change and copying. Ordinary outer gradient training is allowed and declared.

Take a fixed bounded nearest-neighbor host graph A with N sites. At site i, retain:

    h_i in R²                 runtime state;
    Theta_i in R^(2x2)        local transition code;
    c_i in R²                 affinity/boundary code;
    o_i in [0,1]              active occupancy.

Three different learned directed gates on neighboring sites control:

    E_ij: activation execution;
    P_ij: local parameter updates;
    C_ij: construction/copying into inactive material.

For each ordered neighboring pair let z_ij concatenate h_i,h_j,vec(Theta_i),vec(Theta_j),c_i,c_j,o_i,o_j,u_i,u_j and scalar feedback rho_t, for 21 scalar inputs (u_i,u_j are scalars). A fixed two-hidden-layer MLP H_psi: R^21 -> R^10 has width 32, tanh hidden activations and a linear output. Its outputs are three gate logits, four proposed parameter-update values, two affinity-update values and one birth logit. Thus no unspecified “discover a module” function is invoked; the local law is finite and learnable. Feedback rho, if supplied, is external and may be delayed.

Using the first three outputs q^E,q^P,q^C,

    E_ij = sigmoid(q^E_ij - ||c_i-c_j||²/tau),
    P_ij = sigmoid(q^P_ij - ||c_i-c_j||²/tau),
    C_ij = sigmoid(q^C_ij - ||c_i-c_j||²/tau).

Let R_ij be the next four outputs reshaped as a 2x2 matrix, v_ij the next two, and b_ij=sigmoid(last output). Let n_i=sum_j A_ij o_j; use n_i^+=max(1,n_i) only as a bounded normalization, not a learned oracle. A synchronous executed step is:

    h_i' = o_i * tanh(Theta_i h_i
             + W sum_j A_ij o_j E_ij h_j/n_i^+ + B u_i),

    Theta_i^learn = Theta_i + eta_P o_i sum_j A_ij o_j P_ij R_ij/n_i^+,
    c_i' = c_i + eta_c sum_j A_ij o_j v_ij/n_i^+.

Construction rate and parent code are

    r_i = 1-exp[-eta_C sum_j A_ij o_j C_ij b_ij],
    p_ij = A_ij o_j C_ij b_ij / max(epsilon, sum_l A_il o_l C_il b_il),
    Theta_i^copy = sum_j p_ij Theta_j,
    birth_i = (1-o_i)*r_i,
    o_i' = o_i + birth_i,
    Theta_i' = (1-birth_i)*Theta_i^learn + birth_i*Theta_i^copy.

Newly active material starts with h_i=0 at the next step; Theta is inherited from local parent material, possibly with the declared learned update on later steps. Occupancy only increases in this minimal version, and a cost/budget limits it. Adding death/recycling is a separate unverified change, not assumed. Birth is soft activation of already allocated host material; it is not dynamic host allocation. W and B are shared learned matrices. The sigmoid law gives soft permeability, not exact topological cuts. A threshold or compact support would be an explicit architectural change if exact barriers were later studied.

This rule is intentionally ordinary enough to execute, but its object of learning is unusual: a useful local computation may learn different permeability for *performing* its operation, *changing* its code and *building* an additional instance. A desirable outcome would permit execution channels between two scopes while preventing accidental sharing of local state or parameter mutation, and would copy a useful Theta into fresh material without copying the old runtime h. It need not be an organism or evolutionary system.

### Adaptation and acquisition problem are not hidden

Fix an episodic supervised task distribution D. Reset h and o to a specified small seed, inherit learned initial Theta,c, and unroll the transition for a fixed T. Train psi,W,B and the initial seed parameters by BPTT on

    J = E_D[sum_t ||O h_t-y_t||²/2
             + lambda_o sum_i o_i(T)
             + lambda_p sum_(ij),t P_ij(t)²
             + lambda_c sum_(ij),t C_ij(t)²].

The output map O and input placement are designed interfaces. All outer gradient computation, episode data, input placement and initialization are part of the cost. Inner Theta changes use the learned R_ij, not an undeclared SGD solver. No evidence currently says this outer optimization will learn reusable scope construction; it may simply learn a sparse recurrent network or exploit soft fractional occupancy.

At bounded graph degree Delta, a local step costs O(N*Delta*32²) MLP work plus the small matrix state updates; naive BPTT uses O(T*N) activations times constant feature width. The host reserves N potential sites in advance. This proposal is not free parallel computation, and copying Theta has nonzero local event and material costs.

### The specific claim worth attacking

HYPOTHESIS: coupling acquired causal boundaries across E,P,C can make a learned stateful neural computation reusable in more independent instances, or more robust to subsequent local adaptation, at matched host/interaction/training budgets than learning execution addressing alone.

This is a single consequential gap—construction of reusable *instances* of computation—not a claim of construction of all mathematical functions. The source-guided novelty boundary is sharp: if ordinary differentiable memory or a dynamic graph with execution gates alone explains the behavior at matched budget, the stronger claim has failed. Renaming gates as membranes does not rescue it.

What would count as actual acquisition: Theta and boundaries initially do not implement the target procedure, training acquires it, then the trained local rule builds additional usable instances under new interleavings/instance counts. Intervening on the memory or local plasticity of one instance should spare another's current function. Merely plotting visually distinct clusters or completing training examples would not count.

No target procedure is preinstalled in the full rule above. The simple accumulator calculation deliberately *does* preinstall it, so it cannot be used as evidence that the full acquisition happened. This is the hardest remaining bridge.

## Weakest links and discriminating next decision

1. **Closest rival:** NTM-style learned addressing or ordinary sparse adaptive recurrent connectivity may capture all benefit. The Gaussian worked case is already within this rival. A prior-art investigation of learned modularity, neural cellular automata and differentiable allocation is required before a novelty claim. It was not completed in this independent phase.
2. **Causal confounding:** barriers that prevent corruption may simply remove learning opportunities or allocate more memory. All host sites, active sites, local interactions, optimizer steps and reset/replay costs must be counted.
3. **Symmetry:** the reduced model needs nonzero perturbation; exact symmetry never splits. In the full local graph a seed/geometry may already encode the desired division. That must be controlled.
4. **Irreversibility:** after two histories have been merged into a rank-one state, their missing difference cannot be reconstructed by merely moving keys. Between-episode learning/reset is explicit; claims of repairing an already-pooled runtime state would be false.
5. **Soft cheating:** fractional occupancy, tiny permeability and unbounded precision can produce impressive loss without discrete independent mechanisms. Causal interventions and finite-precision robustness are needed.
6. **Learning discovery:** H_psi is a fully specified small neural law, but an optimizer finding a reusable constructor is OPEN. Specifying a trainable network does not solve that discovery problem. The task distribution must not hand the desired routine to it through privileged intermediate targets.

The next single decision is whether the three-process boundary hypothesis merits an independent prior-art/equivalence attack. The present certificate does **not** authorize a large training run. A small future test, if retained after that attack, should freeze criteria for shared-code/independent-state reuse and compare matched execution-only versus E/P/C laws. Results could be FALSIFIED or merely an incremental memory architecture.

## Research certificate

CLAIM | STATUS | EVIDENCE

- The printed matrix-chemistry threshold convention is internally inconsistent in the N=4 literal case. | VERIFIED in the stated equations | page image + 256-pair enumeration.
- von Neumann/Burks copy-consume convention needs three description copies to leave parent and child both complete. | SOURCE + VERIFIED finite accounting | PDF 23 / printed 85 + probe.
- Task loss can destabilize a colliding two-mode field and acquire separation under the stated scalar model. | VERIFIED analytic/finite calculation | derivative check + 500-update trace; convergence-tolerance attempt failed.
- The Gaussian model alone is a novel neural construction principle. | NOT SUPPORTED; close special case of differentiable memory | NTM §§3.1–3.3 comparison.
- Learned E/P/C boundaries can construct reusable neural computation instances beyond addressing alone. | HYPOTHESIS, unbuilt | explicit local transition/adaptation laws; no acquisition experiment.

WEAKEST LINK: joint acquisition and causal usefulness of the three-process boundary.
FAILED BRANCHES: broad novelty of simple kernel-addressed scopes; exact-symmetry separation; assumed 500-step convergence tolerance.
NEXT SINGLE DECISION: independent equivalence/prior-art attack on the E/P/C mechanism, before an end-to-end training commitment.
