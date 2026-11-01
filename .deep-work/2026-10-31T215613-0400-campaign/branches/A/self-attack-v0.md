# A serious self-attack, before cross-branch input

## Attack 1: the demonstrated Gaussian mechanism is learned addressing

TARGET: “A new computational scope has been acquired in a fundamentally different way from prior neural memory.”

Construct the explicit finite-grid correspondence. Let K_i(k)>=0 be the Gaussian write profile at cell i and Z(k)=sum_i K_i(k). Define normalized NTM head weights w_i(k)=K_i(k)/Z(k). A write of x with no erase and add value a=Z(k)*x gives

    M_i' = M_i + w_i*a = M_i + K_i*x.

A read gives r=sum_i w_i M_i; multiply r by Z(k) to obtain the field read sum_i K_i M_i. Multiple scalar channels are simply memory-vector coordinates. Thus the worked model uses no new basic memory mechanism compared with weighted differentiable write/read. The scalar encoder outputs need not even be discovered by a recurrent controller; they are trained parameters in the reduction.

STATUS: the broad novelty claim for the minimal model is rejected. The analytic bifurcation remains a useful mechanism calculation, but it must not become an architecture novelty claim. Source: NTM §§3.1–3.3, https://arxiv.org/pdf/1410.5401, checked primary equations.

## Attack 2: the proposed three causal fields can be absorbed into ordinary messages

TARGET: “Separate learned execution E, plasticity P and construction C constitute a new form of computational capability.”

In the full local rule define effective messages

    m^E_ij = E_ij*h_j,
    m^P_ij = P_ij*R_ij,
    m^C_ij = C_ij*b_ij.

A local dynamic graph with a multi-output message function that emits these three quantities can reproduce the proposed updates without naming three fields. It uses the same products and message dimensions. If the baseline already permits learned parameter-changing and occupancy-changing messages, giving the proposal E/P/C labels does not distinguish the executed mechanism. The factorization may supply an inductive bias or useful regularization, but that is a narrower claim about learnability, finite budget and reuse—not a new reachable set. This is a specific same-cost reparameterization objection, not a universal-emulation veto.

STATUS: broad representation/capability novelty unsupported. A matched comparison must preserve code/state separation, output channels, multiplication/gating capacity and allocated host resources. Comparing against a baseline forbidden to change parameters or activate material would conflate feature access with improved self-organization.

## Attack 3: the simple task does not require construction

TARGET: “Solving two interleaved accumulators establishes learned construction of a reusable mechanism.”

Counterexample: a fixed two-dimensional state s=(s_A,s_B) with writes s_A+=x for cue A and s_B+=x for cue B solves the task exactly, with a fixed selector for queries. It never constructs a scope. Even a fixed distributed two-mode representation can achieve the same input/output behavior. The analytic field calculation shows why separation can lower interference, not that task performance identifies construction.

STATUS: task accuracy alone is insufficient. A decisive future experiment must check new-instance creation or reuse beyond the trained fixed allocation, inspect causal state/plasticity isolation, and match total host/memory capacity. Adding more contexts without these checks could merely test more storage.

## Attack 4: boundaries may suppress adaptation instead of improving it

TARGET: “A protected scope helps subsequent learning.”

Set P=0 everywhere after learning. Old computation is perfectly protected because nothing can change; future adaptation is impossible. An apparently favorable retention score does not establish that boundaries improve learning or construction. Similarly set C=0 and occupancy cost is low but no new instance is built. The proposed regularizers can favor these degeneracies unless D includes tasks whose performance requires continued local adaptation or additional instances.

STATUS: a present-task demonstration is not enough. Compare future capability with variation opportunities, feedback, active sites and local-update budget matched. A retention-versus-adaptation tradeoff must be explicit.

## Decision

REVISE THE CLAIM, RETAIN THE HIGHER-UPSIDe HYPOTHESIS FOR A CONSTRUCTIVE REPAIR: ordinary soft scoping is not original, and a named E/P/C factorization is not a capability proof. The remaining possible contribution is a *learned causal organization that copies and protects a procedure's mutable instances while preserving later adaptation*, obtained with a measurable acquisition/resource advantage from the three-process factorization. No such advantage is established here. The immediate missing step is an independent prior-art/equivalence investigation and a sharper acquisition task; the missing general program learner is not itself the reason to close this narrow hypothesis.
