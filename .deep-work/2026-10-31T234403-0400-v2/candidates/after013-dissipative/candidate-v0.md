# after013 · Learned dissipative relations (v0 frozen before source comparison)

AGENT PROPOSAL. Constructed independently from the mandate and constitutional/workflow files; no old report, candidate or catalog has been read. Effective model/mode: unknown. File isolation: instruction_only. No experiment has been run. This is an alternative foundational assumption, not a change to the MAIN topology / PARK online-organization allocation.

## Exact scientific question and success counterfactual

Can a neural system acquire internal *relations*, rather than only input-to-output maps, such that the same learned pieces can subsequently compute under different clamping and composition conditions by dissipating inconsistency? The proposed reusable object is a set of state constraints plus their relaxation field. Loss of microscopic state information is a computational operation, not a defect to be eliminated by invertible transport.

If successful in the strong sense, training on end-task feedback would construct a shared, sparse family of latent relations whose subsequent recombination can execute new computations without retraining each new forward map. It would change the organization of internal computation: one learned object could act as a gate, a partial inverse, an error-corrector, or part of a larger implicit calculation according to current boundary conditions. The relation vocabulary must be learned, not supplied as named gates. Merely fitting an energy or reproducing ordinary associative completion would not establish the strong result.

## State, execution and formation laws

Fix finite neural capacity n latent scalar units, K candidate local factors, finite precision delta, and an inference limit T. A state is z in [-R,R]^n. An episode supplies a clamp mask C, observed boundary values b_C, and a requested output mask O. Only the mask and values enter execution; no target values on O enter free inference.

Each factor is a learned scalar residual r_k(z;theta_k,m_k), represented by a small smooth neural network. A learned mask m_k in [0,1]^n controls its effective support. Define

    E_theta(z;C,b) = sum_k (rho_k/2) r_k(z;theta_k,m_k)^2
                    + (alpha/2) ||z||^2

with rho_k >= 0 and alpha >= 0. The explicit external clamp replaces z_C by b_C after every update. The free-coordinate update is

    z^{t+1}_{not C} = Q_delta clip_R[z^t_{not C}
                         - h partial_{z_not C} E_theta(z^t;C,b)]
    z^{t+1}_C = b_C.                                  (1)

Q_delta is the specified finite-precision quantizer; T, h, initialization and delta are accounted resources. This is a discrete dissipative neural executor. Smooth finite-time gradient flow alone is not asserted to be exactly noninvertible: the functional many-to-one object is the basin/settled-state map, and the actual finite executor has quantization/clipping. No thermodynamic power advantage follows from a digital simulation.

Train across boundary tasks tau=(C,b,O,y) using exactly the executed T-step map:

    L(theta) = E_tau ||z^T_O(theta;C,b) - y||^2
               + lambda_s sum_k ||m_k||_1
               + lambda_c sum_k rho_k
    theta <- theta - eta grad_theta L(theta).          (2)

The gradients can be obtained by ordinary differentiation through (1), using a smooth surrogate only during parameter updates if Q_delta prevents differentiation. Such a surrogate is a mismatch to be checked, not hidden. Alternatively freeze delta=0 during formation and explicitly test a quantized executor later. Masks use a continuous relaxation and are thresholded only under a separately frozen version. There is no promise that these gradients discover useful factors. A zero residual family is not a solution to supervised output loss unless the initialization already predicts all requested outputs; dependence on initialization is therefore a decisive control.

What is manually designed: factor-capable neural substrate, descent executor, masks, loss, clamp semantics, capacity and resource bounds. What is externally optimized: residual functions, supports and weights. What is executed locally: state relaxation. What is retained: trained parameters and masks. What may be learned: reusable relational structure. Evolution, autonomous online organization, and open-endedness are not asserted.

## Worked construction: a reusable relation rather than a forward gate label

This example establishes executable semantics only; it supplies the factors manually and does NOT establish acquisition. Let x_1,x_2,x_3 in {-1,+1}; use free states a,b and potential

    E(a,b|x) = (1/2)(a - x_1 x_2)^2
                 + (1/2)(b - a x_3)^2.                (3)

The unique minimum is a=x_1x_2, b=x_1x_2x_3. For a,b the Hessian is

    H = [[2, -x_3],[-x_3,1]],

with eigenvalues (3 +/- sqrt(5))/2. Thus unquantized synchronous descent converges from every finite initial a,b whenever 0<h<2/lambda_max. No state path must remain distinct: all initial a,b converge to the same answer for these boundary values. Small roundoff is continually corrected rather than requiring preservation of trajectory labels.

A fully relational three-variable factor can instead use

    e(u,v,w) = (1/2)(u v w - 1)^2.                    (4)

Clamping any two signs lets the third descend to their product. The learned object could therefore support different query directions and compose by identifying shared state coordinates. A feedforward product gate does not supply all these boundary problems automatically. This is a functional distinction; it is not an equal-resource advantage over a sufficiently well-designed bidirectional rival.

The nontrivial formation target is that (2), supplied with only selected end-task boundary examples, produces factorization equivalent to (4) or another useful latent algebra *without being given product factors, gate roles, intermediate labels, or task decomposition*. Merely putting arbitrary neural residuals into (1) does not prove this target.

## Fragile step and an immediate resource warning

The weakest step is learned factor formation and subsequent useful reuse. End-task loss can choose dense global constraints, encode task-specific solutions, make irrelevant directions flat, or create bad local minima. Sparsity alone does not identify a useful algebra. A task distribution must leave a real held-out recombination problem, rather than putting the answer into boundary encoding or choosing a human-designed factor graph.

Even exact factorization does not ensure efficient execution. For a length-L feedforward chain of squared residuals, the relevant Hessian is a signed path Laplacian with a fixed boundary. Its smallest eigenvalue is Theta(L^-2), while its largest is O(1). Synchronous local relaxation therefore needs Theta(L^2 log(1/epsilon)) sweeps in the worst slow mode, with Theta(L) factor work per sweep. A forward sequential evaluator needs only Theta(L) gate applications on that DAG. A particular architecture or update schedule may alter this comparison; the generic energy prescription supplies no speedup. This derivation is analytic and has not been independently checked or numerically tested.

## Status at first freeze

CLAIM: task feedback can construct a reusable dissipative relational mechanism rather than only fit maps.
STATUS: HYPOTHESIS / OPEN; the executor and hand-built example are explicit, factor acquisition is not established.
EVIDENCE/ARTIFACT: equations (1)-(4) and hand derivation in this file; no experiment or source comparison yet.
WEAKEST REMAINING LINK: identifiability and acquisition of reusable factors under matched information/resources.
FAILED BRANCHES: none tested; long-chain relaxation cost is an identified attack.
NEXT SINGLE DECISION: read closest original mechanisms and compare to a same-information learned factor graph / energy model before requesting any experiment.

---

## Post-freeze source audit: the v0 executor is concretely covered

SOURCE: Du, Li and Mordatch (2020), Section 3.1-3.2, printed pp. 3-4, equations (1)-(5), already give neural energy functions, gradient-based sampling, and conjunction by adding energies. Du et al. (2021), COMET, Section 3.1-3.2, printed pp. 3-4, equations (1)-(2) and Algorithm 1, go further: a learned encoder supplies components; their energy sum is minimized by finite gradient steps; image reconstruction trains those components. Du et al. (2022), IREM, Section 3.1-3.2, printed p. 3, equations (2)-(6), learn algorithmic answers by regressing the results of gradient steps on a neural energy. These are direct mechanism comparisons, not a claim based on universal simulability. Precise reading records and links are in `sources/after013-dissipative/reading.md`.

RECONSTRUCTION: Arbitrary residual networks, learned masks, end-task regression and dissipation do not by themselves distinguish v0 from that family. Changing the application from image concepts to computation is not a new formation principle. A factor graph with the same neural factor family, masks, optimizer and tasks is an especially strong rival: after squaring the residuals its free-state update can be exactly equation (1). There is then no proposed experiment that can establish an executor-level distinction.

## One substantive formation repair: counterexample-triggered constraint creation

This is a separate repair, not a retroactive change to frozen v0. It replaces the vague expectation that sparsity will discover useful factors with an explicit event that creates a factor and identifies its scope. It is deliberately stronger in information access than passive examples, and its cost is exposed below. It still has a serious prior-work/importance limit.

**Finite domain and feedback.** Let X=(X_1,...,X_n) have a finite product domain D^n, with a target behavioral relation R subset D^n. A full-assignment validity query V(x) returns 1 iff x is in R. The current model starts with an empty forbidden-pattern set N, so it over-accepts. Boundary tasks ask it to complete partial x by the dissipative executor. This is an offline formation stage with a finite query budget; no claim of autonomous online organization is involved.

**Creation event.** Whenever a candidate completion x satisfies all current constraints but V(x)=0, the model must add a constraint that excludes x. It starts with S={1,...,n}. For each i in S, test whether the partial assignment x_{S without i} has any valid full completion. If no such completion exists, delete i. Repeating to a fixed point yields an inclusion-minimal nonextendable assignment a=x_S. Add (S,a) to N. This is a concrete state-dependent structure update, not a gradient nudging a preassigned factor:

    N <- N union {(S,a)} whenever x is accepted by N but V(x)=0.       (5)

Every added pattern is sound: no valid full tuple matches all its specified coordinates. Hence a formerly valid tuple is never excluded by an exact creation step. Inclusion-minimal is not minimum-cardinality, and the learned set need not be a smallest constraint basis.

**No hidden extension oracle.** Ordinary V does not answer the extension question cheaply. The exact implementation enumerates every assignment of unspecified coordinates and calls V, stopping early if it finds a positive completion. A certificate of nonextendability may require |D|^(n-|S|) full queries. One greedy scope search has at most n deletion checks per pass, and a conservative upper bound is O(n|D|^n) full validity calls. If a task supplies an extension oracle or explicit local-conflict feedback, that is additional information and must be given to rivals too. This repair supplies no general efficient substitute.

**Actual neural realization.** For Boolean sign variables and a forbidden pattern a on S, define a product unit

    p_(S,a)(z) = product_{i in S} (1 + a_i z_i)/2,
    e_(S,a)(z) = gamma p_(S,a)(z)^2.                              (6)

At sign-valued vertices p is exactly the indicator that the forbidden assignment is matched. Its parameters and support are generated by the counterexample certificate; they are not picked from named human gates. Sum these energies, add a specified well potential if required, and execute the same clamped descent. Equation (6) is an explicit neural product primitive, not a claim that product itself emerged. Vertex correctness of the constraint classifier does not prove that continuous relaxation finds a valid completion.

**Compression event.** Once several certified no-goods share a scope S, their Boolean union indicator f_S may be expanded exactly in the Walsh basis:

    f_S(z) = sum_{A subset S} c_A product_{i in A} z_i,
    c_A = 2^(-|S|) sum_{a in {-1,+1}^S} f_S(a) product_{i in A} a_i. (7)

Only coefficients exactly equal to zero are discarded. A sparse polynomial is stored as one residual with its monomial supports. This can turn multiple negative certificates into a compact reusable relation and a new correction field. Computing (7) takes O(|S|2^|S|) arithmetic with a fast transform plus the stored table; a sparse result is not promised. Replacing separate energies by the squared compact residual preserves the forbidden set on vertices, but changes the interior relaxation and requires a separately frozen execution check. No approximate compression is called verified.

**Worked formation trace, without a supplied gate label.** Suppose the unknown relation requires X_1=X_2 and leaves other variables free. A negative full tuple with X_1=+1,X_2=-1 lets scope deletion remove every other variable: all their completions still fail V. Dropping either of the first two variables admits a valid completion, so the certificate scope is {1,2}. The opposite mismatch produces a second certificate. Their union indicator simplifies through (7) to

    f(z_1,z_2) = [(1+z_1)(1-z_2)+(1-z_1)(1+z_2)]/4
               = (1-z_1 z_2)/2.                                 (8)

For clamped z_1 in {-1,+1}, minimizing f^2 sends z_2 to z_1; clamping z_2 instead reverses the query direction. The relation and its support have been acquired from validity responses, with explicit potentially expensive scope tests. This is a worked hand derivation, not an experiment, novelty demonstration, or demonstration that hidden computational decompositions were acquired.

## Strongest attack on the repair

SOURCE: Bessiere et al. (2013), QUACQ, Section 2 and Algorithm 1/FindScope/FindC, already make rejected examples trigger scope discovery and addition of a constraint. Their learner assumes a constraint-language bias and *partial-query feedback about violation of target constraints*. That is not the same as an existential extension oracle; a partial assignment can avoid all fully scoped local violations while still having no global solution. The distinction changes the query cost. Our complete no-good language avoids a fixed small gate catalog but can be exponential; this is an ordinary expressivity/resource tradeoff, not evidence of a new principle.

The strongest same-information rival receives the same full validity queries, the same query budget and variable vocabulary, stores the exact same no-good certificates, applies the same Boolean-polynomial compression, and uses a symbolic constraint solver or a learned factor graph for execution. It gets no oracle or intermediate labels denied to the neural system. On acquired factor identity this rival is identical to the proposed rule. A neural potential may interpolate differently, but a useful interpolation or acquisition advantage has not been derived.

The repair acquires constraints on *observable task variables*. It does not discover hidden intermediate variables that give a much shorter algorithm, and it has no justified efficient rule for introducing such variables. Asking V whether an arbitrary latent neural state is correct would silently provide an unavailable semantic oracle. A broad neural-mechanism formation claim therefore remains unsupported even after this repair.

## Analytical refinement and remaining formation defects

RECONSTRUCTION: For the chain warning above, sign changes conjugate the Hessian to the tridiagonal matrix with diagonal (2,...,2,1) and off-diagonal -1. An eigenvector v_j=sin(j omega), v_0=0, obeys the final boundary v_{L+1}=v_L. Thus omega_k=(2k-1)pi/(2L+1), and

    lambda_k = 2 - 2 cos[(2k-1)pi/(2L+1)], k=1,...,L.              (9)

For stable fixed h of order one, the slowest-mode contraction factor is 1-Theta(L^-2), giving Theta(L^2 log(1/epsilon)) synchronous sweeps for relative error in that mode. With L scalar factor evaluations per sweep this is Theta(L^3 log(1/epsilon)) factor work, versus Theta(L) for a known forward DAG. The conclusion concerns this executor, not all dissipative computation.

ATTACK: v0's mask/rho regularizers have scaling degeneracies unless factor weights and mask usage are norm-bounded: a small mask can be offset by large first-layer weights, and small rho by rescaling its residual. Such bounds would be a needed engineering/identifiability repair. They would not create the missing formation principle.

## Current finite decision

CLAIM: Dissipative learned relations provide a distinct foundation from invertible transport.
STATUS: RECONSTRUCTION of a legitimate but substantially existing computational organization; no novelty claim.
EVIDENCE/ARTIFACT: explicit execution/formation laws; analytic small construction; checked primary-body comparisons; exact no-good formation repair with costs.
WEAKEST REMAINING LINK: an information- and resource-justified rule for discovering compact hidden computational relations, beyond established energy/factor/constraint learning.
FAILED BRANCHES: generic backprop+sparsity is covered by close mechanisms; the counterexample repair supplies structure creation but transfers the burden into potentially exponential queries and an equivalent symbolic acquisition rival.
NEXT SINGLE DECISION: do not promote this candidate or run a parity/chain demonstration as scientific novelty. A continuation would require a specific new compact-formation mechanism with a real difference against the matched acquisition rival. No such mechanism is claimed here; topology remains MAIN and the forbidden PARK branch was not investigated.

---

## Response to root's necessary-composition counterexample: a real representation repair and its price

After the formation repair, I read `root-composition-attack.md`. Its point is stronger than a bad initialization at zero: individually exact factors can create a strict wrong local minimum under scalar continuous descent. I independently checked the displayed Hessian structure symbolically by differentiation: for pair residuals plus the beta-weighted triple residual at a=b=c=t, diagonal terms are 2t^2+beta t^4 and off-diagonals are 2t^2-1+beta(2t^4-t). The symmetric/orthogonal eigenvalues follow. This is an analytic check, not an independent experiment or formal proof certificate.

One actual representation change removes this obstruction: replace one continuously valued assignment z by a **joint distribution p over complete discrete assignments** s in {-1,+1}^n. The entire correlation structure is retained. With full-support p_0, execute

    p_{t+1}(s) = p_t(s) exp[-eta E(s)]
                 / sum_r p_t(r) exp[-eta E(r)].                    (10)

Read out a maximum-weight discrete assignment; do not independently threshold marginals, which could again violate a joint relation. This is multiplicative dissipation of inconsistent joint states. It changes state representation and the executed rule; it is not beta or step-size tuning in the original descent.

RECONSTRUCTION: Iteration gives p_t(s)=p_0(s)exp[-eta t E(s)]/Z_t. If a unique correct s_* has E(s_*)=0 and all others have E(s)>=Delta>0, then

    1-p_t(s_*) <= [(1-p_0(s_*))/p_0(s_*)] exp[-eta t Delta].        (11)

No interior stable wrong distribution survives. In root's specific clauses, the all-negative vertex has energy 0.2, the all-positive vertex has energy 0, and every other sign assignment has energy at least 4; therefore Delta=0.2. The weight ratio of all-negative to all-positive decreases exactly by exp(-0.2 eta) per step. Full-support initialization is essential; zeros never recover in (10).

**The cost defeats a promotion claim.** Exact p stores 2^n numbers, and evaluating/normalizing an update touches the whole joint state space. With uniform p_0, finding its MAP after one update already computes the minimum-energy assignment by evaluating every candidate: the apparent repair is explicit enumeration. Existing exact factor inference expresses the same operation, and root's GF2 solver solves this particular clause family much more cheaply. Restricting p to independent marginals removes the very joint information used by this repair. Compact structured beliefs could help selected graph families, but no new generally compact representation or learning rule is constructed here.

This repair shows the obstruction is not an impossibility of dissipative computation. It also shows why the original claim needs a resource-sensitive execution mechanism, rather than the statement that the right assignment is a global minimum. It does not rescue hidden-factor formation, and it adds no evidence that neural computation has acquired a new internal algorithm. Recommendation remains: retain this finite counterexample/repair analysis and do not broaden the branch into generic inference engineering.
