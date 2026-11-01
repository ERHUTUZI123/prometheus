# Shared plasticity inside an anchor-preserving covering chart

**AGENT PROPOSAL / RECONSTRUCTION; mathematical construction only.** Same V2, 2026-11-01. MAIN=TOPOLOGY; PRIORITY=USER_LOCKED; MAIN_CHANGE=HUMAN_ONLY; ONLINE=USER-PARK; PERSISTENT_STATE=PAUSED_STANDALONE_RESERVE. No experiment, code audit, training, dependency, model change or resource change was performed. This is scientific support inside topology, not a separately allocated research direction. Effective provider mode is unknown.

## Judgment and exact question

There can be **functionally useful shared parameter directions even when every old based-loop operation is exactly fixed**. The remaining directions must affect an observation outside that protected endpoint interface. Here the new observation is an actual continuous state coordinate at a declared intermediate port, with its decoder fixed before learning. An anchored invertible neural coupling changes this output, the relation and the old trajectories, while preserving old operations on their literal original resting states.

This construction answers an existence/learnability question, not the favorable-total-cost hypothesis. It also exposes a strong limitation: the whole family factorizes into a fixed canonical cover and a learned invertible adapter. A conventional implementation can exploit that factorization. The important uncertainty is whether a less factorized useful task can retain the cheap semantic certificate without paying more in execution and restricted plasticity than modular alternatives. No positive answer to that uncertainty is assumed here.

Coverage: **DIRECT** for existence of a nonzero, useful shared learning direction with exact mathematical old endpoint preservation; **PARTIAL** for useful continuing adaptation at favorable total cost; **OUT_OF_SCOPE** for learning a new root-loop permutation on the fixed cover, neural superiority, a physical energy advantage or historical novelty. This is a new prospective interface, not a reinterpretation of the completed quartic experiment.

## 1. Limitation, retained state and new demand

The earlier coefficient update preserved acquired operations, but improved a sampled sensitivity proxy by only about 1.6e-8 relatively and did not improve measured tracking work. It established no useful functional plasticity. Conversely, after012's fixed-graph theorem proves that preserving the full generating holonomies fixes every root-loop operator. We must not count motion of an unobserved frame as a new computation.

Retain an already owned regular covering relation P(w,u)=0, actual running state z, control u, a resting control u0, its finite root anchors a_i, and previously acquired executable control paths. The new mutable parameters theta enter a common map h_theta used by **all** old and new relation execution:

    F_theta(z,u) = P(h_theta(z),u).

The canonical relation P is fixed in this version. Thus this is a factored shared implementation with an immutable mathematical core, not a claim that arbitrary old weights remain freely plastic. Nevertheless theta is not a new-task-only branch: it changes the physical/interpreted z trajectory of each old operation whenever that trajectory passes a region where h_theta changes.

Old capability has its original fixed interface: start at literal z=a_i at u0, run an old closed path, and report its endpoint using the original fixed branch decoder. Numeric anchors will not be transported or relabeled. The new task uses a previously declared open path to a control u, starting on an old selected branch, and predicts a scalar from a **fixed physical z coordinate**. Its environment supplies that continuous target label, not a target h, relation coefficient, desired permutation, hidden canonical root or loop class. Those are additional labels for this new task; the old run did not supply them.

This matters to the Fundamental Question as a sharply limited mechanism: an existing executed branch computation supplies reusable state, while later task evidence changes a shared continuous transformation without replaying every composed old operation. It does not construct an unrestricted new operation algebra.

## 2. An exact parameter family, not a tangent-only proposal

Write z=(x,y) and identify R^2 with C only inside P. Use the globally invertible additive coupling

    h_theta(x,y) = (x, y+f_theta(x)),
    h_theta^-1(a,b) = (a, b-f_theta(a)).                 (1)

For a worked cover use P(w,u)=w^3-3w-u, u0=0. Its anchors are -sqrt(3), 0, +sqrt(3), all with imaginary part zero. Its two critical control values are +2 and -2. The familiar two loops provide noncommuting transpositions, but this note proposes no rerun of that witness.

Set

    psi(x) = x(x^2-3)/(1+x^6),
    c_j(x) = psi(x) phi_j(x),
    f_theta(x) = sum_(j=1)^p theta_j c_j(x),            (2)

where phi_j(x)=tanh(alpha_j x+b_j) are fixed bounded neural features, or include the constant feature 1. These supplied features are a restricted, inspectable acquisition family. Output weights theta are learned. No claim is made that the features, coupling architecture or cover emerged.

Because psi vanishes at all three anchors, h_theta(a_i)=a_i for **every finite theta**. Equation (1) is a global diffeomorphism for any smooth f; its Jacobian determinant is exactly 1. There is no finite-step retraction approximation, residual relation penalty or empirical invertibility test.

A general finite anchor fiber can be treated with a factor vanishing at each distinct anchor x-coordinate, including coincident coordinates only once. That requires storing/constructing the anchor factor, and can worsen degree, feature conditioning and evaluation cost. Nothing here proves dimension-independent or anchor-count-independent efficiency for that extension.

## 3. Actual execution and parameter migration

Let w=h_theta(z). Away from the canonical critical locus, execute

    w_dot = (u_dot-kappa P(w,u))/(3w^2-3),
    x_dot = Re(w_dot),
    y_dot = Im(w_dot) - f_theta'(x) x_dot
                         - sum_j c_j(x) theta_dot_j.    (3)

Complex division stands for an ordinary real 2-by-2 solve. The last term is essential when parameters change. Equations (1)–(3) give dP/dt=-kappa P. They carry the existing branch; they never initialize it from an independently selected root. A bounded numerical executor would still need predictor/corrector accuracy and a termination contract; none was executed here.

For a finite learning step at fixed u, migrate state exactly by

    x_new = x_old,
    y_new = y_old + f_theta_old(x_old)-f_theta_new(x_old). (4)

Thus h_theta_new(z_new)=h_theta_old(z_old). This is an explicit state transition through a parameter change, not a free branch-setting actuator. It can also be implemented by integrating the last term of (3). At an anchor, (4) is the identity.

The original critical control values do not move: the real state Jacobian of F is D_wP Dh_theta and Dh_theta is everywhere nonsingular. Thus criticality occurs precisely when the canonical derivative vanishes. For general P the same chain-rule statement holds. A once-checked old path remains disjoint from the same critical control set for every admitted theta. Unlike the earlier coefficient-ball certificate, one need not rediscover and enclose moving critical values after each theta step.

## 4. Full old-operation preservation, and exactly where new behavior lives

For any old control path gamma, its lifted transition obeys

    T_gamma^theta = h_theta^-1 o T_gamma^0 o h_theta.    (5)

This follows by applying (3) to the unique canonical lift. For a closed based path and every anchor a_i,

    T_gamma^theta(a_i)
      = h_theta^-1(a_(sigma_gamma(i)))
      = a_(sigma_gamma(i)).                            (6)

The same fixed anchors work for all theta and all generators. Therefore (6) holds for every finite word and inverse word in the acquired old library, using one common identification, here literally the identity. The claim is exact ideal mathematics, not unlimited finite-precision reliability.

Old trajectories genuinely move: at a canonical point w=(a,b),

    z_theta=(a,b-f_theta(a)),
    partial z_theta/partial theta_j = (0,-c_j(a)).       (7)

When c_j(a) is nonzero this is an actual change of the old relation's state coordinates and transition vector field under the shared parameters. At the fixed resting endpoints it cancels exactly.

Equation (5) also explains the obstruction. If only root-loop endpoint labels are observed, every theta gives exactly the same computation. This family **cannot acquire a new root-loop permutation**. New functionality comes from an explicitly fixed intermediate-port measurement

    Y_theta(i,gamma) = Im(z_theta,end)
                    = Im(w_i,gamma)-theta^T c(Re(w_i,gamma)). (8)

The readout is never transformed with h_theta. Consequently changing theta changes an observable answer in (8), not merely the label of a hidden coordinate. If one also transported that readout, all behavior would be a reencoding and the acquisition claim would disappear. The fixed-graph theorem is respected: the newly observed port/interface is outside its protected root-loop-only conclusion.

## 5. A learning law with a complete useful consequence

For observed new-task examples k, execute the supplied input/open path from its actually prepared branch and record the prediction in (8). Targets y_k arrive through the new-task endpoint feedback. The learner can compute its own canonical w=h_theta(z) from inspectable state; this is not extra environment information. Set c_k=c(Re w_k), b_k=Im w_k and

    L(theta) = (1/2n) sum_k (b_k-theta^T c_k-y_k)^2,
    theta_new = Proj_Theta[theta + eta/n sum_k
                         (b_k-theta^T c_k-y_k)c_k],     (9)

then perform (4). Old operations, target labels and input paths are not refitted after the update. Theta is a fixed coefficient box, discussed below. Feature evaluation, feedback, fitting and state migration are all charged.

**RECONSTRUCTION, exact restricted-family learning condition.** Suppose noiseless targets satisfy y_k=b_k-theta_*^T c_k for an admissible theta_*; let G=(1/n) sum_k c_k c_k^T. If G is positive definite and 0<eta<=1/lambda_max(G), then projected full-batch updates satisfy

    ||theta_next-theta_*||_2
      <= (1-eta lambda_min(G)) ||theta-theta_*||_2.       (10)

Before projection the error is (I-eta G)(theta-theta_*). Euclidean projection on the coefficient box is nonexpansive and fixes theta_*, proving (10). A held-out target in this same feature family obeys

    |Y_theta-Y_*| <= ||c(Re w)||_2 ||theta-theta_*||_2.   (11)

Thus useful function learning and complete old mathematical endpoint retention occur simultaneously over all update iterations. Useful directions require feature excitation, not parameter movement alone. Rank deficiency, incompatible target functions, noisy labels, an active box boundary or failed branch execution can block acquisition; no topological certificate repairs those defects.

**Full one-parameter arithmetic.** Choose phi=1, theta_*=1. Starting on +sqrt(3), follow the real open path from u=0 to u=-sqrt(2). That interval avoids both critical control values; its positive branch ends at w=sqrt(2), since w^3-3w=-sqrt(2). Here

    c = psi(sqrt(2)) = -sqrt(2)/9,
    Y_theta = theta sqrt(2)/9,
    y_* = sqrt(2)/9.

At theta0=0 the new answer is wrong. With eta=81/4, (9) gives theta_next=(theta+1)/2: theta is 1/2, 3/4, 7/8, ... and new squared loss falls by a factor 1/4 each step. Equation (4) physically moves y by the corresponding nonzero increment. Every old based-loop answer at all three anchors remains exactly its original value throughout. The new effect is an order-one fraction of the specified output scale, not a tiny sensitivity-proxy improvement. This is manual arithmetic, not an executed learning result. The larger rank-p statement, not the single datum, specifies the prospective continuous task family.

Preparing branches uses retained old paths or charged initial branch access. Return from an open-path sample uses the actual inverse path under the new parameters, after (4) has retained its canonical branch. An independently solved endpoint is never inserted as execution state. Reading several branch/input examples costs those several preparations and traversals.

## 6. Why this is not yet the total-cost answer

The construction deliberately makes a useful safe direction explicit before attacking it. Its fixed-core factorization is also a serious rival explanation, not an objection based on universal simulation.

### Bounded conditioning and precision are retention costs

Take the coefficient box |theta_j|<=B/p and |alpha_j|<=S. Direct differentiation gives

    |psi|<=4,   |psi'|<=24,
    |f_theta|<=4B,
    |f_theta'|<=(24+4S)B=:M.                            (12)

For |x|<=1 the numerator bounds suffice; for |x|>=1 divide the degree bounds by the denominator's leading power. These are intentionally coarse global inequalities. In particular

    ||Dh_theta||_2, ||Dh_theta^-1||_2 <= 1+M,
    cond_2(Dh_theta)<= (1+M)^2.                         (13)

The exact critical control set is unchanged, but root/state numerical error can be amplified. If canonical endpoint error is delta, physical z error is at most (1+M)delta, before feature-evaluation error. To require physical error tau one must allocate delta<=tau/(1+M) and bound feature arithmetic too. The corresponding extra precision is at least relevant at the scale log2(1+M), rather than being free. Branch labels remain reliable only if the resulting total error stays below the fixed anchor-separation/readout margin. An ideal all-word proof supplies no uniform finite-length tracker theorem.

Box clipping costs O(p) per update. The invertibility/anchor proof requires no new discriminant search or loop replay per step **for this fixed audited architecture**. That is a certificate by construction, not a claim that arbitrary model edits need no certificate. Changing P, the anchor factor, observation ports or architecture exits the certificate.

### Cost vector and amortization boundary

Let U be learning updates, N feedback examples, V the total number of tracked open/old/reset steps and C_phi(p) the feature evaluation cost (O(p) for the displayed features). A full account contains:

| Component | Actual obligation in this construction |
|---|---|
| Initial mechanism | Acquisition/source of P, branch anchors and old path library; their original construction and critical/path certificate costs remain common inherited costs, never declared zero |
| New representation | Construct psi, choose/store all features and coefficient bounds; O(p) parameters here; general anchor-factor cost grows with the fiber description |
| New evidence | N target labels, actual branch preparation and open-path/inverse executions; no free all-branch observations or canonical-root oracle |
| Training | O(Np) per full pass plus bounded coefficient projection; forming G costs O(Np^2) if one actually uses/checks it, and an eigensolve or lower-bound computation is additional work; condition (10) is not a free empirical certificate |
| Shared migration | Two f evaluations and the state change (4) per parameter update; no anchor migration in the exact family |
| Retention | One architecture/anchor/old-path argument, O(Up) box membership/projection; numerical error bounds, rejected tracking and extra precision remain charged |
| Execution | Canonical continuation work plus coupling/derivative evaluations when using actual F_theta; O(V C_phi(p)) coupling work in a straightforward fully materialized executor |
| Storage | P, paths, anchors, theta/features, feedback/cache/optimizer state and precision; retaining the N examples costs O(Np) if feature rows are cached, O(N) input/output records plus reevaluation otherwise |
| Failure | Rank-deficient or incompatible feedback, parameter-box saturation, tracking failures, rejected samples and retries all count; no candidate/path sweep is hidden in the feature choice |

There is no measured runtime, FLOP, RSS or energy result. If scalarizing these counts, weights and workload frequencies must be specified prospectively. In particular avoided old replay is not automatically a win: frozen-module strategies can avoid replay too. The coupling cost can be paid on every old execution long after learning, whereas a once-only replay/certificate expense may amortize differently.

## 7. Strong same-information alternatives and an exact economic boundary

**Same-mechanism implementation.** A source-aware continuation program retains w, executes the fixed canonical P, and evaluates h_theta^-1 when a z output is needed. It uses exactly the same owned source, features, targets, old anchors and branch/preparation access. Its gradient on (8) is exactly (9), and its finite state migration is exactly (4). This is an explicit finite transformation of the program, not an appeal to universal simulation. It establishes that storing the entire monolithic F_theta does not create an additional learning law.

**Independent modular strategy.** Retain the old learned operation implementation unchanged, and train an ordinary scalar adapter on the canonical state exposed by that owned implementation for the new continuous task. The same feature rows and targets suffice; direct least squares or ordinary projected gradient is allowed. Store no task-specific copy of P. Under an interface that observes z only at the old resting endpoints and the new endpoint, this strategy can omit all unobserved intermediate coupling evaluations. It can match (8) while using no more representation and potentially less execution work. If only symbolic old answers matter, it can additionally use the already acquired old permutation tables. Give it that legitimate optimization; requiring continuous physical z at every intermediate step solely to exclude it would manufacture an advantage.

The modular strategy and the fully materialized shared executor have different causal organization: the latter actually changes old intermediate z trajectories; the former does not need those trajectories to meet the declared observation contract. They nevertheless have identical observable answers in this construction. Thus a claim of lower total digital computation cost for the materialized shared executor is not supported, and the avoidable coupling overhead is a concrete adverse boundary. An actual continuous actuation problem could make those trajectories consequential, but that would require an independently motivated task/interface, not a retrofit to protect this proposal.

**Primitive prior work.** [Real NVP, §§3.2–3.5](https://arxiv.org/html/1605.08803) supplies learned affine coupling layers, analytic inverses and compositions. The shear (1) is its unit-scale special case. [Invertible Residual Networks, §2](https://proceedings.mlr.press/v97/behrmann19a/behrmann19a.pdf) gives another invertible parameter family and explicitly charges iterative inversion. Neither source is being credited with our anchor/old-word calculation, and neither is being claimed absent from the wider literature. No new invertible-network primitive is proposed. Exact prior coverage of the complete semantic-preservation application has not been established by these two readings.

**Ablations have narrow meaning.** Removing the anchor factor permits changed old numeric endpoints even if branch semantics could be transported. Removing invertibility can destroy the covering. Both distinguish mathematical roles in this construction; defeating such ablations would not establish an advantage over the modular alternative.

## 8. Protected acquisition is not indefinitely extensible for free

The new analog outputs are **not automatically protected** by the old topological certificate. A later theta update can change them freely. If the next task treats these outputs as old capabilities, then this same chart needs additional constraints, data retention/replay, a more informative invariant, or more capacity. Counting only the original discrete library while silently allowing acquired analog capabilities to be forgotten would not answer the human's continuing-acquisition question.

Likewise, if a new task requires a root-loop permutation outside the old monodromy action, equations (5)–(6) make acquisition impossible in this chart. If it requires an intermediate output outside the feature family or box, the best attainable residual can remain positive. These are specific failures of this representation and protected set, not evidence against topology MAIN. The exact finite learning calculation merely proves that useful shared plasticity is not logically excluded by full old loop preservation once fixed additional observations matter.

The prospective meaningful success beyond the arithmetic example would be a substantial family of branch-conditioned continuous computations learned from task evidence while a previously acquired noncommuting operation library retains its fixed original interface, with complete retention/acquisition/execution cost competitive with the modular strategy. Merely fitting the one coefficient or showing old loops again would not test that claim.

## 9. Consequential unknown and one useful next decision

**OPEN:** Can a useful continuing task sequence admit a cheap topological semantic invariant while retaining meaningful shared functional plasticity that does not reduce, at the stated observation interface, to a cheaper frozen-core/endpoint-adapter computation? This construction does not resolve that issue. It makes both sides precise: the useful safe learning rank is rank(G), the exact protected behavior is (6), the unprotected new behavior is (8), and the current factorization allows an explicit cheaper-or-equal modular route.

For the coordinator's present topology development window, the next useful action from this support note is **a representation/interface diagnostic on the selected main mechanism**, not an experiment on this cubic: identify a candidate parameter direction, its actual useful observable derivative, and whether an equally informed retained-core implementation can defer that direction entirely to an endpoint adapter. If it can, count the deferred implementation and ask whether the claimed advantage concerned actual continuous execution for independently justified reasons. If it cannot, derive the specific compositional/trajectory dependence that prevents deferral. This provides a concrete cost and identifiability check for a stronger topology construction without another seed/path/degree pilot or a demand for a prior benchmark win.

Provenance: this context personally read the original topology candidates and Reviews011–013 before construction. Before completion, root described its new fixed-idle-encoding/pinned-controller quantum construction, its Duhamel bound and K's dissipative phase-pump/faster-execution idea. Those are pre-freeze exposures, not source bodies personally inspected here, and no independent-blindness claim is made. They are not needed for equations (1)–(13). The construction uses familiar neural coupling primitives and makes no historical novelty claim.

CLAIM | STATUS | COVERAGE | EVIDENCE | WEAKEST LINK | NEXT SINGLE DECISION
--- | --- | --- | --- | --- | ---
Useful shared parameter learning can coexist with full old loop endpoint retention | RECONSTRUCTION, mathematical restricted-family result | DIRECT for this existence claim | Exact transition/migration, conjugacy/anchor proof, rank condition and scalar worked process | Fixed new port and realizable/excited feature family; numerical executor untested | Use the invariant/observable diagnostic on the selected mechanism
This provides favorable total continuing-acquisition cost | OPEN; not established in this family | PARTIAL | Full cost ledger and explicit modular deferral route | No protection of newly learned analog outputs; factorized rival can avoid old-step coupling work | Do not treat this construction as a positive resource result
Fixed-cover continuous learning acquires a new root-loop permutation | Excluded by the actual parameterization | OUT_OF_SCOPE as a goal; false if claimed here | Equation (5) with anchor fixing | None within stated family | Change the stated representation/protected interface only for a justified new subquestion
Historical new principle or neural advantage | UNESTABLISHED | OUT_OF_SCOPE | Coupling originals personally entered; no broader novelty finding | Exact application literature and consequential task/resource evidence absent | Preserve result and boundary; no automatic MAIN change
