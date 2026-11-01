# BF v1: preserve a semantic law while learning its execution geometry

Frozen constructive reconstruction, 2026-10-31. AGENT PROPOSAL / RECONSTRUCTION, not a novelty claim. Inherited current user-authorized Ultra route; the tool surface exposes no independently reliable backend/effort attestation. No Pro claim. Only the assigned B and F frozen artifacts/source records, the run contract, and the assigned role paragraphs were read. No coordinator proposal, A/C/D/E, old architecture report, or B-independent-baseline was opened.

## Reconstruction before repair

B learns Fourier relations and turns them into periodic energy factors. Algebraic elimination creates an entailed factor. Entailment preserves the exactly satisfying configurations; it does not preserve the equilibria of the physical gradient solver. Its five-edge winding countermodel is exact. A winding here is a nonzero phase error distributed across an energy cycle. It is not a closed controller trajectory.

F learns a connection `dh=sum_i A_i(q)h dq_i`, stores controller loops, and composes their payload transports. Controller closure restores q; it does not retain the payload map when A changes. A loop here is an ordered control path, not an energy-factor cycle. In the frozen instance every geometric candidate has an identical cached action word, with identical scores and payload action. That equivalence must remain a baseline.

Their common missing interface is between **the relation/map counted as correct** and **the dynamics used to reach/execute it**. They do not supply a common physical cycle mechanism. The proposed repair therefore does not add a holonomy term to an oscillator energy.

## Protected repair attempt 1: genuine connection plasticity with exact old-loop retention

Use F's sign convention `dh=A_i(q)h dq_i`. Let an invertible learned neural field `G_psi(q)` obey `G_psi(q0)=I`. Set

`A_i^psi=(partial_i G_psi)G_psi^-1+G_psi A_i G_psi^-1`.

This changes the actual state-transition field. With `h'=G_psi(q)h`, direct differentiation gives its execution. Every path transport changes as

`H_psi(gamma)=G_psi(qT) H(gamma) G_psi(q0)^-1`.

Thus every based retained loop, not merely its trace or eigenvalues, retains its full action exactly. Primitive field weights can change substantially; exact semantic retention is possible without freezing each field weight. No Hessian inverse or task oracle is used. One finite parameterization is `G_psi(q)=exp(f_psi(q) N)` with a fixed nilpotent N and `f_psi(q0)=0`; the exponential terminates.

**Substantive attack:** This is a gauge change. It preserves *every* based loop action, including all candidate discovery scores on endpoint feedback. If the allowed paths, path costs, endpoint loss, and exact execution are unchanged, no new endpoint operation becomes possible or easier to discover by searching those same paths. A finite-difference learner using endpoint task feedback receives exactly zero signal for psi. If a benefit comes from numerical conditioning, intermediate readout, noise, altered actuator cost, or a changed path metric, that additional physical criterion must be specified and paid for. In particular a genuine condition-number improvement is not new functional reachability.

For noncommuting fields this statement is stronger than F's elementary determinant invariant and does not require choosing only one old composite to retain. But it does not escape cached-sequence equivalence. It is a successful exact-retention repair and a failed mechanism-discovery repair. No numerical identity check is needed to decide it: the displayed change of variables proves the full-path statement.

## Protected repair attempt 2: learn mobility, not another semantic potential

Retain B's observed relation memory `(C,delta,K,q,N)` and its energy

`E(theta)=-sum_m K_m cos(c_m theta-delta_m)`.

Instead of turning an entailed shortcut into an additional energy factor, construct a learned low-rank **mobility channel**. The execution law is

`theta_dot=-M_psi(theta) grad E(theta)`,

`M_psi(theta)=epsilon I+L_psi(theta)L_psi(theta)^T`, `epsilon>0`.

L can be a neural field or, in the finite instance, a learned constant vector. This is an actual change to the transition law. It changes execution geometry while the acquired relation parameters remain fixed. Its guarantees do not depend on the learner discovering a special vector:

1. `dE/dt=-grad(E)^T M grad(E)<=-epsilon ||grad(E)||^2`.
2. `theta_dot=0` iff `grad(E)=0`: no new equilibrium, correct or incorrect, is created.
3. At a critical state, the linearization is `-M H_E`; it is similar to `-M^(1/2) H_E M^(1/2)`. Consequently the inertia and nondegenerate stability classification are unchanged. Derivatives of a state-dependent M vanish from this linearization because grad E is zero there.

This is materially different from B's entailed potential. It repairs the precise failure: the new learned interaction can change transient accessibility without introducing the demonstrated new winding attractor. It does not remove old spurious minima already present in E. An inconsistent or wrongly learned relation remains wrong. Context/role allocation and useful-coordinate discovery are still unsolved.

### Constructing a channel without an inverse-Hessian oracle

For n free carriers, use a unit vector v and fixed `beta=1/2`:

`M_v=(1-beta) I+n beta v v^T`.

`trace M_v=n`, so the mobility trace equals the unmodified solver. It is strictly positive for every v. During successful ordinary parent-only executions, record force samples at predefined late times. For each nonzero sample f form `u=f/||f||`, then update

`v <- normalize(v+eta u(u^T v))`, `eta=0.2`.

This is normalized Hebbian/Oja-style online principal-direction estimation of slow residual forces. No desired output, eigensolver, inverse Hessian, backpropagated oracle, or choice of the best tested v enters the update. Late-force selection is designed and charged. For the stated symmetric chain linearization, late forces generically align with the slow eigenmode; it is a hypothesis that the finite nonlinear episodes learn a useful direction. All training trajectories and samples count. v is frozen during each held-out inference. Neural constant outputs can implement v; useful nonlinear metric learning is not demonstrated by that observation.

### Worked computation and exact safety comparison

Keep B's four-parent countermodel in deviations from a compatible acquired operation: `y0=0`, free `y1..y4`, `e_i=y_i-y_(i-1)`. Parent energy is

`E0=-sum_(i=1)^4 cos(e_i)`.

The relation matrix from y to e is invertible over the torus. Its critical states have each residual 0 or pi. Its sole local minimum is all-zero residual; every critical state with a pi residual has a negative Hessian direction. Therefore the mobility repair cannot create a stable wrong answer in this tree family, regardless of what v learning produces. Almost-all convergence uses the finite hyperbolic-critical-state gradient-like structure; this is not claimed for arbitrary factor graphs.

B's bad shortcut gives `E1=E0-cos(y4)`. At `alpha=2pi/5`, `y_i=i alpha`, it has the exact stable wrong output `wrap(y4)=-alpha`. Reweighting all five factors by 0.8 matches the simple total edge-strength budget of four unit parents and preserves this false attractor. The mobility law instead keeps E0: at that same state the first three gradient entries vanish and the fourth is `sin(alpha)>0`; positive definite M cannot leave it stationary.

For a real relational query, use acquired offsets d_i and clamped inputs to define compatible desired phases y*_i by their parent relations; the identical error-coordinate law applies to y-y*. No acquisition example is changed by the mobility update. For the illustrative y*=0 query, start at the old winding state plus `[0.02,-0.01,0.015,-0.005]`; parent-only and mobility dynamics should restore y=0 modulo 2pi, whereas the naive shortcut should retain the incorrect winding. This is an actual discriminating execution, not a transport identity check.

### Costs and baseline commitments

For n=4, a parent force needs four sine calls and six scalar additions/subtractions, excluding integrator operations. Applying the rank-one mobility adds 12 scalar multiplications and seven additions with `n beta v` precomputed. A force step has one four-carrier reduction and broadcast; this is nonlocal communication and not free. v stores four learned scalars; its norm and positivity are enforced by the parameterization. At general n the channel costs O(n) memory/arithmetic/communication and has reduction/broadcast latency. Neural state-dependent L would add its network cost.

Equal trace is a limited resource proxy, not matched wall time, power, or FLOPs. Report both number of actual force evaluations and operation-weighted costs with sine cost left explicit. Count all training forces, all Oja updates, and amortization. The tree solver is the same-representation baseline; the naive entailed shortcut is the unsafe consolidation baseline. There is no hidden optimization of beta, eta, horizons, or a selection over candidate channels.

## Finite verification contract, frozen before any probe

QUESTION: Can a directly learned positive mobility channel avoid the *specific* stable wrong computation introduced by entailed-potential consolidation, retain parent semantics exactly, and reduce held-out settling cost under transparent resource accounting?

GIVEN: The two reconstructed mechanisms, exact equations above, and their existing checked source boundaries. Strongest fragile premise: late residual-force Hebbian learning finds a useful direction cheaply enough to repay its overhead.

NON-GOALS: No novelty, new reachability, general continual learning, arbitrary-coordinate discovery, neural biological realization, efficient self-generated controller, or superiority over a learned preconditioner. No numerical gauge identity test. No reinterpretation of metric conditioning as a new correct computation.

VERIFIER: One inspected isolated script. Seed 20261031. Sixteen independent uniform[-pi,pi]^4 parent-only training starts; RK4 step 0.025, horizon 60; eight late-force samples per trajectory at evenly spaced times 25..60. One pass of eta=0.2 normalized Hebbian updates. Held-out: 64 fresh uniform torus starts, paired across original tree and learned mobility; fixed horizon 180; correctness/settling threshold max absolute wrapped parent residual 1e-6. Report failures/censoring. Separate exact bad-winding start plus stated perturbation under tree, learned mobility and five-factor shortcut (all strengths .8). Local linear rates may be computed for description but never enter channel learning. Validate the same central held-out claim at half step on the winding start and four preselected held-out starts, with no further learning.

BUDGET: Two genuine repair approaches (gauge-normalized connection; positive mobility), one exact-equivalence attack and the winding adversary; one numerical script with a 90-second outer timeout, no network, no parameter sweep, and only BF writes. Source retrieval is bounded to nearest primary precedents. STOP after this finite discrimination, including a negative resource result. No next experiment is implied.

OUTPUT: Frozen v1, source records, the small actual probe and compact certificate. The likely scientific outcome is narrower than a new neural architecture: a precise criterion for dynamics-faithful mechanism improvement, with a known-style safe construction and an unresolved discovery/retention benefit.
