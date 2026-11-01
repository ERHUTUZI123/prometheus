# Exact error projections, physical locality, and the missing benefit mechanism

**ANALYTIC SUPPORT / SOURCE-LIMITED PHYSICAL HYPOTHESIS, 2026-11-01.** Same V2. MAIN=TOPOLOGY / USER_LOCKED / HUMAN_ONLY_CHANGE. Online USER-PARK; Persistent State standalone paused/reserve. M0 is active and NOT PASSED; M1–M3 are later milestones and M3 remains the final evidence ambition. Provider-effective mode unknown. No numerical execution, code, shots, training, installation or resource change. Existing notes are left unchanged. Root's current pilot remains the frozen memoryless/no-other-noise pilot; this note does not add noise to rescue its outcome or decide the next research allocation.

## Judgment

The concrete distinction is a **projection selection rule**, not a blanket claim that a topological controller is robust. For the gapped star Hamiltonian, every infinitesimal change of its center–outer coefficients has scalar projection into its instantaneous logical ground code. An outer–outer perturbation can instead project onto a logical Pauli. Ideal counterdiabatic control is a special transverse outer–outer combination with zero such projection; an arbitrary error in that control need not be transverse.

This supplies a causal mechanism worth examining: change shared physical controls in directions that avoid direct logical energy splitting, while task feedback selects a holonomy. It does not itself retain an old holonomy, prevent transitions, or beat conventional robust holonomic/direct control. There is also no proof that an extra noise process is logically necessary for a difference in the current pilot: its gap already filters some transverse control errors, an effect absent from a triangle Duhamel score. What is missing is an important, cost-complete advantage from this or another specific mechanism against the admitted strong rivals.

The smallest physically meaningful extension would be a **common microscopic perturbation-to-actuator model** showing which error projections are suppressed, which are exposed when a logical coupling is activated, and the spectral/time regime where suppression survives execution. Merely adding an arbitrary noise channel to the direct arm is not that model. The original sources below establish physical locality/gap protection in their stated models; they do not establish its strength throughout this proposed six-mode learner.

Connection to the Fundamental Question: these are candidate physical directions in which a shared learned implementation could remain plastic while old executable operations survive. The learning claim additionally needs task-driven operation acquisition, fixed old holonomy/accuracy contracts, and a resource advantage. Encoding protection alone answers none of those learning questions.

## 1. Independent check of root's uniform finite bound

Actually read the entire `root-uniform-finite-bound.md` and `root-uniform-chart.md`, alongside the previously read mechanism/calibration/six-mode notes. The following is an independent by-hand reconstruction, not a numerical verification.

- A two-rectangle half-loop has four north/equator meridian sweeps. Each contributes `pi/4` correction area. Its equator widths, transformed by a feasible nominal `X`, sum to `a` because both moments vanish. Hence `A_half=pi+pi*a/4`.
- Two half-loops and two physical X calls, each X requiring two three-edge exchanges, give `A_echo=2A_half+4(3pi/4)=5pi+pi*a/2`.
- `max P1=1/4`, `max |P2|=sqrt(3)/18`, and `max |P1'|=max |P2'|=1`. Thus root's `F=(pi/2)(delta1/4+sqrt(3)delta2/18)` is a valid angular-difference bound.
- A meridian correction axis changes by Euclidean distance `2|sin(delta_phi/2)|<=|delta_phi|`. Eight sweeps give at most `2pi F`. Two half-loop equator contributions give `pi*a*(delta1+delta2)/2`; four exchanges give `pi*(delta1+delta2)`. These reproduce root's geometric reference-mismatch terms.
- The three outer bilinears of an active Y anticommute and square to I. A componentwise relative response envelope `e` therefore gives exact instantaneous residual norm at most `e ||h_cd,hat||`, and its integral is at most `e A_echo`. It does not require a vector-gain fiction.
- A full direct phase gate with `beta=pi*a` needs idle area `A_direct=beta/2=pi*a/2` and no c1/c2 estimate. A valid uniform inverse-response envelope therefore gives its all-target response score `A_direct e`.

**Result:** root's two displayed sufficient bounds and the direct uniform support check out under the stated static geometry/response epoch, monotone feasible nominal X, actual componentwise inversion, complete paths/echoes, and implementation/error allowances. There is no identified correction to send back. Uniform contract coverage follows on the SAME calibration event; it is not an independent event for each new neural update.

Two scope qualifications matter. First, a c(t) changing during a gate adds derivatives and destroys the fixed-coefficient moment premise unless separately bounded. Second, all of these are sufficient norm scores. A larger area or looser score does not order true errors, especially when an error is transverse to a gap. That distinction belongs inside the frozen model; it is not a positive pilot-performance claim.

## 2. Exact instantaneous projection in the actual six-mode code

Use `{gamma_i,gamma_j}=2 delta_ij`, fixed total parity, the active outer triple `(gamma1,gamma2,gamma3)`, and spectators gamma4,gamma5. For a real unit vector n, write

    gamma_n=sum_i n_i gamma_i,
    S=i gamma0 gamma_n,     H0=g S,     g>0,
    P=(I-S)/2

within the fixed-parity block. P has rank two; the gap to the other block is `2g`. Choose orthonormal real tangent vectors t1,t2 with `(t1,t2,n)` positively oriented. The two active zero Majoranas are `gamma_t1,gamma_t2`. Define

    Omega=i gamma_t1 gamma_t2.

Omega commutes with S, squares to I and is a nontrivial logical Pauli on the ground code. The spectator modes make this a genuine logical qubit at fixed TOTAL parity; the isolated four-mode active Y is not being counted as such.

### Star-channel changes have no first-order logical splitting

Decompose a real center–outer perturbation as

    delta Delta=delta g n+v,       v dot n=0,
    delta H_star=i gamma0 delta Delta dot gamma
                =delta g S+i gamma0 gamma_v.

The transverse term anticommutes with S. Therefore

    P delta H_star P= -delta g P,                         (1)
    P(i gamma0 gamma_v)P=0.

This is exact algebra for any real perturbation vector; it is a statement about its projection, not an exact statement about the finite-time propagator. The transverse term has nonzero `Q delta H_star P` and can excite the other block. A static change remaining entirely in the star family still has exactly degenerate logical ground energies, but its projector and subsequent holonomy generally change.

### General outer-channel changes can split the logical code

Write

    H_outer=h12 i gamma1 gamma2+h13 i gamma1 gamma3
                              +h23 i gamma2 gamma3,
    b=(h23,-h13,h12).

The wedge component wholly inside the tangent zero-mode plane is `(b dot n) Omega`. Components containing gamma_n connect the gapped pair to a tangent zero mode and have zero P–P projection. Consequently

    P H_outer P=(b dot n) Omega P.                        (2)

At idle `n=e3`, `P(i gamma1 gamma2)P=Omega P`. Thus a direct idle pulse `h12 Z` deliberately creates the logical relative energy whose integral implements the phase gate. An uncertainty in that coefficient is directly a logical phase uncertainty, even though the code remains invariant and parity is conserved.

Equations (1)–(2) are an encoding/gap selection rule. They are not new topology theorems, and a dark-state holonomic construction can have analogous rules. The topology/locality question is whether a real device keeps its dominant perturbations in, or close to, the scalar-projection sector while permitting the required paths.

## 3. Counterdiabatic control must be separated into its actual error components

The supplied convention gives

    h_cd,ij=(n_j dot n_i-n_i dot n_j)/2,
    b_cd=-(n cross dot n)/2.

Therefore `b_cd dot n=0` and

    P Hcd P=0.                                           (3)

It would be false to say that every activated outer counterdiabatic term necessarily creates an instantaneous logical splitting. Ideal Hcd is exactly transverse to the instantaneous zero-mode plane. Its purpose is to cancel transport leakage of the moving projector. Inaccurate Hcd is still not topologically protected merely because (3) holds for the ideal expression.

For an actual outer residual `delta b`, the two distinct quantities are

    logical component:       n dot delta b,
    leakage component:       delta b-(n dot delta b)n.     (4)

A common scalar error `delta b=epsilon b_cd` has zero first-order logical projection by (3), although it reintroduces finite-time transition error. An independent-channel error, wrong geometric axis, delay, or componentwise nonlinear response can create the first component in (4).

For example, for uncompensated common componentwise cubic response on a fixed n,

    b_actual=kappa0 b_cd+kappa1 (b_cd)^(componentwise 3),
    n dot b_actual=kappa1 sum_i n_i (b_cd,i)^3.            (5)

The right side is generally nonzero even though `n dot b_cd=0`. On a meridian, with `n=(sin(theta)cos(phi),sin(theta)sin(phi),cos(theta))`, it is proportional to

    sin(theta)sin(phi)cos(phi)[sin²(phi)-cos²(phi)]
                                      (dot theta/2)^3.

Nominal inversion removes (5) only for the nominal plant; the retained residual still needs its actual projection/error bound. If the geometry itself fluctuates, use the ACTUAL n: to first order the longitudinal mismatch includes both `n dot delta b` and `delta n dot b_cd`. An exact counterdiabatic correction of the actual fluctuating path would keep their sum zero, but the current controller is not given that path for free.

This is why the honest conclusion is more specific than either “all CD is unprotected” or “the gap protects CD”: its dependence on the actual trajectory and the longitudinal/transverse decomposition decide the error. Root's Duhamel bound correctly covers both, conservatively. It does not exploit rapid relative phase accumulation across a gap to reduce transverse effects. In a slow limit that dynamical distinction can already matter inside the current memoryless model, subject to time/action and direct-composite competition; it is not an observed pilot result.

## 4. What original microscopic physics actually supports

These are targeted original readings during this support allocation, not an exhaustive survey and not hardware evidence.

**SOURCE — Kitaev, *Unpaired Majorana fermions in quantum wires*.** Personally entered [original body, §1, Eqs.6–15](https://arxiv.org/html/cond-mat/0010440), also [PDF pp.4–8, with Eq.15 on p.8](https://arxiv.org/pdf/cond-mat/0010440). Its local chain has unpaired end Majoranas at the dimer point. Away from that point, its boundary-mode solution gives `H_eff=(i/2)t b' b''`, with t exponentially small in wire length when the bulk remains gapped. This is an actual microscopic separation/gap protection law, not a theorem about arbitrary driven control hardware.

Here is a direct elementary reconstruction of the local selection rule from that chain. At `w=|Delta|>0, mu=0`, set `D_j=i c_(2j)c_(2j+1)` for `j=1,...,L-1`. The Hamiltonian is `w sum_j D_j`, and the ground projector P_chain fixes every `D_j=-1`. A local density perturbation is

    delta mu_j (n_j-1/2)=delta mu_j i c_(2j-1)c_(2j)/2.

For every site in a chain of length at least two, this bilinear anticommutes with at least one fixed dimer D_j. Hence its ground-space projection is zero. More generally, a parity-even operator of support unable to connect BOTH end zero modes reduces, after projection onto the fixed dimers, to a scalar or zero. A term proportional to `i c1 c_(2L)` acts nontrivially but spans the separated ends. The exponential-tail result quantifies the finite-overlap correction in the source's model. To encode a usable qubit at fixed total parity requires four end modes, e.g. two such chains, not the opposite-parity doublet of a single chain alone.

This establishes a physical basis for **local, parity-preserving perturbations and separated modes**. Parity-changing poisoning, gap closure, perturbations connecting ends, and deliberate activating links are outside that simple protection statement. Their susceptibility cannot be set to zero by invoking Kitaev's chain.

**SOURCE — Alicea et al., *Non-Abelian statistics and topological quantum information processing in 1D wire networks*.** Personally entered [§I, the local gate/gap discussion after Eq.7; supplementary §IV.2.1](https://arxiv.org/html/1006.4395). It explicitly distinguishes local gated transport maintaining a finite gap from a uniform gap-closing switch, and assumes well-separated zero modes when constructing its degenerate ground states. This supports the need for real geometry/separation and a gap during transport. It does not certify our entire finite six-mode control family against a generic bath.

**SOURCE — Karzig et al., *Shortcuts to non-Abelian braiding*.** Personally entered [pp.201102-2–3, Eqs.15–17, practical implementation and Robustness paragraphs](https://www.physik.fu-berlin.de/en/einrichtungen/ag/ag-von-oppen/papers/karzi15a.pdf). The adiabatic star and additional outer shortcut couplings are distinct controls; the latter depend on the actual path for exact cancellation. Its discussed overlap/gap, gate and quantum-dot controls provide physical actuator examples. They do not establish a universal low noise price for those controls. Equations (1)–(5) above are our projection reconstruction, not claims imported from its numerical plots.

**SOURCE — Karzig et al., *A universal geometric path to a robust Majorana magic gate*.** Personally entered [§III, §IV Eqs.16–17 and §V; §VI.3; Appendix D](https://arxiv.org/html/1511.05161v3). This source's boundary restrictions support robust geometric cancellation; its interior all-arm couplings also generate parasitic tip couplings of scale `Delta_i Delta_j/Delta_SC`. Its echo requires sufficient repeatability, and time variation limits cancellation. Thus the fractional gate lacks the full protection of an exchange. No source numerical noise amplitude or claimed performance is imported as a measured prior here.

The microscopic support is real but limited: separated modes can suppress particular projected local perturbations. **OPEN:** which physical layout/control law realizes that suppression throughout each learned interior path, including spectators, routing, switching, induced outer couplings and readout? The abstract six-mode Hamiltonian alone has no spatial coordinate, locality metric, bath or noise spectrum from which to answer it.

## 5. The smallest useful next physical interface, without a fabricated win

A defensible extension starts with ONE common microscopic control/noise parameter p_r per actual physical source, not independent penalties assigned by algorithm name. To first order,

    H_noise(t)=sum_r xi_r(t) O_r[lambda(t)],
    O_r=partial H_eff(lambda,p)/partial p_r.

The same p_r process and the same operator-to-actuator map apply to direct, composite, adiabatic, counterdiabatic and UGD programs. In the retained active block one may decompose a specified O_r as

    O_r=i gamma0 v_r dot gamma+H_outer(b_r)+O_spectator,r,
    P O_r P=alpha_r P+(n dot b_r) Omega P
                              +P O_spectator,r P.          (6)

The first term is common phase. The others and `Q O_r P` must be bounded, including cross-correlations and any dependence on an activated mediator. If a logical outer link is turned on to generate a direct phase, its susceptibility can be a nontrivial logical operator. The SAME susceptibility is charged whenever another method uses that link. Conversely, a mediated link may be physically local while connecting two modes of the logical encoding; the word “local” alone does not imply scalar projection.

The actual minimum missing evidence is that this common map yields a **useful protected operating region**: physically justified small longitudinal logical response and controllable leakage while the holonomic program still acquires the demanded operation within full resource bounds. A low-frequency or slow-drift label alone is insufficient. One needs the relevant spectrum/rate and a window such as transport slow relative to `1/(2g_min)`, repeatable over the echo, yet short relative to poisoning/irreversible memory loss. This is not a known admissible window for the proposed device; no numerical times or noise strengths are supplied here.

For orientation, a single direct Z pulse with a small commuting susceptibility `chi_D(t)` has the exact phase-area error

    delta A_D=integral chi_D(t) xi(t)dt.

The corresponding first-order holonomic logical error uses the transported projection `chi_T(t)=n(t) dot b_r(t)`, together with any spectator contribution and the ACTUAL echo sign. For one stationary classical noise source the variance of either such scalar first-order term is

    Var(delta A)=integral d omega/(2pi) S_xi(omega)
                  |integral y(t)chi(t) exp(i omega t)dt|².  (7)

Equation (7) is a conditional linear-response calculation, not a physical estimate. Noncommuting leakage needs its gap/toggling-frame calculation rather than this scalar replacement. A direct composite has its own nontrivial toggling-frame filter; it cannot be evaluated as an uncorrected single Z pulse. In particular, slowly varying common gain is precisely a regime where conventional composites can cancel errors. The protected arm also loses its simple static moment/echo cancellation under sufficiently fast geometry drift.

A comparator is free to avoid noisy direct links and use the SAME gapped holonomic paths, published UGD, analytic moment chart or optimized adiabatic schedule. Therefore, even a verified topology-versus-dynamical-control advantage would establish value of that physical organization, not an advantage of a neural learner over UGD. To address M0, the actual joint learning/retention mechanism must still beat that source-defined conventional controller on a relevant complete cost or capability dimension.

## 6. Learning freedom and the precise failure to avoid

Equation (1) is not an old-operation-retention certificate. A closed path can change its Berry area while having scalar projected Hamiltonian variation at every point. In the present coordinates its logical angle changes with the enclosed area. Protecting old operations therefore requires BOTH:

1. Physical error/learning directions whose relevant instantaneous projections and leakage are controlled.
2. A path-level holonomy constraint, such as the already verified moment chart, and the actual finite-time/error/deadline contract.

The first can reduce a physical precision burden; the second fixes which operation is retained. New task information chooses a new area/program rather than emerging from the scalar projection identity. Shared shape/radial/timing updates may remain possible, but their useful freedom, acquisition cost and conventional comparators must be measured or derived. It would be a false inference to turn `P delta H P proportional to P` into automatic semantic retention or continual learning.

## 7. Necessary interfaces for a trainable, scalable computational architecture

The latest human constraint, delivered by Root, makes the boundary explicit: quantum control is a research vehicle; the final contribution must explain a trainable/scalable general computational architecture with important operation-acquisition/retention value. The following are **independent conditional deductions**, not additional source claims and not a completed architecture. They specify what a transferable mechanism would have to solve.

### Error protection rank and semantic plasticity rank are different

Let a candidate have shared trainable coordinates `theta in R^d`, retained executable programs `p_j`, and actual semantic operators `U_j(theta,p_j)` on a declared state/code space. In a local coordinate chart for operators, quotienting only the allowed common phase, stack the derivatives of the OLD semantics into `J_old`. At a regular point of fixed rank r, first-order semantic-preserving shared updates lie in

    K_old=ker J_old,       dim K_old=d-r.                  (8)

Physical feasibility, old error and resource contracts can restrict this set further. If a proposed first-order learning mechanism needs q independent new semantic coordinates from SHARED updates, the restriction of its new-task Jacobian to `K_old` must have rank at least q. If it is zero, its first-order shared update cannot supply those coordinates. Structural program growth, additional parameters, discrete changes or higher-order paths can evade that LOCAL obstruction, but their actual rule/cost must be specified; equation (8) is not a global impossibility theorem.

In contrast, the instantaneous physical error map is

    L_err: xi -> [P(sum_r xi_r O_r)P] modulo scalar P.

The star selection rule makes this map zero for its specified center–outer errors. It says nothing by itself about r in (8): holonomy can vary even when every projected Hamiltonian derivative is scalar. A general architecture must keep both maps under control. Confusing low `rank L_err` with many useful directions in `ker J_old` would repeat the original retention error at a more abstract level.

If every acquired operator imposes independent new constraints on a fixed d, the useful tangent dimension can be exhausted. A scalable proposal must instead exhibit dependent/factored semantic constraints, structural growth with accounted cost, or another explicit way to retain useful acquisition directions. A uniform constructor can make `J_old=0` for arbitrarily many targets; the existing direct area-normalized controller already has that property in its ideal phase family. Thus uniform invariance alone does not distinguish the proposed learning mechanism.

### Semantic information must accumulate somewhere

For the current class of classical controller/program records, suppose N independently acquired target coordinates each lie in a fixed interval of width D and must later be recalled to error at most epsilon. A grid of spacing `3epsilon` has `M=floor[D/(3epsilon)]+1` sufficiently separated targets and gives `M^N` distinguishable histories. Deterministic worst-case recall from a retained finite-precision classical state therefore needs at least `N log2 M` bits. No topology argument eliminates this semantic information requirement. This is a counting reconstruction for the stated record class, not a general quantum-memory theorem or a sharp bound for stochastic recall.

Sublinear growth is possible only with stated structure, e.g. targets share a learnable rule or programs compose previously acquired operations. A scalable architecture must show that structure actually learned and reused, rather than assume an ever-growing supply of bespoke task rules. Raw parameter count, extra turning points and more named library entries do not establish capability growth. Storing one scalar per learned phase is cheap and legitimate for both direct and geometric controllers, but it is not by itself a general computational architecture.

### Calibration dimension must follow reusable physical primitives, with word costs exposed

Protection could be valuable if a uniform physical certificate depends on a bounded set of genuinely shared primitive susceptibilities instead of separately estimating every acquired program. The same opportunity belongs to a conventional controller. If new primitives/couplers appear, their independent logical and leakage responses cannot be transferred from old observations without an actual shared law. A regular r-dimensional identifiable nuisance family requires probe likelihoods with sufficient derivative rank to resolve those r directions, unless a valid robustness argument makes the unresolved directions irrelevant. A low-rank projected error map helps only if it also covers changing code/projector geometry, leakage and the actual composed programs uniformly.

Even a constant-dimensional primitive uncertainty set can demand increasing precision. For a word of L unitary calls, telescoping gives `||U_word-V_word||<=sum_l ||U_l-V_l||`. A sufficient equal primitive allocation is therefore of order `epsilon_word/L`; simple finite-shot mean estimation can then require order `L²` shots per primitive, aside from confidence factors. This is a sufficient-prescription consequence, not an optimal lower bound. A proposed architecture must account for composition length, precision, drift refresh and rejected updates, rather than infer constant maintenance from a fixed parameter count.

### A meaningful transferred mechanism must satisfy all three interfaces

The candidate would need: a code/state representation with physically or mathematically justified small harmful error projection; an explicit path/program parameterization leaving useful shared learning directions after old semantic/resource constraints; and a formation rule that uses task feedback to acquire and compose new executable operators with scalable information/calibration cost. These interfaces can be expressed without Majoranas, but generic relabeling is not a transfer result. They must be instantiated on richer computations and compared against ordinary constrained optimization, modular programs and robust compilation with equal information and resources.

**Root-reported pilot outcome, not independently data-checked in this support task:** three new target gates were acquired/retained in the frozen finite model, while direct control was cheaper and UGD matched; M0 remains unpassed. This note neither alters that pilot nor claims that its new physical hypothesis overturns the outcome. A next development question must earn its significance through one of the concrete physical/plasticity/calibration interfaces above, not through more control tuning.

**Status/next decision:** the uniform finite bound checks out; the projection decomposition is VERIFIED algebra in the stated six-mode model; locality/exponential splitting has SOURCE support for specified microscopic wire models; transfer to the whole learned device/noise interface and to a general computational architecture is OPEN. The completed pilot retains its frozen conditions and its actual evidence boundary. No new experiment or alternative route is authorized by this support note.

`CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

`Protected versus exposed perturbation directions as a candidate physical retention mechanism | conditional projection law verified; full benefit/architecture OPEN | equations (1)–(8), original microscopic sources, checked root bound | common device-level perturbation map, nonexhausting useful plasticity and complete cost against direct/composite/UGD | blanket CD-splitting claim and algorithm-specific noise exemption rejected | Root to choose one consequential projection/holonomy/plasticity uncertainty for the next in-scope development segment, preserving the completed pilot's result.`
