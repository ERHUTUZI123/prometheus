# M0 candidate: task-driven formation of robust holonomic operation programs

AGENT PROPOSAL / SOURCE-INFORMED RECONSTRUCTION. Same V2, MAIN=TOPOLOGY / USER_LOCKED / HUMAN_ONLY_CHANGE. Persistent State remains standalone USER-PAUSED; Online remains USER-PARK. No execution, shots, code, training, installation or hardware action. **M0 is NOT PASSED.** M1–M3 are future gates, not parallel work or promised results.

This is the strongest concrete K operation-acquisition proposal after the explicit human correction. A stream containing only shorter implementations of old braids was abandoned before writing it. The candidate below forms new physical operation programs from quantum task examples. Its central unresolved issue is whether this organization adds important learning freedom or full-cost value beyond existing robust holonomic compilation and direct control. The proposed non-Clifford robustness scaffold is published prior work, not our invention.

## 1. New learned object, significance, and immediate limits

Retain the six-Majorana/fixed-total-parity model in `quantum-shared-control.md`: one idle gapped pair and four zero modes give one encoded qubit. Old routed exchanges implement a specified Clifford library. Changing only their phase/time profiles cannot acquire a non-Clifford gate.

The new acquired object is instead a closed **interior coupling path**, with a learned sequence of turning points and a shared learned execution profile. Its logical phase is inferred from actual task-state observations, not supplied as a braid word or turning-point recipe. The algorithm may grow the program by inserting new sweeps when a smaller program cannot meet its uncertainty/error contract. It retains that executable program, its task evidence and its certified semantic/error contract for later composition.

The important prospect is that boundary protection plus algebraic cancellation supplies a family of semantic-preserving changes to these programs and the shared controller. The physical paths may continue changing while previously learned non-Clifford transformations stay fixed within declared error. This is more than changing an old deadline, but it is not automatically a new learning principle: process estimation followed by robust geometric control is a very strong existing explanation.

No demand is made that a new program lie outside the universal closure of all physical primitives. Stage1 is genuinely outside the initial Clifford operation group; later stages acquire distinct native transformations, with ordinary compilation from the growing old library a mandatory competitor. A renamed context, a cached old word, or speed improvement alone does not count as a new transformation. We do not claim new physical controllability or universality from the stages below.

## 2. Existing physical family and actual extra assumptions

Use six modes gamma0,...,gamma5, fixed total parity, common idle H_idle=i*g_idle*gamma0*gamma3, and the routed active triples already specified by root. An active Y has

    H0=i*g*n·(gamma0*gamma1, gamma0*gamma2, gamma0*gamma3),
    |n|=1,       g_min<=g<=g_max.

Spectator modes are retained and counted. For a closed normalized coupling path, use polar coordinates vartheta,varphi on its positive octant and set

    x=2*varphi/pi,       y=1-cos(vartheta),       0<=x,y<=1.

In the chosen sign convention, a loop of signed square area a implements Rz(beta), beta=(pi/2)*a, up to an irrelevant common phase for **uncontrolled** gate calls. The full boundary gives the old exchange; fractional-area loops can give non-Clifford phases. This is geometric holonomy, not topological invariance of arbitrary interior paths.

SOURCE boundary: Karzig et al.2016 derive this area representation, explain the unprotected naive magic gate, and give geometric cancellation and a parity echo. The following finite response class, adaptive program-growth rule and learning/admission process are our proposed study, not a new attribution to that source.

Assume a static geometry response during a complete gate:

    X(x)=x+c1*P1(x)+c2*P2(x),       Y=y,
    P1=x(1-x),       P2=x(1-x)(2x-1),
    (c1,c2) in {-1/10,+1/10}².                              (1)

It has X(0)=0, X(1)=1 and X'>=4/5. This is a concrete boundary-pinned, monotone response class; finite observations cannot establish it against arbitrary unmodeled distortions. The prior is an assumption to test/justify, available equally to all controls. It is not obtained by calling topology a sensor.

Retain nonlinear shared correction-channel response

    f_kappa(q)=kappa0*q+kappa1*q³,
    kappa in {(9/10,0),(9/10,1/5),(11/10,0),(11/10,1/5)},
    |q|<=1.                                                 (2)

The response is odd and strictly increasing. A COMMON response across active correction channels is a strong transfer premise. If gains differ, duplicate the corresponding parameters/calibration configurations and costs; do not transfer one channel's observations by fiat. Gap-channel response is assumed known or separately bounded; it is not inferred from (2).

A time-stable parasitic logical-Z splitting during an interior loop can be handled by the charged echo below. Other Hermitian residuals have a declared norm bound r. Unknown splitting drift, nonunitary noise, SPAM bias, leakage outside the modeled modes and model mismatch need their own bounds; they are not silently included in a free scalar calibration envelope.

The counterdiabatic term remains Karzig's actual outer-pair coupling formula. With a distorted geometry its correct coefficients depend on the ACTUAL n(t), not just the requested n(t). Thus it is incorrect to combine an unknown distorted H0 with a perfectly known nominal Hcd and infer transitionless execution. Our admission rule explicitly propagates every retained geometry/correction hypothesis, including this mismatch. Alternatively a sufficiently slow ordinary controller may be competitive; no finite-time success is inferred merely from the area identities.

## 3. Program representation and structural formation

A primitive geometric program has 2N ordered turning points

    0<=x1<=x2<=...<=x_(2N)<=1.

It alternates downward/upward sweeps between y=0 and y=1, with boundary connectors and a common north-pole idle. One may equivalently execute the N disjoint rectangular loops [x_(2j-1),x_(2j)]. Direction is chosen so

    a(q)=sum_j (x_(2j)-x_(2j-1)).

Boundary connectors, pauses/smoothing and all sweeps are physical controls, with their duration, amplitude, slew and cost included. Under (1), the actual area is EXACTLY

    a_actual=a(q)+c1*M1(q)+c2*M2(q),
    Mk(q)=sum_j [Pk(x_(2j))-Pk(x_(2j-1))].                   (3)

Consequently the constraints

    Phi(q):=(a(q),M1(q),M2(q))=(a_target,0,0)                (4)

retain the ideal geometric operation for the whole stated response class. This is the useful robust representation. It does not remove finite-time/control errors or make arbitrary non-Clifford phases intrinsically topological.

### A worked formation step, not an unexplained constructor

Start a candidate with N=1. Area a requires x2-x1=a. Cancelling P1 requires x1+x2=1, hence x1=(1-a)/2 and x2=(1+a)/2. But then

    M2=a(1-a²)/2 !=0          for 0<a<1.                    (5)

A two-turn program therefore cannot satisfy (4). This is a specific structural trigger, rather than a general claim that more parameters will help.

Insert a collocated opposite-sweep pair (c,c) into the ordered program. At birth the inserted pair contributes zero to area and both error moments. Try c from the declared finite grid {1/4,1/2,3/4}, omitting collisions with existing turns, then open the pair by one of the declared small separations {1/64,1/32}. Solve the three moment equations using bounded projected Gauss–Newton/interval refinement in the ordered box. Each proposal, failed solve and certification is charged; an unsuccessful bounded solver is an actual algorithm failure. No exponential unspecified search or semantic oracle is hidden here. Root must freeze iteration/precision caps before any code.

There is an explicit positive four-turn witness for every 0<a<1. Let d=a/4 and

    m=[1-sqrt((1-4d²)/3)]/2,
    q=(m-d, m+d, 1-m-d, 1-m+d).                             (6)

These points are ordered and lie in [0,1]. Their area is a. Symmetry cancels M1. Since P2=-2x³+3x²-x,

    P2(m+d)-P2(m-d)=2d[-6m²+6m-1-2d²]=0,

so M2 also vanishes. This is our elementary reconstruction of a small robust contour, closely related to the source's published universal construction. It is not claimed as a new theorem. A direct controller may use (6) immediately; the candidate does not receive credit for making its iterative solver imitate that known answer expensively.

Program growth is accepted only when the smaller representation fails a required contract and the larger program passes the full finite-time/resource check. Creating turns alone is not capability growth. A successful bounded solver still has not learned the task's angle; that information comes from the observations below.

## 4. Acquiring an operation from task feedback

The task interface supplies independently prepared examples of an unknown Z-axis unitary, namely states Rz(beta*)|+>. The learner can choose paid logical X or Y measurement. It is not given beta*, a desired area, turning points, target Hamiltonian or teacher waveform. Preparation of these examples is a TASK ORACLE, with its number/time/cost exposed equally to every baseline. It is additional supervision, not supplied by the Majorana substrate.

The two observable probabilities are

    p_X=(1+cos beta*)/2,       p_Y=(1+sin beta*)/2.            (7)

Restrict beta* prospectively to [pi/10,2pi/5], avoiding a phase-wrap ambiguity. From n shots of EACH observable and per-stage failure allowance delta_j, Hoeffding gives probability intervals of radius

    eps_p=sqrt(log(4/delta_j)/(2n)).

Intersect their inverse images with the prior interval to obtain a target-angle confidence set B_j. An empty intersection is a recorded confidence/model failure. No narrowing comes from a desired outcome. A conservative width calculation is |beta_hat-beta*|<=4sqrt(2)*eps_p when the Bloch-vector error is small enough; the exact interval intersection should be retained. If this is inadequate for admission, purchase another predeclared batch with its own confidence allocation. A fixed number of target samples cannot identify an arbitrary angle exactly.

The angle record, not its midpoint alone, remains stored. Task-derived midpoint beta_hat determines the candidate area's target; uncertainty contributes at most half the angle half-width to the operator-norm error of Rz. It must remain in every later old-operation contract. Task data are genuine new semantic information; inspecting one's own graph does not replace them.

### Echo makes use of an old operation in forming the new one

Use the retained encoded Pauli X to cancel a static diagonal dynamical phase, preserving TOTAL fermion parity by using the spectator modes as required by the routed code. No forbidden flip of total parity is assumed. A logical X made from two exchanges consumes both exchanges and their errors; it is not a free primitive.

If a forward half-loop implements Rz(b+d_dyn), its reversed path implements Rz(-b+d_dyn) under the stated static splitting assumption. Execute forward half-loop, X, reversed half-loop, X. The resulting logical angle is 2b, while d_dyn cancels. Each half uses area

    a_target=beta_hat/pi,                                   (8)

in (4). This is an actual composition of an old operation with the newly formed geometric program. Both X calls, returning paths and shared-controller errors count. Nonstatic/noncommuting residuals are covered only by an explicit additional error bound, not by the echo slogan.

## 5. One continuing SHARED neural controller

State is (theta, uncertainty set U, retained task-angle sets B_j, program structures N_j, performance contracts, actual idle-code state, raw observation/cost records). Theta is a SINGLE controller over phase position and context, not a frozen network per gate.

It produces raw turning-point proposals, bounded gap/time profiles and odd monotone correction commands q_theta(h_desired,e). A monotone piecewise-linear neural/spline map with q_theta(0)=0 is one explicit command family; (2) composed with such a map is piecewise cubic, so maximum response residuals over a bounded amplitude interval can be certified by segment endpoints and derivative roots, rather than point samples. All evaluations, inversions, normalization and certification are paid. No universal NN capability is assumed.

For an admitted geometric context j, raw turning points v_theta(j) are converted to

    Pi_(a_j)(v)=argmin_ordered q ||q-v||²
                subject to Phi(q)=(a_j,0,0).                 (9)

This is a bounded, certified constraint solve, not an automatic cheap projection. Store its branch/warm start, check ordering and Jacobian regularity, and reject if uniqueness/precision/iteration caps fail. Where regular, differentiate the constrained equations/implicit KKT system; do not use an unexplained straight-through gradient through hard projection. Exact moment preservation is execution parameterization, rather than a soft penalty that permits accumulated semantic drift.

New observations and a new operation drive shared updates through finite modeled gate/process loss and actual control cost. A concrete proposal is a projected gradient step on

    worst_lambda∈U new-operation execution error
      +lambda_E commanded quadratic action
      +lambda_T total duration
      +lambda_S slew/turn cost.                             (10)

The task target is its retained confidence set; model gradients are internal calculations CONDITIONAL on U. If only noisy actual control evaluations are used, their shots and finite-difference/score work are charged instead. Minimax nonsmoothness can be handled by a declared active-hypothesis subgradient, with ties/step/reject rules frozen before execution.

The same shared weights generate OLD contexts too. Admission recomputes their actual profiles and tests their old contracts. If only task labels/library entries change and old generated waveforms never change, report lookup/program growth, not the requested substantial shared plasticity. If all old heads are frozen or copied, report the modular control, not this mechanism.

Projection (9) can retain geometric angles while waveform/gap/control maps change. Its maintenance is at least proportional to the old contexts actually checked unless a valid uniform-in-context certificate is obtained. We do not grant constant-cost preservation to an arbitrary generator or hide one full waveform per context. A direct formula such as (6) may be simpler and cheaper than the neural generator and must remain a competitor.

## 6. Paid calibration and finite-time admission

Use the existing preparation/evolution/measurement interface, with fresh calibration states between task episodes. At an appropriate static idle where B=i*gamma_i*gamma_j is a logical Z, prepare X+, command q for time tau, measure X. Under (2), the ideal binary likelihood is

    p_lambda=cos²(f_kappa(q)*tau).                            (11)

It is not a gain oracle. For q in {1/4,3/4}, tau=1, all four prior phases lie in (0,pi/2), so this particular query is nonaliasing. A single amplitude may already distinguish the FINITE prior; never force two amplitudes when a strong design can use one. Calibration of channels requiring another idle gapped partner involves charged routing/configuration changes. The common-response transfer is used only under its explicit assumption.

For geometry parameters, the query menu may contain known asymmetric rectangle loops at widths 1/3 and 2/3, with fresh X/Y measurement. Their adiabatic phases separate the P1/P2 contributions, but an ACTUAL finite query must use its full modeled Hamiltonian likelihood, including nonadiabatic and correction errors. It is invalid to substitute the ideal geometric angle into a noisy finite-time observation model. A finite set of hypotheses does not guarantee that a selected experiment identifies them; if all available predictions coincide, leave them unresolved.

Retain U as the subset of finite (c1,c2,kappa) hypotheses whose predicted Born probabilities lie in ALL actual confidence intervals. Choose the next query from the declared finite menu by greatest minimum separation of still-relevant predicted probabilities, or a declared information-per-cost objective. Allocate a batchwise confidence budget (or use an anytime-valid rule); cap queries/shots before execution. Empty U, unresolved hypotheses at the cap, drift and out-of-class observations are failure/uncertainty outcomes. No fictional response has been generated here.

Because geometry and calibration enter Hcd, admission is based on the full finite model for EVERY lambda in U:

    err_j <= sup_lambda ||U_lambda(theta,program_j)V_code
                             -V_code*Rz(beta_hat_j)||
                    +rad(B_j)/2 +r_j*T_j +epsilon_num.       (12)

For old Clifford contexts replace Rz by their specified braid; for the echo include both actual X calls. Code preparation/readout/word budgets are additional. Use a common phase convention appropriate to uncontrolled calls; controlled-gate global phases are outside this contract.

One explicit rigorous propagation route is piecewise-constant approximation plus a Hamiltonian variation bound and bounded matrix-exponential truncation error. If H is Lipschitz with bound L_H on a segment of width dt, midpoint replacement costs at most L_H*dt²/4 in operator norm. Sum segment errors and exponential-tail bounds. Nonuniform segmentation/interval arithmetic is allowed and charged. Merely observing a converged floating solution is not this certificate. The six-mode/full-parity matrix and its fixed-parity restriction, every hypothesis, rejected profile and precision increase count. No exponential many-body computation is hidden in this fixed four-dimensional parity block; no scalable many-qubit consequence is claimed.

Accept theta_trial only inside the physical family/caps and when (12), deadline, action and slew contracts hold for EVERY retained operation. Otherwise reject/backtrack, buy justified information or change the candidate program under its budget. A new accepted context becomes an old contract. Topology and moment identities protect neither a newly demanded deadline nor a newly tightened physical fidelity automatically.

## 7. Finite three-stage task stream and what would count

Use an episodic workload: the same six modes are prepared, acted on and measured; calibration occurs between completed episodes. This is retention of executable operation capabilities, not preservation of an unknown coherent data state while the sole register is destructively calibrated. That harder workload needs extra storage/calibration hardware or paid idle protection.

Initial library: routed Clifford exchanges, including an X implementation comprising two exchanges. Its actual code/error certificates must exist before the stream; they are not inferred from the ideal Hamiltonian. A proposed finite scale is exchange duration10, X duration20, all old error budgets fixed prospectively. Their physical feasibility has not been computed here and is an entry obligation for a future contract, not a claimed result.

The evaluator, not the learner, sets target phases beta1=pi/4, beta2=pi/6, beta3=pi/5. Each stage exposes only the task-copy interface (7); evaluator formulas remain inaccessible. New-operation norms are compared with the stated confidence allowance. Proposed total native-program deadlines are120,100,80 and quadratic-action caps150,130,110 respectively, INCLUDING the two X calls, boundary returns, smoothing and correction controls. Use per-new-gate norm budget .03 and modeled residual-norm tolerance1e-4. These are proposed finite contracts to inspect, not numerically passed values. A tighter old X allowance must reserve both echo errors. Gap, amplitude, slew, state-preparation/readout and worst-case word length must be frozen consistently before any process.

Each round: acquire task samples; update U only from actual paid calibration; form/repair the program; update shared theta; recertify all retained actual profiles; admit the new operation only after its full contract passes. Then execute predeclared mixed words using fresh inputs and the carried common idle code, with no intermediate root/state reseeding. Include an ordered noncommuting use, such as the new Z rotation composed with an old X-axis Clifford, not only three commutative angles.

Stage1 is not another Clifford or an old timing context. Subsequent phases are distinct transformations; they are not counted as new simply because a new name was assigned. The old library plus a strong compiler may implement them accurately already. Count its compilation, word length, time, shots, execution and storage; if the candidate merely caches that old word, call it ordinary program compilation. Exact exclusion from a universal primitive closure is not a prerequisite, but resource/function identity cannot be ignored.

The stream does NOT guarantee success or advantage. Source-style robust loops can be slow; full uncertainty propagation can be expensive; nonlinear correction calibration may consume most of the lifetime budget; repeated constraints may exhaust useful shared freedom. If any proposed deadline/action tuple is infeasible, record the rejected task rather than silently relax old contracts or rename a speed result as operation acquisition. No pilot has established feasibility or a training trajectory.

## 8. Strong rivals and falsifiable M0 outcomes

1. **Published robust geometric controller:** implement Karzig et al.'s universal geometric sequence, including its error assumptions and parity echo. It may solve these tasks directly with fewer learning steps. Our small moment formulas already give it a cheap exact comparator in the finite class. Do not cripple it by requiring an NN or withholding the response prior.
2. **Direct robust control/program learning:** same task copies, priors, Born queries, matrices, actuators and bounds; optimally estimate the target/plant and solve for an actual waveform. Allow direct cubic inversion, lookup/spline caching, model-based design and full control optimization. It may match or dominate the entire proposed pipeline.
3. **Old-library compilation:** same learned targets and calibration, best applicable compiler/search, paid query/word/precision costs. It tests whether a new native protocol matters beyond ordinary composition.
4. **Frozen/modular controllers:** preserve old waveforms and add a new controller per operation; grant legal calibration reuse, count storage/routing and all later execution. Do not force it to waste one full model per task if it can store a scalar plus a shared analytic controller.
5. **Mechanism interventions:** remove moment projection while preserving learning capacity/access; separately relax boundary pinning under an EXPLICIT changed physical response model. These test distinct mechanisms and are not substitutes for strong same-plant baselines. Release of boundary pinning is not a fabricated measurement disadvantage assigned only to one competitor.

All costs include target copies, calibration preparations and measurements, routing/parity/code preparation, rejected programs/updates, matrix propagation/certification, inversion/controller evaluation, full waveform storage or generation, slew/action, echo calls, coherent idle storage when promised, and every native/compiled execution. Report cost vectors where physical weights are unknown. Classical simulation of six modes is not a quantum resource advantage.

Useful positive evidence would require actual MULTIPLE new transformation acquisitions; continued substantial changes to the shared executed controller; old task/word contracts retained; and a reproducible important advantage against these controls. Topological causal evidence would need show which boundary-protection/moment property enables that advantage, with finite-time/calibration contributions separated.

Negative patterns include: the direct formula/compiler matches all outcomes more cheaply; the NN merely caches targets; projection/uncertainty refresh grows faster than modular storage; new gates lose fidelity under later learning; finite-time correction costs erase geometric robustness; task examples or hidden targets leak; relaxed-boundary changes explain a result unrelated to the actual physical regime. These do not close topology MAIN, but they prevent claiming this version passes M0.

## 9. Decisive scientific gap

This is an explicit operation-formation proposal rather than another speed-only stream: a task angle is learned from Born examples, a robust executable path may gain new sweeps, an old X is reused in its echo, and a shared controller continues adapting under old-operation contracts. Its load-bearing robust contour law and direct inverse/control alternatives are known. No historical novelty is claimed by appending a neural generator.

The immediate attack should be whether this proposed formation law reduces exactly to existing robust holonomic compilation plus process estimation, with no consequential remaining difference. On the finite two-mode response class it very plausibly does: (6) is a formidable direct solution. That equivalence is not a veto based on universal simulability; it is a specific same-information mechanism/resource comparison. A meaningful M0 contribution would need an actual learning-freedom, capability-growth or full-cost consequence beyond that comparator. This note makes that gap concrete but does not supply its victory.

**Status:** constructive M0 attempt, unfinished and UNEXECUTED. Root should assess this formation/retention mechanism and its known-compiler boundary before authorizing any experiment. Old speed/calibration results remain valid in scope; none, including this proposal, is M0 completion.
