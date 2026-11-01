# Concrete quantum realization: shared learned control geometry, retained braid, explicit finite-time cost

SOURCE SCAFFOLD + ROOT CONSTRUCTION/DERIVATION. Same human-locked topology MAIN. No quantum hardware, simulation, training or new resource route. This supplies the genuine protected family missing from the earlier generic Hamiltonian scaffold. It is not a claim of inventing Majorana braiding, counterdiabatic driving or a novel optimizer.

SOURCE: Karzig et al.2015, original pp201102-2–3, Eq15/17 and the three-step exchange. Root personally read those passages. Their model and exact correction are constructive ingredients here; original quantum experimental results are not reproduced.

## Actual shared controller and executed rule

Let gamma0,...,gamma5 obey {gamma_i,gamma_j}=2delta_ij. The selected active Y uses gamma0,...,gamma3; gamma4,gamma5 are uncoupled spectators during that primitive. Work at fixed TOTAL parity and count all six modes and their preparation. Use the stated active Y-junction family

    H0(t)=i sum_(j=1)^3 Delta_j(t) gamma0 gamma_j.

A nontrivial qubit needs the added spectators: six Majoranas give eight full states, four at fixed parity; fixing the gapped pair in its ground occupancy leaves a two-dimensional logical ground code. The standalone four-Majorana active Y would instead have only a one-dimensional ground code at fixed parity and must not be called a protected qubit by itself.

A retained exchange uses three pairwise-coupling legs, along the positive octant edges, from the Delta3 axis to Delta1, then Delta2, then Delta3. On each leg i,j,

    Delta_i=g_theta(s) sin phi_theta(s),
    Delta_j=g_theta(s) cos phi_theta(s),   other Delta=0,
    s=t/T_leg,    phi_theta(0)=0, phi_theta(1)=pi/2.

Directions/order can be permuted or reversed under the same specified exchange interface. Endpoints have the common fixed g_idle, and gap is kept positive. In this particular source-defined family the ideal logical exchange is the stated braid exp(pi gamma1 gamma2/4), not a generic claim about every gapped Berry loop. Repeated retained exchange words use their common physical idle code; finite-size splitting, quasiparticle poisoning and other physical imperfections are additional errors, not removed by the model.

Use SHARED neural weights theta for the phase/gap profiles across the library, not a frozen network per old exchange. One explicit smooth positive phase parameterization is

    w_theta(s,e,a)=s²(1-s)² exp(n_theta(s,e,a)),
    phi_theta(s)=(pi/2)*integral_0^s w_theta(r)dr
                              /integral_0^1 w_theta(r)dr.

The context a is the executed leg/exchange and e observed calibration evidence. The fixed endpoint/edge wiring and monotone normalized phase are supplied architecture. Numerical integration/normalization/derivatives, bounds on n, evaluating the network and maintaining the context library all cost work. The zero endpoint slope pins the correction to zero at idle. A bounded gap profile with fixed g_idle endpoints can be parameterized separately by the SAME network; it is not allowed to make the gap arbitrarily large for free.

Implement the source correction

    Hcd(t)= i/(2||Delta||²) sum_(i<j)
                 (Delta_j dotDelta_i-Delta_i dotDelta_j) gamma_i gamma_j.

On a two-coupling leg this reduces exactly to i*(dotphi/2)*gamma_i gamma_j, with the sign determined by the leg convention. Radial derivatives cancel. The actually executed unitary obeys

    i U_dot=(H0+Hcd+E_actual)U.

Theta can change both the shared implementation and its actual work/latency while every admitted ideal exchange still has the same logical braid. Install theta at idle: Delta endpoints/g_idle and Hcd=0 are fixed, so no uncharged basis reset or state assignment is needed. Real residual control/calibration errors E_actual must still be bounded/observed. The finite-time correction is NOT itself topologically protected.

## A complete prospective learning law

Acquire actual demanded deadline/error and hardware work/calibration feedback for a new or improved exchange context. Differentiate the finite modeled Hamiltonian evolution/ordinary process loss when those derivatives are available, or use paid finite-difference/experimental control queries otherwise:

    theta_trial=theta-eta grad_theta[L_new(actual gate error;deadline)
                                    +lambda C_control].

Retained contexts share this theta. Accept inside the pinned edge/gap family and a declared old execution-error envelope; measure useful new performance rather than only a parameter displacement. Uniform error bounds must cover the actual old contexts and waveform range, not just a sampled point. Architecture makes ideal semantics retention cheap; calibrating a uniform actual envelope can remain expensive. Refused updates, model acquisition, correction controls, precision, network evaluation and validation are included in total cost.

This rule is ordinary constrained quantum control on a topology-protected family. The research hypothesis concerns its continuing learning/retention economics, not whether ordinary gradients are new. An ordinary model-aware counterdiabatic controller gets the SAME coefficients, derivative access and actuators. It can be better than a learned approximation and must remain in the comparison.

## Completed resource derivation: speed has a nonzero correction cost

ROOT RECONSTRUCTION from the source Hamiltonian. On one leg, ||i gamma_i gamma_j||=1, so

    integral_leg ||Hcd||² dt
      =(1/4) integral_leg dotphi² dt
      >= pi²/(16 T_leg).

Cauchy–Schwarz uses the fixed phase integral pi/2. Summing three legs of total duration T and using sum 1/T_leg>=9/T gives

    C_cd := integral ||Hcd||² dt >= 9pi²/(16T).

It is a lower bound on a declared squared-control functional, not measured electrical energy. Constant phase velocity/equal leg durations attain the unconstrained bound; with pinned zero endpoint velocities it is an infimum approached only if increasingly sharp ramps are allowed. Actual slew-rate, acceleration or smoothness limits can strengthen it. Their costs must not be omitted.

If ||Hcd||<=hmax is an actual hardware cap, the same phase-integral argument gives T>=3pi/(4hmax), with stricter time if finite endpoint ramps are required. Thus no learned geometry can make the exchange arbitrarily fast at bounded correction amplitude. This is a quantitative cost constraint, not a veto on useful learning.

For constant radial gap g on a leg, H0 and that leg's Hcd anticommute, so ||H0+Hcd||²=g²+dotphi²/4. The modeled quadratic-control action is therefore bounded by

    C_action >= g² T + 9pi²/(16T).

Its unconstrained best T is3pi/(4g), with minimum3pi*g/2; real hardware/noise/error limits may exclude that point. Variable-g/slew constraints require their actual functional, not this simple formula. Reporting faster gates while ignoring a growing correction term would incorrectly claim lower TOTAL cost.

## What old-capability retention actually needs

If the only correction calibration error is a multiplicative e(t) with |e|<=e_max, then E=e Hcd and

    ||(U_actual V-V G_braid)|| <= integral ||E||dt
                                <= (3pi/4) e_max,

up to the declared global-phase interface and other errors. This conditional bound holds for every monotone admitted phase profile; it separates geometry learning from correction accuracy. It does not show how to obtain e_max from finite data or certify general unknown plant errors. If physical imperfection is amplitude-dependent, changing theta can invalidate that same e_max and trigger additional maintenance.

Word errors and preparation/readout/migration use the earlier common-code bounds. Newly admitted shorter deadlines become old performance obligations; preserving only the braid class while later losing their speed/fidelity is not retention of the new capability. Stored-code noise during idle, repeated gates and controller drift are part of a lifetime comparison.

## Scientific status

This is an explicit quantum-mechanical family in which shared neural parameters may keep changing, exact ideal braid semantics remain fixed, and meaningful finite-time/control costs can change. It provides no new gate algebra, quantum hardware result, automatic calibration guarantee or demonstrated TOTAL-cost advantage. It is consistent with the human's motivating quantum-topology direction without assuming that motivation is a verified benefit.

The consequential next question is now concrete: can learning useful shared profiles/calibration reduce lifetime work (including maintaining the error envelope) compared with direct model-aware control, freezing/modular controllers and other same-actuator protection strategies? The classical phase/mobility constructions isolate timing, flow-direction and basin-certification issues for this SAME question. They are support models, not a switch away from topology MAIN or a substitute claim of quantum advantage.

## Actual independent physical-interface correction

Review024 identified the fixed-parity degeneracy issue; the six-mode construction above is the applied repair. This does not change the source active-block Hamiltonian/norm algebra, but it changes the physical resource count and makes a nontrivial encoded qubit explicit. For a noncommuting retained library, a larger routed star/network may activate triples {i,j,3} with i,j among the idle zero modes {1,2,4,5}, using gamma0 as center and gamma3 as the common idle gapped partner. Each exchange leaves the same idle code, with the other two modes spectators during that primitive. Required center/outer and correction couplers, routing, calibration and spectator preparation are real additional resources. A single active Y gate is not silently promoted to a universal or noncommuting implemented library. No physical device was added or accessed; this is a corrected proposed mathematical/hardware model.

## Continuing research action: replace a cost-free calibration bound with a paid response class

JOINT constructive development with the actual independent reviewer. The uniform e_max above was an assumption, not free knowledge. Use an explicit prospective response class, for example a relative correction error bounded by e0+L|h(t)| on the currently active correction channel, plus Hermitian residual generator norm at most r. These are actual model assumptions to calibrate/validate, not results read from hardware. Time-delay/non-Markovian/decoherence or unmodeled channels need further terms; this class does not cover them by naming them small.

Let A=integral ||Hcd||dt=3pi/4 for the monotone three-leg exchange. Then the Duhamel certificate score is

    epsilon_cert <= A e0 + L C_cd + r T.

The speed/control lower bound gives the best UNCONSTRAINED infimum of this SUFFICIENT SCORE at fixed T:

    epsilon_score,inf(T)=A e0 + L A²/T + rT.

For L,r>0 its unconstrained minimizing duration is T=A sqrt(L/r), with score A(e0+2sqrt(Lr)). Pinned endpoint/slew/gap/amplitude limits can raise that infimum. This is NOT a lower bound on actual gate error: destructive cancellation, a sharper certificate or another legal controller may perform better. Failure of this sufficient verifier is not failure of the topological direction.

For a demanded old gate norm budget epsilon_old, this particular verifier requires, even at its unconstrained profile optimum,

    rT² -(epsilon_old-Ae0)T + L A² <=0.

For r>0 with a nonnegative discriminant, the feasible durations lie between the two quadratic roots, further intersected with physical control/slew limits. For r=0 the inequality has its linear limiting form; if epsilon_old>Ae0 it instead requires T>=L A²/(epsilon_old-Ae0), with no upper duration bound from this particular score. Both shortening too much and holding too long can therefore consume the declared error budget for different modeled reasons. Useful learned profiles change C_cd and potentially the calibrated validity range; a stable ideal braid does not guarantee those retained performance contracts automatically.

An explicit continuing admission rule is now available. Retain an uncertainty set U over (e0,L,r) justified by actual calibration responses/noise/model assumptions. For each shared theta proposal, enforce the old braid family and

    sup_(e0,L,r in U) [A e0+L C_cd,a(theta)+r T_a(theta)]
         <= epsilon_a

for each retained generator/performance contract, with the declared word-length/preparation/readout allowances. If this fails, either reject/revise the proposal, improve a conservative bound, or buy additional calibration information and count it. Update U only by intersecting with constraints justified by the ACTUAL paid responses; no narrowing from desired success. No such responses or experimental success have been obtained in this session.

A newly admitted useful deadline or correction protocol is inserted into the retained contract set; it is not protected just by its logical class. Complete work includes generating/evaluating waveforms, calibration queries, maintaining U, all rejected updates, correction amplitude/slew, process/state preparation, numerical precision, idle coherent-state protection if the task requires it, routing/spectator modes and physical execution. Direct model-aware robust control gets the SAME U, feedback and actuators; classical same-rule implementation is not a separate competitive result. This is a concrete cost-bearing research mechanism to develop, with neither a claimed new optimizer nor a current superiority conclusion.

Finite observations alone cannot establish this uniform response class without regularity/parametric assumptions: an unobserved between-sample response can invalidate it. The constructive task is to obtain a justified, affordable uncertainty envelope permissive enough for actual useful shared learning, then test the lifetime tradeoff against legal strong methods. This now states exactly which information and maintenance burden the topology itself does not supply.
