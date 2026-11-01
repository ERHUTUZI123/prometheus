# Paid nonlinear calibration supporting protected continual acquisition

**RECONSTRUCTION / prospective protocol, 2026-11-01.** Same V2; MAIN=TOPOLOGY, PRIORITY=USER_LOCKED, MAIN_CHANGE=HUMAN_ONLY; Online USER-PARK; Persistent State standalone paused. Provider-effective mode unknown. No hardware, Bernoulli shots, simulation, training, code or dependency was executed. Only this new artifact and its assigned source record are written.

Read the actual complete `quantum-shared-control.md`, `calibration-observation.md`, Review024, and canonical `water/now.md` Core M Axis (M0–M3, six Permanent Evaluation Rules, Historical recognition). **M0 is active and UNPASSED.** Calibration/profile adaptation is support, not a substitute for actual multistage acquisition of new executable operations, topological causal retention and an important reproducible advantage. M1–M3 are later stages, not parallel allocations. This note does not change MAIN or prescribe a new candidate competition.

## Judgment

An identifiable bounded cubic response class permits an explicit paid Born-observation protocol and a **two-dimensional polygon uncertainty set**. Finite probability intervals become linear coefficient inequalities. After nominal inverse compensation, the exact worst relative error over a continuous command-amplitude interval occurs at its two endpoints. Thus elementary classical postprocessing can provide a profile-uniform sufficient old-gate error envelope without measuring every waveform. This is conditional on the response/domain/observation prior, not empirical validation of that prior.

Adaptive calibration rounds require a simultaneous confidence construction; drift requires set expansion, not indefinite intersection with stale constraints. A direct idle zero-mode pulse is a strong legal comparator under the already supplied correction actuators and response prior. It implements the same exchange with ideal area pi/4 instead of 3pi/4, and receives the same calibration set. These valid cost insights do not by themselves pass or refute M0.

Coverage: **DIRECT** for the conditional identifiability, confidence, error-envelope and direct-pulse algebra below; **PARTIAL** for lifetime resource economics; **OUT_OF_SCOPE** for observed learning, physical calibration success, historical novelty, quantum advantage or completion of protected continual acquisition.

## 1. An explicit response and observation prior

Let c be a requested nonnegative correction-channel amplitude, with units of inverse time. On a **declared** memoryless command domain 0<=c<=H_model, assume

    a_kappa(c)=kappa0 c+kappa1 c^3,
    kappa=(kappa0,kappa1) in K,
    0<k0_min<=kappa0<=k0_max,  0<=kappa1<=k1_max.        (1)

K is a known compact polygon/box, not a posterior learned from no data. Then a'(c)=kappa0+3kappa1 c^2>0, so a has a unique nonnegative inverse on its actual range. The coefficients are constant during a stationary calibration epoch; drift is treated separately below. A negative cubic coefficient could be allowed with additional monotonicity inequalities, but is not silently included here.

The model is a plausible local nonlinear transfer approximation, not an established description of a Majorana device. Its required premises include the physical amplitude/coherence range, absence or bounds of delay/hysteresis/crosstalk, and transfer across channels. Independent channel gains require independent sets and identifiable probes. Negative commands require an explicit odd-response premise a(-c)=-a(c), a calibrated sign-selecting actuator, or separate signed probes/sets. Positive-amplitude data alone do not establish that symmetry.

A concrete moderate prior example uses a fixed reference amplitude H_ref (never silently changed when a later waveform expands):

    beta0=kappa0 in [9/10,11/10],
    beta1=kappa1 H_ref^2 in [0,1/5],
    h1=H_ref/2,   h2=H_ref.                            (2)

This example assumes H_ref is inside the valid model/hardware range. It is an inspectable prior for a weak cubic distortion, not measured evidence.

At a calibration idle configuration, a permitted zero-mode pair B=i gamma_i gamma_j acts as a logical Pauli. Prepare its X+ state, apply the constant requested amplitude h for duration tau under H=a_kappa(h)B, and measure X. With the stipulated preparation/readout and no unmodeled term, each paid binary outcome has

    Pr(X+|kappa,h,tau)=cos^2[tau a_kappa(h)].            (3)

Preparation, spectator/gap/routing configuration, pulse generation, evolution and measurement are real resources. An unknown retained logical input is not copied or destructively reused for calibration. One cannot infer this measurement service merely from specifying the Hamiltonian.

## 2. Fixed nonalias probes and exact identifiability

For a probe amplitude h>0, compute a_min(h),a_max(h) from K. If a_max/a_min<=3, fix **before its outcomes**

    tau_h=pi/[8 a_min(h)].                             (4)

The whole prior has angle tau a(h) in [pi/8,3pi/8], where cos^2 is one-to-one and |sin(2 tau a)|>=1/sqrt(2). These times must fit the actual coherence/timing capabilities. Increasing h, tau or shot count is not free. A wider prior requires a different justified nonalias design or less favorable conditioning, not inversion of an aliased frequency as though unique.

The example (2) satisfies this prior-ratio condition at both amplitudes. Here tau_j h_j=5pi/36. In the noiseless infinite-data limit the two probabilities determine

    g_j=a(h_j)/h_j=kappa0+kappa1 h_j^2,
    kappa1=(g2-g1)/(h2^2-h1^2),
    kappa0=(h2^2 g1-h1^2 g2)/(h2^2-h1^2).              (5)

The determinant is nonzero exactly when h1,h2 are nonzero and distinct. Two repetitions at one amplitude identify only one combination. This is structural identifiability inside (1), not arbitrary Hamiltonian identification. Unknown contrast, SPAM, T2 or additional Hamiltonian coefficients add unknowns and require extra informative designs or prior bounds.

## 3. Paid probability intervals become linear coefficient strips

For a batch j, choose h_j,tau_j,n_j before its outcomes, possibly from previous completed data. Let p_hat_j be the X+ frequency from n_j conditionally independent preparations. A valid fixed-batch interval has

    e_p,j=sqrt[log(2/delta_j)/(2 n_j)],
    I_p,j=[p_hat_j-e_p,j,p_hat_j+e_p,j] intersect [0,1]
              intersect the prior nonalias probability image. (6)

If this intersection is empty, retain the conflict and refuse unsupported admission; do not invent a narrower successful response. It can reflect a confidence failure or a model/observation defect, not automatically a diagnosed cause.

Since angle is monotone on the stipulated interval, invert with outward numerical enclosures:

    a_low,j=arccos(sqrt(sup I_p,j))/tau_j,
    a_up,j =arccos(sqrt(inf I_p,j))/tau_j,
    a_low,j/h_j <= kappa0+kappa1 h_j^2 <= a_up,j/h_j.    (7)

Update the stationary set by intersection with these two halfplanes:

    U_j=U_(j-1) intersect strip_j,   U_0=K.             (8)

No coefficient oracle was queried. All shrinkage comes from the actual future Bernoulli records through (6)–(8). This session has no such records and therefore has not actually narrowed U.

On the nonalias image the inverse-angle derivative has magnitude <=sqrt(2). Thus the resulting effective-gain interval has half-width at most

    eta_j <= sqrt(2) e_p,j/(tau_j h_j),
    n_j >= log(2/delta_j)/[(tau_j h_j)^2 eta_j^2]        (9)

as a sufficient count for a chosen effective-gain precision. This is a fixed-time upper bound for this simple protocol, not an optimal information lower bound.

For completeness the concentration bound requires no Bayesian-credible-set interpretation: for a Bernoulli variable, the centered log moment generating function has second derivative at most 1/4 and value/first derivative zero at zero. Hence it is <=lambda^2/8. Conditional independence plus Chernoff optimization gives Pr(|p_hat-p|>e)<=2exp(-2n e^2), which yields (6). Approximate particle/posterior regions do not automatically provide this simultaneous frequentist guarantee.

### Optional rounds and stopping are not free confidence

For potentially adaptive numbers of completed batches, assign a single global failure budget delta and

    delta_j=6 delta/(pi^2 j^2),  j=1,2,... .            (10)

Given past data, each next batch design and count are fixed before that batch's outcomes. Its conditional coverage fails with probability at most delta_j. A union bound gives, with probability >=1-delta, coverage at **all** completed batches, so stopping after any completed batch and adapting controller proposals is safe on that common event. Controller updates themselves consume no additional statistical budget when they merely use this event and stay inside the model domain.

This does not authorize stopping inside a batch using its fixed-n interval. One elementary anytime alternative for a fixed h,tau batch uses per-prefix budgets

    delta_(j,n)=36 delta/(pi^4 j^2 n^2)

and (6) at every prefix n. Their total is delta; intersection/update can then be used at an arbitrary prefix. It pays the additional logarithmic confidence width. More efficient confidence sequences are possible, but must actually be specified rather than asserted. A finite predeclared total batch count permits a simpler finite allocation. All physical shot/process caps remain external and finite.

## 4. Classical postprocessing is small, but not costless

Each strip in (7) is linear in two real coefficients. Starting from a bounded polygon, after J batches it has O(J) edges/vertices in the worst case. Incremental clipping by one halfplane costs O(V) arithmetic for a stored V-vertex polygon; two halfplanes per batch suffice. A linear support query scans V vertices in O(V). Retain coefficient/constraint precision, failed intersections and all outcome records needed for audit. Repeated parameter drift or independent channels multiply this work.

There is no quantum computation required to propagate U. Directed rounding for probability inversion, clipping and objective evaluation must enclose rather than accidentally exclude feasible parameters. The program/arithmetic has not been implemented or verified here. Keeping only separate coefficient intervals is simpler but loses correlations; the polygon can give materially tighter waveform envelopes.

## 5. A uniform profile-amplitude envelope from two endpoint LPs

Choose a nominal hat_kappa in U with positive kappa0 and monotone response. Suppose the desired correction coefficient is d(t)>=0 and command it by

    c(t)=a_hat_kappa^-1[d(t)],
    0<=c(t)<=C<=H_model.                              (11)

Nonlinear inversion, lookup/interpolation precision and command/slew limits are actual control work. If the inverse is approximated, add its certified residual separately. Calibration alone does not correct a known gain bias; using unmodified commands corresponds to hat_kappa=(1,0), whose systematic mismatch need not vanish with more data.

For a fixed true kappa,

    [a_kappa(c)-d]/d
      =[(kappa0-hat_kappa0)+(kappa1-hat_kappa1)c^2]
                          /[hat_kappa0+hat_kappa1 c^2]. (12)

Its continuous limit at c=0 is well-defined. As a function of q=c^2 the derivative has constant sign, since its numerator is kappa1 hat_kappa0-kappa0 hat_kappa1 and the denominator is a positive square. Therefore the maximum absolute relative error over [0,C] is at q=0 or q=C^2, for every fixed kappa.

Let

    l(q)=min_(kappa in U)(kappa0+q kappa1),
    u(q)=max_(kappa in U)(kappa0+q kappa1),
    d_hat(q)=hat_kappa0+q hat_kappa1.

The exact uniform relative uncertainty bound within (1) is

    E_rel(U,hat_kappa,C)
      = max_(q in {0,C^2})
          max{1-l(q)/d_hat(q), u(q)/d_hat(q)-1}.        (13)

Only four small linear support queries are needed. This is a uniform continuous-amplitude result, not a probability interval checked only at sampled waveform peaks. At c=0 the absolute error is zero; the limiting relative bound is a convenient sufficient control certificate.

For the retained monotone three-leg braid, A=integral ||Hcd|| dt=3pi/4. If the only execution error is this response uncertainty plus an independently justified Hermitian residual ||R||<=r_H, Duhamel gives the sufficient old-gate norm envelope

    epsilon_a,cert <= A E_rel(U,hat_kappa,C_a)+r_H T_a,
                                                        (14)

plus preparation/readout/idle and other explicitly modeled errors. It holds for **every** admitted profile of that command-amplitude range on the common parameter-coverage event, even when profiles are chosen adaptively from the calibration data. No new independence assumption over infinitely many theta updates is needed. Independent channels use their individual U and the corresponding per-channel absolute correction areas; a more general simultaneous-control bound uses the sum of those areas.

The confidence failure probability delta is epistemic coverage risk, not a physical per-gate stochastic error rate or a measured fidelity. A common bad coverage event can affect the whole later library; it is not an independent fault for each call. Retain the conditional norm envelope and the simultaneous confidence statement separately when reporting an operational guarantee.

Rabi data in (3) do not identify a general residual r_H, leakage, quasiparticle poisoning or nonunitary error. These need their own observations/validated bounds and, for nonunitary errors, the appropriate channel-level guarantee. Setting them to zero is an explicit restricted model, not calibration success.

### A complete two-amplitude precision consequence

In the normalized example (2), q1=1/4 and q2=1. If inverted effective-gain intervals have midpoints z1,z2 and common half-width eta, the two center equations give a nominal beta_hat (provided it lies in the prior). The true intercept error is at most 5eta/3; the q=1 gain error is at most eta. Since beta_hat0>=9/10, (13) gives

    E_rel <= 50 eta/27.

Thus eta<=27 epsilon_gate/(50A) suffices for (14)'s response term alone. Combining with tau_j h_j=5pi/36 and (9) specifies two actual fixed-time shot budgets. Their evolution time is n1 tau1+n2 tau2, not merely two observations. With equal counts n it is 5pi n/(12 H_ref), plus 2n preparations/readouts and all classical/control work. No outcome values are fabricated by this consequence.

If the center solution falls outside the prior, use a feasible nominal point and the actual LP bound (13). A simple outcome-independent conservative consequence also exists: any two feasible parameters have effective-gain differences <=2eta at both probes, yielding E_rel<=100eta/27 for any hat_kappa in nonempty U. Thus eta<=27 epsilon_gate/(100A) guarantees the same response allocation on the coverage event without assuming that the noisy midpoint solution is feasible. This costs four times the preceding sufficient shot count, and can be sharpened by the polygon rather than assumed away.

For a worst-case word length L and word norm budget epsilon_word, telescoping needs an allocated per-gate error of order epsilon_word/L. This simple shot prescription then scales as L^2, up to confidence factors and other budgets. It can be independent of the number of contexts sharing the same valid response event, while depending strongly on accuracy, channels, amplitude domain and drift. This is not an optimal physical lower bound.

## 6. Changing amplitude/profile domain: information versus an assumed model

Every proposal needs a **uniform** command-range bound, not just waveform samples. For root's normalized phase family w=s^2(1-s)^2 exp(n_theta), if |n_theta|<=N by architecture, then

    integral_0^1 w ds >= exp(-N)/30,
    max w <= exp(N)/16,
    max |dot phi|/2 <= 15pi exp(2N)/(32 T_leg).         (15)

Thus a conservative desired correction bound D_max is available analytically for each leg duration. The nominal inverse maps it to C<=a_hat^-1(D_max). Checking C<=H_model and the actual hardware cap is part of admission, as is waveform/inverse approximation error. If the cap is on applied rather than requested amplitude, a uniform peak bound is C u(C^2), using the same support function as (13). The bound can be conservative; obtaining a tighter bound costs work. Numerical quadrature/normalization errors are not certified just by writing an integral parameterization. A bound on |n_theta| alone does not bound its derivative or pulse slew; those need a separate architecture/derivative/control cap. Inverse compensation gives dot c=dot d/a_hat'(c), which must also be included in that check.

If a later C grows while remaining inside the **original** uniform cubic domain, (13) remains valid: no new shot is logically compulsory merely because C was not a previous probe amplitude. Evaluate the expanded endpoint support. Uncertainty in kappa1 can make the new certificate too loose, in which case paid high-amplitude probes, slower profiles or rejection are legitimate choices. Newly chosen h,tau must still obey nonalias/coherence constraints and the confidence budget.

If C exceeds the domain on which the cubic prior was justified, neither old confidence coverage nor a few additional calibration points establish that response law uniformly. One must justify an enlarged physical/model class, add bounded discrepancy or refuse the expansion. The normalized beta coefficients in (2) retain their original H_ref; redefining H_ref after expansion without transforming the parameter set would be incorrect.

For example, a known uniform remainder |zeta(c)|<=R5 c^5 can be admitted as a model assumption. Probe strips widen by the corresponding response remainder after a nonalias inversion covering the full allowed response. The gate envelope adds

    sup_(0<c<=C) |zeta(c)|/a_hat(c)
        <= R5 C^4/(hat_kappa0+hat_kappa1 C^2)          (16)

for the nonnegative nominal coefficients. More shots do not remove this floor. A held-out amplitude can challenge the model but passing finitely many tests does not prove the absence of all between-sample departures.

Static Rabi probes also do not establish memoryless behavior at arbitrarily fast slew. A known delay bound tau_d, for instance, adds an amplitude error bounded by L_a tau_d max|dot c|, or an integrated variation bound; L_a can be bounded from the response class. Identifying that delay/bandwidth bound requires additional paid experiments or physical knowledge. New channels, sign changes, leakage or cross-coupling require their own stated extension. Do not hide this uniform-model burden inside cheap polygon updates.

Known T2 gives a known contrast factor exp(-tau/T2); invert the affine contrast map before (7), paying the corresponding larger statistical interval/shot count. Unknown T2 or SPAM contrast/bias cannot simply be fitted away by more repetitions at the same two design points. An independently bounded probability bias b widens (6) by b and leaves a nonvanishing uncertainty floor. Additional residual Hamiltonians during calibration need their own effect on the likelihood; for example a known norm bound r_H over a pulse gives a conservative probability perturbation <=2 r_H tau by Duhamel. None of these quantities is inferred by declaring the cubic model identifiable.

## 7. Drift: expand old information, then buy fresh constraints

Suppose a separately justified drift prior gives

    |dot kappa0|<=d0,    |dot kappa1|<=d1.             (17)

From a valid reference-time set U(t0), propagate

    U_prior(t)=[U(t0) plus
        [-d0 Delta,d0 Delta] x [-d1 Delta,d1 Delta]] intersect K,
    Delta=t-t0.                                      (18)

Then intersect a new batch's constraints on the **current** coefficients. Intersecting stale strips as though kappa were constant can exclude the true plant. If drift is unknown/unbounded, finite past data supply no future uniform envelope; model/time-range information is indispensable.

A constant-within-batch model permits (6) directly at that batch's coefficient value. If drift acts during the batch, let t_ref be its end. The ith probability depends on the phase integral over its pulse. The reference cubic probability differs from its mean batch probability by at most

    b_drift <= tau_j(h_j d0+h_j^3 d1)
                 max_i |t_ref-t_pulse,i|,             (19)

with the maximum including pulse interiors. This follows from the drift bound on integrated phase and |d cos^2 x/dx|<=1. Widen the probability interval by b_drift before inversion; Hoeffding for independent varying-mean shots, or the corresponding bounded martingale differences, controls fluctuation about the batch's average conditional mean. Nonaliasing concerns the reference prior used for inversion, enlarged for any stipulated model remainder.

Preparation/readout and waiting time age the data too. For serial shots, a crude batch-age bound grows with n_j(t_prepare+tau_j+t_measure); increasing n reduces statistical error but increases this conservative drift term. A long batch is not automatically more informative about the plant **now**. Shorter blocks, a better drift estimator/model, parallel resources or slower requested operation each have costs and need their actual assumptions. This tradeoff is not an optimal metrology lower bound.

For a useful refresh deadline, let

    Lambda(C)=max_(q in {0,C^2})
                       (d0+q d1)/(hat_kappa0+q hat_kappa1).

From (13),(18), before clipping by K,

    E_rel(U(t),hat_kappa,C)
       <= E_rel(U(t0),hat_kappa,C)+Delta Lambda(C).     (20)

An old generator's remaining error slack s_a after all allocated terms therefore requires Delta<=s_a/[A_a Lambda(C_a)] when the denominator is nonzero. Use the **end of the intended gate/word interval**, not just its start, for this age. Take the most restrictive retained contract, with a fresh calculation after shared profile or nominal-compensation changes. Data references are batch-end sets incorporating (19), not fictitiously timeless coefficients.

The global confidence allocation (10) can cover these time-indexed sets if each new constraint is valid at its declared reference and the deterministic drift premise holds. It does not create a confidence bound for the drift rate itself. Fresh data, reference propagation and an empty/conflicting set are auditable outcomes, not reasons to shrink U until the desired task passes.

## 8. A strong direct idle-pulse comparator is already available

At the common idle configuration H_idle=i g gamma0 gamma3, let i,j be distinct idle zero modes among {1,2,4,5}. The correction coupler B_ij=i gamma_i gamma_j is already required for the corresponding routed exchange. Disjoint even Majorana bilinears commute:

    [i gamma0 gamma3, i gamma_i gamma_j]=0,
    [B_ij,total parity]=0.

It preserves the gapped pair's occupancy and acts within the same two-dimensional idle logical code. It can split the logical degeneracy during the pulse; no passive topological protection of that phase accumulation is claimed. Different noise consequences require their actual model and costs. A direct pulse H_control=h(t) B_ij therefore implements, apart from the specified common idle phase,

    U_direct=exp[gamma_i gamma_j integral h(t)dt].

Choosing A_d=integral h dt=pi/4 gives the same ideal

    G_ij=exp(pi gamma_i gamma_j/4)                      (21)

as the retained exchange. This is an actual finite-dimensional operator identity, not a universal-simulation argument. Coherently controlled-call interfaces may require explicit matching of the known idle/global phase; pad or correct it and charge its time/control rather than automatically discarding it.

For a compensated direct pulse, a scalar angle error delta_A gives

    ||U_actual-U_ideal||=2|sin(delta_A/2)|<=|delta_A|,
    |delta_A|<=A_d E_rel(U,hat_kappa,C).                (22)

The three-leg counterdiabatic braid instead has A_b=3pi/4=3A_d. Under this response-only sufficient verifier, direct control needs a relative envelope three times less tight for the same gate-error allocation. The simple fixed-time shot prescription scales quadratically with demanded precision; this gives a factor-nine difference in that **sufficient prescription**, not a universal information lower bound or a theorem about the actual braid error, which may have cancellation.

The direct controller receives the same U and observations. For a constant command c, compute a_L(c),a_U(c) by two LP supports of (1). If both are positive, the minimax duration

    T_d=2 A_d/[a_L(c)+a_U(c)]

balances the extreme angle errors, with

    max |delta_A|=A_d [a_U(c)-a_L(c)]/[a_U(c)+a_L(c)].  (23)

This is cheap direct robust control under the same parametric class, not a free gain oracle. Command cap, pulse duration, ramps, calibration age and other residual error terms must still pass. With finite ramps use the actual shaped-pulse moments/interval integration; no instantaneous-switching assumption should be silently donated to one method.

### Concrete resource comparisons and their scope

For the declared ideal coefficient-squared functional and common total duration T,

    inf C_direct=A_d^2/T,
    inf C_braid=A_b^2/T=9 A_d^2/T.                    (24)

Likewise an ideal coefficient cap h_max gives the area-only lower bounds T_direct>=A_d/h_max and T_braid>=3A_d/h_max. Pinned smooth endpoints/slew can raise the bounds. These are infima of specified control functionals, not electrical energy measurements. Nonlinear inverse compensation changes the command-cost functional; compute it from actual c(t), not by reusing (24) as command power.

A particularly concrete legal strategy copies the correction waveform of the exchange's i,j leg and applies it to that same i,j coupler at idle. It uses the same actual amplitude/slew domain and calibrations, has the required direct area pi/4, and omits the other two correction legs and the axis-changing H0 path. Its correction-only uncertainty contribution is one summand of the braid's triangle-bound score. Actual gate errors can cancel differently in a braid/composite sequence, so this is not a lower bound on actual braid fidelity.

Absolute full-Hamiltonian norm-squared costs are also not identical to sums of coupling-drive squares. For constant gap g and an ideal constant direct pulse, H_idle and B commute, giving ||H_idle+hB||=g+h on the modeled full fixed-parity space, while the source braid leg's active terms anticommute. Thus direct full-Hamiltonian action is g^2 T+2gA_d+A_d^2/T, compared with the braid's g^2 T+9A_d^2/T lower infimum. Do not claim universal fixed-T dominance of that different functional. Routing, idling, coherence, preparation and actual hardware power also require their actual prices.

Robust composite direct pulses are legitimate whenever the actual available zero-mode couplers provide the needed axes/sign controls. A noncommuting routed logical library can provide such axes; a single coupler does not automatically provide all of them. Do not forbid that search/control class, or charge it fictional per-old-task recalibration while the learned topology controller reuses U. Conversely, do not claim an unimplemented composite sequence is already certified. All competitors obey the same paid preparation/readout, response/domain, uncertainty/drift and actuator access.

This comparator narrows claims about the current supplied-actuator example. It does not show that topology has no possible protected-acquisition benefit under other actual noise/actuator/formation regimes. In particular its direct pulse is exposed to calibration uncertainty; a different genuinely passive protection mechanism can have different economics. The current response class has not established such an advantage.

## 9. Full lifetime accounting and the open M0 step

| Item | Required account for every method |
|---|---|
| Model/identifiability | Justification of the cubic/range/channel/sign/memoryless prior; separate SPAM, drift, delay, leakage and decoherence assumptions/observations; no finite-fit-to-global-law promotion |
| Quantum evidence | Every preparation, pulse duration/amplitude/ramp, destructive readout, routing/spectator state, failed/conflicting batch and refresh; count actual Bernoulli records |
| Statistical maintenance | Batch/prefix confidence allocation, interval inversion, polygon propagation, drift expansion and numerical enclosure; confidence consumed by channels/rounds, not hidden independence |
| Shared learning | Actual operation-generation feedback and fitting, all refused proposals, waveform/network/inverse evaluation, normalization, profile extrema and slew/precision checks |
| Old retention | Gate/word/error/deadline contracts, A_a,C_a,T_a, U age/domain, remaining error slack, preparation/readout and idle protection; add newly admitted capabilities to future obligations |
| Physical execution | Actual command and gap/correction couplers, duration/coherence, amplitude/slew and whichever hardware cost is justified; ideal norm action is labeled as modeled |
| State/storage | Uncertainty polygons, outcomes, contexts, old operations/programs, network and optimizer, modular/frozen controller instances and precision; common gain transfer only where justified |

Freezing a waveform avoids some shared-update amplitude expansion; it does not freeze the physical plant or eliminate drift refresh. Modular controls may select robust pulses valid over K and spend less data but more execution time/work. A direct robust optimizer can choose (23), numerical optimal control or legal composites from the same information. Same-rule reimplementation alone is not an independent historical-novelty verdict, but the explicitly different direct idle operation must remain a strong competitor.

**OPEN M0 interface:** which genuinely new executable operations are acquired at successive stages, from what task feedback, with what growing freedom, and why does topology causally improve retention or full relevant cost against these controls? The present calibration derivation solves neither operation creation nor that advantage. It supplies an honest finite-observation support mechanism and exposes a potentially unfavorable baseline. Root/K's formation construction must provide the new operations rather than relabel existing exchanges or count only their speed changes.

One informative use of this artifact is to freeze, for that same selected operation-acquisition mechanism, the actual response prior/domain, paid batch policy, U-based old-contract check, new-operation evidence and direct robust/composite comparator before an authorized process. There is no process authorization here. No data, narrowed U, learned operation or milestone passage is claimed.

Primary-source boundary: personally entered [Granade et al., original arXiv:1207.1655v2, §§II, V, VII.1–VII.2](https://arxiv.org/html/1207.1655v2), including the binary preparation/evolution/measurement likelihood, known versus unknown T2 and credible/confidence distinction. This supports the observation scaffolding. The cubic strips, endpoint-envelope proof, alpha-spending, drift propagation and direct-pulse calculations above are reconstructions; no experimental or performance result from that paper is imported. Details and exposure chronology are in the assigned p-reading.md.
