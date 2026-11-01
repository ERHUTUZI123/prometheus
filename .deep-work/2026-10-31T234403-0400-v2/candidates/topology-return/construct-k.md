# Shared phase-map learning with a finite-rate semantic certificate

AGENT PROPOSAL / RECONSTRUCTION. Same V2; first substantive saved construction in this topology-return allocation. No experiment, training, import, compilation, installation or hardware claim. The eight learning steps below are an analytic illustration. Historical novelty and favorable total cost against an independent strong rival remain OPEN.

**Priority received and read on disk:** MAIN=TOPOLOGY; PRIORITY=USER_LOCKED; MAIN_CHANGE=HUMAN_ONLY; ONLINE=USER-PARK; PERSISTENT_STATE=PAUSED_STANDALONE_RESERVE. This is topology work, not resumed Persistent State work. Only the human can change MAIN.

## 1. Intended change and exact limitation

Can a shared continuous implementation learn a substantially faster useful computation while retaining the semantics of every composition of old operations, without retaining or replaying every old episode? The proposed organization separates a winding-based operation identity from the geometry and speed of its attracting execution. Continuous learning changes that geometry. A finite-rate invariant tube, additional to topology, protects actual execution.

The worked result is **new finite-budget accessibility of an existing function**: an increment that initially fails a specified short deadline later succeeds at that deadline. It is not a new permutation, an enlarged reachable group, discovery of an unknown task function, or quantum computation. Old tasks keep their original longer deadlines throughout. This directly addresses useful shared implementation change and retention, but is only PARTIAL evidence toward the larger hypothesis of a favorable total acquisition/retention tradeoff. No comparison has been run.

The prior topology constructor preserved a measured finite permutation library under a tiny shared coefficient step; its task answers factored through that library and its execution work did not improve. Here the derived consequence concerns actual elapsed dynamical time under an unchanged force bound. The old endpoint-factorization observation still applies to discrete answers after ignoring time. It cannot, by itself, decide this different resource claim.

## 2. State, supplied structure and information

Fix a finite integer M>=3. The carried state is a classical coordinate

    q in R/(2*pi*M Z).

There are M physically/readout-distinct wells at q=2*pi*j, j in Z/MZ. Although the force uses exp(i*q), the readout resolves the larger circumference, not q modulo 2*pi. Reading only exp(i*q) would erase j and would not implement the proposed computation. Unbounded integer memory is not supplied: increasing M increases coordinate/readout range and precision requirements.

An operation supplies a control phase s in R/Z. The common learned map is

    h_theta(s) = rho_theta(s) exp(i*phi_theta(s)),   rho_theta(s)>0,
    phi_theta(s+1)=phi_theta(s)+2*pi,
    h_theta(0)>0 real.

The last condition fixes the home wells. All old and new commands use the SAME theta and h; no old network is frozen or copied. A general neural realization can use a periodic network p_theta and positive rho_theta, with phi=2*pi*s+p_theta(s)-p_theta(0). Degree one and nonzero amplitude are then supplied architectural restrictions, not learned discoveries. Computing trig functions, generating the command clock, the M-well substrate and its readout are also supplied primitives. The worked realization below is a one-weight piecewise-linear phase network plus a fixed complex exponential.

Executed dynamics, with damping and force scales normalized, are

    dq/dt = Im(h_theta(s(t))*exp(-i*q)) + eta(t).                 (1)

This is gradient descent in the periodic potential -rho*cos(q-phi), with an optional bounded execution perturbation |eta|<=nu. It is classical dissipative dynamics. A digital simulation pays numerical integration and state precision; no physical or quantum energy advantage follows.

Learner state is (theta, carried q, command library, deadline contracts, retained rate certificate, new-task feedback, cost record). An old command entry stores its input phase trajectory, direction, duration and semantic label +/-1. The new task asks for the already known increment at a shorter fixed deadline. Its feedback is endpoint success/failure at that deadline. Internal phase, parameters and derivatives of the learner's own implementation are inspectable and paid. These do not supply hidden environment information. An unknown physical plant would require its own identification/error bounds before the certificate could be used.

## 3. Topological identity and the missing execution condition

Every h_theta above has degree one as a circle map. For a closed input path of winding w, a continuous lift of arg h_theta changes by 2*pi*w. Nonzero-map homotopies preserve this integer. This is the topological part of the retention rule. It permits large geometric changes without changing the operation identity.

Topology alone does not imply that q actually follows that lift at finite speed. Choose delta in (0,pi/2), and write e=q-phi-2*pi*j along a proposed lifted execution. Almost everywhere,

    de/dt = -rho*sin(e) - dphi/dt + eta.                       (2)

If throughout that execution

    rho_min*sin(delta) > omega_max + nu,
    |dphi/dt|<=omega_max,                                    (3)

then [-delta,delta] is forward invariant. At its upper boundary the derivative is negative; at its lower boundary it is positive. Piecewise differentiability of phi suffices. Starting in the tube, the command therefore ends within delta of well j+w. The nearest-well decoder returns j+w modulo M. A fixed readout error smaller than pi-delta cannot change that result.

This is a direct barrier/invariance derivation, not a claim that arbitrary noise is harmless. Gaussian/Brownian noise is unbounded and does not satisfy this deterministic forever guarantee. Finite-horizon probabilistic bounds would be an additional obligation. Numerical rounding and discretization likewise need a declared effective error or an independent certified integrator; a floating simulation is not automatically covered by (3).

## 4. Composition, shared writes, migration and reset

All command paths begin and end at s=0 in R/Z, so h is positive real at the interface. At an endpoint, the next command starts with the same actual q, not a reseeded root or desired well. Re-indexing the proof's j to the decoded new well is mathematical bookkeeping, not a physical write. The residual error remains inside the same tube. Induction therefore preserves every finite word in certified increments and decrements, with no accumulated phase error outside the tube under the stated bounded-perturbation assumptions. This is a finite-state cyclic operation family, not a universal computational language.

Parameter writes occur with the control at home. Because h_theta(0)>0 real for every admitted theta, all home well centers stay at 2*pi*j. In the worked family rho(0)=1 too, so updating theta does not even change the home force law. There is no anchor migration or free branch assignment hidden here. General maps that move the home phase must physically transport q and the decoder; their phase-rate, migration duration and preparation costs must be charged and certified. That more general case is not used in the example.

No exact settling/reset is required between certified words: the tube is already an admissible input set. If a task requires a smaller initial error delta0, a home dwell is paid. For rho=1, eta=0 and |e|<pi, the exact home relaxation is

    tan(|e(t)|/2)=tan(|e(0)|/2)*exp(-t).

Thus finite tolerance, not exact zero, specifies the dwell. Changing the desired logical input j needs actual preparation operations (or explicitly supplied preparation access); reading j or knowing its label is not a free actuator. Inverse commands consume their full time. Simulating M separately prepared inputs costs M trajectories unless translation symmetry is used only as a mathematical proof, as below.

## 5. A complete worked learning cycle

### 5.1 Shared neural phase map and stable old task

Set epsilon=1/100 and define the continuous lift

    f_epsilon(s)=0                             for 0<=s<=1-epsilon,
                 (s-1+epsilon)/epsilon         for 1-epsilon<s<=1,
    phi_lambda(s)=2*pi*((1-lambda)*f_epsilon(s)+lambda*s),
    h_lambda(s)=exp(i*phi_lambda(s)),           0<=lambda<=1.    (4)

Here f is a single scaled ReLU on the command chart; h is continuous across the chart seam, although its derivative can jump. Its degree is one for every lambda. The one shared weight lambda changes the force throughout the interiors of every old command, from a long hold followed by a concentrated phase turn to a spread phase turn. Its change is not a relabeling of the endpoint table.

Old +/- commands traverse s linearly through one circle in either direction over the unchanged duration

    T_old=10*pi/epsilon.

Take delta=pi/6. Since rho=1 and

    max_s |dphi_lambda/ds|
      =2*pi*(lambda+(1-lambda)/epsilon) <=2*pi/epsilon,

every old command, for every lambda in [0,1], has omega_max<=1/5. The old tube condition has slack 1/2-1/5=3/10 before perturbations. One uniform family certificate therefore protects all old words during every forthcoming update. Old deadlines and outputs do not change.

### 5.2 New deadline and a proved initial failure

The new requested interface is increment within

    T_new=6*pi,

using the same linear input phase sweep and the same force bound |dq/dt|<=1 when eta=0. Initially lambda=0. Starting at any well center, h=1 for the first (1-epsilon)*T_new, so q stays there. During the remaining epsilon*T_new, its total displacement is at most

    epsilon*T_new=3*pi/50 < pi.

It therefore decodes to the original well and fails the requested increment. This is an actual execution bound, not failure of a sufficient certificate misreported as failure of the computation. A start already within delta also cannot reach the adjacent decoder region: delta+3*pi/50<pi. Home relaxation during the initial hold only reduces that initial error.

### 5.3 Trigger, candidate formation and learning signal

The observed missed deadline triggers one candidate: a faster implementation of the known +1 operation, obtained by changing the SAME lambda used by the old library. It does not trigger discovery of a new winding or a hidden target function. The supplied phase-network family includes both the bad initial schedule and a useful spread schedule; this strong prior is exposed.

The task label identifies the requested integer operation and its deadline. The learner then uses a calculable implementation objective from its OWN phase map,

    C(lambda)=integral_0^1 (dphi_lambda/ds)^2 ds
             =4*pi^2*[1+(1/epsilon-1)*(1-lambda)^2].            (5)

This signal needs neither teacher intermediate states nor new environment labels. It encourages distributing phase motion instead of concentrating it. It is still a proxy until (3) proves actual deadline performance. For this family, use its declared normalization

    L_impl=(C/(4*pi^2)-1)/(2*(1/epsilon-1))
           =.5*(1-lambda)^2.

Propose lambda'=clip_[0,1](lambda-.5*dL_impl/dlambda). Keep the proposal only if the old family/rate certificate passes; otherwise backtrack/reject and count the work. In this analytic case every proposal is admitted, and

    lambda_n=1-2^(-n).                                      (6)

The endpoint task loss is not incorrectly differentiated through the nearest-well decoder. Full finite-time smooth execution gradients could also be used before decoding, but none are needed or claimed here. The fixed internal objective, its normalization and learning rate are a designed optimizer, available to every rival.

### 5.4 Eight analytic updates imply actual short-deadline success

At n=8, lambda=255/256. On the T_new command,

    omega_max = (1/3)*(1+99/256)=355/768 < 1/2.

Thus (3) passes with exact slack 29/768 at delta=pi/6 when eta=0, and also for any bounded perturbation nu<29/768. Starting in the tube, the actual coordinate ends within pi/6 of the next well and the decoder returns +1. All old commands retained their original semantics at every intervening update. Both force amplitude and M stayed fixed. The certified time ratio T_old/T_new is 500/3, about 166.7; these are dimensionless dynamical times, not measured CPU seconds or an empirical speedup against a trained baseline.

One may register the new shorter-duration interface only after its rate certificate passes and its paid endpoint check is obtained. The foregoing is a manual derivation of what such a check should observe. No pilot was executed and no observed success record exists. The construction changes finite-budget accessibility of the existing increment; it does not claim a new abstract operation. The arbitrary poor initial schedule is an illustrative witness, not evidence that ordinary learned systems typically suffer this factor.

## 6. General learning/retention law and complete costs

The broader candidate is to train a shared nonzero phase map by task and internal implementation losses while retaining registered winding semantics. For each proposed theta update:

1. Differentiate the learner's own finite-time execution or a declared implementation objective. Record which task feedback, internal traces and derivative access were used.
2. Certify the proposed map homotopy stays nonzero on the retained control domain, preserves the home interface, and satisfies an old-operation finite-rate bound. An explicit degree-one positive-amplitude parameterization pays for this restriction architecturally. An unrestricted complex NN must instead pay for nonzero homotopy certification; finite samples do not establish it.
3. Install at home, with physical migration if required. Reject or backtrack if any necessary certificate fails. New semantics cannot be obtained by secretly crossing a protected singularity.
4. Admit a new deadline/path interface only after its own finite-time contract is supported. Protecting winding does not certify a new latency, noise level or analog output.

In the example, initial topological certification is the explicit lift formula; the old rate bound is one uniform analytic inequality. Each of eight proposals needs constant-size arithmetic, one gradient and a parameter write. New admission requires a constant-size maximum-slope comparison. No old episode replay is required by the proved model. At least initial-failure and final-endpoint queries are charged if actually performed. Clock generation, phase-map evaluation, integration/physical elapsed time, finite readout, preparation and inverse/dwell work remain execution costs. Storage includes one shared parameter plus the fixed phase basis, M/range information, command durations/paths and certificates; it does not silently discard the supplied architecture cost.

For a general NN, derivative bounds and nonzero-map bounds can be expensive or overly conservative. If a global control-domain certificate fails, covering every retained path may require work proportional to all its segments and network evaluations, with subdivision/precision growth. This is not a free replacement of old replay. A family of old programs can share one generator/domain certificate, but acquiring that certificate and every recertification/rejected update count. All state migration, model identification, failed task queries and simulator steps count as well. No exponential decoder or solver is hidden in the worked one-weight case; no polynomial scaling theorem is asserted for rich networks.

## 7. Strong same-information alternatives

**Ordinary same-actuator optimal control:** it receives the exact force law, phase family, derivative access, initial state/noise bounds and task/deadline. It can directly choose lambda=1 without eight gradient steps, or optimize the phase waveform conventionally and prove the same invariant tube. This is an especially strong independent algorithmic alternative, not a weak-SGD baseline. The present example does not beat it. Its retention reasoning can use the same known physics; being classical is no veto or advantage.

**Runtime retiming/lazy execution:** if the input clock is under the controller's control, it can spend most of its time in the small active segment of the INITIAL map and traverse its plateau quickly, subject to the same input-speed/actuator constraints. This may realize the useful force waveform without changing lambda at all. A fixed externally imposed clock is a legitimate distinct task interface only when specified for all methods in advance; it cannot be added afterward to exclude this rival. The worked construction establishes shared-map adaptation for the stated linear sweep, not that such adaptation is always necessary.

**Frozen old implementation plus a new module:** this keeps old timing/semantics by construction and learns a fast module separately. It may incur extra storage/routing but zero old recertification. Match useful execution constraints and report the tradeoff; do not prohibit it solely to force shared learning. Its method changes the shared-implementation condition, but it remains a legitimate total-cost competitor.

**Digital counter:** if only symbolic answers and deadlines measured in ordinary software operations matter, an M-state register increments directly and can be far cheaper than this pump. It is mandatory as an output-only rival. A same-force physical-coordinate task instead requires it to control the same q; it receives no free q-setting actuator. These are separate regimes, not a post hoc hardware excuse.

**Same-mechanism implementation:** a conventional classical implementation of this exact phase-map law must match the mathematical result. That correspondence can check attribution/cost; it is not an independent victory, a novelty refutation by universal simulability, or evidence of quantum advantage.

## 8. What remains scientifically open

The checked mathematical prospect is specific: topology supplies a large class of semantic-preserving shared edits, while an independent finite-rate condition connects learned geometry to actual executable computation. Here a substantial useful edit and all-old-word retention are derived together; neither is inferred from parameter displacement or a tiny sensitivity decrease.

The weakest step in a favorable-benefit claim is total cost against ordinary waveform optimization/retiming and modular/digital implementations. The one-weight witness supplies no advantage there. Its degree-one architecture also fixes the operation algebra in advance. It cannot autonomously acquire an unknown winding target, noncommuting operation family, or irreversible computation. Extending it by merely asserting that a richer NN learns these objects would hide the actual formation problem.

A consequential longer-term prospect is a learned continuous implementation whose semantic certificates cover many reused operations, allowing shared geometric changes that improve actual deadline/noise contracts with less total old-retention work than the best alternatives. Whether the certificate remains cheaper and permissive enough in a rich task-driven operation family is OPEN. The current finite construction makes that question inspectable; it does not answer it. No new experiment is authorized by this artifact.

CLAIM | STATUS | EVIDENCE | WEAKEST REMAINING LINK
--- | --- | --- | ---
Winding is retained under the stated phase-map family | RECONSTRUCTION, algebraic/topological | explicit degree-one lift, positive amplitude | supplied family fixes the integer semantics
All finite old words execute correctly under the tube assumptions | RECONSTRUCTION, analytic | boundary signs in (2)–(3), common carried state | numerical/hardware error and readout must satisfy declared bounds
Shared adaptation can make the specified failed deadline succeed | RECONSTRUCTION, analytic illustration, UNEXECUTED | initial displacement bound; (5)–(6); exact 29/768 margin | designed poor initialization/family, not empirical typicality
Topology yields lower total retention/acquisition cost than strong ordinary methods | OPEN; current construction PARTIAL | explicit costs and rivals above | optimal control/retiming may tie or dominate; rich certificate cost unknown

Next single decision for root: assess whether the finite-rate certificate and shared-learning distinction merit development inside the current topology lineage; preserve the strong controller/retiming rival and the absence of a cost win. This note does not close, change or PARK MAIN.
