# Mobility repair: learn shared flow directions while retaining a root-cover interface

AGENT PROPOSAL / RECONSTRUCTION, same USER_LOCKED_TOPOLOGY V2. This continues `construct-k.md` after the actual `review-023.md` and `retiming-frontier.md`; it does not replace MAIN or reopen Persistent State/Online. No training, integration, experiment, import, compilation or installation. All finite updates and outcomes below are analytic illustrations. No novelty, physical advantage or favorable total-cost result is claimed.

## 1. Repair and provenance

The previous shared phase map could be emulated by a known inverse clock. Root proposed the substantive repair: keep an attracting potential's stationary covering fixed while learning a shared positive mobility, thereby changing state-space directions rather than only traversal speed. I derived the explicit two-dimensional construction, local learning signal, finite execution/restoration bounds and resource limits below. I subsequently read root's saved `mobility-root.md`; our residual argument uses the same underlying energy method, with the factor-of-two convention below made explicit. P's separate proof support has not been read here.

The intended important gap is **continuing useful change to an executed shared implementation while retaining reusable old operation semantics**. The finite case improves the deadline for an existing swap. It does not discover a new abstract function, acquire an unknown root topology, or prove a learning-cost advantage. Its value beyond the phase-map prototype is an actual direction-changing plasticity rule, an old certificate uniform over its updates, and a complete new-operation interface including return to the old error set.

## 2. Model and actual state

Let z=x+iy be the carried classical state in R² and let the command u be complex. For registered operations use the existing circle u=exp(i*phi). Allowing other waveforms with |u|<=1 will not remove the initial nominal lower bound proved below. This is an explicit mathematical control domain, not a measured hardware restriction or a retrospective clock bound. A rival with larger allowed |u|, additive force, direct state setting, a different potential or a digital output register is a different legitimate interface and must be evaluated when available.

Fix the owned potential

    P(z,u)=z²-u,       V0(z,u)=|P(z,u)|²/4.                     (1)

The factor 1/4 differs from root's 1/2 convention. Its simple roots over u!=0 are the two-sheeted minima cover. The already retained circle operation swaps the two home wells +/-1. We do not repeat any old root/path discovery or experiment. Home admissible components are

    X_plus/minus = {z: |z²-1|<=a, +/-Re(z)>0},    a=1/4.        (2)

The decoder is sign(Re z). In X_plus/minus, |Re z|>=sqrt(3/4); a readout error smaller than that margin cannot flip the answer. The actual initial component is an incumbent capability/preparation assumption, not a root-solving oracle or free physical initialization. Preparing another logical input, restoring after a failed trial and checking preparation errors all cost execution. No global convergence from arbitrary z is claimed.

The learned executor is

    z_dot = -c(z,u,b) M_b(z) grad V0(z,u) + xi(t),
    ||xi||<=nu,
    c=min(1, Fmax/||M_b grad V0||),        Fmax=1,               (3)

with c=1 when the denominator is zero. This common force cap is available to all methods. It is inactive throughout the certified old and successful new executions below. Including it makes the failed initial implementation subject to the SAME force budget; saturation cannot increase its angular speed. Clock/control generation and changing mobility are not free energy sources: their costs remain separate.

### Shared bounded mobility, without a zero-factor gradient trick

For r=|z|>0 define unit radial n=z/r and tangent t=i*z/r, interpreted as real columns. Choose epsilon=1/100 and one shared trainable coefficient b in [0,1]. On r>=1/2 set

    M_b = m_r n n^T + m_t t t^T,
    m_r=1-(1-epsilon)b,    m_t=epsilon+(1-epsilon)b.             (4)

Thus epsilon I<=M_b<=I, trace M_b=1+epsilon, and both radial and tangential execution change when b changes. The trace is an architecture/resource proxy, not established physical cost. It cannot be used to ban a scalar M=I competitor under a spectral/force-only budget.

For a globally regular definition, let chi(r)=0 for r<=1/4, chi(r)=1 for r>=1/2, with chi=3s²-2s³, s=4r-1, in between. Set

    M_b=epsilon I+(1-epsilon)chi(r)
                  [(1-b)n n^T+b t t^T],                       (5)

and M_b(0)=epsilon I. This is locally Lipschitz, positive definite and bounded; no singular polar feature is evaluated at the origin. At a given z its trace is independent of b. The certified tubes all lie at r>1/2, where (4) is exact. In particular, at b=0 tangential mobility is epsilon even inside the cutoff.

Equation (5) has an epsilon I+L L^T factorization, but we train the mixture coefficient b directly and differentiate M_b, not zero-initialized L columns. Differentiating L L^T at a zero L would give zero gradient and would not implement the learning rule below. A factorization for execution is allowed and paid; its square-root derivative at b=0 is not silently used. A bounded neural coefficient field b_theta(z,u) is a broader realization with the same old spectral guarantee, but its new fast performance is not inferred from the scalar example.

## 3. This changes the flow image

At a fixed held input u=1, use z=exp(i*pi/48), inside the old interface (2). The gradient's radial and tangential components are

    g_r=1-cos(pi/24)>0,    g_t=sin(pi/24)>0.

The cap is inactive. The determinant, in the radial/tangent frame, of the b=0 and b>0 velocity vectors is

    (m_t-epsilon*m_r) g_r g_t
      =b(1-epsilon²) g_r g_t !=0.                             (6)

Their directions therefore differ at the SAME state and input. A scalar reparameterization of that held-input trajectory cannot reproduce the change. This is a local statement about vector-field directions, not a denial of conventional control: a planner allowed to change u, add feedforward force, or change M may find an equally good or cheaper motion.

## 4. Uniform old-operation protection, including updates

Write F=P(z,u), R=|F|. With the real Jacobian J of z², JJ^T=4|z|² I and grad V0=J^T F/2. On R<=a, |u|=1,

    sqrt(1-a)<=|z|<=sqrt(1+a),
    ||grad V0||=|z| R <=sqrt(5/4)/4<1.

Thus c=1 for every b. The residual satisfies, at R>0,

    R_dot <= -2 epsilon |z|² R + |u_dot|+2|z| nu
           <= -2 epsilon(1-a) R+|u_dot|+2sqrt(1+a)nu.           (7)

Choose all original circle commands to have |u_dot|=1/1000, duration T_old=2000*pi, and nu<=1/1000. At the boundary R=a=1/4, the restoring term is 3/800. Since 2sqrt(5/4)=sqrt(5)<9/4, disturbance plus drive is at most 1/1000+9/4000=13/4000. The strict residual-boundary slack is at least

    3/800-13/4000=1/2000.                                    (8)

This is a complete uniform certificate over all b in [0,1], even time-varying b with a well-posed executor. The potential does not depend on b, so no omitted partial_b V0 term appears. The roots and home decoder anchors stay fixed during learning; no anchor migration or branch reset is needed.

The residual tube avoids z=0 and lifts the root cover continuously. Its two components cannot be exchanged without leaving the tube; after one circle their continuation exchanges home components. All finite old words therefore compose on the SAME carried z with no exact settling or free reset. Reverse paths consume their time; an input in the other component needs actual preparation. Uniform bounded execution error, rather than stochastic noise of unbounded support, is assumed. Numerical integration requires a separately justified discretization-error budget.

This does not assume that products of positive matrices are symmetric positive. The proof uses residual energy/J M J^T. It is also local. Positive M preserves every stationary point of V0, including unwanted ones: at any strict wrong minimum with Hessian H>0, -M H is similar to a negative definite symmetric matrix and is locally attracting. I read the actual after013 `root-composition-attack.md`; this construction does not evade its stable wrong-minimum counterexample, solve SAT, or grant free basin access. Here z=0 itself is an inadmissible stationary point, outside (2).

## 5. New deadline: actual failure, learning and reusable success

### 5.1 The requested computation and initial failure

The new task asks for the retained swap within a shorter deadline, with its result back in (2) so old programs can immediately resume. Define its prospective execution as one circle move of duration 4*pi followed by a PAID home hold of duration 8:

    T_new=4*pi+8.                                            (9)

Start nominally at +1 with b=0. For any |u|<=1 waveform, not just the linear clock, polar differentiation of (1)–(5) gives

    |alpha_dot|<=epsilon,    alpha=arg z,                      (10)

when xi=0; force saturation can only reduce it. With finite bounded control and r(0)>0 the radial equation does not reach r=0 in finite time. Therefore over T_new the angular displacement is at most epsilon*(4*pi+8)<pi/2. The state cannot decode to the negative well. This is actual nominal failure, not absence of a certificate miscalled failure. It also defeats mere scalar retiming of the old implementation within the stated unit-disc control domain. A failed physical trial could leave the old admissible tube; if performed, its restoration/preparation must be charged. The analytic construction below does not pretend such a trial was executed and rolled back for free.

### 5.2 Task-triggered formation of a mobility target

The learner receives the request for a faster realization of its KNOWN swap. Model-predicted failure via (10), or a paid actual failure, triggers mobility adaptation. No unknown task function or hidden intermediate branch label is inferred. The own relation and new command supply an internal demanded root velocity

    v_star = -(P_z)^(-1) P_u u_dot = u_dot/(2z_star).           (11)

For a unit-circle root and forward circle command, its direction d=v_star/|v_star| is tangent to that root. Using a registered/model anchor is a virtual owned-source calculation, not setting the physical z. Computing/validating it and its precision costs work and is equally available to a conventional planner. If the physical potential/model is uncertain, this target is not magically correct; its error must enter the execution bounds.

Form the demanded mobility projector

    M_d=epsilon I+(1-epsilon)d d^T.

Fit the shared b at this virtual home probe using

    L_mob(b)=||M_b-M_d||_F²/[4(1-epsilon)²]
            =.5*(1-b)².                                    (12)

For this supplied radial/tangent feature family, this one target generalizes over the circle by the exact geometric parameterization. That is a strong designed equivariance prior; one home sample does not teach an arbitrary neural field its global behavior. The learned object is the shared allocation of mobility, subsequently used at every old/new state. The target comes from an actual requested operation's internal differential, rather than a supplied latent bit or a claim that a network discovers an unspecified preconditioner.

Use two declared projected-gradient steps b<-clip_[0,1](b-.5*(b-1)). From b=0 they give b=1/2 and b=3/4. Derivatives are nonzero initially. These are two manual analytic updates, NOT observed training outcomes. Old certificate (8) remains valid during both shared writes. Directly selecting b=3/4 or b=1 from the same known model is a legitimate cheaper controller; no discovery-cost win is claimed.

### 5.3 Actual move certificate after the two updates

On r>1/2, write e=alpha-phi/2 for the branch lifted from the actual starting component. As long as the cap is inactive,

    r_dot = -m_r*r*(r²-cos(2e)) + xi_r,
    e_dot = -m_t*sin(2e)-phi_dot/2+xi_t/r.                    (13)

At b=3/4, m_r=103/400 and m_t=301/400. On the move, phi_dot=1/2. Take the rectangle

    17/20<=r<=23/20,        |e|<=delta=pi/12.                 (14)

Every actual input in (2) lies in (14): its radius is between sqrt(3/4) and sqrt(5/4), and its branch angle has magnitude less than pi/12. At the phase boundaries the nominal inward slack is

    (301/400)sin(pi/6)-1/4=101/800.

With nu<=1/1000, |xi_t/r|<=20nu/17, still strictly smaller. At the lower radial boundary cos(2e)>=sqrt(3)/2>43/50, giving inward radial drift greater than

    (103/400)*(17/20)*(11/80)=19261/640000>nu.

At the upper boundary the inward drift is at least (103/400)*(23/20)*(129/400)>nu. Also throughout (14), R²=r^4+1-2r² cos(2e)<49/100: using cos>=43/50, the convex quadratic in r² has its maximum at an endpoint; at r=23/20 the upper bound is .47430625<.49, and the lower endpoint is smaller. Thus R<7/10 and ||grad V0||=rR<161/200<1. The force cap is indeed inactive, closing rather than assuming the tube argument.

After the move the state is on the negative lifted branch, with R<7/10. Merely stopping here would not yet establish reusable composition on (2).

### 5.4 Paid return to the old interface

Hold u=1 for time 8, keeping b=3/4. On R<=7/10, |z|²>=3/10, |z|<4/3 and min eigenvalue M=103/400. The force is below (4/3)*(7/10)=14/15<1. Equation (7), with this sharper mobility bound, gives

    R_dot <= -(309/2000)R+(8/3)nu
           <= -(309/2000)(R-1/50).                           (15)

The outer R=7/10 boundary points inward. At time 8,

    R <=1/50+(7/10-1/50)exp(-309/250)<37/150<1/4.              (16)

For the strict middle inequality, exp(309/250)>exp(6/5)>3, already certified by the first five positive terms of its exponential series. No numerical integration is used. The hold stays in the same negative residual component because R<1 throughout. It therefore returns an actual negative state in X_minus with the original readout margin and no reset. Old commands can immediately follow; new commands can likewise be applied from either old component by symmetry. Preparation of a different logical input still costs work.

The newly acquired time contract must itself be retained under later learning. For example b in [3/5,3/4] keeps all the above move/restoration bounds: tangential phase slack is at least 13/250 before disturbance, radial mobility and restoration are no worse than the bounds used above. Subsequent updates can be projected into that interval. The abstract swap remains protected for all b, but its NEW short deadline does not. This distinction is essential; topology does not preserve arbitrary learned analog accuracy or speed for free.

## 6. Resource bounds, and why faster does not mean cheaper

No global gain was increased in the adaptation: M<=I always, its on-tube trace stays 1+epsilon, the potential and force cap stay fixed, and both radial/tangent responses change. The actual force-image change is (6). There is nevertheless no free energy claim.

Let g=grad V0 and let v_c=-c M g be the controlled velocity. Because M²<=M and c<=1,

    ||v_c||²<=c*g^T M g.

The energy identity is

    V0_dot=-c*g^T M g -.5 Re(conj(F)u_dot)+g·xi.               (17)

Thus controlled quadratic effort is bounded by initial potential plus absolute input work and disturbance work. For the successful move/restoration, initial V0<=1/64, R<=7/10 during the circle and |g|<14/15 throughout. A valid conservative bound is

    integral ||v_c||² dt
      <=1/64+(7/10)*pi+(14/15)nu*(4*pi+8).                    (18)

This is an upper bound, not actual measured effort or a comparison win. Physical dissipation, controller energy and the cost of realizing/changing mobility require an actual device model. In particular the fast circle has input quadratic effort integral |u_dot|²dt=pi; the original slow circle has pi/500. The time improvement costs a faster input waveform. All rivals get that waveform access and its cost.

The finite learning ledger includes two gradients/parameter writes, construction of one demanded-direction projector from owned derivatives, the prescribed feature computation, validation/certification and endpoint feedback if actually obtained. Registering a new interface needs a paid endpoint check in a real executor; none has happened here. Mathematical preparation is the existing input set, not an executed reset. A failed trial, restart, extra anchor computation, approximate derivative, numerical precision or controller-network evaluation must be counted. General neural fields can have substantial per-step/certificate costs even though the old spectral certificate is parameter-uniform.

For repeated use report at least

    C_total=C_model+training+certificate+storage/routing
            +preparation/migration+all command/control execution.

An actual positive tradeoff would require saved per-call cost to exceed its complete acquisition/maintenance overhead. No such saved-cost number has been established.

## 7. Strong alternatives and exact counterconditions

**Scalar M=I:** this is legal under the SAME spectral/force bounds. Its tangential response is at least as fast and radial restoration is stronger than the learned allocation in this example. It can execute the new command without the two updates and without an anisotropic network. A fixed trace restriction cannot be invented to remove it. If trace/conductance/parameter count corresponds to a real independently justified resource, report that additional cost axis for BOTH; it is presently only a model proxy. The worked construction therefore provides no cost win over this baseline.

**Ordinary adaptive preconditioning/model-based control:** give it the same V0, J, state, task/deadline, matrix family and force cap. It can construct (11)–(12), choose b directly or train its own SPD mobility. Positive-metric optimization is established scaffolding. A neural implementation of the same map does not by itself establish a different learning principle. Yang–Amari's original natural-gradient rule uses an inverse metric in parameter space; here the metric acts on executed state. That distinction identifies the proposed interface, not historical novelty.

**Retiming and full control:** retiming cannot change the direction in (6), and the initial angular bound (10) holds for any unit-disc control waveform. This is stronger than the earlier clock-only witness. A broader controller with different u amplitude, additive/feedforward actuation, modified potential, state-setting or an alternative physical realization may bypass that bound and remains legitimate if available. The unit-disc potential/force model is not evidence that real devices have exactly this interface.

**Frozen/modular and digital alternatives:** a separate fast controller can retain the old implementation without recertification; count its storage/routing and compare total resources. For symbolic outputs a two-state bit flip is much cheaper and mandatory. A same-physical-coordinate controller must actually move z, but does not lose other legitimate actuators merely because the candidate uses mobility.

**Wrong minima and model error:** a stationary wrong answer remains stationary under every invertible mobility. Unknown potential/semantic-model errors are not bounded hardware noise by declaration. If the source model, basin preparation or uniform constants are unavailable, the claimed execution certificate is unavailable too. This version makes no global solver claim.

## 8. Scientific boundary and next single decision

Derived: a shared direction-changing adaptation, a uniform old root-cover interface over every admitted update, initial fast-command failure not explained away by scalar retiming, and a successful new deadline with PAID interface restoration. These are analytic, unexecuted consequences within a supplied two-sheet family and a known-model learning scaffold. Both the topological identity and finite-time realization are explicit.

OPEN: whether directional shared plasticity plus topology-derived retention yields a favorable TOTAL learning/retention/resource frontier in a consequential family where the strongest scalar, adaptive preconditioning, waveform, modular and digital methods do not already dominate. This example exposes scalar-I dominance rather than concealing it. It does not refute the broader topology hypothesis, but it prevents promoting this particular witness into a resource advantage or general structure-formation result.

Next single decision for root: integrate the completed directional/return-interface mechanism and its scalar-I countercondition into the current topology lineage, then choose the one remaining cost-bearing uncertainty. No experiment is authorized by this note, and no MAIN change is enacted.
