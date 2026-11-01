# Mobility repair — metric, finite branch tracking, learning and cost audit

**RECONSTRUCTION / constructive mathematical support, 2026-11-01; no experiment.** Same V2. MAIN=TOPOLOGY; PRIORITY=USER_LOCKED; MAIN_CHANGE=HUMAN_ONLY; ONLINE=USER-PARK; PERSISTENT_STATE=PAUSED_STANDALONE_RESERVE. This supports the selected topology force-direction repair, not a new allocation. Only this new file is written. No training, numerical trajectory, old pilot, code audit, installation or model/resource change was executed. Effective provider mode is unknown.

Actual new reads: full `retiming-frontier.md` (the actual filename; the initially supplied root-retiming-frontier.md was absent), full Review023, and full `mobility-root.md`. Earlier actual topology/source reads are preserved in p-reading.md. Root's latest symmetric bounded-mobility chart was supplied by message after my residual derivation. No new historical-priority claim or independent-blindness claim is made.

## Judgment and coverage

Root's residual-energy certificate is correct under its **uniform, paid, local-neighborhood assumptions**. It supplies finite-time old branch tracking through arbitrary admitted mobility updates without assuming that a product of SPD matrices is Euclidean-positive. Below are explicit bounds and a constant-mobility metric comparison; an exact ODE learning derivative, including a zero-initialization trap; and a force-capped holomorphic example proving an actual directional resource distinction from scalar retiming.

Coverage: **DIRECT** for these conditional mathematical statements; **PARTIAL** for the complete retention/acquisition/resource hypothesis; **OUT_OF_SCOPE** for a new root-loop algebra, global minimization, a learned positive result, superiority over ordinary adaptive preconditioning or hardware energy savings. The directional example does not establish that its best mobility is autonomously learned or difficult for an ordinary controller to construct.

## 1. State, protection and allowed updates

Let P(x,u):R^2 x R^k -> R^2 represent an owned complex holomorphic relation in x, written in real coordinates. Let Gamma be the fixed acquired old control-path family, avoiding critical control values. A root lift a_i(t) is determined by its actual initial branch and u(t), not supplied as an execution oracle. Use

    V(x,u)=||P(x,u)||^2/2,
    g=grad_x V=J^T P,                 J=D_x P,
    x_dot=-M_theta(x,u) g + xi(t),    ||xi||<=nu.         (1)

The mobility is shared across every old/new path. Require local Lipschitz regularity in x, measurable/piecewise continuous bounded control/parameter updates, and

    m I <= M_theta(x,u) <= Mbar I,
    0<m<=Mbar<infinity                                      (2)

throughout the certified domain, uniformly in all admitted theta. A bounded smooth-output architecture can enforce (2), instead of checking it only at trace samples. For example L=ell R_theta/sqrt(1+||R_theta||_F^2) in M=epsilon I+LL^T gives m=epsilon and Mbar=epsilon+ell^2. The familiar unnormalized M=epsilon I+LL^T enforces the lower bound automatically; the upper bound still requires an actual bound on L.

P and its root covering remain fixed. Because M is invertible, the stationary points at fixed u are exactly those of V. Every simple root stays a stable local minimum. Other roots, nonzero critical points and their possibly harmful basin boundaries are not removed. Positive mobility does not cure the old wrong-attractor/wrong-minimum limitations, prove globally efficient root solving, or recover the right branch after an uncertified excursion.

At a theta write x stays continuous; no root or decoder-anchor migration is needed because V and its zeros do not change. Between writes the actual vector-field direction can change. The old finite-time/precision/deadline contract, rather than topology alone, decides whether that changed motion still realizes the old operation.

## 2. The tempting Euclidean proof is false

Writing e=x-a gives, for a fixed control, g=Hbar_e e with a symmetric positive average Hessian in a strongly convex local neighborhood. It is invalid to infer e^T M Hbar_e e>0 merely because both factors are SPD. For example

    H=diag(1,100),  M=[[1,9/10],[9/10,1]], e=(10,-1),
    M H e=(-80,-91),   e^T M H e=-709.                  (3)

Thus d(||e||^2/2)/dt=+709 in this quadratic example, although V decreases: (He)^T M(He)=8300. The product has positive eigenvalues, but its symmetric part need not be positive. Euclidean instantaneous distance can increase during stable preconditioned descent. No monotonic-distance argument should be substituted for a branch-tube proof.

## 3. Uniform residual tube: completed proof and quantitative bound

Let U_i(u) be disjoint moving neighborhoods of the intended simple roots along the old corridor. Here i is the carried lift label along the path, not a globally single-valued root section over a region with nontrivial monodromy. Their boundaries must be continuous enough for a first-exit argument. Assume uniformly in their closures

    sigma_min(J)>=s>0,  ||J||<=S,
    ||P_u u_dot||<=B,
    ||P||>=beta>0 on each moving neighborhood boundary. (4)

For a scalar complex holomorphic P, JJ^T=|P_z|^2 I, so s<=|P_z|<=S supplies the Jacobian bounds. These are mathematical uniform bounds, not minima over finitely sampled trace points. Their source/isolation/interval cost is real.

Set r(t)=||P(x(t),u(t))||. Wherever r>0,

    V_dot = -g^T M g + P^T P_u u_dot + g^T xi
          <= -m s^2 r^2 + r(B+S nu),
    D^+ r <= -k r + D,
    k=m s^2,        D=B+S nu.                         (5)

The inequality extends at r=0 in the upper-Dini/comparison sense. Consequently, while inside the neighborhoods,

    r(t) <= exp(-k t) r(0)
             + (D/k)(1-exp(-k t)).                    (6)

Choose a strict slack

    D/k < r0 < beta,       r(0)<=r0.                   (7)

Then r(t)<=r0, and exit from U_i is impossible: a first exit would have residual at least beta>r0. The state therefore stays in the component attached to the unique intended lifted root. At home its branch decoder is correct if U_i or the smaller residual sublevel lies inside the fixed readout margin. This checks root's stated proof and adds the full finite-time comparison (6).

No derivative of M appears in (5): V is fixed, and the dynamics enters only through its instantaneous dissipative quadratic form. Thus the same proof holds for state/input-dependent or time-varying M, and for abrupt parameter writes satisfying (2), subject to well-posedness and continuous state. This is the main benefit over a carelessly changing distance metric.

When u is held at home, replace B by zero. To reach a residual tolerance r_acc>D_home/k from residual r_in>r_acc, a sufficient dwell is

    T_dwell >= (1/k) log[(r_in-D_home/k)/(r_acc-D_home/k)],
    D_home=S nu.                                      (8)

This is a charged positive dwell, not exact settling in finite time. The formula applies only in the stated ordering; no dwell is required when the initial bound already meets the tolerance.

### Obtaining explicit local constants

Suppose the old lifted roots have minimum pairwise separation Delta>0; at their centers sigma_min(J)>=s_a>0 and ||J||<=S_a. If J is K-Lipschitz on radius-R balls with

    2R<Delta,   K R<s_a,

Taylor's formula supplies, for e=x-a_i(u),

    ||P(a_i+e,u)|| >= (s_a-K R/2)||e||,
    s=s_a-K R,    S=S_a+K R,
    beta=(s_a-K R/2) R.                              (9)

The first inequality uses the vector remainder bound, not a false integral-of-singular-values assertion. It also calibrates residual error to actual state error:

    ||x-a_i|| <= r/(s_a-K R/2).                       (10)

With known source P, the root separation, derivative/remainder and P_u bounds can be enclosed on the complete old corridors. Obtaining them can be difficult and conservative. If they are unavailable, retention remains conditional/unverified rather than passed.

### Composition and numerical limits

Old forward/inverse paths compose while the same rate/force/noise bounds, preparation and component/readout margin remain valid. Residual error stays inside the common invariant tube; an exact root reset after each word is unnecessary. This does not prove an arbitrary finite-precision integrator obeys (1). A numerical defect may be included in nu only after bounding the defect of a continuous interpolant, including interstep behavior. Alternatively, if an exact flow step of size h is followed by a state perturbation delta_x, its residual increase is at most S||delta_x||+K||delta_x||^2/2 in the certified domain; this must fit the available one-step slack in (6). Checking only residuals at sparse sample times does not certify no wrong-branch crossing between samples.

Newly learned shorter deadlines or tighter analog error tolerances are not protected merely by the old root topology. They become retained obligations only if their own finite-time constraints are included in later admissibility. A mobility that is always old-safe can still erase a previous speedup.

## 4. Constant-M distance metric, and why variable M needs extra terms

For a fixed SPD M and a locally strongly convex V with Hessian >=mu I, let

    rho=||e||_(M^-1),       e=x-a(u),
    d=xi-a_dot,             ||d||_(M^-1)<=D_M.

Then

    (rho^2/2)_dot = -e^T g + e^T M^-1 d
                    <= -mu m rho^2 + rho D_M,
    D^+ rho <= -mu m rho + D_M.                      (11)

A sufficient invariant metric ball rho<=R_M needs D_M<mu m R_M and sqrt(Mbar) R_M to fit inside the strongly convex Euclidean neighborhood. Since a_dot=-J(a,u)^-1 P_u u_dot, one may use D_M<=(nu+B/s_a)/sqrt(m). These are conservative explicit bounds.

If M varies in time/state, differentiating e^T M^-1 e/2 adds

    -(1/2) e^T M^-1 (dM/dt) M^-1 e.                  (12)

The total derivative includes M_x x_dot, M_u u_dot and parameter changes. A bound q_M=||M^-1/2 (dM/dt) M^-1/2||/2 reduces the certified contraction rate to at least mu m-q_M; a parameter jump can also change rho instantaneously. Omitting these terms invalidates (11). The residual-energy method (5) avoids this complication and needs no M inverse in execution.

## 5. Exact learning derivative and a repairable blind start

For fixed exogenous u(t), let f_theta=-M_theta g. Given a smooth actual terminal/output loss Phi and optional drift-effort penalty

    J_learning=Phi(x(T))+lambda/2 integral_0^T ||f_theta||^2 dt,

the sensitivity S_j=partial x/partial theta_j satisfies

    S_j_dot = f_x S_j + f_theta_j,   S_j(0)=0,
    f_x[v]=-(D_x M[v])g-M H v,     H=D_x^2 V,
    f_theta_j=-(partial_theta_j M)g.                  (13)

If initial preparation depends on theta, its actual derivative replaces the zero initial condition. No stopped-gradient or discrete branch-ID derivative is being invented.

The adjoint p has terminal p(T)=grad Phi and

    -p_dot=f_x^T p+partial_x(lambda ||f||^2/2),
    grad_theta J_learning
       = integral (p+lambda f)^T f_theta dt,           (14)

plus any explicit parameter/terminal costs. All partial derivatives include the mobility normalization/capping and shared state dependence. Target observations, repeated preparations, backward integration, saved trajectories/recomputation and higher derivatives are costs. For V=||P||^2/2,

    H=J^T J+sum_a P_a Hessian(P_a),                   (15)

so differentiating the actual flow generally needs Hessian-vector work, not only a free first derivative.

For M=epsilon I+LL^T, a variation gives

    delta f=-(delta L L^T+L delta L^T)g,
    grad_L integrand= -[(p+lambda f)g^T
                         +g(p+lambda f)^T] L.         (16)

If L is identically zero along the training trajectories, the task gradient through M is exactly zero. The positive floor epsilon alone does not solve acquisition from this initialization. A nonzero initialized factor or a different smooth SPD chart repairs this without extra latent labels.

Root's message supplies a useful two-dimensional chart with a nonzero derivative at isotropy. Put

    B_b=[[b1,b2],[b2,-b1]], d=sqrt(1+b1^2+b2^2),
    c=(m+Mbar)/2, a=(Mbar-m)/2,
    M_b=c I+(a/d) B_b.                               (17)

Its eigenvalues c +/- a||b||/sqrt(1+||b||^2) lie strictly between m and Mbar. With E_i=partial B_b/partial b_i,

    partial M_b/partial b_i
       =a(E_i/d-b_i B_b/d^3),
    partial f/partial b_i at b=0 = -a E_i g.           (18)

Thus the anisotropy gradient is generally nonzero at an isotropic start. A neural network can output b and receive ordinary gradients (14),(18). Trace is fixed at 2c in this chart; that is a representation choice, not a demonstrated physical resource law. Mbar I remains a legitimate competitor whenever the actual task allows it. Section 6 instead compares a common drift cap without excluding the strongest scalar retimer.

## 6. A finite, force-capped directional consequence in the holomorphic class

The elementary noncollinearity example already suffices: for V=||x||^2/2, starting at (1,1), M=diag(1,1/4) gives a curve x2=x1^(1/4), whereas every positive scalar mobility preserves x2/x1. It proves a direction change, not an improvement. The following stronger lemma gives a strictly shorter actual route under a common drift-speed cap, using a genuinely holomorphic P.

### Fixed potential, same start/target, same amplitude cap

Take the static slice u=1 of P(z,u)=z^2-u. Let z=X+iY, the intended known home root be a=1, e=(X-1,Y), and start at z=1+i. The endpoint task is to reach ||e||<=rho, with rho=1/4. There is no new loop/degree experiment here; this is a local resource lemma for the mobility rule. The potential and gradient are

    V=[(X^2+Y^2)^2-2X^2+2Y^2+1]/2,
    g=(2X(X^2+Y^2-1), 2Y(X^2+Y^2+1)).                (19)

Use the same cap ||x_dot||<=F, F>0, for both methods, with no noise in this lemma. This is explicitly a **modeled drift/velocity cap**, or a force cap only if the chosen physical units justify that identification. It is not an empirical hardware constraint. Mbar I and every scalar retimer remain legal subject to this same cap. Holding u fixed defines this local relaxation comparison; if u excursions are allowed, their possible shortcuts belong in the larger matched control frontier. The unrestricted-control/preconditioner baseline below also remains legal.

Define

    Hbar(e)=integral_0^1 Hessian V(a+t e) dt,
    M_0=F/||g|| I,
    M_1=F/||e|| Hbar(e)^-1.                           (20)

In the relevant region Hbar is SPD and g=Hbar e, hence

    -M_0 g=-F g/||g||,
    -M_1 g=-F e/||e||.                               (21)

Both drifts use exactly the same norm F. The second goes straight to the target ball and takes

    T_1=(1-rho)/F=3/(4F).                            (22)

This is not free access to an unknown new root: a=1 is the already known home anchor at this static control. For other moving/unknown branches, obtaining or approximating an analogous direction is a separate acquisition cost. The example proves a possible useful edit in the actual class, not a global root oracle.

Hbar is explicitly computable here. With e=(h,k), q=h^2+k^2 and e1=(1,0),

    Hbar=[4+2h+(2/3)q] I
           +2(e e1^T+e1 e^T)+(4/3)e e^T.             (23)

It requires ordinary arithmetic and a 2-by-2 inverse. A numerical or learned implementation must pay those costs or its approximation error.

### Exact best scalar-retiming frontier and a strict finite gap

For any positive scalar mobility alpha(t,z) I the geometric gradient trajectory is unchanged. From (19), starting at (1,1), its coordinates have the exact parameterization

    X(r)=1/sqrt(1+r-r^2),
    Y(r)=r/sqrt(1+r-r^2),      r decreases from 1 to 0. (24)

One derivation: under the accumulated scalar clock, A=X^2 and B=Y^2 satisfy B/A=r^2 and 1/A=1+r-r^2. Direct substitution in (19) verifies the curve. Scalar pauses do not shorten it, and the largest permitted speed F attains the minimum duration along it. Thus, writing r_rho for its first intersection with the target ball,

    T_scalar,min = (1/F) integral_(r_rho)^1
             sqrt(5)*sqrt(1+r^2)/[2(1+r-r^2)^(3/2)] dr. (25)

At r=1/2 it passes z_*=(2/sqrt(5),1/sqrt(5)), which is outside the rho=1/4 ball. Triangle inequalities for the two portions of the curve give the explicit bound

    F T_scalar,min
       >= sqrt(3-6/sqrt(5)) + sqrt(2-4/sqrt(5)) - 1/4,
    F(T_scalar,min-T_1)
       >= sqrt(3-6/sqrt(5)) + sqrt(2-4/sqrt(5)) - 1
       =: Delta_length >0.                           (26)

Strict positivity follows because z_* is not on the straight segment from 1+i to 1. No numerical rollout or tiny-proxy interpretation is involved: (26) is a finite deadline/route-length difference under equal drift amplitude, against the **best scalar retiming**, not just one fixed clock.

### The common spectral budget is feasible; trace need not be restricted

The Hessian eigenvalues are

    lambda_+/- = 4|z|^2 +/- 2|z^2-1|.

On X>=3/4 its lower eigenvalue is at least 3 sqrt(3)-4>1. To check this, put t=|z|^2 and use |z^2-1|^2<=(t+1)^2-9/4, then minimize 4t-2 sqrt((t+1)^2-9/4). On the compact rectangle swept by (24) and the radial path, an upper bound 14 suffices. For a single larger domain containing those paths and the small neighborhood used below, take D={3/4<=X<=9/8, 0<=Y<=9/8}; throughout D the upper bound 18 suffices, as does the lower bound 1. D contains the straight segments from a to its points. In D intersected with rho<=||e||<=2, both matrices in (20) therefore have eigenvalues between F/36 and 4F. Choosing the common architecture bounds m<F/36 and Mbar>4F admits both, with strict slack. These are compact-domain bounds, not a claimed global Hessian upper bound on the whole half-plane.

Mbar I plus norm capping is included in (25), since it is scalar and can attain speed F when Mbar is sufficiently large. The gain is not obtained by imposing equal trace and excluding that competitor. Convex interpolation M_theta=(1-theta)M_0+theta M_1, theta in [0,1], is SPD with the same spectral bounds and has drift norm <=F. This is a general bounded-SPD interpolation family; its trace can vary, so it is not identified with the fixed-trace chart (17). The zero-factor gradient trap is absent in this direct interpolation chart.

This is a local field construction on the declared static task domain. A shared global mobility network must represent/approximate and extend it while satisfying its old-corridor bounds, regularity and force cap; these obligations are not certified by (26) alone. The old residual tube can remain uniform in that global mobility because it uses only (2),(4), rather than the specific local route. The example does not establish that every intermediate mobility learns or globally reaches the intended branch from arbitrary initialization.

### A real task gradient is available at the isotropic start

This can be made stronger than an existence comparison of two manually selected endpoints. For the interpolation above, use the fixed-horizon task loss Phi=||x(T)-a||^2/2. At z0=1+i,

    f_0=-F(1,3)/sqrt(10),   f_1=(0,-F),
    partial_theta Phi at theta=0
        = -F T[1-3/sqrt(10)] + O(T^2).                (27)

Thus a positive mobility edit has the correct ordinary task-gradient sign for sufficiently small positive T. The gradient is of a calibrated distance to a known branch anchor, not a discrete branch-ID gradient or a new privileged target matrix.

Here is an explicit conservative horizon condition rather than an unspecified infinitesimal. In the radius-1/8 coordinate box around z0, ||e||>=7/8, Hessian V<=18 and Hessian V>=I. Consequently

    ||D f_0||<=144F/7,
    ||D(f_1-f_0)||<=152F/7,
    ||f_1-f_0||<=2F.

The sensitivity equation bounds the remainder in (27) by 84 F^2 T^2 whenever T<=min{1/(8F),7/(144F)}: use ||S(T)-T(f_1-f_0)(z0)||<=(508/7) F^2 T^2 and ||e(T)||<=9/8. With d0=1-3/sqrt(10)>0, the further restriction

    0<T<=d0/(168F)

gives partial_theta Phi|_0 <= -F T d0/2<0. A sufficiently small positive projected gradient step therefore improves the actual new endpoint task. This is a deliberately conservative finite mathematical learning condition, not a tuned hyperparameter result. The global speed-optimal endpoint M_1 is still a supplied constructive witness; no theorem says task learning will discover it efficiently.

## 7. What the directional gain does and does not cost

At a fixed control, dissipation satisfies

    integral g^T M g dt = V(start)-V(end)

for equal potential endpoints and zero disturbance. Calling this quantity a reduced energy expense merely because T falls would be false. For the explicitly declared squared-drift proxy, the norm-F routes instead have integral ||x_dot||^2 dt=F^2 T, so the shorter route also uses less of that modeled proxy. Neither identity specifies physical controller energy, friction cost or hardware power for changing mobility.

Every comparison must include:

| Cost | Concrete obligation |
|---|---|
| Potential/old library | Owned P and its source, branch preparation, root/corridor isolation, old operations and fixed decoder; common initial learning is not zero cost |
| Certificate | Uniform s,S,B,beta and neighborhoods; initial residual/readout margin; force/regularity/precision bounds; refresh if the admitted architecture, potential, paths or deadlines change |
| Field evaluation | V/P derivatives, the neural mobility output, normalization and products on every integration/actuation step; for d-state rank-r L, Mg=epsilon g+L(L^T g) costs O(dr) after evaluating L |
| Inverses | No M inverse is needed in (1),(5). Constant-M metric evaluation can factor M once; the explicit M_1 preconditioner needs the 2-by-2 inverse (23), or O(d^3) dense factorization in general, unless structure/reuse is proved |
| Training | Actual targets/rewards, preparations, all failed updates, forward traces, adjoint/sensitivity and Hessian-vector/field-Jacobian work; finite precision and the zero-factor initialization issue |
| Actuation | Equal control-path access/clock, the same declared cap on drift or physical force, duration and error tolerance; theta writes and routing/normalization cannot be free |
| Integration | Stable timestep and global-error control; rejecting/halving steps; cost can increase with ||M||, derivatives of M and stiffness, despite a faster continuous-time path |
| Storage | Mobility network/optimizer, examples/traces/checkpoints, old path/certificate/anchor metadata and precision; count any model-based root predictor used as a teacher |
| New retention | A new speed/accuracy achievement needs its own subsequent admission constraints; the old root covering preserves branch identity, not all previously optimized deadlines |

A sufficient conservative applied-drift cap on the old tube is Mbar S r0<=F. If clipping instead enforces the cap, the clipped field and its derivatives belong in the learning/certificate calculation. On a compact certified tube, scalar clipping retains a positive lower mobility bound only after bounding the unclipped force; use that reduced m in (5). Force clipping cannot silently retain the old unmodified contraction constant.

### Strong conventional alternatives under the same information

1. **Scalar timing/preconditioning:** best state-dependent scalar mobility with the same cap and source is exactly the class in (25). The strict difference (26) survives this comparator. It does not survive all directional control.
2. **Ordinary matrix preconditioning:** a model-based controller can compute Hbar^-1 or a suitable Hessian/quasi-Newton approximation from the same source/probes, and use the same force cap. In this small known polynomial, (23) is cheap; no neural discovery advantage is derived. The same admissible field with the same update is a same-mechanism implementation, not an independent empirical victory or an automatic historical-novelty veto.
3. **Direct root feedback/continuation:** when an additive drift actuator is allowed and the known target is a=1, the ordinary controller can apply -F e/||e|| directly, matching (22) without constructing a mobility inverse. A moving-branch predictor/continuation solver may do the analogous job, paying for its root/derivative work. Forbidding it requires an independently justified actuator interface, not a desire to force a topology win.
4. **Frozen/modular policies and direct numerical solves:** these can retain old policy instances or compile symbolic old outputs, then solve the new task separately. They are legitimate in their actual execution regimes; topology claims must account for their storage/routing/preparation and allow them to exploit the same owned source.

Learning a cheap approximate preconditioner might amortize repeated exact curvature/control work on a consequential task family. That is a plausible **OPEN** mechanism consequence, not shown by the two-dimensional polynomial. If it saves a measured Delta_call per call and costs A in acquisition/certification/maintenance, the relevant break-even condition is N_calls Delta_call>A, including any changed force/error/integration/storage expense. Here a conventional source-aware controller can already obtain the supplied useful direction analytically, so a learning-cost advantage is absent from the derivation.

## 8. One consequential uncertainty and the current scientific boundary

The repair has two load-bearing mathematical results: a metric-correct uniform old-basin certificate, and a genuinely directional useful edit that cannot be compiled into a scalar clock under the declared common amplitude cap. Its successful task gradient need not be zero at isotropy. It therefore passes a stronger constructive milestone than merely writing an SPD matrix or asserting noncollinearity.

**OPEN:** Can task experience acquire an inexpensive shared directional field whose amortized execution benefit exceeds its observation/training/field/integration/certificate cost against source-aware preconditioning and direct continuation control, while later useful improvements remain retained under explicit deadline/accuracy contracts? The old tube proves neither the acquisition efficiency nor that stronger new-retention condition. Wrong-basin excursions, overly conservative m, shrinking force slack, increased stiffness and cheap exact preconditioners are substantive possible adverse outcomes.

The useful next within-lineage action is to choose the actual modeled resource/actuator interface and derive its **matrix-preconditioner/direct-control frontier**, using (26) only as a scalar-retiming separation. If the intended gain is computational rather than physical time, include actual mobility/Hessian/solver work and integration error; if it is physical execution, justify the actuator and mobility-change costs. No pilot merely reproducing this analytic lemma is needed, and no experiment is authorized by this note. MAIN remains human-locked topology.
