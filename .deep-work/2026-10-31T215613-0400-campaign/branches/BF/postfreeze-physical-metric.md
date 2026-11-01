# Postfreeze coordinator question: can a fixed physical metric make gauge-related implementations different?

Provenance: The coordinator asked this question after receiving the frozen v1 and probe outcome, and supplied Stock et al. 2022 as a primary lead. This is therefore a separately authorized **coordinator-proposed extension**, not part of independent v1 or its predeclared probe. v1 is unchanged. No coordinator mobility variant was read. No extra numerical experiment was run.

**Judgment:** Yes. A fixed physical activity metric can distinguish implementations with exactly identical retained endpoint maps. The following concrete connection learner shows the distinction without runtime coordinate conversion. It removes an avoidable internal excursion; it does not discover a new operation or defeat F's cached-action baseline. Synaptic balancing is close prior for the general learning principle.

## Actual state and transition-law change

Use the same physical payload h=(a,b,c), controller q=(x,y), fixed clock, and F's learned lower-triangular hat connection A_x^0. Add one trainable central connection parameter:

`A_x^kappa=A_x^0`, `A_y^kappa=kappa E31`.

All strict-lower 3x3 matrices commute with E31. Therefore `G_kappa(q)=I+kappa y E31` has inverse `I-kappa y E31`, equals I at the controller basepoint (0,0), and transforms A^0 into A^kappa. The actual physical c-register includes an additional `kappa y a` during a controller excursion. It is removed again at every based return.

Changing kappa is a real connection-weight update with the same physical register units. Every based loop retains its full payload map for every input, including acquired A, B and C, even though intermediate trajectories change. The norm/cost measurement is **not** transformed along with G. If one transformed the metric to pull it back with G, the putative gain would disappear; that would be coordinate bookkeeping.

This implementation need not evaluate G, its derivative, its inverse, or a full matrix exponential at runtime. Those expressions prove the parameterization. The compiled physical field is simply the displayed constant central A_y weight. A sparse evaluator computes `c_dot=kappa*a*u_y` on vertical segments, in addition to existing A_x dynamics. A dense evaluator still pays the complete 3x3 field-matrix action costs. The theorem alone does not license skipping those operations in a hardware comparison.

## A worked run and a learner with an actual nonzero signal

Execute F's A loop at unit axis speed, total path length/time six:

`(0,0)->(0,1)->(2,1)->(2,0)->(0,0)`.

Let the acquired A_x field be E21 times the unit-integral first hat, and input h=(1,0,0). For kappa=3, the physical states at vertices are

`(1,0,0) -> (1,0,3) -> (1,1,3) -> (1,1,0) -> (1,1,0)`.

For kappa=0 they are

`(1,0,0) -> (1,0,0) -> (1,1,0) -> (1,1,0) -> (1,1,0)`.

Endpoint semantics and timing are identical. Peak physical payload norm changes from sqrt(11) to sqrt(2). This peak cannot be explained away by assigning a new output coordinate, because the measured registers and metric stay fixed.

Use the physical learning objective `J(kappa)=integral_0^6 ||h(t)||^2 dt` in those fixed units. This is an activity objective, not asserted to equal dissipated energy. The exact calculation is short. Along the top segment the acquired b-coordinate is F(x), where F(x)=x^2/2 on [0,1] and F(x)=1-(2-x)^2/2 on [1,2]. Hence `integral F(x)^2 dx=23/30`. The unchanged a and b contributions give `J(0)=6+3+23/30=293/30`. The extra c activity contributes

`kappa^2 integral y(t)^2 dt=kappa^2(1/3+2+1/3)=8 kappa^2/3`.

Therefore

`J(kappa)=293/30+8 kappa^2/3`.

Unlike endpoint loss, this measured physical objective has a nonzero gauge-direction gradient. An explicit learner uses delta=1/4, eta=1/8, and four symmetric finite-difference updates:

`kappa <- kappa - eta [J(kappa+delta)-J(kappa-delta)]/(2 delta)`.

The displayed quadratic gives `kappa_new=kappa/3` exactly. Starting at 3, four updates yield `3 -> 1 -> 1/3 -> 1/9 -> 1/27`. Every intervening field preserves all endpoint operations, including the evaluation perturbations. The resulting activity is `293/30+8/2187`; peak norm is sqrt(2+1/729). No inverse Hessian, operation-discovery oracle, or endpoint retraining is hidden in the rule. The analytical calculations verify this defined example; an apparatus measuring J or a numerical activity integrator has not been executed here.

## Full cost and strongest attacks

- Learning uses eight complete physical loop evaluations, 32 geometric edges, controller path length/time 48, eight scalar activity feedback values, four scalar parameter updates, and one persistent kappa parameter. All control paths and the activity sensor are designed. An online physical activity accumulator must square three payload components, sum them, and integrate over time; its bandwidth, precision and power have not been specified as hardware. The exact analytical integral is a proof, not a free experimental sensor.
- Sparse exact transport of each vertical central segment performs `c <- c+(kappa Delta_y)a`: two multiplications and one addition, or one multiplication/one addition when the segment coefficient is cached. There are two such segments per A loop. Lower nonzero kappa changes activity magnitude, not the arithmetic count. At kappa=0 an implementation that legitimately removes zero operations can skip both segments; the four-step learner has not reached exact zero.
- A fixed dense evaluator pays nine multiplications and six additions per 3x3 matrix-vector product before control-weighting/accumulation; two control fields must be charged when both are evaluated. Gauge normalization does not turn those evaluations into free gates. There is no extra amplitude-restoration law here: the model is reversible and driftless, with fixed timing. Biological restoration, actuator dissipation, stiffness and finite-precision error would be additional model assumptions, not established benefits.
- Under J alone, eight learning evaluations cost `162152/1215` activity units. Replacing kappa=3 by 1/27 saves `24-8/2187` per subsequent A execution. Thus this deliberately supplied nuisance can repay that particular training-activity expenditure after six A uses. This exact toy calculation excludes the sensing/update machinery and makes no hardware-energy or wall-time claim.
- The example begins with a removable pure-gauge nuisance absent from F v0 (`A_y=0` there). It demonstrates that the physical-metric question is nonempty; it does not demonstrate that F's original learned geometry has an exploitable inefficiency, or that the system can find a useful orbit beyond this supplied one-parameter family.
- A cached-action implementation with the same internal trajectories/weights and the same physical activity objective can make the identical kappa update. Holonomy is a retention certificate here, not a unique source of task advantage. Merely comparing endpoints against a baseline that cannot represent its own physical implementation would be unfair.

## Primary precedent and remaining claim

Stock et al., *Synaptic balancing* (2022), was read directly from the author-hosted original, PDF pp. 3–10. Equations (4)–(5) give task-preserving diagonal similarity for homogeneous rate RNNs; equations (6)–(11) define and analyze a fixed internal-noise sensitivity cost on that orbit; equations (17)–(20) give local learning. The text explicitly separates invariance of input-to-output sensitivity from changes to internal-noise sensitivity. Their dynamical stability result requires additional connectivity conditions. [Primary source](https://ganguli-gang.stanford.edu/pdf/22.SynapticBalancing.pdf).

Our construction uses a state-dependent central shear connection, fixed basepoint ports, and an integrated-activity objective rather than their diagonal homogeneous-RNN similarity and Jacobian sensitivity proxy. Those differences require a substantive learning/resource advantage before supporting originality. The broad principle—learn on a task-preserving orbit to improve a physically fixed secondary criterion—is already established.

**Decision:** Keep F's geometry/retention direction open in this narrow, nonempty sense. Close any claim that the present construction supplies a new operation, expanded reachability, an advantage over a fully matched cached-action learner, or novelty for task-preserving physical optimization. A scientifically new result would have to identify a specific missing mechanism beyond existing balancing/preconditioning and demonstrate a real fixed-resource gain without presupposing the useful orbit. No further experiment was started.
