# B v0: learn relations, execute by relaxation, construct shortcuts by elimination

Status: AGENT PROPOSAL / HYPOTHESIS with explicit RECONSTRUCTIONS. Independently frozen before reading any other branch or old architecture report. The actual configured route is inherited current model / Ultra as authorized by the user and conveyed by the root agent. No Pro call was made; the tool interface exposes no independently checkable deployment ID.

## Scientific opportunity and protected construction

The opportunity is to make a learned relation an executable, reusable internal object. A relation is stronger than a remembered output pattern: holding different subsets of its arguments should invoke forward computation, inverse computation, or completion. Sharing an argument should compose relations without training a new end-to-end map. Repeated successful composition should create an additional interaction that removes a dynamical bottleneck. This addresses construction, composition and improvement of internal computational mechanisms within one deliberately restricted algebra.

The high-upside hypothesis is that a network can grow an internal algebra by learning stable resonances and then forming new resonances through local cancellation of shared carriers. In this construction, a mechanism is an inspectable coupling among phases, not the trajectory's visual appearance. The proposed shortcut process changes the actual vector field. It is not merely a name for observed synchronization.

The substrate, search grammar and cancellation rule are designed. The network discovers which grammar elements are supported by examples, their phase offsets, which stored relations co-occur, and therefore which cancellation opportunities arise. It does not discover multiplication from an unrestricted space of possible physical laws. That limitation is central to the judgment.

## Source-grounded changes to assumptions

Source records are frozen separately in `sources/B/load-bearing-records.md`.

1. Kelso's learning scans and symmetry transfer, printed pp. 168–179, redirected the target from isolated learned attractor points to transferable relations and changes to the surrounding dynamics. They do not establish the neural learning rule below.
2. Izhikevich's equations (10.12)–(10.17), printed pp. 407–411, force a distinction between a defined phase substrate and a demonstrated reduction from actual neurons. Standard weak pairwise coupling yields pairwise averaged interactions at first order. I explicitly assume a nonlinear multi-carrier coupling primitive.
3. Strogatz section 4.5 gives a concrete detuning attack: a relation with one free phase has finite tracking bandwidth and a calculable bias. No instantaneous streaming arithmetic claim is allowed.
4. Gerstner's equations (19.12)–(19.15), printed pp. 501–503, require named plasticity traces and event gates. Calling this Hebbian would otherwise conceal the implementation and its high-order observation cost.
5. Existing oscillator equilibrium propagation already learns XOR; adaptive higher-order networks already exist. The possible contribution must concern learned reusable relations and autonomous structural consolidation, not oscillators, XOR, plasticity or hyperedges in general.

## Exact state and execution rule

There are `n` carriers, each with phase `theta_j in R/(2pi Z)`. In the laboratory frame every free carrier has a common natural rotation `omega`; below use the rotating frame. A stored relation `m` consists of:

- a primitive integer incidence row `c_m in Z^n`, with `sum_j c_mj=0`;
- a complex observation statistic `q_m`, sample count `N_m`, confidence `rho_m=|q_m|`, offset `delta_m=arg(q_m)`;
- a nonnegative interaction budget `K_m` and a retained/provisional flag.

The zero-sum condition makes relations invariant to arbitrary common phase origin. Primitive means greatest common divisor of nonzero coefficients is one; rows `c` and `-c` denote the same relation with conjugate statistics, so store one canonical orientation.

For active relations,

`E(theta) = - sum_m K_m cos(e_m)`,

`e_m = c_m theta - delta_m  (mod 2pi)`,

`theta_dot_j = - sum_m K_m c_mj sin(e_m)`

on free ports. Clamped ports follow the externally supplied observation/input trajectory. Their clamp status is explicit. No learned digital readout does the operation. Output is the phase of a free carrier after settling. With static clamps, `dE/dt=-sum_free(theta_dot_j)^2 <=0`. This proves energy descent only, not the absence of unwanted local minima in arbitrary networks.

A possible complex oscillator realization is `Z_j=r_j exp(i theta_j)`, radial restoration `r_dot_j=lambda(1-r_j^2)r_j`, plus tangential coupling `i Z_j theta_dot_j`. Products of `Z` and conjugates can express `exp(i c_m theta)` on the unit torus. For a four-port relation `e=theta_z-theta_x-theta_y+theta_r-delta`, the forcing target for `z` is `P_z=exp(i delta) X Y R*`. The tangent term `(K/2)(P_z-Z_z^2 P_z*)` gives precisely `theta_dot_z=-K sin(e)` when amplitudes equal one. This is an explicit nonlinear interaction requirement; its cheap biological implementation is OPEN. The exact phase model does not need an unverified weak-coupling reduction.

## Actual learning and structural allocation

Initially expose a bounded grammar `D_4`: all nonzero canonical primitive rows with coefficients in `{-1,0,1}`, zero sum and support at most four. The implementation may restrict candidate supports to locally co-observed carriers; a globally exhaustive dictionary costs `O(n^4)` and is not scalable.

For each fully observed/clamped sample of a candidate's ports, update

`N_c <- N_c+1`,

`q_c <- q_c + (exp(i c theta)-q_c)/N_c`.

Only genuinely observed samples count; settling under the network's own coupling is not fresh evidence for a world relation. A per-port observed flag and the product of participating flags implement this gate. This is designed training infrastructure. It is not an oracle for which relation is correct.

After at least `N_min` samples, promote candidates with concentration above a fixed threshold and held-out observed concentration. Store their rows/statistics and allocate coupling budget. Ordinary circular Fourier moments identify the relation; there is no task-specific function identifier. Retention means these stored statistics are frozen during inference and unaffected by adding a different relation. Online forgetting, context discrimination and contradictory evidence are not solved by freezing.

Worked identifiability calculation: present independent uniform phases `r,a,b`, and set `x=r+a`, `y=r+b`, `z=r+a+b+delta`, all modulo `2pi`. For a row `d`,

`d theta=(sum d)r+(d_x+d_z)a+(d_y+d_z)b+d_z delta`.

The expected Fourier moment vanishes unless all three random-phase coefficients vanish. Within `D_4`, the sole nontrivial primitive relation, up to sign, is `c_r=+1,c_x=-1,c_y=-1,c_z=+1`. Its statistic is `q=exp(i delta)`. Pairwise moments vanish. This proves identifiability for this data-generating family and this fixed dictionary; it does not solve useful-coordinate discovery. For independent Gaussian relation noise with standard deviation `sigma`, the population concentration becomes `exp(-sigma^2/2)`.

## A real operation, including inverse use

Suppose the training examples have `delta=0.43`. After training, clamp `r=0`, `x=0.7`, `y=-0.4`, and release `z`. The learned vector field is

`z_dot=-K sin(z-0.73)`.

For every initial phase except the unstable antipodal point, it converges to `z=0.73 (mod 2pi)`. In relative complex values this computes multiplication: `exp(i(z-r))=exp(i delta) exp(i(x-r)) exp(i(y-r))`. Binary phases yield parity, but continuous examples are needed to distinguish coefficient signs and avoid binary aliasing.

Now clamp `z=0.73`, `y=-0.4`, `r=0` and release `x`; the exact same stored interaction converges to `x=0.7`. This is inverse execution without retraining or selecting a separately learned inverse module. Each query still requires designed port clamping and a stopping criterion.

If a single free output has residual detuning `nu`, its error obeys `e_dot=nu-K sin(e)`. Stable locking is possible only for `|nu|<K`, with error `asin(nu/K)`. At or beyond the boundary exact relation execution fails by the familiar locking bifurcation.

## Composition is a physical state transition

Learn relation A on `(r,x,y,z)` with `delta_A=0.43`, and learn relation B separately on `(r,z,w,v)` with `delta_B=-0.31`. There are no composed training examples.

For global carrier ordering `(r,x,y,z,w,v)`:

`c_A=(1,-1,-1,1,0,0)`;

`c_B=(1,0,0,-1,-1,1)`.

Pin `r,x,y,w`; release `z,v` and activate both learned couplings simultaneously. No serial controller computes intermediate values. For `x=0.7,y=-0.4,w=1.2,r=0`, the feasible state is `z=0.73,v=1.62`. The two errors are

`e_A=z-x-y+r-delta_A`,

`e_B=v-z-w+r-delta_B`.

The physical transitions are `z_dot=-K_A sin(e_A)+K_B sin(e_B)` and `v_dot=-K_B sin(e_B)`. Relation B back-acts on the shared carrier; composition is not free. Near the feasible state, for `K_A=K_B=1`, the Jacobian in `(z,v)` is `[[-2,1],[1,-1]]`. Its eigenvalues are `-(3 +/- sqrt(5))/2`, hence the slow local contraction rate is approximately `0.381966`. The individual one-port relation's rate was one. Longer chains acquire increasingly slow modes.

Both original relations remain functionally present at the new feasible state. This statement assumes compatibility of offsets/ports; arbitrary constraints can be inconsistent and destroy this property. Neither retained parameters nor current-task equilibrium implies greater future discovery ability.

## A concrete mechanism that grows an improved mechanism

Each pair of retained, co-used relations may attempt a merger only if an integer multiple/orientation makes one shared carrier cancel. For v0 use the minimal case of opposite unit coefficients and row addition. Co-use eligibility is a bounded trace of both factors' activation and small residuals. Propose only a finite number of pairs with high eligibility; canonicalize rows, reject duplicates, enforce an arity/coefficient cap, and never delete parent factors in v0.

The merger is an actual structural rule:

`c_new <- c_A+c_B`, `delta_new <- delta_A+delta_B (mod 2pi)`.

Here it creates `c_new=(2,-1,-1,0,-1,1)` and the five-port constraint `v-x-y-w+2r=delta_A+delta_B`. Its residual equals `e_A+e_B`; its zero is entailed by the two parent relations. This creates an executable shortcut that does not involve intermediate carrier `z`. The row did not exist in the initial `D_4`; the grammar is now expanded by its own learned factors. It is a fixed algebraic growth rule, not unexplained emergence.

Initialize its offset by the entailed identity. Treat that as internal derivation, not new external evidence. On later observed trials, maintain its own statistic. Own-generated replay can test execution equivalence but cannot establish that any parent relation describes the world. A noisy population's composite moment need not equal the product of parent moments unless the corresponding noises are independent.

For static compatible clamps, adding `K_new>0` contributes the positive-semidefinite local stiffness `K_new c_new,free^T c_new,free` at the same feasible state. With all three strengths one, the free-port Jacobian becomes `[[-2,1],[1,-2]]`, eigenvalues `-1,-3`: the slow local rate rises from `0.381966` to one. This uses more interactions, so it is not yet an efficiency result.

A predeclared stricter proxy is `sum_m K_m ||c_m||_2^2` over ALL ports, including clamped ones. The parent pair at strengths `(1,1)` has cost eight. The three-factor system at strengths `(0.5,0.5,0.5)` also has cost eight because the new row's squared norm is eight. Its Jacobian is `[[-1,0.5],[0.5,-1]]` with rates `0.5,1.5`. Thus its slow local rate improves by about 30.9% under this proxy while its fastest rate falls. This is a prediction to check, not a hardware or wall-clock efficiency claim. The new higher-order interaction has real implementation and evidence costs that this proxy does not settle.

The candidate learning/improvement loop is: observe phase relations -> allocate supported factors -> execute compositions by shared state -> accumulate co-use eligibility -> create cancellation factors -> retain a shortcut only if its measured relaxation improvement survives a fixed cost rule and independent world-data checks. In v0 the acceptance experiment is a small external scientific evaluator. An autonomous local acceptance rule for arbitrary contexts is still OPEN; do not hide it in an unspecified internal controller.

## Strong attacks and what would reverse the judgment

1. **Known algebra disguised as new neural computation.** This is Fourier relation estimation plus energy-based constraint execution plus algebraic elimination. All components have close precedents. A novelty claim is unestablished; oscillator EP, higher-order associative memory, vector-symbolic binding and resonator systems are mandatory comparisons. It remains worth constructing because it states a concrete candidate way to turn learned relations into new physical interactions.
2. **Primitive smuggling.** Multiplicative phase interaction is supplied, and row cancellation is supplied. The model discovers relations/offsets/shortcuts within that algebra, not arbitrary computational primitives. If useful tasks do not have efficiently discoverable coordinates in this algebra, the proposal loses significance.
3. **Discovery/representation bottleneck.** Exhaustive `O(n^4)` candidate scans, arbitrary phase encodings, and high-order shortcuts may cost more than ordinary networks. The hardest missing step is an evidence-driven process that creates the useful carriers/coordinates and candidate interactions without presupposing the desired operation. This is a core scientific gap, not merely engineering.
4. **Self-confirmation.** Its own low-energy states cannot train evidence for their validity. A false parent relation generates false entailed shortcuts. External observations/negative examples and explicit provenance are required.
5. **Global dynamics.** A local Hessian improvement can coexist with additional spurious minima, smaller basins or slower finite-time response. Probe nonlinear settling separately, including random starts. A contradictory relation can create frustration.
6. **Binding and context.** Shared carriers are manually selected in the operational example. Uncontrolled reuse causes interference. Learning portable relation roles, allocating fresh carriers and deciding which constraints to activate remain explicit scaffolding/open questions.
7. **Resource accounting.** Polynomial degree, coefficient size, trace memory, sample count, candidate proposals and coupling strength all count. More hyperedges alone do not establish improved computation.

## Frozen lightweight verification contract

QUESTION: In this explicit substrate and fixed data family, does the observation rule recover the planted relation, does its actual ODE execute and compose held-out operations, and does the algebraically generated shortcut change local relaxation as predicted without calling added resources free?

GIVEN: Exact rows, learning rule, offsets, data distribution, ODE and cost proxy above; actual source records. The data generator supplies continuous phase coordinates and observed port examples.

NON-GOALS: No claim of general intelligence, arbitrary mechanism invention, biological implementation, open-ended evolution, novelty, or future evolvability. No large experiment.

VERIFIER: One reproducible bounded script: deterministic/random-seed Fourier identifiability and shuffled-output control; forward/inverse/held-out-composition ODE integration; analytic-versus-finite-difference Jacobians; fixed-resource local-rate comparison; bounded random-start nonlinear relaxation. These touch the actual stated equations. They cannot independently validate the representation's scientific value.

BUDGET: Two construction levels (learned relation; generated cancellation shortcut), one substantive attack (known-algebra/discovery bottleneck plus shuffled-control failure), one executed lightweight check. At most 256 observed training samples per module, 64 held-out queries per condition, and fixed ODE horizons; no parameter search or unbounded extension.

STOP: If source constraints invalidate the substrate, if the exact equations fail their advertised operation, if shuffled data also promote the planted factor, or once the bounded check has decided these finite claims. Preserve failures. No next experiment is implied by this certificate.

OUTPUT: This frozen candidate, source records, executable probe/output and a compact evidence certificate.
