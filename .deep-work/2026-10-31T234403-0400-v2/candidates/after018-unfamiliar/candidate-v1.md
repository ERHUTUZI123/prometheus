# Learned expansion of the optimization tangent by computational commutators — v1

**AGENT PROPOSAL / CONSTRUCTIVE DERIVATION, unexecuted.** Same V2 run. No experiment, training, installation, new paid route or model override. Effective provider mode unknown. Root is the sole coordinator and this worker makes no MAIN/campaign decision.

## What was discarded, and what is different here

V0 is preserved at its frozen hash. After freezing it, read actual after014 `candidate-v0.md`, `raw-stream-v1.md`, and `acquisition-v2.md`, including their neural emission, whole-law search, own-trace edits and selective-capture protocols. Learned pair-local execution is already much more developed there. V0's principal-expansion witness supplies a standard partial-evaluation certificate, without a different formation mechanism. **Do not retain v0 as an independent promising architecture.**

The new entry is nonholonomic geometric control, entered through Murray–Sastry's original 1993 paper. It changes a different assumption: learning need not keep the coordinates of its *allowable computational updates* fixed. A composite update whose effect is higher order in primitive controls can be made a first-order learning coordinate. The candidate is an explicit rule for creating those coordinates, not a claim to invent Lie brackets or controllability.

Connection to the Fundamental Question: the learner would construct executable internal update operations from already acquired neural updates, compose them recursively, and expose their effects to later gradient learning. The proposed change concerns **finite-budget access to useful computational transformations**, not new computable functions.

## 1. Actual state and executable updates

Runtime state is `h in R^n`, with an externally bounded precision and compact operating region. Input x and task context q are ordinary network inputs. A fixed encoder initializes h; a decoder D produces a terminal prediction. Their costs and training count. For the formation argument below, freeze their current versions so the coordinate/gradient comparison has a definite meaning.

There are m existing smooth neural update fields `f_i(h)`. Use invertible additive neural coupling updates so inverses do not require an unspecified oracle: split coordinates into A_i and B_i, and define

`T_i^a(h)_A = h_A + a * F_i(h_B;theta_i)`,

`T_i^a(h)_B = h_B`.

F_i is an ordinary smooth neural network. This is the exact time-a flow of a shear field, and `T_i^(-a)` is its inverse in real arithmetic. In finite arithmetic the inverse has rounding error and must be accounted for. The split, smooth activation, signed control, allocator and exact indexing are supplied architecture. The f_i can have been learned in an earlier task stage; this candidate does not explain the emergence of those initial primitives. In the worked calculation they are manually fixed.

A dictionary entry j stores a finite **word** `W_j(epsilon)` in these signed base updates, a positive integer order d_j, the associated leading field B_j, and actual leaf-execution cost. Base entries have `W_i(epsilon)=T_i^epsilon`, d_i=1, B_i=f_i. Inverse words reverse the leaf order and negate every leaf amplitude. The executable meaning is always the stored word, not an unexecuted derivative field.

Assuming an entry has, uniformly on its operating region,

`W_j(epsilon)(h)=h+epsilon^(d_j)*B_j(h)+O(epsilon^(d_j+1))`,

give it a scalar control coordinate c:

`M_j(c,h)=W_j(c^(1/d_j))(h)` if c>0;

`M_j(c,h)=W_j((-c)^(1/d_j))^(-1)(h)` if c<0;

`M_j(0,h)=h`.

Thus `M_j(c,h)=h+c*B_j(h)+O(|c|^(1+1/d_j))`. In exact mathematics its derivative at c=0 is B_j. At zero a custom derivative must use this derived limit; naïvely differentiating a branch containing a fractional power is invalid. Computing B_j costs actual Jacobian-vector products. At nonzero c, differentiate the actual finite word. No claim of favorable floating-point conditioning follows.

A finite neural controller outputs coefficients c_t and entry choices at at most H internal update sites. Its transition is `h_(t+1)=M_(j_t)(c_t,h_t)`. Existing selection can use a fixed controller or categorical neural routing with ordinary score-function estimation. For the construction below, the site being enlarged has one new coefficient head initialized to zero, so addition of an unused entry changes no forward trace. All task losses are terminal; no correct operator, intermediate target h, or bracket label is supplied by a teacher.

## 2. A concrete operation-formation and adaptation law

At an existing update site, record states h_a from the learner's own training traces. Backpropagate terminal supervised loss through the frozen remainder to obtain `g_a = gradient_h loss_a`. This requires differentiability and the ordinary external target y; it is stronger information than scalar reinforcement alone and must be supplied equally to competitors.

Let `V_a=[B_1(h_a),...,B_K(h_a)]`, with current dictionary size K. Let `r_a=(I-V_a V_a^+)g_a`, using the declared Euclidean metric and a declared numerical rank tolerance. Geometry evaluation, pseudoinverse and own-trace storage are metered. Scores are coordinate/metric dependent even though rank and bracket-generation statements have an invariant interpretation.

For each bounded eligible pair (i,j), calculate

`B_ij = D B_j * B_i - D B_i * B_j`.

Use a predetermined degree cap and a finite pair-proposal cap. A pair is not eligible merely because a symbolic expression exists: JVPs, precision and the operating-region derivative conditions must be available. For the controller's current features phi(h,x,q), compute

`s_ij = ||mean_a [phi_a * (r_a dot B_ij(h_a))]||^2 / C_ij`,

where C_ij is the declared leaf execution count plus the charged formation-work proxy. This is an explicit heuristic, not an established acquisition theorem. It aims at task-gradient components missing from the present first-order operation span. If no positive resolved score exists, this formation round adds nothing.

Choose the highest scored pair. Construct its new word by executing, in order,

`W_i(epsilon), W_j(epsilon), W_i(epsilon)^(-1), W_j(epsilon)^(-1)`.

The new order is d_i+d_j; its leading field is B_ij under the sign convention above. This recursive leading-order statement requires sufficient smoothness and a uniform expansion of the existing words and their spatial derivatives. First-order pairs are derived below; the full arbitrary-depth error induction is **an explicit proof obligation**, not a checked theorem of this artifact. Higher-depth promotion is disabled until that obligation is met. First-order promotion is already a complete provisional algorithm.

Append a coefficient head `c_new=w dot phi`, initially w=0. Update w and the existing controller by ordinary terminal gradients, including the actual limit derivative at zero. For the chosen head, the true task gradient at zero is `mean_a[(g_a dot B_ij(h_a))*phi_a]`; the residual r is a proposal score, not a silently substituted loss gradient. A zero residual score does not prove no useful finite operation exists, and cancellation across examples can obscure a useful contextual head.

Accept/retain the new dictionary version only after metered finite-coefficient replay improves a frozen training-bank objective `mean task_loss + lambda * actual total work + mu * stored bits`, with old examples included when retention matters. Rejected trials and coefficient searches count. This is model selection on training data, not a held-out result. A finite line search is needed because operation work has a fixed positive cost whenever c is nonzero. Do not claim the infinitesimal task derivative alone defeats that resource cost. Entries and their base-network versions are immutable once retained; changing a base theta creates a new version and requires replay. Stored operation meaning can therefore be retained exactly at the program level, while old task behavior can still be lost through controller changes.

## 3. The consequential local derivation

Let f,g be C^3 fields with local flows and define chronological execution `f for epsilon; g for epsilon; f for -epsilon; g for -epsilon`. Taylor expansion of each flow through second order cancels the two first-order displacements and the pure f/f and g/g terms. The surviving cross term is

`C_fg(epsilon,h)=h+epsilon^2*(Dg(h) f(h)-Df(h) g(h))+O(epsilon^3)`.

With signed coefficient c and epsilon=sqrt(|c|), choosing the inverse word for negative c gives

`M_fg(c,h)=h+c*B_fg(h)+O(|c|^(3/2))`.

Now take any fixed finite primitive word `P(a_1,...,a_L,h)=T_(i_L)^(a_L) ... T_(i_1)^(a_1)(h)` and differentiable scalar continuation loss ell. At a=0, every flow is the identity, hence

`partial_(a_l) ell(P(0,h)) = gradient ell(h) dot f_(i_l)(h)`.

If the loss gradient is perpendicular to the span of all present f_i(h), **every primitive-duration derivative at that zero-control initialization vanishes**, regardless of the word length. If it is not perpendicular to B_fg(h), then

`d/dc ell(M_fg(c,h)) at c=0 = gradient ell(h) dot B_fg(h) != 0`.

Promotion has therefore made a second-order computational effect first-order visible to the selected gradient coordinates. This is a conditional local learning fact, not a universal neural-network limitation, an impossibility theorem, or proof that the promoted optimizer wins. Nonzero initialization, second-order derivatives, perturbation search, directly adding a residual neural field or another parameterization can remove the same obstruction.

The potentially broad principle is **growth of the optimization tangent through acquired computational composition**. The task-gradient span available to later learning can enlarge without changing the underlying base fields or their global reachability set. Whether recurrent task learning benefits after full formation costs is OPEN.

## 4. Fully worked internal computation and resource attack

Use state `(x,y,z)` and two supplied base shears:

`T_1^a(x,y,z)=(x+a,y,z)`,

`T_2^b(x,y,z)=(x,y+b,z+b*x)`.

T_2 is an additive coupling update: its unchanged conditioner is x and its updated block is (y,z). These are linear/smooth neural coupling functions, not novel primitives. Their Lie effect is standard nonholonomic geometry.

Chronological four-step execution gives, from arbitrary starting `(x,y,z)`:

1. T1(a): `(x+a,y,z)`.
2. T2(b): `(x+a,y+b,z+b*x+a*b)`.
3. T1(-a): `(x,y+b,z+b*x+a*b)`.
4. T2(-b): `(x,y,z+a*b)`.

Thus the acquired composite can act as a persistent z-update while restoring scratch x and y. Setting a=b=sqrt(c) for c>0 and reversing the word for c<0 implements `(x,y,z)->(x,y,z+c)` exactly in real arithmetic, with four base calls. Two such calls compose as z<-z+c1+c2. This is an executed-state-transition description on paper; **no machine run or acquisition has occurred**.

For initial h=0 and terminal loss `ell=.5*(z-t)^2`, t!=0, all zero primitive-duration derivatives vanish. The promoted coefficient has derivative -t. A gradient step c<-c+eta*t lowers task loss for 0<eta<2. The generic formation score can choose this pair from its nonzero task-gradient correlation without being told the target operation's name.

Resource qualification is decisive. With charged extra work penalty 4*lambda, the best possible task-loss reduction t^2/2 outweighs it only if `t^2/2 > 4*lambda`, before formation/storage cost. An objective charging physical control effort proportional to |a|+|b| would have a sqrt(|c|) cost near zero, which can defeat infinitesimal descent entirely. Our proposed digital account charges neural evaluations and finite arithmetic; it makes no physical energy claim. The square-root coordinate can also amplify finite-precision effects. These are actual restrictions, not cosmetic caveats.

## 5. Actual closest mechanisms and same-information alternatives

SOURCE boundary, originals entered: Murray–Sastry (1993) already supplies local commutator motions, bracket filtrations and steering algorithms. Walker et al.'s Log-NCDEs (2024), §§2.5,3.1,3.3–3.5, already construct neural vector fields with iterated brackets and JVPs. Morteza's ChowFlow preprint (2026), §3/Algorithm 1, learns scalar neural controls over fixed bracket-generating fields. Therefore **brackets in neural computation, low-dimensional controls and controllability are not this proposal's novelty**. Source reading details and limits are in `reading-v1.md`.

The describable difference in this inspected material is task-residual-triggered allocation of new **executable normalized word coordinates**, and the explicitly derived change in first-order task sensitivity. This is not a priority claim: adaptive Lie extensions, motion-primitive learning, boosting/new-feature growth and high-order saddle reparameterizations still require closer reading. A prior method doing this formation law would cover the candidate directly.

Strong same-information rivals must include:

- The same base coupling networks, terminal gradients, states, inverses and task bank, with all primitive-duration sequences optimized from nonzero initializations, noise and a competent Hessian-vector-product/trust-region method. Giving this rival only zero-initialized first-order descent would manufacture an advantage.
- A fixed bounded Hall/bracket dictionary with learned coefficients, or ordinary task-guided control/Lie-extension synthesis. It gets exactly the same derivatives and total formation+execution+storage budget. If adaptive allocation only prunes this existing basis, the result may be routine feature selection.
- Directly allocating one comparably budgeted neural residual update using the same gradient information. It need not obey a coupling/commutator constraint unless that constraint is justified by the scientific question. Artificially forbidding this removes an obvious alternative.

The canonical three-state example cannot decide among these. It only checks the local mechanism. No toy training should be launched to display the predicted zero-init contrast.

## 6. Success counterfactual, fragile premise and one decision

If successful beyond the canonical example, this would be a learning organization that repeatedly **makes previously higher-order internal effects directly adjustable**, so acquired computation changes the coordinates of subsequent learning. It could matter for compositional capability acquisition when useful operations are weakly visible in the current update tangent. The structural reason for considering broader tasks is that noncommutation is generic in nonlinear updates; the relevant *task-aligned, reusable and economical* part is not generic and must be demonstrated.

The most fragile premise is that this coordinate growth offers anything after ordinary saddle escape/second-order learning and after the rapidly growing word/JVP costs. A second fragility is that the neural base fields may not possess reusable task-aligned brackets at all; random noncommutation is not an algorithm. A third is that the controller may merely exploit an operation within training contexts without compositional reuse.

What reverses the construction's scientific prospect: an actual closest source already has the same acquisition law; a same-information fixed-basis/second-order learner subsumes its measured benefit; only the zero-initialization singularity yields an advantage; high-order formation or precision cost dominates; or the best plausible success is just an optimizer preconditioner on hand-picked dynamics. Any of these warrants closing this candidate rather than growing a control benchmark.

**Next single decision for root:** determine whether the local learning-coordinate distinction survives the closest adaptive-Lie-extension / feature-growth precedent and the strong second-order rival strongly enough to justify further constructive work. No experiment is requested or admitted here. This worker does not promote this candidate to MAIN.

`CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

`First-order primitive-control stationarity can coexist with a nonzero promoted commutator derivative | HAND DERIVATION, unexecuted | §3 plus exact four-shear calculation §4 | external mathematical audit; finite precision; local initialization scope | v0 graph-rule repetition discarded | source/rival comparison`

`A neural learner can form a growing dictionary of such operations from terminal-gradient residuals | EXPLICIT ALGORITHM / HYPOTHESIS | §2 state, scores, word constructor, zero head and replay rule | no efficiency/generalization evidence; higher-depth error induction pending | no empirical branch | same comparison before implementation`

`This changes broad capability acquisition beyond known control synthesis or second-order optimization | OPEN | success counterfactual only | closest-prior coverage and total-cost advantage absent | no positive scientific result claimed | root's scientific allocation judgment`
