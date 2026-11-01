# Candidate v1: learn a mechanism that can teach its own learning interface

AGENT PROPOSAL / RECONSTRUCTION / OPEN. One further constructive pass inside the same V2 run. V0 and its reading ledger remain immutable. No experiment, training, installation, new paid route, model change, host-code generation, publication, or MAIN-completion decision. Effective provider/model/mode remain unavailable.

## Judgment

**The new proposed unit of retained computation is a neural mechanism that can rebuild both a task computation and the ability to build another such mechanism, through finite learning interactions.** Task acquisition changes its ordinary execution; a second, explicitly coupled learning process then repairs the mechanism's ability to transmit that acquisition through its own teaching behavior. A learned mechanism consequently changes which learning episodes the system can construct internally, rather than merely adding another output feature or cached update.

The proposed organizing pressure is precise: **evaluate the learner produced by a learner produced by the current mechanism**, and send that outcome back through the actual learning interactions. First-descendant task success alone does not ensure that the acquired teaching procedure survives. A small exact countermodel below distinguishes these outcomes and shows the extra gradient path. The general nonlinear law is specified, but its economical acquisition and scientific importance remain OPEN.

This is a serious formation proposal, not a claim of novelty, a general self-improvement guarantee, or evidence of open-ended evolution. Learned teaching, self-generated training data, neural replication, and iterated learning already exist. In particular, this proposal must be compared directly with self-editing meta-learning and self-replicating neural programs. Its potentially consequential residual is **preserving and revising a learned learning interface while using that interface to construct later computations**, rather than copying weights or preserving a task response alone. If ordinary multi-step meta-learning already implements the same state and objective at the same resource level, that correspondence should be reported rather than protected by a new name.

## 1. Limitation selected after v0

V0's lesson was not that complicated endpoint functions are unlearnable. It was that an apparently compact acquisition construction can hide a powerful supplied algebra or informative query interface. Inspecting one's own internal computation is legitimate, but it does not create external facts absent from the evidence.

The new limitation concerns the learner's own computations and is different. A system may acquire a useful mechanism while erasing or failing to acquire the procedure that would reconstruct, specialize, or extend that mechanism in another internal instance. A stored task solver and a stored procedure for producing a trainable solver are different acquired capabilities. An algorithm that makes the current model better can also make its later self-generated learning signals worse. Success on the present task does not expose that damage.

This matters if a growing neural computation repeatedly needs new, independently adaptable instances of an acquired capability. Freezing and copying every old network is one strong solution, and will often be cheaper. The candidate asks whether a different learned retention form can leave behind a reusable *development procedure*: an executable teaching interaction that constructs a child, including the child's own ability to construct successors. Unlike a passive archive of examples, this procedure conditions on the actual developing neural state and can be revised when later tasks reveal missing teaching competence.

Connection to the Fundamental Question: construction is the actual training of a new internal neural mechanism; composition is use of acquired mechanisms as callable participants in later teaching and execution; improvement is a change in the mechanism's own inherited development procedure, selected by the capability of actual descendants. The permanent ambition would be served only if this changes finite-budget acquisition of substantial new computations. Self-copying alone is not that result.

## 2. What changed from the authorized comparisons

The following actual bodies were read in full before this v1 construction: after017-outside-principle/candidate-v0.md and comparison-v1.md; resumed-r/candidate.md and nullspace-rival.md; resumed-association/repair-after-execution.md. They ruled out three tempting replacements for a missing formation mechanism:

- A learned relation plus exact linear realization is already ordinary system identification and solving.
- A named finite update program can be matched by an identified-response controller or parameterized option.
- Shared update coordinates and additional source probes can be matched by conditional latent-factor learning, with the same observations.

V1 therefore abandons the commuting-mask class. It does not add a nonlinear feature map to it, extend it to another group, repair the parked online-organization/AD-frontier line, or claim that changing information access is automatically a new computation. The proposed new persistent consequence is that the acquired learning interaction must itself pass through the learning process it constructs.

Provenance: the parent suggested, as an optional direction, asking whether acquired mechanisms change subsequent experiments/information. That prompt contributed to the direction; the specific two-descendant teaching law below is AGENT PROPOSAL. The parent's earlier source names SRWM and learned loss were exposed before the v0 hash, as recorded there; no associated equations or current bodies were supplied. During v1 formation, the sibling audit sent point-query/coset bounds and the warning that self-inspection of transcript-derived hypotheses does not generate hidden facts. That warning is incorporated. No audit body was needed to construct this mechanism.

## 3. Actual retained state and manually supplied substrate

For one retained mechanism, let N_theta be a finite neural program with two callable modes:

    EXEC(c,x,h) -> (answer,h_next)
    TEACH(c, state_of_child, teaching_history) -> lesson

The mode bit is designed. The same parameter collection theta contains both behaviors and may share internal computations; it is not assumed to be neatly separated into expert-designed modules. A finite graph, differentiable numerical primitives, allocation of an initially bounded child, SGD, and a legal lesson format are supplied. There is no demand that these primitives emerge.

A lesson is a finite tuple

    ell_t = (mode_t, x_t, v_t, weight_t, learning_rate_t).

It requests a supervised learning interaction at an EXEC or TEACH interface. Targets v_t may be synthetic. They are control signals proposed by the teacher, not asserted external truth. Their utility is judged by independently obtained external task outcomes. Bounds apply to lesson count, target magnitude, weights, learning rates, graph size and memory. In the first fully differentiable version, the mode schedule is fixed and both modes receive some lessons; the lesson contents are neural outputs. Learning the schedule is not required to make the initial rule concrete.

The teacher may inspect the actual child's weights, graph, loss values, gradients and finite activity probes. Their construction and reading costs count. A graph encoder may compress them, but its adequacy is then an additional learned assumption. It receives no hidden task solution, privileged future label, or reference intermediate state. The strongest baseline receives the same observations.

Persistent state is

    S = (theta, finite task/context archive D, accepted mechanism handles,
         lesson/birth records with actual costs and failures).

Active developmental state is

    s_dev = (w_t, child fast state, t, actual lesson history).

theta and the archive persist across tasks. Active child state is finite and can be discarded. Children are ordinary bounded in-memory neural objects; this proposal does not generate or execute unrestricted source code.

## 4. Actual learning and construction law

### 4.1 Construct one child by the current mechanism's own output

Sample an allowed initialization w_0 and use the current N_theta to produce K lessons. The child's actual persistent weights change by

    ell_t = N_theta.TEACH(c, inspect(w_t), H_t),
    w_(t+1) = w_t - eta_t grad_w
                 [omega_t * loss(N_w(mode_t,x_t), v_t)] evaluated at w_t.
                                                               (1)

In the student's gradient in (1), the already emitted lesson is held fixed; the student does not differentiate its target through the teacher. The outer derivative does follow how teacher outputs change when the actual developing child changes. H_t includes what actually happened during prior lessons. A requested target is not confused with a realized change. Each lesson and gradient step is charged. At K, the child is the actual neural mechanism N_wK. Denote this finite construction

    w = B_theta(c, epsilon),                              (2)

where epsilon includes the paid initialization and sampling randomness. B is a sequence of real learning updates, not a hypernetwork assignment of w, and no task-solving theorem is assumed for it.

The initial v1 rule explicitly includes teaching-mode lessons so that the child is trained to acquire a usable TEACH computation, not merely its EXEC response. This is a specified construction choice, not a theorem that every other way of learning the teaching mode is impossible. The general implementation fits that behavior through the same finite lesson channel as other neural behavior. Its teaching targets need not be copied from a parent word for word; gradients through the later descendant can select different internal teaching behavior.

### 4.2 The child's acquired teaching procedure constructs the next child

Use the acquired child's actual TEACH mode, with a fresh allowed context c' and initialization:

    v = B_w(c', epsilon').                               (3)

The parent does not step in to teach v. If w's teaching behavior is broken, it actually fails here. This is the concrete coupling absent from a first-child-only evaluation.

External task evaluators supply finite losses on fresh paid examples. Let L_exec(theta;D) denote task loss of EXEC, not a measure of teaching fidelity. One explicit outer learning objective is

    J(theta) = L_exec(theta; D_current)
             + alpha E[L_exec(w; D_child)]
             + beta E[L_exec(v; D_grandchild)]
             + gamma L_protected(theta,w,v; D_past)
             + lambda actual_birth_and_storage_cost,

    theta <- theta - rho grad_theta J(theta),             (4)

with w and v given by actual equations (1)-(3). The gradient goes through both finite training processes, including teacher outputs and the required higher derivatives. A nondifferentiable actual cost is charged at acceptance rather than assigned an invented derivative; under a fixed graph and fixed K it is a constant in this differentiable update. These computations are ordinary, potentially expensive meta-gradients. They are not cached AD-frontier repair or a claim that AD itself is acquired. Their full work and memory count.

D_current, D_child and D_grandchild are actual external training/audit examples available at their stated times. When c' describes a new task, its support information arrives before the child's TEACH call; its scored query labels are revealed only afterward. New facts in D_child or D_grandchild are not secretly assumed to have been present in the parent's context. Held-out evaluation is not reused as training without being relabeled as training.

Protected loss is over explicitly retained earlier capabilities and contexts. It prevents a trivial positive interpretation in which teaching one new computation silently deletes all older ones. It proves nothing off the finite protected distribution. Separate earlier mechanisms or snapshots remain legitimate strong alternatives.

### 4.3 Promotion and later improvement

After an ordinary new-task update improves EXEC, equations (1)-(4) train or repair the inherited teaching behavior. A candidate child replaces a retained parent only after separate finite checks of current execution, protected capability, and its own descendant construction. The check has a fixed descendant depth and budget. Failure retains the old parent and the actual failed record; no success follows from a model's predicted utility.

New retained mechanisms can participate in EXEC or TEACH of later mechanisms through literal callable handles. Calls, scope, retained parameters and physical effects count. This permits a previously acquired operation to generate part of a later learning episode. It does not by itself solve which old operations should be selected or establish a cheap composition search.

## 5. Why the extra outcome is not already measured by ordinary task success

Here is an exact, deliberately small realization of the proposal, exposing the causal path rather than asserting a nonlinear learning theorem.

A mechanism has three scalar neural coefficients (a,b,d). On a scalar context/input x, its execution is a*x. Its teaching behavior emits three coordinate lessons with targets (b,d,d). A child begins at (0,0,0). Each coordinate is a one-weight linear neural port; a unit-rate gradient step on half squared error at input 1 sets its weight to the emitted target. All three steps therefore construct

    B(a,b,d) = (b,d,d).                                 (5)

The duplicated final teaching target is an explicitly designed convention in this diagnostic realization. It is not claimed to have emerged. The general proposal uses the same-mode neural interface of sections 3-4; equation (5) is its transparent finite instance.

Suppose the external task is y=t*x and the audit has E[x^2]=1. The present, first-child and second-child losses are respectively

    L0 = (a-t)^2/2,
    L1 = (b-t)^2/2,
    L2 = (d-t)^2/2.                                    (6)

Consequently

    grad_(a,b,d) (L0 + alpha L1) = (a-t, alpha(b-t), 0),
    grad_(a,b,d) (L0 + alpha L1 + beta L2)
                                         = (a-t, alpha(b-t), beta(d-t)). (7)

At t=1 and (a,b,d)=(1,1,0), the current mechanism and its child execute the task perfectly. The child is (1,0,0); its child is (0,0,0) and fails. A first-child-only learner receives zero gradient in every parameter of (6)'s measured objective. The extra outcome exposes d: with beta=1, a unit outer step makes d=1. The reconstructed child is now (1,1,1), and applying B again preserves (1,1,1). The system has acquired not a new scalar answer but the missing ability to reconstruct a capable teaching mechanism.

If a new task requires t=-1, directly changing a to -1 makes the current mechanism correct while leaving b and d at +1. Descendants still implement the old task. The descendant terms provide the actual signals that update both transmitted execution and transmitted teaching. In a multi-task system, task context or separate physical mechanisms is required to preserve the old +1 behavior; the scalar model does not magically retain contradictory tasks at one unconditioned input.

This is a hand derivation, not an experiment. It is not a novelty theorem: anyone can construct (5) and optimize (6). Its role is to establish three distinct operational capabilities and one missing causal learning signal:

1. Current task performance.
2. Constructing a child with task performance.
3. Constructing a child that can construct another task-performing child.

These distinctions persist for nonlinear neural teachers; the clean diagonal gradients do not. A finite-depth objective can still overfit the assessed depth. The example cannot establish indefinite transmission, useful architecture formation, efficiency, or a richer algorithm.

## 6. Load-bearing nonlinear learning path

For a general differentiable inner step, let ell_theta(w) denote the teacher's emitted lesson, and define the actual student's gradient field by

    G_t(w,theta) = partial_u loss(N_u(mode,x),target)
                         evaluated at u=w and (mode,x,target)=ell_theta(w).

The partial derivative with respect to u holds the lesson fixed. In contrast, the total Jacobians A_t=D_w G_t and C_t=D_theta G_t include the effect of the teacher responding to w. A_t need not be symmetric or a Hessian: it contains the additional term (D_ell G_t)(D_w ell_theta). Omitting this term would silently change the adaptive teaching law. The derivative P_t = partial w_t / partial theta obeys

    P_(t+1) = (I - eta_t A_t) P_t - eta_t C_t,

with additional product-rule terms if eta_t depends on theta. These terms are part of the actual law, not silently discarded. The child-to-grandchild derivative contains both the dependence of v on its developing weights and its dependence on w as the *teacher*. The second component is exactly what the scalar d path makes visible.

A stop-gradient at the child teacher interface removes that path. A surrogate reward that scores only w's EXEC also removes it. Thus the proposal makes a falsifiable causal prediction: if a learned teaching interface is destroyed while current execution and first-child task ability are held fixed, descendant acquisition should fail, and the additional descendant outcome should be the feedback that repairs it. Randomly corrupting every weight would not isolate this effect.

The serious constructive wager is that neural training can find useful low-complexity teaching behavior under this coupled objective. Nothing in the recursion guarantees it. Long meta-gradients may be ill conditioned; the teacher can encode weights through examples; or the best outcome may be a constant family unable to acquire new capabilities. These are mechanism-level failure possibilities, not implementation footnotes.

## 7. Serious repairs attempted and what each does

**Repair 1: prevent teaching collapse by scoring an actual later task.** A constant self-reproducing network can minimize a replication discrepancy. Equation (4) instead anchors both descendant losses to external task outcomes, with support contexts varying across training. A constant family that cannot distinguish tasks then fails. This only works to the extent that the training task distribution tests meaningful variation; a narrow distribution can still select a sterile reproducer.

**Repair 2: preserve a teaching procedure, not identical weights.** Do not minimize parameter distance between generations. Teach through bounded examples/directives and score actual behavior. Neural coordinate permutations, changed internal encodings or a different legal child realization may survive. This avoids imposing weight identity as the scientific target. It does not guarantee useful compression or transfer to arbitrary architectures.

**Repair 3: do not insert a permanent, stronger teacher behind the loop.** In (3), w itself supplies the teaching episode. A frozen offboard teacher would make first-child success look like inherited teaching when it is merely external support. Such a teacher remains a legitimate strong baseline, with its storage and usage charged, but it does not satisfy this candidate's acquired capability criterion.

**Repair 4: distinguish semantic retention from simulation artifacts.** Store a finite old-task/context archive and audit whole EXEC behavior after the proposed update and descendant construction. Do not call preservation of a teaching syntax or an internal norm preservation of the capability. Finite tests remain finite evidence.

**Repair 5: charge the entire constructor.** Two descendants cost two inner learning trajectories, neural teaching evaluations, inspection, saved states or recomputation, external labels, and the outer derivative. A teaching program is not a constant-time macro. If the new interface does not repay its formation/maintenance cost in subsequent construction tasks, its practical acquisition claim fails.

These repairs make the hypothesis concrete. They do not supply the missing positive evidence that this organization improves substantial computation acquisition.

## 8. Closest primary boundaries and strongest alternatives

SOURCE: DreamCoder already uses learned programs to generate training problems for neural synthesis. Learning-to-Teach already makes student progress the teacher's feedback. SEAL directly generates its own training data/directives, applies resulting weight updates, and trains edit generation on the changed model's outcome. Self-replicating neural programs already reproduce the code that creates/trains them. SeRANN already couples genotype replication and a task. The paired v1 source note records original method locations rather than relying on abstracts. These precedents remove any broad claim to self-generated learning, neural self-replication, or feedback from a learner.

The proposed specific coupling is an acquired teaching computation that is itself constructed through the teaching channel, with real later-descendant task outcomes shaping its transmission. In this pass I have not established whether that exact objective/state has a prior implementation or theorem. It is a precise comparison question, not a novelty assertion. The 2021 neural-program source is especially close in reconstructing its own training machinery; reproducing arbitrary source code is not the same implementation as the bounded lesson channel, but a representational difference alone does not establish an important new consequence.

The strongest same-information conventional alternatives are:

- **Direct weight copying plus ordinary meta-learning.** It often preserves both modes for far less cost and must be allowed. If exact copying is legal and all future instances have the same parameter semantics, the candidate currently has no replication-efficiency advantage.
- **A retained external neural teacher / curriculum learner plus task solver.** It gets the identical contexts, student inspection, lesson grammar, external task stream, memory and compute. It may outperform the recursive constraint. Compare whether self-transmission changes later capability rather than simply spending more meta-training.
- **SEAL-style self-edits with a two-step meta-objective.** Give it the same ability to generate lessons and to evaluate the changed model's subsequent editing/teaching. If this is exactly equations (1)-(4) after renaming, it is a direct known-mechanism correspondence. No separate principle should then be claimed.
- **Program-library learning plus dataset/curriculum distillation.** It can retain the teaching code or data and reuse earlier programs. Matching only a fixed one-generation teacher would be weak; a competitor that retains successful constructors is required.

The scalar example is covered by trivial coordinate optimization and copying. Its importance is the exact capability distinction, not a solver advantage. The nonlinear proposal cannot be dismissed merely because an unrestricted program can simulate it, but neither can generic expressivity or a renamed meta-objective certify a new acquisition organization. An actual comparison must identify the learning channel, retained state, selective pressure, or resource consequence that differs.

## 9. Important success counterfactual

If this works in a consequential setting, learning a new computation would produce a mechanism that also carries a revisable way to reconstruct and specialize that computation in later internal instances. The ability to construct a learner would become a retained computational capability, exposed to task feedback through descendants. Such learning could accumulate a family of development procedures rather than requiring an ever-growing collection of frozen weight snapshots or an unchanged external teaching algorithm.

The potential paradigm-level prospect is an organization in which improvements to internal computation and improvements to the process that generates internal computation become causally coupled and heritable through actual learning. The word heritable here refers to the declared in-memory transmission process. There is no claimed ecology, autonomous spending, unlimited population, or open-ended evolution.

The decisive unproved step is whether that self-transmission pressure helps acquisition of new algorithms or useful compositions after all construction costs, rather than merely preserving a narrow self-reproducer or imposing an unnecessary burden. The best plausible success must include held-out *new capability acquisition* by descendants and preservation of the teaching ability after subsequent learning. A copied scalar, lower training loss, or additional successful generations alone would not suffice.

## 10. Return to root for independent review

This v1 provides one actual new constructive coupling after the v0 negative boundary. It is ready for conceptual/source comparison before any experiment: explicit neural modes, actual child updates, transmitted teaching, a second-descendant feedback law, a worked cause, costs, repairs and strong conventional alternatives. No training or expanded run is proposed as an automatic next step.

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION

Current task success and first-child success can coexist with failure to transmit the learning interface; later-descendant feedback can expose and repair that failure | RECONSTRUCTION, hand-checked finite example | Equations (5)-(7) and actual update law (1)-(4) | General nonlinear learning and useful new-computation acquisition not demonstrated | V0 algebra supplied; a one-child-only objective misses d; weight-identity replication is too narrow | Root independently assesses whether the explicit coupling is already a known two-step self-edit/teaching mechanism and whether a consequential residual survives.

A self-transmitted teaching procedure can support increasingly capable internal mechanism acquisition at an honest total cost | HYPOTHESIS / OPEN | Fully specified finite construction; primary-source boundaries; no empirical evidence | Costly recursive meta-learning may add a burden without a new useful acquisition effect | Constant replication, depth overfitting, synthetic-example weight encoding, direct-copy and permanent-teacher rivals remain strong | One source/mechanism judgment by root before deciding any further work; do not automatically execute.
