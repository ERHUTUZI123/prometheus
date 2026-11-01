# Acquisition v2: exact comparison and one representation repair

AGENT RECONSTRUCTION / ATTACK / CONSTRUCTIVE REPAIR. Same V2 campaign. Frozen v0/v1 candidates and source notes are unchanged. No experiment, training, installation, model/resource change, or MAIN-closing decision. Requested/effective provider/model/mode remain unavailable.

## Judgment

**Do not promote v1 or the repaired version below as a new mechanism-acquisition principle.** V1 specifies a real causal dependence: an acquired teacher affects the next learner's learning, and task performance alone can miss damage to that teacher. But its complete finite state and updates are a recursive self-edit/meta-learning graph, with parameter tying. A direct same-information implementation reproduces the proposed transitions and gradient, not merely its input-output function.

I gave one repair a constructive pass: retain a finite development recipe as the learning coordinate, regenerate the neural mechanism through actual SGD, and apply later task updates to that recipe. This changes subsequent learning even when present execution is identical. The effect is explicit and can be useful. However, the load-bearing general derivative is ordinary differentiation through a parameterization, and a fully worked finite case is exactly a conventional factorized parameterization. Dataset distillation already supplies the core representation of retained knowledge as synthetic training data.

The finite result is therefore an honest boundary: **transmitting teaching competence is a distinct outcome to measure, but neither the extra outcome nor the specified recipe representation currently supplies a different causal organization from the strongest ordinary alternative.** This is a branch-level recommendation returned to root, not a theorem against neural self-construction and not a MAIN decision.

## 1. Correct developmental state before comparing algorithms

V1's TEACH explicitly reads teaching history and fast state. Its weight-only derivative formula is sufficient only when those values are fixed or uniquely included in the differentiated weight state. The full algorithm needs augmented state. Preserve v1's scalar derivation, but use the following general law in this repair.

Let

    z_t = (w_t, H_t, child_fast_t, teacher_fast_t, other_retained_state_t).

Fixed context c, random seeds, and the finite step index are explicit inputs. A transition first produces the real lesson

    ell_t = Teach_theta(c, inspect(z_t)),

then updates child weights with that emitted lesson held fixed inside the child's gradient, updates actual fast states, and appends the actual lesson/outcome to H. Write the entire operation as

    z_(t+1) = Phi_theta(z_t; c, epsilon_t).                  (1)

With Q_t = D_theta z_t, the correct outer derivative is

    Q_(t+1) = D_z Phi_theta(z_t) Q_t + D_theta Phi_theta(z_t). (2)

The weight component makes the extra terms concrete. If

    w_next = w - eta(z,theta) omega(z,theta) G(w,ell(z,theta)),

then its total differential is

    d w_next = d w
               - eta omega [(partial_w G) d w
                              + (partial_ell G) d ell]
               - omega G d eta - eta G d omega,             (3)

where d ell includes the teacher's response to the history, fast state and inspected child as well as its direct parameter dependence. Neither these terms nor the H/fast-state update can be removed merely by calling G a gradient. The total state Jacobian generally need not be symmetric. A discrete lesson choice needs its actual discrete optimization rule or an explicitly declared surrogate; no exact derivative through that choice is invented.

Equation (2) also exposes the actual cost: all differentiable state needed in the reverse computation must be stored or recomputed. If the teacher inspects gradients, differentiating through those inspections can require additional derivatives. Bounded depth does not make these costs disappear.

The supervisor independently identified the history/fast-state issue before this v2; root passed it on. The augmentation above incorporates that contribution. V1's B(a,b,d) example has no such hidden state, so its hand calculation remains valid.

## 2. Actual same-information recursive meta-learning comparator

Use a conventional model with a task-answer mode and a data/directive-generation mode, with exactly the same tied parameter tensor theta. Give it the identical lesson grammar, init distribution, learner inspection, contexts, archives, protected examples, random streams, K-step budgets and numerical precision.

Define Adapt(teacher, child_init, context) as the actual K transitions (1). The conventional procedure is

    w = Adapt(theta, init_1, c),
    v = Adapt(w, init_2, c'),
    theta <- theta - rho D_theta J(theta,w,v).             (4)

It is essential that the second call uses **w as the data/directive generator**. This is neither an immutable external teacher nor a baseline that forgets the acquired TEACH behavior. The procedure retains all of z and differentiates through the two actual calls using (2).

| V1 state or action | Ordinary recursive meta-learning realization |
|---|---|
| N_theta.EXEC and N_theta.TEACH | Two modes of the same tied neural model |
| Generated lessons and update directives | Differentiable synthetic training data, weights and step sizes |
| Child weights/fast state/history | Inner optimization state and its actual trace |
| Child's acquired TEACH constructs v | Reuse the adapted model as the next edit/data generator |
| External current/child/grandchild outcomes | The same finite unrolled meta-objective |
| Parent update | The same reverse derivative through the same operations |
| Protected archive, failure records and handles | The identical retained state and acceptance procedure |

There is a pathwise correspondence, stronger than universal simulation: starting from equal full states and the same random stream, each emitted lesson is equal; each actual SGD/fast-state/history transition is equal; induction through K steps gives equal w, then equal v, task outcomes and outer gradients. Stored tensor sizes, numerical operations and gradient paths can be identical. No program search over v1 is needed by the rival.

SOURCE boundary: the SEAL methods already read in v1 implement own-generated edits, real inner SFT, and outcome-based outer learning. The inspected paper is not being claimed to have implemented exactly (4), nor does its stop-gradient RL estimator equal v1's full derivative. Equation (4) is the explicit recursive meta-learning comparator we construct here. The relevant fact is that v1 supplies no additional transition, observation, state or constructor beyond it. Parameter tying and using the adapted generator again are its complete additional coupling, not evidence of a different learning substrate.

The scalar missing-gradient result survives this comparison unchanged: the ordinary implementation also receives the second-descendant gradient to d. It is a real outcome distinction, not a separation between mechanisms.

## 3. What direct copying and a permanent teacher actually decide

### Direct copying

If a new internal instance has the same legal graph and parameter semantics, copying every persistent parameter and required fast state preserves both execution and teaching, to the copy's numerical fidelity. It costs a read/write of the retained state and allocation; it does not need K gradient steps to reconstruct that state.

For a task newly learned by changing theta, copying the changed theta transmits that acquired execution immediately. V1 instead needs to train its teaching behavior to transmit the change. A baseline that copies only EXEC and deliberately omits TEACH would be unfair.

Copying a broken teacher does not make its teaching computation correct. For example it copies (1,1,0) exactly rather than repairing d. But a copying baseline can construct later task-performing instances by copying again, without using that broken teaching path. Thus the scalar example does not show that self-teaching is necessary for the relevant task-acquisition outcome. Requiring every baseline to use the self-teaching channel would make that channel a task requirement, not demonstrate why it improves acquisition.

This is scoped: direct copying does not automatically give new algorithms, nonidentical task specialization, good variation, or arbitrary substrate transfer. None of those claims is used against v1. The issue is simply that, in v1's actual same-graph construction setting, no compensating reason for its more expensive reconstruction channel has been supplied.

### A retained permanent teacher

A permanent teacher psi can produce lessons for successive children without requiring each child to transmit the teaching procedure. Its causal graph differs from v1: damage to the child's TEACH can leave the next child unaffected because psi supplies the next lessons. That is a real intervention difference.

It does not imply that v1 has an advantage. One retained psi may teach many children sequentially. V1 already pays to retain neural mechanisms and all states needed for learning. No memory, locality or communication constraint in the proposal has made psi unavailable or shown that maintaining it is costlier. Artificially forbidding a permanent teacher or direct copy would create an interface separation, not establish the project's mechanism claim.

A teacher that learns from the same future-task stream, or retains successful conditional construction procedures, is the strong comparator. A static narrow teacher denied that evidence is insufficient.

## 4. One constructive repair: make the development recipe the retained coordinate

The original proposal retained theta, a full neural mechanism whose TEACH had to be reconstructed. This still left ordinary weight copying as the immediate inheritance method. I tried changing the retained representation itself.

Retain a bounded recipe xi instead of a finished weight tensor. xi contains a finite sequence of synthetic examples/targets/directives, or parameters of an adaptive lesson generator with its finite state. A declared initializer and actual K-step learner construct

    w = D(xi; w_init, seed).                              (5)

Current neural execution is F_w. A new internal instance can be regenerated by (5). When a new task supplies loss L(w), update the retained recipe by

    xi_next = xi - rho D_xi [L(D(xi))],
    w_next  = D(xi_next).                                 (6)

For this repair I abandon self-transmission as the supposed distinguishing property. No third/fourth descendant, extra task family, new loss, or heterogeneous child is introduced. The actual proposal is that *retaining and modifying a developmental recipe could organize subsequent variation more usefully than modifying a completed neural tensor*. This is a change of representation and transition law with an explicit finite cost.

The success counterfactual is consequential in principle: two systems with identical present computation could differ in which useful new computations a small later learning step reaches, because their retained development representations differ. That is an evolvability/learnability prospect, not a present-task performance claim. It needs a mechanism and comparison, which follow.

## 5. Load-bearing consequence and its exact rival

### General local consequence

For fixed init/seed and a frozen constructor/decoder during this particular later update, let J_D = D_xi D(xi) and g = grad_w L(w). Equation (6) uses ordinary Euclidean gradient descent in the declared recipe coordinates and yields

    delta xi = -rho J_D^T g,
    delta w  = -rho J_D J_D^T g + O(rho^2).               (7)

The first-order accessible directions are im J_D. The inherited recipe can therefore change the effective update metric and its nullspace, even for the same w and the same task gradient. That is a genuine causal representation effect under the specified coordinate metric; changing the recipe norm changes the learning law. A singular J_D can also make a needed change locally inaccessible. Nothing in (7) guarantees that the selected directions help a later task.

This is ordinary differentiation through a latent parameterization, not a new theorem about teaching. A conventional model with retained coordinate xi and decoder D has the identical pullback gradient. For a full-rank local metric, an optimizer with the same retained J_D J_D^T can reproduce (7) to first order. This local statement alone does not prove full finite-step equivalence of arbitrary nonlinear recipe learning; the following case does.

### Exact finite worked case

A one-weight linear neural student computes f_w(u)=w*u. It starts at w_init=0. The retained recipe is one synthetic training example (x,y), and a unit-rate step on half squared error gives

    D(x,y) = w = x*y.                                    (8)

For a new external scalar task target t, set L(w)=(w-t)^2/2 and e=xy-t. Equation (6) performs the actual recipe update

    x_next = x-rho*e*y,
    y_next = y-rho*e*x,

and regeneration gives exactly

    w_next = xy - rho*e*(x^2+y^2) + rho^2*e^2*xy.         (9)

This is a finite nonlinear learning consequence, not just an asymptotic derivative. Recipes (x,y)=(1,1) and (2,1/2) both execute w=1. With t=0 and rho=1/10, the first produces w_next=81/100; the second produces

    x_next=39/20,   y_next=3/10,   w_next=117/200.

The present function is identical, but the next acquired function differs under identical task information and recipe learning rate. The first-order metrics are 2 and 17/4 respectively. These values agree with (7).

The exact conventional rival is a factorized parameterization w=x*y trained by ordinary gradient descent on L(x*y). It uses the same two stored scalars and performs exactly the same updates in (9), for arbitrary finite rho and every t. It can compile regeneration into one multiplication; no student-gradient execution is needed to implement (8). This is not a black-box universal simulator or the entire new proposal assigned a different name. It is the elementary standard factorized model exposed by the actual recipe.

There is also a direct failure: at x=y=0, the recipe gradient is zero for every nonzero target t, although grad_w L at w=0 is -t. A direct weight update can move immediately. Indirect learning can make a simple acquisition harder. Conversely, rescaling x by s and y by 1/s changes the effective learning rate without changing execution; a speedup produced only by that scale must face ordinary step-size/preconditioning controls.

These are hand calculations, not executed results. They establish that the recipe changes future learning, and also that the worked change is fully covered by a simpler ordinary parameterization.

## 6. Stability and costs do not rescue the interpretation automatically

The source-driven repair has genuine possible uses: compact retention, regeneration, or a learned bias over later changes. None is free. A retained recipe must count the initializer, seed or distribution, generator weights, all targets, numerical precision and its K actual learning steps. Distilling a large external dataset is not automatically compression relative to storing the final internal mechanism.

In (8), the recipe stores two scalars to reproduce one weight, while a copy stores one. The compiled factorized rival gets the same representation geometry more cheaply. In a vector one-step construction, w=eta X^T y, retaining X and y may exceed the size of w; restricting X to a small learned subspace changes what can be reconstructed. A conventional latent model retaining X and y can execute exactly that map with the same information. A useful memory/variation tradeoff would need to be demonstrated for a substantive mechanism, not inferred from the existence of a recipe.

An attracting developmental family could make regeneration robust. But attraction alone can select a constant or uninformative family. Any stable-family repair would still have to preserve task-conditioned variation and show a useful later acquisition consequence with copying and the same latent/meta-learning controls allowed. No such new family has been constructed in this pass. Adding a stability penalty or naming a fixed point would not supply it.

Likewise, any finite tested descendant depth admits a delayed failure beyond that depth. For instance a finite transmission state can shift a future bad teaching value toward the active position only after the assessed number of births. Passing a finite chain is evidence at that depth, not indefinite closure. I do not repair this by adding another descendant.

## 7. Primary-work boundary and precise closure

SOURCE: Wang et al.'s *Dataset Distillation*, original methods section 3.1, makes updated neural weights a differentiable function of synthetic data and optimizes that data through the update. Sections 3.2 and 3.4 cover initialization distributions and multiple training steps. This directly covers the core idea of retaining knowledge as a finite training recipe. The new source note records exact locations. No source performance numbers are needed here.

SOURCE: Rusu et al.'s *Meta-Learning with Latent Embedding Optimization* explicitly decodes a latent representation into weights, updates the latent representation through that decoder, and trains the generator using later task loss. That is a concrete prior for the repaired adaptation geometry. Its inspected method does not establish coverage of the full self-transmitted TEACH coupling; the exact v1 comparator is supplied in section 2 instead.

RECONSTRUCTION: equation (7) explains how changing retained learning data can change later acquisition geometry; equations (8)-(9) show a complete finite case. These are not a novelty claim. The particular original recipe-representation wording does not create a different mechanism from differentiation through a learned parameterization.

The closure is precise and limited:

- V1's full dynamics have an exact recursive meta-learning realization with the same state, information and operations.
- Its teaching-transmission outcome is genuinely distinct from current execution and first-child execution, but direct copying can satisfy the important task-construction outcome without repairing that channel in the stated same-graph setting.
- Replacing retained weights by a development recipe creates a real variation/learning effect; the constructed effect is exactly a conventional factorized parameterization and its general derivative is standard latent-coordinate learning.
- No specific mechanism has yet been supplied that makes self-transmitted teaching or retained recipes improve acquisition of substantial new computation after honest cost against these alternatives.

This does not establish impossibility of learning useful development procedures, neural self-construction, or a meaningful future stable family. It says the present construction has exhausted its specified difference. A numerical descendant experiment, an additional task, or more generations would currently test implementation or a known meta-learning effect rather than the missing organizing principle.

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION

V1's acquired teaching affects subsequent internal learners | RECONSTRUCTION; exact finite causal distinction | Frozen v1 B(a,b,d); full augmented transition (1)-(4) here | No distinct mechanism beyond recursive self-edit/meta-learning | A weight-only recurrence omitted history/fast-state Jacobians in the general setting; finite depth does not imply closure | Root records the corrected finite interpretation.

Retaining a learning recipe changes later acquisition while preserving current execution | RECONSTRUCTION; exact hand consequence | (7)-(9), including 81/100 versus 117/200 from identical w=1 | A useful new organization beyond latent parameterization is absent | One-lesson repair is exactly factorized optimization; zero recipe blocks a directly available update; extra storage/work is real | Root should close this candidate for new-principle promotion, retain the verified distinctions, and allocate any further work independently. No MAIN completion is asserted.
