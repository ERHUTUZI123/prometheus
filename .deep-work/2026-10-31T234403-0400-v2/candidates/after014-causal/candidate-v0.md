# Causal acquisition after014 — v0, unfinished construction

**AGENT PROPOSAL / HYPOTHESIS.** The candidate is **learning an executable repair operation from task-validated splices of unsuccessful computations**. It does not first solve a task, capture a correct subgraph, and train a contract for its reconstruction. It searches internal interventions before a correct operation exists; a successful intervention becomes an imitation target for a new callable transition. This is the concrete revision after reading old C/CD, which already covers extraction, distillation, learned behavioral contracts, and context accommodation. No novelty or improvement is established.

Mode requested by parent: existing same-campaign route. Effective mode unknown. Parent reports real fork-none, but the supplied seed contains prior comparisons/exclusions and is not pristine independence. This child does not independently verify host context isolation. Its source helper used fork-all. File isolation is instruction_only. No experiments or code commitment. Own files: this document and sources/after014-causal/reading.md only.

## Exact question and success counterfactual

Can task-grounded internal splice search discover **state transformations that were absent from successful executions**, install them as reusable neural calls, and thereby acquire later compositions with lower total search/training cost than the same-info combination of dense self-imitation and program/skill recombination?

If successful, this changes the unit of useful learning evidence: two failed computations can supply a correct internal transition that neither execution supplied as a whole. The ambition is autonomous construction of the operations that later computation can call, not more accurate interpretation of a fixed solver. The structural prospect is that a useful partial computation may be much more common than a complete successful trajectory. The unresolved question is whether causal splice discovery and amortization can exploit that gap after all donor search, validation, formation, and maintenance costs.

## Current machine and information sources

Runtime state is (h,t,L), with h in a bounded neural workspace and L a persistent bank of callable residual transitions. Initially L is empty. The designed fallback is a bounded recurrent net F_theta; affine maps, nonlinearities, gradient descent, a bounded clock and generic token inputs are legitimate supplied primitives.

    h[t+1] = F_theta(h[t], v[t])                  fallback transition
    h[t+1] = M_j(h[t], v[t])                     installed call
    M_j(h,v) = h + W_j g_phi_j(R_j h, v)         candidate read/write interface

R_j and W_j have learned bounded rank; they are not an alignment to human variables. A bounded router selects fallback or a library call and is trained from ordinary task loss/reward. Read/write rank, module count, execution depth and precision are charged, not free abstractions.

Training receives raw task instances (x,q), input tokens v, and a terminal evaluator V(x,q,y). V consumes **the original raw task instance and the actual emitted answer**, not hidden activations, a latent variable label, or a hypothetical recombined input. Supervised I/O corresponds to V=1[y=y*]; an environment may supply a terminal scalar reward instead. Its query cost is charged. The mechanism does not manufacture correctness labels from neural self-consistency.

The algorithm may read and intervene on its own workspace. It has no supplied high-level SCM, semantic module labels, causal alignment, correct intermediate states, valid input-recombination map, or evaluator for arbitrary hidden states. Public source tokens are available as potential operands. The choice of task family and evaluator is still designed information.

## New learning-state transition

Replay D stores unsuccessful and successful bounded trajectories, including their raw task data, neural states, outputs and evaluator scores. An intervention proposal is a base trajectory b, time t, donor trajectory s, time u, and a bounded-rank projector P. Its actual execution is

    I_P(h_b,h_s) = h_b + P(h_s-h_b)
    y_cf = Continue_theta,L(x_b,q_b,t,I_P(h_b,h_s))
    r_cf = V(x_b,q_b,y_cf).

The continuation does not receive donor task data. It receives whatever ordinary base input stream remains; any ability to recompute from it is measured with matched controls. A separate forced-call audit can withhold already-consumed input rather than silently assume it is absent.

The proposal law is finite and executable: sample trajectory pairs/times and P from bounded dictionaries or a learned proposal distribution; score by r_cf-r_base minus measured execution/query costs. Score-function learning can amortize proposal selection. No oracle chooses a semantically correct projector. Brute search is the fallback and may be prohibitive.

**Installation event (the actual acquisition rule):** collect task-validated patches into a proposed class j, with examples (h_b, v_s, h_cf), where h_cf=I_P(h_b,h_s) and v_s is an explicitly stored public source token/window. Fit a fresh residual transition by

    J_j = E[ || M_j(h_b,v_s)-stopgrad(h_cf) ||^2
              + lambda * L_task(Continue_b(M_j(h_b,v_s)), y_b*) ]
          + beta * (rank(R_j)+rank(W_j)+stored_bits+runtime_cost).

For non-differentiable V the second term is optimized by sampled reward; the state-imitation term remains differentiable. This may be inefficient. Hard masks may replace relaxed ranks after development; this is not an executed design choice.

The class assignment/proposal j is learned from predictive/task benefit, not a semantic label. A candidate j is promoted into L only if a **donor-free forced-call** realizes task improvement on development instances and operands excluded from its fitting records, repays its measured construction cost on the declared horizon, and survives sham patches with the same perturbation size. If no class generalizes, reject it; do not retain an activation lookup and call it an operation. Router fitting and every attempted rejected installation are charged.

At later normal execution the archived donor and intervention search are absent. M_j computes a new state from current h and operand v. The new machine can execute a call that was not in L initially. A fresh later task can compose calls without semantic names, using only its own task feedback. This is an actual change to the available executable transition set, not a probe or a representation-only score.

## Fragile step already visible

Task-valid patches are **not automatically a consistent operation**. Different donor states may encode unrelated ways to get one answer right. State imitation can conflict, and continuation can repair the patch or ignore it. Therefore task validity alone does not justify module promotion. Generalizable donor-free fitting and fresh-context forced-call utility are separate, unpaid obligations. No result yet shows they hold.

### Serious repair: discover the interface by prediction conflicts

A successful patch is not admitted as a clean target merely because its terminal reward is positive. Within proposed class j define delta=h_cf-h_b and z=(R_j h_b,v_s). A deterministic donor-free operation requires conditional target variance E[Var(delta|z,j)] to be small on the declared distribution. The exact zero condition is an analytic necessary condition, not an observable finite-data certificate.

When paired records with similar z disagree in delta, the learner tries one of three **charged structural changes**: append a read direction from the omitted h (rank-one expansion); enlarge the public operand window by one token; or split j with a learned router using only runtime-visible (h,v). Rank-one candidates can be sampled or fitted to held-out residual-prediction reduction. Accept a change only if the donor-free task loss and validation prediction error fall enough to pay its added cost. If the residual depends on donor state not recoverable from these allowed inputs, reject the operation. A source-activation lookup is not a repair.

The write space W_j is similarly expanded only if fresh task improvements require residuals outside its current range. These rules make read/write structure itself an acquired object. They do not guarantee that expansion terminates in a small useful operation. Enlarging to the whole task solver is explicitly a failure of reusable module formation.

## Worked construction: two wrong internal transitions can train one correct call

**RECONSTRUCTION / explanatory construction, unexecuted.** Consider raw binary addition tasks. A recurrent workspace processes digit pair (a,b) with a carried state c. The analyst describes the desired full-adder transition by

    s = a XOR b XOR c
    k = (a AND b) OR (a AND c) OR (b AND c).

These equations explain this example; the learner is supplied only raw addends, emitted answer bits, terminal answer evaluation, and its own neural traces. They are not supplied as a high-level SCM or intermediate labels. Assume two stochastic fallback computations currently produce intermediate pairs

    A(a,b,c) = (s, 1-k),
    B(a,b,c) = (1-s, k).

Each complete two-bit result is wrong on every input triple. Neither provides a successful full-adder transition. With P projecting onto the second coordinate,

    I_P(A,B) = A + P(B-A) = (s,k).

At (a,b,c)=(1,1,1), A=(1,0), B=(0,1), while the actual splice is (1,1), encoding the correct total 3. V receives the original raw numbers and the emitted 3 and accepts. It never sees or labels the internal words “sum” and “carry.” A finite search over coordinate masks and donor pairs can find this patch from its actual terminal consequence. This describes a favorable internal organization; it does not show that an arbitrary distributed network has it or learns it.

Fitting one donor-free M on such validated state targets can realize the nonlinear transition (c,a,b)->(s,k). Later recurrent calls can use its k as the next digit's carried state. A longer addition is executed by repeatedly calling the acquired transition; neither archived donor nor patch search is required. However, learning operand binding and repeated routing from later task feedback is still necessary. One cannot infer that composition was learned merely because the mathematical transition composes.

### The same-info rival wins the explanatory example

Give an ordinary network both A and B activations, raw input/output examples, and the same evaluations. It can learn a two-output adapter selecting A's first component and B's second component; ordinary supervision on the emitted two-bit answer identifies that choice. Its executable transition is identical to M. A program constructor with feature reuse can make the same pair. Causal patch validation supplies no additional task information and pays extra queries here. Thus this example establishes **possibility of forming a correct operation from unsuccessful executions**, not an advantage or an original principle. It is not a discriminating experiment and should not be implemented as one.

Unknown semantic names do not rescue the comparison. A coordinate permutation only adds an equally available adapter search; a nonlinear scrambling can invalidate the low-rank splice assumption or make both learners' inference harder. “Distributed” is not an advantage certificate.

## Strong attack on the broader prospect, and what would repair it

Partial computations can be abundant while useful assembly remains inaccessible. In the limiting assay with k binary choices, no informative observations about the right combination, and only the all-or-nothing equality reward, there are 2^k indistinguishable correct-target possibilities before queries. Under a uniform target, q distinct tested assemblies have success probability q/2^k (for q<=2^k). Adaptivity does not help before the first positive answer, since all prior answers are the same failure. This is a direct finite counting argument under the stated information assumption, not a general learning lower bound.

Therefore the candidate's motivating abundance argument is insufficient. It needs an **additional structural premise**: the raw task family must contain multiple continuations/contexts whose ordinary task feedback separates candidate internal effects, and these contexts must be discoverable within the same metered search budget. No coordinate-correctness oracle is granted.

The proposed repair is active **context-witness acquisition**. Store, for each proposed j, ordinary raw tasks/continuations on which replacing its patch by a size-matched alternative changes terminal reward. A witness is an executed original-task contrast, not a symbolic claim about the correct latent variable. Let the proposal policy prioritize contexts with high empirical disagreement among candidate M_j and route those contexts through the actual task evaluator. Keep both positive and negative patch records. The library grows only when its donor-free operation fits these distinguishing contexts and repays all witness search. A conventional active self-imitation/program-synthesis rival receives those exact records too.

This is a real rule revision: unsuccessful computations generate candidate internal transitions; learned distinguishing contexts drive read-interface growth and admission. It does not fall back to a correct captured teacher or a supplied causal model. Its scientific value still depends on demonstrating cheaper reusable formation than the rival, which is **OPEN**. There is no current reason to admit a large experiment. The next useful check is a theoretical or small source-grounded task construction where witness acquisition provably/executably avoids hidden high-level labels and has a concrete resource difference from equal-information dense/module learning.

## Prior-work boundary entered before implementation

Old C original body: branches/C/candidate-v0.md, §§runtime–capture, captures an already successful subgraph, teacher-perturbs its ports, learns a trace energy and reconstructs it. Old D original body: archive/D-v0.md, learns Get/Put views and lifted editable operations. Old CD actual full body branches/CD/v1.md now read: source-version transitions provide field-residual repair targets; the complete finite realization is state assignment/circuit synthesis, and equal-information frozen-plus-adapter emits identical code. Those mechanisms are not claimed anew. This candidate changes the target's origin: task-validated recombinations of failed runs, not encoded correct source transitions.

Original contemporary methods: IIT supplies a high-level model/alignment and uses its counterfactual label; DAS learns alignment while models are fixed; Asiaee 2026 discovers and compiles mechanism replacements without a semantic high-level model; UCAD 2026 discovers factors from low-level interventional regimes; SIL 2026 predicts structural intervention consequences and uses them for action. Consequently “no supplied high-level model,” “self-intervention,” and “validated executable rewrite” are **not** the candidate's claimed difference. Exact source notes belong in reading.md.

**Current provisional distinction:** promote reusable donor-free transitions from task-validated hidden splices among failed executions, rather than preserve/interpret an already correct network. Closest actual rivals include neural trajectory stitching, policy/skill composition, hindsight/self-imitation learning, library-learning program synthesis, and any modular meta-learner allowed the same hidden replay and successful splice demonstrations. Primary comparison incomplete.

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION
---|---|---|---|---|---
Task-oracle splice evidence can generate a new callable transition without a supplied latent causal label | explicit unexecuted mechanism; no advantage established | state/intervention/installation/interface-growth laws; worked nonlinear transition; equality-query counting attack | finding economical separating contexts and a consistent donor-free operation | naive capture overlaps old C/CD; unsupervised rewrite covered; simple full-adder realization ties ordinary feature adapter | find a concrete resource distinction for context-witness acquisition before any code, otherwise PARK this version
