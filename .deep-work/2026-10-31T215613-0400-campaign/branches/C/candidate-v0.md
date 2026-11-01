# Branch C frozen v0 - operations inherited as regenerative constraints

Frozen independently on 2026-10-31, before exposure to other branches or earlier Prometheus architecture reports. Route: inherited current user-authorized model / Ultra, as supplied by coordinator; not Pro. No separate model backend identifier or measured inference budget is available to this branch.

## Judgment and question

**HYPOTHESIS / AGENT PROPOSAL:** An internal computational operation can be stored partly as a *learned criterion for what must remain true*, together with a learned process that reconstructs a circuit satisfying it. It need not be inherited only as frozen weights or a frozen subgraph. When several such operations are composed, their circuit realizations can adapt together while their behavioral criteria remain active. The scientific opportunity is not extra representational universality; it is reducing the simultaneous coordination needed to construct a useful new operation without destroying its supporting functions.

Exact question: **Can learned local behavioral constraints and a reusable construction process make new stateful operations accessible under a fixed total learning/search budget, beyond an equally large library of frozen circuits with equally trained boundary adapters?**

The candidate answers the Fundamental Question by changing the object a neural learner constructs and retains: from a numerical activation or a reusable fixed circuit to a *regenerating computational mechanism*. It is not yet a demonstrated learning system. There is no novelty claim, no evolution claim, and no claim of open-endedness.

## What the reading changed

**SOURCE:** Kirschner/Gerhart, selected printed pp. 111-118, 143-152, 171-176, 219-222, treats weak linkage as an indirect, minimally demanding connection, not merely a small edge weight. Its microtubule example separates a broadly generating process from local stabilization of its products. The limb discussion suggests an important role for support processes that accommodate a changed structure without a separate inherited change in every support tissue. The book explicitly restricts facilitated variation to a component of evolvability rather than proving population-level future success (printed p. 220).

**RECONSTRUCTION:** The transferable move is to preserve a generative-and-repairing process and change what stabilizes its products. A learned operation may therefore retain a *stabilization rule*, rather than retain every detail of a successful implementation. This is more specific than modularity or ordinary reuse.

**SOURCE:** Alon's chemotaxis reconstruction (printed pp. 146-149; PDF pp. 163-166) depends on a constant reference flux and an opposing flux that directly senses the activity being regulated. Its robustness breaks under saturation failure or demethylation of inappropriate receptor states. It preserves a particular observable, not arbitrary function.

**RECONSTRUCTION:** A neural analog needs a functionally adequate acceptance signal and a feasible correction channel. Mean activity, sparsity, energy, or prediction confidence alone cannot certify that a learned computation survived. Hence the candidate uses *behavioral trace constraints*, and this is its hardest learned component.

**SOURCE:** Ashby, sections 13/9-13/10 and 14/4-14/6, separates directly choosing repeated responses from constructing a machine that produces them, and attributes apparent amplification to supplementation from other information sources. Sections 11/5-11/9 give conditional bounds rather than a magic law creating useful variety.

**RECONSTRUCTION:** The generator cannot be an unexplained oracle. Its information comes from prior task feedback, executed boundary traces, interventions, and the environment. Its costs must be counted. A neural contract can itself hide the desired computation; removing it during execution is a decisive audit.

## Representation and executable laws

### 1. Runtime substrate

Use real-valued recurrent circuits with a graph G, weights theta, and state h. Available elementary nodes are bounded affine sum, smooth threshold, product, and one-step state delay. These are designed primitives. No write/read register, equality tester, timer, parser, stack, or whole task solver is installed as a primitive. A bounded arithmetic circuit can approximate the proposed example; this does not establish efficient discovery.

One execution tick is h[t+1] = F[G,theta](h[t], u[t]); y[t] = R[G,theta](h[t], u[t]). A graph edit changes a *future executable transition law*, not the record of previous outputs. All runtime state resets and graph changes are logged separately.

### 2. Retained operation

Store an operation O_i = (E_psi_i, q_phi_i, G_seed_i, theta_seed_i, port schema, reset rule, resource bound).

- E_psi_i(U,Y) >= 0 is a small learned neural energy over short ordered input/output boundary traces. It sees no hidden task answer at deployment. It states which boundary relations the operation should preserve.
- q_phi_i(e | G, local trace, energy residual, context) is a neural policy for bounded graph edits: add an elementary node, add/remove an edge, duplicate a subgraph, insert a retained operation instance, change a port attachment, or stop. q starts with exploratory edits and is trained from measured improvement. Maximum graph size and rewrite depth are external hard limits.
- A seed graph can speed construction but is not obligatory identity of the operation. Different contexts may lead to different interior graphs realizing the same tested relations.

This is **inheritance across episodes** (stored parameters and circuit descriptions), not biological heredity or a population process. Selection means an explicit acceptance decision below, not evolution.

### 3. Construction and accommodation

A task graph holds several operation instances. For a fixed bounded batch B of actual context traces, minimize

J(G,theta) = L_task(B;G,theta) + sum_i lambda_i E_psi_i(U_i,Y_i) + rho C(G) + eta C_build.

The task labels/rewards occur only during authorized training episodes. The energies are previously learned internal constraints. C includes active nodes, executed operations, stored contract parameters and graph size. C_build counts proposed rewrites, teacher calls, gradient steps, and replay evaluations; they do not become free because they happen during development.

One construction transition is:

1. Sample e from q_phi and apply it to a private tentative graph.
2. Execute the graph on B; do B_theta bounded gradient steps on theta using task and contract gradients.
3. Update lambda_i <- clip(lambda_i + kappa*(E_i - epsilon_i), 0, lambda_max). This is a discrete integral pressure on *constraint violation*. It is not a proof of stable integral control for a nonlinear changing graph.
4. Compare held-out development loss and old-operation violations with a sham edit at equal budget. Accept only if task gain minus measured cost is positive and old-operation violations remain within tolerances. Otherwise revert. All checks are finite-sample checks.
5. Train the edit policy by REINFORCE, Delta phi proportional to (accepted utility - baseline) grad_phi log q_phi(e|...). No undefined optimal edit chooser is present. Ordinary structural search is the fallback when the learned policy has no information.

Accommodating old instances is the key organizational interaction. Adding a new candidate does not force their numerical states or interior weights to remain fixed; the old instances can reconstruct within the equivalence class approximately defined by their behavioral constraints. Hard-freezing all supporting details may unnecessarily make a useful new composition unreachable within a small edit budget.

### 4. How an operation is captured without a semantic oracle

After a successful training graph exists, sample a connected subgraph R using a bounded learned boundary-selection policy p_kappa. No source-code semantic analyzer selects the perfect subroutine. Record its boundary input/output trajectories. Perturb each boundary input within a predefined range, reset state, and execute the *original R* to generate additional teacher traces. These calls are charged.

Train E_psi as a conditional energy: low on R's paired traces, high on corrupted output traces (time shifts, state-reset changes, output swaps across different inputs, and sampled numerical perturbations). Use a contrastive margin or binary logistic loss. Corruptions are designed; the task-specific relation is not supplied in symbols.

Train q_phi to reconstruct low-energy, low-cost implementations from multiple perturbed seeds, using the rule above. The original R is retained as a teacher during capture, then withheld in the reconstruction audit. Selection of the boundary and retention of the operation use downstream task loss after replacing R by a reconstructed instance, less capture/storage/build costs. A REINFORCE update to kappa supplies the boundary selector's only task-directed information.

This is a specified but potentially inefficient procedure. It does not assume that every useful concept has a small subgraph, that negatives identify the correct relation, or that energy generalization follows from fitting traces. It also does not claim an energy critic is computationally cheaper than R. Those are open conditions.

At normal execution, construction and all contract critics are *removed*; the realized circuit runs F[G,theta]. Rebuild/repair can occur between bounded episodes, but a result requiring the critic to keep solving the task at every runtime step is classified as energy-based inference, not as successful construction of a persistent operation.

## Worked conceptual computation: capture a memory, then change what can be built

This is an **unexecuted explanatory example**, not a training result. Suppose task feedback has led a graph built from sums, products and delay to realize

m[t+1] = m[t] + w[t]*(v[t] - m[t]);   y[t] = r[t]*m[t].

Here w is a write pulse, r a read pulse, and v an input value. The update is a learned composition of designed elementary operations, not a preinstalled memory-cell instruction. A capture boundary around m's transition creates an operation with ports (w,v,r) and reset m[0]=0. The *ideal relation* its learned energy would need to approximate is: after a write, later reads reproduce the most recent written value despite intervening distractors. That relation is written here to explain behavior; it is not supplied to the learner as a handcrafted contract.

Concrete ticks (output is read from the state before that tick's write):

| t | (w,v,r) | m before | y | m after |
|---|---|---:|---:|---:|
| 0 | (1,3,0) | 0 | 0 | 3 |
| 1 | (0,9,0) | 3 | 0 | 3 |
| 2 | (0,-4,1) | 3 | 3 | 3 |
| 3 | (1,5,0) | 3 | 0 | 5 |
| 4 | (0,8,1) | 5 | 5 | 5 |

On a later task, the edit policy can insert a newly reconstructed memory instance and wire its output into a new subtraction/comparison process: write the first measurement a, delay through distractors, then emit sign(b-m). For a=3, b=5 the derived value is positive; for a=5,b=3 it is negative. The latter capability was not an elementary instruction. The new retained memory operation changes the probability and edit length of building this process: a policy can request the memory relation in one structural edit, while the generator supplies a contextual realization.

What is potentially new versus an ordinary library call is **joint developmental accommodation**. The memory realization may alter its internal gain, placement of state, or circuit decomposition when embedded with a consumer, while the boundary relation remains tested. The new consumer and its supporting memory need not be independently correct before they are combined; old-function constraints and present task feedback can guide a compatible joint realization. This is a finite-budget accessibility hypothesis. Mere rescaling is only a change of representation and is not counted as a new capability.

The example does not show that the learner discovers the correct memory boundary, learns the relation rather than finite traces, or constructs the comparator economically. In fact, if a sufficiently expressive boundary adapter around a frozen memory does equally well, this example supplies no reason to prefer the candidate.

## Designed / learned / selected / retained

| Dimension | What exists in the candidate |
|---|---|
| Designed representation | Elementary sum, threshold, product, delay; graph representation; bounded ports and traces |
| Learned representation | Circuit graphs/weights; operation boundaries; neural trace energies; edit distributions |
| Local execution | Explicit recurrent graph transition F; no oracle or simulator narration |
| Online adaptation | Weight updates and graph edits within an authorized construction phase; none silently during frozen execution |
| External feedback | Supervised targets or scalar rewards for training tasks and development acceptance |
| Internal stabilization | Previously learned trace energies and bounded multiplier updates |
| Variation | Sampled graph edits and gradient steps; learned q changes their probabilities |
| Selection | Measured improvement after equal-budget sham comparison and old-behavior checks |
| Retention | Stored energy/generator/seed/ports/reset rule; selected circuits persist until replaced |
| Evolution | Not claimed; no population inheritance/selection is specified |

## Strongest rival and central unresolved step

**ATTACK:** This may be neural architecture search plus distillation plus program-library learning, with an expensive critic replacing the original function and extra replay accounting for any gain. Conditional generative modeling, neural developmental encodings, energy-based inference, and adaptive modules are close conceptual neighbors. A new combination is not yet a new mechanism result.

**Central OPEN step:** Can a small learned stabilization signal define a sufficiently rich *functional equivalence class* to allow useful reconstruction and composition, without itself performing the entire desired computation or admitting catastrophic shortcuts? A memory trace critic trained on short windows can accept a leaky memory that fails at longer delays. Good training reconstruction alone does not settle this.

Constructive repair worth protecting: retain a small *set of counterfactual trace families* for each operation, grow that set only when an execution failure exposes a missing distinction, and co-train the construction policy and critic on multiple embeddings. The operation then carries a finite executable test distribution, not an asserted full semantic specification. This costs memory and does not guarantee generalization, but it removes the requirement of learning a perfect symbolic contract in one step. All test generation uses the original learned operation or observed task failures; no arbitrary correctness oracle is introduced.

## What would change the judgment

The discriminating comparison is not against a tiny fixed RNN. It is against (1) frozen module library plus equally trained adapters and the same replay budget; (2) deformable modules with no learned trace critic; (3) a flat learned graph-edit policy with equal total capacity and proposals; and (4) ordinary program-library capture with the same teacher calls. Count all critic, construction, capture, and storage costs.

Development tasks should require acquiring stateful operations before held-out compositions and delays. Lock evaluation combinations, seeds, graph ceilings and total transition budgets independently before running. Measure acquisition of a *new tested operation*, preservation after later construction, and transfer to held-out compositions separately. Do not infer one from another.

A favorable result would require: a persistent circuit that runs without critics; retained old behavior after controlled structural perturbations; improved probability or budget-to-discovery on held-out new compositions; and an intervention showing the improvement disappears when stabilization is broken while current capability and search budget remain matched. No such result exists yet.

Stop this candidate if the critic does the runtime task, if generalization requires hand-coded task contracts, if only representation rescalings improve, or if equally budgeted adapters/flat search explain the gain. Keep it open if useful behavior is constructed but the critics remain too broad; that is a repairable identification problem, not an immediate contradiction.

## Unread scope at freeze

Read selected passages only, as itemized in source-reconstruction.md. Not read: the full books, their full source chains, earlier Prometheus architecture reports, fire/, other branches, or a systematic neural construction/prior-art survey. The specific novelty boundary remains OPEN. Freeze precedes the closest-prior-art attack.
