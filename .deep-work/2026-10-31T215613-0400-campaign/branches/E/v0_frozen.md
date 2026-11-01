# Branch E: frozen construction before closest-prior attack

Frozen 2026-11-01T02:13:44Z. Inherited host route is the current user-authorized Ultra context; no claim of Pro, model ID or measured effort beyond host-provided metadata. Instruction-only isolation: only the common seed, contract and Independent researcher role have been read. No branch or earlier architecture report read. Original food PDFs unchanged.

## QUESTION and scope

Can a network learn a reusable **conditional transition mechanism**, by allocating material that changes activation barriers rather than treating its state variables as weighted-sum neurons? The important potential change is what is learned: a physically executed, stateful local program (including its holding time and reset), rather than an extra feature in a static input/output map. This is a candidate route to the Fundamental Question, not a claim that a biochemical substrate is necessary or superior.

GIVEN: elementary stochastic chemical kinetics, explicit scalar terminal feedback, a designed library of conformational sites and two-arm catalyst scaffolds, and a bounded scaffold pool. NON-GOALS: autonomous invention of chemistry; emergence of the learning rule; open-ended evolution; solving arbitrary algorithm synthesis. VERIFIER: an independently integrated finite master equation and a small trajectory-learning probe; they can check this mechanism and reveal leakage, not establish scalable discovery or physical fabrication. BUDGET: one construction, one closest-prior attack, one bounded probe with at most a documented numerical repair; no purchase or external release. STOP: deliver a grounded certificate even if the candidate is an existing mechanism or fails.

## Reading that changed the construction

SOURCE (book exposition, not a new empirical result): Sterling and Laughlin, *Principles of Neural Design*, printed pp. 105–106, 117–123 (PDF pp. 130–131, 142–148), treats neural computation at protein binding/conformational/catalytic scale. Figure 5.5, printed p. 118, gives an actual multi-state AND program: A binding exposes kinase sites; B-dependent phosphorylation exposes a catalytic site; substrate digestion yields C; release/reset completes the cycle. Thus a scalar synaptic weight is a modeling decision, not the only internal mechanism a neural system could construct. Printed pp. 120–121 separate amplification, shutoff, and reset; the latter cannot be omitted when composing mechanisms.

SOURCE: Pfeifer and Bongard, printed pp. 107–109, 123–125 (PDF pp. 132–134, 148–150), makes the task-distribution point concrete. The passive walker works only in a restricted slope/material niche; muscle/tendon elasticity implements return to a default pose. This changes the proposed test: merely observing a useful trajectory is insufficient; input preparation, reset, timing and downstream load must be named parts of the computation.

SOURCE: Peliti and Pigolotti, printed pp. 38–41 (PDF pp. 61–64), equations 3.2–3.9, separates an energy difference from the symmetric intrinsic jump rate and imposes local detailed balance under driving. Printed pp. 60–62 (PDF pp. 83–85), equations 3.76–3.81, warns that combining physical channels can hide entropy production. Printed pp. 112–115 (PDF pp. 135–138), equations 5.18–5.23, distinguishes energetic from kinetic discrimination. This makes **learned transition-state barriers with fixed energy minima** a concrete possibility, while forbidding a one-way "free" gate.

A useful correction: Sterling/Laughlin's printed pp. 122–123 uses loose "register a bit" Landauer language. Peliti/Pigolotti p. 112 explicitly says copying into a prepared reference need not have a Landauer lower bound; finite speed, reset and errors matter. I will account for actual chemical work of individual channels, and will not equate a numerical loss or every transition with kT ln 2.

A local source skim of the opening four pages of Liao et al.'s 2026 SAM paper and Caravelli et al.'s SOMN perspective was made, but no technical result from either is used in this construction. Neither is an independent verification.

## Candidate: material-supported kinetic gates

State at time t is (x, c), where x_j in {0,1} is a conformational memory bit, and c_e >= 0 is material assigned to a two-arm scaffold e=(i,s_i;k,s_k -> j). The gate is g_e(x,u)=1[x_i=s_i]1[x_k=s_k]; an always-present ligand allows a unary gate. Inputs u are prescribed ligand pulses. The scaffold holds two binding sites close enough that their joint occupation catalyzes a reaction changing j. There is no learned additive force from i to j. Scaffolds change **how fast** a reaction channel runs. Their microscopic binding and sequestration are initially coarse-grained, an explicit physical weak link.

Energy of an active bit relative to inactive is epsilon_j > 0, fixed during learning. Fuel F and waste W have maintained chemical-potential difference Delta_mu. With beta=1/(k_B T), every gate has two distinguishable channels:

- Fuel channel 0 -> 1: k_e^+ = alpha c_e g_e.
- Reverse fuel channel 1 -> 0: k_e^- = alpha c_e g_e exp[beta(epsilon_j-Delta_mu)].
- Passive reset channel 1 -> 0: gamma_j.
- Reverse passive reset channel 0 -> 1: gamma_j exp[-beta epsilon_j].

A fuel-forward event consumes one F and yields W; its reverse synthesizes F. Hence k_e^+/k_e^- = exp[beta(Delta_mu-epsilon_j)]. The passive forward/reverse ratio is exp[-beta epsilon_j]. The two channels must remain distinct even when they have the same endpoints. Execution is a continuous-time Markov jump process with these rates, sampled by Gillespie or integrated as a master equation. No truth-table evaluation is performed by the simulator. With Delta_mu=0 the equilibrium distribution is the same product Boltzmann distribution for all scaffold assignments; kinetic structure can still affect transients. With maintained fuel the same minima support different causal programs.

### How learning actually changes the mechanism

All geometrically compatible scaffold contacts are initially weakly occupied. c_e = M softmax(theta)_e gives an exact finite material budget, with an optional small training-only exploration floor documented separately. For a trajectory, the local rate eligibility is

E_e = N_e^+ + N_e^- - integral_0^T (k_e^+ 1[x_j=0] + k_e^- 1[x_j=1]) dt.

This is the derivative of its path log-probability with respect to log c_e. A terminal scalar R supplies task information, not an edge label, state target or trajectory. With independent baseline b, h_e=(R-b)E_e, the resource-constrained update is

theta_e <- theta_e + eta (h_e - p_e sum_f h_f), p=softmax(theta).

The integral is a local no-event trace, not an absent-event oracle; omitting it would reward activity whether useful or not. A chemical implementation of this signed trace and reward broadcast is NOT established. A digital simulation can implement it exactly; that is a designed learning process on a physically consistent execution model. Material thresholds can prune effective edges after training, but continuous reallocation alone is not literal creation of new degrees of freedom. No claim of topology invention will be made merely from a small c_e.

The high-upside hypothesis is that learning can construct **persistent conditional clocks and state transitions** as reusable internal mechanisms, with geometry/material supporting them. Existing sites, gate valency and the rule remain designed. An ambitious extension would let scaffolds recruit spare conformational sites; this v0 does not contain a solved credit rule for recruitment and will not count it as accomplished.

### Worked program before learning

One memory site m and one output site z suffice for ordered A-then-B detection:

1. c_(A -> m) is large, so an A pulse spends fuel to set m.
2. gamma_m is slow enough that m persists through a silent interval.
3. c_(B AND m -> z) is large, so a later B pulse spends fuel to set z only if m is still present.
4. Reading z at the deadline reports success. B-before-A and B-only should fail. A-only must also fail; this prevents solving the task by merely recognizing recent A or B.

The four internal states 00, 10, 01, 11 are explicitly executed; A -> m and B AND m -> z are the constructed two transitions. The prescribed example establishes expressibility, not discoverability. Exact success probability depends on alpha*c, reset gamma, pulse duration and deadline. It will be computed rather than asserted.

The candidate pool for the first probe is generic for two sites: for target m, unary and pairwise positive gates from {A,B,z}; for target z, gates from {A,B,m}; no gate reads its own target. This has 12 contacts and includes feedback and misleading direct-output gates. It is an intentionally small finite discovery problem, not evidence that finding a large algorithm is easy. No intermediate m target or correct-contact supervision is supplied.

### Composition, retention and costs

A trained z can act as a catalytic ligand to a newly constructed stage. For example, add a site q and scaffold (z AND C -> q), then A–B–C produces q. The old material assignment can remain unchanged. Retention here means preservation under specified later training, and must be checked on old sequences; it is not proved by copying parameters. Composition requires independent held-out timing combinations, not replay of A–B–C used to train the new gate. A sequential circuit is only an example of computation; no universal compositional discovery result is claimed.

The coarse-grained ideal catalyst does not consume its source state, but a real source can be bound/sequestered. Its downstream load changes free ligand, delay and error. Maintaining or copying a source to drive multiple recipients requires material, time, and potentially fuel. An explicit loading test is mandatory before calling the mechanism composable.

Per-trajectory driven work is W_chem=Delta_mu * sum_e(N_e^+ - N_e^-). Heat released on fuel jumps is Delta_mu-epsilon_j for 0->1 and its negative for reverse; reset heat is +/-epsilon_j. Input clamping, state preparation/reset, scaffold synthesis/turnover, trace measurement, and controller costs are additional. lambda*sum c or a penalty on c is a numerical/resource proxy, not automatically joules. The simulator's CPU work is also not this chemical work. If fuel goes to zero, no thermodynamic free lunch is inferred from transient information in prepared inputs.

## Frozen probe criterion

Before inspecting outcomes: pulse duration 1, gap 1, read immediately after the final slot; positives A then B; matched negatives B then A, A only, B only, and empty, including B-only in the final slot. Check the hand-constructed two-contact circuit by exact finite-state integration and channel-resolved work. Then attempt terminal-reward-only material learning within a fixed small trajectory budget (at most 20,000 training trajectories per seed; initially at most four seeds). Report all seeds and evaluation separate from training. Core finite acceptance: constructed circuit balanced accuracy >=0.9 for a documented parameter setting; learning success is separately reported and not required to preserve expressibility. Adversarial tests: Delta_mu=0, contact removal, absence of memory persistence, and ideal versus explicit sequestration. No claim of scalable discovery follows even if all pass.

## Hardest unresolved step

Can low-information feedback discover useful kinetic state machines at acceptable material/trajectory cost, and can the resulting gates be reused without hidden buffering/control that does the computation? Local detailed balance and a valid score gradient do not solve either problem. The first serious prior attack will check whether this is already a standard reinforced chemical reaction network or kinetic-network design; the first probe will expose whether learning the two-transition motif already needs excessive search or task-specific scaffolding.
