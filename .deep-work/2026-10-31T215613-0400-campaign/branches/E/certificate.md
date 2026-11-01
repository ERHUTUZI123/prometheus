# Branch E first-stage certificate

**Judgment:** A fuel-driven network can learn a small stateful internal program by reallocating catalyst/scaffold material using terminal reward alone. In this finite model the acquisition is real and independently checked. **Composition is not free:** explicitly limiting the availability of a shared catalytic source causes a severe fanout failure. The useful research opportunity is load-aware construction and reuse of kinetic mechanisms. The broad ingredients and the score learning rule have close precedents; this is not a novelty or breakthrough certificate.

## CLAIM | STATUS

1. **Conditional kinetic gates can express an ordered two-event program with fixed state-energy minima. VERIFIED within the stated four-state CTMC.** A memory site is set by A; a second scaffold sets output only when B and memory coincide. Removing either learned transition destroys performance. This connects to the Fundamental Question by changing an internal computational primitive from an instantaneous weighted input to a learned conditional transition with holding time, fuel use and reset.
2. **Scalar terminal reward can acquire the two dominant transitions in a generic 12-contact pool. EXECUTED OBSERVATION in four predetermined seeds, not a scaling claim.** No intermediate memory targets, correct-edge labels, exact gradient or master-equation values enter training.
3. **Material-supported gates compose without meaningful additional resource/control costs. FALSIFIED for the explicit finite-source loading model.** Catalytic nonconsumption does not imply simultaneous availability to multiple downstream mechanisms.
4. **This is a new learning principle or an established route to autonomous, open-ended construction. NOT ESTABLISHED.** The local score is a known likelihood-ratio estimator. Catalyst reinforcement and conserved material pools have specific prior examples. Creation of new state variables, physically autonomous trace learning, and retention under later learning remain open.

## Mechanism, execution and information boundary

State consists of binary conformational sites x and material c on candidate scaffold contacts. A two-arm scaffold gates target j only when its two ligands have the specified conformations; a constant ligand permits a unary gate. In the executed probe the sources are A, B, memory m and output z. Each target admits unary and pairwise positive gates from all other sources, including feedback. This gives 12 designed possible contacts. Material is continuous and constrained to sum c=4. Thus the experiment learns an effective kinetic circuit on a designed primitive pool; it does not demonstrate fabrication of new chemistry or creation of new degrees of freedom.

For a gate with activation g, the distinguishable fuel channels are

    0 -> 1: alpha*c*g
    1 -> 0: alpha*c*g*exp[(epsilon-Delta_mu)/(k_B*T)]

and the distinguishable passive channels are

    1 -> 0: gamma
    0 -> 1: gamma*exp[-epsilon/(k_B*T)].

The fuel-channel ratio is exp[(Delta_mu-epsilon)/(k_B*T)], while passive reset has the energy-only ratio. A forward fuel event consumes F and produces W; its reverse synthesizes F. Rates, channel identity and every state transition are explicit. There is no hidden truth-table interpreter. The probe uses epsilon=8 kT, Delta_mu=16 kT, alpha=4 and gamma=0.03, with unit pulse duration and unit silent gap. Inputs and deadline are externally specified. Each episode is prepared at m=z=0; that preparation is a real omitted physical cost, not an autonomous reset result.

Learning stores theta with c=4*softmax(theta). Each contact accumulates its actual forward-plus-reverse fuel-event count minus its integrated fuel-channel hazard. The scalar correctness reward multiplies this local trace; the softmax Jacobian enforces the material budget. The learner uses a prior-episode running reward baseline, batch size 64 and fixed eta=1.5. This is a digital controller implementing a specified learning rule on a thermodynamically consistent execution model. A chemical implementation of the signed dwell/event trace, resource transfers and baseline is not supplied. That controller's memory and work must not be attributed to the execution substrate.

The training distribution is balanced by class: 1/2 A–B positive and 1/8 each B–A, A-only, late B-only, and empty. Thus recency of B alone cannot solve the task. All timing, class balance, site types and primitive valency are designed task/representation information. The only online task-dependent feedback to the update is one terminal correctness scalar.

## EVIDENCE / ARTIFACT

`probes/E/results.json` preserves all cases, aggregate work/heat, state probabilities, gradients, material allocations, checkpoints, seeds and loading outcomes. `probes/E/channel_counts.json` additionally preserves exact expected counts for every individual forward/reverse fuel channel and passive-reset direction, for all five test cases of the hand circuit and each learned seed. Its independent work/heat reconstruction differs from the earlier totals by at most 3.20e-14 kT. `probes/E/run.log` preserves the original run output. No failed seed was removed and no hyperparameter search or repair was performed.

| Seed | Training trajectories | Initial exact balanced accuracy | Final exact balanced accuracy | Dominant material: A -> m / B AND m -> z |
|---|---:|---:|---:|---:|
| 1301 | 19,968 | 0.64649 | 0.97182 | 2.05728 / 1.72978 |
| 1302 | 19,968 | 0.64659 | 0.97207 | 1.65160 / 2.05861 |
| 1303 | 19,968 | 0.64523 | 0.96451 | 1.89991 / 1.67146 |
| 1304 | 19,968 | 0.64564 | 0.97383 | 1.79527 / 2.01616 |

The hand circuit has exact balanced accuracy 0.97900 and positive-sequence output probability 0.95804. Removing A -> m gives 0.50001; removing B AND m -> z gives 0.50000. Setting fuel affinity to zero gives 0.50000. Raising reset gamma from 0.03 to 3 gives 0.50257; that intervention speeds both sites' reset and therefore attacks persistence generally rather than isolating memory alone.

The independent verifier integrates the four-state master equation with matrix exponentials. A separate 6,000-trajectory check agrees with exact reward probability and a central finite-difference gradient: maximum coordinate discrepancy 1.73 estimated standard errors. For zero fuel affinity, the designed Boltzmann distribution is stationary for arbitrary material allocations, with maximum tested residual 3.11e-15. In the channel-resolved hand circuit, the first-law residual W-Q-Delta E is below 1e-14 kT. These are numerical checks of this model, not wet-lab measurements or mathematical proofs of physical realizability.

### Cost and composition boundary

The hand positive sequence uses expected driven chemical work **32.07961 kT**, releases **16.91427 kT** as modeled heat and leaves **15.16535 kT** in active internal sites. Negative A-only and B–A cases still spend about 16.35 kT because memory is set. Zero chemical work is not zero total cost: input preparation, clamping, final read/reset, scaffold construction/turnover, and learning-controller work are outside this accounting. The full four-seed acquisition budget was 79,872 training trajectories and 123,844 executed training jumps, plus 6,000 verification trajectories. For seeds 1301/1302/1303/1304 respectively, net training fuel consumption was 28,968 / 28,236 / 30,670 / 28,674 and modeled chemical training work was 463,488 / 451,776 / 490,720 / 458,784 kT. The JSON preserves these costs, which exclude between-episode preparation and the controller. Measured training-loop times were 0.35–0.38 seconds per seed in the supplied local environment; source reading, verification and this report are not included in those measurements.

A separate ideal-model composition check freezes each learned old circuit and explicitly supplies one new site q and the contact z AND C -> q. It adds two material units, a 50% increase; it does not learn that new contact. Across nine held-out pairs of A–B and B–C gaps in {0.5,1,1.5}, the four seeds' mean exact balanced accuracies are 0.95257, 0.95287, 0.94422 and 0.95471. Their worst timing accuracies are 0.93869, 0.93900, 0.93024 and 0.94081. This demonstrates supplied composition's expressibility under an ideal catalyst assumption. No back-action is built into that model; unchanged old behavior is consequently an assumption of execution, not independent retention evidence. No later-learning retention test was run.

The explicit loading challenge replaces a free ideal source by **one scarce active memory site that can bind one downstream scaffold at a time**. Binding/unbinding rates are kon=50 and koff=5; a bound scaffold catalyzes at kcat=8. The same total downstream material, 2, is split across recipients. This is a finite binding kinetic model, not a fully closed microscopic thermodynamic model; forced release on source reset omits binding free-energy work.

| Recipients | Per-recipient positive output probability | Probability all recipients succeed |
|---|---:|---:|
| 1 | 0.95720 | 0.95720 |
| 2 | 0.81405 | 0.66824 |
| 4 | 0.55337 | 0.05732 |

The loading result is the decisive adversarial evidence. Restoring reliable fanout would require more source copies, faster catalytic turnover, serial time, buffering, or another mechanism whose learning/material/energy costs must be counted. Parameter freezing alone cannot establish reusable physical computation.

## Closest precedent and changed assumptions

Substantive entry passages and original-source checks, with exact pages/equations and limitations, are preserved in `sources/E/reading_and_prior_attack.md`. Important original references are:

- [Warren & Allen 2012](https://arxiv.org/abs/1202.4704), PDF pp. 2–4, equations 4 and 6–11: the exact event-minus-hazard estimator and its variance problem. This defeats novelty of our credit formula.
- [Sartori & Pigolotti 2013](https://arxiv.org/abs/1209.4566), PDF pp. 1–5, equations 1–6 and Fig. 2: kinetic versus energetic discrimination. This makes finite-time stopping/reset a substantive resource.
- [Fil & Chu 2020](https://arxiv.org/abs/2009.13207), PDF pp. 2–6, Table 1 and section 2.4: autonomous chemical Hebbian learning and resource depletion when nonlinear binding withdraws signal molecules. This made the loading attack obligatory.
- [Arredondo & Lakin 2022](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010676), PDF pp. 4–7, 11, 14, 16–17: reinforcement of conserved catalyst weights and an explicit external-environment computation boundary. This defeats the broad combination's novelty claim.

## WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION

**Weakest link:** learning reallocates ideal catalytic material, but physically instantiating conditional gates and feeding a downstream mechanism can change the effective function. A fully autonomous signed trace is an additional unverified component. The twelve-contact pool and two-arm gate prior make the finite discovery problem unusually favorable.

**Failed claims/branches preserved:** broad mechanism novelty; cost-free catalytic fanout; usefulness without fuel or persistence; inference that hand-supplied downstream composition is discovered composition; inference that parameter freezing is retention under new learning. There were no discarded empirical seed failures and no numerical repairs.

**Next single decision:** whether to invest in one load-aware instantiation/reconstruction that matches source copies, binding-site number, total scaffold material, time, and fuel, and asks if a learned gate remains a useful reusable internal mechanism. This certificate ends the present bounded attempt; it does not authorize that follow-up or claim work will continue.

No NOW, original source PDF or other branch was modified. Mode provenance remains inherited current user-authorized Ultra; exact host model/effort identifiers were not independently exposed to this branch, and no Pro claim is made.
