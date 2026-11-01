# AE v1 — construct private kinetic instances, including their output impedance

**Judgment:** a concrete bridge is possible as a finite, costly mechanism: build an unlabelled local compartment, synthesize a fresh sample of a shared learned kinetic recipe inside it, and construct a powered output interface that presents expendable carriers to downstream binding. A boundary alone does not provide reusable kinetics through an open port. The executed repair protects E's learned source and substantially restores its output under one specified load. The constructor and output interface were supplied, **not learned**. At present this is a construction target with a strong known explanation—compartmentalized chemical circuitry plus a load driver—not a demonstrated new neural primitive.

AGENT PROPOSAL / RECONSTRUCTION. Only A and E frozen candidates were supplied to this context. Exact host model/effort is not independently instrumented here; inherited route is the currently user-authorized Ultra context, never labelled Pro. Originals remain intact. No root, food, NOW, or other branch edits.

## 1. Reconstructing the two mechanisms before the bridge

**A:** a learned rule needs separate mutable instances. Its worked memory field changes cue addresses until their kernels overlap less; this creates better-conditioned independent state modes, although Gaussian tails retain interference. The demonstrated operation is an accumulator already supplied by the designer. A's own constructive alternative is a finite local graph whose learned law can separately gate execution, local parameter changes and activation of spare material. Its missing result is acquisition of a useful constructor. The simple field has an explicit NTM-style correspondence; merely renaming addressing as a membrane adds nothing.

**E:** scaffold quantities allocate a finite material budget among conditional, fueled reaction channels. The source program is an actual stochastic sequence: A activates m, m persists, then B together with m activates z. All forward/reverse driven and passive channels are named. Terminal-feedback learning reallocates scaffolds; E's first executed training seed acquired useful amounts over 19,968 trajectories. Here the learned object is a conditional transition-rate law with a holding time. The ideal model's catalytic read of a source does not include its sequestration. Once downstream material can bind the source, its availability and timing depend on what is connected to it. Nor does a learned rate vector create fresh state carriers or a second physical copy of its catalysts.

**The actual conflict:** A separates state but has not learned a new transition law or installer; E learns a transition law but does not separate its mutable instances or provide a load-invariant interface. Putting E's catalysts behind an A-style boundary prevents unwanted mixing, yet a usable output still has to cross that boundary. If the downstream gate binds the protected z directly, the boundary has passed the problematic coupling through its intended port. Spatial isolation and output insulation are different operations.

## 2. Strong construction, with the designed scaffold exposed

### Joint state without prelabelled module slots

Use a finite homogeneous nearest-neighbor host graph G=(V,E). A site is generic material, not a named module. The state is

    q_i          local active-material occupancy;
    a_i          bounded growth/age token;
    b_ij         wall material on a host edge;
    n_i,s        counts of ordinary molecular species;
    c_i,e        counts of the twelve E scaffold types;
    xi_i,r       local eligibility traces;
    R_i          local precursor/fuel/waste counts.

Molecular species include inactive/active m and z carriers, inactive/active exported Y, an output sensor S, and source/sensor and Y/load complexes. Local recurrent activations h_i used during construction are additional finite state. No list of instances or instance ID is an input to a transition. Connected low-permeability regions are identified afterward for analysis.

The common learned recipe is p_e=softmax(theta)_e. It controls the relative synthesis propensity of the twelve scaffold types. It is **a shared recipe, not a single catalyst molecule operating everywhere**. Each region must obtain fresh physical copies. A software realization can share an immutable parameter vector; that is a different material implementation with different costs.

During a build phase a transient request ligand marks a spatial origin. It says “start an instance here,” not which answer to produce. Local growth consumes precursor; wall synthesis consumes wall material/fuel; uptake/synthesis draws scaffold types from the common recipe; inactive m/z and output carriers must be made. The task's A/B pulses begin only after the build deadline. Input delivery at an origin and a construction request are designed interfaces. They do not provide prelabelled state slots.

### An actual local instantiation witness

For a supplied integer radius R, initialize a_i=-1 on empty material and a_i=R where a request arrives. A synchronous nearest-neighbor transition is

    a_i' = max(a_i, max_(j adjacent i)(a_j-1)),
    q_i  = 1[a_i>=0],
    desired_wall_ij = 1[q_i != q_j].

Every newly occupied site consumes one generic support precursor and receives new inactive state carriers plus locally synthesized scaffold molecules. Creating or removing a wall segment is a counted event; moving a frontier is not free. Wall production/recycling reactions approach the desired wall field at finite rates. The Boolean wall in the finite witness is a digital construction idealization, not a measured impermeable physical membrane.

On an 81-site line, three requests at 8,27,52 with R=2 actually created regions [6..10], [25..29], [50..54]. The rule had no component IDs. It used 15 active sites, 18 wall-synthesis events and 12 disassembly events, leaving six wall segments. Requests at 8 and 11 **merged** into [6..13]. This preserves an important failure: without collision handling the rule cannot guarantee two instances from nearby requests. Supplying separate identifiers afterward would conceal that failure.

This witness is stronger than assuming an allocator, but weaker than learning one: its growth rule and radius were supplied. It demonstrates a feasible local route from uncommitted host material to private domains, under a separation precondition. Arbitrary collision-safe variable binding remains OPEN.

### A neural construction policy that does not already compute the answer

Replace the supplied growth/sealing rates with a local neural policy, while retaining finite reaction masks and resource stoichiometry. One explicit choice is a width-16 recurrent rule

    h_i' = tanh(W h_i + U v_i),
    log lambda_i,r = (O h_i')_r,  r=1,...,9.

v_i has 14 scalars: local request, occupancy, age, mean neighboring occupancy, mean wall density, free support precursor, free wall precursor, free scaffold precursor, inactive m precursor, inactive z precursor, inactive Y precursor, sensor precursor, local fuel fraction, and local measured load-capacity demand. The nine rate outputs control growth attempts, wall synthesis/removal, scaffold synthesis, m/z carrier synthesis, sensor synthesis, Y precursor synthesis and Y reset capacity. Masks suppress unavailable reactions and make conservation explicit. Counts of molecules and neural arithmetic are separate costs.

Crucially this policy does **not** observe the subsequent A/B sequence, m/z task state or task label, and it cannot directly emit an answer bit. Load and start positions must be randomized independently of task class. It chooses an installation from request/resources/load; the learned E kinetics subsequently perform the computation. This is a substantial scaffold, including a build/execute protocol, a fixed chemical alphabet and source/output roles. It is not an unrestricted solution to discovering all neural mechanisms.

Runtime transport across edge ij for species s has finite propensity

    D_s * n_i,s * exp(-kappa_s b_ij).

Thus walls protect internal state and local scaffold/update material while chosen input/output species have separately designed transport or membrane transduction. At finite b, isolation is approximate. A union bound on expected cross-wall events over T is

    P(any leakage) <= T * sum_(directed boundary ij,s) D_s n_i,s exp(-kappa_s b_ij),

when n_i,s is bounded over the interval. More reliable isolation costs wall material or lower transport and can slow assembly. The supplied witness does not verify a physical wall achieving a particular b.

### Executing the inherited transition law inside a new region

After synthesis, E's reactions use the local scaffold concentrations c_e and local m/z. With fixed energy epsilon=8 kT, fuel difference mu=16 kT, alpha=4 and passive reset gamma=.03, the driven channel rates are

    a_e^+ = alpha c_e g_e(x,u),
    a_e^- = alpha c_e g_e(x,u) exp(epsilon-mu).

Passive reset and its reverse remain present. Internal catalyst binding is still coarse-grained as in E; the new verifier explicitly expands source/output binding only. New regions start with m,z inactive; runtime state is not copied from an old region. Independent regions sample their own jumps and retain their own state. The common recipe may remain fixed even if one instance changes its private state or private scaffold patch.

If all cross-region channels are removed and resource concentrations are fixed, the joint generator is the sum of the local generators. Product initial distributions then remain products conditional on inputs and recipes. This is the precise ideal independence statement. Finite permeability, a shared depleted precursor/fuel pool, common input fluctuations and sampled recipe differences all violate its premises. None is silently assumed absent from physical cost accounting.

## 3. The necessary repair at the output boundary

Keep the task's z protected. Add a small sensor pool S and an abundant, separately manufactured output carrier Y:

    Z_active + S <-> C,
    C + Y0 + F_y <-> C + Y1 + W_y,
    Y1 <-> Y0,
    Y1 + L <-> D.

Z is actually sequestered while in C. C catalyzes output conversion without being consumed. The downstream load binds Y1, not Z. All driven forward/reverse rates and both passive directions are retained. The sensor can still perturb the source, and the carrier pool can still be overwhelmed; there is no ideal one-way wire.

For the finite verifier S_total=.01, sensor association=100, dissociation=10,000; Y_total=100; load association=10, dissociation=1. The output reaction uses epsilon_y=16, mu_y=32, passive reset G, and

    kcat = G*(koff_sensor/kon_sensor)/(S_total*Y_total).

At G=50, kcat=5,000. Low sensor occupancy reduces input loading; fast conversion/reset and a large finite Y reserve support an output near the source signal. At small G the carrier fails to track it. These choices were supplied before the probe. They are **not** learned by AE.

This is not a novel insulation circuit: [Del Vecchio et al. (2008)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2267736/) already requires fast conversion, sufficient carrier material and low input loading, and [Mishra et al. (2014)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4262674/) gives a load-driver implementation. The new candidate question would be whether a network can acquire such an interface together with the act of installing a private kinetic instance. The interface's existence is established prior art.

## 4. Learning, inheritance and the copying bill

### A changed adaptation law, not a hidden copy operator

Once catalyst numbers are drawn during construction, a physical instance does not directly reference theta. Let M scaffold synthesis events select types with probabilities p=softmax(theta); then N~Multinomial(M,p), and c_e=4 N_e/M at the chosen concentration scale. For terminal scalar reward R, the exact recipe sampling score is

    d log P(N|theta)/d theta_e = N_e - M p_e,
    theta_e <- theta_e + eta*(R-baseline)*(N_e-M p_e).

It would be wrong to reuse E's continuous-rate gradient while pretending theta still directly sets already assembled stochastic counts. For local neural construction propensities lambda_r(psi), the likelihood-ratio update instead uses event-log-rate derivatives minus integrated propensity derivatives, multiplied by the same delayed reward. A declared objective can be

    R = mean_instance task_score
        - lambda_material*consumed_precursors
        - lambda_work*channel_resolved_chemical_work
        - lambda_time*build_and_execute_time
        - lambda_leak*specified_cross-instance_damage.

This is a defined stochastic training problem with finite scores, not a demonstrated good optimizer. The physical implementation of signed traces, a reward broadcast and neural-to-synthesis control is not established. A digital trainer can compute them, with its own memory/compute cost.

An optional local adaptation phase can retain private scaffold adjustments and eligibility traces inside each region; its material must not flow across the wall. Updating the common recipe changes future production; it does not instantaneously change every existing catalyst molecule. Existing instances keep old material until replacement. Shared ancestral law and private local adaptation must therefore be distinguished. No local adaptation was run in AE.

### Sharing a recipe has finite fidelity

The relative standard deviation of contact e in a sampled copy is

    sqrt((1-p_e)/(M p_e)).

The executed 64-copy sample at each of M=100,1,000,10,000 gave dominant-contact relative standard deviations about 10–11%, 3.4–3.8%, and 1.05–1.10%, respectively. Mean L1 error of the entire concentration recipe was .46849, .16554 and .05156. Even at M=10,000 the least frequent contact has a .10192 chance of being absent. Those small contacts may be unimportant, but no functional pruning test was run; their harmlessness is not assumed.

At fixed concentration, increasing M also increases volume and the number of state molecules needed at a fixed state concentration. It is not a free precision increase in the same material volume. Exact equality of all rates is not obtained by a finite random draw. A meaningful reuse claim needs functional tolerances, not a claim of literal perfect copying.

For a fixed-duration finite-state program, a coarse Duhamel bound gives TV error <= T*alpha*||c-c_hat||_1 when all channel gates are bounded by one and reverse multipliers are <=1. This often becomes vacuous at small M; it does not turn the sample variance into a performance guarantee. AE does not report successful task execution for sampled copies.

### What persists, what is learned, what was supplied

- **Inherited from the already executed E result:** seed 1301's 12-contact recipe, acquired with 19,968 training trajectories; its reported driven training work is 463,488 kT in E's concentration/unit convention. Two dominant amounts are A->m=2.05728 and B+m->z=1.72978; the other contacts are retained.
- **Shared design:** species alphabet, conditional scaffold valency, rate/energy conventions, host locality, request/input protocol, output role and neural policy architecture.
- **New supplied witness:** local growth law/R=2, wall construction idealization, template sampling, sensor/carrier interface and its numerical parameters.
- **Actually learned in AE:** nothing. AE reused E's learned program and tested a proposed installation/interface scaffold.
- **Potentially learned in the full hypothesis:** local growth/sealing/material allocation and carrier turnover from terminal task/resource feedback, while recipe theta and policy psi persist between training episodes. This is optimizer-mediated retention, not demonstrated evolutionary heredity.

## 5. Executed worked computation and costs

Frozen pre-execution contract: `probes/AE/contract-v1.md`. Script: `probes/AE/probe_v1.py`; output: `results_v1.json`; enforced environment: `execution_v1.log`. The six source states are m in {0,1} times z in {inactive,active-free,active-bound}; two finite output-pool variables and two integrated work counters are coupled by mass action. It is a deterministic population/master-equation coupling, **not** an exact single-molecule joint distribution for sensor/carrier pools.

Same E recipe, pulse schedule A for one unit, silence for one, B for one:

| Installation | Final free port signal | Additional buffer fuel work |
| --- | ---: | ---: |
| Isolated learned source | .957524 | 0 |
| Perfect compartment, direct output, load 10 | .010595 | 0 |
| Buffered, G=1, load 10 | .018706 | 26.887 kT |
| Buffered, G=10, load 10 | .310034 | 260.565 kT |
| Buffered, G=50, load 10 | .848397 | 1,251.350 kT |
| G=50 buffer, drive removed, load 10 | .0000113 | 0 |

The source alone spends 32.133 kT on its driven program in the isolated positive trial; with the G=50 buffer it spends 32.133 kT plus the listed buffer work. Extra fast chemistry is doing real work. The loaded G=50 analog balanced terminal score over the pre-existing five cases is .920182, versus .505188 for direct load and .971822 for the isolated source. These analog scores are not a newly verified stochastic Boolean success rate.

The frozen central criterion passed: across all five cases/times, average absolute free-port error fell from .0607960 to .0200826 (67.0% reduction, better than the required factor two); maximum source total-active trajectory error was 4.703e-7, below .02. The positive-trial port error by itself remains substantial: mean absolute error .087925, and terminal deficit .109127. A threefold average improvement is not exact reuse or load invariance. The G=1 result demonstrates why source protection alone is insufficient.

Tightening solver tolerance changed the positive G=50 port trajectory by at most 8.08e-9 and source trajectory by 1.09e-8. Probe execution took 1.405 seconds internally. Network access was denied outside the simulation; environment contained no inherited credentials; writes were restricted to probes/AE; CPU and wall limits were enforced. No numerical repair or second training run was needed.

**Material/energy units:** .01 sensor and 100 carriers are concentrations relative to a source population of concentration one. They are not fractional physical molecules. For example, with 10,000 molecules per concentration unit, a compartment contains 100 sensor molecules and one million carrier molecules, plus 10,000 m and 10,000 z carriers and approximately 40,000 scaffold molecules at concentration four. The reported concentration-unit work scales by that molecule number. This is expensive compared with a two-bit software state.

Wall synthesis, scaffold/carrier synthesis, input clamping, initial preparation/reset, replacement of depleted fuel, physical trace/neural training implementation and host arithmetic are additional costs. If wall assembly consumes one driven token per synthesis event at chemical potential mu_b, the three-region witness needs 18*mu_b gross wall-assembly work before accounting for a specified recycling scheme; there are 12 wall-removal events and no assumed perfect recovery. No complete physical first-law certificate for all construction channels is claimed.

## 6. Strongest attack, what changed, and the exact missing step

**Closest simpler explanation:** a conventional compiler/assembler installs several copies of a learned chemical circuit in dynamically grown compartments, with a standard load driver at each output. A dynamic graph network could encode the same local construction/update messages. The present result establishes neither a new reachable class of computations nor an acquisition advantage over those mechanisms. It should be rejected as a novelty claim if it rests only on new names for their parts.

**What the repair genuinely changed:** the unit being instantiated is now not just a rate vector or a memory slot. It includes new state carriers, paid scaffold synthesis with fidelity limits, local boundary creation, and an output interface whose gain/material/work can be tested. This exposes a consequential joint learning target: learning a transition mechanism is insufficient if installation does not preserve its functional timecourse under the loads that its own reuse creates. A recipe's useful instantiation includes its permissible load and reset regime.

**Strongest missing step:** no network has acquired the construction policy, collision behavior, physical barrier selectivity and export capacity from task feedback while also acquiring the kinetic law, under the same finite material/training budget as a supplied compartment/allocator plus standard load-driver baseline. In particular, our fixed policy never had to discover that active insulation is needed; we designed it after the source reading. Training a differentiable or score-based installer is an explicit possible route, not evidence that this key step works. Exact independence is additionally unattainable under arbitrary finite shared-resource perturbations; the claim must remain bounded by specified tolerance and load.

**What could reverse the favorable finite result:** explicit stochastic low-copy simulation may reveal sensor/carrier noise; physical gate binding may add further sequestration; membrane leakage/collision may destroy instance independence; or the required speed/material/fuel may exceed a matched conventional implementation. The current probe includes source/output binding but leaves E's internal product gates coarse-grained and assumes controlled input placement.

## Certificate

CLAIM | STATUS | EVIDENCE / LIMIT

- A supplied local rule can grow three spatially separate unlabelled compartments from generic sites. | EXECUTED OBSERVATION | 81-site witness; nearby requests merge; no learned allocator.
- Sampling a shared catalyst recipe provides approximate law inheritance at a material/fidelity cost. | VERIFIED sampling calculation | 192 recipe draws; no task-level fidelity result.
- A passive scope boundary alone makes an exposed kinetic output reusable under load. | FALSIFIED in the specified model | direct loaded port .010595 instead of .957524.
- An explicit finite-rate, powered output interface can repair much of that output loss while preserving the source. | EXECUTED OBSERVATION | frozen error criterion passed; high added work/material; known insulation explanation.
- A neural system has learned to construct these reusable instances without prelabelled slots. | OPEN / NOT EXECUTED | explicit state, event grammar and score updates only.
- AE establishes a novel neural computational primitive. | NOT SUPPORTED | closest explanation is compartmentalized CRN plus load driver.

NEXT SINGLE DECISION: decide whether **learning installation together with a mechanism's finite-load contract** is consequential enough to retain as one focused acquisition question. This certificate does not authorize further training. No new toy task is required to preserve the useful negative result that a boundary and a reusable kinetic interface are different objects.
