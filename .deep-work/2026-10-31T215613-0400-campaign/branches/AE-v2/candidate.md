# AE-v2 — delayed-feedback acquisition of a finite installer

**Judgment: partial acquisition is demonstrated in this finite model; load-sensitive interface acquisition is not.** A previously random neural policy learned to assemble state carriers, fresh copies of E's recipe, sensor/carrier stock, turnover capacity and a connector from an empty supplied compartment. It never observed the input sequence or answer label. All three saved neural policies meet the frozen acquisition criterion. However, they almost always build the expensive driver even when load is zero, and removing their load observation scarcely changes performance. The strongest explanation is ordinary reinforcement learning of a mostly fixed assembly priority under resource masks. The ambitious adaptive-installer claim remains OPEN.

AGENT PROPOSAL / VERIFIED finite observation. This is an explicitly authorized follow-on repair to AE-v1, not a fresh independent invention. Current inherited Ultra route; no verified Pro/model-effort claim. No independent-rebuild materials were read.

## Exact question and connection to the research question

Can delayed functional feedback teach a neural controller **which pieces of a reusable stateful computation to install**, rather than supplying an already complete compartment and buffer? This bears directly on constructing a mechanism instead of only adjusting its current state. AE-v1 left every installation decision supplied. AE-v2 supplies the component alphabet and assembly physics but starts with zero installed functional components; the new learned object is a finite assembly policy.

The narrowing is substantial. The private boundary, request/execute protocol, E source recipe and load-driver chemistry remain supplied. The controller is an external digital trainer/installer for the modeled subsystem. No boundary placement, new chemistry, arbitrary neural topology, simultaneous multiple-instance reuse, online self-modification, retention after later learning, or evolutionary heredity is established. This is a partial acquisition bridge, not the full Fundamental Question solved.

## Executable construction

Assembly state is

    q = (m,z,r,s,y,k,p),
    m,z in {0,1}; r,s in {0,1,2}; y,k in {0,...,4};
    p in {none,direct,driver}; q_initial = 0.

A chosen action adds an m carrier, z carrier, fresh recipe amount, .01 sensor, 25 output carriers, .05 turnover catalyst, direct connector, or driver connector; the ninth action is STOP. A connector can be selected once. At most 15 actual assembly actions are executed. Generic precursor consumption for these eight actions is respectively 1,1,4,.01,25,.05,.1,.1 concentration units. No mask requires a useful combination: the policy can waste material, omit memory, stop early or install a connector without a working circuit. Masks enforce only resource availability and component maxima.

These are designed species roles and assembly primitives, including two designed wiring choices. Neither the favorable final count vector nor an imperative “build a buffer now” macro is supplied to the policy. Nevertheless the library itself already contains an insulation circuit; learning chooses and provisions it rather than inventing its chemistry. Concentration-based synthesis assumes the inherited recipe is copied at its mean proportions; finite-copy fidelity and fabrication are not reverified here.

The width-16 policy has 329 parameters:

    x = (1, log(1+load)/log(11), budget/115,
         remaining_budget/115, q/component_maxima),
    h = tanh(W x),
    pi(a|q,load,budget) = masked_softmax(V [1;h]).

The neural controller receives **no A/B pulse, task label, m/z reaction state, correct action target, installation oracle, or evaluator derivative**. It finishes construction before any task pulse. A matched linear policy uses the same x and action masks, with 99 parameters.

After construction, the installed material executes the actual AE source/sensor/carrier/load model. Its six source population states are m in {0,1} and z in {inactive,active-free,active-bound}. E's twelve scaffold concentrations are r*C from inherited seed 1301. Missing m/z carriers suppress their transitions. Every fueled source contact has its E forward/reverse rates and passive reverse/reset channels.

The direct connector exposes z to reversible load binding. The driver connector instead lets installed sensor S bind z; the bound sensor catalyzes Y0/Y1 conversion, and downstream load binds Y1. Installed pools are S=.01s and Ytotal=25y. With k turnover units, kcat=1250k and passive turnover G=12.5k. Association/dissociation constants remain sensor 100/10000 and load 10/1; carrier epsilon=16 and fuel chemical-potential difference=32. Finite source sequestration, finite carrier stock, reverse conversion and passive reverse turnover remain present. No unperturbable numerical source signal substitutes for the source reaction system.

The simulator couples a finite-state source master equation to continuous, finite mean-field pools; it is not an exact microscopic molecular simulation. See the frozen `probes/AE-v2/contract.md` and literal `acquire.py` for the state/action/reaction law.

## Delayed feedback and finite learning budget

Each installation is assayed on AE/E's five fixed sequences: AB, BA, A-only, B-only and empty. Each has three unit slots; all task state starts reset. Let f_j be the clipped free terminal port signal and t_j its answer. The analog task score is

    A = sum_j w_j [t_j f_j + (1-t_j)(1-f_j)],
    w = (.5,.125,.125,.125,.125),
    R = A - .0002*consumed_precursor
          - .0005*assembly_actions
          - 1e-5*weighted_source_and_buffer_work.

Only this terminal scalar is returned. Evaluating all five cases makes feedback unusually low variance and more informative than one sampled Boolean trial. It is an explicit acquisition scaffold, not an unreported capability of the learning rule.

For action sequence a_1...a_T, the update uses the likelihood-ratio score

    g = (R-b_context) sum_t grad log pi(a_t|q_t,context),

with a baseline from earlier episodes of the same context, then Adam learning rate .008. No reward gradient flows through the chemical solver. The source recipe is fixed during this new learning.

Three seeds, 631/632/633, receive 2,000 installations each. Training contexts are load {0,1,10} crossed with precursor {35,65,115}. Held-out contexts are load {.5,4,7} crossed with precursor {50,90}; evaluation makes no weight changes and samples 64 installations per context. One learning attempt was executed; there was no search over learning rates, seeds or checkpoints.

## Results, including the decisive attack

The first process saved all six final policy files and the reference table, then failed at final JSON serialization on a numpy Boolean. It did not save its in-memory training curves or training cost totals. The failure log is retained. A separately authorized **evaluation-only recovery** loaded those frozen weights, repeated the predeclared sanity checks and evaluated a newly fixed replay stream. No training or tuning was repeated. The numbers below are recovered frozen-policy evaluations, not recovered original evaluation draws.

| Frozen policy/reference | Held-out analog task score | Held-out penalized reward | Material | Weighted execution work |
|---|---:|---:|---:|---:|
| Neural, mean of all 3 seeds | .839135 | .812909 | 60.3455 | 766.550 |
| Matched linear policy, all 3 | .835684 | .804468 | 60.2638 | 1,272.256 |
| Matched-query ordinary random allocator, all 3 | .741567 | .726402 | 38.2094 | 263.425 |
| Exhaustive ordinary allocator | .852864 | .835023 | 52.7083 | 254.963 |

Material and work use the model's concentration convention; material is a resource proxy and work is chemical driving work, not controller energy or gross five-assay cost. No neural superiority is claimed from these small differences.

Neural training-context reward increases over each seed's initial random policy are **.373215, .346012 and .318255**. Held-out task scores are .839453, .838498 and .839453. The frozen criterion required every seed's training reward improvement >=.05 and mean held-out task score >=.75; it passes.

**ATTACK:** this success does not establish adaptive insulation. Removing the load observation reduces mean neural held-out reward only from .812909 to .811961; this small sampled difference is not decisive evidence of load use. Modal final configurations are identical across physical loads at each budget. All neural policies almost always select the driver. For example, at precursor 35 the common installation is (1,1,1,2,1,4,driver); at precursor 115 it is (1,1,2,2,4,4,driver). Changing stock with budget is substantially explained by supplied feasibility masks.

The ordinary exhaustive allocator makes a causally meaningful different decision: at **zero load**, every budget's optimum is the cheap direct installation (1,1,1,0,0,0,direct), material 6.1, weighted work 20.2823, reward .968399. The neural policies instead obtain about .71379, .92827 and .92633 at budgets 35/65/115. Thus the model contains a valuable load-dependent installation distinction, but the learner has not acquired it. This is a concrete failure, not merely a disclaimer about future scale.

## Verification and its limits

The vectorized solver uses exact source, reversible-binding and carrier-conversion subflows with step .02. Eight configurations/cases were fixed in advance and compared with a separately integrated full scipy BDF right-hand side, including positive/order-reversed trials, no load, strong load and absent source components. Recovered maximum terminal-port discrepancy is .00153126; maximum checked case-reward discrepancy bound .00160141; maximum step-halving port change .00124449. Source probability mass error in the checked batch is 7.11e-15; minimum state is -7.91e-20. The analytic neural log-score derivative agrees with finite differences within 9.02e-11.

These checks pass their frozen tolerances. The BDF RHS shares the declared source generator, so this is an integration and gradient check, not an independently invented model. It is not a uniform error bound for every tabulated installation. Its error scale and three-seed scope do not support a fine neural-versus-linear advantage claim.

## Complete cost boundary

- **Inherited acquisition:** E used 19,968 training trajectories and reported 463,488 kT driven source work. AE-v2 did not reacquire that recipe.
- **New learning, static code accounting:** 6 policies x 2,000 installations = 12,000 delayed-feedback queries, each logically using five assays, or 60,000 assay evaluations. Neural and linear methods each receive 6,000 installations. The ordinary allocator has 2,000 queries/seed, 6,000 total, with no functional feedback from held-out contexts.
- **Reference/oracle cost:** the evaluator precomputed 2,700 terminal configurations x 15 contexts x 5 sequences = 202,500 reaction-model assays. Policies access only scalar queries; no argmax, target vector or gradient enters training. Caching makes subsequent queries cheap computationally. This extra table/oracle cost must not disappear from an acquisition efficiency claim. The exhaustive reference has access to the complete table and is not a matched-search-budget learner.
- **Evaluation overhead:** each set of six policies has 960 sampled installations before training, 960 after training and 960 with load hidden, per policy: 17,280 installation evaluations in the original code and again as evaluation-only recovery. These are cached logical assays, not fresh training. Numerical verification additionally repeats the eight BDF checks and specified step checks.
- **Missing measurements:** first-run training curves, unique-query counts, summed training material and chemical-work traces were not written before the serialization error and are unavailable. Weighted work in the table is an expected execution cost under the case weights, not the gross cost of physically running all five assays. Neither that gross acquisition work nor digital controller energy was measured. The first execution's internal duration was also not saved; recovery took 7.23756 seconds. Both processes had enforced 120-second wall limits and 110/115-second CPU limits.
- **Supplied/unpriced physical scaffolds:** private shell and assembly apparatus, state preparation/reset, input clamping, template fidelity, synthesis and fuel regeneration, reward measurement/broadcast, policy arithmetic/optimizer memory and synthesis stoichiometry. The assembly mass model treats components in common concentration resource units; it is not a fabrication or thermodynamic first-law certificate. Neural policy arithmetic is external digital computation, not free chemistry.

The numerical processes used the existing runner with denied network and a scrubbed environment. Their only scripted input files were the inherited E result and saved AE-v2 artifacts. The runner permits writes within the campaign; inspected code writes only probes/AE-v2. Source acquisition was separate. No root/food edits, purchases, children, new service or external release occurred.

## Prior-art boundary and certificate

The interface is inherited known load-driver chemistry. The optimizer is established neural combinatorial optimization: the checked source is [Bello et al., §4 and §6](https://arxiv.org/html/1611.09940v3). `sources/AE-v2/prior-boundary.md` records the exact reading boundary. The result is a finite application and acquisition observation, not a novelty or hardware claim.

CLAIM | STATUS | EVIDENCE

- A neural controller can acquire a usable finite installation from blank functional material using delayed scalar feedback in this supplied alphabet. | VERIFIED in this model and frozen criterion | all 3 saved weights; `results_recovered.json`; no task-label input.
- The policy learns when load warrants insulation and economizes at low load. | NOT ESTABLISHED; the decisive zero-load comparison fails | load-hidden replay; modal installs; exhaustive cheap direct baseline.
- The controller discovers the load-driver chemistry, learns a private boundary or outperforms ordinary allocation. | NOT CLAIMED | these primitives are supplied; linear policy performs similarly and exhaustive ordinary allocation is better.
- Full acquisition costs or detailed training dynamics were measured successfully. | FALSE | serialization failure; missing training traces explicitly retained.

WEAKEST REMAINING LINK: whether a learned installer can acquire the valuable **conditional** use of structure instead of a nearly fixed resource-filling recipe, once complete acquisition costs and stronger physical constraints are enforced.

FAILED BRANCHES: final JSON serialization; interpreting variable-budget count changes as learned load sensitivity; treating a predefined load-driver action library as invention of insulation.

NEXT SINGLE DECISION: whether this restricted installation question is scientifically valuable enough to retain. The finite authorized objective is closed; no further training is implied.
