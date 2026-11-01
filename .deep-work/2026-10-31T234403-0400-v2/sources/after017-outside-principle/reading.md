# after017 outside-principle — primary reading ledger

## Provenance and bounds

Fresh reserve subtask in the existing V2 run; instruction_only separation on shared filesystem. Before candidate-v0 freeze I read only effective AGENTS.md, README.md, deep-work/WORKFLOW.md, and external primary-source material below. No current library index, NOW, candidates, reports, or reviews entered. The original theory was independently chosen; it was not supplied by the coordinator. No experiments or training. Only this new ledger and its paired new candidate-v0.md were written.

This ledger records substantive original text entered through the web tool, not a claim of cover-to-cover reading. Browser extraction is sufficient for the equations used below; some mathematical glyphs in the state-map section were corrupted. I therefore did not rely on the exact extracted minimal-state formula as a verified derivation. The candidate's finite-matrix proofs are independently written and narrower than the source's differential-system theory.

## Entered original work

Jan C. Willems, **The Behavioral Approach to Open and Interconnected Systems**, IEEE Control Systems Magazine, December 2007, pp. 46–99. DOI 10.1109/MCS.2007.906923. Author-hosted original PDF:

<https://homes.esat.kuleuven.be/~sistawww/smc/jwillems/Articles/JournalArticles/2007.1.pdf>

Actual substantive passages entered:

- PDF pp. 1–4 / printed pp. 46–49: motivation, variable sharing, and tearing/zooming/linking, including the terminal-variable example.
- PDF p. 25 / printed p. 70: state as a special latent variable, characterized by legal trajectory concatenation at matching state.
- PDF pp. 28–29 / printed pp. 73–74: construction of a minimal state map and input selector from a kernel law; shift-and-cut on polynomial matrices. Read the prose and principal definitions, not a clean visual verification of every formula.
- PDF pp. 42–44 / printed pp. 87–89: Theorem 2 and the sidebar on the fundamental principle and elimination, including the role of the function space and left syzygies.

### Checked source content, narrowly stated

SOURCE: a behavior is a set of trajectories; inputs/outputs are additional structure. Interconnection shares variables. For the paper's linear time-invariant differential behaviors, projecting away latent variables preserves membership in that class. In the construction R(D)w=M(D)h, generators N of the left syzygy of M yield the manifest law NR(D)w=0. The paper warns that nonlinear elimination can fail and that solution-space choices matter. State and input selectors can be constructed from laws rather than assumed initially.

Those claims are source results. No claim in that article was checked here about neural learning, autonomous feature discovery, catastrophic forgetting, computational superiority, or new capability acquisition.

### What was actually reconstructed

The candidate uses only a finite real-matrix specialization: for L_w w+L_h h=0, compute rows N spanning ker L_h^T. An h exists iff N L_w w=0. This follows directly from im L_h=(ker L_h^T)^perp; it does not need the differential-system elimination theorem, a completeness assumption on infinite traces, or a claim about polynomial nonlinear elimination.

My reconstruction also separates three facts that are easy to blur:

1. A relation can be represented without naming inputs and outputs.
2. A particular boundary has a solution and unique target values.
3. The chosen runtime can find those values cheaply and stably.

Only the first is free from directional commitment. The latter two require rank, conditioning, and cost analysis. The candidate's uniqueness condition is ker[R;P] subseteq ker T, derived in finite dimensions. A relation by itself is not an efficient inverse algorithm.

### How the reading changed the candidate

Before the original passage, the broad direction was to question the usual learned function as the unit of acquired computation. The concrete change made by the source was to use an admissible behavior, preserve latent variables during composition, and create a new operation through elimination. This yielded a genuine compiler rule, rather than attaching the word relational to a usual routed neural module. The source's nonlinear warning became a central attack, not a footnote.

The further proposed learning loop—trace covariance, neural port proposals, task-cost acceptance, acquired operator registration—is AGENT PROPOSAL/RECONSTRUCTION. It is not attributed to Willems and is not claimed original relative to machine learning or control literature.

## Additional source routes and limitations

Willems, **From Time Series to Linear System—Part I. Finite Dimensional Linear Time Invariant Systems**, Automatica 22(5), 1986, pp. 561–580, DOI 10.1016/0005-1098(86)90066-X.

Author original PDF:

<https://homes.esat.kuleuven.be/~sistawww/smc/jwillems/Articles/JournalArticles/1986.1.pdf>

The web tool retrieved PDF metadata and search-extracted first-page material but no readable line extraction. Screenshot calls did not supply a usable image to this subtask. I therefore do not count this as substantive original reading or rely on it for a theorem statement. The 2007 original was the actual load-bearing entry.

Willems, Rapisarda, Markovsky, De Moor, **A note on persistency of excitation**, Systems & Control Letters 54 (2005), pp. 325–329, DOI 10.1016/j.sysconle.2004.09.003.

Primary author/institutional repository route:

<https://eprints.soton.ac.uk/262195/1/PersistencyExcitation.pdf>

Search extraction exposed its introduction and proposition scope: spanning observed windows requires excitation conditions. Direct PDF open returned HTTP 403, so I did not personally enter its proof and do not invoke its theorem for the autonomous recurrence example. The candidate instead explicitly assumes window spanning and proves the elementary finite statement. A separate 1986 Part II PDF route timed out. No unavailable original is marked read.

The 2007 autobiographical article **In Control, Almost from the Beginning Until the Day After Tomorrow**, DOI 10.3166/EJC.13.71–81, was opened while finding the longer original. Introductory material was exposed; it supplied no load-bearing result and is not used in the candidate.

## Known-alternative audit before freeze

The linear comparator was constructed directly, not inferred from a slogan: an ordinary model learner that receives the same windows can compute their covariance/nullspace and use the same Gaussian elimination. It therefore matches the entire finite linear acquisition/composition/inverse example. This is actual mathematical coverage of that case, independent of naming a literature family.

Potential broader predecessors needing original reading after the freeze, only if the coordinator retains this reserve: neural implicit/energy models with variable clamping; learned factor models; differentiable declarative solvers; behavioral data-driven control; learned latent linearization. These are search directions, not assertions that their papers have been read or that the candidate is uncovered. No novelty claim is supported by this ledger.

## Evidence boundary

Checked: original behavioral definitions and linear-elimination scope; independent finite linear derivation; explicit matched alternative; observational, nonlinear-projection, and conditioning counterexamples.

Not checked: neural representation formation, training dynamics, total-resource gains, operation retention beyond archived witnesses, closest full machine-learning literature, or distinctness from the existing V2 architecture. No computational execution, numerical outcome, or neural learning success is claimed.
