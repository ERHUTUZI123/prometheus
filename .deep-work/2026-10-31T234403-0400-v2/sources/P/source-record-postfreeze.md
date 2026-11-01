# P primary reading after v0 freeze

No other candidate or prior campaign artifact has been opened. These sources were selected to check whether replacing exact physical framing by future behavioral sufficiency would merely rediscover known abstraction theory.

## S5 — Singh, Littman, Sutton and Stone, Learning Predictive State Representations

Original author-hosted technical PDF: https://web.eecs.umich.edu/~baveja/Papers/nipspsr.pdf . The retrieved copy is visibly marked as a draft; no publication-version claim is made here.

Read Section 2, equations (1)–(3), and Section 3. A set of core action-observation tests is sufficient if predictions of every other test are functions of the core predictions. For linear PSRs these functions are linear; the recursive state update is a ratio of extension-test predictions. Section 3 assumes an epsilon-soft known behavior policy and estimates parameters from executed extension tests, using importance weights. The document explicitly distinguishes learning correct predictions for a given test set from discovering the core test set; the latter is deferred.

Boundary: future-test sufficiency is established prior work. Merely defining internal state through possible continuation outcomes is not P's contribution. The source does not establish P's learned module-construction loop or its ability to accommodate newly added operations.

## S6 — Li, Walsh and Littman, Towards a Unified Theory of State Abstraction for MDPs (2006)

Original author-hosted PDF: https://thomasjwalsh.net/pub/aima06Towards.pdf

Read Section 3, Definitions 1–3 and Theorems 2–4. The state map partitions ground states and uses a normalized weighting function to define abstract transitions/rewards. Model-irrelevance preserves action rewards and transition mass into every abstract block. Other criteria preserve all-policy values, optimal values, or progressively weaker optimal-action information. Theorem 4's learning claims explicitly assume infinite state-action visits and appropriate step-size decay; weaker action abstraction has counterexamples to unrestricted convergence or optimality. Section 2 also records using options to encapsulate abstraction information as prior work.

Boundary: one cannot conflate preserving current best behavior with preserving a representation suitable for subsequent learning. The source treats fixed MDP/abstraction relationships; it is not evidence that an expanding neural operation language retains sufficient interfaces automatically.
