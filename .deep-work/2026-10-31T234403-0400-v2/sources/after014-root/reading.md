# Root actual reading after explicit after014 continuation

Primary entries; no claim from an abstract alone, no experiment yet. These are constructive constraints, not a demand that every component be unprecedented.

## IIT (Geiger et al., ICML2022)
https://proceedings.mlr.press/v162/geiger22a/geiger22a.pdf . Root actually read §3, printed pp2–4, Eq1–4 and Algorithm1: interchange internal values from a source execution into a base run, compare the neural output with the counterfactual output of a supplied causal model, and differentiate through that intervention. A high-level model and alignment supply the target. Full all-input-pair agreement implies the stated abstraction; sampled agreement is weaker. Our inference: merely making intervention tests trainable is known. A candidate without a supplied high-level model must specify where counterfactual truth comes from and how intended intervention semantics are identified. No raw benchmarks/full causal-abstraction theorem audit.

## DreamCoder (Ellis et al., PLDI2021)
https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf . Root read printed pp836–839, §§2.1–2.4, Fig2–3 and Eq1–3: finite beams of synthesized programs, library refactoring using e-graphs, and neural search trained on replayed/fantasized program-task pairs. The base typed language, scope/evaluation semantics and domain/task corpus are supplied. Refactoring exposes higher-order shared abstractions such as map. Our inference: learned reusable library plus search amortization is a concrete existing mechanism; changing labels to 'internal operation' is not enough. Full e-graph algorithm and experiments not yet checked; no reported enormous-search-time claim adopted.

## Koza, Genetic Programming II — actual local library entry

Read through existing read-only campaign venv pypdf, no install. Local food/Genetic Programming II_ Automatic Discovery of Reusable Programs (Complex Adaptive Systems).pdf, 768PDFpages. Page index=printedpage+21. Root entered complete printed575–580,594–595,601–604, not just TOC. Ch22 evolves primitive truth tables and arities together with ADF/program architecture; a fixed constrained LISP table interpreter remains. Ch24 lets UNDEFINED act as a third internal value and selects programs with defined boundary outputs; this differs from simply treating every undefined intermediate as fatal. No numerical performance claim is adopted as independent evidence. The scanned/OCR LISP punctuation/tables are imperfect; no exact source-code transcription is used. Remaining chapters and reported runs not checked. Constructive implication: primitive-meaning or alphabet extension alone is already insufficient for novelty; learning binding/evaluation organization would be a different question only if those semantics are not supplied by the input encoding.

## Root recent causal originals

SOURCE https://arxiv.org/html/2602.24266v1 , Asiaee2026: actually read §§3.1–3.4 and4 plus5.1 Eq4–5; constant/affine mechanism replacement, restricted retained-coordinate interventions and exact downstream folding. No semantic high-level model is supplied, but this constructs an approximation of an existing model rather than a new task operation. Surrogate risk is not IIA itself. Benchmarks/full proofs not audited.

SOURCE https://arxiv.org/html/2608.14894v1 , Tomaszewski2026: actually read §§3.1–3.7 and relevant introduction. It learns g_phi(I) from measured functional discrepancies c(I) then selects a structural action; no claim of universal superiority over direct empirical memory/search is adopted. Underlying experiments/their preregistration not audited here. Predictive self-intervention alone is existing scope; failed-trace operator compilation is a different unverified candidate.

## Root nominal original

SOURCE https://www.gabbay.org.uk/papers/newaas.pdf , Gabbay/Pitts conference version, actually entered §§2–3 including Eq2, finite support, fresh-atom assumptions and Remark3.3 on choice, PDFpp2–4. Journal version newaas-jv.pdf had corrupt browser extraction and is unread beyond metadata. This supports name-permutation/freshness as established foundations; root's finite machine correspondence proof is its own stated reconstruction. Full name-abstraction/universality proofs not checked.

## Neural GPU — root direct reading

SOURCE https://arxiv.org/html/1511.08228 , v3, root entered complete §2 and §3.3: raw token embeddings initialize one grid column; shared gated convolution runs recurrently, with final token loss. No symbolic parser/evaluator is installed. The parameter-sharing relaxation initially trains temporal parameter versions with a growing agreement penalty, then averages and fine-tunes a shared transition. The reported large hyperparameter search is part of acquisition cost, not free evidence of reliable learning. Full implementation and task results not independently reproduced. Mutable exact graph links, rather than raw-I/O algorithm learning alone, are the candidate's remaining architectural hypothesis.

## Neural Random-Access Machines — root direct reading

SOURCE https://arxiv.org/pdf/1511.06392 , printed pp2–7, §§3–4.1 and module list §4.3. Controller selects fuzzy circuits over supplied gates; register updates use softmax mixtures (Eq1–3). Registers expose only zero-probabilities to the controller. READ is M-transpose times pointer distribution; WRITE erases/adds at addressed cells. Terminal I/O likelihood and Adam train without access traces. Discretizing selections enables ordinary exact pointers when learned choices permit it; dense training READ costs quadratic memory-size work. Supplied arithmetic, comparison and read/write modules are explicit scaffolding. Root did not audit the implementation or full experimental tables. This is a concrete learned mutable-memory/pointer rival, not evidence it acquires lexical evaluation at matching cost.

## Neural Language Interpreter (2026) — root direct reading

SOURCE https://arxiv.org/html/2604.18907v1 , Macfarlane et al., actual §§2–3.4, Eq1–3 and Algorithm1. A transformer inductor infers discrete latent tokens from I/O specifications; a shared neural executor processes them sequentially, predicts output distributions and re-embeds the result. Leave-one-out output loss and token-reuse regularization train the model; Gumbel relaxations and latent test-time gradient search provide acquisition/search. No symbolic executor or task program trace is required. The sequential execution interface and codebook organization remain designed. Full results, code and cost comparisons not audited; reported superiority is not adopted here. This is close concrete coverage of jointly learned primitives and execution semantics, not exact generated pair-local graph surgery.

## Lafont — root direct reading

SOURCE https://chorasimilarity.wordpress.com/wp-content/uploads/2024/01/ic-lafont-1.pdf , Interaction Combinators (1997), printed pp69–74, especially §§1.1–1.2 and Proposition1. Root checked finite port partition, interface-preserving pair rules, symmetry/uniqueness and the disjoint-redex diamond proof. Local erasure/duplication are explicitly distinguished from treating arbitrary copying as atomic. These deterministic execution foundations are prior work; root has not reconstructed the entire universality/copy-code proof or all graphical reductions. The candidate's new learning claim must lie elsewhere.

## Sinot — root direct reading

SOURCE https://lsv.ens-paris-saclay.fr/Publis/PAPERS/PDF/sinot-tlca05.pdf , TLCA2005, author-PDF pp1–11: introduction, §§3.2–3.7 and call-by-value small-step equations in §4. Root checked the supplied syntax-to-net translation groups occurrences and attaches binder backpointers; evaluation/return and explicit copy/erase control are local. Recursive host evaluator control would otherwise hide stack work. Proposition3 covers closed-package copying, not unrestricted arbitrary active/cyclic graphs. Graphical rules were inspected through extracted equations/prose, not every image; full formal simulation/correctness was not independently proved. The candidate must learn its raw parsing/binding organization rather than claim this known evaluator as new.

## Parameter exploration — root direct check

SOURCE https://arxiv.org/pdf/1706.01905 , Plappert et al. (ICLR2018), §3 printed pp2–3, including episode-fixed sampled neural parameters, repeated-state consistency and noise-scale Eq1. Root entered the actual method passage independently after the constructor's reading. This covers the broad coherent-policy exploration rationale; it does not provide the finite-bit mixture's exact distribution or a theorem that graph-interpreter weights have appreciable successful mass. Experiments, complete RL variants and code were not audited here.

## Willems behavioral theory — root direct reading after reserve freeze

SOURCE https://homes.esat.kuleuven.be/~sistawww/smc/jwillems/Articles/JournalArticles/2007.1.pdf , printed pp46–49 and87–89, Theorem2, Eq27–28 and elimination sidebar. Root entered variable-sharing/behavior definitions and the full target elimination passage. The theorem concerns specified linear differential behaviors and function spaces; nonlinear projection and L2/compact-support variants do not inherit it. Arbitrary annihilators give only a vacuous characterization unless a small computable family is available. The finite left-nullspace example in the source matches the reserve's elementary specialization. Root independently checked its covariance, finite elimination proof, recurrence composition and uniqueness condition. No dynamic/PDE theorem formalization, original experiment or nonlinear neural-learning result was checked.
