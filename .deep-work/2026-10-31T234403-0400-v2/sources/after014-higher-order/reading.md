# Original reading for after014 higher-order construction

Same V2 continuation. These are selected original-body readings through browser PDF extraction and existing local source text, not a full literature review, replicated experiment or novelty certificate. Source text was read by this worker. Parent-reported Koza reading is separately labeled below. No raw source download, code or experiment was added by this worker; only this file and the owned candidate were written.

## Lafont — Interaction Combinators (1997)

Original article, *Information and Computation* 137, 69–101. DOI [10.1006/inco.1997.2643](https://doi.org/10.1006/inco.1997.2643). Read a [scan of the original article](https://chorasimilarity.wordpress.com/wp-content/uploads/2024/01/ic-lafont-1.pdf). The CORE URL returned 403; the scan was available. PDF text, not all graphical diagrams, was inspected.

Actual passages entered: §§1.1–1.2, printed pp.71–73 (one principal port, auxiliary ports, interface-preserving active-pair rules, uniqueness and Proposition 1); §§1.5–1.7, pp.76–80 (erasure/duplication, reduced/principal nets, translations); §§2.5–2.7, pp.86–90 (duplication obstruction, codes/copiers/decoder, general translation); beginning of §3.1, p.91 (two-stack execution interpretation).

SOURCE: the model's deterministic local compatibility is established prior work. Duplication of arbitrary executable packages is not a primitive consequence of having a duplicator; the code/decoder construction matters. The detailed graphical universality construction was not independently reconstructed, and no optimal-reduction result is claimed. Our affine beta wiring and stochastic causal-key proof are explicit reconstructions in the candidate. The reading changed the design by preventing an unexamined `COPY(function)` primitive and by making nonoverlap an architectural constraint.

## Didolkar et al. — Neural Production Systems (2021)

[Original NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/d785bf9067f8af9e078b93cf26de2b54-Paper.pdf). Entered §§2–3.2, PDF pp.3–5, particularly the rule embeddings/MLPs, primary/contextual slots, Gumbel selection and sequential/parallel alternatives. The primary-host indexed rendering of Appendix Algorithm 1 was also inspected; no appendix-wide reading is claimed.

SOURCE: learned sparse pair-conditioned rules and dynamic rule/slot selection are existing mechanisms. The inspected model changes slot states by rule-conditioned residual updates. It does not implement our proposed principal-pair port surgery or growing causal ancestry. That is a difference from these passages, not proof that no neural graph-rewrite architecture exists elsewhere. Source experiments were not replicated or independently audited.

## Ellis et al. — DreamCoder, arXiv v1 (2020)

Read the actual original-body text in the existing file `sources/rival/dreamcoder-2020.txt`, PDF pp.4–5: wake/sleep overview, task-program inference, library abstraction/refactoring and neural replay/fantasy training, including equation (1). Public original: [arXiv:2006.08381](https://arxiv.org/pdf/2006.08381). The read local version identifies itself as v1, June 2020; this record does not silently call it the final PLDI version.

SOURCE: jointly learning reusable program abstractions and a neural search guide is prior work. The proposed distinction would have to concern learning local evaluator semantics/encoding and causal execution, not the bare existence of a library or neural proposal loop. The candidate's port-graph-emitter compilation is our explicit matched-rival construction; it is not a claim about features implemented in released DreamCoder.

## Díaz-Caro and Martínez — Confluence in Probabilistic Rewriting (2017)

[Author-hosted original](https://mtzguido.github.io/pubs/confl.pdf), LSFA 2017 manuscript. Actual passages entered: introduction PDF p.2 (duplicating an unevaluated random expression changes result law); §2.2 pp.4–5 (PARS distributions, computation trees and scheduler distinction); §4 discussion pp.7–9 and Criterion 4.10 (distribution diamond); §6.1 pp.12–13 (affine/protected duplication restrictions and Theorem 6.3); concluding comparison pp.14–15.

SOURCE: probability plus nondeterministic redex choice requires its own confluence treatment; ordinary beta confluence does not imply stochastic result-law independence. Their distribution-level diamond is direct prior coverage, and their duplication example is a warning that our affine worked trace does not solve unrestricted function duplication. The full paper's proofs were not independently verified. Our pathwise ancestry-field proof is a restricted direct derivation, not an attribution of that exact construction to this paper.

## Faggian — Probabilistic Rewriting: Normalization, Termination, and Unique Normal Forms (2019)

[Original FSCD article](https://drops.dagstuhl.de/storage/00lipics/lipics-vol131-fscd2019/LIPIcs.FSCD.2019.19/LIPIcs.FSCD.2019.19.pdf), DOI [10.4230/LIPIcs.FSCD.2019.19](https://doi.org/10.4230/LIPIcs.FSCD.2019.19). Entered introductory distinctions pp.19:2–19:3 and §5 pp.19:14–19:15, especially Definition 5.1, Theorem 5.4 and Proposition 5.5. The proof of the local/global characterization on p.19:15 was read; not every appendix proof was read.

SOURCE: probabilistic random descent can give equal normal-form probabilities at every step bound, equal limit and expected steps under appropriate equivalence/local hypotheses. It is established theory. Almost-sure termination is a limit property; one must not move an ordinary finite well-founded induction into unrestricted PARS without justification. Our candidate instead freezes an independent random field and proves the finite-witness property in the resulting deterministic labeled system. This is a special mechanism argument, not a new generalization of Faggian's results.

## Varacca and Yoshida — Probabilistic pi-Calculus and Event Structures (2007)

[Author-hosted original](https://www.lacl.fr/~dvaracca/papers/probpi.pdf), DOI [10.1016/j.entcs.2007.07.009](https://doi.org/10.1016/j.entcs.2007.07.009). Author-PDF page numbers below differ from the journal's 147–166 pagination.

Actual original passages entered: pp.2–3 (local probability and schedulers), pp.5–6 (linear use and confusion-free composition), pp.8–9 (§4 event structures, cells, local probabilities), pp.12–14 (§6 conversion to Segala automata, fair scheduler definition, Theorem 6.2, operational correspondence and the explicit untyped schedule-dependent counterexample). Source says some proofs are omitted for space; no claim of having checked omitted proofs.

SOURCE: independent local probabilistic choices on a suitable event structure admit scheduler-independent event probabilities; Theorem 6.2 explicitly gives equality across fair schedulers. Their untyped race example shows why an execution discipline matters. This directly covers the broad stochastic schedule-independence principle. The neural generated-code kernel is not discussed in the inspected passages. Combining that kernel with the known discipline remains an architecture proposal, with usefulness and novelty OPEN.

## Abbes and Mairesse — Uniform and Bernoulli measures on the boundary of trace monoids (2015)

[Original arXiv v3](https://arxiv.org/pdf/1407.5879). The author-hosted IRIF URL was unavailable through the browser; the original arXiv PDF was readable. Entered pp.2–3's two failed naive probability constructions and §§2.1–2.3 pp.4–7: fixed independence alphabet, traces, overlapping cylinders, Proposition 2.1 and Definition 2.2's multiplicative Bernoulli condition. Did not reconstruct the full Möbius characterization or realization theorem.

SOURCE: a probability on serial words does not automatically become uniform on concurrent traces; trace-prefix cylinders overlap and require careful compatibility. This prevented calling our local-kernel event measure a uniform trace-monoid prior. Our event probability is defined by once-only normalized emissions under a canonical scheduler and shown insensitive to independent reordering; dynamic graph generation is not directly the source's fixed finite independence pair. Probabilistic event structures are the closer rival.

## Schulman, Heess, Weber and Abbeel — Gradient Estimation Using Stochastic Computation Graphs (2015)

[Original NeurIPS paper](https://papers.nips.cc/paper_files/paper/2015/file/de03beffeed9da5f3639a621bcab5dd4-Paper.pdf). Actual passages entered: §2.1, PDF pp.2–3, equations (1)–(4), differentiability caveats and score-function/pathwise distinction; §3.1 and §4, PDF pp.5–7, Condition 1, Theorem 1 and the baseline statement of Theorem 2. The appendix proofs were not independently reconstructed. No claim of new gradient machinery is made.

SOURCE: score-function and pathwise estimators and their combination are established. In our exact b-bit stochastic variant, the likelihood includes emitted code bits so no unexamined derivative through quantization is used. Our event-score sum is an application of the ordinary likelihood identity after establishing that complete causal configurations have a scheduler-independent product law. Unbounded stopping still needs integrability; the candidate explicitly supplies sufficient finite-bound conditions instead of claiming automatic validity.

## Actual old campaign bodies read before any experimental commitment

- Old `2026-10-31T215613-0400-campaign/branches/A/candidate-v0.md`, selected local-neural graph/E-P-C formation passages: local learned execution/plasticity/construction and dynamic occupancy were already explicitly proposed; local graph growth alone is not new here.
- Old `branches/AE-v2/candidate.md`, actual finite installer mechanism, scalar-feedback learning and failure of conditional driver acquisition: choosing an assembly from a supplied alphabet is not discovering an evaluator.
- Old `branches/D/v0.md`, actual encoder/editable-representation/program construction, plus `sources/D/prior-attack.md`: editable lenses and neural/library synthesis already cover broad recognizer-plus-operation learning stories.
- Current sealed `candidates/R/candidate-v0.md`, `candidate-v1.md`, `candidate-v2.md`, and `disposition.md`, actual equations and update laws. V1 directly executes learned first-order rewrite semantics and compares same-grammar CEGIS; v2's finite matrix/edit algebra is not reopened. The present finite rewrite-table route would repeat that boundary. The generated local kernel and root-seeded stochastic semantics are explicitly separated from it.
- Old `branches/BF/candidate-v1.md` was also read while checking the exclusion: its gauge-preservation/positive-mobility repairs do not supply the present interaction-law mechanism and are not repackaged.

No old original body was edited. Existing summaries were not substituted for these bodies. Root additionally reports having read Koza GP II's primitive-semantic and UNDEFINED passages from the original library PDF. **PARENT REPORT**, not this worker's independent SOURCE: primitive-meaning/arity evolution is established coverage; the LISP evaluator/binding apparatus remains supplied there.

## Evidence boundary

No code/experiment was run, no new implementation or trained weights exist, and no empirical success is reported. All graph reconnections, matching counterexamples and the causal-law proof are written derivations in the owned candidate. Reading a source theorem or obtaining another agent's agreement does not independently verify those derivations. Strongest remaining issue: a formation law that makes useful generated interaction semantics learnable beyond same-information probabilistic graph-evaluator induction.

## After the raw-stream v1 construction: Sinot original token-passing execution

François-Régis Sinot, *Call-by-Name and Call-by-Value as Token-Passing Interaction Nets*, TLCA 2005, pp.386–400. [Author/institution-hosted original](https://lsv.ens-paris-saclay.fr/Publis/PAPERS/PDF/sinot-tlca05.pdf), DOI [10.1007/11417170_28](https://doi.org/10.1007/11417170_28). This was first read after saving `raw-stream-v1.md`; it did not supply the witness tables in that initial save. Read introduction; §§3.1–3.7, PDF pp.4–10, especially the syntax-to-net translation, copy/erase rules, closed-term Proposition 3 and explicit eval/return control; §4, PDF pp.11–12, the call-by-value transition equations and propositions. Extracted prose/equations were entered; not every graphical wiring figure was visually audited and the full correctness theorem was not independently proved.

SOURCE: a local token-passing implementation with explicit evaluation order, return, copying and erasure is established. The translation in §3.4 links bound occurrences to their binder and supplies sharing structure. That inspected input is not a raw token chain whose parser/binding algorithm was learned from terminal labels. Therefore the witness's execution organization has close concrete precedent, while its proposed *acquisition interface* is a different question, not a demonstrated novel method. The author also explicitly observes that recursive host evaluation would hide stack/control work; our local continuation and parser rows expose those costs.

Related author originals opened: [Sinot's DCM 2005 preliminary call-by-need paper](https://lsv.ens-paris-saclay.fr/Publis/PAPERS/PDF/sinot-dcm05.pdf), introduction and §3's supplied binder-backpointer translation; [2006 call-by-need manuscript](https://lsv.ens-paris-saclay.fr/Publis/PAPERS/PDF/Sinot-mscs06.pdf), introduction only. Neither paper's full sharing/fully-lazy machinery was reconstructed. No claim of an efficient sharing implementation or optimal reduction follows from these readings. The shorter call-by-need manuscript describes a multiple-principal-port extension; our witness retains one principal port and passive environment-tree copying, so it must not silently inherit that extension's results.

SUPPORT, not this worker's independent reading: `conventional-rival.md` was read after v1's first save. The supporting worker entered actual Neural GPU/NPS/NTM original bodies and proposes the unchanged CGRU Neural GPU as the strongest concrete raw-input conventional comparator. Its reading provenance remains with that artifact. No hypothetical duplicate of our new emitter is claimed as historical proof of coverage.

## During acquisition v2: coherent parameter exploration is established

Plappert et al., *Parameter Space Noise for Exploration*, ICLR 2018, [original arXiv v2](https://arxiv.org/pdf/1706.01905). Actual body entered: §3, PDF pp.2–3, especially the passage sampling perturbed neural parameters at the beginning of an episode and holding them fixed throughout the rollout; repeated-state consistency; parameter-noise adaptation equation (1). Did not reproduce the empirical results or read every appendix.

SOURCE: coherent parameter-space exploration is prior work; it is not the new principle in acquisition-v2. Our whole-finite-W mixture search uses the ordinary once-per-law likelihood score. The elementary conditional p^R versus p^N comparison is our restricted explanatory derivation, not a bound attributed to this source or a guarantee for the neural kernel. Actual Neural GPU/NRAM competitors may use the same optimizer family.

## During acquisition v2: capture/sharing tradeoffs are established closure theory

Shao and Appel, *Space-Efficient Closure Representations*, LFP 1994, original printed pp.150–161. Entered introduction and §2, printed pp.150–152, from a [scan of the original proceedings article](https://3e8.org/pub/scheme/doc/lisp-pointers/v7i3/p150-shao.pdf). Actual passages: closure environments for free variables; flat versus linked versus safely linked representations; linked-retention space example; copying costs; selective sharing of lifetime-grouped data. No source benchmark numbers are used as our evidence, and the full compiler algorithm is not reconstructed.

SOURCE: selective free-variable capture and safe sharing/copying tradeoffs are established. The v2 FV/Project controller protocol was saved before this reading and is a local reconstruction, not new closure conversion. It repairs one checked environment-growth family and does not imply general efficient sharing.

The [author's 2000 extended paper](https://flint.cs.yale.edu/shao/papers/escc.pdf) was opened first, but its browser text extraction was badly garbled; a university mirror timed out. Those failed attempts are not recorded as substantive original-body reading. The readable 1994 original supplies the comparison above. Root/other worker's subsequent relevant-capture counterfamily is separate derived evidence, not a source theorem.
