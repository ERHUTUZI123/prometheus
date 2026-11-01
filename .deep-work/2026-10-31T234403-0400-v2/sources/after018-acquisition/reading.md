# Original reading for after018-acquisition v0

Read before old/current candidate bodies or reviews. Sources were accessed read-only through web tools. No source PDF was saved to disk. PDF skill was read; local pypdf and Poppler were unavailable, so no installation was attempted. This note records what was actually entered, not a claim of a completed literature landscape. UTC freeze date: 2026-11-01.

Before hash freeze, after both substantive files were first written, the parent named SRWM and learned loss as its own reading directions and requested a provenance correction. These two names are pre-hash-freeze exposure; no associated equations or current candidate/review bodies were supplied or read. They did not supply the independently chosen mechanism already written here. Isolation was an instruction with shared filesystem access, not technical blindness.

## Source 1: predictive tests and nonlinear compact state

Rudary, Matthew R., and Satinder Singh. *A Nonlinear Predictive State Representation*. NeurIPS 2003.

Primary body: https://papers.neurips.cc/paper_files/paper/2003/file/72e6d3238361fe70f22fb0ac624a7072-Paper.pdf

Entered: PDF pp. 3-5, section 3.1, Figure 1, Lemmas 2-4 and update equation (5); the k-bit-register dimension comparison is on PDF p. 5. The construction starts from known POMDP outcome vectors; the authors explicitly defer learning those vectors from observations. Their end-tests use an action sequence and only its ending observation. The register example shows that compact predictive state and nonlinear prediction are old, and that linear-state dimension is an inadequate lone baseline. This reading does not establish that all later acquisition work left the same gap open.

SOURCE / RECONSTRUCTION boundary: this source supports the prior-work warning, not v0's particular residual-to-recurrent compiler. V0 independently derives the character-basis mechanism in its own restricted command class.

## Source 2: product learning and counterexample-based refinement

Moerman, Joshua. *Learning Product Automata: Revisiting a trick by Rivest and Schapire*. arXiv:1705.02850v1, 2017.

Primary full text: https://arxiv.org/html/1705.02850

Entered: sections II-III, definitions 1-4, register-machine example, Lemmas 1-2, Figure 2, and section III-B. Product transitions run components in parallel; the observation-table algorithm refines component consistency and closedness. Learning assumes known input/output alphabets, product output projections, membership queries, and equivalence queries. The latter either certify the entire hypothesis or return a counterexample. This establishes that factored learning and counterexample-based state refinement are prior mechanisms. V0's resettable scalar endpoint access differs from product output projections and does not include an uncharged equivalence oracle.

Not independently checked: the 1994 Rivest-Schapi­re paper cited there. Its historical priority is not claimed as independently verified here.

## Source 3: learned action-group structure, with explicit observability assumptions

Dang-Nhu, Barthélémy, Louis Annabi, and Sylvain Argentieri. *Disentangled Representation Learning through Unsupervised Symmetry Group Discovery*. arXiv:2603.11790v1, 2026; ICLR 2026 version located.

Primary body: https://arxiv.org/html/2603.11790v1

Entered: section 4.2, assumptions 1-3, equations (5)-(6), theorem 2, and Appendix G. Assumption 1 requires an injective observation map, modulo interaction equivalence. Assumptions 2-3 constrain available actions relative to a subgroup decomposition. Theorem 2 additionally uses finite world state, all transitions in the dataset, and convergence of the first-stage loss. The action-distance criterion uses learned representation matrices and bounded powers of available actions. Appendix G explains why their observation-encoder definition needs observability. Learning group structure is therefore not new. A history-based setting changes the premise but is not itself a novelty claim.

Access limitation: the OpenReview PDF presented a browser verification page; the arXiv full body above was readable. The relationship between versions was not compared line by line.

## Source 4: the acquisition subroutine is conventional query learning

Bshouty, Nader H., and Vitaly Feldman. *On Using Extended Statistical Queries to Avoid Membership Queries*. JMLR 2 (2002), 359-395.

Primary body: https://jmlr.csail.mit.edu/papers/volume2/bshouty02a/bshouty02a.pdf

Entered: printed pp. 367-369, sections 2.4-2.5; their section 3.3 was also inspected. The body explicitly supplies character products, correlation coefficients, prefix squared-energy estimation, and branching by a heavy-coefficient threshold. The real-valued extension is Remark 7. V0's acquisition routine reconstructs that established subroutine; its paired-query estimator is derived in the candidate. No novelty is claimed for sparse-Fourier discovery or its polynomial dependence on inverse threshold.

Attribution limitation: this is an original primary research paper, but its section 2.5 reconstructs Kushilevitz-Mansour rather than originating that algorithm. The original KM PDF was located at https://www.cs.princeton.edu/courses/archive/spring09/cos522/KushilevitsMansour.pdf but web retrieval returned 403. A local memory-only extraction attempt stopped immediately because pypdf was unavailable; it did not read the PDF or install anything. Consequently this note does not claim a full original-KM-paper reading.

## Additional original body entered but not load-bearing for v0

Quessard, Robin, Thomas D. Barrett, and William R. Clements. *Learning Disentangled Representations and Group Structure of Dynamical Environments*. NeurIPS 2020.

Primary body: https://proceedings.neurips.cc/paper/2020/file/e449b9317dad920c0dd5ad0a2a2d5e49-Paper.pdf

Entered: PDF pp. 3-4, section 3's equivariance relation and section 4.1's learned special-orthogonal transformation parameterization. This was an early check that learned latent transformations and disentanglement were established. No priority or performance inference from this source is needed for the v0 judgment.

## Reading conclusion and remaining uncertainty

SOURCE: all four load-bearing neighboring ingredients have substantive precedents. RECONSTRUCTION: the candidate's sign-state execution, basis-rank statement, learning-cost accounting, and worked process are hand derivations, with limits stated in the candidate. OPEN: there is no landscape-complete answer about acquisition of an unknown useful operator algebra under partial observation. More importantly, v0 has not yet constructed an advantage there; an unsearched gap is not a contribution.

The strongest same-information explanation does not need a literature-complete novelty verdict: ordinary sparse-Fourier learning, binary elimination, and a register compiler already reproduce every proved behavior in the candidate. That equivalence is a direct check of v0, not a source's empirical result.
