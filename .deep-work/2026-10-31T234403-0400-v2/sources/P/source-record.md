# P source-reading record before v0 freeze

All records below concern original technical documents read through web retrieval. No paper code was downloaded or executed. Short quotations stay below 25 words per source; remaining notes are concise paraphrases. Sources support the stated prior mechanisms, not the candidate's efficacy or novelty.

## S1 — Ellis et al., DreamCoder, arXiv:2006.08381v1 (2020)

Original PDF: https://courses.cs.washington.edu/courses/cse599j1/22sp/papers/dreamcoder.pdf

Read PDF pp. 5–7 (zero-index pages 4–6), especially Figure 2, library abstraction, and representation. Figure 2 wake search uses P[rho|x,L] proportional to P[x|rho]P[rho|L]. Its abstraction step proposes routines from subtrees of equivalent program refactorings and improves a library-prior/program-probability objective. Its dreaming step trains Q on fantasies and replay. Page 7 specifies polymorphically typed lambda-calculus expressions. Load-bearing passage: “Propose new library routines from subtrees of refactorings of programs” (Figure 2). Read pp. 10–13 for higher-order primitives and comparisons including whole-solution memorization.

Boundary: already growing reusable program languages and neural guidance. P cannot claim reusable primitives or hierarchical acquisition as new. Potential difference must concern discovering interfaces in distributed executed state, rather than mining already explicit program structure.

## S2 — Geiger et al., Inducing Causal Structure for Interpretable Neural Networks, arXiv:2112.00826v2 (2022)

Original HTML: https://arxiv.org/html/2112.00826

Read Section 3, equations (1)–(4), and Appendix A, equations (13)–(15). IntInv replaces aligned internal variables on a base run by their source-run values. A supplied high-level model and a neural model are trained to agree on those counterfactual outputs. Their alignment maps intermediate high-level variables to nonoverlapping low-level subsets. Appendix A's zero-loss implication sums a nonnegative loss over every base/source pair; each zero addend then entails the required equality under a stated output map kappa. It does not justify replacing full pair coverage by sampled low loss.

Boundary: intervention training and intervention-based abstraction guarantees are prior work. P lacks the supplied high-level causal model; its teacher is an executable neural segment. This removes labels but does not establish meaningful counterfactual domains or make extraction easy.

## S3 — Geiger et al., Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability, JMLR 26(83), 2025

Original PDF: https://jmlr.org/papers/volume26/23-0058/23-0058.pdf

Read Sections 2.2–2.3.1, especially Definitions 11, 25, 28, Theorem 30; Section 2.5 Definition 46; Section 3.6. Definition 28 translates mechanisms by a bijection tau, conjugating the complete mechanism vector with tau and its inverse before projection. Theorem 30 gives an exact transformation under the explicitly constructed corresponding intervention algebra. Definition 46 uses translation, interchange in translated coordinates, and inverse translation for distributed interventions. Section 3.6 discusses both unsupervised modular-feature discovery and supervised alignment; Section 2.6 identifies an orthogonal learned translation in the hierarchical-equality example.

Boundary: a transformed coordinate system and distributed causal interfaces are not new ideas. P's equation (1) lies within this existing mathematics. The intended research claim must instead concern acquisition of useful executable modules and their effect on later learning, which these formal conditions alone cannot establish.

## S4 — Goyal et al., Recurrent Independent Mechanisms, arXiv:1909.10893

Original HTML: https://arxiv.org/html/1909.10893

Read Section 2 and Section 2.2, equations (1)–(2). The architecture divides recurrence into parameterized subsystems. Per-RIM queries attend to input keys and values, including a null row; top-k activation selects modules with least attention to the null input. Selected modules update and communicate through attention. The module family and sparse-interaction scaffold are supplied, while specialized dynamics and activation are learned.

Boundary: learned neural mechanisms and selective interaction are prior art. P proposes learning a callable boundary from an initially distributed transition family. A fixed modular system remains a strong rival; success cannot be inferred merely from a more flexible construction.

## Additional read, not relied upon for a distinct candidate claim

Geiger et al., Causal Abstractions of Neural Networks, arXiv:2106.02997v2: https://arxiv.org/html/2106.02997 . Read Section 3 equations (1)–(3), Appendix F Definitions F.6–F.7. The abstraction requires surjectivity of the state map and induced intervention map plus intervention agreement. Finite sampled interchange accuracy is evidence with a limited domain, not universal proof.

## Retrieval failures and limits

The Berkeley dissertation URL titled Neural Software Abstractions was surfaced by search but direct web opening returned an access error. Its technical content was not read and supports no source claim here. OpenReview's direct RIM PDF returned a verification page; the original arXiv HTML was read instead. No claim that the literature map is complete. No other candidate was read.
