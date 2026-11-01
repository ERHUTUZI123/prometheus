# Selective source reading for R v2

The sources below were selected to test the load-bearing computation/learning differences. Their existence is not treated as a veto of a new organization of known operations. No empirical claims from these works were reproduced here.

## SOURCE — Tallec and Ollivier, 2017/2018

*Unbiased Online Recurrent Optimization*, arXiv:1702.05043v3, ICLR 2018. Original: <https://arxiv.org/pdf/1702.05043>.

Read §3.1 equations (4)–(5), §3.2 Proposition 1/equations (6)–(15), §3.3 implementation and the footnote on changing parameters. The source reconstructs exact recurrent sensitivities as a state-by-parameter array and replaces them with a rank-one stochastic approximation whose expectation follows the sensitivity recursion. Random signs cancel cross terms in expectation. R v2 therefore cannot claim the first history-free recurrent credit algorithm. Its proposed distinction is learned exact structural closure with an explicitly limited edit family; UORO is an actual alternative when such closure is large or absent. The source's changing-parameter limitation is retained in the candidate.

## SOURCE — Evci et al., 2020

*Rigging the Lottery: Making All Tickets Winners*, ICML/PMLR 119. Original: <https://proceedings.mlr.press/v119/evci20a/evci20a.pdf>.

Read §3, especially drop/grow criteria and Algorithm 1 on pp. 3–4. Connections are removed by a magnitude criterion and grown using high-magnitude instantaneous gradients; newly activated connections start at zero, and infrequent dense-gradient computation is part of the cost. Thus R cannot claim gradient-selected structural birth as new. The possible additional object is a compositionally closed credit carrier whose derived channels are not necessarily independent trainable connections. Full source experiments and FLOP claims are not independently verified.

## SOURCE — Farajtabar et al., 2020

*Orthogonal Gradient Descent for Continual Learning*, AISTATS/PMLR 108. Original: <https://proceedings.mlr.press/v108/farajtabar20a/farajtabar20a.pdf>.

Read §§2–3, equations (4), (7)–(9), Algorithm 1 and the stale-gradient limitation. Prediction gradients define a protected subspace; new-task gradients are projected orthogonally to it. This is direct precedent for R's projection arithmetic. The proposed protected object in R is a scoped operational identity whose contextual consequences can cover unseen programs, instead of selected old input/output predictions. That distinction is a candidate scientific issue, not proof of a better method; old task performance is not guaranteed by retaining an operational identity.

## SOURCE — Ong and Veličković, 2022

*Learnable Commutative Monoids for Graph Neural Networks*, LoG/PMLR 198. Original: <https://proceedings.mlr.press/v198/ong22a/ong22a.pdf>.

Read §§2.4–2.5 and Appendix E's node regularizers. The checked construction balances a learned reduction tree and enforces/regularizes commutativity and associativity. Strong commutativity comes from symmetrization; associativity is weakly regularized in the inspected architecture. This is close precedent against presenting learned algebraic aggregation or shorter composition depth as new. R v2 concerns transported edit spaces and structural changes required by credit, which those inspected passages do not settle.

## SOURCE — Chen, Goodfellow and Shlens, 2016

*Net2Net: Accelerating Learning via Knowledge Transfer*, arXiv:1511.05641v4. Original: <https://arxiv.org/html/1511.05641>.

Read §§2.2–2.4, particularly the widening duplication/rescaling equations and the requirements for identity-layer insertion. The construction initializes a larger network to preserve the teacher's function, then continues ordinary training. Therefore preserving existing computation when adding capacity is not itself a new mechanism. R must distinguish learned edit transport and continued scoped laws from this actual comparator, with growth and training costs counted.

## RECONSTRUCTION versus novelty

Paired multiplication `(a,m)(b,n)=(ab,an+mb)`, matrix-unit closure and derivative transport are standard algebra/calculus. R v2 uses them as designed mechanisms. No priority claim is made for them. The open proposal is to acquire and retain the computational algebra together with the closed space of useful edits, and to let escaped credit inform resource construction. Exact structural forward-mode compilation may already cover much of this; that is a targeted unresolved comparison, not a completed literature exclusion.
