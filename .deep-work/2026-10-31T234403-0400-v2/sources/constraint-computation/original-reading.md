# Original reading for the relation-transport candidate

Read-only source use; this note records actual passages entered, not a complete literature map. No source PDF or library index was modified.

## Willems 2007

Jan C. Willems, *The Behavioral Approach to Open and Interconnected Systems*, IEEE Control Systems Magazine 27(6), 46–99. [Original PDF](https://homes.esat.kuleuven.be/~sistawww/smc/jwillems/Articles/JournalArticles/2007.1.pdf).

SOURCE: Entered printed pp.46–53 (PDF pp.1–8), pp.55–59 (PDF pp.10–14), and pp.87–89 (PDF pp.42–44), including the hydraulic construction and the elimination argument. The theory defines a behavior as admissible trajectories; interconnection shares variables. Its latent-variable presentation induces manifest behavior by existential projection. Input/output structure can be derived subsequently and can fail globally for nonlinear systems. The linear constant-coefficient elimination argument computes a left annihilator N of M in R(d/dt)w=M(d/dt)l, obtaining NR(d/dt)w=0. Its general nonlinear extension is not asserted.

RECONSTRUCTION: A learned relation need not privilege a fixed inference direction. This reading changed the starting assumption that an internal acquired mechanism must be a function with a permanent input/output partition. Willems does not establish neural acquisition, efficient inference, or the transport proposal.

## Hatcher, Algebraic Topology, chapter 1

Allen Hatcher, [author-hosted chapter](https://pi.math.cornell.edu/~hatcher/AT/ATch1.pdf).

SOURCE: Entered printed pp.25–31 (PDF pp.5–11), especially the covering definition and proof through finitely many evenly covered neighborhoods; also printed pp.60–62 (PDF pp.40–42), Propositions 1.30–1.34. A chosen starting lift gives a unique lifted path; homotopies lift and preserve fixed lifted endpoints. The proof constructs a path segment by segment using local inverse charts and proves agreement by connectedness. Thus endpoint branch depends on loop homotopy class, rather than on endpoint alone.

RECONSTRUCTION: Use the retained lift as executable state. Closed boundary paths can generate operators on its discrete fiber. Applying this to learned neural relations, identifying operations with such loops, and the cubic calculation are our reconstruction, not results asserted by this source. Neural approximation does not automatically preserve a covering.

## Duff et al., 2018 original monodromy-solver paper

Timothy Duff, Cvetelina Hill, Anders Jensen, Kisun Lee, Anton Leykin, Jeff Sommars, *Solving polynomial systems via homotopy continuation and monodromy*, [original manuscript](https://arxiv.org/pdf/1609.08722), v4. Published IMA Journal of Numerical Analysis 39(3), 1421–1446 (2019), online 2018.

SOURCE: Entered manuscript pp.4–8, sections 2.1–2.3 and worked graph example, together with algorithm overview. The solver regards polynomial solutions as fibers over parameter space away from its discriminant. Lifted loops permute roots. Its graph stores known roots at vertices and partial root correspondences along homotopy edges; execution tracks unmatched roots, records correspondences and can add graph vertices/edges. This directly supplies the classical execution rival for our proposal. Its purpose is polynomial solution discovery, not neural acquisition of internal computational relations.

## Jipdi et al., 2026 braid programming

M. N. Jipdi et al., *Topologically Protected Learning from Exceptional Point Braiding: Toward Braid Programming*, [original HTML](https://arxiv.org/html/2608.15829v1), arXiv:2608.15829v1. This source was found after our v0 freeze.

SOURCE: Entered sections IV, V.5 and V.6 (especially equations 29–30 and the genetic-search specification). V.6 prescribes input/output examples, a bounded braid-word length and a predefined alphabet of gate generators/inverses. It proposes genetic search over those words. Thus fixed-substrate learning by compositional braid search is explicitly covered. We do not adopt its broader generalization/forgetting claims or certify its physical dynamics or numerical results. The descriptive residual in our repair is structural acquisition of the relation/generators while retaining existing corridor operations, not ordinary braid programming.

## Other actual reading, limited use

Zanardi and Rasetti, [*Holonomic Quantum Computation* (1999)](https://arxiv.org/pdf/quant-ph/9904011), pp.1–3 entered for the control-loop composition mechanism. Chen, Fuertes and Deng, [*Topological Computation by non-Abelian Braiding in Classical Metamaterials* (2025)](https://arxiv.org/html/2502.16006v1), sections III–IV entered for classical physical gate encoding. These establish close control-loop computation precedents; no quantum capability or physical robustness result transfers automatically to a digital neural relation.

## Angluin 1987, state-acquisition rival

Dana Angluin, [*Learning Regular Sets from Queries and Counterexamples*](https://people.eecs.berkeley.edu/~dawnsong/teaching/s10/papers/angluin87.pdf), Information and Computation 75, 87–106.

SOURCE: Entered printed pp.88–95, especially observation tables in section 2.1, the row-based state construction and consistency argument, and section 2.2/Figure 1. State candidates are distinguished by their responses to suffix experiments. Inconsistency adds a distinguishing suffix; lack of closure adds a new representative. Membership queries and counterexamples have explicit teacher assumptions. Therefore our proposed history-aliasing signal for adding a branch is a classical state-refinement signal, not by itself a new acquisition mechanism. These teacher assumptions cannot be transferred to a sampled neural training regime without being provided and costed.

## Caution about the 2026 braid-programming source

Root independently checked V.5–V.6 and flagged technical reasons not to adopt its strong claims: for 2-by-2 unitary matrices the displayed (1/4)|Tr(U_target^dagger U)| is at most 1/2 and is invariant to global phase, unlike the surrounding interpretation/table; a half-encirclement is not a closed-loop monodromy operation. This note uses only the checked fixed-alphabet search formulation as precedent. It does not use those disputed formulas or claims as evidence for our construction.
