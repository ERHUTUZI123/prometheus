# R source comparison after immutable v0

These are independently selected public original works. Comparisons below are limited to the inspected passages; no paper's claimed theorem, experiment or broad significance is accepted merely because it was printed.

## SOURCE — Phillips, 2021

*A reconstruction theory of relational schema induction*, PLOS Computational Biology, DOI 10.1371/journal.pcbi.1008641. Original: <https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008641>.

Read “Basic category theory”, “Reconstruction theory”, “Induction as reconstruction”, “Schema computation”, “Transfer as completion”, and definitions 70–71. Reconstruction: actions on task-specific sets and equivariant maps provide a common structure; natural compatibility is used to reconstruct a monoid; a free M-set construction supports transfer. Thus “learn shared action algebra then obtain a representation” is already an explicit concept. The source does not, in the checked passages, provide an executed neural algorithm that acquires a new internal memory topology with matched total search cost. V1's distinction is proposed internal semantics, not a claim that monoid reconstruction or free representation is new. The full-category reconstruction result must not be turned into a finite-feedback identification guarantee.

## SOURCE — van Heerdt, Kappé, Rot, Sammartino, Silva, 2022

*A Categorical Framework for Learning Generalised Tree Automata*, DOI 10.1007/978-3-031-10736-8_4. Original published text: <https://pure.uva.nl/ws/files/117731314/978_3_031_10736_8_4.pdf>.

Substantive reading: §2 assumptions/Definition 1, §5 local closure and consistency/Algorithm 4/Theorem 25, §6 tree and unordered-tree examples, §8 limitations. Reconstruction: for fixed categorical scaffold and functor F, wrappers separate represented states from experimental distinctions; repairs enlarge representation/refine quotients until the hypothesis is correct under stated conditions. Theorem 25 requires finitely many relevant subobject/quotient isomorphism classes. It is not a termination guarantee for an unbounded stack. F is fixed; §8 explicitly limits quotient equations to its framework. R cannot claim generic algebra learning or counterexample-driven representation growth. Acquiring the operational signature/theory and generating resource families is the remaining proposed gap.

## SOURCE — Sargsyan, 2026 preprint

*Functorial Neural Architectures from Higher Inductive Types*, arXiv:2603.16123v1. Original: <https://arxiv.org/html/2603.16123v1>.

Read §2.1–2.3, Construction 3.2, Theorem 3.3 and its displayed proof, plus appendix discussion of architectural identities. The checked construction takes a supplied generators/relations specification, assigns generator networks, composes segments structurally, and supplies networks for relation homotopies. This is direct precedent against describing specification-to-neural-architecture compilation as new. It does not show acquisition of the specification from task feedback in the inspected passages. I did not execute its code or verify its formalization, experimental comparisons or general attention impossibility claims, and R does not rely on those claims.

## SOURCE — Knuth and Bendix, 1970

*Simple Word Problems in Universal Algebras*, pp. 263–297, DOI 10.1016/B978-0-08-012975-4.50028-X. Original: <https://www.cs.tufts.edu/~nr/cs257/archive/don-knuth/knuth-bendix.pdf>.

Read §§1–3 on terms, substitutions and order; §4 Theorem 4; §5 superpositions/Theorem 5 and corollary; §6 completion and the failure example. Reconstruction: reductions use an ordering stable under substitutions; this prevents infinite reduction chains. Local competing reductions can be checked through finitely represented superpositions and joining reductions. Complete systems supply canonical representatives of the generated equality theory. Completion can fail or fail to terminate; changing the orientation/order can change that behavior. This supplies the v1 trusted-scaffold route and its limitations. The compiler/checker would be classical, and it has not been implemented by the executed R probe.

## SOURCE — Christian Wild, September 1987

*Analysis and Synthesis of Abstract Data Types through Generalization from Examples*, ICASE 87-59 / NASA CR-178369. Original: <https://ntrs.nasa.gov/api/citations/19880001137/downloads/19880001137.pdf>.

Substantive reading: §§2–4 (printed pp. 2–8), queue algebraic specification and canonical constructor forms. Important boundary: despite the title, this setup supplies an algebraic ADT specification and known syntax, generates examples by symbolic execution, then proves learned generalizations using the supplied axioms. Constructor-only canonical forms simplify learning. The inspected algorithm starts with predefined transformation features counting events. This is strong conceptual precedent for ADT/example analysis, but it does not establish acquisition of a previously unsupplied operational theory. V1 must not mistake a title-level match for complete mechanism coverage.

## SOURCE — Hawkins et al., 2012, limited passages

*Concurrent Data Representation Synthesis*, PLDI 2012. Original: <https://theory.stanford.edu/~aiken/publications/papers/pldi12a.pdf>.

Read §1 contribution details and §2 relational specifications/operations. The explicit setup supplies a relational specification and builds cooperating structures from a container library, with representation and lock choices. It is precedent for representation synthesis and workload-dependent selection, but the inspected scope does not acquire a generative operational law from task feedback. Full synthesis proofs and experiments were not read or checked; no claim about them is used here.

## Still OPEN

Direct comparison with inductive functional/logic synthesis, modern ADT synthesis and program-library learning is not complete. The explicit v1 should currently be classified as a possible neural-guided program/ADT synthesis construction. No novelty claim is justified by the uncovered differences above.
