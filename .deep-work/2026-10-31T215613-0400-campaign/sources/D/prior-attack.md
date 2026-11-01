# Post-freeze primary-source attack

These sources were first searched/downloaded after the v0 freeze at 2026-11-01T02:13:53Z. Manifest contains exact URLs, hashes and download times. Only the selected pages below were read; no claim of full-paper verification.

## SOURCE: Foster et al., Combinators for Bi-Directional Tree Transformations

Author-hosted [POPL paper](https://www.cis.upenn.edu/~bcpierce/papers/newlenses-popl.pdf), local foster2005.pdf, PDF pages 3–4, sections 3–4. The local pagination has no printed page number on these pages.

The source defines a get function, a putback using the old source, GetPut/PutGet laws, optional PutPut, totality and composition. Its composition formula is the same as v0's nested-view formula. Therefore v0's algebra is an independent reconstruction of existing lens semantics, not a new mechanism. Crucially, the source does not require PutPut for all combinators: its important map combinator can fail that law. Requiring all three laws universally, as v0 initially did, would exclude useful editable representations. This is a substantive correction, not a numerical tweak.

## SOURCE: Miltner et al., Synthesizing Bijective Lenses (arXiv v1, 2017)

Author paper [arXiv PDF](https://arxiv.org/pdf/1710.03248), local optician2017.pdf, PDF pages 2–3 / printed 1:2–1:3 and PDF 10–11 / printed 1:10–1:11.

Optician receives source/target regular expressions plus disambiguating examples, converts types to disjunctive normal form, synthesizes in that representation, and converts back to a bijective lens. Types supply much information and control search. Named subformats support type propagation and selective expansion; normalization can blow up exponentially. Thus automatically constructing bidirectional transformations is prior work, and giving the schema/type is a scientifically important scaffold. The paper's completeness and benchmark results are SOURCE claims, not independently reproved/replicated here. D's remaining question removes a supplied target schema and chooses it from downstream task usefulness; this comparison alone does not establish that question as novel.

## SOURCE: Ellis et al., DreamCoder (PLDI 2021)

Author-hosted [paper](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf), local dreamcoder2021.pdf, PDF pages 4–7 / printed 838–841, particularly equations 1–4 and Figure 3.

DreamCoder jointly infers programs and a typed lambda-calculus library, uses finite task-program beams to approximate its Bayesian objective, grows library abstractions by refactoring/compression, and trains a neural recognition model from solved-program replays and executed sampled-program fantasies. V0's neural proposal/library bootstrap loop therefore has close precedent. Adding executable views to such a loop is not a novel learning principle by itself. A comparison would need isolate what jointly inducing editable state interfaces adds beyond ordinary library refactoring, including mining/checking time. Only the relevant mechanism sections were read; D did not run DreamCoder.

## SOURCE: Maina et al., Synthesizing Quotient Lenses (ICFP 2018)

Author-hosted [paper](https://tupl.cs.tufts.edu/papers/synthesizing-quotient-lenses.pdf), local quotient2018.pdf, PDF pages 2,7,8,17 / printed 80:2,80:7,80:8,80:17.

Quotient regular expressions specify both formats and equivalence relations. Canonizers map into representative kernels; synthesis can operate between kernels and place canonizers at the edges. Equation 2 weakens round trips to the specified equivalences. The normalize combinator identifies equivalence through a surjective idempotent function's fibers. This attacks any claim that learning an encoding/equivalence plus transported edits is new in the abstract. In this source the relevant format/equivalence specifications are supplied; automatic discovery from downstream tasks is not demonstrated by the passages read. The normal-form theorem is reported, not independently verified here.

## What the comparison establishes

VERIFIED in the primary-source-comparison sense: D's algebraic core and most of its learning-loop organization have established precedents. OPEN: whether the joint editable-type discovery problem has a useful, sufficiently differentiated unsolved mechanism; whether a neural-guided system can solve it at lower total cost. Do not describe an incomplete literature search as proof of novelty.

The selected-page text files preserve the reading evidence. Pypdf emitted warnings about optional font decoding on these papers; equations central to the comparison were also checked against the browser PDF extraction. No source claim depends on an unread appendix.
