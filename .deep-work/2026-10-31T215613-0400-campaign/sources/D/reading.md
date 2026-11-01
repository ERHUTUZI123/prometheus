# D primary reading record

Read on 2026-11-01 UTC (2026-10-31 America/Toronto), before cross-branch exposure. Original food PDFs unchanged. Extraction text and rendered page checks are local evidence, not a substitute for the primary books.

## SOURCE: Hofstadter and FARG, Fluid Concepts and Creative Analogies (1995)

Substantive passages: chapter 4, PDF 179–182 / printed 169–172; chapter 5, PDF 215–223 / printed 205–213 and PDF 224–232 / printed 214–222; PDF 247–249 / printed 237–239. Partial page 250 was extracted but not used. The book says exact numerical formulas/parameters are in Mitchell (1993), not supplied in this chapter (PDF 222 / printed 212). I therefore do not claim to have reconstructed the original numerical simulator.

Mechanism reconstruction:

1. Workspace begins with raw letter tokens, alphabet identity and edge descriptors (printed 217).
2. Scouts inspect adjacent letters; a sameness bond can be proposed, assessed for support and then created by an effector (printed 221). A uniform bond fabric permits a group, which itself becomes an object that can receive descriptions and participate in higher groups (printed 218).
3. A bridge between left `aa` and right `kk` carries both sameness identity and a leftmost/rightmost slippage (printed 218–219). Building such a bridge activates the concept `opposite` (printed 220).
4. `opposite` labels several links, including right/left and successor/predecessor. Its activation shortens those links together, making the relevant conceptual slips more probable (printed 214).
5. Activated concepts spawn top-down scouts. Urgency changes their sampling rates; one codelet is sampled and removed each step. This makes representation formation and rule interpretation mutually influence each other (printed 221–222).
6. Worked length example: `abc -> abd; mrrjjj -> ?`. Same-symbol grouping yields lengths 1,2,3, allowing a successor interpretation over lengths and the possible answer `mrrjjjj` (printed 239). This is one interpretation, not a uniquely logically required output.

Decisive boundary: no new permanent Slipnet nodes or topology are created, and changes are not retained between runs (PDF 226 / printed 216, visually checked at pages/page-226.png). This changed my candidate from persistent Copycat-like activation dynamics to construction of executable recognizers and edit semantics. The representational objects alone cannot count as learned computational primitives.

## SOURCE: Herbert A. Simon, The Sciences of the Artificial, 3rd ed. (1996)

Substantive passages: chapter 5, PDF 143–148 / printed 131–136; chapter 6, PDF 153–158 / printed 141–146.

Exact computational example reconstructed: number scrabble has nine cards 1…9; a player wins with exactly three summing to 15. Simon maps cards to the magic square `4 9 2 / 3 5 7 / 8 1 6` (PDF 143 / printed 131, visually checked). Its eight rows/columns/diagonals are exactly the eight triples summing to 15, so a familiar tic-tac-toe strategy transports through the map (printed 132). This proves an isomorphism for this finite game; it does not explain automatic discovery of the map or strategy.

Representation changes what is operationally apparent, but Simon explicitly says there is no systematic theory here for generating effective representations (printed 131). The State Department communication example (printed 143–144) changes the bottleneck from printer throughput to human attention; optimizing the first can worsen the true task. This changed the candidate's account: charge construction and checking costs, not only the short resulting program. The goal must not quietly change from learnable computation to pretty compressions.

## SOURCE: Abelson, Sussman and Sussman, SICP, 2nd ed. (1996)

Local PDF is a 521-page reflowed electronic edition with no reliable printed-page numbers. Exact references: chapter 4 introduction PDF 307–309; section 4.1.7 PDF 332–334; section 4.3.3 PDF 360–368. PDF page 365 (assignment) and 367 (amb choice) were visually checked.

Load-bearing mechanism reconstructed:

An ordinary analyzed expression maps environment to value. The amb evaluator's analyzed expression instead receives `(environment, succeed, fail)`. On success, it calls `succeed(value, next-failure)`; on dead end, it calls `fail()`. An `amb(a,b)` execution passes branch a a failure closure that tries b, and the last branch inherits the incoming failure closure (PDF 367). Assignment captures the old binding and passes a failure closure that restores it before invoking the prior failure (PDF 365).

Worked causal trace (RECONSTRUCTION, not a reported book experiment): initialize `x=0`; choose k from `[1,2]`; assign x=k; require x=2. First branch sets x=1; requirement invokes its failure closure; the closure restores x=0 then tries k=2; second branch sets x=2 and succeeds. If the caller requests another result, the next-failure restores x=0 and exhausts choices. Without the assignment-specific failure closure, speculative updates leak across branches.

What changed: a computational primitive's meaning includes how it updates and reverses state, not only its input/output label. A proposed learned representation therefore needs executable writeback and explicit effects. SICP implements new semantics by human-written evaluator changes; it does not supply a learning law for discovering them.

## SOURCE: John R. Koza, Genetic Programming II (1994)

Substantive unfamiliar sections: chapter 22, PDF 599–605 / printed 577–583; chapter 23 PDF 620–621 / printed 598–599; chapter 26 PDF 641–643 / printed 619–621 and PDF 661–663 / printed 639–641. PDF 599 was visually checked; some table OCR is poor, so no exact table-result claim rests on unverified OCR.

Load-bearing primitive-construction reconstruction:

Each individual can contain primitive-defining branches, ADF definitions and a result-producing branch. A k-argument primitive is represented by 2^k Boolean output leaves. Function name, arguments and the nested IF decision structure are invariant; truth-table leaves vary. Initialization chooses branch number/arity within prescribed maxima and independent Boolean leaves. Type-preserving crossover can replace truth-table entries with Boolean constants while call sites can change separately (printed 577–580). For k=2, leaves corresponding to inputs 00,01,10,11 equal `[1,1,1,0]` implement NAND; nested IF evaluation traverses the input tests to return the corresponding leaf. Editing the 11 leaf to 1 makes the same callable primitive constant true and changes every live call site's semantics. This is actual primitive evolution inside a fixed interpreter, not only module selection.

The chapter's reported success frequencies are SOURCE claims, not our independent replications. In chapter 26 the lens effect is a change in the distribution of scores of randomly generated programs with ADFs; the chapter deliberately considers initial random populations, not the causal effect of later learning (printed 619,640). Hence a promising initial-search histogram is neither learned representation nor proof of evolutionary advantage.

What changed: do not claim that learned function definitions alone are new. The candidate must explicitly say which source-state equivalence, object addresses and update semantics are learned, and it must pay the search debt.

## Files and pagination checks

Extracts: fluid-selected.txt, fluid-mechanism.txt, simon-design.txt, simon-representation.txt, sicp-evaluators.txt, gp2-primitives-representation.txt. Readability was verified directly in extraction output for the passages above. Rendered pages live in sources/D/pages/. Printed pagination visually confirmed for Fluid 216, GP II 577, Simon 131; SICP has none in this reflowed edition.

No website or secondary report was used to infer these book claims. No other branch's work was read.
