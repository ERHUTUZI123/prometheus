# K v2 technical receipt

## Main current mechanism

Construct a compact executable representation of a module's learning transitions, preserving both present computation and all future updates under a declared action interface. In the exact factorized-matrix instance, hidden plasticity state survives compaction. A neural constructor could subsequently edit bounded plasticity state without changing current outputs, and learn such edits from later task performance. That neural/meta-learning stage is specified but unexecuted.

## Frozen artifacts

- `candidate-v2.md`: SHA256 bc2cbfefd1c9ef3a976b96fce18510aa83a8ecee28afe5525788f945ac443277; frozen 2026-11-01 04:15:15 UTC.
- `probes/K/plasticity-test-contract.md`: SHA256 a2a55585caf15121a083def3db580261c553742b1f63b77131db084b0ba6efe2.
- `probes/K/check_plasticity.py`: SHA256 ded3eb7d0ded190df4ed8d8d252a393bc2ccb7c225c47d1f9f8dc616c1a99727; inspected and frozen with final contract at 04:26:00 UTC, before the only run.

## Execution and actual construction

The inspected stdlib test ran once through `probes/infrastructure/run_isolated.py`, external wall/CPU timeout 45 seconds, network denied, scrubbed environment, writes restricted to probes/K. Exit code 0. No failed runs, retries, tolerance tuning or post-result code changes. Serialization smoke preceded calculation. Completion was observed by 04:26:36 UTC.

The generic polynomial routine receives source update programs and present output observables. It does not receive the expected Gram sufficient statistics as candidates. It expands pullbacks, groups coefficients of free update controls, adds linearly independent coefficient functions, and continues until closure. Expected dimensions appear only as assertions after construction, not as target features or a search filter.

Results:

1. Scalar three-path source: six parameter variables -> two generated observables, nine total monomial terms in the retained statistic programs. The missing squared-norm statistic was constructed from pullback coefficients.
2. Matrix 2x2 width-2 source: four output observables -> ten closed quadratic observables. The tenth is introduced only by pulling back an already inferred plasticity statistic, from a quadratic control coefficient. Its polynomial is v00^2+v01^2. Thus first output-response closure was not sufficient. This small width-2 case has eight source variables and ten algebraically constrained observables; it is a discovery/correctness example, not a storage reduction at that width.
3. Independent code paths for direct factor SGD and the manually derived block-state laws match exactly after every action in four-step replay, including a physical function-preserving growth from width six to seven. Rectangular boundary replay also matches. In this replay the factor parameter count is 24 then 28, while the symmetric compact matrix has ten independent stored entries. No runtime speedup is claimed.
4. Same current function and same complete initial first-order response do not imply the same finite-step learning future. The specified witness has equal W1, then W2[1,2]=-37/32 versus -77/64. Exact difference -3/64.
5. Direct factor/velocity momentum and augmented-state congruence agree after all four actions. Deliberately using an already-updated factor in the second factor update changes the output and is detected.

The inferred JSON update programs are symbolically closed under the source pullback routine. Numerical replay in this test uses the separately derived block formulas, not a numeric interpreter of those generated JSON programs. A fresh independent rebuilder can usefully execute the generated program or reconstruct from micro laws; do not describe this as already independently rebuilt.

## Raw evidence hashes

- `scalar-closure.json`: 9232de9e9f0fe2638f9e102d5e6ff2b7ab2f18702936fcdba82055b3b63f85dc
- `matrix-closure.json`: 559ada1ed1a20fb969141e201cc11cae736b45a7fec57eca31a0b54a66c5aef9
- `plasticity-raw.log`: 586a81d98ca1e4f5e3c012423991f08272026f69796cc1473a820dd75eb005ab
- `plasticity-result.json`: 2a94ee16a1ae295cfb36de52c318808085051832717ff7b7bee975fa5c07be2b

## Source and inference boundaries

Covariance learning dynamics, functionally invisible plasticity variables, function-preserving neural growth and learned preconditioners have close prior art; see `sources/K/source-record-v2.md`. No broad novelty of those ingredients is claimed. The local test supports automatic symbolic construction of a closed update state in the supplied polynomial class and exact finite examples of its behavioral importance.

It does not show a neural policy learned compaction, a meta-policy learned useful plasticity allocation, a task-learning improvement, broad nonlinear-network compaction, immunity to forgetting, or paradigm-level scientific value. The exact scope is redundant factorized linear blocks under specified shared-linear update laws, including placement inside a nonlinear surrounding network that respects the interface.

The live invention is the coupling of mechanism construction, discovery of learning-sufficient state, compaction preserving future edits, and learned plasticity construction/reuse. Its hardest missing step is finding useful compact closures and learned construction actions for richer mechanisms without making state/discovery costs as large as the original implementation. Existing algorithms are legitimate components; the actual established comparators, rather than a hypothetical baseline endowed with the entire proposed coupling, determine any later scientific gain.

## Critical interface for independent checking

`micro-law-input.md` supplies only source laws and optional source extensions. It omits this macro-state answer and all expected dimensions/numerical outcomes. Coordinator derivations shared during this turn are same-model convergence, not verification. The coordinator plans separate checks; those are not included among K's completed evidence.

CLAIM | STATUS | EVIDENCE | WEAKEST LINK | NEXT SINGLE DECISION

Finite learning-state closure can preserve behavior absent from current-function/initial-kernel compaction | VERIFIED in stated symbolic/finite tests plus explicit reconstruction | frozen code, inferred programs, raw log | independent rebuild pending | rebuild from neutral micro laws.

Neural systems can learn to discover and construct useful compact learning mechanisms | HYPOTHESIS | v2 organization and explicit adaptation law | learned discovery/selection beyond the known exact class unexecuted | continue constructive mechanism development after independent checking, with actual comparators and costs.
