# Branch C: first attack and constructive repair

This follows the independent v0 freeze. Candidate-v0.md remains unchanged. No other branch or old architecture report was read. The source-comparison summaries below are deliberately narrow; primary passages were inspected, but the full papers were not read end to end.

## Executed prerequisite attack

Reproduce: `python3 .deep-work/2026-10-31T215613-0400-campaign/probes/C/stabilization_attack.py`.

Complete deterministic output: `probes/C/stabilization_attack_results.json`.

1. **EXECUTED OBSERVATION:** identity and sign-reversal maps on {-1,+1} have identical mean outputs but functional MSE 4. A mean-activity homeostat has zero error signal despite complete reversal of the computed relation. This attacks a simple neural homeostasis analog, not the richer learned trace energy specified in v0.
2. **EXECUTED OBSERVATION / RECONSTRUCTION:** for y=a*m+d, dm/dt=1-y, a=0.2 and d=0.4, the bounded Euler run ends with error about 2.72e-5 after 50 time units. Reversing the response sign a=-0.2 gives error about 13,202.67. With a=0.2,d=0 and |m|<=2, error remains 0.6. Stability and control range are independent assumptions; increasing integral pressure cannot repair the wrong sign or unavailable actuator range. These are simple checks, not validation of the candidate's nonlinear multiplier rule.
3. **EXECUTED OBSERVATION:** a leaky memory m_next=0.999*m after writing 1 passes a 2% retention tolerance at delays 1,2,4,8,16, but loses 63.23% at delay 1000 and 99.995% at delay 10000. This attacks an implication from accepted short traces to general functional preservation. It is not evidence that all learned contracts fail, and no actual learned critic was trained.

**Judgment:** the biological analogy does not supply a sufficient learning signal. The proposal remains viable only if it learns and tests enough behavior to distinguish materially wrong computations, and only inside a measured reconstruction range.

## Closest checked mechanisms

### DreamCoder: most of the library-and-search loop is established

Primary: Ellis et al., *DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning*, PLDI 2021. https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf . Inspected sections 2-2.1 (PDF p. 3), 3.2-4 (PDF p. 7) and evaluation baseline passage (PDF pp. 9-10).

**SOURCE:** it jointly learns a reusable program library and a neural search policy, mines abstractions from solved programs, and trains search on replayed and generated tasks. It explicitly compares against adding whole solved programs to the library. Its corpus assumptions require useful abstractions to recur across easier and harder problems.

**ATTACK:** “a learned operation shortens future search” is already an implemented mechanism here. That broad claim is not a contribution of branch C. The remaining candidate difference is preserving an operation through learned behavioral stabilization while allowing its implementation to change jointly with its embedding. Whether that difference has any advantage remains OPEN. A frozen-library-plus-adapters baseline is mandatory.

### Growing Neural Cellular Automata: learned regeneration is established

Primary: Mordvintsev et al., *Growing Neural Cellular Automata*, Distill 2020. https://distill.pub/2020/growing-ca/ . Inspected Model/Cellular Automaton rule, sample-pool pseudocode, and Experiment 3.

**SOURCE:** local neural rules update a multichannel lattice; external training loss rewards a target pattern. A pool of previous states trains persistence; damage augmentation expands the recovery conditions. Regeneration can vary substantially among models, and training for growth alone need not ensure recovery.

**ATTACK:** retaining a neural process that regenerates an outcome is not new. The branch's remaining interaction is between *acquired computational operations*, learned acceptance constraints, and reconstitution during later composition. Pattern reconstruction alone would not establish this interaction or computational innovation. It would be useful scaffolding only if a circuit's task behavior, rather than a desired picture, is reconstructed and then runs autonomously.

### Developmental Symbolic Encoding: identified but not substantively checked

Search found *Evolving scalable and modular adaptive networks with Developmental Symbolic Encoding* (2011), https://pmc.ncbi.nlm.nih.gov/articles/PMC3161195/ . The open request hit an access check. Its mechanism is therefore an unread prior-art lead, not an established comparison in this report. Same for HyperNEAT variants and learned conditional synthesis/energy models. Novelty remains OPEN.

## Repair v0.1: turn a contract into an actively challenged finite acceptance process

**AGENT PROPOSAL / HYPOTHESIS:** retain the old learned operation R as a temporary executable teacher during capture, and use *accepted-but-wrong reconstructions* to improve its stabilization criterion. This strengthens the candidate's existing finite-trace repair without inserting a symbolic semantic oracle.

State adds an operation-specific trace set T_i and a pool of reconstructed candidate implementations. For a candidate G that the current energy accepts, search for a boundary input trace U maximizing

  discrepancy(F_G(U),F_R(U)) - beta*input_cost(U)

under the same input range, reset rules, horizon cap and transition budget. Inputs can be optimized numerically when differentiable or sampled by a small policy trained on discrepancy. The teacher is an *already learned circuit*, not an external solver of the new task. If a discrepancy exceeds tolerance, add that paired teacher trace to T_i and train the energy to reject the faulty reconstruction. Charge both executions and all adversarial search.

Then reconstruct again from an independently perturbed seed. The acceptance process stops at its budget, including failure if no satisfactory reconstruction exists. It makes a conditional finite claim: no counterexample found within the checked trace distribution/budget. It never asserts exact equivalence or unlimited retention. Freeze the acceptance set before an independent evaluation horizon/embedding set is exposed.

Why it is constructive: this lets a useful yet initially broad critic improve from actual errors. The learner need not discover a perfect memory law from the first handful of traces. A long-delay counterexample supplies a new boundary distinction; a changed embedding can supply another. The mechanism can develop progressively, rather than being rejected for lacking complete semantics at birth.

Why it may still fail: adversarial trace search has its own discovery problem; R's distribution can be inadequate; the critic may grow to copy R; horizons may be prohibitively costly; replacing R with a critic plus builder may never amortize. This repair is closely related in spirit to counterexample-guided synthesis and active testing. It is not claimed novel, and the proposal's specific neural/compositional benefit remains to be measured.

## No new experiment claimed

The only executed computation in this branch is the small prerequisite attack. There is no trained neural architecture, learned operation boundary, learned critic, successful graph reconstruction, comparison with DreamCoder, or measured gain in subsequent discovery. The operational trace in v0 is explanatory arithmetic. Keep that separation in any synthesis.
