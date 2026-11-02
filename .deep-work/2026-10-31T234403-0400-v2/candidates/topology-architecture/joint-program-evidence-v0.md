# Candidate v0 — Conservative learning over joint executable programs

**AGENT PROPOSAL / HYPOTHESIS.** Same V2; MAIN remains human-locked TOPOLOGY. This is one constructive candidate, not a new Method, roadmap, result, or experiment authorization. No code, training, core module, or canonical Method was changed for this proposal. The latest acceptance criterion permits established components if the resulting architecture delivers an important, reproducible advantage; originality is not assumed or used as an admission gate.

## 1. The consequential limitation to change

The current learner optimizes a smoothed distribution of executions while deploying a deterministic neural mean. It also represents uncertain calls through products of local soft mixtures. Neither an arithmetic mean of loop coordinates nor independent marginal call choices necessarily denotes a program supported by the successful executions. More terminal-gradient iterations can therefore move a numerical surrogate without accumulating a coherent reusable computation.

The new candidate changes the **persistent learning state**: retain a correlated distribution over whole executable programs, and retain the evidence for those programs independently of the mutable neural proposal network. Monodromy supplies certificates under which continuous implementation changes leave that evidence valid. New program hypotheses may inherit old evidence only through a checked conservative refinement.

If successful, this could reduce repeated discovery and replay when new tasks extend an ambiguous earlier mechanism, while allowing shared controller weights to keep learning. The intended benefit is acquisition/retention under a complete cost vector, not merely lower surrogate loss or less neural P achieved by hiding programs outside the model.

**Existing evidence boundary.** Actual current controls all succeed in the fixed 30P diagnostic: ordinary event32 transactions, chamber-plus-head learning15, program search10. Their final held-out native accuracy is1. These are read from `probes/topology-architecture/runs/parameter-hybrid-conventional-controls-v1/learn/result.json`. They do not support an independent acquisition benefit for the present coupled rule. This proposal preserves that result and does not spend another budget on tiny chamber variants.

## 2. Actual state and execution rule

The proposed module state is

\[
S_t=(\theta_t,\mathcal A_t,\mathcal H_t,s_t,\mathcal K_t).
\]

- `theta`: neural input/call proposal, loop controllers, and readout parameters.
- `A`: actually acquired executable atoms. Each atom contains a loop source/version, a paid endpoint observation, and its current protection contract. Reverse paths implement inverse actions where the interface permits them; they are not cost-free physical execution.
- `H`: a bounded set or factored DAG of **complete program hypotheses**. A hypothesis specifies the same operation choice whenever a shared symbol/port is reused, the call sequence, and the relevant hard input/readout binding. It is not a new independent draw at every occurrence.
- `s(P)`: accumulated native-evidence score for each supported hypothesis. These adaptive scalars count as learned state; their storage and parameter-equivalent capacity must be reported.
- `K`: executable-domain, old binding-closure, and finite-length soft/native contracts, plus conservative-refinement receipts.

For context `c`, choose one supported complete program

\[
P_t^*(c)=\arg\max_{P\in\mathcal H_t(c)}q_t(P\mid c)
\]

with a fixed tie rule, then execute that actual program on the input state. The model never executes mean coordinates of mutually exclusive programs. An inference sample from `q` is also coherent, but deterministic MAP and stochastic prediction must be reported separately.

All program codes, scores, controller state and receipts belong in the actual model checkpoint. They must not be a successful graph sidecar disconnected from the deployed neural module. Amortization can later predict a program from raw input/context, but its old input--program binding still needs protection. The current public-task-context benchmark makes `c` explicit to every method; a no-task-ID recognizer is a separate unresolved interface.

## 3. Native evidence update, not soft-mixture evidence

Use a fixed observation model on the **native** prediction. For classification with `N` labels, one possible declared model is

\[
\lambda_\epsilon(y\mid\widehat f_P(x))=
\begin{cases}
1-\epsilon,&y=\widehat f_P(x),\\
\epsilon/(N-1),&y\ne\widehat f_P(x),
\end{cases}
\qquad 0<\epsilon<1/2.
\]

Given one new terminal example `(x_t,y_t)`, update

\[
s_{t+1}(P)=s_t(P)+\log\lambda_\epsilon(y_t\mid\widehat f_P(x_t)),
\qquad
q_{t+1}(P)=\frac{\exp s_{t+1}(P)}{\sum_Q\exp s_{t+1}(Q)}.
\]

This is ordinary likelihood/exponential-weight updating, deliberately used as a component rather than claimed as new mathematics. Each distinct external terminal label is charged once when acquired; every program execution and repeated label evaluation is separately charged. The score is based on the fixed native observation model, not a neural soft probability whose value changes after unrelated gradient updates.

The neural proposal network `pi_theta(P|context, observations)` is trained to amortize the retained program distribution:

\[
L_{\rm proposal}(\theta)=-\sum_{P\in\mathcal H_t}q_t(P)\log\pi_\theta(P\mid c,\xi_t).
\]

`xi_t` contains permitted observations/history, not hidden teacher states or desired routes. A proposal loop still requires actual execution and a valid receipt before becoming an atom. Training the proposer does **not** replace `q` by its changed prior or silently recount earlier data. The canonical evidence state is distinct from the mutable search heuristic.

## 4. The topological bridge: evidence transport under shared learning

Shared controller updates can be trained by the current terminal objective and boundary estimator, but they must preserve the classes of acquired atoms and the required old native bindings. These are the existing, paid contracts, not an assumption that local updates preserve a task automatically.

If all atom actions used by a supported program remain fixed and its input/readout binding is preserved, its native prediction is fixed for every protected old word. Therefore its earlier native log-likelihood terms are fixed. Its accumulated `s(P)` is still valid after the shared neural implementation changes, even though the proposer, soft confidence and loop geometry may all change.

The useful compositional point is that **one atom-level certificate protects the evidence of every program built from those atoms**, including retained counterfactual programs. It does not require reexecuting every old training example for every hypothesis. Native relation closure and finite-length soft confidence remain separate obligations. Numerical errors, changed atom classes, or changed observation models invalidate this argument.

That statement requires a **common marked input/readout interface**, not just the present winner's task-level closure. A concrete sufficient rule keeps old hard encoder/readout maps fixed while their logits learn within those hard-choice chambers. A more permissive rule supplies one relation that contains the old encoded input pairs, agrees on decoded outputs, and is closed under **every retained atom**, rather than only the winner's selected token transitions. Induction then covers every candidate word over that atom set at once. Passing the current winner's weaker closure alone is insufficient and must not silently authorize evidence reuse. The stronger bank-level relation and any loss of plasticity are additional charged costs.

This is the candidate's concrete integration target: a learning system may alter a shared implementation while keeping an inspectable distribution of computational explanations and its evidence semantically stationary. It is stronger than maintaining one current prediction score, but its advantage over ordinary immutable program libraries or equally strong semantic constraints is unproved.

## 5. Growing hypotheses without inventing missing history

Arbitrarily introducing a new program after deleting old labels does not provide its old likelihood. The following rule makes the required information explicit.

Let a proposed refinement provide a parent map `r:H_new→H_old` and a checked contract

\[
\widehat f_{P'}(x)=\widehat f_{r(P')}(x)
\quad\text{on the old protected observation domain}.
\tag{C}
\]

The domain must be inspectable: for the current finite program interface it can be all old encoded starts and words over the admitted alphabet, certified by atom classes plus native binding closure. Newly exposed ports/tokens may differ. A finite list of matching outputs alone does not establish the universal contract.

Choose an explicit normalized conditional mass `kappa(P'|P)` for children and lift

\[
q_t^-(P')=q_t(r(P'))\kappa(P'\mid r(P')),
\qquad
\sum_{P':r(P')=P}q_t^-(P')=q_t(P).
\]

In the stored log-score representation this is `s_minus(P') = s(parent) + log kappa(P'|parent)` up to the shared normalization constant, followed only by genuinely new likelihood terms.

By (C), every child has the parent's accumulated old log-likelihood. Splitting the parent's prior mass is therefore consistent with all retained old evidence. New terminal examples can distinguish the children. The proposal distribution may be learned, but its use is recorded as an adaptive search/refinement choice; no fixed-prior Bayesian consistency claim is made for arbitrary data-dependent `kappa`.

Children without a valid parent contract must obtain the missing old information by retained replay, new permitted queries, or an explicitly narrower guarantee. Local plasticity, a timestamp or a favorable new loss cannot supply it. This restriction is essential: the mechanism is not a way to learn arbitrary new functions from evidence that was erased.

Program hypotheses can be coalesced only when their **declared executable interfaces** agree. Equal final output on one task is insufficient if future tasks can access different internal ports. Geometry aliases with identical observed atom actions and the same complete binding can share a score; neutral alternative factorizations that differ at reusable ports must remain distinct or be retained as conditional structure.

## 6. Worked conceptual example

This is a finite hand construction, not a measured result. Use the fixed cubic cover, identity-class loop `gamma_1` and transposition-class loop `gamma_3`, with actions `I` and `A=(01)`. A new two-slot macro has four complete hypotheses:

\[
(I,I),\ (I,A),\ (A,I),\ (A,A).
\]

Start with a uniform joint distribution. A terminal example requires the macro to map sheet0 to1. Under noiseless consistency, only `(I,A)` and `(A,I)` survive, each with weight1/2. Under the fixed noisy model they receive the corresponding high posterior mass.

Both surviving **whole programs** execute `A` correctly. Their factor marginals remain uniform. Independently taking marginal argmaxes can choose `(I,I)`, which is wrong. Averaging the corresponding raw scalar loop coordinates gives `(2,2)`, exactly on the supplied discriminant. A joint MAP program instead chooses one of the actually supported regular computations.

This identifies a definite representation/update failure of means and factorized modes; it does not show faster discovery than program search. Indeed, ordinary joint search solves this four-hypothesis example.

If a later task accesses one of the reusable slots, the two explanations cease to be interchangeable. Their correlation must survive. If a previously admitted macro `A` is refined into these two children, both old macro predictions are identical to the parent; contract (C) lets both inherit old evidence without replay. New feedback about the exposed slot supplies the information that chooses between them. No old observation can determine that choice by itself.

New atoms can likewise be proposed from unadmitted loop coordinates using terminal feedback, followed by actual class identification. Their insertion into protected old programs requires an explicit conservative wiring contract. The example does not claim that the system has already invented a new operation, grown state capacity, or solved arbitrary irreversible tasks.

## 7. Full learning process

```text
state: neural module theta; acquired loop atoms/receipts;
       joint program DAG H with adaptive log weights; old contracts K

for each new allowed terminal experience:
    execute supported complete native programs (bounded work)
    add fixed native-evidence log likelihood to their scores
    retain joint correlations; never replace them with coordinate means

    if a bounded new hypothesis/refinement is proposed:
        execute/identify any new loop atoms; charge discovery
        prove the old-domain parent contract, or buy missing old information
        split parent mass explicitly; inherit only justified old evidence

    train the neural proposer from the retained weighted programs
    propose shared controller/E/R updates
    check atom class, old native binding, and soft/native contracts
    commit only valid actual module updates; retain all rejected work

    deploy one joint supported program; serialize the actual module state
```

Candidate generation is an unresolved search problem, not the phrase “automatic discovery.” An implementable first version can use the existing loop-coordinate proposer, paid branch observations, and bounded program edits. It needs a declared support cap; dropping low-weight hypotheses then becomes an approximation whose lost mass and failures are recorded. There is no assertion that finite particles solve exponentially rare discovery or that an empty beam proves the representation impossible.

## 8. Nearest strong alternative and the real remaining difference

**SOURCE checked, not a landscape survey:** Ellis et al., *DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning*, PLDI2021, original [author-hosted paper](https://people.csail.mit.edu/asolar/papers/EllisWNSMHCST21.pdf), printed pp.836–839, §2.1–2.4 and Eqs.1–3. It already uses whole-program latent hypotheses, finite program beams, learned neural search and library refactoring. Therefore joint program beliefs, program abstraction, and amortized search are not new contributions here. The original equations and body were read; code and full empirical reconstruction were not.

An ordinary program learner with immutable primitives already has stationary old likelihoods. Give that control joint beliefs, conservative rewrites, the same observations, caches, proposal budget, storage, and allowed certificates. It may match the candidate completely. The proposed experimental distinction is whether **mutable shared learned implementations with class-level certificates** deliver useful new acquisition/implementation learning without paying the cost of either freezing separate implementations or rescoring/replaying prior experience.

A second strong control is the same joint-program learner using ordinary endpoint/behavior constraints instead of monodromy certificates. If it matches the complete effect and cost, the benefit belongs to program-consistent learning or generic semantic protection, not independently to topology. Under the newest human criterion, the full organization may still be useful if it demonstrates important broad competitive advantage; this artifact predicts neither that advantage nor its attribution.

## 9. Cost, fragility, and decision boundary

Count neural weights, adaptive log weights, discrete program code and its description length, atom geometry, parent/refinement receipts, caches, optimizer state, retained replay, every native execution, class identification, and verification. A posterior is model memory, not free external bookkeeping. A factored program DAG can share structure, but no subexponential search or general inference bound is assumed. A certificate's cost can still grow with atoms, input domains or state size.

The weakest step is constructive refinement: can the learner propose useful new computational explanations that preserve the old contract cheaply enough, rather than retaining a large useless beam or simply implementing ordinary program synthesis with an extra executor? The mean/parity example settles only representation consistency. It does not settle this question.

The next discriminating development, if Root accepts the candidate, should address this step inside the existing larger continual-composition tasks, with joint ordinary program learning and frozen/modular/replay controls. Repeating the cubic scalar example or proving the score-transport identity again would not answer it. No experiment or main-architecture change is started by this artifact.

**CLAIM:** A concrete joint-program learning state and conservative evidence-transfer rule can connect monodromy-protected shared implementation updates to retained computational uncertainty.

**STATUS:** Explicit proposed algorithm; finite worked derivation; no implementation, acquisition result, cost advantage or verified novelty.

**EVIDENCE:** Current ordinary controls; the hand joint-versus-marginal example; conditional score/refinement identities; checked DreamCoder primary equations.

**WEAKEST LINK:** Useful, cheap refinement/discovery under fair same-information program-learning alternatives.

**FAILED INTERPRETATIONS:** Coordinate averages are not executable semantic averages; posterior memory is not free; deleted labels do not score arbitrary new hypotheses; ordinary whole-program inference is established prior work; strong current controls remain successful.

**NEXT SINGLE DECISION:** Whether this evidence-retaining organization warrants a bounded constructive implementation in TODO2 alongside the separate general-Block work; Root has not yet adopted it as the canonical Method.
