# Mechanism v1: use a continuation witness to train a new memory mode near the loss of information

**AGENT PROPOSAL / MANUAL CONSTRUCTION; no experiment or training.** Same V2 and the one protected hypothesis selected by root: task-feedback-driven formation of persistent internal dynamical modes when current state aliases future-relevant histories. Read updated AGENTS §5.1a and WORKFLOW §§6.1/7.1. Root owns `candidate.md` and allocation decisions; this worker does not edit them or make a MAIN decision. Provider-effective mode unknown. Online remains PARK and topology/coded-closure results remain sealed.

Before the user interruption there was only a directional sketch and a read of the continuation contract. No mechanism artifact, new original reading or experiment had occurred. The construction below was developed after root selected this hypothesis. The root's RCC original-source pointer arrived after the local-contrast derivation began but before this artifact/source record. It is recorded as external input, not pristine independent source selection.

## 1. Important unknown and proposed constructive move

A learner can lack a state distinction needed by a new task. Adding a recurrent unit is established, but its terminal error may have to traverse a long, contracting hold interval before it can train the earlier write. A persistent attractor does not solve that credit problem: information can survive in its basin while derivatives of its value with respect to old write parameters vanish.

The proposed move is to extract **an intermediate separation constraint from observed task feedback and a matched future continuation**. It does not receive a latent bit or an intermediate correct state from a teacher. If two retained histories have the same full incumbent state, the same remaining inputs/task context, and different deterministic endpoint labels, then a correct deterministic extension must distinguish those histories at that checkpoint. Train the new recurrent mode at the earliest recorded merger using the short prefixes; then train its carry dynamics using one-step paired separation constraints along the observed common suffix. Full terminal replay remains decisive for read/update usefulness.

This is an explicit candidate loss/formation rule. Its useful effect beyond RCC-style residual correlation or a strong gated recurrent learner is **OPEN**, as is efficient discovery of such witnesses in general tasks. The protected development question is whether the derived separation evidence can actually train write/hold/update/read without a supplied hidden memory label. The first construction supplies a concrete route and a discriminating credit consequence, not a completed general learning theory.

## 2. Exact feedback and a legitimate alias witness

Let the frozen incumbent have state `h_t=F(h_(t-1),x_t)` and old readouts D_old. The full state includes every variable available for future computation: recurrent registers, explicit clock if present, task context and any controller memory. Equality of a rounded plot or a readout is not equality of full machine state.

Retain completed episodes `(x_1:T,q,y)` and either actual incumbent state traces or enough raw input to replay them. All storage, teacher queries, resets and re-execution count. Search for two actual episodes with aligned checkpoints tau such that:

1. Their full incumbent states at tau are identical, under the stated exact/finite-precision semantics.
2. The entire remaining input word `w=x_(tau+1:T)` and available future task context are identical.
3. Their deterministic endpoint labels differ.

Call this pair a continuation witness. If a deterministic predictor uses only the incumbent state and that common future, it must give the same answer to both, so it cannot be correct on both. For targets +1 and -1, any common prediction c has mean squared error `((c-1)^2+(c+1)^2)/2=c^2+1>=1`.

For near states, a label difference alone is not an exact alias proof. One may instead state a bound using an explicitly bounded read/transition sensitivity, or use the pair as a heuristic. Noise in labels can generate apparent contradictory pairs without missing memory; repeated observations/conditional-output differences need their own statistical evidence. No noisy-label contradiction is treated as proof here.

For a pair whose raw inputs have a common suffix, replay/inspect the actual old trajectories to locate the earliest merger followed by identical remaining inputs. This moves the constraint to the first documented loss of a necessary distinction. The search is not free: a naive comparison of M stored length-L episodes can cost O(M^2 L); suffix indexing can reduce search overhead but still stores/processes the actual words and states. There is no generic promise that informative pairs occur frequently.

**Information boundary:** the past is not reconstructed from the aliased h. New candidate states must be initialized at the episode start and rerun on retained raw prefixes, or trained on new fresh episodes. If histories were discarded and cannot be replayed, the missing distinction in an already-running episode is unavailable. Self-inspection does not evade the earlier point-oracle information audit.

## 3. State, transition and candidate formation

Add a provisional scalar mode r, initialized to zero for each episode, while leaving incumbent dynamics untouched:

`h_t = F(h_(t-1),x_t)`,

`r_t = tanh(alpha * k_theta(chi_t) * r_(t-1) + u_theta(chi_t))`.       (1)

Here `chi_t=(x_t,h_(t-1),retained_older_modes_(t-1),q_if_observed)` contains only currently available inputs/state. k is a bounded neural gate in [0,1], u is a bounded neural drive, and alpha has a declared finite range. Their widths, precision and parameter budget must be frozen before a process. Signed drive, nonlinear recurrence and gating are supplied building blocks; their task meanings, weights and admission are to be acquired. No prefix ID, endpoint label or future word is an input to (1) during execution.

The gate permits a learned distinction between carry, overwrite and clear: a large effective recurrent gain can hold a mode, while suppressing k permits u to install a new value or zero it. Those behaviors are representable, not already learned. The first worked subfamily below uses k=1 and u=beta*x; it does not establish acquisition of a reset gate.

Candidate trigger: a recorded continuation witness plus nonzero current-task error proposes a new mode. It is stronger evidence for a missing distinction than an ordinary training plateau, although it does not prove that one scalar mode suffices or that expanding the network is the best intervention. A finite candidate pool may vary alpha/gate initialization and feature connections; every candidate's construction, replay and failure is charged. Root will set process bounds before any execution.

A new readout for the new task may use `(h_t,r_t,observed q)`. Old readouts continue using the old h/program. The addition is block-triangular: r never feeds back into h or an older frozen mode. Thus the mathematical old computation is unchanged for every input while the addition remains one-way. This retains old program semantics, not a claim that a shared mutable readout cannot forget.

## 4. Learning rule derived from the witness

### 4.1 Seed write: train at the early merger, not only at the endpoint

Choose one seed witness pair. Give its two candidate states at the merger an arbitrary consistent orientation, e.g. orient the pair by ordered endpoint labels and request `r_tau^+>=m`, `r_tau^-<=-m`, for a declared margin 0<m<1. The **need for separation** follows from task observations; the ± orientation and scalar margin are a designed representational convention. They are not claimed to be the true latent bit or an externally supplied subgoal.

A seed loss is

`L_seed = ((r_tau^+ - m)^2 + (r_tau^- + m)^2)/2`.             (2)

Compute its true derivatives by replaying the two actual prefixes through (1), with BPTT/RTRL only through those prefixes. This can still be difficult if the earliest merger has a long or complex prefix. The method removes the **observed common suffix** from this write-training path; it does not remove all temporal learning difficulty.

For other witness pairs, use an orientation-free separation loss rather than assign globally consistent ± labels by fiat:

`L_pair(t) = max(0, 2m - |r_t^i-r_t^j|)^2`.                 (3)

At a completely collapsed pair the absolute-value symmetry gives no useful preferred direction. Equation (2) or a declared nonzero candidate initialization breaks that symmetry for the seed; do not claim (3) alone escapes exact zero. Multiple witnesses may need more states/modes or incompatible scalar conventions. This is a concrete remaining allocation/learning issue, not automatically solved by the loss.

### 4.2 Carry: use one-step local contrast on the observed common suffix

Once the seed produces a distinction, replay the common suffix. At every later aligned time before the endpoint, the two histories still have the same remaining observed continuation and different endpoint labels, so a correct deterministic state must remain distinguishable there too.

For carry training, feed each candidate's **actual replayed** previous state to one step of (1), stop the derivative at that previous state, and differentiate (3) through that one transition. This is the exact gradient of a conditional one-step auxiliary loss. It is **not** the exact gradient of terminal loss or the full unrolled contrast objective, and it must not be reported as such. Recompute candidate trajectories after updates; detached states are not supplied correct hidden states.

A pair-distance constraint allows the internal orientation to change; it does not impose that a hidden semantic bit always has the same sign. The positive-self-gain scalar family is a persistence bias. More complex task updates may need gates or additional modes; orientation-free contrast alone does not teach the right transition among all relevant histories.

### 4.3 Terminal replay trains read/update and rejects spurious separation

Interleave (2)/(3) with full actual task loss on retained/new complete episodes. Fit the new readout from terminal labels. Include episodes with changed suffixes, new writes and irrelevant input variations when those are actually available; gathering their labels costs real queries. A mode that distinguishes two memorized prefixes but cannot execute correct later updates is not accepted as a useful memory mechanism merely because its contrast loss is low.

The training objective is a declared combination of true task loss, seed and local-pair losses, plus actual parameter/state/execution cost. Weights and schedules are finite design choices for a future bounded process. The auxiliary losses are hypotheses about useful credit organization, not proof that optimization will select the intended mode. A continuation-conditioned decoder or ordinary gated RNN may use the same witness-derived constraints in a matched comparison.

When a candidate improves the frozen finite task bank under the declared criterion, retain its actual dynamics/readout and witness provenance. Freeze its incoming/self-control weights for later one-way reuse, as a conservative initial retention policy. No universal generalization certificate comes from a finite bank. Improvement of an old frozen mode initially creates a new version with explicit routing/re-encoding/replay cost; old consumers remain on the old version unless an actual replacement is validated.

## 5. Fully worked sequence: necessary memory, an early gradient and a persistent mode

### 5.1 Actual alias sequence

Take an incumbent that stores only the current scalar input: `h_t=x_t`, with an old capability that reports the current input. The new task labels a completed sequence by its early cue. Two observed sequences are

`(+1,0,0,...,0) -> +1`,

`(-1,0,0,...,0) -> -1`.

They have equal lengths and identical task context. At t=1 the incumbent states differ; at t=2 both equal zero, and all remaining inputs are zero. Thus tau=2 is the first actual merger in this example. This is a hand-specified witness to analyze the learning rule, not a proposed benchmark or a claim of newly learned behavior.

Use the subfamily `r_t=tanh(A*r_(t-1)+beta*x_t)`, with r_0=0 and A>0. Then

`r_1^s=tanh(beta*s)`,

`r_2^s=tanh(A*tanh(beta*s))`, for s in {+1,-1}.

At beta=0, the seed objective (2) has

`d L_seed / d beta = -2*m*A`.                              (4)

So for any A>0 there is a nonzero write-learning signal at the earliest merger, independent of how many neutral inputs follow. The signal is derived from the two terminal labels plus the verified common continuation; no intermediate cue-bit label was queried.

For a terminal-only squared readout objective initialized in the same small-signal regime, `d r_T/d beta=A^(T-1)` at beta=0. With a contracting A<1 this decays with the entire hold interval. Equation (4) removes that suffix factor **for this auxiliary write update**. It does not prove a global optimizer benefit, and A>1 at zero introduces the separate familiar issue of expansion near the unstable origin.

### 5.2 A persistent state can coexist with vanishing write sensitivity

For zero input and A>1, the recurrence `r_next=tanh(A*r)` has stable nonzero fixed points ±r*. On the positive half-line, `tanh(A*r)-r` is initially increasing, strictly concave and negative at r=1, so there is a unique positive fixed point. Its derivative is below one by strict concavity. Nonzero positive/negative states approach the corresponding attractor. This is standard scalar neural dynamics, not a new bifurcation result.

For an explicit regime take A=2 and beta=3. The first cue places |r_1| above 0.9; zero-input dynamics keep |r| above 0.9, preserving its sign. Each hold-step derivative has magnitude

`2*(1-tanh(2*r)^2) < 0.38` on that region.

Hence sensitivity to the original write parameters can shrink at least geometrically across the hold while the sign memory remains robust. Memory persistence and terminal gradient transmission are separate properties. The local carry update bypasses that long product by assigning a one-step separation loss to each actual paired state.

For symmetric paired states with detached previous magnitude a>0 and the positive branch below margin m, the local loss is `(m-tanh(A*a))^2`, and

`d L_local/d A = -2*(m-tanh(A*a))*sech(A*a)^2*a < 0`.        (5)

Thus a local auxiliary update can increase effective carry gain when the pair is losing its margin, even when terminal write sensitivity is tiny. Finite-horizon zero auxiliary loss does not prove infinite persistence or guarantee that learning reaches A>1. Recomputing real trajectories and checking the actual dynamics remain necessary.

### 5.3 Updateability is not silently supplied by the hold example

With A=2,beta=3, a later positive cue drives `2*r+3>0` for every r in [-1,1]; a negative cue drives `2*r-3<0`. Therefore this particular hand mechanism can overwrite its sign and then hold the new sign. For instance `(+1,0^a,-1,0^b)` ends with negative state, while reversing the last cue ends positive. The old h/current-input capability is unchanged throughout.

This demonstrates a **possible complete write–hold–overwrite–read process** in the chosen candidate family. The current two-label seed does not teach or identify that rule on its own. Additional actual episode feedback is needed to distinguish “first cue,” “last cue,” parity, prefix memorization or another task rule. A clear/reset behavior requires the learned gate and relevant feedback; it has not been constructed as acquired in this first stage. The task-derived contrast evidence may guide that next step, but does not supply an unseen subgoal.

## 6. What is retained, composed and improved

Retain a mode's actual neural transition parameters, its reset convention, its read/access head, formation witnesses, finite evidence and resource cost. Later candidates may read older modes one-way, giving a concrete route to composition without expanding every joint history. Old computation remains a prefix of the directed architecture; each new mode adds its own state and per-step work. A future readout can combine such features, but generalization to withheld compositions is not established by this structural possibility.

Freezing is a supplied retention policy and can be inefficient. A system may accumulate redundant modes, or a new task may need to reorganize an old state rather than append another. This version first preserves old behavior exactly in the specified program semantics; selective replacement, mode merging and a general economy of reuse are OPEN within the selected lineage, not separate hypotheses opened here.

## 7. Sources, controls and current boundary

SOURCE: RCC already proposes new self-recurrent candidate units on residual-error plateaus, trains their correlation with residuals, and freezes incoming/self weights when retaining a unit. Its original report explicitly discusses positive self-weights as flip-flop-like memory and negative ones as oscillators. Therefore “experience grows recurrent memory modes” is already established architecture, not this pass's novelty.

The present proposed change is **matched-continuation evidence transported into early write and one-step carry constraints**, with full endpoint replay preserving task truth. Its closest coverage by predictive-state/distinguishing-suffix learning and auxiliary temporal supervision needs comparison; no claim of novelty follows from not yet completing that comparison.

Constructive same-signal controls include RCC-style residual-correlation growth with the same candidate family; a capacity-matched gated recurrent/LSTM learner with full BPTT; ordinary recurrent-unit growth with the same witnesses and auxiliary losses; and a program/state-machine learner allowed the same raw histories, continuations, terminal labels and storage. A finite-state learner can use distinguishing continuations directly. Giving only this candidate saved histories or derived pair information would confound its result with information access. Their ability to implement the rule is not itself a historical novelty verdict or evidence that its empirical effect is zero.

All formation cost matters: acquiring useful pairs, state checks, raw history storage, candidate replay, repeated full task episodes, local objectives, retained modes and later per-step work. Exact matching may be rare; approximate matching may introduce false supervision; noisy labels may create false growth. A candidate can preserve a spurious pair distinction or overfit margin while failing task execution. These are current technical unknowns to investigate, not reasons to relabel a failure as success or abandon the lineage before its informative next action.

## 8. Next informative action and current claim scope

Root owns the process contract and first representation-versus-credit diagnostic. This worker has run no such process. The first learning question after that diagnostic is narrow: on a legitimate retained continuation witness, does the explicit local separation rule train a new mode's write/carry behavior differently from endpoint-only credit, and what remains missing for updates/read/reset under new actual feedback? That question can be informative before novelty or superiority is settled.

`CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

`Matched future plus identical full incumbent state and different deterministic labels implies a missing necessary state distinction | HAND DERIVATION | §2 and sequence §5.1 | exact state/context equality and reliable labels; finite-precision artifacts must be declared | none run | root's state-vs-credit diagnostic`

`A witness can generate an early write loss and one-step carry losses without hidden-state teacher labels | EXPLICIT ALGORITHM / HYPOTHESIS | §§3–4, gradients (4)–(5) | surrogate constraints may be inconsistent or unhelpful; witness acquisition cost | none run | examine actual first learning/transition gap under bounded contract`

`The candidate family can write, persist and overwrite a state while old computation is retained | MANUAL REPRESENTATIONAL WITNESS | A=2,beta=3 trace and one-way architecture | no training or broad composition/generalization evidence; reset still open | no empirical result | continue development of update/read acquisition`
