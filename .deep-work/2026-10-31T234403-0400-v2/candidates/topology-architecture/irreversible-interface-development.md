# An irreversible logical interface with reversible, protected execution

**RECONSTRUCTION / AGENT PROPOSAL · same Prometheus V2 · MAIN=TOPOLOGY, USER_LOCKED, human-only change.** This is a finite TODO 3 development artifact, not another Method, dashboard, campaign, or publication claim. No neural source, current Method, paper, SSH process, training, or experiment was changed or executed. Provider-effective mode remains unknown.

## Judgment

Root's two one-use bounds are correct under different contracts. For a full old frame of $N$ distinguishable logical states, one isolated use of $f$ needs at least $2N-|\operatorname{im}f|$ physical states. Requiring every old permutation to keep its meaning **after that use**, including on the resulting garbage states, raises the tight bound to

$$
D_{1,G}^{\min}=\sum_{O\in Y/G}|O|\max\!\left(1,\max_{y\in O}|f^{-1}(y)|\right). \tag{1}
$$

For the actually acquired three-state $A=(01),B=(12)$ algebra, $G=S_3$ is transitive. One composable reset therefore needs nine states, not five. Five states suffice while the retained group is only $\langle B\rangle$, whose orbits are $\{0\},\{1,2\}$. The required interface capacity can increase when a new reversible operation joins previously separate logical orbits.

No fixed finite reversible machine with a closed logical quotient can implement a non-surjective reset for arbitrary mixed repeat words. The concrete repair developed here is **a learned local reversible lift with a fresh blank ancilla for each irreversible call, an explicit bounded history arena, and charged cleanup**. Topology protects the complete acquired lift and its inverse; it does not eliminate the ancilla, initialization, history, copy, or cleanup costs.

The important future-use distinction is that preserving the visible irreversible function alone does not preserve the meaning of its garbage. Later inverse cleanup can fail even though every current forward logical test passes. Protecting the complete reversible lift prevents that failure. Ordinary exact reversible programs, protected full-lift maps, or frozen modules can also prevent it; all remain required comparators. This is a useful Block contract and a possible protection-cost question, not a new reversible-computing principle or a topology advantage result.

## 1. Domains and contracts: what $D$ counts

Let $Y$ be a finite logical state set, $|Y|=N$. Let $S$ be a finite physical data-state set, $|S|=D$, with an encoder section and decoder

$$
E:Y\hookrightarrow S,\qquad R:S\to Y,\qquad R E=\operatorname{id}_Y.
$$

Thus all $N$ old logical states remain distinguishable, including under an empty program. A physical primitive is a permutation of $S$, as in the current software monodromy semantics. $D$ counts data configurations/sheets, **not** all neural weights, program-control states, clocks, precision bits, caches, or the external resource monitor. Those additional costs must be reported separately. A factorized memory with exponentially many configurations need not have exponentially many parameters or stored values.

Let $G\le\operatorname{Sym}(Y)$ be the retained logical permutation group. Its physical operations $U_g$ preserve the old frame, or at minimum implement its declared decoded semantics. For the composable theorem, require a finite relevant set $W\subseteq S$ such that:

1. $E(Y)\subseteq W$ and $U_f(E(Y))\subseteq W$;
2. $U_g(W)=W$ for every retained $g$;
3. $R U_g(s)=g R(s)$ for every $s\in W$;
4. $U_f\in\operatorname{Sym}(S)$ and $R U_f E=f$.

The physical $G$ action may be taken coherent, $U_gU_h=U_{gh}$, in the construction. Cardinality necessity only uses the stated invariant-set bijections. The order convention does not affect the bounds.

This is a **one-use** contract: a word may contain any old operations before or after one $f$. It does not yet license a second $f$ on a previously used garbage state. If only original frame points have the equivariance guarantee, condition 3 does not cover post-$f$ execution and the orbit inflation in (1) is unjustified.

If the old frame has fewer than $N$ accessible/distinguishable states, if some states already have latent multiplicity, or if a task-specific readout is allowed to redefine the state after each program, use the actual interface instead. These bounds do not prove that every apparent many-to-one output requires newly allocated states. A constant terminal readout is also not evidence that an internal reset has been acquired and will compose with later operations.

## 2. Tight isolated one-use bound

Write $k_y=|f^{-1}(y)|$ and $S_y=R^{-1}(y)$. The old section puts at least one state in each $S_y$. Injectivity of $U_f$ puts $k_y$ distinct images of old frame points in $S_y$. Hence

$$
|S_y|\ge\max(1,k_y),\qquad
D\ge\sum_y\max(1,k_y)
=N+N-|\operatorname{im}f|. \tag{2}
$$

The last identity uses $\sum_y k_y=N$ and one additional state for every empty output fiber. This counts existing old frame states, not merely extra garbage states.

**Sufficiency.** Give fiber $y$ exactly $\max(1,k_y)$ points and choose one as $E(y)$. Assign the $k_y$ different inputs of $f^{-1}(y)$ to distinct points in that fiber. This defines an injective partial assignment of $U_f$ on the $N$ points $E(Y)$. Any injection between two equal-size subsets of a finite set extends to a permutation: biject the remaining domain points to the remaining range points. No old post-$f$ equivariance follows from this construction.

### A five-state reset that fails old composition

For $Y=\{0,1,2\}$, take $S=\{0,1,2,a,b\}$, $E(y)=y$, and

$$
R(0)=R(a)=R(b)=0,\quad R(1)=1,\quad R(2)=2,\qquad
U_f=(1\ a)(2\ b).
$$

Then $R U_f E(y)=0$ for every input: one reset is realized with $D=5$.

Extend old $A=(01)$ and $B=(12)$ on the original three points and fix $a,b$. Every old word still works on its original frame. But starting at 1, reset sends the state to $a$. Calling $A$ next leaves $a$ fixed, which decodes as 0; the required logical result is $A(0)=1$. Testing reset only from starting state 0 would miss this failure.

With only $G=\langle B\rangle$, this same five-state construction **does** satisfy post-reset equivariance: $B$ fixes logical 0 and both garbage points. This is a counterexample to applying the transitive bound when the retained group is not transitive, not a counterexample to (1).

## 3. Tight one-use bound with old-operation composition

Let $W_y=W\cap R^{-1}(y)$. Equivariance and $U_g(W)=W$ imply that $U_g$ is a bijection $W_y\to W_{gy}$. Therefore $|W_y|$ is constant on each $G$ orbit. Conditions 1 and 4 give $|W_y|\ge1$ and $|W_y|\ge k_y$. Each orbit consequently contributes at least its size times its largest required multiplicity, proving necessity of (1).

For sufficiency, put

$$
m_O=\max\!\left(1,\max_{y\in O}k_y\right),\quad
S=\bigsqcup_{O\in Y/G}O\times\{0,\ldots,m_O-1\},
$$

with $R(y,h)=y$, $E(y)=(y,0)$, and $U_g(y,h)=(gy,h)$. Choose distinct output slots for each preimage of $f$, as in §2, and extend the resulting partial injection to a permutation $U_f$ of $S$. The old group preserves the first coordinate's meaning on all of $S$. This gives every allowed old-word–$f$–old-word computation, and proves equality in (1).

For transitive $G$, (1) becomes $D=N\max_y k_y$. A reset has $k_0=N$, hence $D=N^2$. For $N=3$, a two-to-one merge with preimage counts $(2,1,0)$ needs six states under transitive old composition, versus four for an isolated use. The three-to-one reset needs nine versus five.

The $G$-invariant relevant set matters, not irrelevant unreachable machine states. It is sufficient to impose equivariance on the $G$-saturation of $E(Y)\cup U_f(E(Y))$. If later operations reach more states, the contract must expand with them. If its finite closure was already included in $W$, the proof applies there without claiming anything about the rest of $S$.

### Acquisition can increase this requirement without changing $f$

When only $B$ is retained, the minimal reset fibers have sizes $(3,1,1)$. Adding $A$ joins the three logical states into one orbit and requires $(3,3,3)$. Four additional microstates suffice. They can be added as two new copies in each of fibers 1 and 2; extend $B$ to swap the corresponding new fiber-1/fiber-2 copies, and let $A$ swap corresponding fiber-0/fiber-1 copies. The original five points and the previous reset assignment can retain their old transitions.

Thus the obstruction does not demand discarding the old result or recoding its existing history. It identifies a concrete capacity cost of retaining its *composability* when the surrounding operation algebra grows. This is a finite reconstruction, not a novelty claim or a proof of favorable neural scaling.

## 4. What fails for arbitrary repeated reset

Suppose a fixed finite forward-reachable set $W$ contains $E(Y)$ and satisfies

$$
U_f(W)\subseteq W,\qquad R U_f(s)=f R(s)\quad(s\in W).
$$

Since $U_f$ is injective and $W$ is finite, $U_f(W)=W$. Consequently

$$
Y=R(W)=R U_f(W)=f(R(W))=f(Y),
$$

so $f$ must be surjective. A non-surjective reset cannot satisfy these premises. The same conclusion holds if one demands equivariance on an entire infinite $S$ on which $U_f$ is a bijection and $R$ is onto. Infinite fresh-history implementations escape by requiring the logical rule only on a forward-reachable subset on which the physical map is not onto.

The finite result does **not** exclude bounded horizons, fresh prepared inputs, measurement/dissipation, returning to a source input, externally versioned clocks, or restricting the admissible program language. Each changes a premise and has its own cost. A second reset can already fail the five-state example: input 1 goes $1\to a\to1$, whose second output is 1 instead of 0.

### A tight finite-horizon state count, not an exponential parameter claim

For a transitive old group and reset to 0, let $W_t$ contain all states reachable from the old frame using at most $t$ resets and arbitrary old operations. Assume the intended semantics hold through reset count $T$. Since $U_f(W_{t-1})\subseteq(W_t)_0$ is injective and old-group symmetry equalizes the $N$ fiber sizes in $W_t$,

$$
|W_t|\ge N|W_{t-1}|,\qquad |W_T|\ge N^{T+1}. \tag{3}
$$

This bound is attained at the data-state level by $T+1$ $N$-ary registers. Initialize $(y,0,\ldots,0)$, apply every $g$ to the first register only, and use the same reversible rotation

$$
U_f(y,a_1,\ldots,a_T)=(a_1,\ldots,a_T,y).
$$

Before reset number $T+1$, the next incoming register is blank, so the first register is reset correctly. At that next call, old history can return and the logical reset contract ends.

The $N^{T+1}$ configurations are represented by only $(T+1)\log_2N$ data bits (up to integer encoding), not a table with $N^{T+1}$ learned entries. Register rotation costs moves; a fresh-cell/address implementation costs an explicit pointer and addressing. Program and resource-monitor state were excluded from $D$ and must be added to a whole-machine account. No exponential neural-parameter lower bound follows from (3).

For a single fixed program/initial condition, or a program with no arbitrary interposed old operations, the reachable set can be much smaller. In particular, a lone known repeated reset does not lose $\log N$ new bits on every step once the core is already zero. The transitive/all-mixed-words premise supplies the growing family of distinguishable pre-reset states in (3).

## 5. One concrete extension: local lifts, fresh cells, and a cleanup contract

Use a factorized logical core register $q\in Y$ and a bounded history arena. A declared irreversible call receives one fresh blank $N$-ary cell $a=0$. The local physical operation is a learned permutation

$$
U_r:Y\times\{0,\ldots,N-1\}\to Y\times\{0,\ldots,N-1\}.
$$

Its desired logical interface is

$$
\pi_Y U_r(q,0)=f_r(q). \tag{4}
$$

No output value for the garbage coordinate is supplied. An exact permutation executor already guarantees that different inputs have distinct joint outputs. Using a full $N$-ary cell is sufficient for any finite $f_r$ and deliberately exposes a conservative resource cost. If complete evidence establishes $K_r=\max_y|f_r^{-1}(y)|$, an $N\times K_r$ local product suffices; this version does not assume that a finite partial sample reveals that maximum.

### State and actual transitions

The native state is $(q,H,p,\mathcal V)$: current core, allocated garbage cells, actual executed gate/port log, and fixed source/interface versions. A read/write address or remaining-capacity counter is explicit runtime state. Every old $g$ executes $q\mapsto gq$ and leaves all history cells unchanged. For a new irreversible call:

1. Verify that the arena has a fresh cell with the prescribed blank value; otherwise halt with resource exhaustion.
2. Execute the learned local $U_r$ on $(q,0)$, obtaining $(q',a')$ by the actual protected primitive interface.
3. Keep $a'$ in that allocated cell. Do not reset, overwrite, or reinterpret it as another fresh cell.
4. Record the actual primitive ID, typed ports/cell address, and source/semantic version needed for inversion. The visible state is $q'$.

These local operations act as $U_r\otimes I$ on all other cells. Old permutations act as $g\otimes I_H$, so their post-$f$ logical semantics hold for every garbage value. This structural tensor action replaces enumeration of the arena's Cartesian-product state space. The same factorization is offered to classical rivals.

This is the current **software** Block interface. A local $N^2$-sheet executor must have an explicitly paid encoding of $(q,a)$ into its resting fiber, actual cold lift/identification, and return decoding; a native cache may then apply the observed joint map. Existing $N$-state primitives and new pair primitives can share typed register ports. This is not a free physical teleportation protocol between different quantum or continuous devices. Claiming one physical carrier would additionally require a compatible state-preparation/interoperation construction and its costs. No old polynomial growth/critical-loop compiler is rerun or declared solved here.

### Formation from the same terminal feedback

Keep the current learned logical frame fixed while a history arena is live. New input/output adapters may be learned at safe job boundaries under the existing binding contract; changes of physical register interpretation are not silently applied to live garbage.

For a forming bank slot, the neural controller proposes local pair-operation paths. Actual continuation yields a permutation $U_r$, not a supplied target table. Terminal feedback observes the core result after complete mixed programs, including old operations after the candidate operation. Its label never specifies a garbage code or the correct reversible completion. Use the already derived coupled within-chamber/event credit rule on these logical terminal losses. The continuous cover, pair port, blank-cell convention, allocation cap, and permutation grammar are supplied inductive biases and are identical for rivals.

If training produces two or more distinguished input states with the same required output, their multiplicity is a lower-bound witness for garbage capacity. It does not establish the unseen maximum, and training cannot fabricate blank cells from that witness. With the conservative $N$-ary cell, formation may proceed without first learning the exact minimum.

Admission requires the declared logical criterion on fresh-cell starts, complete physical permutation validity from execution, the old input/call/readout contract, a frozen precision/execution domain, and the resource ledger. Old logical tests alone cannot certify off-slice behavior or universal learning. Once admitted, **protect the entire local $U_r$ class and its inverse**, including the garbage semantics. Merely freezing its visible restriction (4) is inadequate for the cleanup operation below.

### The soft training interface can remain on the logical core

From the actually executed local lift define the derived row-one-hot channel

$$
[T_r]_{q,q'}=\mathbf1\{\pi_Y U_r(q,0)=q'\}. \tag{5}
$$

Rows may coincide. There is no trainable permutation $W_r$ hiding behind this channel. Existing learned encoder, readout, and soft call mixtures can use $T_r$ in their terminal likelihood, with the event term changing the underlying lift class. This logical marginal is sufficient for **forward** computation because each call gets a fresh blank cell and future forward operations never read old garbage. That is a Markov interface condition, not a general claim that correlations may be discarded.

The actual native computation still retains the full garbage and gate log. It cannot invert $T_r$, which is non-injective. Inversion uses $U_r^{-1}$ on the actual joint register state. If future forward computation reads garbage, or if an ancilla is reused without cleanup, (5) no longer specifies the required state and the soft interface must be enlarged. Full joint distributions over a long soft program history must not be assumed factorized for free.

### One cleanup rule, with its retained information

At a finite job boundary, reversibly copy the classical final core label to a blank output register, then replay the **recorded actual primitive IDs and ports** in reverse order, using inverse loops. A fixed modular-addition copy $(q,b)\mapsto(q,b+q\bmod N)$ is a legitimate supplied primitive on a blank $N$-ary output register. This is copying a classical label, not cloning an unknown quantum state.

The replay restores the initial physical core and all history cells to their blank states, leaving the copied result. It therefore frees the bounded arena for another job. It still retains the original input and the copied output. Their eventual disposal or external output-stream storage is not free. A neural router that has since changed must not select a new inverse program: the log bypasses it and names the acquired full lifts that actually wrote the garbage.

Shared path implementations may change while these primitive classes, physical register labels, and source versions remain protected. Their current inverse paths then undo the old full actions. The logical input/output adapters must remain compatible with the live register frame, or the job must close before changing it. This restriction, including any lost plasticity, is part of the proposed contract.

## 6. Analytic reset example and a hidden-garbage counterexample

For the three-state transitive algebra, take the local product $Y\times Y$ with decoder $R(q,a)=q$ and blank $a=0$. One valid reversible reset lift is

$$
U(q,a)=(a,q),\qquad U(q,0)=(0,q).
$$

This is an **existence witness**, not a teacher lift to be supplied to learning. Its nine states meet (1) exactly. For two reset calls use two fresh cells. Starting from $(x,0,0)$ and interposing arbitrary old permutations $g_0,g_1,g_2$ gives

$$
(x,0,0)\to(g_0x,0,0)\to(0,g_0x,0)
\to(g_1 0,g_0x,0)\to(0,g_0x,g_1 0)
\to(g_2 0,g_0x,g_1 0).
$$

The two dirty cells are still stored. The 27 possible data configurations fit in three ternary registers; no 27-entry trainable global state table is required. After copying the result, inverse $g_2$, the second pair inverse, inverse $g_1$, the first pair inverse, and inverse $g_0$ restore $(x,0,0)$, leaving the output copy untouched.

**Why visible retention is insufficient.** For a smaller two-state example let $U_0(q,a)=(a,q)$ and let $V$ exchange $(0,0)$ with $(0,1)$ while fixing the other two points. Put $U_1=V U_0$. Both lifts implement exactly the same reset on blank-cell starts: their visible output is always 0. They merely use opposite garbage encodings.

An old run of $U_0$ from $(1,0)$ leaves $(0,1)$. Replacing it by the visibly equivalent $U_1$ passes every fresh-start logical reset test. But $U_1^{-1}(0,1)=(0,0)$: cleanup recovers input 0, not the original input 1. Full monodromy-class preservation rejects this change; an ordinary full four-state-map constraint or immutable reversible program rejects it as well. A coarse reset-output constraint does not.

This counterexample identifies the additional useful state that must be retained and the future behavior that can reveal its loss. It does not prove that topology is the cheapest way to protect that state.

## 7. Complete resource account and strong comparators

For $T$ irreversible calls with local ancilla alphabets $K_r$ (this concrete conservative version uses $K_r=N$), at minimum report:

- Actual core and garbage values: $\log_2N+\sum_{t\le T}\log_2K_{r_t}$ data bits, rounded for the chosen encoding; arena capacity, initialized blanks and address/control state separately.
- Program/history metadata: actual primitive IDs, addresses, call order, source and interface versions; no free inverse-plan oracle.
- Learned parameters: core/call/path-controller parameters and any per-lift head/embedding. Count fixed/preallocated capacity and feature/source metadata; there is no automatic $N^{T+1}$ parameter bill or automatic zero-parameter growth claim.
- Local physical representation: an $N K_r$-state pair interface, root/configuration precision, cold continuation, event discovery, training-label accesses, failed proposals and execution-domain checks. A generic full local map costs $N K_r$ image entries if cached; a logical channel costs $N$ image entries. Both stores are derived from paid execution and are additional to neural parameters.
- Actual forward lifts/cached calls, fresh-cell preparation/access, output copying, and every inverse call for cleanup. At the gate-program level, a forward $L$-gate job followed by one copy and full replay uses $2L+1$ gate invocations, with different per-gate lift/actuation costs retained. This is not a hardware-energy estimate.
- Protection under subsequent shared updates: full-lift invariance and current numeric execution domain, old binding checks, rejection/stalling, and any frame-freezing or version storage. Report new acquisition alongside protection, not only cleanup success.
- Permanent input/output records at job boundaries and the cost or restriction of eventually releasing them. Uncomputation does not turn many-to-one input erasure into a free reversible operation.

The closest structural rival is a **classical reversible local-gate learner/compiler** using the same observed logical examples, blank registers, local gate grammar, bit precision, protected full-lift semantics, compressed representations, and cleanup schedule. It may choose a simple SWAP reset or a canonical preimage-index lift after honestly acquiring the logical function. Giving it an arbitrary dense global history table is not a fair baseline. Fixed or exact local programs can preserve full lift identities without saving a neural checkpoint for every history record.

An **ordinary irreversible state machine/channel** is also a legitimate computational comparator. For digital answers it can overwrite the core directly or use an $N$-state row-one-hot channel. It need not retain garbage just because this architecture uses reversible primitives. Compare actual measured compute, state storage and learning cost, with the execution requirement stated in advance. A thermodynamic energy advantage cannot be inferred from a Python simulation or a branch count.

A **quantum channel/dilation** is a related but different interface. On classical basis states, $|q\rangle|0\rangle\mapsto|f(q)\rangle|c(q)\rangle$ preserves distinct preimages when $c$ separates them. Tracing out the environment hides the information. A reset can use a SWAP with a prepared register. Coherent inputs, measurement, and a prescribed fully dephasing channel impose additional requirements; the finite classical decoder-fiber theorem above is not automatically a minimal-dimension theorem for every quantum channel or measurement model. Ancilla initialization and environment reset remain explicit costs.

## 8. Original sources actually entered, and provenance boundary

**SOURCE.** [Bennett, *Logical Reversibility of Computation*, IBM JRD 17, 525–532 (1973), original scan](https://worrydream.com/refs/Bennett_1973_-_Logical_Reversibility_of_Computation.pdf): printed pp. 525–526, theorem on p. 527, Table 1 on p. 528, and the history/segmentation discussion pp. 529–530 were entered. His construction retains history, copies the desired result, and reverses the computation, ending with input and output rather than a free erased input. The fresh-history and cleanup rules here are a reconstruction/application of that established mechanism; no new general reversible-simulation theorem is claimed. No universal constant time/storage bound is imported from the paper into this learned Block.

**SOURCE.** [Toffoli, *Reversible Computing* (1980), abridged original](https://cqi.inf.usi.ch/qic/80_Toffoli.pdf): printed pp. 634–638, §3's constant/garbage distinction, Theorem 4.1 and the §5 realization discussion were entered. It explicitly distinguishes prepared auxiliary inputs, garbage outputs and temporary-storage channels, and realizes finite functions through invertible functions with restriction/projection. This directly covers the broad construction, while (1) is our separately proved specialization to a retained logical group. The longer MIT technical report was located/opened but its scanned body was not relied on after text extraction failed; the readable abridged primary version is the source used.

**RECONSTRUCTION.** Bounds (1)–(3), the reachable-set qualifications, the five-state/post-$A$ failure, the group-growth capacity calculation, and the garbage-recoding cleanup counterexample were derived here. An independent worker checked the precise one-use/orbit and finite-repeat assumptions and reached the same formulas; that agreement is not an independent mathematical proof. The proofs above are the checkable evidence. They are informal finite mathematics, not executed experiments or machine-checked theorems, and historical originality is not established.

## 9. One next scientific decision

The construction is explicit enough to diagnose **one** consequential interface question: after acquiring a logically irreversible operation, can the same protected learned Block retain both its visible computation and the full historical lift needed for later cleanup while shared learning acquires something else, at a favorable total cost?

A discriminating future test would write garbage with an admitted lift, perform a declared shared update, then use the current inverse and recorded gate IDs to clean that old history. Compare full monodromy protection, a coarse logical constraint, and an equally informed ordinary full-lift/frozen reversible control; include the cheap irreversible channel when only logical answers are required. Measure new acquisition, visible retention, recovered input/blank cells, storage and all forward/cleanup/protection costs. The counterexample predicts a failure the coarse constraint can miss, not a win over the full classical comparator. This would be DIRECT for the history-interface claim and only PARTIAL for the general topology advantage hypothesis.

No such experiment or new implementation is authorized by this artifact. The next single decision is whether this fresh-cell/full-lift cleanup contract is the right TODO 3 extension to implement, with the physical-versus-software port and full-cost scope fixed first. It does not replace the human-locked MAIN or broaden the six-item roadmap.

| CLAIM | STATUS | EVIDENCE / ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION |
|---|---|---|---|---|---|
| A learned reversible lift with fresh ancillas can provide composable irreversible logical operations; full-lift retention protects later cleanup | RECONSTRUCTION / explicit unexecuted extension | Bounds and finite proofs, N3 example, N2 garbage-recoding counterexample, Bennett/Toffoli original passages | Trainable local pair lift, typed execution port, and competitive total lifecycle cost | Bare fixed-state reset quotient; one-use five-state lift without post-operation equivariance; free garbage reuse | Decide whether to implement the bounded fresh-cell/cleanup interface with full classical and irreversible-channel controls |

