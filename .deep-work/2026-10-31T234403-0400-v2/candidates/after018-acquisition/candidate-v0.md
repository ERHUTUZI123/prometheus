# Candidate v0: acquire recurrent state by factoring the algebra of task error

Freeze: 2026-11-01, before access to current or previous candidate/review bodies. Provenance: AGENT PROPOSAL and RECONSTRUCTION unless marked SOURCE. This is a fresh constructive pass inside the existing V2 run, not a new run. Effective provider/model/mode: unavailable; no Pro claim. No experiment, training, installation, publication, or paid route was used.

Chronology correction before hashing: after the substantive candidate and reading files were first written, but before their freeze hashes were taken, two parent messages arrived. One named SRWM and learned loss as the parent's own reading directions; the second requested that this exposure be recorded. Those two source names were therefore exposed before the hash freeze. No equations, current candidate body, or review body from that comparison was supplied or read. This is instruction-only isolation with shared filesystem access, not pristine blindness. The independently chosen mechanism and derivation above were already written when those messages arrived.

## Judgment

**A complete acquisition mechanism can be constructed for a restricted but instructive family: use actively queried endpoint errors to discover recurrent memory coordinates, then learn composition as products of those coordinates. However, its tractable guarantee is reproducible by a conventional sparse-Fourier learner and a register compiler. V0 therefore establishes a constructive route, not a new learning principle or an advantage over the strongest same-information alternative.**

The consequential limitation is the acquisition of a compact internal transition system when a small number of reusable operations generate exponentially many joint states. The mechanism below does address acquisition, rather than merely exhibiting a compact representation. Its weakest step is also explicit: the strong acquisition law depends on a known algebra of admissible transitions and resettable, actively chosen queries. Making those relations genuinely acquired, without losing the acquisition advantage, remains OPEN. Nothing here establishes that gap as novel.

Connection to the Fundamental Question: the acquired objects are executed recurrent operations and the state coordinates they update. The learning process chooses which coordinates exist, how each command changes them, which coordinates are retained, and which previously learned coordinates can be multiplied to express a new function. This is a partial answer to construction, composition, and retention of neural computation. It does not answer arbitrary mechanism invention, evolvability, or open-ended evolution.

## 1. Begin with the capability limitation

Consider a neural learner receiving long sequences of commands and only an endpoint answer. A task may depend on several latent facts accumulated by those commands. A learner can succeed on the current endpoint statistic while failing to construct state that will support another readout later. Alternatively it can represent essentially every joint history/state, paying for combinations rather than for the operations that generate them.

The important question is not whether a fixed recurrent network can represent the correct algorithm. It is whether feedback can *cause the learner to construct the small, recursively executable state variables and their update laws*, with a learning cost tied to the relevant mechanism rather than to the number of joint states or trajectory length.

This is not a claim that ordinary gradient training universally fails. It distinguishes three burdens that must not be conflated:

1. The information needed to distinguish task-relevant states.
2. The search required to discover a recursively updatable representation of them.
3. The cost of executing that representation after acquisition.

Giving the learner intermediate latent bits, the correct memory modules, or an uncharged equivalence oracle would solve much of burden 2 by assumption. V0 instead uses endpoint answers at explicitly counted, learner-chosen command sequences. It still supplies an important algebraic prior; that prior is accounted for below.

## 2. Closest checked precedents and the actual claim boundary

SOURCE: Predictive representations already ground state in future tests; nonlinear predictive representations can compress some register systems exponentially. Product-automata learning already exploits component outputs and counterexamples. Learned group actions and subgroup decomposition are established representation-learning targets. Sparse-Fourier query learning already finds useful parity features without enumerating the cube. Exact bodies/equations read and limitations appear in the paired reading note.

V0 does **not** claim any of these ingredients as new. The constructive proposal combines error-based coordinate discovery with an executable recurrent realization and explicit multiplicative reuse. Its purpose is to see whether this supplies an acquisition principle that survives the best conventional explanation. The answer in the fully worked class is currently negative: an ordinary algorithm can compile exactly the same circuit.

An important nearby distinction is history state versus observation encoding. The 2026 symmetry-discovery paper assumes an injective current observation, up to its interaction equivalence. V0 uses history because the current endpoint observation need not identify the state. This is a difference in setting, not evidence of novelty; predictive-state methods already address history sufficiency.

## 3. Precisely scoped acquisition problem

There are m named commands a_1,...,a_m. Within the v0 theorem class they are commuting involutions. Their effective action may be redundant and is not given to the learner. One realization is

    b_{t+1} = b_t XOR A e_{a_t},        b_0 = 0,

where A is an unknown k-by-m binary matrix, e_a is the command's one-hot vector, and the hidden state b is never exposed. This realization explains the class; the learner does not receive A, k, b, or intermediate outputs.

For a task q, a resettable endpoint interface returns

    f_q(x) = g_q(Ax) in [-1,1],        x in F_2^m.

Here x is implemented by resetting and executing the canonical word containing command a_i once if x_i=1. The learner may choose x and q. A label costs one reset, at most m command executions, and one endpoint evaluation. A later task is not free: all of its labels count. The test-time requirement is prediction after arbitrary command sequences, including sequences longer than acquisition queries, without endpoint queries during execution.

Critical assumptions:

- Commutation and involution are a class prior in the proved construction. A few short successful relation tests do not establish them universally.
- Resets and learner-chosen endpoint queries are available and charged. Passive episodes are a different information regime.
- Useful outputs have a small or discoverable set of sufficiently heavy Fourier coefficients. The relevant minimum coefficient magnitude is a resource parameter, not a free constant.
- Exact binary commands and sign updates are available. The statement is not a robustness theorem for noisy analog recurrent dynamics.
- Task outputs, not a hidden state oracle, determine relevance. A coordinate that no available task can reveal cannot be learned merely because it might help an unspecified future task.

Functional equivalence means identical endpoint outputs for every allowed continuation and task. Distinct encodings of the same behavior do not count as new capabilities.

## 4. The actual neural state and executed transition

The persistent learned object is

    M = (B, {C_q}, metadata),

where the independent rows of B in F_2^{r-by-m} are discovered masks. They are not prescribed hidden bits. For each acquired coordinate j, the episode state is

    z_j in {-1,+1},      z_j(0)=+1.

On command a, the executed transition is

    z_j(t+1) = z_j(t) * (1 - 2 B[j,a]).                 (1)

This is a multiplicative recurrent neural unit with a learned binary command gate. The primitive multiplication and sign gate are designed substrate operations. The mask, whether the unit exists, and its task use are acquired. A neural network need not discover multiplication itself for this partial construction to count.

For any mask sigma in the row span of B, solve beta B = sigma over F_2 and define a product feature

    Phi_beta(z) = product_{j: beta_j=1} z_j.

An output is a learned sparse product network

    fhat_q(z) = sum_beta C[q,beta] Phi_beta(z).         (2)

The constant feature has beta=0. Products provide composition. A new useful mask in the span is a new output operation, not new memory. An independent mask creates a new recurrent unit.

The ordinary coefficient learning law, for labeled endpoint (x,y), can be

    C[q,beta] <- C[q,beta] + eta (y-fhat_q(x)) Phi_beta(z(x)).  (3)

For the clean derivation below, coefficients are estimated by orthogonal correlations instead. The structural law that adds masks is not backpropagation through the trajectory. It is an explicit, nonlocal acquisition routine using task error.

After a task is accepted, its masks and readout are stored. Later tasks may extend B while leaving old readouts fixed. Retention here follows from that explicit storage policy. It is not an emergent resistance to interference under arbitrary further weight updates.

## 5. Structural learning law: expose unexplained operators in the error

Define the currently unexplained endpoint function

    R_q(x) = f_q(x) - fhat_q(x).

For a binary mask sigma, write chi_sigma(x)=(-1)^(sigma dot x). Its residual correlation is

    rhat(sigma) = E_uniform_x [R_q(x) chi_sigma(x)].

The following is a reconstruction of the known prefix-energy method, applied to residual acquisition. It is not an original learning theorem. For a prefix alpha of length ell,

    W_alpha = sum_{suffix nu} rhat(alpha,nu)^2
            = E_{u,v,v'} [ R_q(v,u) R_q(v',u)
                           (-1)^(alpha dot (v XOR v')) ],       (4)

where u in F_2^{m-ell} and v,v' in F_2^ell are independent uniform samples. A sample uses two endpoint queries sharing u. This is correlated query design, not two independent ordinary examples.

Acquisition routine, with threshold theta and fixed query cap:

    candidates := {empty prefix}
    for ell = 0,...,m-1:
        estimate W_{alpha 0}, W_{alpha 1} for each surviving alpha
        retain a child if its estimate is at least 3 theta^2 / 4
    estimate rhat(sigma) for surviving full masks
    accept masks whose estimated coefficient clears the prescribed threshold
    for each accepted sigma:
        if sigma is outside row_span(B): append sigma to B
        otherwise construct Phi_beta from beta B = sigma
        add/refit that output coefficient

Take every energy estimate to additive error at most theta^2/4. A mask of magnitude at least theta survives every prefix, because every ancestor has W at least theta^2. A retained prefix has true W at least theta^2/2, so each depth retains at most 2||R_q||_2^2/theta^2 prefixes. The leaf correlation estimates need their own confidence allocation and threshold margin; no exact support recovery is claimed for coefficients arbitrarily close to a threshold.

Costs: if rho is a known bound on |R_q|, the random variable in (4) lies in [-rho^2,rho^2]. Hoeffding gives a sufficient sample count

    N >= 32 rho^4 theta^(-4) log(2Q/delta)

per energy estimate, where Q is the declared total estimate cap. A union bound then applies even with adaptive branching, using fresh samples for every estimate. For a residual-energy bound E, Q can be capped at a constant multiple of m E/theta^2, with an abort if exceeded. This yields a coarse polynomial query bound in m, E, rho, theta^(-1), and log(1/delta); each energy sample costs two length-at-most-m executions. Coefficient estimation, basis elimination, reset cost, neural forward cost, and acquisition metadata also count. No numerical performance claim follows from this loose bound.

An explicit termination path is reaching depth m, hitting the query cap, or finding no accepted residual feature. A failed cap or unresolved coefficient is a failed/unresolved acquisition, not success. This routine is specified but has not been executed in this subtask.

## 6. Load-bearing derivations

### 6.1 Why equation (4) exposes a feature without enumerating masks

Expand the residual into characters, writing a full mask as (alpha',nu):

    R(v,u) = sum_{alpha',nu} rhat(alpha',nu)
                            (-1)^(alpha' dot v + nu dot u).

Multiply by (-1)^(alpha dot v) and average over v. Orthogonality kills alpha' != alpha, leaving

    F_alpha(u) = sum_nu rhat(alpha,nu) (-1)^(nu dot u).

The independent v,v' in (4) produce F_alpha(u)^2. Averaging u kills all cross terms nu != nu', leaving exactly the sum of squared coefficients. This is the acquisition lever: a prefix may have zero ordinary correlation with the target while the mass of some as-yet-unknown longer mask is detectable through paired interventions.

No statement that all neural gradients are incapable of exploiting such information is used. The learner deliberately asks for different information from a stream of independent examples; comparisons must give that access to every baseline.

### 6.2 Why the acquired coordinate is a reusable recurrent mechanism

Let x(h) be the command parity vector of a history h. Induction on sequence length in (1) gives

    z_j(h) = (-1)^(B_j dot x(h)).                     (5)

Appending a command changes x by e_a, so one exact local multiplication updates the coordinate, independent of the history length. This proves execution on arbitrary finite lengths within the stated deterministic class; it does not depend on length extrapolation in a fitted continuous approximation.

The product identity

    chi_sigma(x) chi_tau(x) = chi_{sigma XOR tau}(x)   (6)

means beta B = sigma implements the new mask exactly from existing recurrent units. A new output can therefore reuse old dynamical state rather than store another copy of the whole history.

### 6.3 Why only rank-many binary recurrent units are needed

Let S be the union of nonzero Fourier supports of the available tasks. Let V=span(S) and r=dim(V). Any shift d in V-perp changes no task output after any continuation. Conversely, if d is not in V-perp, some supported character changes sign. For the task containing that nonzero coefficient, f_q(x+d)-f_q(x) has a nonzero Fourier coefficient and cannot vanish for every x. Some allowed continuation therefore distinguishes the two histories.

Thus the observational equivalence classes of history parity are the 2^r cosets of V-perp. Any exact deterministic finite machine distinguishing them needs at least r binary state bits. B attains that bound for this class. This is a lower bound on binary information state, not on the number of real-valued neurons under unlimited precision, and not a general lower bound against neural networks.

### 6.4 Learning improvement is separate from execution correctness

For exact projection residual R orthogonal to the current output span, adding a character with coefficient c=E[R chi_sigma] and refitting reduces uniform squared error by c^2:

    E[(R-c chi_sigma)^2] = E[R^2] - c^2.

With an estimated coefficient ctilde, reduction is c^2-(ctilde-c)^2. A discovered feature is useful only when estimation uncertainty is smaller than its coefficient. Many arbitrarily weak features may be undiscoverable within a fixed budget even though the eventual circuit is compact. An exponential state-space compression theorem is not automatically an efficient-learning theorem.

## 7. Fully worked acquisition and execution

There are three opaque commands A,B,C. For exposition only, the hidden state is

    b1 = x_A XOR x_C,
    b2 = x_A XOR x_B,
    b3 = x_B.

The learner sees none of these equations. Its first task returns f1=b1 AND b2. Start with the mean output 1/4. The endpoint residual is

    R1(x) = -chi_101(x)/4 - chi_110(x)/4 + chi_011(x)/4.

Use the order (A,B,C) and theta=1/5. Exact energy values, before estimation error, are:

| Prefix length | Nonzero prefix energies |
|---|---|
| 0 | W_empty=3/16 |
| 1 | W_0=1/16; W_1=1/8 |
| 2 | W_01=W_10=W_11=1/16 |
| 3 | W_011=W_101=W_110=1/16 |

The search exposes the three masks. They have rank two: 011=101 XOR 110. The learner may choose 101 and 110 as its stored basis (a deterministic tie rule can choose their order); it constructs the third feature as their product. Call the recurrent coordinates z1=chi_101 and z2=chi_110. The learned endpoint network is

    fhat1 = (1-z1-z2+z1*z2)/4.

The acquired command gates are:

| Command | z1 update | z2 update |
|---|---|---|
| A | negate | negate |
| B | retain | negate |
| C | negate | retain |

This is actual recurrent memory learned from endpoint behavior. The learner was not handed the two hidden bits, although the allowed class supplied the sign-update primitive and global commutative-involution prior.

A later task f2=b3 has mean 1/2 and residual -chi_010/2. Mask 010 is outside the old span, so the structural law adds z3. It is negated by B and retained by A,C. The old output stays exactly the same. A later task f3=b1 XOR b2 has the learned readout (1-z1*z2)/2 and requires no new recurrent state. Its readout still needs task feedback; this is not zero-shot identification of an unknown task.

With all three coordinates present, execute h=A,C,B,A,B:

| Executed prefix | z1 | z2 | z3 | fhat1 | fhat2 |
|---|---:|---:|---:|---:|---:|
| empty | +1 | +1 | +1 | 0 | 0 |
| A | -1 | -1 | +1 | 1 | 0 |
| AC | +1 | -1 | +1 | 0 | 0 |
| ACB | +1 | +1 | -1 | 0 | 1 |
| ACBA | -1 | -1 | -1 | 1 | 1 |
| ACBAB | -1 | +1 | +1 | 0 | 0 |

Every row follows (1); no endpoint answer is needed to perform these transitions. The final command parity is (0,0,1), so the hidden state is (1,0,0), independently agreeing with the circuit's final outputs. This is a hand-checked derivation, not an executed experimental result.

## 8. Strongest attacks and a serious repair attempt

### Attack A: the learner has been given the important mechanism

The primitive sign register and assumption that commands commute and square to identity make all raw command-parity counters a sufficient, obvious state. A conventional system can store m counters, then learn the endpoint function. Learning an r-dimensional basis reduces memory when r << m, but this alone is not a new organization of computation. Our impressive-looking generalization to long sequences largely comes from the supplied algebra and exact recurrent primitive.

Repair attempted: replace the known algebra with empirical relation testing. For histories h and suffix probes s, compare endpoint answers after haa s versus h s, and hab s versus hba s. A disagreement is a concrete witness against the proposed relation. This is a legitimate falsification rule using the same task interface.

But absence of disagreement on finitely many contexts is not a proof. A transition system can imitate commuting toggles for every tested word and diverge only after a longer trigger. The canonical-word reduction f_q(x) is then invalid, equation (5) fails, and the Fourier acquisition law targets the wrong function class. Strong uniform guarantees require a class bound, a separating-context assumption, or an equivalence oracle whose cost is charged. This repair does not currently remove the key prior. OPEN, not solved by adding a neural relation detector.

### Attack B: hidden state may not be learnable from the current task

If the sole task is parity of all latent bits, one recurrent parity coordinate is sufficient. The other bits cannot be justified by the observed task merely because a later task might ask for them. The rank result says exactly what available task-and-continuation feedback identifies. A future task may require adding state and gathering new feedback. Neither perfect current performance nor a learned factor is evidence of increased future discovery ability.

Repair: explicit exploratory probe tasks could make more state distinguishable. Their labels and costs must be counted, and a same-information baseline gets them. This moves the scientific issue to choosing useful probes under finite resources. V0 specifies no general solution to that issue.

### Attack C: compact state can coexist with exponentially expensive acquisition

For an AND of k independent bits, individual Fourier coefficients have magnitude 2^(-k). A heavy-coefficient procedure can therefore demand an exponential budget even when the state and readout admit a short conventional circuit. This directly blocks a claim that the mechanism generally converts compact computation into easy learning.

Repair: reuse already discovered factors and allow direct gradient fitting of the product readout; the AND can be fitted once its factors are acquired. That does not explain acquisition of those factors from the AND task alone. The repaired system still needs another informative output or a stronger structural search primitive. No acquired-scratchpad miracle is claimed.

### Attack D: irreversible or noncommuting computation

Overwriting, comparison-dependent branches, stacks, and pointer updates generally violate the v0 group class. The construction must not be presented as a method for arbitrary internal algorithms. Appending a residual RNN would make the system expressively broader, but supplies no acquisition argument for that residual. That is an unconstructed extension, not a repair counted as success.

## 9. Strong same-information conventional alternatives

The decisive baseline is **a classical sparse-Fourier query learner, Gaussian elimination over discovered masks, binary registers, and a sparse polynomial readout**. It receives the identical named commands, global class assumption, reset ability, endpoint tasks, random-query access, memory accounting, and compute budget. It can emit exactly equations (1)-(3), use exactly (4), and achieve the same derived guarantee. The neural realization currently changes the implementation vocabulary, not the accessible function class or acquisition cost.

A second strong baseline keeps all m raw parity registers and spends acquisition on the readout. A third is a multiplicative recurrent network or learned linear action model trained with the same active query schedule and the same total task evaluations. Comparing against a passive RNN alone would confound information access with learning organization. Comparing with a flat linear PSR alone would confound a restricted representation with an available nonlinear/factored alternative.

V0 supplies no theorem or experiment separating the proposed method from these. It does supply a concrete alternative to unspecific end-to-end training, with a complete law and an auditably supplied prior. That is a reconstruction of a possible learning methodology, not sufficient evidence of a new scientific contribution.

## 10. Success counterfactual and what remains consequential

The motivating success counterfactual is strong: a learner encountering a large, partly observed command interface autonomously discovers a small set of recursively executable latent operations from endpoint error, retains them, and learns later compositions without enumerating joint states or receiving execution traces. That would affect acquisition of interpreters, protocol models, and persistent controllable memory whose useful state is distributed over histories. Its potential importance comes from learning an execution state and operation algebra, rather than only predicting another output.

For V0, only the known commutative-involution instance has been constructed. In that instance, the conventional compiler above already supplies the same result. A counterfactual beyond the toy would require a real advantage in discovering unknown useful update laws from a broad but bounded class, or a new causal account of when feedback selects a reusable algebra. No such advantage is established here; it cannot be inferred from a successful three-bit demonstration.

The finite objective of this pass is complete: one limitation, an explicit construction, a worked acquisition and execution, cost sources, checked precedents, a strong conventional competitor, and a load-bearing failure boundary. No experiment or successor branch is automatically authorized.

## 11. Critical interface and frozen scientific state

| Field | Frozen assessment |
|---|---|
| Exact question | Can endpoint error cause acquisition of compact recurrent state and compositional update operators, rather than merely fit a joint-state input-output map? |
| Candidate mechanism | Active residual-prefix search discovers character masks; basis growth creates recurrent sign units; products reuse state for later readouts. |
| Decisive assumptions | Known commuting involutions, resettable active endpoint access, sufficiently heavy useful spectrum, exact sign transitions. |
| Fragile step | Acquiring or justifying the operator algebra itself without supplying a privileged state/equivalence oracle or incurring joint-state cost. |
| Strongest rival | Sparse-Fourier query learning plus a conventional register compiler reproduces V0 exactly. |
| Evidence obtained | Original source-body checks and explicit hand derivations; no empirical evidence and no proof assistant execution. |
| What could reverse the judgment | A fully specified broader structural law with a new finite-budget acquisition effect against the same-information compiler/model-learning alternatives. A neural implementation or better passive baseline score alone would not suffice. |
| Next single decision | Root should decide whether the known-method equivalence and unresolved algebra-discovery step close this candidate or justify a tightly specified comparison. Do not launch an experiment or broaden the class automatically. |

CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION

Endpoint feedback can construct exact recurrent factors in the stated algebraic class | RECONSTRUCTION, hand-checked in stated sense; not executed | Equations (1)-(6), worked process, paired reading note | Query access and algebra supplied; coefficient detection can be expensive | None executed | Assess against classical compiler equivalence.

This is a new or superior mechanism-acquisition principle | UNESTABLISHED; V0 guarantee has an exact conventional reproduction | Section 9 | No difference remains in the worked class | Empirical relation testing does not certify a global algebra; AND-only feedback defeats heavy-feature efficiency; generic residual-RNN extension gives no acquisition law | Root decides whether to close or ask for one specific missing distinction.
