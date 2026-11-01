# Endpoint-query information audit: unknown linear coordinates followed by AND

**HAND MATHEMATICAL DERIVATION / INFORMATION BOUNDARY.** Same Prometheus V2; root requested this audit. Read the actual `after018-acquisition/candidate-v0.md` and paired original-reading record. No experiment, training, new source sweep or modification of a frozen candidate. Effective provider mode unknown. This is support for mechanism construction, not a novelty claim, a universal neural impossibility result, or authorization to run the parity example.

## Judgment

The root conjecture is correct with explicit qualifications. For `A in GL(m,2)`, `m>=1`, an AND endpoint is a **single-positive-point oracle** at `s=A^(-1)1_m`. With a hard cap q on noiseless point evaluations, a uniform unknown s is queried positively with probability at most `min(q,N)/N`; allowing one final **unverified** guess identifies s with probability at most `min(q+1,N)/N`. Adaptation, neural optimization and self-generated tests do not improve this information bound when their only environment-dependent input is that same query transcript.

If r>=1 actual independent rows B of A are known, and no other information about the unknown rows is present, then `B s=1_r` and the conditional candidate set has `N=2^(m-r)` points. The same bounds hold. For r=0, the correct number is `2^m-1`, not `2^m`. Arbitrary learned parity masks are not automatically rows with positive-point target value one. Additional old-task side information can also invalidate a uniform-coset assumption.

The consequence for acquisition is specific: a compact state/circuit does not guarantee informative endpoint feedback. Inspecting one's own computation can change this boundary **if that computation already contains information about the unknown function**. Merely differentiating a candidate model constructed from the same negative labels cannot create it.

## 1. Exact oracle and randomness assumptions

Work over F_2. Named commands commute and are involutions, and the reset state is zero. A command word has a known parity vector x in F_2^m. Its hidden state is `b=A x`, where A is unknown and invertible. Its sole new-task endpoint label is

`f_A(x)=AND_(i=1..m) (A x)_i`.

The learner may choose any word adaptively. Repetition, word order and longer words give no further information in this specified class; every endpoint label equals the label of the word's parity x. Each actual endpoint evaluation counts. Internal computation, own-model gradients and random bits independent of A are allowed without restricting the learning architecture.

q is a **pathwise maximum number of endpoint evaluations**, a nonnegative integer, not an expected budget and not a count of high-level aggregate calls. The bounds concern the number of evaluations; reset, commands, simulation, arithmetic, storage and gradient computation cost additional resources. Returning a guessed point and then querying it uses another evaluation.

For distributional statements, A is initially uniform on GL(m,2), or uniform on its completions of the stated known rows. For worst-case statements, A is fixed before the learner's random bits are sampled. A different prior can have a different concentration and bound.

## 2. Reduction to a uniformly hidden positive point

### Proposition 1: exact point-oracle reduction

For every invertible A,

`f_A(x)=1 <=> A x=1_m <=> x=s=A^(-1)1_m`.

Because m>=1, `1_m!=0`, hence s!=0. Conversely, every nonzero s is possible for some invertible A. To see this, choose a linear functional ell with `ell(s)=1`; extend ell to a row basis `{ell,v_2,...,v_m}` with `v_i(s)=0`. The rows `{ell,ell+v_2,...,ell+v_m}` remain a basis and all evaluate to one on s. They form a suitable A.

### Proposition 2: uniform A induces uniform s on S=F_2^m minus {0}

For any nonzero s,t there is an invertible T with Tt=s. Right multiplication maps A to AT bijectively, and `(AT)^(-1)1_m=T^(-1)s=t`. Thus the same number of matrices give each nonzero positive point. Consequently `Pr(s=x)=1/(2^m-1)` on S.

Equivalently, observing this entire endpoint function identifies s but generally not the ordered rows of A. There are `|GL(m,2)|/(2^m-1)` matrices per positive point. For m>=2 this is greater than one. Every continuation remains a translated point query, so additional words of this same task cannot distinguish matrices within such a fiber. Actual hidden factors may be behaviorally unidentifiable even after every endpoint has been observed. This does not prevent computing the single task once s is known: raw command-parity state plus comparison with s suffices.

## 3. Adaptive queries: discovery, identification and minimax scope

Let S be any N-point candidate set, s uniform on S, and `f_s(x)=1[x=s]`. Queries outside S return zero and give no information. Fix a deterministic learner and follow its **all-negative branch**. Let Q be the distinct points of S it would query along that branch, so |Q|<=q.

If a first positive response occurs at time t, every preceding response was zero. Therefore that query is exactly the t-th query on this all-negative branch. Conversely, if s lies in Q, its first occurrence on that branch gives a positive response. Hence the hit event is exactly `s in Q` and

`Pr(at least one observed positive) = |Q|/N <= min(q,N)/N`.       (1)

This handles fully adaptive queries: the apparent branches before the first hit all have the same negative answers. Conditional on q distinct negatives, the posterior is uniform on the N-q unqueried candidates.

Suppose the learner may also output one candidate `s_hat` after its queries. On the all-negative branch its output is a fixed point g. Correct identification can occur only when `s in Q union {g}`. Thus

`Pr(s_hat=s) <= min(q+1,N)/N`.                                (2)

The plus one is an unqueried final guess, not a free verified positive. If the result must carry an observed positive endpoint certificate, (1) applies instead. When q=N-1, all-negative answers identify the last point by elimination, even though no positive was observed. This separates exact identification from positive discovery.

For a randomized learner, condition on its random seed, apply the same argument, and average. These are bounds under the uniform target distribution. They do **not** assert the same upper bound for every fixed target: a learner can hard-code one point and hit it immediately. Averaging does imply that every randomized learner has some fixed target with success no greater than (1)/(2). Conversely, uniformly sampling distinct candidate points attains q/N hit probability for every fixed target; adding a uniformly chosen unqueried guess attains (q+1)/N identification probability, clipped at one. Thus these are the exact minimax success values for this oracle and hard cap.

For deterministic worst-case guarantees, q<N allows a fixed target outside Q and therefore no observed positive; q+1<N also allows a fixed target outside `Q union {g}`, defeating identification. Describing an adversary as choosing that consistent target after seeing the all-negative transcript is a proof device: it establishes existence of a fixed target. It must not be confused with allowing the physical environment to change A during execution. For randomized worst-case bounds, the uniform-prior averaging argument avoids choosing A after seeing the private seed.

**Expected-budget caveat.** The formula q/N cannot simply be used with q equal to the expected number of queries. Sampling all points without replacement until the first hit succeeds with probability one and uses `(N+1)/2` queries on average. That does not violate (1), because its pathwise cap is N. This observation does not itself establish an expected-budget bound; the audit uses the hard cap specified above.

## 4. Known rows: realizability, conditional uniformity and exceptions

Let `B in F_2^(r by m)` have full row rank, and suppose these are r **actual known rows** of A. Row order is immaterial to AND, so place them first. For the positive point,

`B s=1_r`.

If r>=1, this affine set S_B excludes zero and has `N=2^(m-r)` elements. Every such s is realizable. Start with any invertible completion of B. Whenever an unknown row evaluates to zero on s, add the first known row to it. These elementary row additions preserve invertibility and B, and make every row evaluate to one on s.

Realizability alone is not a uniform-prior proof. Here is the counting proof for a uniform random invertible completion. Change input coordinates so `B=[I_r 0]`; this is an invertible relabeling of queries. Put n=m-r. Every completion has block form

`A = [ I_r  0 ; C  D ]`,

where C is an arbitrary n-by-r matrix and D is in GL(n,2). Candidate positive points have form `s=(1_r,u)`. The condition on the lower block is

`C 1_r + D u=1_n`.

For each fixed D and u, each row of C has one prescribed dot product with the nonzero vector `1_r`, giving `2^(r-1)` choices per row. Thus every u has exactly

`|GL(n,2)| * 2^((r-1)n)`

compatible completions, out of `|GL(n,2)| * 2^(r n)` total. The ratio is `2^(-n)`. Therefore s is uniform on S_B, and (1)/(2) apply with `N=2^(m-r)`. At r=m, N=1 and s is already known; no endpoint query is needed for identification. Use the usual empty-block convention for this count.

Necessary qualifications:

- **r=0:** zero cannot be a positive point, so N is `2^m-1`, not `2^m`.
- **Arbitrary old parity masks:** a known mask that is the XOR of two latent rows satisfies `B_j s=0`, not one. The correct general statement needs known target constraints `B s=t`. For rank r, its affine solution set has `2^(m-r)` points; exclude zero if t=0 because s!=0. Do not infer t from the mask alone.
- **Alignment is load-bearing:** retaining a rank-r character basis for old tasks does not imply knowing its relation to the new AND's latent basis. Unless the rows are known aligned factors of this A, or their target syndrome `t=B s` is independently known, rank alone gives no `2^r` query saving. For an explicit two-bit example, retain the old coordinate `B=[1 0]`. A new AND may use `A_1=I`, with positive point `(1,1)` and `B s=1`, or `A_2=[1 1;0 1]`, with positive point `(0,1)` and `B s=0`. Both are invertible and supply exactly the same known old coordinate function. More strongly, knowing the full raw coordinate basis `B=I_m` already gives rank m, yet if the new A is independently unknown every nonzero s remains possible. The purported N=1 conclusion would be false. The required object is a **known constraint on the new positive point**, not merely an executable old feature.
- **Other old information:** if old tasks/traces, learned code or parameters reveal more than B, the posterior need not remain uniform on S_B. The precise premise is that side information is only B (or leaves s uniform on the stated set). Coset size is not by itself a distributional bound under arbitrary side information. A positive point previously observed would already identify s.
- **Nonuniform posterior:** conditional success is bounded by the mass of the q most likely remaining points, and identification by the mass of q+1 points. Uniform q/N is a special case.
- **Cost of old capabilities:** acquiring the known rows is not free in a total-learning claim. The bound is conditional on having them; it does not erase their earlier labels, computation or storage.

The effective information bottleneck is the number of unresolved positive candidates. If m-r is logarithmic in m, exhaustive residual search can be polynomial; if a linear number of unknown directions remains, the bound is exponential in that number under the stated uniform interface.

## 5. Prediction is a different success criterion

On the full uniform input cube, the always-zero predictor has error exactly `2^(-m)` for every A. On a uniform candidate set S of size N its error is 1/N. It can therefore obtain extremely high average accuracy without ever finding a positive, identifying s, or acquiring hidden factors. A PAC requirement with epsilon above that mass is already satisfied by this trivial predictor. There is no contradiction between this and the search bound.

Exact agreement on every input/continuation does require identifying s. A balanced evaluation that places substantial mass on the positive point also exposes the always-zero failure, but must remain an evaluation distribution rather than an uncharged source of positive training examples. If a training sampler supplies positives with constant probability, seeing a single such input identifies s: that is genuinely different information access.

Obtaining all rows of A is a still stronger and generally unidentifiable objective from the AND oracle alone. A different A with the same positive point produces the same complete current-task behavior. A future task revealing a latent coordinate can distinguish those representations, but its new labels are new information. Current-task accuracy cannot establish acquisition of those unobserved future distinctions.

## 6. Which extra access changes the information, and which does not

### No change from transcript-only internal work

After an all-negative transcript, every remaining s produces exactly the same observations. A self-generated test, neural representation, program, pseudo-label, own-model gradient or random proposal computed from that transcript has the same conditional distribution under all those s. It cannot distinguish them. It can choose the next query intelligently for a nonuniform prior, but the uniform oracle has no informative asymmetry to exploit.

Backpropagating `(prediction_theta(x)-observed_label)^2` through the learner's own graph is allowed. It differentiates the candidate prediction, not the unknown environment's A. All unqueried-positive worlds still supply the same negative labels. A differentiable simulator with the true hidden A, gradients of the true endpoint with respect to latent state, or access to its computational graph would be **additional information**. Over F_2, an arbitrary real-valued differentiable extension is also extra structure; it is not determined by the finite endpoint oracle.

This leaves open the scientifically different setting in which a learner's **own existing computation** is inspectable. If that retained computation already encodes useful factors, causal traces or a teaching procedure, analyzing it may expose information absent from a bare point oracle. Its origin, formation cost and actual contents then matter. If the learner generated that computation solely from the same previous endpoint answers and independent randomness, inspecting it does not defeat the bound. The audit concerns information acquisition, not a ban on useful self-inspection.

### Access that does change the oracle

- **Component outputs:** access to a latent bit `(A x)_i` exposes one row via m unit-vector queries. All m bits expose A in m vector-output queries, or m^2 scalar-bit labels. Output bandwidth is different and must be counted.
- **Aggregate bit error/Hamming weight:** `w(x)=sum_i (A x)_i` has Fourier expansion `m/2 - (1/2) sum_i chi_(row_i(A))(x)`. There are m nonconstant masks of absolute coefficient 1/2; after normalization into [0,1] the coefficients are 1/(2m). The already specified prefix-energy acquisition method therefore has a polynomial inverse-threshold bound in this class, unlike AND's exponentially weak coefficients. This is a genuinely more informative endpoint signal, even though it is still scalar. No numerical claim or new algorithm is asserted.
- **True equivalence/counterexample oracle:** proposing the all-zero function and receiving a real disagreement immediately supplies s. Finding that disagreement by endpoint enumeration costs the search; self-generated tests cannot be relabeled an equivalence oracle.
- **Subset/group query:** an oracle returning whether s belongs to an arbitrary chosen subset supports binary splitting in logarithmic queries. Implementing the subset test by evaluating ordinary endpoints still pays every actual evaluation; a subset name is not a one-query implementation.
- **A positive-example sampler or informative old tasks:** either changes the posterior directly. The sampler, task acquisition and all labels must be supplied equally to competitors and charged.

### Aggregate notation cannot hide exponential evaluations

An exact sum over all endpoints, exact Fourier coefficient, prefix energy or logical existential test may expose s compactly **if granted as a primitive**. Computing it by `2^m` underlying endpoint evaluations is `2^m` queries, regardless of whether the caller invokes one function. Sampling such a statistic uses its actual sampled endpoints and remains subject to (1)/(2). Until a positive is hit, every product or other statistic of the observed AND labels is zero or otherwise transcript-determined. The audit does not prohibit aggregate estimators; it prevents counting their unobserved answers as supplied evidence.

## 7. Finite scientific use and stopping boundary

The useful result is a repaired finite information statement: endpoint-only acquisition can be hard even when command algebra, compact recurrent state and a short output circuit are available. Known old factors can reduce that burden, but their information and costs must be real. Richer feedback or inspectable retained computation can legitimately change the problem; their access must be specified instead of assumed.

This is not a novel mechanism or a reason to run the parity example. It also does not show that neural self-construction is impossible, that all terminal-feedback problems are unlearnable, or that the project's broader question should be replaced by point search. It marks one source of failure that a new acquisition mechanism must actually address.

`CLAIM | STATUS | EVIDENCE/ARTIFACT | WEAKEST REMAINING LINK | FAILED BRANCHES | NEXT SINGLE DECISION`

`Uniform hidden AND-positive point has q/N hit and (q+1)/N identification bounds under a hard q-query cap | HAND PROOF above | all-negative-branch argument §§2–3 | stated interface/prior/side information; no computational run or proof assistant | no empirical branch | use as an information contract for the next mechanism`

`Known r>=1 actual rows reduce the candidate set to a uniform 2^(m-r) coset under uniform completions | HAND PROOF above | row-completion construction and block count §4 | arbitrary old features/metadata may provide different constraints or posterior | r=0 formula repaired; arbitrary-mask RHS clarified | state exactly which old information is supplied`

`Self-inspection can bypass the bound without new environment information | FALSE in transcript-only setting; OPEN in a genuinely richer retained-computation setting | indistinguishability §6 | origin and contents of inspectable computation must be explicit | own-model gradients are not unknown-environment derivatives | root/P specify the actual information change, if any`

Audit complete; no further experiment or successor task is inferred.
